# 精度与显存：为什么换成 BF16 还会 OOM？

**中文** · [English](precision-and-memory.en.md)

> 最近审阅：2026-10 · 前置：[一次训练](training-step.md)、[LoRA / QLoRA](../../05-post-training/lora-and-qlora.md)

模型权重只占几 GB，训练却放不进显卡，这并不奇怪。权重只是显存账单的一项。反向传播需要中间结果，优化器需要状态，某些算子还要临时工作区。

先分清两件事：**数值能不能可靠地算，数据能不能同时放下。** BF16、梯度累积和重算分别处理其中一部分，没有一个开关能解决全部问题。

## FP16 和 BF16 不是谁全面更精确

下面列的是存储格式，不代表每个算子的累加也使用同一精度。

| 格式 | 符号 / 指数 / 尾数位 | 1 附近相邻数间隔 | 主要特点 |
| --- | --- | --- | --- |
| FP32 | 1 / 8 / 23 | $2^{-23}$ | 范围大，分辨率较细 |
| FP16 | 1 / 5 / 10 | $2^{-10}$ | 比 BF16 更细，但范围小 |
| BF16 | 1 / 8 / 7 | $2^{-7}$ | 指数范围接近 FP32，分辨率较粗 |

例如 1.001 在 BF16 中按最近值舍入会成为 1，在 FP16 中约为 1.0009765625。相反，数值 $10^5$ 已超过 FP16 最大有限值 65504，却在 BF16 范围内。**表示范围和有效精度是两条不同的轴。**

为了看清舍入，下面用 Python 自带的 FP16 打包；BF16 的示例直接按 1 附近的间隔计算，只适用于这个局部例子。

```python
import struct

half_value = struct.unpack('e', struct.pack('e', 1.001))[0]
bfloat_near_one = 1 + round((1.001 - 1) * 128) / 128
assert half_value == 1.0009765625
assert bfloat_near_one == 1.0
```

矩阵乘法的输入、乘法、中间累加和输出存储也可能采用不同精度。不能看见某个张量是 BF16，就断定整个训练“都在 BF16 算”。

## Autocast 不是把模型全部转成 FP16

`model.half()` 改的是模型浮点参数与 buffer 的存储 dtype；`autocast` 则按算子选择计算精度。一次 forward 中，矩阵乘法与某些归约可能采用不同精度。Backward 也不是统一切回 FP32，而是跟随相应前向操作选用的类型。

所以排查问题时，要分别看参数、激活、梯度与累加的 dtype。`state_dict()` 保存的是模型状态，不会因为开过 AMP 就自动把权重转换成 FP32。[PyTorch AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)把 autocast 与 gradient scaling 分开使用：前者选精度，后者处理梯度的数值范围。

## Loss scaling 在救什么？

很小的梯度在 FP16 下可能舍入为 0。把 loss 乘尺度 $s$，反向得到 $s\nabla\mathcal L$，更新前再除以 $s$，在理想算术中没有改变梯度，却有机会避免中间梯度下溢。

它不能救所有情况：如果前向先产生了 Inf，后面缩放 loss 不会把它变回正确值。BF16 通常不像 FP16 那样需要 scaling，但仍可能有溢出、舍入或不稳定训练。

使用动态 scaler 时，顺序应是：

```text
forward → 缩放 loss → backward（完成所有累积）
        → unscale → gradient clipping → optimizer step → 更新 scale
```

不能先裁剪放大后的梯度，再按原来的阈值解释它。一个累积窗口中也不能随意改变 scale，否则累积的梯度不在同一尺度。[PyTorch AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)说明了这些操作的顺序。

拿一个标量梯度算一次：真实梯度为 0.25，scale 为 8，反向得到 2；裁剪阈值为 0.5。正确顺序得到 0.25，不需要裁剪。若先把 2 裁成 0.5，再除以 8，就只剩 0.0625，无意间把更新缩小了 4 倍。

```python
def clip_scalar(gradient, limit):
    return max(-limit, min(gradient, limit))

scaled_gradient = 0.25 * 8
correct_gradient = clip_scalar(scaled_gradient / 8, 0.5)
wrong_gradient = clip_scalar(scaled_gradient, 0.5) / 8
assert correct_gradient == 0.25
assert wrong_gradient == 0.0625
```

这里只用一个标量说明顺序；实际 `clip_grad_norm_` 按参数梯度的整体范数缩放，不是逐元素裁剪。标准动态 `GradScaler` 遇到非有限梯度时会跳过这次参数更新并调整 scale，并不自动保证把同一批数据重新跑一遍。

## 把训练显存拆成 5 项

$$
M_{\mathrm{peak}}\approx M_{\mathrm{weights}}+M_{\mathrm{grads}}
+M_{\mathrm{optimizer}}+M_{\mathrm{activations}}+M_{\mathrm{temporary}}.
$$

这是记账方式，不是保证五项各自峰值同时出现的精确公式。

以一个 **假设的 mixed-precision Adam 配置**为例：每参数 2 bytes 权重、2 bytes 梯度、4 bytes FP32 master copy、8 bytes 两份 FP32 moments，总计 16 bytes。10 亿参数就是约 16 GB，**还没算激活**。另一些实现以 FP32 保存参数或梯度、不单独维护 master copy，必须按真实张量核对，不能硬套 16。

| 症状 | 优先看什么 |
| --- | --- |
| 刚加载模型就 OOM | 权重、重复模型副本、设备放置 |
| forward 随序列变长而 OOM | 激活、attention 中间量、输出 logits |
| 第一次 optimizer step 才 OOM | 懒初始化的 optimizer states |
| 跑很多步后持续上涨 | 是否保留了带计算图的 loss / outputs |
| reserved 很高，allocated 较低 | 分配器与碎片，未必是活跃张量太多 |

`nvidia-smi` 的占用不等于 PyTorch 活跃张量大小；也不能单凭 reserved/allocated 差距就诊断内存泄漏。

## 梯度累积省的是 microbatch 激活

4 次 microbatch、每次 2 条序列，可以在一次 optimizer step 前累积出 8 条序列的梯度。每次反向完成后释放该 microbatch 的计算图，就不用同时保存 8 条的激活。

但权重和 Adam states 不会因此变小；把序列从 2K 拉到 32K，也不一定能靠减小 batch 救回来。BatchNorm、随机性、序列长度和 loss 分母都会影响它是否等价于一个大 batch。

有效 token 数不等时，要按总有效 token 数加权，不要机械地每个 loss 除以累积次数。多卡时还要算上梯度同步的平均，见[分布式训练](../../06-systems/distributed-training.md)。

## Checkpointing：不保存的中间量，反向时重算

这里说的是 **activation checkpointing**，不是往磁盘保存训练进度。

为什么序列一长就需要它？先只算两个张量，不猜整个模型。设 batch 为 2、hidden size 为 1024、attention heads 为 16，每个元素 2 bytes：

| 序列长度 | 一个 $B\times T\times d$ 隐藏状态 | 一个显式 $B\times H\times T\times T$ 分数张量 |
| --- | ---: | ---: |
| 2048 | 8 MiB | 256 MiB |
| 4096 | 16 MiB | 1024 MiB |

```python
def tensor_mib(shape, bytes_per_value=2):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * bytes_per_value / 2**20

assert tensor_mib((2, 2048, 1024)) == 8
assert tensor_mib((2, 16, 2048, 2048)) == 256
```

序列翻倍，前者翻倍，后者变成 4 倍。这不是每层总显存：还要看哪些中间量被保存、什么时候释放、是否融合算子。[FlashAttention](attention-kernels.md)避免把完整分数矩阵写入显存，checkpointing 则减少反向前一直保留的激活。两者能配合，但不能把各自节省的数字直接相加。

```text
普通：x → block 1 → block 2 → block 3 → loss
           保存内部激活，反向时读取
重算：保存分段边界 → 反向到这一段时重新 forward → 算梯度
```

模型参数没有变少，optimizer states 也还在。节省的是某些激活，代价是额外计算；非重入实现可能只重算需要的部分。若分段中有随机数、状态更新或副作用，要保证重算与原 forward 的语义一致。

[PyTorch checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)推荐显式选择非重入实现；具体 API 按所用版本核对。测试应比较 loss 和参数梯度，不只看程序能否跑完。dropout 的 RNG 状态是否恢复也会影响结果。

## 怎样判断一个“省显存方案”值得用？

固定模型、序列分布、有效 tokens 和 optimizer，分别记录峰值显存、每秒有效 tokens、loss/梯度偏差以及恢复训练行为。下面几种方法不是替代关系：

| 方法 | 主要节省 | 主要代价 |
| --- | --- | --- |
| 低精度 | 部分张量存储和计算带宽 | 舍入、范围与 kernel 支持 |
| 梯度累积 | 同时存活的 microbatch 激活 | 更多串行步骤；权重不缩小 |
| Activation checkpointing | 保存的中间激活 | 重算耗时 |
| LoRA | 可训练梯度与 optimizer 状态 | 更新空间受限；基座仍需前向 |
| FSDP / ZeRO | 每卡持有的训练状态 | 通信和临时参数聚合 |

不要只报“显存降了多少”。如果每步慢了一倍，但可以使用更大的有效 batch，最后是否更快要测；如果不 OOM 了却改变了训练目标，就不是同一次比较。
