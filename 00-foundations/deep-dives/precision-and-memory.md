# 精度与显存：为什么换成 BF16 还会 OOM？

**中文** · [English](precision-and-memory.en.md)

> 最近审阅：2026-10 · 前置：[一次训练](training-step.md)、[LoRA / QLoRA](../../05-post-training/lora-and-qlora.md)

模型权重只占几 GB，训练却放不进显卡，这并不奇怪。权重只是显存账单的一项。反向传播需要中间结果，优化器需要状态，某些算子还要临时工作区。

先分清两件事：**数值能不能可靠地算，数据能不能同时放下。** BF16、梯度累积和重算分别处理其中一部分，没有一个开关能解决全部问题。

## FP16 和 BF16 不是谁全面更精确

下面列的是存储格式，不代表每个算子的累加也使用同一精度。这里只讲 FP32、FP16 和 BF16；FP8、FP4 以及 NVFP4 这类按小块缩放的格式，见 [GPU 架构的数据类型一节](../../12-infra/gpu-architecture.md)。

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

### 梯度有数值，为什么权重没动？ {#rounded-away-update}

假设一个权重从 1 开始，每步减去 0.001。这次更新量在 BF16 中并不是 0，但 `1 - 0.001` 写回 BF16 时会舍入成 1。在 1 的下方，最近的 BF16 数是 0.99609375；0.999 离 1 更近。

如果每一步都这样写回，同样的小更新不会自动攒起来。下面固定减去 0.001，重复 100 次；不引入学习率调度、动量或真实训练数据，只看存储舍入：

<figure class="worked-update worked-update--pairs" lang="zh-CN" id="precision-update-paths">
  <figcaption>同样减 100 次，保留更新的精度不同，结果就不同。</figcaption>
  <ol>
    <li><small>每步写回 BF16</small><strong>仍然是 1</strong><span>每次的小变化都被舍入掉，下一步又从 1 开始。</span></li>
    <li><small>在 FP32 中持续更新</small><strong>约为 0.9</strong><span>小变化留下来了；最后转成 BF16，读到 0.8984375。</span></li>
  </ol>
</figure>

<details class="code-drop" markdown="1">
<summary>用 CPU 上的 PyTorch 验证</summary>

```python
import torch

def rounded_update_example():
    low_precision = torch.tensor(1.0, dtype=torch.bfloat16)
    full_precision = torch.tensor(1.0, dtype=torch.float32)
    for step in range(100):
        low_precision.sub_(0.001)
        full_precision.sub_(0.001)
    return low_precision.item(), full_precision.item(), full_precision.to(torch.bfloat16).item()

low_result, full_result, rounded_result = rounded_update_example()
assert torch.tensor(0.001, dtype=torch.bfloat16).item() != 0.0
assert low_result == 1.0
assert abs(full_result - 0.9) < 2e-6
assert rounded_result == 0.8984375
```

</details>

这和“梯度太小，存下来就变成 0”是两种问题。Loss scaling 在更新前要 unscale，不能靠它永久放大步长来绕过权重舍入。保留 FP32 参数或 master weights，是为了让小更新有地方累积；不是说所有低精度训练都必须保留同一套副本。[Mixed Precision Training §3.1–3.2](https://arxiv.org/html/1710.03740v3#S3.SS1)分别讨论了权重更新精度与梯度缩放；上面是本站的 BF16 算例，不是复现该论文的 FP16 实验。

## Autocast 不是把模型全部转成 FP16

`model.half()` 改的是模型浮点参数与 buffer 的存储 dtype；`autocast` 则按算子选择计算精度。一次 forward 中，矩阵乘法与某些归约可能采用不同精度。Backward 也不会统一切回 FP32，每个操作沿用对应前向操作选用的类型。

所以排查问题时，要分别看参数、激活、梯度与累加的 dtype。`state_dict()` 保存的是模型状态，不会因为开过 AMP 就自动把权重转换成 FP32。[PyTorch 2.8 AMP 示例](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html)把 autocast 与 gradient scaling 分开使用：前者选精度，后者处理梯度的数值范围。

例如保留 FP32 的线性层，在 CPU BF16 autocast 里计算 `[3, 4] @ [1, 2] = 11`，再对目标 10 算平方损失。这次运行中，参数是 FP32、线性层输出是 BF16；把输出显式转回 FP32 算 loss，反向后参数的 `.grad` 也是 FP32。**梯度存成 FP32，不代表前面的低精度运算没发生过舍入。**

<details class="code-drop" markdown="1">
<summary>把各处 dtype 打印出来</summary>

```python
import torch

def autocast_dtype_example():
    layer = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = layer(torch.tensor([[3.0, 4.0]], dtype=torch.float32))
        loss = (prediction.float() - 10.0).square().mean()
    loss.backward()
    dtypes = (layer.weight.dtype, prediction.dtype, loss.dtype, layer.weight.grad.dtype)
    return dtypes, layer.weight.grad

dtypes, gradient = autocast_dtype_example()
assert dtypes == (torch.float32, torch.bfloat16, torch.float32, torch.float32)
torch.testing.assert_close(gradient, torch.tensor([[6.0, 8.0]], dtype=torch.float32))
print(dtypes)
```

</details>

这是 PyTorch 2.8.0 / CPU 的接口检查，不是 GPU 速度实验。本页代码都在 2.8.0 上运行；2026-10 对照稳定版 [2.14 文档](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)，下文引用的 AMP 用法、checkpoint 限制和显存计数说明都没有变，但代码没有在 2.14 上重跑。这里 FP32 参数本身就保存更新，不需要再算一份相同用途的 FP32 master copy；换成低精度参数加独立 master weights 的实现，显存账就不同了；后面 16 bytes 的算例假设的正是这种实现。更多算子的选择规则见 [AMP 文档](https://docs.pytorch.org/docs/2.8/amp.html)，不要把这一个线性层的结果推广成所有算子的 dtype 规则。

## Loss scaling 在救什么？

很小的梯度在 FP16 下可能舍入为 0。把 loss 乘尺度 $s$，反向得到 $s\nabla\mathcal L$，更新前再除以 $s$，在理想算术中没有改变梯度，却有机会避免中间梯度下溢。

它不能救所有情况：如果前向先产生了 Inf，后面缩放 loss 不会把它变回正确值。BF16 通常不像 FP16 那样需要 scaling，但仍可能有溢出、舍入或不稳定训练。

使用动态 scaler 时，顺序应是：

```text
forward → 缩放 loss → backward（完成所有累积）
        → unscale → gradient clipping → optimizer step → 更新 scale
```

不能先裁剪放大后的梯度，再按原来的阈值解释它。一个累积窗口中也不能随意改变 scale，否则累积的梯度不在同一尺度。[PyTorch 2.8 AMP 示例](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html)说明了这些操作的顺序。

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
\begin{aligned}
M_{\mathrm{peak}}\approx{}& M_{\mathrm{weights}}+M_{\mathrm{grads}}\\
&+M_{\mathrm{optimizer}}+M_{\mathrm{activations}}\\
&+M_{\mathrm{temporary}}.
\end{aligned}
$$

这是记账方式，不是保证五项各自峰值同时出现的精确公式。

以一个 **假设的 mixed-precision Adam 配置**为例：每参数 2 bytes 权重、2 bytes 梯度、4 bytes FP32 master copy、8 bytes 两份 FP32 moments，总计 16 bytes。10 亿参数就是约 16 GB，**还没算激活**。另一些实现以 FP32 保存参数或梯度、不单独维护 master copy，必须按真实张量核对，不能硬套 16。

### 先算一份看得见的账 {#state-ledger}

沿用这个配置，只算 10 亿参数的常驻训练状态。把 master copy 单独列出来，避免和低精度权重混算：

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>权重</small><strong>2 GB</strong><span>10 亿个 BF16 参数，每个 2 bytes。</span></li>
<li><small>梯度</small><strong>2 GB</strong><span>每个参数保存 1 个 BF16 梯度，也是 2 bytes。</span></li>
<li><small>FP32 主副本 · master copy</small><strong>4 GB</strong><span>每个参数再留 1 份 FP32 副本，每个 4 bytes。</span></li>
<li><small>Adam 状态 · moments</small><strong>8 GB</strong><span>每个参数有 2 个 FP32 矩估计，共 8 bytes。</span></li>
</ol>
<figcaption>合计 16 GB；只算本例的参数相关状态，还没加入激活和临时工作区。</figcaption>
</figure>

这里的 moments 是 Adam 保存的梯度一阶矩和二阶原始矩的估计。可以先理解为对过去梯度及其平方的移动平均；即使当前 batch 已经算完，这两份状态仍要留下来供下一步使用。

GB 按 $10^9$ bytes 算，GiB 按 $2^{30}$ bytes 算，所以这里的 16 GB 约为 **14.90 GiB**。单位不同，不代表凭空省了显存。

<details markdown="1">
<summary>自己换参数算一下（Python）</summary>

```python
def state_bytes(total_parameters, trainable_parameters,
                weight_bytes=2, gradient_bytes=2, master_bytes=4, moment_bytes=8):
    counts = (total_parameters, trainable_parameters)
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
        raise ValueError("parameter counts must be nonnegative integers")
    if trainable_parameters > total_parameters:
        raise ValueError("trainable parameters exceed total parameters")
    widths = (weight_bytes, gradient_bytes, master_bytes, moment_bytes)
    if any(not isinstance(width, int) or isinstance(width, bool) or width < 0 for width in widths):
        raise ValueError("storage widths must be nonnegative integers")
    return {
        "weights": total_parameters * weight_bytes,
        "gradients": trainable_parameters * gradient_bytes,
        "master": trainable_parameters * master_bytes,
        "moments": trainable_parameters * moment_bytes,
    }

full_state = state_bytes(1_000_000_000, 1_000_000_000)
assert sum(full_state.values()) == 16_000_000_000
assert round(sum(full_state.values()) / 2**30, 2) == 14.90
adapter_state = state_bytes(1_010_000_000, 10_000_000)
assert sum(adapter_state.values()) == 2_160_000_000
```

</details>

最后一个算例是假设的 LoRA 账本：基座 10 亿参数，额外 adapter 1000 万参数，全部权重按 2 bytes 计，只有 adapter 有梯度、master copy 和 moments，得到 2.16 GB。**不是把 16 GB 直接乘 1%。** 基座还在；反向所需激活也不在这份账里。真实 adapter 的 dtype 可能不同，量化还会增加 scales 等元数据，要按实现另算。

### 第一次更新为什么还会突然多一块显存？ {#lazy-optimizer-state}

用 CPU 上的 PyTorch 2.8.0 做一个小检查：只有 100 个 FP32 参数，使用普通 AdamW，关闭 foreach，没有 AMP、额外 master copy 或共享存储。逐项数 `numel() * element_size()`，结果如下：

| 时刻 | 权重 | 梯度 | Optimizer 中的 tensor | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 创建 optimizer 后 | 400 B | 0 B | 0 B | 400 B |
| 第一次 backward 后 | 400 B | 400 B | 0 B | 800 B |
| 第一次 step 后 | 400 B | 400 B | 804 B | 1604 B |
| `zero_grad(set_to_none=True)` 后 | 400 B | 0 B | 804 B | 1204 B |

804 B 是两份各 400 B 的 moments，加上本次实现中 4 B 的 step tensor。这里数的是这些 tensor 的逻辑存储量，不包含 Python 对象、allocator 和计算临时量，更不是 CPU 实验测出的 GPU 峰值。可执行检查在 [测试文件](../../site/tests/test_training_engineering.py)中。

这个例子想说明：**只看模型刚加载后的占用，会漏掉还没创建的状态。** 检查 GPU 显存时，至少要经过一次完整更新，也要试代表性的长序列。`element_size()` 能帮你核对 dtype，但 view、共享权重和扁平 buffer 会让简单相加重复计数。

如果把 AdamW 换成 SGD，账也会变。仍用上面的 100 个 FP32 参数，第一次更新之后：

| Optimizer 配置 | 额外的 optimizer tensor 存储 |
| --- | ---: |
| SGD，`momentum=0` | 0 B |
| SGD，`momentum=0.9` | 400 B，保存一份 momentum buffer |
| AdamW，默认非 AMSGrad | 804 B，两份 moments 和 step |

这只是在同一个 CPU 实现里数状态，不包含权重和梯度，也不能当 GPU 峰值。**SGD 不一定有动量；Adam 的 moments 也不是梯度的来源。** 先由 backward 得到梯度，再用梯度更新 moments，最后决定参数怎么走。可对照 [SGD](https://docs.pytorch.org/docs/2.8/generated/torch.optim.SGD.html) 和 [Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html) 的更新顺序。换 dtype、启用 AMSGrad 或换成分片 optimizer，都要重新记账，不能固定说“训练等于推理显存的几倍”。

### 把状态分片后，每卡要存多少？ {#sharded-state}

普通数据并行每卡保留模型副本，不能把总数直接除以设备数。沿用上面 16 GB 的假设，以 8 个 rank 为例，只计算均匀分片后、每卡持有的状态：

| 方式 | 每卡的理想账本 | 每卡 GB |
| --- | --- | ---: |
| 普通数据并行 | 权重 2 + 梯度 2 + master / moments 12 | 16 |
| ZeRO-1 | $2+2+12/8$ | 5.5 |
| ZeRO-2 | $2+(2+12)/8$ | 3.75 |
| ZeRO-3 | $(2+2+12)/8$ | 2 |

分片对象依次增加，这是 [ZeRO 原论文](https://arxiv.org/html/1910.02054v3#S5)讨论的基本思路。表中没有算执行时的参数聚合、通信 buffer、激活或碎片。看到最后一行的 2 GB，不能据此断言 2 GB 显卡就能训练这个模型；聚合某一层参数时，仍会有额外峰值。框架具体怎么切分、预取和释放，要接着看[分布式训练](../../06-systems/distributed-training.md)。

| 症状 | 优先看什么 |
| --- | --- |
| 刚加载模型就 OOM | 权重、重复模型副本、设备放置 |
| forward 随序列变长而 OOM | 激活、attention 中间量、输出 logits |
| 第一次 optimizer step 才 OOM | 懒初始化的 optimizer states |
| 跑很多步后持续上涨 | 是否保留了带计算图的 loss / outputs |
| reserved 很高，allocated 较低 | 分配器与碎片，未必是活跃张量太多 |

`nvidia-smi` 的占用不等于 PyTorch 活跃张量大小；也不能单凭 reserved/allocated 差距就诊断内存泄漏。

实际记录时，把 `memory_allocated`、`memory_reserved` 和峰值分开。先完成需要的预热，再重置峰值计数，跑完要测的完整步骤并等待 GPU 完成；同时单独保留首次更新的峰值。否则预热把一次性开销藏掉了，或计时结束时 GPU 还没做完。[PyTorch 的内存说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management)介绍了这几种指标；`empty_cache()` 只能释放未使用的缓存，不能替你删掉仍被引用的张量。

## 梯度累积省的是 microbatch 激活

4 次 microbatch、每次 2 条序列，可以在一次 optimizer step 前累积出 8 条序列的梯度。每次反向完成后释放该 microbatch 的计算图，就不用同时保存 8 条的激活。

但权重和 Adam states 不会因此变小；把序列从 2K 拉到 32K，也不一定能靠减小 batch 救回来。BatchNorm、随机性、序列长度和 loss 分母都会影响它是否等价于一个大 batch。

有效 token 数不等时，要按总有效 token 数加权，不要机械地每个 loss 除以累积次数。多卡时还要算上梯度同步的平均，见[分布式训练](../../06-systems/distributed-training.md)。

## Checkpointing：不保存的中间量，反向时重算

这里说的是 **activation checkpointing**（也常叫 gradient checkpointing），不是往磁盘[保存训练进度](../../practice/post-training/checkpoint-and-resume.md)。Q、K、V 等是中间张量；只有选定保存与重算策略后，才谈得上哪些是检查点边界，并不是每个中间量都叫 checkpoint。

### 不留中间结果，梯度从哪里来？ {#recompute-example}

只看两次运算：$h=wx$，$\ell=h^2$。令 $x=2$、$w=3$，前向得到 $h=6$、$\ell=36$。反向求权重梯度时，要用到刚才的 $h$：

$$
\frac{\partial\ell}{\partial w}=2hx=24.
$$

<figure class="worked-update worked-update--pairs" lang="zh-CN">
<ol>
<li><small>保留</small><strong>把 h = 6 留到反向</strong><span>需要时直接读取，不再做这次乘法。</span></li>
<li><small>重算</small><strong>用原来的 x 和 w 再算 h</strong><span>2 × 3 仍然得到 6，所以梯度还是 24。</span></li>
</ol>
<figcaption>改变的是拿到中间结果的方式，不是把这条梯度丢掉。一个标量没有省显存的实际意义，只用于说明机制。</figcaption>
</figure>

“原来的”很重要：若反向前擅自把权重改成 4，重算会得到 $h=8$，已经不是那次前向了。真实代码也不能在重算区域里随意修改全局变量、更新统计量或发送外部请求。默认的 determinism check 不会逐元素证明结果相同；仍要比较数值与梯度。[PyTorch 2.8 的说明](https://docs.pytorch.org/docs/2.8/checkpoint.html)列出了重算的限制。

<details markdown="1">
<summary>在 CPU 上对照普通前向与重算</summary>

```python
import torch
from torch.utils.checkpoint import checkpoint

def squared_product(features, weight):
    hidden = features * weight
    return hidden.square().sum()

def checkpoint_example(use_checkpoint):
    features = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    if use_checkpoint:
        loss = checkpoint(squared_product, features, weight, use_reentrant=False)
    else:
        loss = squared_product(features, weight)
    loss.backward()
    return loss.item(), features.grad.item(), weight.grad.item()

assert checkpoint_example(False) == (36.0, 36.0, 24.0)
assert checkpoint_example(True) == checkpoint_example(False)
```

使用 PyTorch 2.8 / CPU，只验证函数值与梯度，不报告 GPU 内存或加速。带 Dropout 的对照还需要对齐 RNG；项目测试另外检查了这一点。不要把这里的精确相等推广到所有算子和硬件。

</details>

### 哪些中间量值得省？ {#recompute-budget}

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

FlashAttention 也不是“反向不需要保存任何东西”。[原论文 §3.1 / Appendix B](https://arxiv.org/html/2205.14135v2#S3.SS1)用 Q/K/V、输出和行归一化统计量重建局部注意力；带 Dropout 时还要处理 RNG。省掉的是完整的二次大小中间矩阵，不是所有训练状态。中间激活与它的 `.grad` 又是两件事：保存前向值供 backward 使用，不等于给每个非叶子张量保留梯度，详见 [autograd](../pytorch/autograd.md)。

```text
普通：x → block 1 → block 2 → block 3 → loss
           保存内部激活，反向时读取
重算：保存分段边界 → 反向到这一段时重新 forward → 算梯度
```

模型参数没有变少，optimizer states 也还在。节省的是某些激活，代价是额外计算；非重入实现可能只重算需要的部分。若分段中有随机数、状态更新或副作用，要保证重算与原 forward 的语义一致。

[PyTorch 2.8 checkpoint 文档](https://docs.pytorch.org/docs/2.8/checkpoint.html)推荐显式设置 `use_reentrant=False`。这条路径可以在需要的中间量已得到时提前结束重算，不保证函数里的日志或计数器恰好执行两次。默认会处理所支持设备的 RNG 状态，但不是自动复制整个程序的可变状态。

也不一定整块都重算。**Selective activation checkpointing** 可以保留某些昂贵算子的结果，让便宜的算子重新执行。假设 A 的结果占 4 MiB、重算耗时 6 ms，B 占 32 MiB、重算耗时 1 ms，且本例中两者都一直留到反向、能独立重算：

| 本例选择 | 留下的内部结果 | 额外重算 |
| --- | ---: | ---: |
| 都保存 | 36 MiB | 0 ms |
| 保存 A，重算 B | 4 MiB | 1 ms |
| 都重算 | 0 MiB | 7 ms |

这里还没算边界输入、输出和重算时的临时量，**0 MiB 不是训练峰值为零**。这些是假设值，不是测试结果。真实依赖、融合 kernel 和共享存储会改变选择；PyTorch 的[选择性重算介绍](https://pytorch.org/blog/activation-checkpointing-techniques/#selective-activation-checkpoint-new)给出了按算子制定策略的方式。先找出峰值时仍存活的张量，再测省下的内存是否值得额外计算。

## 怎样判断一个“省显存方案”值得用？

固定模型、序列分布、有效 tokens 和 optimizer，分别记录峰值显存、每秒有效 tokens、loss/梯度偏差以及恢复训练行为。下面几种方法针对的开销不同，也可以配合使用：

| 方法 | 主要节省 | 主要代价 |
| --- | --- | --- |
| 低精度 | 部分张量存储和计算带宽 | 舍入、范围与 kernel 支持 |
| 梯度累积 | 同时存活的 microbatch 激活 | 更多串行步骤；权重不缩小 |
| Activation checkpointing | 保存的中间激活 | 重算耗时 |
| LoRA | 可训练梯度与 optimizer 状态 | 更新空间受限；基座仍要做前向和反向传播（只是不存基座权重的梯度） |
| FSDP / ZeRO | 每卡持有的训练状态 | 通信和临时参数聚合 |

不要只报“显存降了多少”。如果每步慢了一倍，但可以使用更大的有效 batch，最后是否更快要测；如果不 OOM 了却改变了训练目标，就不是同一次比较。
