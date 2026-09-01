# Module 03 · 数值计算与混合精度

**中文** · [English](03-numerical-computing.en.md) · [返回 AI Infra](../README.md)

> 阅读时间：约 5 分钟 · 难度：Intermediate · 时效性：Fast-moving · 最近审阅：2026-08

## 这一模块解决什么

低精度可以提高吞吐、减少显存和通信，但也会损失范围或精度。本模块建立一个判断框架：**哪些值可以降低精度，哪些操作必须保留更高精度，误差怎样被发现？**

## 学习目标

- 理解浮点数的符号、指数和尾数；
- 比较 FP32、TF32、FP16、BF16、FP8、FP4 与整数格式；
- 解释 overflow、underflow、rounding 和 accumulation error；
- 理解 automatic mixed precision 与 loss scaling；
- 能设计量化质量、显存与速度的对照实验。

## 核心笔记

### 范围和精度不是一回事

指数位主要决定能表示的数量级范围，尾数位主要决定同一数量级里的分辨率。

| 格式 | 指数直觉 | 尾数直觉 | 主要特点 |
| --- | --- | --- | --- |
| FP32 | 大 | 高 | 稳定但存储和带宽成本高 |
| FP16 | 较小 | 高于 BF16 | 容易 overflow/underflow |
| BF16 | 接近 FP32 | 低于 FP16 | 动态范围大，适合训练 |
| FP8 E4M3 | 较小 | FP8 中较高 | 更偏精度 |
| FP8 E5M2 | 较大 | 更低 | 更偏范围 |
| MXFP8 | FP8 值配合 block scale | 更细的局部适应 | 依赖硬件与 layout |
| NVFP4 / INT4 | 很有限 | 很有限 | 强依赖 scaling、kernel 与校准 |

TF32 通常是 Tensor Core 的矩阵计算模式，而不是把模型权重存成一种新的 19-bit tensor 类型。“BF24”不是主流 LLM 工作流中的常用格式，遇到时应检查具体硬件或论文定义。

### Block scaling 也是格式的一部分

现代 FP8/FP4 路径不能只用单个元素的位数描述。MXFP8 为小块数值设置 scale；NVFP4 把小块 scale 与 tensor-level scale 组合起来。这些选择会影响精度、metadata、layout、transpose、支持的 shape 和通信。一个低精度结论如果没有说明 scaling configuration、累加精度、kernel 和硬件，就是不完整的。

### 混合精度

不是所有算子都适合相同精度。常见策略是：

```text
矩阵乘法输入：BF16 / FP16 / FP8
累加：FP32 或更高的内部精度
reduction、normalization、部分 softmax：较高精度
优化器状态：根据算法和稳定性保留较高精度
```

Autocast 根据算子选择数据类型。FP16 训练常需要 loss scaling，避免很小的梯度在反向传播中下溢；BF16 因指数范围更大，通常更少依赖 loss scaling，但仍需检查具体训练行为。

### 量化

量化把连续值映射到有限离散值：

```text
q = clamp(round(x / scale) + zero_point)
x_hat = scale × (q - zero_point)
```

关键选择包括：

- symmetric 或 asymmetric；
- per-tensor、per-channel 或 per-group scale；
- weight-only 或 weight-and-activation；
- static calibration 或 dynamic quantization；
- post-training quantization 或 quantization-aware training。

异常值会迫使 scale 覆盖很大范围，导致大多数普通值的有效分辨率下降。因此量化不仅是格式转换，也是数据分布问题。

## 需要会算

Tensor 的基础存储量：

```text
memory = number of elements × bytes per element
```

量化压缩比的第一近似：

```text
compression ratio ≈ original bits / quantized bits
```

真实收益还要扣除 scale、zero point、padding、临时 buffer 和未量化层。

误差至少应从三层检查：

1. tensor-level：绝对误差、相对误差、cosine similarity；
2. model-level：loss、perplexity、task metrics；
3. system-level：latency、throughput、memory 与 cost。

## 动手验证

1. 打印 FP16 与 BF16 对极大、极小数的表示差异。
2. 用 FP32、FP16、BF16 完成同一个矩阵乘法并比较误差和时间。
3. 对一个小模型记录 autocast 前后的 operator dtype。
4. 实现简单的 per-tensor INT8 quantize/dequantize。
5. 比较 BF16 与一种硬件支持的 FP8 或 FP4 配置，记录精度、显存、吞吐、scale overhead 和不支持的 shape。

## 常见误区

- 位数更少不保证硬件上更快；
- 模型能运行不代表输出质量没有系统性退化；
- 只看平均误差可能掩盖少数严重 outlier；
- 理论压缩比不等于端到端显存下降；
- 量化后的 kernel、layout 和硬件支持往往比格式名称更重要。

## 掌握检查

- BF16 为什么比 FP16 有更大动态范围？
- 为什么矩阵乘法可以低精度输入、高精度累加？
- loss scaling 在解决什么问题？
- per-channel scale 为什么可能优于 per-tensor scale？
- 怎样证明一次低精度优化值得采用？

当前资料：[NVIDIA Transformer Engine FP8、MXFP8 与 NVFP4 指南](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

下一步：[Module 04 · 分布式训练](04-distributed-training.md)
