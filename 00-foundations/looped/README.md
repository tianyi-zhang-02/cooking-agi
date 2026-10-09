# Looped Transformer：把同一个 block 重复用

**中文** · [English](README.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-10-09

把一个 4 层 block 连续算 3 遍，等于走过 12 次层计算，却只存 4 层权重。第二遍接着第一遍的状态算，所以通常不会得到同一个结果。这就是循环模型的基本做法：重复使用参数，多花一些计算，而不是凭空增加知识。

## 深度和参数，本来是绑在一起的

普通的、不共享层参数的 $L$ 层 Transformer，每层有自己的一套权重。沿用这种结构增加层数，计算深度和参数量会一起涨。循环是拆开这两笔账的一种办法；生成更多中间 token、调用工具等，也可以增加解题计算，但不是这里讨论的结构。

## 把 block 循环起来

Looped Transformer 只存一个 $k$ 层的 block，反复用 $L$ 次：

$$h^{(t+1)} = f_\theta\big(h^{(t)},\, e\big), \qquad t = 0, 1, \ldots, L-1$$

每一圈用的都是同一个 $\theta$；$e$ 是输入表示，不少设计会每圈重新注入（input injection）。循环块存的是 $k$ 层权重，执行 $kL$ 次层计算；embedding、输出层和额外 adapter 要另算。它并不等价于拥有 $kL$ 套独立参数的模型。

“同样的参数”不等于“同样的输出”。例如反复算 $h\leftarrow 0.5h+1$，从 0 出发会得到 1、1.5、1.75。状态在变，所以每圈做的事也在变。但若换成 $h\leftarrow 2h$，状态会越变越大：复用本身既不保证收敛，也不保证答案越来越好。

<!-- widget:tx-loop-unroll -->

## 老想法，新用法

- [Universal Transformer](https://arxiv.org/abs/1807.03819)（2018）：跨位置和深度共享 transition，并研究用 ACT 分配计算。
- [ALBERT](https://arxiv.org/abs/1909.11942)（2019）：跨层共享，加上 embedding 因式分解，共同减少参数。共享不会自动减少执行的层数；也不能把 108M → 12M 全归因于共享，或由此断言推理时间完全不变。
- **2025 年的 Huginn 与 Ouro**把循环用于语言模型预训练。圈数可以作为实验变量，但超出训练深度后是否有效，要看具体 checkpoint 和任务，不能当作通用的推理增强开关。

## 这一组怎么读

1. [为什么循环有助于推理](why-loops-reason.md)：串行步数、理论结果，以及和 CoT 的关系（图能拖）
2. [每个 token 该转几圈](adaptive-depth.md)：ACT、PonderNet、Ouro 的退出 gate、Mixture-of-Recursions（图能拖）
3. [现在的 looped LM 长什么样](models.md)：Huginn、Ouro、Relaxed Recursive Transformers
4. [代价与局限](costs.md)：延迟、KV cache、训练稳定性，以及知识容量实验的边界
5. [复习题](review.md)：面试题和自检
