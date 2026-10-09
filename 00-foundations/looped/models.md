# Looped Transformer：现在的 looped LM 长什么样

**中文** · [English](models.en.md)

> 阅读时间：约 4 分钟 · 难度：进阶 · 最近审阅：2026-10-09

## Huginn：prelude、循环块、coda

[Geiping 等（2025）的 Huginn](https://arxiv.org/abs/2502.05171) 是个 3.5B 的模型，结构分三段：

- **prelude**（2 层）：把输入变成 embedding $e$；
- **循环块**（4 层）：反复跑；每一圈都用一个 adapter 把当前状态 $s$ 和 $e$ 拼起来、再压回原来的宽度，相当于每圈把输入重新喂一遍；起始状态 $s_0$ 是随机的；
- **coda**（2 层）：把最后的状态变回下一个 token 的分布。

训练随机采样递归深度，配置期望约 32 圈，反向最多穿过最后 8 圈。更早的状态仍参与前向，只是不保留完整反传链；这不是完整展开目标的无偏梯度。论文在约 800B token 训练的模型上报告了增加推理计算的收益；“50B 等效计算量”不是 50B 的参数容量，更不是每一题随圈数单调变好。

## Ouro：把循环放进预训练

[Ouro（2025）](https://arxiv.org/abs/2510.25741)重复 decoder 层栈，不是每圈重新查一次 token embedding：

| | Ouro 1.4B | Ouro 2.6B |
| --- | --- | --- |
| 层数 × 宽度 | 24 × 2048 | 48 × 2048（由复制层扩展） |
| 循环次数 | 4 | 4 |
| 训练 token | 7.7T | 7.7T |

7.7T 是完整训练路线的 token 账，包括两种尺寸分叉前的共同阶段，不表示 2.6B 从第一步就独立训练了这么多。论文报告过增加圈数的稳定性问题；深度外推也不保证提升。退出 gate 和实际执行循环的区别在[上一篇](adaptive-depth.md)。

## Relaxed Recursive Transformers：从现成模型改过来

[Bae 等的工作](https://arxiv.org/abs/2410.20672)从预训练模型初始化共享层，再继续训练；不同深度的小 LoRA 允许它们有一定差别，也需要额外参数。论文中的模型比较有指定任务和训练预算，并非“绑好权重就无损压缩”。其 2–3× 吞吐潜力来自理论 / 模拟及 oracle early-exit 条件，不是直接部署必得的收益。

## 放在一起看

| | 怎么循环 | 训练时转几圈 | 每圈注入输入 | 深度怎么定 |
| --- | --- | --- | --- | --- |
| Huginn | 中间 4 层 | 随机，平均 32 | 是 | 推理时自己选 |
| Ouro | decoder 层栈 | 4 | 否 | gate 选择输出；是否跳过计算看引擎 |
| Relaxed Recursive | 共享基座与按深度区分的 LoRA | 配置的最大深度 | 否 | 可研究 early exit 与深度组批 |
| Mixture-of-Recursions | 共享 block | 由 router 分配 | 否 | 每个 token 一个深度 |

Mixture-of-Recursions 的路由和缓存边界见[自适应深度](adaptive-depth.md)。选型时至少固定 checkpoint、最大圈数、输入 / 输出长度、dtype、KV 方案和 batch；不然“参数更少”与“服务更快”很容易被混成同一件事。
