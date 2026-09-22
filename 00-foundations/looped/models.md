# Looped Transformer：现在的 looped LM 长什么样

**中文** · [English](models.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## Huginn：prelude、循环块、coda

Geiping 等（2025）的 Huginn 是个 3.5B 的模型，结构分三段：

- **prelude**（2 层）：把输入变成 embedding $e$；
- **循环块**（4 层）：反复跑；每一圈都用一个 adapter 把当前状态 $s$ 和 $e$ 拼起来、再压回原来的宽度，相当于每圈把输入重新喂一遍；起始状态 $s_0$ 是随机的；
- **coda**（2 层）：把最后的状态变回下一个 token 的分布。

训练时每一步转几圈是随机采的，平均 32 圈；反向传播只穿过最后 8 圈，为的是省显存。它在大约 800B token 上训练，论文说推理能力会随圈数一直往上走，直到相当于 50B 参数模型的计算量。

## Ouro：把循环放进预训练

Ouro（2025）是整个模型一起循环：

| | Ouro 1.4B | Ouro 2.6B |
| --- | --- | --- |
| 层数 × 宽度 | 24 × 2048 | 48 层（由复制层得到） |
| 循环次数 | 4 | 4 |
| 训练 token | 7.7T | 7.7T |

论文提到早期试过 8 圈，训练时出现 loss spike，最后定在 4 圈。退出 gate 在[上一篇](adaptive-depth.md)。

## Relaxed Recursive Transformers：从现成模型改过来

不从头训，而是把一个已经训好的 LLM 改成循环的：几层绑成一个 block 反复用，再给每圈配一个小 LoRA，让各圈之间能有点差别（Bae 等，2024，ICLR 2025）。改出来的 Recursive Gemma 1B 打过了 TinyLlama 1.1B 和 Pythia 1B。论文还提出按深度做连续 batching，估计能把吞吐拉高 2 到 3 倍。

## 放在一起看

| | 怎么循环 | 训练时转几圈 | 每圈注入输入 | 深度怎么定 |
| --- | --- | --- | --- | --- |
| Huginn | 中间 4 层 | 随机，平均 32 | 是 | 推理时自己选 |
| Ouro | 整个模型 | 4 | 否 | 退出 gate |
| Relaxed Recursive | 预训练层绑成 block，每圈一个 LoRA | 固定 | 否 | 固定 |
| Mixture-of-Recursions | 共享 block | 由 router 分配 | 否 | 每个 token 一个深度 |
