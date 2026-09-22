# Looped Transformer：代价与局限

**中文** · [English](costs.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## 计算和延迟：圈数是串行的

参数没变，计算量却和圈数成正比。更麻烦的是，同一个 token 的各圈只能一圈接一圈地算，并行不了，所以**延迟也跟着圈数涨**。能做的是在不同 token 之间做流水线，比如 Relaxed Recursive Transformers 提的按深度连续 batching。

## KV cache 随圈数增长

每一圈都要重新跑一遍 attention，所以默认每个（圈，层）都得存自己的 K/V。Ouro 1.4B 是 24 层转 4 圈，公开代码里每个 token 占 96 个 KV 槽位，正好是同样 24 层不循环时的 4 倍。另一篇论文测出来，Ouro-1.4B-Thinking 完整的逐圈 KV 每个 token 大约 0.786 MB。

## 能不能各圈共享 KV

这是目前争议最大的地方：

- **Ouro 论文**：在 prefill 阶段共享 KV，GSM8K 掉 10 分以上；只在解码阶段复用最后一圈的 KV，GSM8K 78.85 对 78.92，MATH-500 80.40 对 82.40，解码显存省 4 倍。
- **Continuous Depth Batching**（2026）：在 Ouro-1.4B 上，GSM8K-CoT 完整 KV 是 77.86，所有圈共用一个槽位只剩 0.23，第一圈单独存、其余共用是 71.34；作者说复现不了 Ouro 报告的几乎无损。同样的操作在 Huginn 上几乎没影响（33.74、33.97、34.80）。
- **MELT**（2026）：几种不需要训练的共享方式，在 Ouro-1.4B-Thinking 上的 AIME、AMC、MATH-500 全部是 0。
- **Mixture-of-Recursions**：只为参与这一圈的 token 存 KV（recursion-wise caching），或者只存第一圈再复用（recursive sharing）；后者对 expert-choice 有害，对 token-choice 有帮助。
- **Huginn**：给每个 token 固定 $k$ 个槽位，第 $i$ 圈读写第 $i \bmod k$ 个，不用额外训练也能工作。

结论是：KV 能不能省，得看模型当初是怎么训的，不能默认它免费。

## 训练更难

圈数越多，训练越不稳：Ouro 试 8 圈时就碰上 loss spike；Huginn 只让梯度穿过最后 8 圈，再靠每圈重新注入输入、随机起始状态这些办法把训练稳住。

## 它不会多记知识

参数就那么多，循环并不会让它多装知识：Ouro 测出来，循不循环都是每个参数大约 2 bit。Saunshi 等也发现，同样计算量下 looped 模型的 perplexity 和记忆更差。要装大量事实的任务，参数量还是硬约束。

## 什么时候值得

- 推理重、知识轻的任务；
- 显存或参数吃紧，但愿意多花点计算和延迟的部署；
- 想要一个推理时能拧的「多想一会儿」旋钮，又不想多吐 token。
