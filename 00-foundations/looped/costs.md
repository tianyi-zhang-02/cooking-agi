# Looped Transformer：代价与局限

**中文** · [English](costs.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## 计算和延迟：圈数是串行的

参数不变，计算量却和圈数成正比。更麻烦的是，同一个 token 的各圈必须一圈接一圈地算，没法并行，所以**延迟也随圈数增长**。能做的是在不同 token 之间流水线化，比如 Relaxed Recursive Transformers 提的按深度连续 batching。

## KV cache 随圈数增长

每一圈都会重新跑 attention，所以默认每个（圈，层）都要存自己的 K/V。Ouro 1.4B 是 24 层循环 4 圈，公开代码里每个 token 有 96 个 KV 槽位，是同样 24 层不循环模型的 4 倍。另一篇论文测得 Ouro-1.4B-Thinking 完整的逐圈 KV 大约是每个 token 0.786 MB。

## 能不能各圈共享 KV

这是目前争议最大的地方：

- **Ouro 论文**：在 prefill 阶段共享 KV，GSM8K 掉 10 分以上；只在解码阶段复用最后一圈的 KV，GSM8K 78.85 对 78.92，MATH-500 80.40 对 82.40，解码显存省 4 倍。
- **Continuous Depth Batching**（2026）：在 Ouro-1.4B 上，GSM8K-CoT 完整 KV 是 77.86，所有圈共用一个槽位只剩 0.23，第一圈单独存、其余共用是 71.34；作者说复现不了 Ouro 报告的几乎无损。同样的操作在 Huginn 上几乎没影响（33.74、33.97、34.80）。
- **MELT**（2026）：几种不需要训练的共享方式，在 Ouro-1.4B-Thinking 上的 AIME、AMC、MATH-500 全部是 0。
- **Mixture-of-Recursions**：只为参与这一圈的 token 存 KV（recursion-wise caching），或者只存第一圈再复用（recursive sharing）；后者对 expert-choice 有害，对 token-choice 有帮助。
- **Huginn**：给每个 token 固定 $k$ 个槽位，第 $i$ 圈读写第 $i \bmod k$ 个，不用额外训练也能工作。

结论：KV 能不能省，取决于模型是怎么训练的，不能默认它是免费的。

## 训练更难

循环越深，训练越不稳定：Ouro 试 8 圈时出现 loss spike；Huginn 只把梯度传过最后 8 圈，并用每圈重新注入输入、随机初始状态这些办法来稳住训练。

## 它不会多记知识

同样的参数，循环不增加知识容量：Ouro 测得循不循环都是每个参数大约 2 bit。Saunshi 等也发现，同样计算量下 looped 模型的 perplexity 和记忆更差。需要大量事实知识的任务，参数量仍然是硬约束。

## 什么时候值得

- 推理重、知识轻的任务；
- 显存或参数受限，但可以接受更多计算和延迟的部署；
- 想要一个推理时可以调的「多想一会儿」旋钮，又不想生成更多 token。
