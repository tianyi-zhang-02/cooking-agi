# Looped Transformers: costs and limits

[中文](costs.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-09

## Compute and latency: loops are serial

Parameters stay fixed, but compute scales with the number of loops. Worse, one token's loops must run one after another and cannot be parallelised, so **latency grows with the loop count too**. What you can do is pipeline across tokens, as in the continuous depth-wise batching proposed with Relaxed Recursive Transformers.

## The KV cache grows with the loop count

Every loop runs attention again, so by default each (loop, layer) pair keeps its own K/V. Ouro 1.4B is 24 layers looped 4 times, and its released code keeps 96 KV slots per token, 4 times as many as the same 24 layers without looping. Another paper measures the full per-loop KV of Ouro-1.4B-Thinking at about 0.786 MB per token.

## Can the loops share KV?

This is where the evidence disagrees most:

- **The Ouro paper**: sharing KV during prefill costs more than 10 points on GSM8K; reusing only the last loop's KV during decoding gives GSM8K 78.85 vs 78.92 and MATH-500 80.40 vs 82.40, with 4 times less decoding memory.
- **Continuous Depth Batching** (2026): on Ouro-1.4B, GSM8K-CoT scores 77.86 with the full KV, 0.23 with one slot shared by every loop, and 71.34 with the first loop kept separate and the rest shared; the authors say they cannot reproduce Ouro's near-lossless result. The same operations barely affect Huginn (33.74, 33.97, 34.80).
- **MELT** (2026): several training-free sharing schemes all score 0 on AIME, AMC, and MATH-500 with Ouro-1.4B-Thinking.
- **Mixture-of-Recursions**: store KV only for tokens taking part in a loop (recursion-wise caching), or store only the first loop and reuse it (recursive sharing); the latter hurts expert-choice and helps token-choice.
- **Huginn**: give each token a fixed $k$ slots, with loop $i$ reading and writing slot $i \bmod k$; it works without extra training.

The conclusion: whether KV can be saved depends on how the model was trained; do not assume it is free.

## Training is harder

Deeper looping makes training less stable: Ouro hit loss spikes when trying 8 loops; Huginn passes gradients through only the last 8 loops and relies on re-injecting the input every loop and a random starting state to keep training steady.

## It does not store more knowledge

At the same parameter count, looping adds no knowledge capacity: Ouro measures about 2 bits per parameter with or without loops. Saunshi et al. also find worse perplexity and memorisation for looped models at the same compute. For tasks that need a lot of factual knowledge, parameters remain the hard limit.

## When it is worth it

- tasks heavy on reasoning and light on knowledge;
- deployments short on memory or parameters that can afford more compute and latency;
- when you want a "think a little longer" dial at inference time without generating more tokens.
