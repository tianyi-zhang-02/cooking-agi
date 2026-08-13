# Vanilla Transformer

[中文](vanilla-transformer.md) · **English**

> Reading time: ~8 min · Level: core · Last reviewed: 2026-08

## In one sentence

The original Transformer is an encoder–decoder. Attention moves information across positions, FFNs transform channels independently, and removing recurrence makes full-sequence training parallel.

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

## Three attention sites

| Site | Query | Key / value | Mask | Role |
| --- | --- | --- | --- | --- |
| encoder self-attention | source | source | padding | bidirectional source understanding |
| decoder self-attention | target prefix | target prefix | causal + padding | prevent future access |
| decoder cross-attention | decoder state | encoder states | source padding | retrieve source evidence |

Attention shortens the path between arbitrary positions to one layer and parallelizes training, but the $T\times T$ score matrix makes standard self-attention approximately quadratic in sequence length.

The 2017 model used sinusoidal positions, post-norm LayerNorm, MHA, and ReLU FFNs. Modern decoder-only models more often use RoPE, pre-norm RMSNorm, GQA, and SwiGLU. Do not collapse these into one architecture.

Run [`../code/vanilla_demo.py`](../code/vanilla_demo.py), then use [the full Transformer deep dive](../transformer.en.md). Next: [Decoder-only](decoder-only.en.md).
