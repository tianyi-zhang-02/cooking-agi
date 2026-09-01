# Online Softmax and IO-Aware Attention

[中文](03-online-softmax.md) · **English** · [Back to project](../../projects/01-attention-from-scratch.en.md)

> Reading time: ~5 minutes · Level: Advanced · Freshness: Stable principle, evolving implementations · Last reviewed: 2026-08

## Core problem

Why should attention avoid writing the complete `N×N` score matrix to HBM?

A conventional implementation produces scores, reads and writes probabilities, and finally computes `PV`. As sequence length grows, intermediate storage and movement grow with `N²`. HBM traffic can dominate even when the mathematical FLOP count is unchanged.

## Online softmax state

Stable softmax needs a row maximum and exponential sum. An online version reads scores by block while maintaining:

```text
m = running maximum
l = running normalization sum
o = accumulated weighted output
```

For a new block:

```text
m_new = max(m, max(score_block))
old_scale = exp(m - m_new)
new_weights = exp(score_block - m_new)
l_new = old_scale × l + sum(new_weights)
o_new = old_scale × o + new_weights × V_block
```

The final output is `o / l`. When the maximum changes, the previous accumulated state must be rescaled; omitting that step creates systematic error.

## IO-aware data flow

The objective is not merely fewer softmax operations. Q, K, and V tiles are reused in closer storage while complete score and probability matrices never return to HBM:

```text
Q tile stays local
→ stream K/V tiles
→ update online softmax state
→ emit output tile
```

This is the central mental model behind FlashAttention-like methods: reorder computation to reduce expensive movement while preserving the mathematical result.

## Correctness sequence

1. Implement one-dimensional online softmax on CPU.
2. Compare with ordinary stable softmax on random and extreme inputs.
3. Add block boundaries and non-multiple sizes.
4. Add the weighted accumulation over V.
5. Only then move to CUDA and fuse.

Masks must apply inside the current score block, and a causal boundary may cross a tile.

## Hands-on check

- Make the running maximum increase in a later block and verify rescaling.
- Compare error across block sizes.
- Measure whether HBM bytes and peak memory actually fall.
- Record latency too; lower memory with excessive computation or synchronization may lose.
- Check long sequences and large scores for NaN and Inf.

## Key conclusions

FlashAttention is not “a faster softmax.” It is an IO-aware reordering of attention. Correctness comes from the online-normalization invariant; performance comes from reducing movement across expensive storage levels.

Next: [Precision, KV Cache, and Benchmarks](04-precision-kv-benchmark.en.md)
