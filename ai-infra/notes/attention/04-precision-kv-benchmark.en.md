# Attention Precision, KV Cache, and Benchmarks

[中文](04-precision-kv-benchmark.md) · **English** · [Back to project](../../projects/01-attention-from-scratch.en.md)

> Reading time: ~5 minutes · Level: Advanced · Freshness: Fast-moving · Last reviewed: 2026-08

## Core problem

How can we prove that a lower-precision or decode-attention optimization is worth using?

The answer must include correctness, latency, throughput, memory, and workload. TFLOP/s or one shape cannot justify a serving decision.

## Precision path

Compare in order:

1. FP32 input with FP32 accumulation;
2. BF16/FP16 input with FP32 accumulation;
3. FP8 or quantized paths only when hardware and tooling explicitly support them.

Check:

- absolute and relative output error;
- softmax row sums;
- NaN and Inf;
- large-magnitude scores;
- long sequences;
- latency, throughput, HBM bytes, and peak memory.

Do not lower every intermediate merely to claim dtype support. Reductions, normalization, and accumulation may require higher precision.

## KV cache

Serving self-attention has two phases:

- **Prefill:** process the prompt and create all historical K/V.
- **Decode:** append one K/V per step and let the new query read the cache.

A simplified cache estimate is:

```text
KV bytes ≈ 2 × layers × tokens × kv_heads × head_dim × bytes
```

Real usage also depends on batch slots, tensor parallelism, padding, allocation, and paged layout.

Implement contiguous cache first and define shape, append position, valid length, batch slot, and query-head-to-KV-head mapping. Study paged allocation afterward.

## Why benchmarks must be separate

Prefill has larger matrices and is commonly more compute-intensive. Decode has a small query at each step but repeatedly reads weights and historical KV, making bandwidth and cache layout more important.

At minimum, separate:

| Workload | Variables |
| --- | --- |
| Prefill | prompt length, batch, heads, dtype |
| Decode | context length, active sequences, KV dtype, GQA |
| End-to-end | TTFT, TPOT, throughput, P95/P99 |

## Hands-on check

1. Compare FP32 and BF16/FP16 on identical inputs.
2. Test cache append, sequence-length boundaries, and batch-slot reuse.
3. Record prefill and decode kernel time separately.
4. Sweep context length and observe bandwidth and latency.
5. Preserve hardware, CUDA, compiler, flags, commit, and complete shapes.

## Key conclusions

Lower precision succeeds only when quality and system gains agree; KV cache succeeds only on realistic decode workloads. “Faster” has no meaning without a workload.

Back to [Project 01 · Attention from Scratch](../../projects/01-attention-from-scratch.en.md)
