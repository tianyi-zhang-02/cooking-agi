# Module 05 · LLM Inference Systems

[中文](05-llm-inference.md) · **English** · [Back to AI Infra](../README.en.md)

> Reading time: ~5 minutes · Level: Intermediate · Freshness: Fast-moving · Last reviewed: 2026-08

## What this module solves

Training optimizes the process that produces a model; inference optimizes how that model serves real requests. This module studies how to control latency, throughput, memory, and cost when request lengths and arrival times are irregular.

## Learning goals

- Explain tokenization, queueing, prefill, decode, and streaming.
- Distinguish TTFT, TPOT, end-to-end latency, and throughput.
- Understand KV cache, paged attention, and continuous batching.
- Explain prefix caching, chunked prefill, and speculative decoding.
- Design a serving benchmark that includes both quality and system metrics.

## Core notes

### Request lifecycle

```text
arrival → admission control → queue → batching
→ prefill → decode loop → streaming → completion or cancellation
```

The system schedules tokens as well as requests. Prompt lengths, output lengths, and stopping times vary, so static batches easily create padding and waiting.

### Prefill and decode

Prefill processes many prompt tokens together, producing larger matrices that can often use Tensor Cores effectively. Decode normally generates one new token per active request per step, repeatedly reading model weights and KV cache, and is often more constrained by memory bandwidth.

A long prefill can block requests already decoding and harm interactivity. Excessively prioritizing decode can starve new requests of their first token. The scheduler manages fairness and throughput across these two forms of work.

### KV cache

Attention reuses keys and values for previous tokens. KV cache avoids recomputing the full history on every decode step, but grows with concurrency, context length, and layer count.

Paged attention divides KV cache into blocks that can be allocated non-contiguously. This reduces reservation and fragmentation and lets variable-length requests share a memory pool more flexibly.

### Continuous batching

A static batch waits for all requests to finish. Continuous batching removes completed requests immediately and inserts new work into later decode iterations. Utilization improves, but the scheduler must maintain each sequence's state, positions, and cache mapping.

### Common optimizations

- **Prefix caching:** reuse KV state for identical prompt prefixes.
- **Chunked prefill:** split long prefills to reduce blocking.
- **Speculative decoding:** let a cheaper draft model propose tokens for the target model to verify.
- **Quantization:** reduce weight or KV-cache storage and bandwidth.
- **Tensor/Pipeline Parallel:** serve models that do not fit on one GPU.
- **Admission control:** limit incoming work before the system becomes overloaded.

## Quantities to calculate

A simplified KV-cache estimate is:

```text
KV bytes ≈ 2 × layers × tokens × kv_heads × head_dim × bytes_per_element
```

The factor of 2 represents keys and values. Sum across all active sequences and account for tensor-parallel sharding, padding, and allocator overhead.

Core metrics are:

```text
TTFT = first token timestamp - request arrival
TPOT = decode duration / generated tokens
throughput = completed tokens or requests / wall-clock time
```

Latency must be reported with percentiles and a workload distribution. One mean value is rarely informative.

## Hands-on work

1. Serve a small model with vLLM, SGLang, or TensorRT-LLM.
2. Vary prompt length, output length, concurrency, and the batch-token budget.
3. Record TTFT, TPOT, throughput, P95/P99, and peak memory.
4. Compare repeated-prefix workloads with prefix caching enabled and disabled.
5. Compare BF16 with one quantized configuration on the same quality and cost workload.

## Common misconceptions

- High tokens/s does not guarantee short user wait times.
- Fixed-length benchmarks overestimate real serving performance.
- The largest batch is rarely the best batch.
- KV-cache OOM is different from model weights not fitting.
- Ignoring cancellation, timeout, retry, and streaming disconnects leads to incorrect capacity estimates.

## Mastery check

- Why do prefill and decode have different bottlenecks?
- Why is continuous batching better suited to LLMs than static batching?
- How does GQA affect KV-cache size?
- Under what conditions does speculative decoding improve speed?
- How can one long prompt be prevented from destroying tail latency for other requests?

Next: [Module 06 · GPU Platforms](06-gpu-platforms.en.md)
