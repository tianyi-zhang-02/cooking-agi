# Project 01 · Attention from Scratch: CPU Reference to CUDA

[中文](01-attention-from-scratch.md) · **English** · [Project ladder](README.en.md)

## Project goal

Starting from matrix multiplication and softmax, implement verifiable scaled dot-product attention and then add multi-head layouts, causal masking, KV cache, CUDA, tiling, and online softmax.

The goal is not to copy a fast kernel immediately. It is to build an evidence chain:

```text
mathematical definition
→ simple reference
→ C++ CPU implementation
→ correctness tests
→ profiler
→ CUDA implementation
→ numerical and performance comparison
```

## Attention mathematics

Given:

```text
Q: [B, H, Nq, D]
K: [B, Hkv, Nk, D]
V: [B, Hkv, Nk, Dv]
```

Standard scaled dot-product attention is:

```text
S = QKᵀ / √D
P = softmax(S + mask)
O = PV
```

where:

- `B` is batch size;
- `H` is the number of query heads;
- `Hkv` is the number of key/value heads;
- `Nq` and `Nk` are query and key sequence lengths;
- `D` is head dimension;
- a causal mask prevents positions from attending to future tokens.

Use `H = Hkv` first, then add MQA/GQA after correctness.

## Suggested layout

```text
attention-from-scratch/
├── CMakeLists.txt
├── include/attention/
│   ├── attention.h
│   └── tensor.h
├── src/
│   ├── attention_cpu.cpp
│   ├── attention_cuda.cu
│   └── online_softmax.cu
├── reference/
│   └── attention_reference.py
├── tests/
├── bench/
├── results/
└── README.md
```

Reuse tensor, matmul, and softmax from [Project 00](00-c-cpp-tensor-core.en.md), or keep a smaller specialized implementation inside this project.

## Milestone 0 · Python correctness oracle

Write the simplest PyTorch reference without a high-level attention API:

1. explicitly compute `Q @ K.transpose(-2, -1)`;
2. divide by `sqrt(D)`;
3. add the mask;
4. apply softmax;
5. compute `P @ V`.

Save a tiny fixed input/output pair for C++ tests. Add randomized property tests as well so the implementation does not overfit one set of golden values.

## Milestone 1 · Single-head CPU C++

Implement:

```text
scores[Nq, Nk]
= matmul(Q[Nq, D], transpose(K[Nk, D]))
→ scale
→ mask
→ row-wise stable softmax
→ matmul(probabilities[Nq, Nk], V[Nk, Dv])
```

Materialize the full score and probability matrices first. This is memory-heavy and slow, but its control flow is transparent and makes an excellent C++ reference.

Check:

- that `1 / sqrt(D)` is applied before softmax;
- causal-mask boundaries;
- every probability row sums approximately to one;
- extreme scores do not create NaN or Inf;
- non-square `Nq != Nk`;
- `Dv != D`.

## Milestone 2 · Multi-head attention

Add batch and head dimensions:

- explicitly choose `[B, H, N, D]` or `[B, N, H, D]`;
- write the stride for each layout;
- avoid repeated expensive index calculation in inner loops;
- execute each `(batch, head)` as independent single-head attention;
- test that permuting heads does not mix them.

Then add MQA/GQA, mapping several query heads onto fewer KV heads. Define the head mapping explicitly instead of relying on implicit broadcasting.

## Milestone 3 · Masks and padding

Support separately:

- causal masks;
- padding or valid-length masks;
- arbitrary additive masks;
- combined causal and padding masks.

Do not assume one finite negative constant is safe for every dtype. Record mask behavior in FP32 and FP16/BF16 references and define output behavior for an entirely masked row.

## Milestone 4 · Optimized CPU version

Optimize only after establishing a clear baseline:

1. improve loop order and contiguous access;
2. parallelize across `(batch, head)` or query rows;
3. use tiled matrix multiplication;
4. remove unnecessary transpose copies;
5. optionally replace matmul with BLAS while keeping the custom softmax;
6. profile whether time is spent in QKᵀ, softmax, PV, or layout conversion.

Report the difference between reference and optimized CPU versions, but preserve the reference for testing.

## Milestone 5 · Naive CUDA attention

Begin with separately verifiable kernels:

```text
QKᵀ kernel
→ scale and mask kernel
→ row softmax kernel
→ PV kernel
```

Each stage can be copied back to the host and compared against the CPU reference. This is not the final high-performance design, but it isolates bugs.

Observe:

- thread and block mapping;
- coalesced loads;
- synchronization in reductions;
- kernel-launch overhead;
- HBM traffic for intermediate score matrices;
- register and shared-memory use.

## Milestone 6 · Tiled attention

Load Q, K, and V tiles into shared memory, reuse them, and reduce global-memory traffic. Optimize QKᵀ and PV separately before attempting fusion.

Use the roofline perspective:

```text
arithmetic intensity = FLOPs / bytes moved
```

Compare measured bytes, bandwidth, and kernel time between naive and tiled versions. Shared memory does not automatically make a kernel faster.

## Milestone 7 · Online softmax and the FlashAttention idea

Materializing a full `N × N` score matrix creates quadratic intermediate storage and HBM traffic. Online softmax processes scores by block while maintaining a running row maximum and normalization sum:

```text
old state: m, l, accumulated output
new block maximum: m_block
new maximum: m_new = max(m, m_block)
rescale old contribution by exp(m - m_new)
add new block contribution scaled by exp(score - m_new)
update normalization l
```

First verify online softmax against ordinary stable softmax on CPU, then move it to CUDA. Fuse the data flow of `QKᵀ`, softmax normalization, and `PV` so the full score/probability matrix is not written to HBM.

The objective is to understand FlashAttention's IO-aware core idea, not to match production-kernel performance immediately.

## Milestone 8 · Precision

Compare in order:

- FP32 input with FP32 accumulation;
- BF16/FP16 input with FP32 accumulation;
- optional FP8 or quantization experiments when hardware and tooling support them.

Check:

- absolute and relative output error;
- softmax row sums;
- NaN and Inf;
- long sequences and large scores;
- latency, throughput, and peak memory.

Do not manually lower every intermediate dtype merely to claim support for a format.

## Milestone 9 · KV cache and decode

Split self-attention into:

- prefill, which creates K/V for the full prompt;
- decode, which appends one new K/V pair and lets one query read the previous cache.

Implement a simple contiguous KV cache before exploring paged layouts. Record cache shape, append rules, valid length, batch slots, and head mapping.

Compare arithmetic intensity and memory access between prefill and decode and explain why the same attention operation can have different bottlenecks in the two phases.

## Correctness test matrix

Cover at least:

| Dimension | Test values |
| --- | --- |
| Batch | 1, >1 |
| Heads | 1, multiple, GQA |
| Sequence | 1, short, non-multiple of tile, longer |
| Head dim | small, common, non-multiple of tile |
| Mask | none, causal, padding, combined |
| Dtype | FP32, BF16/FP16 when supported |
| Values | random, constant, large magnitude, all-masked edge |

Every optimized version must compare against the same reference, not against the previous optimization, which may already contain a bug.

## Benchmark matrix

Record:

| Version | Shape | Dtype | Causal | Latency | TFLOP/s | HBM traffic | Peak memory | Max error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Separate at least:

- end-to-end time;
- QKᵀ;
- softmax and masking;
- PV;
- layout conversion;
- host/device transfer when it belongs to the real path.

## Definition of done

- Mathematical definitions, shapes, strides, and mask semantics are explicit.
- The CPU reference passes fixed and randomized tests.
- Every naive CUDA stage can be validated independently.
- At least one tiled or fused version has profile evidence showing a reduced bottleneck.
- Online softmax has an independent correctness test.
- FP32 and at least one lower-precision configuration have a numerical report.
- Benchmarks preserve hardware, compiler, CUDA, flags, and workload.
- The README explains why one version is faster rather than only reporting numbers.

Related modules:

- [C/C++ foundations](../modules/00-c-cpp-foundations.en.md)
- [GPU programming](../modules/02-gpu-programming.en.md)
- [Numerical computing](../modules/03-numerical-computing.en.md)
- [LLM inference](../modules/05-llm-inference.en.md)
