# Project 01 · Attention from Scratch

[中文](01-attention-from-scratch.md) · **English** · [Project ladder](README.en.md)

> Reading time: ~5 minutes · Level: Intermediate → Advanced · Freshness: Evolving · Last reviewed: 2026-08

## Goal

Starting from matrix multiplication and stable softmax, implement verifiable scaled dot-product attention and then progress through multi-head layouts, GQA, CUDA, tiling, online softmax, and KV cache.

The project preserves an evidence chain:

```text
mathematical definition
→ Python oracle
→ CPU C++ reference
→ correctness tests
→ CUDA kernels
→ profiler
→ numerical and performance report
```

## Mathematical boundary

```text
Q: [B, H, Nq, D]
K: [B, Hkv, Nk, D]
V: [B, Hkv, Nk, Dv]

S = QKᵀ / √D
P = softmax(S + mask)
O = PV
```

Begin with `H = Hkv`, `Nq = Nk`, FP32, and no mask. Every new dimension or optimization continues to validate against the same reference.

## Four short notes

| Note | One question only |
| --- | --- |
| [01 · Reference, shapes, and masks](../notes/attention/01-reference-shapes-masks.en.md) | What exactly does attention compute, and how are edges verified? |
| [02 · CPU to tiled CUDA](../notes/attention/02-cpu-to-tiled-cuda.en.md) | How can optimization remain debuggable? |
| [03 · Online softmax and IO-aware attention](../notes/attention/03-online-softmax.en.md) | Why avoid materializing the `N×N` matrix? |
| [04 · Precision, KV cache, and benchmarks](../notes/attention/04-precision-kv-benchmark.en.md) | How are lower precision and decode paths validated? |

Each note targets five minutes. The implementation can grow over time without accumulating every concept on one page.

## Suggested layout

```text
attention-from-scratch/
├── CMakeLists.txt
├── include/attention/
├── src/
│   ├── attention_cpu.cpp
│   ├── attention_cuda.cu
│   └── online_softmax.cu
├── reference/attention_reference.py
├── tests/
├── bench/
├── results/
└── README.md
```

Reuse tensor, matmul, and softmax from [P00 C/C++ Tensor Core](00-c-cpp-tensor-core.en.md) if useful.

## Milestone path

1. **Oracle:** explicit PyTorch matmul, mask, softmax, and matmul.
2. **CPU single-head:** materialize full score and probability matrices.
3. **Multi-head/GQA:** write shapes, strides, and KV-head mapping explicitly.
4. **Masks:** test causal, padding, additive, and combined masks separately.
5. **Optimized CPU:** improve loop order, tiling, and parallelism.
6. **Naive CUDA:** separately implement and verify QKᵀ, mask/scale, softmax, and PV.
7. **Tiled CUDA:** reuse shared-memory tiles and reduce HBM traffic.
8. **Online softmax:** maintain running maxima, normalizers, and output by block.
9. **Precision:** compare FP32 with BF16/FP16 inputs and FP32 accumulation.
10. **KV cache:** benchmark prefill and single-token decode separately.

## Evidence saved for every version

| Evidence | Contents |
| --- | --- |
| Correctness | absolute/relative error, softmax row sums, NaN/Inf |
| Shape | B, H/Hkv, Nq/Nk, D/Dv, layout, mask |
| Performance | latency, throughput, peak memory, HBM traffic |
| Environment | GPU, CPU, compiler, CUDA, flags, commit |
| Explanation | Why is it faster, and where did the bottleneck move? |

## Definition of done

- The CPU reference passes fixed and randomized tests.
- Masks, GQA, and non-tile-multiple shapes have boundary tests.
- Each naive CUDA stage can be compared independently with the reference.
- At least one tiled or fused version has profile evidence of a reduced bottleneck.
- Online softmax has an independent correctness test.
- At least one lower-precision configuration has a numerical report.
- Prefill and decode are benchmarked separately.
- The README explains results instead of only pasting speed numbers.

Related modules: [C/C++](../modules/00-c-cpp-foundations.en.md) · [GPU](../modules/02-gpu-programming.en.md) · [Numerics](../modules/03-numerical-computing.en.md) · [Inference](../modules/05-llm-inference.en.md)
