# AI Infrastructure Project Ladder

[中文](README.md) · **English** · [Back to AI Infra](../README.en.md)

## Why projects are separate

Understanding a concept only proves that it has been encountered. A project requires correctness, performance, numerical stability, observability, and reproducibility to work together. Every project should produce four forms of evidence:

```text
correctness tests
performance benchmark
profiling evidence
written analysis
```

## Project ladder

| Project | Main skills | Final output |
| --- | --- | --- |
| [P00 · C/C++ Tensor Core](00-c-cpp-tensor-core.en.md) | pointers, RAII, layouts, testing, benchmarking | a small CPU tensor library |
| [P01 · Attention from Scratch](01-attention-from-scratch.en.md) | matmul, softmax, masks, CUDA, online softmax | CPU/CUDA attention and an analysis report |
| P02 · CUDA Kernel Pack | reduction, normalization, fusion, profiling | kernels validated against PyTorch |
| P03 · Mixed Precision Lab | BF16, FP8/INT8, scaling, error | a precision-throughput-memory report |
| P04 · Distributed Training Lab | DDP, FSDP, collectives, timelines | a multi-GPU scaling report |
| P05 · LLM Serving Benchmark | KV cache, batching, TTFT, TPOT | a serving dashboard |
| [P06 · Self-Improving Service](../modules/08-capstone.en.md) | tracing, evaluation, SFT, canary, rollback | a complete learning loop |

The first P02–P05 exercises live in their corresponding modules. They can become dedicated project pages when the experiments grow deep enough.

## Shared project contract

Every project should contain:

- `README`: objective, hypotheses, design, and results;
- `src/`: implementation;
- `tests/`: correctness and edge cases;
- `bench/`: fixed workload distributions and a benchmark harness;
- `results/`: machine, compiler, flags, hardware, and raw results;
- `notes/`: profiles, bottleneck explanations, and next steps.

Use an explicit experiment table:

| Version | Input shape | Dtype | Correct? | Latency | Throughput | Peak memory | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Benchmark discipline

1. Establish correctness before comparing speed.
2. Exclude initialization and warm-up from steady-state timing.
3. Prevent the compiler from deleting unobserved computation.
4. Fix random seeds and workload distributions.
5. Report a median and a tail or distribution, not only the best run.
6. Preserve hardware, compiler, library, and flag versions.
7. Change only a small number of variables in each optimization.
8. Use a profiler to explain changes rather than merely recording them.

## Recommended order

```text
C/C++ tensor primitives
→ CPU attention
→ CUDA operators
→ tiled / online-softmax attention
→ mixed precision
→ multi-GPU training
→ LLM serving
→ continual-learning loop
```

Do not skip the CPU reference. It may be slow, but it becomes the correctness oracle for CUDA, low-precision, and fused implementations.
