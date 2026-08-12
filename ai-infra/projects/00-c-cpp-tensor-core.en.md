# Project 00 · Build a Small Tensor Core in C/C++

[中文](00-c-cpp-tensor-core.md) · **English** · [Project ladder](README.en.md)

> Reading time: ~5 minutes · Level: Foundation → Intermediate · Freshness: Stable · Last reviewed: 2026-08

## Project goal

Implement a small CPU-only tensor library. The first version uses no BLAS, Eigen, or deep-learning framework so that shape, stride, ownership, loop order, and numerical stability remain explicit.

Here “Tensor Core” means a personal learning library, not NVIDIA Tensor Core hardware.

## Final capability

```text
tensor storage and views
+ elementwise operators
+ reduction
+ matrix multiplication
+ stable softmax
+ tests and benchmarks
```

The result becomes the foundation for [Attention from Scratch](01-attention-from-scratch.en.md).

## Suggested layout

```text
cpp-tensor-core/
├── CMakeLists.txt
├── include/tiny_tensor/
│   ├── tensor.h
│   └── ops.h
├── src/
│   ├── tensor.cpp
│   └── ops.cpp
├── tests/
├── bench/
├── results/
└── README.md
```

## Milestone 0 · C primitives

First implement in C:

- vector addition;
- scalar multiplication;
- dot product;
- sum and max reduction;
- stable softmax;
- naive matrix multiplication.

Interfaces explicitly pass pointers, lengths, shapes, and strides. Every allocation and release needs tests and must run under AddressSanitizer.

Success criterion: random small inputs match a Python/NumPy or PyTorch reference within tolerance, with no sanitizer errors.

## Milestone 1 · C++ tensor ownership

Implement an owning `Tensor` with:

- contiguous row-major storage;
- shape and stride;
- `numel()`, `data()`, and bounds-checked indexing;
- `const` and non-`const` access;
- memory managed by `std::vector<float>`;
- the Rule of Zero.

Explicitly distinguish:

- owning tensors;
- borrowed views;
- mutable views;
- `const` views.

Support only `float` in the first version; generalize with templates after correctness.

## Milestone 2 · Shape, stride, and views

Add:

- reshape when element counts agree;
- transpose views that change strides without copying;
- contiguous copies that materialize non-contiguous views;
- slice views with offset, shape, and stride.

Test view lifetime carefully: a view must not access storage after its owner is destroyed. The first version can enforce this through API constraints or shared ownership, but the choice must be explicit in the README.

## Milestone 3 · Operators

Implement:

- elementwise add and multiply;
- row-wise sum and max;
- stable row-wise softmax;
- `matmul(A, B)`;
- transpose-aware matmul;
- a causal-mask helper.

Softmax must use:

```text
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

Keep direct `exp(x_i)` only as a failing comparison.

## Milestone 4 · Correctness tests

Cover at least:

- defined behavior for empty dimensions and zero-sized tensors;
- incompatible shapes;
- non-contiguous views;
- very large and small softmax inputs;
- non-square matrix multiplication;
- aliased inputs;
- copy, move, and object destruction;
- randomized property tests.

Use absolute and relative tolerances rather than direct `==` for floating-point values.

## Milestone 5 · Performance experiments

Compare three matrix multiplications:

1. naive `i-j-k` loops;
2. reordered loops for contiguous access;
3. cache-blocked or tiled matmul.

Record shapes, compiler flags, median latency, GFLOP/s, and profiles. Explain why loop order affects cache locality.

Optional extensions:

- OpenMP parallelism;
- SIMD/vectorization reports;
- aligned allocation;
- BLAS as a performance upper-bound comparison;
- `float` / `double` templates;
- a pybind11 binding.

## Definition of done

- Debug, sanitizer, and release builds all run.
- Ownership and view-lifetime rules are explicit.
- Operators pass against a reference implementation.
- Benchmarks exclude initialization and warm-up.
- At least one optimization is supported by profiler evidence.
- The README explains correctness, complexity, layout, and bottlenecks.
- The library can directly support CPU attention in Project 01.

Next: [Project 01 · Attention from Scratch](01-attention-from-scratch.en.md)
