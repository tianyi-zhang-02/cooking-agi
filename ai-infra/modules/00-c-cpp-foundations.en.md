# Module 00 · C/C++ Foundations for AI Infrastructure

[中文](00-c-cpp-foundations.md) · **English** · [Back to index](../README.en.md)

> Reading time: ~5 minutes · Level: Foundation · Freshness: Stable · Last reviewed: 2026-08

## Why it matters

CUDA, the PyTorch runtime, NCCL, inference engines, and high-performance operators rely heavily on C and C++. The goal is not to memorize both languages. It is to answer four questions:

```text
Where does the data live?
Who owns it?
How long is it valid?
What does the compiler ultimately execute?
```

When these are unclear, the result is often out-of-bounds access, use-after-free, incorrect strides, data races, or an elegant abstraction with no performance.

## The minimum knowledge map

### C: facing memory directly

Learn:

- basic types, arrays, and structs;
- pointers, addresses, dereferencing, and pointer arithmetic;
- `const`, lengths, shapes, and strides;
- `malloc/calloc/realloc/free`;
- headers, source files, translation units, and linking;
- undefined behavior and sanitizers.

A C pointer carries no length or ownership. A `float*` identifies an address that may contain floats but does not say how many, whether they are mutable, or who releases them.

→ [Five minutes: Pointers, arrays, and contiguous memory](../notes/c-cpp/01-pointers-and-memory.en.md)

### C++: expressing ownership

Learn:

- references and `const`;
- `std::vector`, `std::array`, and `std::span`;
- classes, constructors, and invariants;
- RAII, `unique_ptr`, and the Rule of Zero;
- copy, move, borrow, and object lifetime;
- function templates and lambdas.

Modern C++ is valuable because it expresses resource lifetime with optimizable abstractions. Prefer standard containers and RAII; manage raw resources only when ownership is completely understood.

→ [Five minutes: RAII, copy, move, and views](../notes/c-cpp/02-ownership-and-raii.en.md)

### Build, debug, and performance evidence

Understand:

```text
source → preprocess → compile → object → link → executable
```

Keep at least three build configurations:

```text
debug:   -O0 -g -Wall -Wextra -Wpedantic
checked: -O1 -g -fsanitize=address,undefined
release: -O3 -DNDEBUG -march=native
```

Warnings, sanitizers, tests, debuggers, and profilers answer different questions. A release benchmark does not replace a checked build, and one successful run does not prove the absence of undefined behavior.

→ [Five minutes: Compilation, sanitizers, and benchmarks](../notes/c-cpp/03-build-debug-benchmark.en.md)

## Minimal practice path

1. Implement vector add, dot product, stable softmax, and naive matmul in C.
2. Pass lengths, shapes, and strides explicitly to pointer-based APIs.
3. Use AddressSanitizer to find intentionally introduced bounds and lifetime errors.
4. Rewrite a C++ `Tensor` using `std::vector` and RAII.
5. Compare matmul loop orders and explain the result with a profiler.
6. Split the implementation into library, test, and benchmark targets.

Full specification: [Project 00 · C/C++ Tensor Core](../projects/00-c-cpp-tensor-core.en.md)

## Learning check

- What information disappears when an array decays to a pointer?
- Does a raw pointer imply ownership?
- How do stack, heap, and object lifetime relate?
- Why does RAII cover exceptions and early returns?
- What do debug, sanitizer, and release builds each prove?
- Why does successful compilation not imply defined behavior?

Next: [Module 01 · Computer Systems](01-computer-systems.en.md)
