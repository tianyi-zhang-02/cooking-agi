# Compilation, Sanitizers, and Benchmarks

[中文](03-build-debug-benchmark.md) · **English** · [Back to C/C++ module](../../modules/00-c-cpp-foundations.en.md)

> Reading time: ~5 minutes · Level: Foundation · Freshness: Stable · Last reviewed: 2026-08

## The central question

Why does code compiling and running still fail to prove that it is correct or fast?

```text
source → preprocess → compile → object file → link → executable
```

Each `.c/.cpp` file is a translation unit. Headers provide declarations and template definitions; the linker resolves symbols and libraries across files. Compile errors, link errors, runtime failures, and performance regressions are different categories.

## Three builds

### Debug

```text
-O0 -g -Wall -Wextra -Wpedantic
```

Useful for breakpoints, variables, and readable stacks. It does not represent release behavior or performance.

### Checked

```text
-O1 -g -fsanitize=address,undefined
```

AddressSanitizer detects many bounds and use-after-free errors. UndefinedBehaviorSanitizer detects selected forms of UB. Multithreaded data races require a separate ThreadSanitizer build.

### Release

```text
-O3 -DNDEBUG -march=native
```

Used for performance measurement. Optimization can inline, reorder, vectorize, or delete computation with no observable result. Release failures may differ completely from debug failures.

## Minimum benchmark discipline

1. Establish correctness with a reference and tests.
2. Separate initialization, I/O, and warm-up from steady state.
3. Consume outputs to prevent dead-code elimination.
4. Fix input distributions, seeds, thread counts, and affinity.
5. Report a median and distribution, not only the best run.
6. Preserve CPU, compiler, flags, libraries, and commit.
7. Explain changes with a profiler.

A benchmark number is meaningful only when its measurement contract is explicit.

## Debuggers and profilers

- `gdb/lldb`: breakpoints, call stacks, memory, and variables;
- sanitizers: selected correctness violations;
- `perf` / Instruments: CPU sampling, caches, and hotspots;
- compiler reports: vectorization and inlining;
- Nsight: CUDA timelines and kernel metrics.

A profiler does not directly prescribe a fix. It provides evidence about where time and resources go; a systems model explains why.

## Hands-on check

1. Create CMake debug, checked, and release targets.
2. Add a bounds error and signed overflow; compare tool reports.
3. Benchmark three matmul loop orders.
4. Check whether the compiler vectorizes the inner loop.
5. Save an experiment table and explain why the fastest version wins.

## Key conclusions

Tests demonstrate behavior on covered inputs, sanitizers expose selected violations, profilers describe resource use, and benchmarks compare controlled workloads. They complement rather than replace one another.

Next: [Project 00 · C/C++ Tensor Core](../../projects/00-c-cpp-tensor-core.en.md)
