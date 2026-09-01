# Module 02 · GPU Programming and Operator Optimization

[中文](02-gpu-programming.md) · **English** · [Back to AI Infra](../README.en.md)

> Reading time: ~5 minutes · Level: Foundation · Freshness: Stable concepts / Evolving hardware · Last reviewed: 2026-08

## Core problem

GPU performance requires more than launching additional threads. This module studies how a kernel maps onto the GPU and how computation, memory access, synchronization, and scheduling jointly determine performance.

## Learning goals

- Explain grids, blocks, warps, threads, and Streaming Multiprocessors.
- Draw the data path through registers, shared memory, L2, and HBM.
- Recognize uncoalesced access, warp divergence, and synchronization overhead.
- Explain tiling, occupancy, arithmetic intensity, and kernel fusion.
- Distinguish library calls, generated kernels, Triton kernels, and handwritten CUDA.
- Use profiling evidence to decide whether an operator is compute-bound or bandwidth-bound.

## Core notes

### The SIMT execution model

CUDA expresses work as threads, organizes cooperating threads into blocks, and groups all blocks for a kernel into a grid. Hardware usually schedules threads as warps; an NVIDIA warp normally contains 32 threads.

When threads in one warp take different branches, the hardware must execute the paths separately while masking inactive threads. This is warp divergence. Different paths across separate warps do not create the same penalty.

### SM residency

One SM can host multiple blocks and warps. Each block consumes registers and shared memory; high per-block resource use can reduce the number of resident warps.

Occupancy measures active warps relative to a hardware limit, but higher occupancy does not guarantee a faster kernel. The real goal is enough independent work to hide latency without unnecessary movement or instructions.

### Memory access

```text
registers → shared memory / L1 → L2 → HBM
```

- Registers are fastest but limited.
- Threads in one block explicitly cooperate through shared memory.
- HBM is large but expensive to access.
- Neighboring threads accessing neighboring addresses enables memory coalescing.
- Tiling keeps reused data in a closer memory level.

### Tensor Cores and matrix multiplication

Tensor Cores accelerate small matrix multiply-accumulate operations. A high-performance GEMM partitions a large matrix into thread-block, warp, and smaller compute tiles, staging data through global memory, shared memory, and registers.

### Streams and asynchronous execution

Kernel launches, memory copies, and some communication can be enqueued asynchronously in CUDA streams. Different streams may overlap, but dependencies must be represented with events or synchronization. Excessive synchronization leaves CPUs, GPUs, or links idle.

### The modern compiler ladder

Do not begin every optimization by writing CUDA. First identify the lowest layer that needs intervention:

```text
framework graph (`torch.compile` / Inductor)
→ kernel DSL (Triton)
→ template library (CUTLASS)
→ handwritten CUDA
```

Higher layers provide faster iteration and portability; lower layers expose more control over layouts, instructions, and scheduling. Graph breaks, shape guards, or recompilation can erase compiler gains, while a custom kernel can lose end-to-end by adding copies or launch overhead. Compare the whole workload before moving down the ladder.

## Key calculations

Arithmetic intensity is:

```text
arithmetic intensity = floating-point operations / bytes moved
```

Combining it with hardware compute throughput and memory bandwidth in a roofline model indicates whether computation or bandwidth is the likely limit.

A basic lower bound is:

```text
time ≥ max(FLOPs / compute throughput, bytes / memory bandwidth)
```

## Hands-on work

1. Implement vector addition and compare contiguous with strided access.
2. Implement parallel reduction with block-level synchronization.
3. Implement tiled matrix multiplication with and without shared memory.
4. Write a fused softmax or normalization operator in Triton.
5. Run the same operator eagerly and through `torch.compile`; inspect graph breaks, recompilation, generated kernels, and end-to-end time.
6. Use Nsight or PyTorch Profiler to record kernel time, bandwidth, and launch gaps.

## Common misconceptions

- High GPU utilization does not prove Tensor Cores or HBM are used effectively.
- Maximum occupancy is not always optimal.
- A faster individual kernel may not improve the end-to-end system.
- Small operators may be dominated by launch overhead.
- Optimizing without understanding tensor layout often targets the wrong data path.

## Learning check

- Why does divergence within one warp reduce efficiency?
- How does tiling reduce HBM traffic?
- Why can increased register use lower occupancy?
- What does kernel fusion improve, and what costs can it add?
- What evidence distinguishes compute-bound from memory-bound execution?

Current references: [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) · [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)

Next: [Module 03 · Numerical Computing](03-numerical-computing.en.md)
