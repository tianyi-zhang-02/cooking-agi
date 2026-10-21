# CUDA: threads, memory, and kernels

[中文](cuda.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

When you write `c = a + b` in PyTorch, what actually happens on the GPU? The CPU hands the GPU a function called a kernel, and the GPU starts thousands of threads at once, each computing one or two elements. The addition itself is cheap. **Whether a kernel is fast mostly depends on whether each warp uses the bytes it fetches, and on whether the same data is read from device memory again and again.** This note works through both questions with a vector add and a 4×4 matrix multiply, then connects them to cuBLAS, CUTLASS, and Triton.

The rules below were checked against NVIDIA's [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) v13.4.2 (updated 2026-09-10; formerly the CUDA C++ Programming Guide, "the Guide" below) and the [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 13.4 ("BPG" below). For why GPUs are built this way, see [GPU architecture](gpu-architecture.en.md); for how to tell whether a kernel is limited by compute or by bandwidth, see the next note, [Performance](performance.en.md).

## The CPU launches, the GPU executes {#host-device}

CUDA splits the system in two: the CPU and its memory are the host, and the GPU and its memory are the device. A program always starts on the CPU. Host code allocates device memory, copies data between the two sides, starts functions on the GPU, and waits for the results. A function that runs on the GPU and is started from the host is a kernel; starting it is called a launch ([Guide §1.2.1](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#heterogeneous-systems)).

A launch is **asynchronous** with respect to the CPU: when `kernel<<<...>>>(...)` returns, the kernel may not even have started ([Guide §2.1.2](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#launching-kernels)). If you time it without synchronizing first, you may only measure the time to put the work in a queue. Synchronization and timing in a PyTorch training loop are covered in [From batches to a training loop](../00-foundations/pytorch/training-loop.en.md).

## How threads are grouped: thread, warp, block, grid {#thread-hierarchy}

One launch creates one grid. A grid consists of many thread blocks of the same size, and a block contains threads. During execution, every 32 threads of a block form a warp that executes the same instruction together (SIMT).

| Level | Size | Where it lives | What it can do |
| --- | --- | --- | --- |
| thread | 1 | its own registers | knows who it is through `threadIdx` |
| warp | 32 threads | one SM | one instruction at a time; when threads branch differently, the ones not taking the branch are masked off |
| block | up to 1024 threads | the whole block runs on one SM | shares shared memory and synchronizes with `__syncthreads()` |
| cluster (optional) | several blocks, portable maximum 8 | one GPC | blocks can read and write each other's shared memory; only from compute capability 9.0 |
| grid | all blocks | spread across the SMs | no ordering guarantee between blocks, so they must not wait on each other |

These rules come from the Guide's sections on the [programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids) and [thread block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters). Compute capability 9.0 is Hopper (for example the H100); older GPUs have no cluster level. An SM (streaming multiprocessor) is the unit that executes blocks; according to the [H100 architecture whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf), the H100 SXM5 has 132 of them.

In one dimension, every thread computes the index it owns with the same formula. Let $b$ be `blockIdx.x`, $D$ be `blockDim.x`, and $t$ be `threadIdx.x`:

$$
i=b\cdot D+t.
$$

Take $n=1000$ elements and 256 threads per block. You need $\lceil 1000/256\rceil=4$ blocks, 1024 threads in total. In the third block ($b=2$), the thread with $t=5$ handles $2\times256+5=517$. The last block covers 768–1023; the 24 threads for 1000–1023 are out of range and must be skipped with `if (i < n)`. They all sit in that block's last warp (992–1023), so only 8 lanes of that warp do work. Block sizes are chosen as multiples of 32 precisely to avoid a half-empty warp at the end of every block ([Guide §1.2.2.2](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)).

## A complete vector add {#vector-add}

Below is a minimal program that compiles on its own (`nvcc vec_add.cu -o vec_add`). Its structure follows the explicit memory management example in Guide [§2.1.3](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#explicit-memory-management). It was not compiled or run on a GPU for this note.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1000;
    const size_t bytes = n * sizeof(float);

    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);
    for (int i = 0; i < n; ++i) { h_a[i] = i; h_b[i] = 2.0f * i; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("c[999] = %f\n", h_c[999]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    return 0;
}
```

| Code | What it does |
| --- | --- |
| `__global__ void vecAdd(...)` | `__global__` marks a kernel: launched from the host, run on the device, and it must return `void` |
| `int i = blockIdx.x * blockDim.x + threadIdx.x;` | each thread computes its own index with the formula above |
| `if (i < n)` | bounds check; without it, threads 1000–1023 would read and write outside the arrays |
| `cudaMallocHost(&h_a, bytes)` | allocates page-locked (pinned) host memory; asynchronous copies later depend on it |
| `cudaMalloc(&d_a, bytes)` | allocates device global memory; host code cannot dereference this pointer |
| `cudaMemcpy(..., cudaMemcpyHostToDevice)` | a synchronous copy that returns only when it is done |
| `int blocks = (n + threads - 1) / threads;` | integer division rounded up, which gives 4 here |
| `vecAdd<<<blocks, threads>>>(...)` | the launch: 4 blocks of 256 threads; the call returns immediately |
| `cudaGetLastError()` | only says the launch parameters were valid, not that the kernel has finished |
| `cudaDeviceSynchronize()` | waits for all submitted GPU work and returns execution errors such as an out-of-bounds access |
| `cudaMemcpy(..., cudaMemcpyDeviceToHost)` | copies the result back; `h_c[999]` should be $999+1998=2997$ |
| `cudaFree` / `cudaFreeHost` | free the device memory and the pinned host memory respectively |

The error handling follows Guide [§2.1.7](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#error-checking-in-cuda): a kernel launch returns no error code, and execution errors surface only at the next synchronization or API call. Production code should check the return value of every CUDA API call; to keep this short, only the kernel is checked.

For every element this kernel reads 8 bytes, writes 4 bytes, and does one addition. Its speed is set almost entirely by memory bandwidth; [the performance note](performance.en.md#three-ops) turns that sentence into numbers.

## Memory: where it lives and who can see it {#memory-spaces}

| Memory | Who can access it | Lifetime | Physical location | Amount on H100 SXM5 |
| --- | --- | --- | --- | --- |
| register | one thread | while the kernel runs | inside the SM | 64K 32-bit registers per SM (256 KB), at most 255 per thread |
| local | one thread | while the kernel runs | device memory | used only when registers overflow (spill) or arrays use dynamic indices; as slow as global |
| shared | one block (and the other blocks of its cluster) | while the kernel runs | inside the SM, sharing storage with L1 | up to 228 KB per SM, up to 227 KB per block |
| constant | the whole grid, read-only in kernels | the whole application | device, read through the constant cache | 64 KB in total |
| global | the whole grid, plus host-side APIs | until freed | device memory (HBM3) | 80 GB, peak bandwidth 3352 GB/s |

The first four columns come from the Guide's [memory spaces section](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#gpu-device-memory-spaces). In the last column, the register, shared memory and constant limits are the compute capability 9.0 column of [Guide appendix 5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications); memory size and bandwidth are the H100 SXM5 entries in Table 3 of the [H100 architecture whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf). A block that uses more than 48 KB of shared memory must allocate it dynamically and opt in explicitly.

For how the levels compare in bandwidth and capacity, see [GPU architecture](gpu-architecture.en.md). In addition, each SM has an L1 cache that shares one unified data cache with shared memory, and the split can be adjusted per kernel; the whole GPU shares one L2 (50 MB on the H100 SXM5). Caches are managed by hardware, while shared memory is read and written explicitly by the programmer, so you can think of it as a cache you manage yourself.

The most common step in writing a kernel is **to move data that will be reused from global memory into shared memory or registers and reuse it on chip**. The next two sections answer two questions: does a single transfer waste bytes, and how many times is the same data transferred.

## Coalescing: how many bytes a warp reads at once {#coalescing}

Global memory is transferred in **32-byte** units. When a warp executes one load, the hardware merges (coalesces) the addresses of its 32 threads and issues as many transactions as there are 32-byte segments involved (compute capability 6.0 and later; see [BPG §10.2.1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) and [Guide §2.3.4.1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access)). The table assumes each thread reads one element and the array starts on a 32-byte boundary (`cudaMalloc` guarantees at least 256-byte alignment):

| Access pattern | Byte offsets read by the 32 threads | 32-byte segments | Bytes moved | Fraction used |
| --- | --- | ---: | ---: | ---: |
| float32, neighboring threads read neighboring elements (stride 1) | 0, 4, 8, …, 124 | 4 | 128 B | 100% |
| float32, stride 1 but shifted by one element | 4, 8, …, 128 | 5 | 160 B | 80% |
| float32, stride 2 | 0, 8, …, 248 | 8 | 256 B | 50% |
| float32, stride 32 | 0, 128, …, 3968 | 32 | 1024 B | 12.5% |
| BF16, stride 1 | 0, 2, …, 62 | 2 | 64 B | 100% |

With stride 32, neighboring threads are 128 bytes apart and each thread gets a 32-byte segment to itself: 1024 bytes are moved and only 128 are used. Reading a column of a row-major matrix with at least 32 floats per row produces exactly this pattern; in the Guide's [matrix transpose example](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#matrix-transpose-example-using-global-memory), the writes of the unoptimized version look like this.

Coalescing does not require "thread k reads element k". It only requires the addresses of a warp to fall in as few 32-byte segments as possible; permuting them within the segments still costs four transactions (BPG). Also, the table counts transactions, not time: neighboring warps may reuse extra data that is already in cache. BPG measured the shifted copy on a V100 and saw bandwidth drop to about 9/10 of the aligned case, not as low as the 4/5 the segment count would suggest.

## Shared memory tiling: don't read the same data from global memory many times {#tiling}

Take a matrix multiply $C=AB$ where all three matrices are $n\times n$. The most direct kernel lets each thread compute one $C_{ij}$: it reads row $i$ of $A$ and column $j$ of $B$, $n$ elements each.

Take $n=4$. Each of the 16 outputs reads 8 elements, so there are **128** global reads in total, but $A$ and $B$ together hold only 32 distinct elements, so each one is read 4 times. For example, $A_{ik}$ is read once by each of the 4 outputs in row $i$.

Tiling cuts $C$ into $T\times T$ tiles and gives each block one tile. Take $T=2$: 4 blocks of $2\times2$ threads. The summation is split into $n/T=2$ phases. In each phase the block first moves one $2\times2$ sub-tile of $A$ and one of $B$ into shared memory (4 reads each), synchronizes, and then does the multiply-adds on chip.

| Scheme | Global reads per block | Blocks | Total reads |
| --- | ---: | ---: | ---: |
| no tiling (each thread reads one row and one column) | — | — | $2n^3=128$ |
| tiling with $T=2$ | 2 phases × (4 + 4) = 16 | 4 | $2n^3/T=64$ |

<!-- widget:step-tiling-reads -->

In general, tiling cuts global reads from $2n^3$ to $2n^3/T$: each element is read $n/T$ times instead of $n$ times. With $n=4096$, $T=32$ and FP32, the untiled kernel issues about $1.37\times10^{11}$ reads (about 550 GB) and the tiled one about $4.3\times10^{9}$ (about 17 GB), while the three matrices themselves are only about 0.2 GB.

```cpp
#define TILE 32

__global__ void tiledMatMul(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < n / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * n + col] = sum;
}
```

This is a teaching version: it assumes $n$ is a multiple of `TILE` and is launched with `dim3 block(TILE, TILE)` and `dim3 grid(n / TILE, n / TILE)`. The first `__syncthreads()` ensures the whole tile is loaded before anyone uses it; the second ensures every thread is done with it before the next phase overwrites it. When reading `A` and `B`, `threadIdx.x` lands in the lowest address dimension, so neighboring threads read neighboring addresses and the loads themselves are coalesced.

BPG applies the same idea to $C=AB$ ([§10.2.3.2](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)). On a Tesla V100 it raises effective bandwidth from 119.9 GB/s to 195.5 GB/s, and it notes that the gain comes from removing redundant reads, not from better coalescing. Reads drop to 1/32, yet the kernel is nowhere near 32 times faster: caches were already absorbing part of the repeated reads, and shared memory has its own issues. It is split into 32 banks, and when a warp hits different addresses in the same bank, the accesses are split up and serialized. A read count explains the mechanism; it does not predict time.

Real GEMM libraries keep going down the hierarchy: block-level tiles in shared memory, warp-level tiles in registers, the innermost step on Tensor Core MMA instructions, and the transfer of the next tile hidden behind the computation on the current one ([CUTLASS's description of the GEMM hierarchy](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)). FlashAttention applies the same tiling idea to attention, combined with an online softmax; see [FlashAttention and PagedAttention: what does each save?](../00-foundations/deep-dives/attention-kernels.en.md).

## Occupancy: how many warps fit on an SM at once {#occupancy}

While one warp waits for data from global memory, the SM can switch to another warp. The more warps are resident at once, the better the chance of hiding that wait; the hardware side of this latency hiding is covered in [GPU architecture](gpu-architecture.en.md). **Occupancy = active warps on an SM ÷ the maximum number of warps that SM can hold** ([Guide §2.3.7](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy)).

Whether an SM can take one more block depends on which resource runs out first. The H100 is compute capability 9.0, and [Guide appendix 5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications) gives these per-SM limits:

| Resource | Per-SM limit (compute capability 9.0) |
| --- | ---: |
| resident threads | 2048 (64 warps) |
| resident blocks | 32 |
| 32-bit registers | 65536 |
| shared memory | 228 KB |

With 256 threads per block (8 warps), divide resource by resource:

| Kernel usage | Thread limit | Register limit | Shared limit | Resident blocks | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 registers per thread, no shared | 2048/256 = 8 | 65536/(32×256) = 8 | — | 8 | 64/64 = 100% |
| 64 registers per thread, no shared | 8 | 65536/(64×256) = 4 | — | 4 | 32/64 = 50% |
| 32 registers per thread, 48 KB shared per block | 8 | 8 | ⌊228/48⌋ = 4 | 4 | 32/64 = 50% |
| 80 registers per thread, no shared | 8 | ⌊65536/20480⌋ = 3 | — | 3 | 24/64 = 37.5% |

A few lines of Python reproduce the table:

```python
def resident_blocks(threads, regs_per_thread, smem_kb_per_block):
    limits = [32, 2048 // threads, 65536 // (regs_per_thread * threads)]
    if smem_kb_per_block:
        limits.append(228 // smem_kb_per_block)
    return min(limits)

for regs, smem in [(32, 0), (64, 0), (32, 48), (80, 0)]:
    blocks = resident_blocks(256, regs, smem)
    print(regs, smem, blocks, blocks * 256 // 32 / 64)
```

This estimate leaves out two details. First, registers are allocated per warp in units of 256 (explained in BPG [§11.1.1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy) with a compute capability 7.0 example); the register counts above were chosen so that each warp's usage is a multiple of 256. Second, the shared memory a block can use (227 KB) is 1 KB less than the per-SM maximum (228 KB), and the [CUDA 12.4 edition of the Guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#compute-capability-9-0) says that 1 KB is reserved for system use; counting 1 KB more per block (49 KB) still gives 4 blocks, so the conclusion holds. For a real kernel, first read its register and shared memory usage with `nvcc --resource-usage`, then check with the Nsight Compute occupancy calculator or `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

Higher occupancy is not always better. BPG states that beyond some point more occupancy does not make a kernel faster, and that fewer resident warps with more registers per thread can sometimes hide latency through instruction-level parallelism ([BPG §11.1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)). Occupancy answers "are there enough warps to switch between", not "is this kernel fast".

## Streams: overlapping copies with computation {#streams}

A stream is a work queue executed in order: copies and kernels in the same stream complete one after another, while different streams can interleave or even run at the same time ([Guide §2.5.2](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)). For a host–device copy to actually overlap with a kernel, three things must hold together: the host buffer is pinned memory, the copy uses `cudaMemcpyAsync`, and the copy and the kernel are in different non-default streams. These conditions come from [BPG §10.1.2](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation). With ordinary pageable memory, `cudaMemcpyAsync` still works correctly but falls back to synchronous behavior and no longer overlaps with other work ([Guide §2.5.2.3](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#launching-memory-transfers-in-cuda-streams)).

BPG's staged pattern cuts the data into `nStreams` chunks and puts each chunk's "copy → compute" into its own stream. The code below assumes `h_x` is pinned, `stream[]` was created with `cudaStreamCreate`, and `n` is divisible by `nStreams * 256`:

```cpp
size_t chunk = n / nStreams;
for (int s = 0; s < nStreams; ++s) {
    size_t off = s * chunk;
    cudaMemcpyAsync(d_x + off, h_x + off, chunk * sizeof(float),
                    cudaMemcpyHostToDevice, stream[s]);
    process<<<chunk / 256, 256, 0, stream[s]>>>(d_x + off, chunk);
}
cudaDeviceSynchronize();
```

Work it out once: copying everything takes $t_T=8$ ms and computing everything takes $t_E=10$ ms, so doing them in sequence takes 18 ms. With 4 chunks, computation can start as soon as the first chunk arrives, and the remaining copies hide behind computation. BPG's rough estimate is $t_E+t_T/n_{\rm streams}=10+2=12$ ms (for $t_E>t_T$; otherwise roughly $t_T+t_E/n_{\rm streams}$). The milliseconds are teaching assumptions. Whether copies in both directions can run at once also depends on how many copy engines the device has (`asyncEngineCount`).

In PyTorch, the equivalent is a DataLoader with `pin_memory=True` followed by `.to(device, non_blocking=True)` ([CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#use-pinned-memory-buffers)). Tensor lifetime pitfalls with multiple streams are covered in [From batches to a training loop](../00-foundations/pytorch/training-loop.en.md).

"Move while computing" also happens inside kernels: from compute capability 8.0, copies from global to shared memory can be issued asynchronously (LDGSTS), and from 9.0 the Tensor Memory Accelerator (TMA) moves sub-tiles of multidimensional arrays in bulk ([Guide §4.12](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)). Kernels such as GEMM and FlashAttention use these to hide the transfer of the next tile behind the computation on the current one.

## Beyond hand-written kernels: cuBLAS, cuDNN, CUTLASS and Triton {#libraries}

Everyday training and inference rarely need hand-written CUDA. [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) is a BLAS implementation on top of the CUDA runtime (with further interfaces such as cuBLASLt), and PyTorch uses cuBLAS by default for BLAS operations on CUDA (see the [`preferred_blas_library` documentation](https://docs.pytorch.org/docs/2.14/backends.html)). [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html) provides deep learning primitives such as attention, convolution, matrix multiplication and normalization, and can fuse several operations expressed as a graph. [CUTLASS](https://github.com/NVIDIA/cutlass) is an open-source library of CUDA C++ templates that breaks the hierarchical tiling above into composable parts; since CUTLASS 4 it also has the Python CuTe DSL; when checked on 2026-10-21, the version at the top of its README was 4.8.1 (October 2026). The Guide also recommends using Tensor Cores through these libraries rather than through inline PTX ([appendix 5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications)).

[Triton](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) ([Tillet, Kung, and Cox, MAPL 2019](https://doi.org/10.1145/3315508.3329973)) moves the abstraction up one level: the programmer writes a program per block that operates on whole blocks of data, and the compiler handles coalescing, shared memory allocation and synchronization, scheduling of asynchronous copies, and selection of Tensor Core instructions. TorchInductor, the default backend of PyTorch's `torch.compile`, generates kernels with Triton on NVIDIA, AMD and Intel GPUs ([PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html)). CUDA itself added a similar tile programming model in its v13 documentation: cuTile for Python, and CUDA Tile C++ from Toolkit 13.3 onward ([Guide §2.4](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html)). Whichever layer you use, the three questions of this note remain: does each warp use the bytes it moves, how many times is data reused, and does the SM have enough warps to switch between. How frameworks wire these libraries together is covered in [Frameworks](frameworks.en.md).

## What to try first {#try}

| What to try | What you should observe | What you can NOT conclude |
| --- | --- | --- |
| change the vector add block from 256 threads to 250 | each block ends with a partly empty warp, and the result is still correct | that it must be noticeably slower; when bandwidth-bound the difference may be small |
| make the reads stride 32 | Nsight Compute shows close to 32 sectors/request for global loads (the ideal for 32-bit accesses is 4) | that measured time is exactly 8 times slower; caches change the result |
| change `TILE` from 16 to 32 | global reads halve again, and shared memory per block becomes 4 times larger | that it must be faster; occupancy and bank conflicts change too |
| lower register usage with `-maxrregcount` | occupancy rises, and local memory spills may appear | that higher occupancy must be faster |
| replace pinned memory with plain `malloc` | `cudaMemcpyAsync` no longer overlaps with the kernel | that every copy slows down by the same factor |

The ideal sectors/request value is taken from the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). These experiments were not run on a GPU for this note; the second column is the expectation derived from the documents cited above and should be checked on your own hardware.

Continue with [Performance: the roofline, arithmetic intensity, and MFU](performance.en.md) to turn "how many bytes were moved" into a lower bound on time.
