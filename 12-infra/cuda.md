# CUDA：线程、内存与 kernel

**中文** · [English](cuda.en.md)

> 阅读时间：约 13 分钟 · 最近审阅：2026-10

在 PyTorch 里写一句 `c = a + b`，GPU 上到底发生了什么？CPU 把一个叫 kernel 的函数交给 GPU，一次启动可以安排成千上万个线程，每个线程只算一两个元素。加法本身很便宜，数据怎样搬运却会显著影响性能。

对这类 kernel，值得先问：每个 warp 搬来的字节有没有用上？同一份数据有没有被反复从显存读出来？这一篇用一个 vector add 和一个 4×4 的矩阵乘法把这两件事算清楚，再接到 cuBLAS、CUTLASS 和 Triton。

文中的规则核对自 NVIDIA [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) v13.4.2（2026-09-10 更新，旧名 CUDA C++ Programming Guide，下称 Guide）和 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 13.4（下称 BPG）。GPU 为什么做成这样，见 [GPU 架构](gpu-architecture.md)；怎样判断一个 kernel 被算力还是带宽卡住，见下一篇[性能](performance.md)。

## CPU 发起，GPU 执行 {#host-device}

CUDA 把系统分成两边：CPU 和它的内存叫 host，GPU 和它的显存叫 device。程序总是从 CPU 开始运行。host 代码负责分配显存、在两边之间拷数据、启动 GPU 上的函数，再等结果。在 GPU 上执行、由 host 启动的函数叫 kernel，启动这个动作叫 launch（[Guide §1.2.1](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#heterogeneous-systems)）。

Launch 对 CPU 是**异步**的：`kernel<<<...>>>(...)` 返回时，kernel 可能还没开始跑（[Guide §2.1.2](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#launching-kernels)）。所以不先同步就计时，量到的可能只是把任务放进队列的时间。PyTorch 训练循环里的同步与计时问题，见[训练循环](../00-foundations/pytorch/training-loop.md)。

## 线程怎样分组：thread、warp、block、grid {#thread-hierarchy}

一次 launch 创建一个 grid；grid 由许多同样大小的 thread block 组成，block 里是 thread。执行时，block 内每 32 个线程编成一个 warp，一起执行同一条指令（SIMT）。

| 层级 | 大小 | 放在哪里 | 能做什么 |
| --- | --- | --- | --- |
| thread | 1 | 自己的寄存器 | 用 `threadIdx` 知道自己是谁 |
| warp | 32 个 thread | 同一个 SM | 同一条指令；分支不同时，不走这条分支的线程被 mask 掉 |
| block | 最多 1024 个 thread | 整个 block 在同一个 SM 上 | 共用 shared memory，用 `__syncthreads()` 同步 |
| cluster（可选） | 若干 block，可移植上限 8 | 同一个 GPC | 互相读写对方的 shared memory；compute capability 9.0 起才有 |
| grid | 全部 block | 分散到各个 SM | block 之间不保证执行顺序，不能互相等待 |

这些规定见 Guide 的[编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids)和 [thread block cluster](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters) 两节。Compute capability 9.0 对应 Hopper（例如 H100）；更早的 GPU 没有 cluster 这一层。SM（streaming multiprocessor）是 GPU 上执行 block 的单元；按 [H100 架构白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)，H100 SXM5 有 132 个。

一维时，每个线程用同一条公式算出自己负责的下标。记 $b$ 为 `blockIdx.x`，$D$ 为 `blockDim.x`，$t$ 为 `threadIdx.x`：

$$
i=b\cdot D+t.
$$

取 $n=1000$ 个元素、每个 block 256 个线程，需要 $\lceil 1000/256\rceil=4$ 个 block，一共 1024 个线程。第 3 个 block（$b=2$）里 $t=5$ 的线程负责 $2\times256+5=517$。最后一个 block 覆盖 768–1023，其中 1000–1023 这 24 个线程越界，要靠 `if (i < n)` 跳过；它们恰好都在这个 block 最后一个 warp（992–1023）里，所以这个 warp 只有 8 条 lane 在干活。Block 大小取 32 的倍数，就是为了不在每个 block 末尾留下半空的 warp（[Guide §1.2.2.2](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)）。

## 一个完整的 vector add {#vector-add}

下面是可以单独编译的最小程序（`nvcc vec_add.cu -o vec_add`），结构与 Guide [§2.1.3](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#explicit-memory-management) 的显式内存管理示例一致。本页没有在 GPU 上编译运行它。

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

| 代码 | 在做什么 |
| --- | --- |
| `__global__ void vecAdd(...)` | `__global__` 标出 kernel：由 host 启动、在 device 上运行，返回类型必须是 `void` |
| `int i = blockIdx.x * blockDim.x + threadIdx.x;` | 每个线程按上面的公式算出自己的下标 |
| `if (i < n)` | 越界保护；没有它，第 1000–1023 号线程会读写数组以外的内存 |
| `cudaMallocHost(&h_a, bytes)` | 在 host 上分配页锁定（pinned）内存；后面要做异步拷贝就离不开它 |
| `cudaMalloc(&d_a, bytes)` | 在 device 的 global memory 上分配；host 代码不能直接解引用这个指针 |
| `cudaMemcpy(..., cudaMemcpyHostToDevice)` | 同步拷贝，拷完才返回 |
| `int blocks = (n + threads - 1) / threads;` | 整数除法向上取整，这里得到 4 |
| `vecAdd<<<blocks, threads>>>(...)` | launch：4 个 block，每个 256 个线程；调用立即返回 |
| `cudaGetLastError()` | 只说明 launch 参数没出错，不代表 kernel 已经跑完 |
| `cudaDeviceSynchronize()` | 等所有已提交的 GPU 工作结束，同时取回执行期错误，例如越界访问 |
| `cudaMemcpy(..., cudaMemcpyDeviceToHost)` | 把结果拷回；`h_c[999]` 应为 $999+1998=2997$ |
| `cudaFree` / `cudaFreeHost` | 分别释放 device 内存和 pinned host 内存 |

错误检查的写法来自 Guide [§2.1.7](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#error-checking-in-cuda)：kernel launch 本身不返回错误码，执行期错误要等下一次同步或 API 调用才报出来。正式代码应检查每个 CUDA API 的返回值，这里为了短，只检查了 kernel。

这个 kernel 每个元素读 8 字节、写 4 字节，只做 1 次加法，算术强度很低。数组足够大、主要流量确实经过显存时，带宽通常是主要限制；示例里的小数组还可能受 launch 开销、并行度和 cache 影响，不能仅凭算术强度判断实测耗时。[性能篇](performance.md#three-ops)会用大数组把带宽下限算成数字。

## 内存：先分清放在哪里、谁能看见 {#memory-spaces}

| 内存 | 谁能访问 | 存活多久 | 物理位置 | H100 SXM5 上的量 |
| --- | --- | --- | --- | --- |
| register | 单个 thread | kernel 运行期间 | SM 内 | 每 SM 64K 个 32-bit 寄存器（256 KB），每线程最多 255 个 |
| local | 单个 thread | kernel 运行期间 | device 显存 | 寄存器放不下（spill）或数组用动态下标时才用，和 global 一样慢 |
| shared | 同一个 block（cluster 内可互访） | kernel 运行期间 | SM 内，与 L1 共用一块存储 | 每 SM 最多 228 KB，每 block 最多 227 KB |
| constant | 整个 grid，kernel 只读 | 整个应用 | device，经 constant cache 读取 | 共 64 KB |
| global | 整个 grid，以及 host 端 API | 直到释放 | device 显存（HBM3） | 80 GB，峰值带宽 3352 GB/s |

前四列来自 Guide 的[内存空间一节](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#gpu-device-memory-spaces)。最后一列的寄存器、shared memory 和 constant 上限取 [Guide 附录 5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications) 中 compute capability 9.0 一栏；显存容量与带宽取 [H100 架构白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf) 表 3 的 H100 SXM5。每个 block 用超过 48 KB 的 shared memory 时，必须改用动态分配并显式 opt-in。

各级存储的带宽和容量怎样对比，见 [GPU 架构](gpu-architecture.md)。此外，每个 SM 有 L1 cache，它和 shared memory 共用同一块 unified data cache，比例可以按 kernel 调整；全 GPU 共享一个 L2（H100 SXM5 为 50 MB）。Cache 由硬件管理，shared memory 则由程序员显式读写，可以当成一块“自己管的 cache”。

写 kernel 时最常见的一步，就是**把会被反复用到的数据从 global 搬进 shared memory 或寄存器，在片上复用**。下面两节分别回答：一次搬运有没有浪费；同一份数据被搬了几次。

## Coalescing：一个 warp 一次读多少字节 {#coalescing}

Global memory 以 **32 字节**为单位传输。一个 warp 执行同一条 load 时，硬件把 32 个线程的地址合并（coalesce），需要几个 32 字节段，就发几个 transaction（compute capability 6.0 及以上，见 [BPG §10.2.1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) 和 [Guide §2.3.4.1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access)）。下表假设每个线程读一个元素，数组起点按 32 字节对齐（`cudaMalloc` 至少保证 256 字节对齐）：

| 访问方式 | 32 个线程读到的字节偏移 | 32 字节段数 | 实际搬运 | 用上的比例 |
| --- | --- | ---: | ---: | ---: |
| float32，相邻线程读相邻元素（stride 1） | 0, 4, 8, …, 124 | 4 | 128 B | 100% |
| float32，stride 1 但起点错开 1 个元素 | 4, 8, …, 128 | 5 | 160 B | 80% |
| float32，stride 2 | 0, 8, …, 248 | 8 | 256 B | 50% |
| float32，stride 32 | 0, 128, …, 3968 | 32 | 1024 B | 12.5% |
| BF16，stride 1 | 0, 2, …, 62 | 2 | 64 B | 100% |

Stride 32 时，相邻线程相隔 128 字节，每个线程独占一个 32 字节段：搬了 1024 字节，只用上 128 字节。按列去读一个行优先（row-major）、每行至少 32 个 float 的矩阵，就是这种情况；Guide 的[矩阵转置例子](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#matrix-transpose-example-using-global-memory)里，未优化版本的写入正是如此。

Coalescing 不要求“第 k 个线程读第 k 个元素”。只要一个 warp 的地址落在尽量少的 32 字节段里，段内顺序打乱也还是 4 个 transaction（BPG）。另外，表里数的是 transaction，不是耗时：相邻 warp 可能复用 cache 里多拉进来的数据。BPG 在 V100 上测“起点错开”的拷贝，带宽降到不错开时的约 9/10，而不是按 4/5 推算的那么低。

## Shared memory tiling：同一份数据别从 global 读很多遍 {#tiling}

矩阵乘 $C=AB$，三个矩阵都是 $n\times n$。最直接的 kernel 让每个线程算一个 $C_{ij}$：读 $A$ 的第 $i$ 行和 $B$ 的第 $j$ 列，各 $n$ 个元素。

取 $n=4$。16 个输出各读 8 个元素，共 **128 次** global 读取；但 $A$ 和 $B$ 一共只有 32 个不同的元素，每个都被读了 4 遍。比如 $A_{ik}$ 会被第 $i$ 行的 4 个输出各读一次。

分块（tiling）把 $C$ 切成 $T\times T$ 的块，每个 block 负责一块。取 $T=2$：4 个 block，每个 block 有 $2\times2$ 个线程。沿求和方向分 $n/T=2$ 个阶段；每个阶段，block 先把 $A$ 的一个 $2\times2$ 子块和 $B$ 的一个 $2\times2$ 子块搬进 shared memory（各读 4 次），同步之后在片上做乘加。

| 方案 | 每个 block 的 global 读取 | block 数 | 总读取 |
| --- | ---: | ---: | ---: |
| 不分块（每个线程各读一行一列） | — | — | $2n^3=128$ |
| $T=2$ 分块 | 2 个阶段 ×（4 + 4）= 16 | 4 | $2n^3/T=64$ |

<!-- widget:step-tiling-reads -->

一般地，分块把 global 读取从 $2n^3$ 降到 $2n^3/T$：每个元素被读的次数从 $n$ 遍降到 $n/T$ 遍。$n=4096$、$T=32$、FP32 时，不分块要发出约 $1.37\times10^{11}$ 次读取（约 550 GB），分块后约 $4.3\times10^{9}$ 次（约 17 GB），而三个矩阵本身只有约 0.2 GB。

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

这是教学版：假设 $n$ 是 `TILE` 的倍数，用 `dim3 block(TILE, TILE)` 和 `dim3 grid(n / TILE, n / TILE)` 启动。第一个 `__syncthreads()` 保证整块搬完才开始用；第二个保证所有线程都用完了，下一阶段才覆盖。读 `A` 和 `B` 时，`threadIdx.x` 都落在地址的最低维，相邻线程读相邻地址，所以搬运本身也是 coalesced 的。

BPG 用同样的思路优化 $C=AB$（[§10.2.3.2](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)），在 Tesla V100 上把有效带宽从 119.9 GB/s 提到 195.5 GB/s，并指出提升来自去掉重复读取，而不是合并访问变好。读取次数降为原来的 1/32，速度却远没有快 32 倍：一部分重复读取本来就被 cache 挡住了；shared memory 也有自己的问题，它分成 32 个 bank，同一个 warp 访问同一 bank 的不同地址会被拆开串行执行。读取次数说明的是机制，不是耗时预测。

真正的 GEMM 库会继续分层：block 级的 tile 放 shared memory，warp 级的 tile 放寄存器，最内层交给 Tensor Core 的 MMA 指令，并把下一块的搬运藏到当前块的计算后面（[CUTLASS 的 GEMM 层次说明](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)）。FlashAttention 把同样的分块思想用到 attention 上，再配合 online softmax，见 [FlashAttention 和 PagedAttention](../00-foundations/deep-dives/attention-kernels.md)。

## Occupancy：一个 SM 上同时能放几个 warp {#occupancy}

一个 warp 等 global memory 的数据时，SM 可以切换去执行别的 warp。同时驻留的 warp 越多，越有机会把等待藏起来；这套延迟隐藏的硬件原理见 [GPU 架构](gpu-architecture.md)。**Occupancy = SM 上活跃的 warp 数 ÷ 这个 SM 最多能驻留的 warp 数**（[Guide §2.3.7](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy)）。

一个 SM 还能不能再放一个 block，看几项资源里谁先用完。H100 属于 compute capability 9.0，[Guide 附录 5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications) 给出的每 SM 上限是：

| 资源 | 每 SM 上限（compute capability 9.0） |
| --- | ---: |
| 驻留线程 | 2048（即 64 个 warp） |
| 驻留 block | 32 |
| 32-bit 寄存器 | 65536 |
| shared memory | 228 KB |

设每个 block 256 个线程（8 个 warp），逐项去除：

| kernel 的用量 | 线程限制 | 寄存器限制 | shared 限制 | 能驻留的 block | occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 每线程 32 个寄存器，不用 shared | 2048/256 = 8 | 65536/(32×256) = 8 | — | 8 | 64/64 = 100% |
| 每线程 64 个寄存器，不用 shared | 8 | 65536/(64×256) = 4 | — | 4 | 32/64 = 50% |
| 每线程 32 个寄存器，每 block 48 KB shared | 8 | 8 | ⌊228/48⌋ = 4 | 4 | 32/64 = 50% |
| 每线程 80 个寄存器，不用 shared | 8 | ⌊65536/20480⌋ = 3 | — | 3 | 24/64 = 37.5% |

可以用几行 Python 复算这张表：

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

这个估算省掉了两处细节。第一，寄存器按 warp 以 256 个为粒度分配（BPG [§11.1.1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy) 在 compute capability 7.0 的例子里说明）；上面选的寄存器数让每个 warp 的用量正好是 256 的倍数。第二，每个 block 能用的 shared memory（227 KB）比每个 SM 的上限（228 KB）少 1 KB，[CUDA 12.4 版 Guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#compute-capability-9-0) 说明这 1 KB 留给系统使用；按每个 block 多占 1 KB 算（49 KB），仍是 4 个 block，结论不变。实际 kernel 先用 `nvcc --resource-usage` 看寄存器和 shared 用量，再用 Nsight Compute 的 occupancy calculator 或 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 核对。

Occupancy 不是越高越好。BPG 写明，超过某个点后提高 occupancy 不会再变快；少驻留几个 warp、让每个线程多用寄存器，有时能靠指令级并行把等待藏住（[BPG §11.1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)）。Occupancy 回答的是“有没有足够的 warp 可以切换”，不是“这个 kernel 快不快”。

## Streams：让拷贝和计算重叠 {#streams}

Stream 是一条按顺序执行的工作队列：同一个 stream 里的拷贝和 kernel 依次完成，不同 stream 之间可以交错，甚至同时进行（[Guide §2.5.2](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)）。要让 host 与 device 之间的拷贝真正和 kernel 重叠，需要同时满足：host 缓冲区是 pinned 内存；用 `cudaMemcpyAsync`；拷贝和 kernel 放在不同的非默认 stream 里。条件来自 [BPG §10.1.2](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation)。用普通的 pageable 内存时，`cudaMemcpyAsync` 仍然正确，但会退化成同步行为，不再与别的工作重叠（[Guide §2.5.2.3](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#launching-memory-transfers-in-cuda-streams)）。

BPG 的分段写法是把数据切成 `nStreams` 段，每段的“拷贝 → 计算”放进自己的 stream。下面假设 `h_x` 是 pinned 内存，`stream[]` 已经用 `cudaStreamCreate` 创建，`n` 能被 `nStreams * 256` 整除：

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

手算一次：整段拷贝要 $t_T=8$ ms，整段计算要 $t_E=10$ ms，顺序做是 18 ms。切成 4 段后，第 1 段拷完就能开始算，后面的拷贝藏在计算后面。BPG 给的粗略估计是 $t_E+t_T/n_{\rm streams}=10+2=12$ ms（适用于 $t_E>t_T$；反过来约为 $t_T+t_E/n_{\rm streams}$）。这些毫秒数是教学假设。能不能同时做两个方向的拷贝，还取决于设备有几个 copy engine（`asyncEngineCount`）。

在 PyTorch 里，对应的做法是 DataLoader 设 `pin_memory=True`，再用 `.to(device, non_blocking=True)`（[CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#use-pinned-memory-buffers)）。多 stream 下张量生命周期的坑，见[训练循环](../00-foundations/pytorch/training-loop.md)。

“边搬边算”也发生在 kernel 内部：compute capability 8.0 起，global 到 shared 的拷贝可以异步发出（LDGSTS）；9.0 起有 Tensor Memory Accelerator（TMA），专门批量搬运多维数组的子块（[Guide §4.12](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)）。GEMM 和 FlashAttention 这类 kernel 靠它们把下一块数据的搬运藏到当前块的计算后面。

## 手写 kernel 之外：cuBLAS、cuDNN、CUTLASS 与 Triton {#libraries}

日常训练和推理很少需要手写 CUDA。[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) 是建立在 CUDA runtime 上的 BLAS 实现（另有 cuBLASLt 等接口），PyTorch 在 CUDA 上做 BLAS 运算时默认调用 cuBLAS（见 [`preferred_blas_library` 的说明](https://docs.pytorch.org/docs/2.14/backends.html)）。[cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html) 提供 attention、卷积、矩阵乘、归一化等深度学习原语，并支持把多个运算按图融合。[CUTLASS](https://github.com/NVIDIA/cutlass) 是开源的 CUDA C++ 模板库，把上面那种分层 tiling 拆成可以组合的部件；CUTLASS 4 起还有 Python 的 CuTe DSL；2026-10-21 查看时，README 顶部写的版本是 4.8.1（2026 年 10 月）。Guide 也建议通过这些库，而不是内联 PTX 去用 Tensor Core（[附录 5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications)）。

[Triton](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)（[Tillet、Kung、Cox，MAPL 2019](https://doi.org/10.1145/3315508.3329973)）换了一层抽象：程序员按 block 写程序、直接操作整块数据，coalescing、shared memory 的分配与同步、异步拷贝的调度和 Tensor Core 指令的选择交给编译器。PyTorch `torch.compile` 的默认后端 TorchInductor 在 NVIDIA、AMD 和 Intel GPU 上用 Triton 生成 kernel（[PyTorch 2.14 文档](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html)）。CUDA 自己在 v13 的文档里也加入了类似的 tile 编程模型：Python 有 cuTile，CUDA Tile C++ 从 Toolkit 13.3 起提供（[Guide §2.4](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html)）。不管用哪一层，本篇的三个问题都还在：每个 warp 搬的字节用上没有，数据复用了几次，SM 上有没有足够的 warp 可以切换。这些库怎样被框架串起来，见[框架](frameworks.md)。

## 动手时先试什么 {#try}

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| vector add 的 block 从 256 个线程改成 250 个 | 每个 block 末尾多出半空的 warp，结果仍然正确 | 一定会明显变慢；带宽受限时差别可能很小 |
| 把读取改成 stride 32 | Nsight Compute 显示 global load 的 sectors/request 接近 32（32-bit 访问的理想值是 4） | 实测时间正好慢 8 倍；cache 会改变结果 |
| 把 `TILE` 从 16 改成 32 | global 读取再减半，每个 block 的 shared memory 变成 4 倍 | 一定更快；occupancy 和 bank conflict 也跟着变 |
| 用 `-maxrregcount` 压低寄存器 | occupancy 上升，可能出现 local memory spill | occupancy 更高就一定更快 |
| 把 pinned 内存换成普通 `malloc` | `cudaMemcpyAsync` 不再与 kernel 重叠 | 所有拷贝都按同样比例变慢 |

Sectors/request 的理想值引自 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。本页没有在 GPU 上跑这些实验；第二列是根据上面引用的文档推出的预期，要在自己的硬件上验证。

接着读[性能：roofline、算术强度与 MFU](performance.md)，把这里的“搬了多少字节”换算成耗时的理论下限。
