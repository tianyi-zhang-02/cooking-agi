# CPU architecture: pipelines, caches, SIMD, and NUMA

[中文](cpu-architecture.md) · **English**

> Reading time: ~10 min · Last reviewed: 2026-10

An 8-GPU server also contains two CPUs. What are they doing, and why do you so often end up checking the CPU when GPU utilization will not go up? **A CPU core is designed to finish one instruction stream as fast as possible: pipelining, out-of-order execution, branch prediction and the cache hierarchy all exist to hide waiting. In an AI server the CPU reads data, preprocesses it, tokenizes, launches work on the GPU and moves data; when it falls behind, the GPU sits idle.**

## How fast one core wants to run one instruction stream

An instruction goes through fetch, decode, operand read, execute and write-back. **Pipelining** overlaps these steps like an assembly line: while one instruction executes, the next is already being decoded. **Superscalar** cores issue several instructions per cycle. **Out-of-order execution** lets instructions whose inputs are ready run first and then commits results in program order, so independent instructions do not sit idle while one waits for memory.

Branches are the pipeline's problem: before the `if` is resolved, where should the next instruction come from? **Branch prediction** guesses a direction and keeps going; on a wrong guess the work is thrown away and redone. Account for it in cycles per instruction (CPI):

$$
\text{CPI}=\text{CPI}_{\rm base}+f_{\rm branch}\times m\times p.
$$

$f_{\rm branch}$ is the fraction of instructions that are branches, $m$ is the misprediction rate, and $p$ is the number of cycles thrown away per misprediction. **Teaching numbers**: $\text{CPI}_{\rm base}=0.5$ (2 instructions completed per cycle on average), 20% of instructions are branches, $p=15$ cycles.

| Prediction accuracy | Extra CPI | Total CPI | Slower than ideal |
| ---: | ---: | ---: | ---: |
| 100% | 0 | 0.5 | 0% |
| 95% | $0.2\times0.05\times15=0.15$ | 0.65 | 30% |
| 90% | $0.2\times0.10\times15=0.30$ | 0.8 | 60% |

Five percentage points of accuracy cost 30% of the speed. This is also why code over unpredictable data (for example parsing logic that branches on content) runs slower on a CPU than you might expect.

## Caches: most of the time goes to waiting for memory

From registers to memory there is a chain of ever larger and slower storage: L1, L2, L3, then DRAM. Data moves in whole cache lines. Intel's optimization manual gives a few numbers to compare against: on Skylake server cores, a 1 MB L2 per core with a minimum load-to-use latency of 14 cycles and 64-byte lines; on Gracemont cores, an L1 load-to-use latency of typically four cycles ([Intel 64 and IA-32 Optimization Reference Manual, Volume 1](https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf), 248966-049, 2024-01).

Average memory access time (AMAT) chains the levels together:

$$
\text{AMAT}=t_1+m_1\big(t_2+m_2\,(t_3+m_3\,t_{\rm mem})\big).
$$

$t_i$ is the hit time of level $i$ in cycles, and $m_i$ is the fraction of accesses reaching that level that miss there and go further down. **Teaching numbers**: $t_1=4$, $t_2=14$, $t_3=50$, $t_{\rm mem}=300$ cycles; L1 hits 95%, L2 catches 80% of L1 misses, and L3 catches half of the rest.

$$
\text{AMAT}=4+0.05\times\big(14+0.2\times(50+0.5\times300)\big)=4+0.05\times54=6.7\ \text{cycles}.
$$

| Change one condition | AMAT | Note |
| --- | ---: | --- |
| The baseline above | 6.7 | Only 2.7 cycles slower than an L1 hit on average |
| L1 hit rate drops to 90% | $4+0.10\times54=9.4$ | Miss rate doubles; 40% slower on average |
| L2 catches only half | $4+0.05\times114=9.7$ | More accesses fall all the way to DRAM |
| Every access goes to DRAM | 300 | 45x slower; caches keep the average close to L1 |

The lesson: small changes in hit rate get amplified by DRAM latencies of hundreds of cycles. What matters most when writing code is whether data stays in cache and gets reused.

## Same matrix, row by row or column by column

A $64\times64$ float32 matrix stored row-major (the default in C, NumPy and PyTorch): one row is 256 bytes, exactly four 64-byte cache lines, and the matrix is 256 lines. To keep things hand-computable, assume a **toy cache that holds only 8 lines** and evicts the least recently used line when full (LRU).

- **Row by row**: 16 consecutive floats share a line, so 4,096 accesses miss only 256 times, a 6.25% miss rate.
- **Column by column**: consecutive accesses are a full row (256 bytes) apart and land on different lines. One column touches 64 lines while the cache holds 8; by the time you come back for the next column, those lines are long gone. All 4,096 accesses miss.

<!-- widget:step-cache-walk -->

```python
from collections import OrderedDict

def misses(order, n=64, floats_per_line=16, cache_lines=8):
    cache, miss = OrderedDict(), 0
    for i, j in order(n):
        line = (i * n + j) // floats_per_line   # row-major: element (i, j) sits at i*n + j
        if line in cache:
            cache.move_to_end(line)             # LRU: mark as most recently used
        else:
            miss += 1
            cache[line] = True
            if len(cache) > cache_lines:
                cache.popitem(last=False)       # evict the least recently used line
    return miss

by_row = lambda n: ((i, j) for i in range(n) for j in range(n))
by_col = lambda n: ((i, j) for j in range(n) for i in range(n))
assert misses(by_row) == 256
assert misses(by_col) == 4096
assert misses(by_col, cache_lines=64) == 256
```

| Try | Observe | What it does not establish |
| --- | --- | --- |
| Set `cache_lines` to 64 | The column walk also misses only 256 times: one column's 64 lines just fit | Column walks are fine on a real CPU |
| Make the matrix $128\times128$, cache still 64 lines | The column walk misses every time again | The real threshold is here; real caches have several levels, set associativity and hardware prefetch |
| Count the bytes used per line in the column walk | 4 of every 64 bytes moved are used; effective bandwidth drops to 1/16 | Row walks always saturate memory bandwidth |

On a real machine the threshold depends on "rows × 64 bytes" relative to each cache level's capacity, and hardware prefetchers can catch some constant-stride access (Intel's optimization manual recommends constant strides so the prefetcher can recognize them), so measure. In AI code: a PyTorch transpose usually changes only the strides and moves no data, so the next operator may be walking columns; calling `.contiguous()` on a non-contiguous tensor copies the data into row-major order. GPUs have a counterpart problem called memory coalescing; see [GPU architecture](gpu-architecture.en.md).

## SIMD: one instruction, a row of numbers

SIMD lets one instruction process several numbers in a vector at once.

- **AVX-512 (Intel)**: 32 vector registers of 512 bits and 8 mask registers; one register holds 16 float32 or 8 float64 values ([Intel AVX-512 overview](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-avx-512-instructions.html)).
- **AMX (Intel)**: 8 two-dimensional tile registers, each up to 16 rows × 64 bytes, i.e. 1 KB; a matrix-multiply unit called TMUL performs INT8 and BF16 matrix multiplies on the tiles; Intel's sample runs on 4th Gen Xeon Scalable processors ([Intel AMX code sample](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)). Intel's per-core, per-cycle peak figures: AMX does 2,048 INT8 operations versus 256 with AVX-512 VNNI, and 1,024 BF16 operations versus 64 FP32 operations with AVX-512 ([Intel AMX brief](https://cdrdv2-public.intel.com/785250/Intel-AMXBrief-Final-3.17.pdf), peak architectural capability at full CPU utilization).
- **Neon and SVE (Arm)**: Neon vectors are fixed at 128 bits. SVE's vector length is chosen by the hardware implementation, and the same binary runs on machines with different lengths (vector-length agnostic). Arm's February 2025 revision of [Introduction to SVE](https://documentation-service.arm.com/static/67ab35a4091bfc3e0a9478b5) states the length is between 128 and 2048 bits and must be a power of two; the same document's revision history says this restriction is new in that issue, so older material saying "any multiple of 128 bits" is out of date.

A rough upper bound on CPU matrix math: the DGX H100's two Xeon 8480C CPUs have 112 cores at a 2.0 GHz base clock ([DGX H100 user guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)). At Intel's AMX peak, $112\times2.0\times10^9\times1024\approx229$ TFLOP/s of BF16. That is a paper ceiling: the actual clock under full AMX load and memory bandwidth are not included. The same machine's 8 H100 SXM5 GPUs total $8\times989.4\approx7{,}915$ TFLOP/s of dense BF16 ([H100 architecture whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf); the 1,979 per GPU on the product page is the 2:4 sparse figure). A CPU can run inference for small models, but large matrices still go to the GPU.

## Multicore and NUMA: memory has near and far

When a server has two CPUs, each CPU (socket) has its own memory attached. The Linux kernel documentation calls such a grouping a node: a CPU accesses its own node's memory faster and with higher bandwidth than another node's; by default Linux allocates memory from the node of the requesting CPU and falls back to other nodes only when that fails ([Linux kernel NUMA documentation](https://docs.kernel.org/mm/numa.html)).

GPUs and NICs also hang off the PCIe of one particular CPU. Suppose GPU 0 is attached to socket 0, but the worker feeding it runs on socket 1 with its data in socket 1's memory: every batch first crosses between sockets and only then goes over PCIe. You will not notice it once, but with 8 GPUs running together the inter-socket link can fill up.

How to check: `nvidia-smi topo -m` shows which CPU each GPU is close to; `numactl --hardware` shows node numbers and distances; `numactl --cpunodebind=0 --membind=0` binds a process and its memory to the same node. The NUMA section of the CUDA C++ Best Practices Guide also warns that automatic NUMA balancing, enabled by default on some distributions, may degrade GPU applications, and advises tuning it by hand ([CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)).

## x86 and Arm: two paths in the data center

- **x86**: the DGX H100 uses 2 Intel Xeon 8480C CPUs, 56 cores each, with 2 TB of memory (DGX H100 user guide).
- **Arm**: NVIDIA Grace has 72 Arm Neoverse V2 cores, each with 4×128-bit SVE2, up to 480 GB of LPDDR5X per CPU, and memory bandwidth of up to 512 GB/s (for the 120 GB and 240 GB options; 384 GB/s for 480 GB) ([Grace CPU product page](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/)). In Grace Hopper, the CPU and GPU are connected by 900 GB/s NVLink-C2C, which NVIDIA says is 7x PCIe Gen5, with a coherent CPU and GPU memory model ([Grace Hopper product page](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)). A GB200 NVL72 rack holds 36 Grace CPUs and 72 Blackwell GPUs ([GB200 NVL72 product page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)).

For an AI server, the instruction set itself is not the point. Compare instead: how much memory bandwidth the CPU has, how fast the CPU-to-GPU link is, and whether the software you need ships for that architecture.

## What the CPU does in an AI server

| Job | Example | What it looks like when done badly |
| --- | --- | --- |
| Read, decode and augment data | Read images, decode video, assemble batches | GPU utilization drops to 0 periodically while CPU cores are saturated |
| Tokenization and templates | A chat request gets its template and is split into tokens | Time to first token grows while the GPU is not busy |
| Launch GPU work | Python calls the framework, which launches kernels one by one | Gaps all over the GPU timeline at small batch sizes |
| Move data between host and GPU memory | Copy a batch to the GPU, copy results back | Copies and compute do not overlap |
| Network and storage | RPC, writing checkpoints, logging | Tail-latency jitter; training stalls while a checkpoint is saved |

**Work out a data loader by hand** (teaching numbers): 8 GPUs each consume 1,000 samples per second, and the CPU spends 4 ms per sample. The number of CPU cores needed is

$$
8\times1000\ \text{samples/s}\times0.004\ \text{s/sample}=32\ \text{cores}.
$$

A 112-core machine has enough. With heavy augmentation or video decoding at 20 ms per sample, you need 160 cores, more than the whole machine, and the GPUs wait. The fix then is to move decoding elsewhere, preprocess the data ahead of time, or move part of the preprocessing onto the GPU, not to keep raising `num_workers`.

For moving data, remember how large the bandwidth gap is. The H100 SXM5's HBM runs at 3,352 GB/s (whitepaper V1.04); PCIe Gen5 x16 carries 64 GB/s in each direction ([Hopper architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)), a gap of about 50x. The CUDA Best Practices Guide's older example is the V100's 898 GB/s against 16 GB/s for PCIe Gen3, also more than 50x, which is why it lists minimizing transfers between host and device as a high-priority recommendation. It also explains that transfers from pinned (page-locked) memory reach the highest bandwidth; asynchronous copies require pinned host memory, which is what lets copies overlap with kernels; but pinned memory is a scarce resource and should not be overused. In PyTorch, setting `pin_memory=True` on a DataLoader puts fetched batches in pinned memory, and `num_workers` greater than 0 loads with multiple processes ([PyTorch 2.14 DataLoader documentation](https://docs.pytorch.org/docs/2.14/data.html)).

## Why a GPU instead

A CPU spends many of its transistors on making one thread wait less: large caches, out-of-order windows, branch predictors. A GPU goes the other way. In the words of the CUDA C++ Programming Guide: a CPU excels at running one thread as fast as possible and runs a few tens in parallel; a GPU excels at running thousands in parallel, devotes more transistors to data processing rather than caching and flow control, and hides memory latency with computation instead of relying on large caches and complex flow control ([CUDA C++ Programming Guide 12.8.1, §1.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html)). Add high-bandwidth memory such as HBM. In orders of magnitude: one Grace CPU reaches at most 512 GB/s while one H100 SXM5 has 3,352 GB/s; and the AMX paper peak of the two CPUs in the machine above is under 3% of its 8 GPUs.

So the division of labor is: the CPU handles control flow, irregular small tasks and data preparation, and the GPU handles large, regular, parallel matrix math. The next note explains how the GPU does that: [GPU architecture](gpu-architecture.en.md). For how the chips in this machine are made, see [How chips are made](chips-and-fabs.en.md); the map of the whole chain is in the [series guide](README.en.md).
