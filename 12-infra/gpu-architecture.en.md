# GPU architecture: SMs, warps, the memory hierarchy, and Tensor Cores

[中文](gpu-architecture.md) · **English**

> Reading time: ~13 min · Draft, pending review

When an H100 SXM runs a large matrix multiply, its dense BF16 Tensor Core peak is 989.4 TFLOPS; yet when the same card generates tokens one at a time for a large model, it spends most of its time waiting for data. Both are true at once. **A GPU is built for total throughput, not for making one thread fast: it keeps many warps resident and switches to another whenever one is waiting on memory, and whether the Tensor Cores stay busy comes down to how fast data can be moved into the SMs.**

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

Hardware numbers are mostly for the H100 SXM5; final specifications come from the [H100 architecture whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf) (which states that it includes final clocks and TFLOPS), and the rest from NVIDIA product pages, architecture blogs and CUDA documentation; cycle counts and matrix entries in the worked examples are teaching assumptions and are marked as such. How to write kernels is in [CUDA](cuda.en.md), the roofline is in [Performance](performance.en.md), and the cross-generation comparison is in [NVIDIA GPU generations](nvidia-generations.en.md).

## Two design goals: wait less, or do more at once

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html) puts it plainly: a CPU is built to finish a serial sequence of operations as fast as possible and runs a few tens of threads at once; a GPU is built to run thousands of threads in parallel, trading lower single-thread performance for much higher total throughput. So a CPU spends more of its transistors on caching and control logic, and a GPU spends more on compute units. How a CPU shortens the wait of a single thread is covered in [CPU architecture](cpu-architecture.en.md).

| | Latency first (CPU) | Throughput first (GPU) |
| --- | --- | --- |
| Transistors mostly go to | Caches and control logic | Compute units and a large register file |
| When a thread must wait on memory | Make the wait short, or find other work in the same thread | Switch to another warp that is ready and keep issuing |
| Good at | Branchy code, long dependency chains, low-latency work | The same operation over lots of data, such as matrix multiply |

This is a design tendency, not a hard line: CPUs have SIMD vector units too, and GPU threads can branch; the costs are just different.

## The SM: a GPU is a hundred-odd processors

In CUDA's hardware model, a GPU is a set of Streaming Multiprocessors (SMs) grouped into GPCs; each SM has its own register file, an on-chip memory that can be split between L1 and shared memory, and compute units ([programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)). The H100 SXM5 has 132 SMs ([Hopper architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)), and each SM contains:

| Component | Per SM on H100 SXM5 | What it does |
| --- | --- | --- |
| FP32 CUDA cores | 128 | Ordinary floating-point work, such as element-wise adds and activations |
| Tensor Cores (4th gen) | 4 | Small matrix multiply-accumulates; most of the FLOPs happen here |
| Register file | 256 KB, i.e. 65,536 32-bit registers | Each thread's private variables; the fastest level |
| L1 / shared memory | 256 KB combined, up to 228 KB as shared | Scratchpad shared by the threads of a block, managed by the programmer |
| Warp schedulers | 4 processing blocks, one each | Each cycle, pick a ready warp and issue its instruction |
| Residency limits | 64 warps, 2,048 threads, 32 blocks | How much work can be in flight at once |

The "four processing blocks, one warp scheduler each" layout is described in the [Volta whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf); the [Ampere whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) says the A100 has one Tensor Core per partition, and the H100 also has 4 per SM. Residency limits come from the [compute capability tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html); the H100 is compute capability 9.0.

On the software side, a kernel launch creates a grid made of thread blocks. **All threads of a block run on the same SM**, which is why they can exchange data through shared memory and synchronize; there is no ordering guarantee between blocks. From compute capability 9.0 there are also thread block clusters: the blocks of a cluster are scheduled onto one GPC and can read and write each other's shared memory. The thread hierarchy in detail is in [CUDA](cuda.en.md#thread-hierarchy).

## Warps and SIMT: 32 threads share one instruction

An SM schedules threads in groups of 32, called warps. The [CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt) calls this SIMT: the threads of a warp execute the same code, but each can take its own branch. When they disagree, the threads not on the current path are masked off and get their turn when the other path runs. This is warp divergence.

Work out the cost once. A warp hits an `if`: when the condition is true it runs path A (10 instructions), otherwise path B (20 instructions). The instruction counts are a teaching assumption.

| Case | Instruction slots the warp occupies | Useful "threads × instructions" | Utilization |
| --- | ---: | ---: | ---: |
| All 32 threads take A | 10 | 32 × 10 = 320 | 100% |
| All 32 threads take B | 20 | 32 × 20 = 640 | 100% |
| 16 take A, 16 take B | 10 + 20 = 30 | 16 × 10 + 16 × 20 = 480 | 480 / (32 × 30) = 50% |

**A branch is not expensive in itself; splitting within one warp is.** Different warps taking different branches pay no such cost.

Two common misreadings. First, SIMT does not mean "the 32 threads are always in lockstep". Since compute capability 7.0 (Volta) there is independent thread scheduling: each thread has its own program counter, and threads in a warp can diverge and reconverge at finer granularity; old code that relies on implicit synchronization must use `__syncwarp()` ([advanced kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#independent-thread-scheduling)). Second, the difference between SIMT and SIMD is the interface: SIMD exposes the vector width to software, while SIMT lets you write code for a single thread and the hardware groups threads into warps.

## Latency hiding: keep many warps resident

The CUDA guide explains that each warp's execution context (program counter, registers) stays on chip for the warp's whole lifetime, so switching between warps costs nothing; at every issue cycle a warp scheduler picks one of the ready warps ([hardware multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading)). If one warp is waiting on memory, another one goes.

A simple model tells you how many warps you need. Suppose each warp repeats two things: issue $W$ independent instructions (taking $W$ issue cycles), then wait $L$ cycles for a memory load. With $N$ warps resident on one scheduler, the fraction of issue slots used is about:

$$
U \approx \min\left(1,\ \frac{N\,W}{W+L}\right).
$$

Take $L=300$ and $W=30$ (teaching assumptions, not measured H100 latencies): a warp uses only 30 issue cycles out of every 330, so you need $N\ge 330/30=11$ warps to fill the slots.

<!-- widget:step-latency-hiding -->

How many warps fit is often set by registers. An H100 SM has 65,536 32-bit registers and at most 64 resident warps; split evenly over 4 schedulers, that is at most 16 each.

| Registers per thread | Registers per warp | Resident warps per SM | Per scheduler | $U$ in this example |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1,024 | 64 (the limit) | 16 | 100% |
| 64 | 2,048 | 32 | 8 | 8 × 30 / 330 ≈ 73% |
| 128 | 4,096 | 16 | 4 | 4 × 30 / 330 ≈ 36% |

This ignores register allocation granularity and shared memory limits; the full occupancy calculation is in [CUDA](cuda.en.md#occupancy).

| What to try | What you should see | What you can NOT conclude |
| --- | --- | --- |
| Raise $W$ from 30 to 60 (each thread computes several independent elements) | With the same $L$, 6 warps are enough to fill the slots | Lower occupancy is better |
| Raise registers per thread from 64 to 128 | Resident warps halve; $U$ here drops from about 73% to about 36% | Kernels that use many registers are always slow |
| Switch to asynchronous copies (hardware async copy from Ampere, TMA on Hopper) | Data movement overlaps compute, and fewer warps can stay busy | Async copies get you past the bandwidth limit |

The last row deserves a word. High-performance matrix multiply kernels often use many registers per thread and run at modest occupancy, hiding the wait with instruction-level parallelism and asynchronous data movement; the CUDA guide also describes async copies as a way to overlap movement and compute beyond what warp switching does ([asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#asynchronous-execution-features)). **Latency can be hidden; the bandwidth ceiling cannot.** That is the next section.

## The memory hierarchy: closer is smaller and faster

Capacities and official bandwidths for each level of the H100 SXM5 (whitepaper V1.04, [Hopper blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [H100 product page](https://www.nvidia.com/en-us/data-center/h100/), compute capability tables):

| Level | Capacity (H100 SXM5) | Who can access it | Bandwidth (official figure) |
| --- | --- | --- | --- |
| Registers | 256 KB per SM, 33,792 KB across 132 SMs | One thread | Not published |
| Shared memory / L1 | 256 KB per SM combined, up to 228 KB as shared | One block; across SMs within a cluster | 32 banks, each 32 bits per cycle |
| L2 cache | 50 MB | All SMs | Not published for H100 |
| HBM3 | 80 GB | All SMs; also how kernels hand data to each other | 3,352 GB/s |
| NVLink (4th gen) | — | Other GPUs | 900 GB/s total per GPU |
| PCIe Gen5 | — | CPU and host memory | 128 GB/s |

Note a discrepancy between sources: the 2022 Hopper architecture blog gives HBM3 bandwidth as 3,000 GB/s, a preliminary figure; whitepaper V1.04 gives 3,352 GB/s, consistent with the product page's 3.35 TB/s. In the V1.04 spec table the memory bandwidth row label still carries an old "Not Finalized for H100" note, but the cover says the final memory clocks are included, so this note uses 3,352 GB/s.

NVIDIA does not publish absolute on-chip bandwidths for the H100, but the public A100 material is enough for an order-of-magnitude check. The A100 whitepaper gives an L2 read bandwidth of 5,120 bytes per clock and a boost clock of 1,410 MHz; the [CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-bank-conflicts) says shared memory has 32 banks of 32 bits per cycle each, i.e. 128 bytes per SM per cycle. The A100 has 108 SMs:

$$
\begin{aligned}
\text{shared memory, all SMs} &\approx 108\times128\ \text{B}\times1.41\ \text{GHz}\approx19.5\ \text{TB/s},\\
\text{L2} &\approx 5120\ \text{B}\times1.41\ \text{GHz}\approx7.2\ \text{TB/s},\\
\text{HBM2e (A100 80GB SXM)} &= 2.039\ \text{TB/s}.
\end{aligned}
$$

The first two lines are theoretical peaks converted from specifications, not measurements; bank conflicts and cache misses lower the real numbers. The HBM figure comes from the [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). The ratios already make the point: the closer to the compute units, the higher the bandwidth and the smaller the capacity. Hence the basic recipe of high-performance kernels is tiling: read a block from HBM into shared memory or registers, reuse it many times there, then write back. How to write it is in [the tiling section of CUDA](cuda.en.md#tiling); [FlashAttention](../00-foundations/deep-dives/attention-kernels.en.md) is the same idea applied to attention.

Blackwell adds one more level. Fifth-generation Tensor Cores have a dedicated Tensor Memory, which [PTX ISA §9.7.18.1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) describes on sm_100a as 128 rows × 512 columns of 32-bit cells per CTA, i.e. 256 KB; the matrix-multiply accumulator $D$ lives there instead of in registers.

## Tensor Cores: one instruction, one small matrix multiply-accumulate

A CUDA core's FMA instruction does one multiply-add. A Tensor Core does a small block at once: $D=A\times B+C$. In the Volta whitepaper, each Tensor Core works on 4×4 matrices and performs 64 FMAs per clock.

### A 4×4 multiply-accumulate by hand

Take these small matrices (teaching numbers), with $C$ the all-ones matrix:

$$
A=\begin{bmatrix}1&2&0&1\\0&1&1&0\\2&0&1&0\\1&1&1&1\end{bmatrix},\qquad
B=\begin{bmatrix}1&0&2&0\\0&1&0&1\\1&1&0&0\\0&2&1&1\end{bmatrix}.
$$

Row 1, column 2 of $D$ is the dot product of row 1 of $A$ with column 2 of $B$, plus the matching entry of $C$: $d_{12}=1\cdot0+2\cdot1+0\cdot1+1\cdot2+1=5$. Doing all 16 entries gives:

$$
D=AB+C=\begin{bmatrix}2&5&4&4\\2&3&1&2\\4&2&5&1\\3&5&4&3\end{bmatrix}.
$$

Each output entry takes 4 multiply-adds, i.e. 4 FMAs; 16 entries take $16\times4=64$ FMAs, exactly one Volta Tensor Core's work per clock. Counting an FMA as 2 FLOPs, this tile is 128 FLOPs.

On precision, Volta's $A$ and $B$ are FP16, the products are kept at full precision and accumulated in FP32; $C$ and $D$ may be FP16 or FP32. "Low-precision inputs, higher-precision accumulation" carries through to the later FP8 and FP4.

### From one tile to the whole card's peak

Peak = number of Tensor Cores × FMAs per Tensor Core per cycle × 2 × clock. Check against NVIDIA's published numbers:

| GPU | Tensor Cores | Dense FMAs per core per cycle | Clock | Computed | Official figure |
| --- | ---: | ---: | ---: | ---: | ---: |
| V100 SXM2 | 640 | 64 (FP16) | 1,530 MHz | 125.3 TFLOPS | 125 TFLOPS |
| A100 SXM | 432 | 256 (FP16/BF16) | 1,410 MHz | 311.9 TFLOPS | 312 TFLOPS |
| H100 SXM | 528 | 512 (FP16/BF16) | 1,830 MHz | 989.4 TFLOPS | 989.4 TFLOPS |

The V100 row comes from the Volta whitepaper and the A100 row from the [Ampere architecture blog](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/). The H100 row's clock and official figure come from whitepaper V1.04: a boost clock of 1,830 MHz for Tensor Core operations, 989.4 TFLOPS dense FP16/BF16 (FP32 accumulate), and 1,978.9 TFLOPS sparse. The 512 FMAs per Tensor Core per cycle is this note's inference: the whitepaper says the per-SM MMA rate is 2× the A100's at the same clock; plugging in $528\times512\times2\times1.83\ \text{GHz}$ gives exactly 989.4 TFLOPS.

The "sparse" figures in datasheets come from 2:4 structured sparsity, introduced with Ampere: when at most 2 of every 4 values are non-zero, Tensor Core throughput doubles. Ordinary dense models do not get that factor, so convert to dense before comparing. Every Tensor row on the H100 product page is marked "with sparsity".

### Bigger instructions, issued by fewer threads

Programs never write 4×4 directly. At the PTX level you can see how the interface changed with each generation ([PTX ISA §9.7.16–9.7.18](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)):

| Instruction | Issued by | Shape of one instruction (FP16/BF16) | FLOPs per instruction | Requires |
| --- | --- | --- | ---: | --- |
| `mma.sync.m16n8k16` | One warp (32 threads) | A 16×16, B 16×8 | 2 × 16 × 8 × 16 = 4,096 | sm_80 and later |
| `wgmma.mma_async.m64n256k16` | One warpgroup (4 contiguous warps) | A 64×16, B 16×256 | 2 × 64 × 256 × 16 = 524,288 | sm_90a |
| `tcgen05.mma` | A single thread of the CTA | Accumulator $D$ in Tensor Memory | Depends on shape | sm_100a and others |

One `mma.sync` equals 32 of the 4×4×4 tiles above (4,096 / 128). The trend: each instruction does more work, runs more asynchronously, and is issued by fewer threads. With fewer threads and registers spent feeding the Tensor Cores, the remaining warps can move data or compute softmax; [FlashAttention-3](https://arxiv.org/abs/2407.08608) on Hopper uses warp specialization to overlap data movement with compute and interleaves block-wise matmul with softmax (see also [attention kernels](../00-foundations/deep-dives/attention-kernels.en.md)). The actual code is left to [CUDA](cuda.en.md) and libraries such as CUTLASS.

<!-- paper-figure:fa3-pingpong -->

## Data types: what each Tensor Core generation added

| Format | Sign / exponent / mantissa bits | First Tensor Core support | Notes |
| --- | --- | --- | --- |
| FP16 | 1 / 5 / 10 | Volta (V100) | Can accumulate in FP32 |
| INT8, INT4 | Integer | Integer `wmma` from sm_72, INT4 from sm_75 | Mainly for quantized inference |
| TF32 | 1 / 8 / 10 | Ampere (A100) | Reads FP32 data, computes with FP32 range and reduced internal precision, outputs FP32 |
| BF16 | 1 / 8 / 7 | Ampere (A100) | Same range as FP32 |
| FP8 (E4M3, E5M2) | 1 / 4 / 3; 1 / 5 / 2 | Hopper (H100); Ada (compute capability 8.9) also supports it | E4M3 has more precision, E5M2 more range |
| FP6 (E3M2, E2M3), FP4 (E2M1) | 1 / 3 / 2, 1 / 2 / 3; 1 / 2 / 1 | Blackwell (B200) | Used together with block scale factors |

Sources: FP16 from the Volta whitepaper; TF32 and BF16 from the Ampere blog; the two FP8 formats from the Hopper blog; which input types each compute capability supports from the compute capability tables; FP8 `mma` needing sm_89 and the FP6/FP4 format names from the PTX ISA. The FP16 versus BF16 trade-off is in [Precision and memory](../00-foundations/deep-dives/precision-and-memory.en.md) and is not repeated here.

The narrower the format, the more it depends on scaling. FP4 E2M1 can only represent 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives. [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) gives every 16 values a shared FP8 (E4M3) scale factor and the whole tensor a further FP32 scale, for 4.5 bits per value on average; the MXFP4 format it is compared with shares one E8M0 scale, restricted to powers of two, across every 32 values.

Work a simplified version by hand (only 4 values; ignore the tensor-level scale and the rounding of the scale itself). The block is $[0.3,-1.2,2.4,3.0]$; its largest magnitude is 3.0, so the scale is $3.0/6=0.5$. Dividing by 0.5 gives $[0.6,-2.4,4.8,6.0]$; rounding to the nearest representable values gives $[0.5,-2,4,6]$; multiplying back gives $[0.25,-1.0,2.0,3.0]$. If a 30 lands in the same block, the scale becomes 5, and 0.3 and −1.2 both round to 0 after dividing by 5: one outlier wipes out the small values of the whole block. The smaller the block, the fewer values one outlier drags down, which is the intuition behind NVFP4 using 16 rather than 32.

## NVLink and NVSwitch: between cards

The H100 SXM5 has 18 fourth-generation NVLink links at 25 GB/s per direction each, 900 GB/s in total; its PCIe Gen5 link is 128 GB/s (Hopper blog, H100 product page). NVLink is a point-to-point link; NVSwitch is a switch chip that lets every GPU in a group talk to every other at full NVLink bandwidth. A group of GPUs that can reach each other this way is called an NVLink domain.

| Interconnect | Bandwidth | Reach |
| --- | ---: | --- |
| PCIe Gen5 (H100) | 128 GB/s | GPU to CPU |
| NVLink 4 (H100) | 900 GB/s per GPU | Inside one 8-GPU HGX machine |
| NVLink 5 (GB200 NVL72) | 1.8 TB/s per GPU, 130 TB/s across 72 GPUs | One rack |
| NICs (HGX B200) | 0.8 TB/s for 8 GPUs, about 100 GB/s per GPU | Between machines |

The last two rows come from the [GB200 NVL72 page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) and the [HGX page](https://www.nvidia.com/en-us/data-center/hgx/). Both numbers add the two directions: HGX B200 has 8 NICs at 400 Gb/s (the [DGX B200 page](https://www.nvidia.com/en-us/data-center/dgx-b200/) lists 8 single-port ConnectX-7), 400 GB/s per direction in total, 0.8 TB/s counting both. Compared on the same basis, NVLink bandwidth in the same Blackwell generation is about 18× the off-machine network (1.8 TB/s versus about 0.1 TB/s per GPU). That is why tensor parallelism, the most communication-heavy split, usually stays inside an NVLink domain, while only lighter-communication splits cross machines; see [Tensor parallelism](../06-systems/tensor-parallel.en.md) and [Distributed training](../06-systems/distributed-training.en.md). NVL72 stretches the NVLink domain from one 8-GPU machine to a whole rack.

## What this note leaves out

- How to write kernels and use Nsight to read occupancy and bandwidth: see [CUDA](cuda.en.md).
- One roofline number only: the H100 SXM has 989.4 dense BF16 TFLOPS and 3,352 GB/s of HBM, a ratio of about 295 FLOPs per byte. A kernel that cannot do that many operations per byte read from memory hits the bandwidth limit first. How to reason about this is in [Performance](performance.en.md).
- Per-generation specs and the worked example of how many GPUs a 70B model needs: see [NVIDIA GPU generations](nvidia-generations.en.md). Process nodes, HBM and packaging: see [How chips are made](chips-and-fabs.en.md).
