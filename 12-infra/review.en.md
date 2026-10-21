# AI infrastructure: review questions

[中文](review.md) · **English**

> Reading time: ~8 min · Last reviewed: 2026-10

AI infrastructure interviews often cover several connected stages: chip manufacturing, GPU kernel execution, framework kernel launches, and cluster-level GPU allocation. Start by stating the basis for comparison, including precision, dense or sparse computation, and the specific SKU. Then identify the layer containing the bottleneck and use the relevant figures to estimate its limits. Every number below comes from the notes in this series, which provide the derivations and sources.

## Interview questions

<details class="interview" markdown="1">
<summary>Why is a process node name such as "7 nm" or "N4" not a physical size? What should you compare when comparing two chips?</summary>

Early node names roughly matched a physical dimension of the transistor; today a node name is closer to the name of a process generation. A 2020 IEEE Spectrum article points out that most critical features of a "7 nm" transistor are much larger than 7 nm, and describes LMC, an alternative metric proposed by Wong and colleagues (logic density, memory density, and the density of connections between them). Vendor figures need their baseline checked: TSMC calls N4 an enhanced version of N5, and N4P's "11% performance boost" is relative to N5; N2P versus N3E is 18% faster at the same power or 36% lower power at the same speed, which usually cannot be had at once. NVIDIA's 4N and 4NP are custom processes, and no public statement was found on how they differ from TSMC's public N4 family.

So the node name is only a starting point; in the end, compare measured performance, power and cost on the same workload. The constraint that really caps a single die is the exposure field: 26 mm × 33 mm, or 858 mm², and the GH100's 814 mm² already takes 95% of it. See [How chips are made](chips-and-fabs.en.md).

</details>

<details class="interview" markdown="1">
<summary>Why is batch-1 decode limited by memory bandwidth? What does a larger batch change, and what does it not change?</summary>

To generate one token, each weight does a single multiply-add (2 FLOP) but costs 2 bytes to read in BF16, an arithmetic intensity of about 1 FLOP per byte; the H100 SXM5's ridge point is its dense BF16 989.4 TFLOPS divided by 3,352 GB/s, about 295. A 7B model's BF16 weights are about $1.4\times10^{10}$ bytes; reading them once takes at least 4.18 ms, so a single request tops out at about 239 tokens/s, while the arithmetic itself needs only 0.014 ms.

A larger batch shares the weights: counting weights only, the intensity is roughly the batch size $B$, and while $B$ is below about 295, step time barely changes and throughput grows linearly with $B$. What cannot be shared is the KV: every request has its own, and the attention part has an intensity of only about 4 FLOP per byte, independent of the batch. With a 4096-token context per request, the step lower bound is already 14.4 ms at $B=64$, for a throughput bound of about 4,440 tokens/s; at $B=128$ the weights plus KV take about 77 GiB. KV bandwidth and capacity run out before compute does. See [Performance](performance.en.md#decode-batching).

</details>

<details class="interview" markdown="1">
<summary>What determines a kernel's occupancy? Is higher always better?</summary>

Occupancy is the number of active warps on an SM divided by the maximum number of warps that SM can hold, and it is set by whichever resource runs out first. At compute capability 9.0 (for example the H100), each SM holds at most 2048 threads (64 warps), 32 blocks, 65,536 32-bit registers and 228 KB of shared memory. With 256 threads per block: 32 registers per thread fits 8 blocks, 100%; 64 registers leaves 4 blocks, 50%; 48 KB of shared memory per block also gives 4, 50%; 80 registers leaves 3, 37.5%. Registers are also rounded up to 256 per warp; for a real kernel, read usage with `nvcc --resource-usage` and check with Nsight Compute or `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

Occupancy answers "are there enough warps to switch between", not "is this kernel fast". The Best Practices Guide says that past a certain point, higher occupancy does not improve performance; high-performance matrix-multiply kernels often use many registers per thread with low occupancy, hiding the wait with instruction-level parallelism and asynchronous copies (hardware async copy from Ampere, TMA on Hopper). Latency can be hidden; the bandwidth ceiling cannot. See [CUDA](cuda.en.md#occupancy) and [GPU architecture](gpu-architecture.en.md).

</details>

<details class="interview" markdown="1">
<summary>What is memory coalescing, and why does strided access waste bandwidth?</summary>

Global memory moves data in 32-byte units. When a warp executes one load, the hardware coalesces the 32 threads' addresses and issues one transaction per 32-byte segment touched. With each thread reading one float32 and an aligned start: adjacent threads reading adjacent elements use 4 segments, all 128 bytes; offsetting the start by one element takes 5 segments, 80% used; at stride 32 each thread gets its own segment, so 1024 bytes are moved for 128 used, 12.5%. Reading a row-major matrix by column is exactly this case.

Segment counts are not time: the Best Practices Guide measured the offset copy on a V100 at about 9/10 of the aligned bandwidth, not the 4/5 the segment count suggests, because neighbouring warps reuse data already pulled into cache. When debugging, check sectors/request in Nsight Compute: ideal is 4 for 32-bit accesses, close to 32 at stride 32. The fix is to make `threadIdx.x` index the lowest dimension of the address. See [CUDA](cuda.en.md#coalescing).

</details>

<details class="interview" markdown="1">
<summary>How do dense and sparse datasheet numbers differ? What must you align before comparing two GPU generations?</summary>

Sparse figures come from 2:4 structured sparsity, introduced with Ampere: when at most 2 of every 4 values are non-zero, Tensor Core throughput doubles, which an ordinary dense model does not get. The H100 product page's BF16 1,979 TFLOPS is marked with sparsity; whitepaper V1.04's dense value is 989.4 TFLOPS. But you cannot always divide by 2: the GB300 NVL72 page lists FP4 as 1,440 (sparse) versus 1,080 (dense).

Three more things need aligning: the SKU (both are H100, but the SXM part has 80 GB and 3.35 TB/s, the NVL part 94 GB and 3.9 TB/s); the unit counted (a GB200 superchip is 1 Grace plus 2 GPUs, and the HGX page gives 8-GPU totals); and preliminary versus final specs (the 2022 Hopper blog's 1,000 TFLOPS and 3,000 GB/s were preliminary). Once aligned, from V100 FP16 to GB300 NVFP4 the dense peak rose 120× while memory bandwidth rose only about 9×, so for read-heavy, compute-light work such as decode, generational speedups mostly follow bandwidth. Official pages also disagree with each other: Rubin's per-GPU memory bandwidth, for example, is given as both 22 and 19.2 TB/s. See [NVIDIA GPU generations](nvidia-generations.en.md).

</details>

<details class="interview" markdown="1">
<summary>What does MFU measure? How does it differ from GPU utilization in nvidia-smi, and from HFU?</summary>

MFU was introduced in the PaLM paper: observed tokens/s divided by the tokens/s the hardware could reach in theory at peak FLOPs. The theoretical value counts only the operations required for the forward and backward passes (about $6N$ per token for a dense decoder, plus $12LHQT$ for the attention matmuls), **not recomputation**. HFU also counts recomputation, depends on the implementation and is not suited to cross-system comparison; the paper reports an MFU of 46.2% and an HFU of 57.8% for PaLM 540B. `nvidia-smi` utilization is something else again: NVML defines it as the fraction of the sample period during which at least one kernel was running.

Change the basis and the number changes. In one teaching example (a 7B model on 8 H100 SXM5s at 72,018 tokens/s): counting only $6N$ gives 38.2%, adding attention gives 44.1%, HFU with full recomputation is 50.9%, and wrongly using the sparse peak as the denominator gives only 19.1%. When reporting MFU, state whether attention is counted, which precision the peak refers to, dense or sparse, and whether tokens/s includes padding. See [Performance](performance.en.md#mfu).

</details>

<details class="interview" markdown="1">
<summary>What does torch.compile fuse, and why does that make things faster?</summary>

In eager mode, `y = torch.relu(x @ w + b)` is 3 ATen operators and at least 3 kernels: each operator goes through a Python call, the dispatcher, output allocation and kernel submission, and each elementwise operator reads and writes its intermediate result in memory. `torch.compile` has TorchDynamo capture an FX graph (with guards, recompiling when they fail; unsupported Python causes a graph break), AOTAutograd capture the backward pass ahead of time, and TorchInductor generate Triton code for the GPU and fuse kernels.

What it saves is not multiplications. With 4096×4096, BF16 and a teaching bandwidth of 2 TB/s: eager's add and relu read and write 128 MiB, about 67 µs; one fused kernel moves 64 MiB, about 34 µs; with bias and ReLU both in the GEMM epilogue there is no extra traffic. At a teaching compute of 500 TFLOP/s the GEMM itself is about 275 µs, so eager's elementwise part is about 24% of it. With $M$ cut to 8, the bytes become negligible and the saving is mostly CPU submissions; `reduce-overhead` uses CUDA Graphs to merge a chain of submissions into one. The costs are first-call compile time, recompilation when shapes change, and graph breaks. See [Frameworks](frameworks.en.md).

</details>

<details class="interview" markdown="1">
<summary>Why does distributed training on Kubernetes need gang scheduling?</summary>

The default kube-scheduler places one Pod at a time, but distributed training needs every rank present before it can start; and a GPU is an extended resource that comes only in whole numbers and cannot be overcommitted. Work it by hand: 3 nodes with 8 GPUs each; in order, two 4-GPU jobs arrive, then a 16-GPU training job T16 (2 Pods of 8 GPUs each), then an 8-GPU job T8. Default scoring looks only at CPU and memory, so the two 4-GPU jobs may land on different nodes; one T16 Pod takes the third node and the other does not fit, so the half that started holds 8 GPUs while waiting; the two remaining halves of 4 free GPUs cannot form T8 either. Only 8 of 24 GPUs are working, and another T16 would also queue. Two jobs waiting on each other require an extra condition: with only two whole nodes available for training and the third occupied long-term, each T16 may hold one node and wait for its other half. Deadlock then requires that no resource release, timeout-based reclamation or preemption resolves the wait; the original table's fragmentation does not establish this by itself.

Gang scheduling binds nothing if the whole group does not fit, leaving the third node for T8, so 16 GPUs work; adding GPU bin packing (MostAllocated with GPU weighted) packs the two small jobs onto one node, and all 24 work, with T8 now queued. There are several implementations: core Kubernetes gang scheduling is beta and off by default in v1.37; Kueue uses `waitForPodsReady` to evict and requeue on timeout; Volcano uses the PodGroup's `minMember`. Note that "gang scheduling" in the Slurm docs means time-slicing, not all-or-nothing. See [Kubernetes](kubernetes.en.md).

</details>

## Self-check

<div class="taste-check">
  <strong>Use these questions to check your understanding:</strong>
  <ol>
    <li>why, at the same $D_0=0.1$ per cm², a monolithic 800 mm² die gives about 29 good dies per wafer while two 400 mm² halves give 48 pairs, and which costs that arithmetic leaves out;</li>
    <li>why tiling cuts a matrix multiply's global reads from $2n^3$ to $2n^3/T$, yet the Best Practices Guide's V100 example only raises effective bandwidth from 119.9 GB/s to 195.5 GB/s;</li>
    <li>what basis NVLink 900 GB/s, PCIe 128 GB/s and InfiniBand 400 Gb/s are each quoted on, and how far apart HBM and the network card are once all are converted to bytes per direction;</li>
    <li>why, for a 70B model at batch 1, FP8 on one H100 and BF16 on two H100s share the same 20.9 ms per-token floor, and why two A100s are slower than five V100s;</li>
    <li>why, with the busbw values assumed in the text, the same 14 GB of BF16 gradients take only about 1.9% of a step to all-reduce on one 8-GPU node but 18% on two 8-GPU nodes, how much of that DDP buckets can hide, and why tensor parallelism cannot hide it.</li>
  </ol>
</div>

## References

- [NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf): final H100 SXM5 specifications, dense and sparse peaks, memory bandwidth
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/): the 2022 architecture blog, with preliminary specs, NVLink and PCIe bandwidth, and FP8
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/): TF32, BF16 and 2:4 structured sparsity
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html): thread hierarchy, memory spaces, occupancy and per-compute-capability limits
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): coalescing, shared memory tiling, occupancy, overlapping copies with compute
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785): Williams, Waterman and Patterson, CACM 2009, the roofline model
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102): Pope et al., 2022; loading weights dominates at small batch, loading the KV cache at larger batches and longer sequences
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311): §4.1 and Appendix B define MFU and HFU
- [Bandwidth optimal all-reduce algorithms for clusters of workstations](https://doi.org/10.1016/j.jpdc.2008.09.002): Patarasuk and Yuan, JPDC 2009, the communication lower bound for ring all-reduce
- [PyTorch 2.14 docs: torch.compiler](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html): TorchDynamo, AOTAutograd and TorchInductor
- [PyTorch 2.14 docs: CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html): asynchronous execution, streams, pinned memory and CUDA Graphs
- [Kubernetes docs: scheduling GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/): device plugins and `nvidia.com/gpu`
- [Kubernetes docs: gang scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/gang-scheduling/): all-or-nothing placement for a group of Pods
- [Kueue](https://kueue.sigs.k8s.io/docs/overview/): job-level queuing, quotas and admission on top of kube-scheduler
- [A Better Way to Measure Progress in Semiconductors](https://spectrum.ieee.org/a-better-way-to-measure-progress-in-semiconductors): IEEE Spectrum, 2020; node names detached from physical size, and the LMC metric
