# Performance: the roofline, arithmetic intensity, and MFU

[中文](performance.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

`nvidia-smi` shows 100% GPU utilization, yet a 7B model generating one token at a time produces only a couple of hundred tokens per second. Is the card too slow, or is it being used badly? First look at what that 100% measures: NVML defines it as the fraction of the sample period during which at least one kernel was running ([NVML documentation](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlUtilization__t.html)). It says nothing about how much compute is being used. **The more useful question is: for every byte this operation moves from memory, how many operations does it perform? Compare that with the card's "peak compute ÷ memory bandwidth": below it, you need to move fewer bytes; only above it does compute become the limit.** Using the published specifications of one H100, this note puts a large matrix multiply, an elementwise add, and 7B-model decode on the same chart, then works out MFU and multi-GPU communication, and ends with an order for profiling.

This note follows [CUDA: threads, memory, and kernels](cuda.en.md). Unless a number is marked as reported by a paper, it is an upper bound computed from published specifications or a teaching assumption, not a benchmark.

## The roofline: two lines that bound performance {#roofline}

The roofline model of Williams, Waterman, and Patterson ([CACM 52(4), 2009](https://doi.org/10.1145/1498765.1498785); [2008 technical report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)) uses only two hardware numbers and one program number:

$$
P_{\rm attainable}=\min\left(P_{\rm peak},\;B\times I\right).
$$

$P_{\rm peak}$ is peak compute (FLOP/s), $B$ is memory bandwidth (byte/s), and $I$ is how many floating-point operations are done per byte moved (FLOP/byte). The original paper calls $I$ operational intensity and deliberately counts bytes as the traffic that actually reaches DRAM after the caches have filtered it. NVIDIA's [performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) counts the reads and writes the algorithm itself requires, calls this arithmetic intensity, and calls $P_{\rm peak}/B$ the ops:byte ratio. The former is measured traffic, the latter an ideal count; this note uses the latter for hand calculations, and you look at the former when measuring a real kernel.

The two lines meet at $I^*=P_{\rm peak}/B$, the ridge point. When $I<I^*$, the bound is the sloped line $B\times I$: the operation is memory-bound, and more compute does not help. When $I>I^*$, the bound is the flat line: the operation is compute-bound.

This note uses the H100 SXM5 throughout only because its specifications are public and easy to compute with; it is not a recommendation. The numbers come from Tables 1 and 3 of NVIDIA's [H100 architecture whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf):

| Metric | H100 SXM5 | Basis |
| --- | ---: | --- |
| BF16 Tensor Core peak | 989.4 TFLOPS | dense, FP32 accumulate; the same table gives 1978.9 with sparsity |
| BF16 non-Tensor Core peak | 133.8 TFLOPS | the path elementwise operations take |
| Memory bandwidth | 3352 GB/s | HBM3, 80 GB capacity |
| Ridge point (BF16 Tensor Core) | 989.4 ÷ 3.352 ≈ 295 FLOP/byte | computed from the two rows above |

The [product page](https://www.nvidia.com/en-us/data-center/h100/) lists BF16 at 1,979 TFLOPS and 3.35 TB/s; the former is marked "with sparsity" and is twice the dense value. An earlier 2022 version of the whitepaper (V1.02) listed preliminary values from before the clocks were final (BF16 1000 TFLOPS, 3000 GB/s), while V1.04 states that it includes the final specifications. When numbers disagree, first check the SKU, the document version, the precision, and dense versus sparse.

## Three operations on one chart {#three-ops}

Bytes are counted for the ideal case where every input is read once and every output written once. The time lower bound is $\max(\text{FLOPs}/P_{\rm peak},\ \text{bytes}/B)$, which assumes computation and data movement overlap perfectly.

| Operation | FLOPs | Minimum bytes moved | Arithmetic intensity (FLOP/byte) | Limited by | Time lower bound |
| --- | ---: | ---: | ---: | --- | ---: |
| BF16 GEMM, $M=N=K=8192$ | $2\cdot8192^3\approx1.10\times10^{12}$ | $3\cdot8192^2\cdot2\approx4.03\times10^{8}$ | ≈ 2731 | compute | 1.11 ms |
| BF16 elementwise add, $10^8$ elements | $10^{8}$ | $6\times10^{8}$ | ≈ 0.17 | bandwidth | 0.18 ms |
| 7B model decoding 1 token (batch 1) | $\approx2N=1.4\times10^{10}$ | $\approx1.4\times10^{10}$ (weights) | ≈ 1 | bandwidth | 4.18 ms |

**GEMM.** Multiplying $M\times K$ by $K\times N$ takes $2MNK$ operations (a multiply-add counts as 2); reading and writing the three matrices once takes $2(MK+KN+MN)$ bytes. So $I=MNK/(MK+KN+MN)$, which is $n/3$ for square matrices ([NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)). $8192/3\approx2731$, far above 295. At peak it takes 1.11 ms, while moving the data takes only 0.12 ms; the bottleneck is compute.

**Elementwise add.** Each element reads two BF16 values (4 bytes), writes one (2 bytes), and does a single addition, so $I=1/6$. It does not use Tensor Cores; even against the non-Tensor Core ridge point of $133.8/3.352\approx40$, it is more than two orders of magnitude lower. Its ceiling is $3.352\times10^{12}\times1/6\approx0.56$ TFLOPS, about 0.06% of the Tensor Core peak. That is why elementwise operations such as bias and activation are fused into the epilogue of the preceding GEMM: saving one round trip to memory matters far more than making the addition faster. The [vector add in the CUDA note](cuda.en.md#vector-add) (FP32, 8 bytes read, 4 bytes written, one addition, $I=1/12$) belongs to the same class.

**Decode.** When generating one token, each weight takes part in a single multiply-add (2 FLOP) but costs 2 bytes to read from memory, so $I\approx1$. A 7B model's BF16 weights are about $1.4\times10^{10}$ bytes; reading them once takes at least $1.4\times10^{10}/3.352\times10^{12}\approx4.18$ ms, so a single request tops out at about 239 tokens/s, while the arithmetic itself needs only 0.014 ms. NVIDIA's guide gives an example of the same kind: a fully connected layer at batch size 1 is a matrix-vector product (GEMV) with arithmetic intensity below 1, and is always memory-bound.

<!-- widget:step-roofline -->

## Decode: why a bigger batch moves toward compute-bound {#decode-batching}

### Weights only

In one decode step the weights are read once no matter how many requests are in the batch, while the work grows in proportion to the number of requests $B$. Counting weights only, $I\approx B$.

| batch $B$ | Read weights | Compute | Step lower bound | tokens/s upper bound |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.18 ms | 0.014 ms | 4.18 ms | 239 |
| 64 | 4.18 ms | 0.91 ms | 4.18 ms | 15,300 |
| 295 | 4.18 ms | 4.17 ms | 4.18 ms | 70,600 |
| 512 | 4.18 ms | 7.24 ms | 7.24 ms | 70,700 |

While $B$ is below 295, that is, while $I$ is still left of the ridge point, adding a request barely changes the step time, so throughput grows linearly with $B$. Past $B\approx295$, throughput stops growing; each step just gets slower and every request waits longer. This is why serving engines work so hard to form batches.

### Adding the KV cache

But every request also has its own KV cache, which cannot be shared between requests (except for shared prefixes). Reuse the assumed configuration from [the KV cache note](../00-foundations/deep-dives/kv-cache-and-inference.en.md): 32 layers, 32 query heads, 8 KV heads, head dimension 128, BF16. Each token's KV takes $2\times32\times8\times128\times2=131072$ bytes (128 KiB). With a context of 4096 tokens per request, that is 512 MiB.

Attention work has to be counted too: in each layer, each query head does a length-128 dot product with each of the $T$ positions and then a weighted sum, $4\times128\times T$ FLOP in total; with 32 layers, 32 heads and $T=4096$, that is about $2.15\times10^{9}$ FLOP/token. Each KV element (2 bytes) is shared by 4 query heads, each doing one multiply-add, so the arithmetic intensity of this part is only $4\times2/2=4$ FLOP/byte, **independent of the batch size**.

| batch $B$ | Bytes per step (weights + KV) | Arithmetic intensity | Step lower bound | tokens/s upper bound |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.5 GB | 1.1 | 4.34 ms | 231 |
| 8 | 18.3 GB | 7.1 | 5.46 ms | 1,470 |
| 64 | 48.4 GB | 21.4 | 14.4 ms | 4,440 |

However large $B$ gets, the arithmetic intensity only approaches $(1.4\times10^{10}+2.15\times10^{9})/5.37\times10^{8}\approx30$, far from 295; and at $B=128$ the weights plus KV already take about 77 GiB, nearly all of the 80 GB of memory. At this context length, KV bandwidth and capacity run out before compute does. GQA, MLA, KV quantization and paged management all reduce these bytes or the waste around them; see [the KV memory budget](../00-foundations/deep-dives/kv-cache-and-inference.en.md#inference-budget) and [FlashAttention and PagedAttention](../00-foundations/deep-dives/attention-kernels.en.md).

Prefill is the opposite: it processes thousands of tokens of one request at once. Counting weight bytes only, as in the batch table above, the arithmetic intensity of a linear layer is roughly the number of tokens processed together; counting activation reads and writes as well brings it down (about 1,365 for hidden size 4096 and 4,096 tokens at once), still far above 295. The two phases of the same model sit on opposite sides of the roofline, which is one reason [how an LLM request is served](../06-systems/llm-serving.en.md) treats time to first token and the gap between output tokens separately. All numbers in this section use peak values and do not subtract activations, kernel launches, or bandwidth that is not achieved in practice.

## MFU and HFU: what goes in the numerator and denominator {#mfu}

The PaLM paper ([Chowdhery et al., 2022, §4.1 and Appendix B](https://arxiv.org/abs/2204.02311)) proposed model FLOPs utilization: observed tokens/s divided by the tokens/s the hardware could theoretically reach at peak compute. The key point is that the theoretical maximum throughput, in the paper's words, "only accounts for the required operations to compute the forward+backward passes, and not rematerialization" ([§4.1](https://arxiv.org/html/2204.02311v5#S4.SS1)). For a dense decoder, each token needs $6N$ matrix-multiply operations ($2N$ forward, $4N$ backward), plus $12LHQT$ for the matrix multiplies inside attention:

$$
\mathrm{MFU}=\frac{\text{tokens/s}\times(6N+12LHQT)}{n_{\rm GPU}\times P_{\rm peak}}.
$$

$N$ is the parameter count, $L$ the number of layers, $H$ the number of heads, $Q$ the head dimension, $T$ the sequence length, and $P_{\rm peak}$ the peak matrix-multiply throughput of one device. HFU (hardware FLOPs utilization) counts every operation actually executed, including recomputation; the paper argues it depends on the implementation and is not suited to comparisons across systems. The paper reports an MFU of 46.2% (counting attention; 45.7% without it) and an HFU of 57.8% for PaLM 540B on 6144 TPU v4 chips.

**A teaching example.** A 7B dense model ($N=7\times10^9$, $L=32$, $H=32$, $Q=128$), sequence length $T=4096$, 8 H100 SXM5 GPUs, 64 sequences per step (8 per GPU), 262,144 tokens in total. Assume a measured step time of 3.64 s, i.e. 72,018 tokens/s. Counting only $6N$:

$$
\mathrm{MFU}=\frac{72018\times4.2\times10^{10}}{8\times989.4\times10^{12}}\approx38.2\%.
$$

| Convention | FLOPs counted per token | Result |
| --- | ---: | ---: |
| MFU, $6N$ only | $4.2\times10^{10}$ | 38.2% |
| MFU, plus $12LHQT$ | $4.2\times10^{10}+6.44\times10^{9}$ | 44.1% |
| HFU with full activation recomputation (one extra forward, counted as $8N$, attention excluded) | $5.6\times10^{10}$ | 50.9% |
| sparse peak of 1978.9 TFLOPS wrongly used as the denominator | $4.2\times10^{10}$ | 19.1% |

The statement that full recomputation costs one extra forward pass comes from [Korthikanti et al. (2022)](https://arxiv.org/abs/2205.05198), who also separate MFU from HFU in the way PaLM does. The same training run can be reported as four different numbers between 19% and 51% just by changing the convention. When reporting MFU, state whether attention is counted, which precision the peak refers to, dense or sparse, and whether padding is included in tokens/s (including it inflates the result). A higher HFU than MFU does not mean the run is faster; recomputation is extra work by definition. PaLM chose recomputation because it allows a larger batch and therefore more tokens/s. The memory side of recomputation is covered in [Precision and memory](../00-foundations/deep-dives/precision-and-memory.en.md); the same $6N$ shorthand appears in the training compute estimate $C\approx6ND$, see [Pretraining](../00-foundations/deep-dives/pretraining-pipeline.en.md).

## Multiple GPUs: how long does one all-reduce take {#communication}

The ring all-reduce derivation is already worked out in [Distributed training](../06-systems/distributed-training.en.md): with $n$ ranks each holding $S$ bytes, each rank sends $2(n-1)S/n$. Patarasuk and Yuan proved this is a lower bound (without compressing the data, at least one process must communicate this much) and gave a ring algorithm that reaches it ([JPDC 2009](https://doi.org/10.1016/j.jpdc.2008.09.002)). NVIDIA's [nccl-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) uses this to define bus bandwidth, $B_{\rm bus}=\frac{S}{t}\cdot\frac{2(n-1)}{n}$, which rearranges to:

$$
t_{\rm all\text{-}reduce}\approx\frac{S}{B_{\rm bus}}\cdot\frac{2(n-1)}{n}.
$$

This $B_{\rm bus}$ can be compared directly with link bandwidth and does not change with the number of ranks. Plugging in the busbw that nccl-tests measures on your own machines is more reliable than using numbers from a specification sheet.

Continuing the 7B example, synchronize BF16 gradients with DDP: $S=7\times10^9\times2=14$ GB. The work per GPU per step stays the same (8 sequences × 4096 tokens), so a step still takes 3.64 s.

| Setup | Ranks $n$ | Assumed busbw | Sent per rank | All-reduce time | Share of one step |
| --- | ---: | ---: | ---: | ---: | ---: |
| one node, 8 GPUs | 8 | 360 GB/s | 24.5 GB | 68 ms | 1.9% |
| two nodes, 16 GPUs | 16 | 40 GB/s | 26.25 GB | 656 ms | 18% |

Both busbw values are teaching assumptions. For reference, the whitepaper gives the H100 SXM5 900 GB/s of total NVLink bandwidth (both directions added), so 450 GB/s in one direction is the ceiling inside a node; across nodes the network adapters are usually the limit. The 18% is the upper bound with no overlap at all: [PyTorch DDP](https://docs.pytorch.org/docs/2.14/notes/ddp.html) groups gradients into buckets and starts an asynchronous all-reduce on a bucket as soon as all of its gradients are ready during the backward pass, so how much is hidden depends on whether enough backward computation remains. With gradient accumulation, `no_sync()` skips the intermediate synchronizations so that each optimizer step synchronizes once, which lowers the share further.

Data-parallel communication can overlap with computation; tensor parallelism is different. It synchronizes inside every layer in both the forward and backward passes, the result is needed right away, and so it sits on the critical path and usually depends more on fast interconnects; see [Tensor parallelism](../06-systems/tensor-parallel.en.md). ZeRO-3 / FSDP sends about 1.5 times as much per step as DDP; see [Distributed training](../06-systems/distributed-training.en.md#zero-communication).

## Profiling: three tools, three questions {#profiling}

| Tool | Question it answers | Granularity and cost |
| --- | --- | --- |
| PyTorch profiler (`torch.profiler`) | which PyTorch operators are most expensive, what their input shapes, call stacks and memory allocations are, and how CPU and GPU activity line up | operator level; records GPU kernels through CUPTI; recording shapes and stacks adds overhead |
| Nsight Systems (`nsys`) | what the whole system does on a timeline: CPU threads, CUDA API calls, kernels, copies, NCCL, NVTX ranges | system-wide timeline; good for finding idle GPU time, serial waits, and communication that does not overlap |
| Nsight Compute (`ncu`) | why one particular kernel is slow: Speed Of Light, roofline charts, memory workload (such as sectors/request), occupancy | a single kernel; collects metrics by replaying it several times, serializes kernels, high overhead |

Sources: the [PyTorch 2.14 profiler documentation](https://docs.pytorch.org/docs/2.14/profiler.html), the [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) (v2026.5), and the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) (v2026.3.1). The Nsight Systems guide describes the same flow: use its CUDA trace to locate a suspicious kernel, then run Nsight Compute on that kernel.

The order can be simple:

1. Fix a repeatable workload (same shapes, warm-up steps skipped) and measure each step's wall-clock time after synchronizing.
2. Capture a few steps with the PyTorch profiler or Nsight Systems: are there gaps on the GPU? Is the time going to GEMMs, elementwise operations, NCCL, or copies?
3. Take the few kernels that cost the most, compute their FLOPs and bytes by hand as above, and see how far each is from its bound.
4. Use Nsight Compute only on kernels that are far from their bound.
5. Change one thing at a time and measure again.

```python
from torch.profiler import ProfilerActivity, profile, schedule

def on_ready(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    prof.export_chrome_trace(f"trace_{prof.step_num}.json")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
             on_trace_ready=on_ready, record_shapes=True) as prof:
    for _ in range(6):
        train_step()
        prof.step()
```

`train_step()` stands for one step of your own training. The `schedule` skips step 1, warms up for 1 step and records 3, following the example in the PyTorch documentation. Minimal command-line usage of the other two tools (see their documentation for options):

```text
nsys profile -t cuda,nvtx,osrt -o step_trace python train.py
ncu --set full -k regex:gemm -c 1 -o gemm_report python train.py
```

The second line captures only the first kernel whose name matches `gemm` and collects the full metric set; without `-k` and `-c`, every kernel is replayed several times, which is very expensive.

## Common bottlenecks: symptom and first check {#bottlenecks}

| Symptom | Possible cause | What to check first |
| --- | --- | --- |
| gaps between kernels on the timeline; the GPU is often idle | the CPU cannot keep up: data loading, Python overhead, host synchronization | what the CPU threads are doing during the gaps; look for synchronizations such as `cudaStreamSynchronize` or `.item()`; if copies do not overlap computation, check `pin_memory` and `non_blocking` (see [CUDA](cuda.en.md#streams)) |
| thousands of kernels per step, each running a few microseconds | launch and scheduling overhead dominates | kernels per step and their average duration; try fusion (for example `torch.compile`) or CUDA Graphs, whose main benefit the PyTorch documentation describes as lower CPU overhead |
| elementwise and normalization kernels take most of the time | bandwidth-bound, with data making several round trips to memory | compute intensity from FLOPs and bytes; in Nsight Compute, check whether memory throughput is already near peak; consider fusion; for a hand-written memory-bound kernel, also check sectors/request (ideal is 4 for 32-bit accesses; see [CUDA](cuda.en.md#coalescing)) |
| a large GEMM reaches only a small fraction of Tensor Core peak | wrong dtype (FP32 not using TF32 / BF16), shapes too small or misaligned, tile / wave quantization | print the GEMM's M, N, K and dtype; NVIDIA recommends dimensions that are multiples of 16 bytes (8 elements for FP16 / BF16; 128 bytes on A100) |
| low occupancy, with warps mostly waiting | heavy register or shared memory usage | `nvcc --resource-usage` and the occupancy section of Nsight Compute; compare with the hand calculation in the CUDA note |
| NCCL kernels are long on multiple GPUs and do not overlap computation | communication is exposed on the critical path | whether NCCL overlaps the backward pass in Nsight Systems; measure busbw with nccl-tests and compare with the formula above |
| decode throughput saturates quickly as the batch grows | KV reads or memory capacity hit the ceiling first | use this note's byte accounting to split bytes between weights and KV |

The alignment advice and tile / wave quantization are described in the [NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html); CUDA Graphs are explained in [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs). The table only says where to look first; the cause has to be confirmed by measurement.

## What these estimates cannot tell you {#limits}

Peak compute and bandwidth are specification maxima. Real kernels do not reach them; the roofline gives a ceiling, not a prediction of actual time. The ideal byte counts assume each input is read once: insufficient tiling causes repeated reads, while L2 (50 MB on the H100 SXM5) may absorb some of them, which is why Williams et al. insist on measured DRAM traffic. Taking the $\max$ as the time lower bound assumes perfect overlap of computation and data movement; a poorly pipelined kernel is closer to the sum of the two. With multiple GPUs, topology, the algorithm NCCL chooses (ring, tree, and others) and link contention all change busbw. MFU numbers are comparable only under the same counting convention.

These formulas are best used to ask two questions: how far is this kernel from its bound, and should the optimization reduce bytes or increase reuse? To go back to what to change inside a kernel, read [CUDA](cuda.en.md); for system-level parallelism choices, see [Distributed training](../06-systems/distributed-training.en.md); for end-to-end latency in inference serving, see [how an LLM request is served](../06-systems/llm-serving.en.md).
