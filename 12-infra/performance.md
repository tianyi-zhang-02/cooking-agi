# 性能：roofline、算术强度与 MFU

**中文** · [English](performance.en.md)

> 阅读时间：约 13 分钟 · 草稿，待审核

`nvidia-smi` 显示 GPU 利用率 100%，一个 7B 模型逐个生成 token，却只有每秒两百来个。是卡不够快，还是没用好？先看那个 100% 量的是什么：NVML 把它定义为采样周期里“至少有一个 kernel 在跑”的时间比例（[NVML 文档](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlUtilization__t.html)），它并不说明算力用掉了多少。**更有用的问题是：这个操作每从显存搬 1 字节，要做几次运算？拿它和这张卡的“峰值算力 ÷ 显存带宽”比：低了就该减少搬运，高了才轮到算力。** 下面用同一张 H100 的公开规格，把大矩阵乘、逐元素加法和 7B 模型的 decode 放到同一张图上，再算 MFU、多卡通信，最后排出 profiling 的顺序。

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

本篇接着 [CUDA：线程、内存与 kernel](cuda.md)。除非注明是论文报告的结果，文中数字都是按公开规格算出的上限或教学假设，不是 benchmark。

## Roofline：两条线围出的上限 {#roofline}

Williams、Waterman 和 Patterson 的 roofline 模型（也叫屋脊模型；[CACM 52(4), 2009](https://doi.org/10.1145/1498765.1498785)；[2008 年技术报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)）只用两个硬件数字和一个程序数字：

$$
P_{\rm attainable}=\min\left(P_{\rm peak},\;B\times I\right).
$$

$P_{\rm peak}$ 是峰值算力（FLOP/s），$B$ 是显存带宽（byte/s），$I$ 是每搬 1 字节做多少次浮点运算（FLOP/byte）。原论文把 $I$ 叫 operational intensity，并特意规定字节按经过 cache 过滤、真正到达 DRAM 的流量算。NVIDIA 的[性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)按算法本身的读写量来数，叫 arithmetic intensity，并把 $P_{\rm peak}/B$ 叫 ops:byte ratio。前者是实测流量，后者是理想计数；本篇手算用后者，量真实 kernel 时看前者。

两条线的交点 $I^*=P_{\rm peak}/B$ 叫 ridge point（屋脊点）。$I<I^*$ 时，上限是斜线 $B\times I$，属于带宽受限（memory-bound），多给算力也没用；$I>I^*$ 时，上限是水平线，属于算力受限（compute-bound）。

本篇统一用 H100 SXM5，只因为它的规格公开、方便手算，不代表推荐。数字取自 NVIDIA [H100 架构白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf) 的表 1 与表 3：

| 指标 | H100 SXM5 | 口径 |
| --- | ---: | --- |
| BF16 Tensor Core 峰值 | 989.4 TFLOPS | dense，FP32 累加；同表 sparse 为 1978.9 |
| BF16 非 Tensor Core 峰值 | 133.8 TFLOPS | 逐元素运算走这条路 |
| 显存带宽 | 3352 GB/s | HBM3，容量 80 GB |
| Ridge point（BF16 Tensor Core） | 989.4 ÷ 3.352 ≈ 295 FLOP/byte | 由前面两行算出 |

[产品页](https://www.nvidia.com/en-us/data-center/h100/)写的是 BF16 1,979 TFLOPS 和 3.35 TB/s；前者标注 with sparsity，是 dense 值的两倍。2022 年的白皮书早期版本（V1.02）还列过时钟未定稿时的预估值（BF16 1000 TFLOPS、3000 GB/s），V1.04 则注明已包含最终规格。遇到对不上的数字，先确认 SKU、文档版本、精度和 dense / sparse。

## 三个操作放上同一张图 {#three-ops}

字节数按“每个输入读一次、输出写一次”的理想情况算。时间下限取 $\max(\text{FLOPs}/P_{\rm peak},\ \text{bytes}/B)$，也就是假设计算和搬运完全重叠。

| 操作 | FLOPs | 最少搬运字节 | 算术强度（FLOP/byte） | 受谁限制 | 时间下限 |
| --- | ---: | ---: | ---: | --- | ---: |
| BF16 GEMM，$M=N=K=8192$ | $2\cdot8192^3\approx1.10\times10^{12}$ | $3\cdot8192^2\cdot2\approx4.03\times10^{8}$ | ≈ 2731 | 算力 | 1.11 ms |
| BF16 逐元素加，$10^8$ 个元素 | $10^{8}$ | $6\times10^{8}$ | ≈ 0.17 | 带宽 | 0.18 ms |
| 7B 模型 decode 1 个 token（batch 1） | $\approx2N=1.4\times10^{10}$ | $\approx1.4\times10^{10}$（权重） | ≈ 1 | 带宽 | 4.18 ms |

**GEMM。** $M\times K$ 乘 $K\times N$ 要 $2MNK$ 次运算（一次乘加算 2 次）；三个矩阵各读写一次，共 $2(MK+KN+MN)$ 字节。所以 $I=MNK/(MK+KN+MN)$，方阵时等于 $n/3$（[NVIDIA 矩阵乘指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)）。$8192/3\approx2731$，远高于 295。按峰值要 1.11 ms，搬运只要 0.12 ms，瓶颈在算力。

**逐元素加。** 每个元素读两个 BF16（4 字节）、写一个（2 字节），只做 1 次加法，$I=1/6$。它不走 Tensor Core；即使和非 Tensor Core 的 ridge point $133.8/3.352\approx40$ 比，也低了两个多数量级。它能达到的上限是 $3.352\times10^{12}\times1/6\approx0.56$ TFLOPS，约为 Tensor Core 峰值的 0.06%。所以要把 bias、激活这类逐元素运算融合进前一个 GEMM 的收尾：省掉一次显存往返，比把加法算快有用得多。[CUDA 篇的 vector add](cuda.md#vector-add)（FP32，读 8 字节、写 4 字节、1 次加法，$I=1/12$）也属于这一类。

**Decode。** 生成一个 token 时，每个权重只参与一次乘加（2 FLOP），却要从显存读 2 字节，$I\approx1$。7B 模型的 BF16 权重约 $1.4\times10^{10}$ 字节，读一遍至少要 $1.4\times10^{10}/3.352\times10^{12}\approx4.18$ ms，所以单条请求最多约 239 tokens/s；把这些运算做完却只要 0.014 ms。NVIDIA 指南举过同类的例子：batch 1 的全连接层是矩阵乘向量（GEMV），算术强度小于 1，总是带宽受限。

<!-- widget:step-roofline -->

## Decode：batch 变大，为什么会走向算力受限 {#decode-batching}

### 先只看权重

一步 decode 不管 batch 里有几条请求，权重都只读一遍；运算量却和请求数 $B$ 成正比。只算权重时，$I\approx B$。

| batch $B$ | 读权重 | 计算 | 每步下限 | tokens/s 上限 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.18 ms | 0.014 ms | 4.18 ms | 239 |
| 64 | 4.18 ms | 0.91 ms | 4.18 ms | 15,300 |
| 295 | 4.18 ms | 4.17 ms | 4.18 ms | 70,600 |
| 512 | 4.18 ms | 7.24 ms | 7.24 ms | 70,700 |

$B$ 小于 295，也就是 $I$ 还在 ridge point 左边时，多加一条请求几乎不增加每步时间，吞吐随 $B$ 线性上涨；过了 $B\approx295$，吞吐不再涨，只是每一步更慢，每条请求等得更久。这就是 serving 引擎拼命组 batch 的原因。

### 再把 KV cache 算进来

可是每条请求还有自己的 KV cache，KV 不能在请求之间共用（共享前缀除外）。沿用 [KV cache 一篇](../00-foundations/deep-dives/kv-cache-and-inference.md)的假设配置：32 层、32 个 query heads、8 个 KV heads、head dimension 128、BF16，每个 token 的 KV 占 $2\times32\times8\times128\times2=131072$ 字节（128 KiB）。每条请求的上下文取 4096 个 token，就是 512 MiB。

Attention 的运算也要算进来：每层每个 query head 要和 $T$ 个位置各做一次长度 128 的点积，再做一次加权求和，共 $4\times128\times T$ FLOP；32 层、32 个 head、$T=4096$ 时约 $2.15\times10^{9}$ FLOP/token。每个 KV 元素（2 字节）被 4 个 query head 共用，各做一次乘加，所以这部分的算术强度只有 $4\times2/2=4$ FLOP/byte，**和 batch 大小无关**。

| batch $B$ | 每步搬运（权重 + KV） | 算术强度 | 每步下限 | tokens/s 上限 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.5 GB | 1.1 | 4.34 ms | 231 |
| 8 | 18.3 GB | 7.1 | 5.46 ms | 1,470 |
| 64 | 48.4 GB | 21.4 | 14.4 ms | 4,440 |

$B$ 再大，算术强度也只会逼近 $(1.4\times10^{10}+2.15\times10^{9})/5.37\times10^{8}\approx30$，离 295 很远；而 $B=128$ 时，权重加 KV 已约 77 GiB，几乎占满 80 GB 显存。所以在这个上下文长度下，先碰到的是 KV 的带宽和容量，而不是算力。GQA、MLA、KV 量化和分页管理都是在减这部分字节或浪费，见 [KV 显存预算](../00-foundations/deep-dives/kv-cache-and-inference.md#inference-budget)和 [FlashAttention 与 PagedAttention](../00-foundations/deep-dives/attention-kernels.md)。

反过来，prefill 一次处理一条请求的几千个 token。和上面的 batch 表一样只算权重字节时，线性层的算术强度约等于一次处理的 token 数；把激活的读写也算上会低一些（hidden size 4096、一次 4096 个 token 时约 1,365），仍远高于 295。同一个模型的两个阶段落在 roofline 两侧，这也是[一次 LLM 请求](../06-systems/llm-serving.md)里要把首 token 延迟和输出间隔分开看的一个原因。本节数字都按峰值算，没有扣除激活、kernel launch 和实际达不到的带宽。

## MFU 与 HFU：分子分母各算什么 {#mfu}

PaLM 论文（[Chowdhery 等，2022，§4.1 与附录 B](https://arxiv.org/abs/2204.02311)）提出 model FLOPs utilization：观测到的 tokens/s，除以硬件按峰值算力运行时理论上能达到的 tokens/s。关键在于，论文写明理论最大吞吐“only accounts for the required operations to compute the forward+backward passes, and not rematerialization”（[§4.1](https://arxiv.org/html/2204.02311v5#S4.SS1)），即只计前向加反向必需的运算，**不计重算（rematerialization）**。对 dense decoder，每个 token 需要 $6N$ 次矩阵乘运算（前向 $2N$、反向 $4N$），attention 里的矩阵乘再加 $12LHQT$：

$$
\mathrm{MFU}=\frac{\text{tokens/s}\times(6N+12LHQT)}{n_{\rm GPU}\times P_{\rm peak}}.
$$

$N$ 是参数量，$L$ 是层数，$H$ 是 head 数，$Q$ 是 head dimension，$T$ 是序列长度，$P_{\rm peak}$ 是每张卡的峰值矩阵乘算力。HFU（hardware FLOPs utilization）则把实际执行的运算都算进去，包括重算；论文认为它依赖实现，不适合横向比较。论文报告 PaLM 540B 在 6144 块 TPU v4 上的 MFU 为 46.2%（计入 attention；不计时 45.7%），HFU 为 57.8%。

**一个教学算例。** 7B dense 模型（$N=7\times10^9$，$L=32$，$H=32$，$Q=128$），序列长 $T=4096$，8 张 H100 SXM5，每步 64 条序列（每卡 8 条），共 262,144 个 token。假设实测每步 3.64 s，即 72,018 tokens/s。只算 $6N$ 时：

$$
\mathrm{MFU}=\frac{72018\times4.2\times10^{10}}{8\times989.4\times10^{12}}\approx38.2\%.
$$

| 口径 | 每 token 计入的 FLOPs | 结果 |
| --- | ---: | ---: |
| MFU，只算 $6N$ | $4.2\times10^{10}$ | 38.2% |
| MFU，加上 $12LHQT$ | $4.2\times10^{10}+6.44\times10^{9}$ | 44.1% |
| HFU，全量 activation 重算（多一次前向，记 $8N$，不含 attention） | $5.6\times10^{10}$ | 50.9% |
| 错把 sparse 峰值 1978.9 TFLOPS 当分母 | $4.2\times10^{10}$ | 19.1% |

“全量重算多一次前向”的说法来自 [Korthikanti 等（2022）](https://arxiv.org/abs/2205.05198)，他们也按 PaLM 的方式区分 MFU 与 HFU。同一次训练，换个口径就能写出 19% 到 51% 的四个数字。报告 MFU 时要写明：是否计 attention、峰值用的哪种精度、dense 还是 sparse、tokens/s 有没有把 padding 算进去（算进去会虚高）。HFU 比 MFU 高，不代表跑得更快，重算本来就是多做的功；PaLM 选择重算，是因为它允许更大的 batch，从而提高 tokens/s。重算的显存账见[精度与显存](../00-foundations/deep-dives/precision-and-memory.md)；$6N$ 这个简式在训练总算力 $C\approx6ND$ 里也会出现，见[预训练流程](../00-foundations/deep-dives/pretraining-pipeline.md)。

## 多卡：一次 all-reduce 要多久 {#communication}

Ring all-reduce 的推导在[多卡训练](../06-systems/distributed-training.md)里已经算过：$n$ 个 rank 各有 $S$ 字节，每个 rank 要发送 $2(n-1)S/n$。Patarasuk 和 Yuan 证明这是下界：数据不压缩时，至少有一个进程要通信这么多；他们也给出了达到这个下界的 ring 算法（[JPDC 2009](https://doi.org/10.1016/j.jpdc.2008.09.002)）。NVIDIA 的 [nccl-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) 据此定义 bus bandwidth，$B_{\rm bus}=\frac{S}{t}\cdot\frac{2(n-1)}{n}$，反过来就是：

$$
t_{\rm all\text{-}reduce}\approx\frac{S}{B_{\rm bus}}\cdot\frac{2(n-1)}{n}.
$$

这个 $B_{\rm bus}$ 可以直接和链路带宽比，也不随 rank 数变化。拿 nccl-tests 在自己机器上测出的 busbw 代入，比用规格表上的数字可靠。

接着上面的 7B 例子，用 DDP 同步 BF16 梯度：$S=7\times10^9\times2=14$ GB。每卡每步的计算量不变（8 条 × 4096 token），一步仍按 3.64 s 算。

| 场景 | rank 数 $n$ | 假设的 busbw | 每 rank 发送 | all-reduce 时间 | 占一步的比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单机 8 卡 | 8 | 360 GB/s | 24.5 GB | 68 ms | 1.9% |
| 两机 16 卡 | 16 | 40 GB/s | 26.25 GB | 656 ms | 18% |

两个 busbw 都是教学假设。作为参照，白皮书给 H100 SXM5 的 NVLink 是 900 GB/s 总带宽（两个方向相加），单方向 450 GB/s 是机内的上限；跨机时一般受网卡限制。18% 是完全不重叠时的上限：[PyTorch DDP](https://docs.pytorch.org/docs/2.14/notes/ddp.html) 把梯度分成 bucket，一个 bucket 的梯度在反向中全部算好，就异步发起它的 all-reduce，所以能藏掉多少，取决于剩下的反向计算够不够长。梯度累积时用 `no_sync()` 跳过中间几次同步，一个优化步只同步一次，比例还会再降。

Data parallel 的通信可以和计算重叠，tensor parallel 不一样：它在每一层的前向和反向里都要同步，结果紧接着就要用，正在关键路径上，所以通常更依赖高速互联，见[张量并行](../06-systems/tensor-parallel.md)。ZeRO-3 / FSDP 每步的通信量约是 DDP 的 1.5 倍，见[多卡训练](../06-systems/distributed-training.md#zero-communication)。

## Profiling：三个工具各回答一个问题 {#profiling}

| 工具 | 回答什么问题 | 粒度与代价 |
| --- | --- | --- |
| PyTorch profiler（`torch.profiler`） | 哪些 PyTorch 算子最贵，输入 shape、调用栈和显存分配是什么，CPU 与 GPU 活动怎样对应 | 算子级；通过 CUPTI 记录 GPU kernel；记录 shape 和调用栈会增加开销 |
| Nsight Systems（`nsys`） | 整个系统在时间线上做什么：CPU 线程、CUDA API、kernel、拷贝、NCCL、NVTX 区间 | 全系统时间线；适合找 GPU 空闲、串行等待和没重叠的通信 |
| Nsight Compute（`ncu`） | 某一个 kernel 为什么慢：Speed Of Light、roofline 图、memory workload（如 sectors/request）、occupancy | 单个 kernel；靠多次 replay 收集指标，kernel 会被串行化，开销大 |

依据：[PyTorch 2.14 profiler 文档](https://docs.pytorch.org/docs/2.14/profiler.html)、[Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)（v2026.5）和 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)（v2026.3.1）。Nsight Systems 的指南里描述的流程也是这样：先用它的 CUDA trace 找到可疑的 kernel，再对那个 kernel 跑 Nsight Compute。

顺序可以很朴素：

1. 固定一个可重复的工作负载（相同 shape，跳过预热步），用同步后的墙钟时间量每步耗时。
2. 用 PyTorch profiler 或 Nsight Systems 抓几步时间线：GPU 有没有空隙？时间花在 GEMM、逐元素运算、NCCL 还是拷贝上？
3. 挑耗时最多的几个 kernel，用前面的方法手算 FLOPs 和字节，看离各自的上限有多远。
4. 只对离上限很远的 kernel 用 Nsight Compute 深挖。
5. 一次只改一处，重新测量。

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

`train_step()` 代表你自己的一步训练。`schedule` 跳过第 1 步、预热 1 步、记录 3 步，写法与 PyTorch 文档的示例一致。命令行工具的最小用法（选项见各自文档）：

```text
nsys profile -t cuda,nvtx,osrt -o step_trace python train.py
ncu --set full -k regex:gemm -c 1 -o gemm_report python train.py
```

第二行只抓第一个名字匹配 `gemm` 的 kernel，并收集完整的指标集合；不加 `-k` 和 `-c`，每个 kernel 都会被 replay 多遍，开销很大。

## 常见瓶颈：症状与第一步检查 {#bottlenecks}

| 症状 | 可能的原因 | 先查什么 |
| --- | --- | --- |
| 时间线上 kernel 之间有空隙，GPU 经常闲着 | CPU 跟不上：数据加载、Python 开销、host 同步 | 空隙期间 CPU 线程在做什么；有没有 `cudaStreamSynchronize`、`.item()` 一类同步；拷贝没和计算重叠时查 `pin_memory` 与 `non_blocking`（见 [CUDA](cuda.md#streams)） |
| 每步有成千上万个只跑几微秒的小 kernel | launch 和调度开销占主导 | 每步 kernel 数和平均时长；试融合（如 `torch.compile`）或 CUDA Graphs，PyTorch 文档说后者的主要收益是减少 CPU 开销 |
| 逐元素、归一化 kernel 占了大头 | 带宽受限，且数据多次往返显存 | 按 FLOPs 和字节算强度；Nsight Compute 看显存吞吐是否已接近峰值；考虑融合。自己写的访存 kernel 还要看 sectors/request（32-bit 访问的理想值是 4，见 [CUDA](cuda.md#coalescing)） |
| 大 GEMM 只达到 Tensor Core 峰值的很小比例 | dtype 不对（FP32 没走 TF32 / BF16）、shape 太小或没对齐、tile / wave quantization | 打印 GEMM 的 M、N、K 和 dtype；NVIDIA 建议维度是 16 字节的倍数（FP16 / BF16 即 8 个元素；A100 上建议 128 字节） |
| Occupancy 低，warp 大多在等待 | 寄存器或 shared memory 用量大 | `nvcc --resource-usage` 与 Nsight Compute 的 occupancy 一节；对照 CUDA 篇的手算 |
| 多卡时 NCCL kernel 很长，且和计算没有重叠 | 通信暴露在关键路径上 | Nsight Systems 里 NCCL 与反向是否重叠；用 nccl-tests 测 busbw，和本页公式比 |
| decode 吞吐随 batch 增长很快就饱和 | KV 读取或显存容量先到顶 | 用本页的字节账算出权重和 KV 各占多少 |

对齐建议和 tile / wave quantization 见 [NVIDIA 矩阵乘指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)，CUDA Graphs 的说明见 [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)。这张表只给出第一步该看哪里，原因要靠测量确认。

## 这些估算不能告诉你什么 {#limits}

峰值算力和带宽是规格上限。真实 kernel 达不到，roofline 只给出天花板，不预测实际耗时。理想字节数假设每个输入只读一次：tiling 不够时会重复读，L2（H100 SXM5 为 50 MB）又可能挡掉一部分读取，所以 Williams 等人坚持用实测的 DRAM 流量。取 $\max$ 作为时间下限，假设计算和搬运完全重叠；流水做得不好的 kernel 更接近两者相加。多卡时，拓扑、NCCL 选择的算法（ring、tree 等）和链路争用都会改变 busbw。MFU 只有在同一套计数口径下才能比较。

所以这些公式最适合用来问两个问题：这个 kernel 离它的上限还有多远？优化方向是减少字节，还是增加复用？想回到 kernel 层面怎么改，读 [CUDA](cuda.md)；系统层面的并行选择见[多卡训练](../06-systems/distributed-training.md)，推理服务的整体时延见[一次 LLM 请求](../06-systems/llm-serving.md)。
