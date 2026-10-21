# AI 基础设施：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 8 分钟 · 最近审阅：2026-10

AI 基础设施的面试题常涉及几个相互关联的环节：芯片怎么制造、GPU 怎么运行 kernel、框架怎么发射 kernel、集群怎么分配 GPU。回答时，先说清比较口径，包括精度、dense 还是 sparse，以及具体 SKU；再判断瓶颈在哪一层，用相应的数字估算限制。下面的数字都来自本系列正文，推导和出处在对应的笔记里。

## 面试常见问题

<details class="interview" markdown="1">
<summary>为什么说“7 nm”“N4”这样的制程节点名不是一个物理尺寸？比较两颗芯片时该看什么？</summary>

早年的节点名大致对应晶体管的某个物理尺寸，今天更像一代工艺的名字：IEEE Spectrum 2020 年的文章指出，“7 nm”晶体管的大部分关键尺寸远大于 7 nm，并介绍了 Wong 等人提出的 LMC 指标（逻辑密度、内存密度和两者之间的连接密度）。厂商的提升数字要先看基准：TSMC 说 N4 是 N5 的增强版，N4P 的“性能高 11%”是相对 N5；N2P 相对 N3E 是同功耗快 18% 或同速度功耗低 36%，两者一般不能同时拿到。NVIDIA 的 4N、4NP 是定制工艺，和 TSMC 公开的 N4 系列差在哪里，没有找到公开说明。

所以节点名只是起点，最后要比同一个 workload 下实测的性能、功耗和成本。真正卡住单块 die 面积的是一次曝光的视场：26 mm × 33 mm，即 858 mm²，GH100 的 814 mm² 已经占了 95%。见[芯片怎么造出来](chips-and-fabs.md)。

</details>

<details class="interview" markdown="1">
<summary>为什么 batch 为 1 的 decode 受显存带宽限制？加大 batch 改变了什么，又没改变什么？</summary>

生成一个 token，每个权重只做一次乘加（2 FLOP），BF16 下却要读 2 字节，算术强度约 1 FLOP/字节；H100 SXM5 的屋脊点是 dense BF16 的 989.4 TFLOPS 除以 3,352 GB/s，约 295。7B 模型的 BF16 权重约 $1.4\times10^{10}$ 字节，读一遍至少 4.18 ms，单请求最多约 239 tokens/s，而这些运算本身只要 0.014 ms。

加大 batch 分摊的是权重：只算权重时，强度约等于 batch 大小 $B$；$B$ 小于约 295 时每步时间几乎不变，吞吐随 $B$ 线性上涨。分摊不了的是 KV：每条请求有自己的 KV，attention 这部分的强度只有约 4 FLOP/字节，和 batch 无关。按每条请求 4096 token 上下文算，$B=64$ 时每步下限已是 14.4 ms，吞吐上限约 4,440 tokens/s；$B=128$ 时权重加 KV 约 77 GiB。先碰到的是 KV 的带宽和容量，不是算力。见[性能](performance.md#decode-batching)。

</details>

<details class="interview" markdown="1">
<summary>一个 kernel 的 occupancy 由什么决定？是不是越高越好？</summary>

Occupancy 是一个 SM 上活跃的 warp 数，除以这个 SM 最多能驻留的 warp 数，由最先用完的那项资源决定。Compute capability 9.0（如 H100）每个 SM 最多 2048 个线程（64 个 warp）、32 个 block、65,536 个 32 位寄存器、228 KB shared memory。每个 block 256 个线程时：每线程 32 个寄存器能驻留 8 个 block，100%；64 个寄存器只剩 4 个，50%；每个 block 用 48 KB shared 也是 4 个，50%；80 个寄存器只剩 3 个，37.5%。寄存器还按每个 warp 256 个向上取整；实际先用 `nvcc --resource-usage` 看用量，再用 Nsight Compute 或 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 核对。

Occupancy 回答的是“有没有足够的 warp 可以切换”，不是“这个 kernel 快不快”。Best Practices Guide 写明，超过某个点再提高 occupancy 不会更快；高性能矩阵乘 kernel 常常每线程用很多寄存器、occupancy 不高，靠指令级并行和异步拷贝（Ampere 起的硬件异步拷贝、Hopper 的 TMA）把等待藏起来。延迟可以藏，带宽上限藏不掉。见 [CUDA](cuda.md#occupancy) 和 [GPU 架构](gpu-architecture.md)。

</details>

<details class="interview" markdown="1">
<summary>什么是访存合并（coalescing）？stride 访问为什么浪费带宽？</summary>

Global memory 以 32 字节为单位传输。一个 warp 执行同一条 load 时，硬件把 32 个线程的地址合并，碰到几个 32 字节段就发几个 transaction。每个线程读一个 float32、起点对齐时：相邻线程读相邻元素，4 个段、128 字节全部用上；起点错开一个元素，5 个段，用上 80%；stride 32 时每个线程独占一段，搬 1024 字节只用上 128 字节，即 12.5%。按列读一个行优先的矩阵就是这种情况。

段数不等于耗时：Best Practices Guide 在 V100 上测“起点错开”的拷贝，带宽约为对齐时的 9/10，而不是按段数推出的 4/5，因为相邻 warp 会用上 cache 里多拉进来的数据。排查时看 Nsight Compute 的 sectors/request：32 位访问的理想值是 4，stride 32 时接近 32。改法是让 `threadIdx.x` 落在地址的最低维。见 [CUDA](cuda.md#coalescing)。

</details>

<details class="interview" markdown="1">
<summary>数据表上的 dense 和 sparse 数字差在哪里？比较两代 GPU 之前要先对齐什么？</summary>

Sparse 数字来自 Ampere 引入的 2:4 结构化稀疏：每 4 个值里最多 2 个非零时，Tensor Core 吞吐翻倍，普通 dense 模型拿不到这个倍数。H100 产品页的 BF16 1,979 TFLOPS 注明 with sparsity，白皮书 V1.04 的 dense 值是 989.4 TFLOPS。但不能一律除以 2：GB300 NVL72 页面的 FP4 是 1,440（sparse）对 1,080（dense）。

还要对齐三件事：SKU（同叫 H100，SXM 版 80 GB、3.35 TB/s，NVL 版 94 GB、3.9 TB/s）；计数单位（GB200 superchip 是 1 颗 Grace 加 2 颗 GPU，HGX 页面给 8 卡合计）；初步还是最终规格（2022 年 Hopper 博客的 1,000 TFLOPS、3,000 GB/s 是初步值）。对齐之后，从 V100 的 FP16 到 GB300 的 NVFP4，dense 峰值涨了 120 倍，显存带宽只涨约 9 倍，所以 decode 这类读得多、算得少的工作，代际提速主要跟着带宽走。官方页面之间也会对不上，例如 Rubin 的每卡显存带宽有 22 和 19.2 TB/s 两种写法。见 [NVIDIA 各代 GPU](nvidia-generations.md)。

</details>

<details class="interview" markdown="1">
<summary>MFU 衡量的是什么？它和 nvidia-smi 的 GPU 利用率、HFU 有什么不同？</summary>

MFU 由 PaLM 论文提出：观测到的 tokens/s，除以硬件按峰值算力运行时理论上能达到的 tokens/s。理论值只计前向和反向必需的运算（dense decoder 每个 token 约 $6N$，attention 里的矩阵乘再加 $12LHQT$），**不计重算**。HFU 把重算也算进去，依赖实现，不适合横向比较；论文报告 PaLM 540B 的 MFU 为 46.2%，HFU 为 57.8%。`nvidia-smi` 的利用率是另一回事：NVML 把它定义为采样周期里至少有一个 kernel 在跑的时间比例。

口径一变，数字就变。同一个教学算例（7B 模型、8 张 H100 SXM5、72,018 tokens/s）：只算 $6N$ 是 38.2%，加上 attention 是 44.1%，全量重算的 HFU 是 50.9%，误把 sparse 峰值当分母只剩 19.1%。报告时要写明是否计 attention、峰值是哪种精度、dense 还是 sparse、tokens/s 有没有算 padding。见[性能](performance.md#mfu)。

</details>

<details class="interview" markdown="1">
<summary>torch.compile 融合了什么？为什么能变快？</summary>

Eager 模式下，`y = torch.relu(x @ w + b)` 是 3 个 ATen 算子、至少 3 个 kernel：每个算子都要走 Python 调用、dispatcher、分配输出和提交 kernel，逐元素算子还各自把中间结果从显存读写一遍。`torch.compile` 由 TorchDynamo 抓出 FX graph（带 guard，条件不满足就重编译；遇到不支持的 Python 会 graph break），AOTAutograd 提前抓下反向，TorchInductor 为 GPU 生成 Triton 代码并融合 kernel。

省下的不是乘法。按 4096×4096、BF16 和 2 TB/s 的教学带宽：eager 的 add 和 relu 共读写 128 MiB，约 67 µs；融合成一个 kernel 是 64 MiB，约 34 µs；bias 和 ReLU 都进 GEMM 的 epilogue 就没有额外读写。按 500 TFLOP/s 的教学算力，GEMM 本身约 275 µs，eager 的逐元素部分约相当于它的 24%。把 $M$ 降到 8，字节可以忽略，省的主要是 CPU 提交次数；`reduce-overhead` 用 CUDA Graphs 把一串提交合成一次。代价是首次编译时间、形状变化引起的重编译和 graph break。见[框架](frameworks.md)。

</details>

<details class="interview" markdown="1">
<summary>为什么分布式训练在 Kubernetes 上需要成组调度（gang scheduling）？</summary>

默认的 kube-scheduler 一次只给一个 Pod 找节点，分布式训练却要所有 rank 到齐才能开始；GPU 又是只能取整数、不能超卖的 extended resource。手算一次：3 台节点各 8 卡；依次来了两个 4 卡任务、一个 16 卡训练 T16（2 个 Pod，各 8 卡）和一个 8 卡任务 T8。默认打分只看 CPU 和内存，两个 4 卡任务可能各占一台；T16 的一个 Pod 拿下第三台，另一个放不下，起来的一半占着 8 张卡空等；剩下两台各空 4 张，也拼不出 T8。24 张卡只有 8 张在干活，再来一个 T16 也只能排队。双任务互等需要额外条件：若只有两台整机可供训练、第三台长期被其他任务占用，两个 T16 可能各占一台、各等另一半；没有资源释放、超时回收或抢占时才会死锁。原表展示的碎片化不能单独证明这一点。

成组调度放不下就一个都不绑定，第三台留给 T8，16 张卡在干活；再按 GPU 装箱（MostAllocated 并给 GPU 加权），两个小任务挤进一台，24 张都在干活，轮到 T8 排队。实现方式不止一种：Kubernetes 核心的 gang scheduling 在 v1.37 是 beta、默认关闭；Kueue 用 `waitForPodsReady` 超时驱逐、重新排队；Volcano 用 PodGroup 的 `minMember`。另外，Slurm 文档里的 “gang scheduling” 指分时轮流，和这里的全有或全无不是一回事。见 [Kubernetes](kubernetes.md)。

</details>

## 自检

<div class="taste-check">
  <strong>可以用下面的问题检查理解：</strong>
  <ol>
    <li>同样设 $D_0=0.1$ 个/cm²，800 mm² 的整块 die 每片晶圆约 29 块好 die，拆成两块 400 mm² 却能配出 48 对；这笔账漏掉了哪些成本？</li>
    <li>分块把矩阵乘的 global 读取从 $2n^3$ 降到 $2n^3/T$，为什么 Best Practices Guide 在 V100 上只把有效带宽从 119.9 GB/s 提到 195.5 GB/s？</li>
    <li>NVLink 900 GB/s、PCIe 128 GB/s、InfiniBand 400 Gb/s 各是什么口径？都换成每个方向的字节后，从 HBM 到网卡差多少？</li>
    <li>70B 模型 batch 为 1 时，为什么 H100 上 FP8 一张卡和 BF16 两张卡的每 token 下限都是 20.9 ms，而两张 A100 比五张 V100 还慢？</li>
    <li>同样 14 GB 的 BF16 梯度，按文中假设的 busbw，为什么单机 8 卡的 all-reduce 只占一步约 1.9%，两机 16 卡却占 18%？DDP 的 bucket 能藏掉多少，tensor parallel 为什么藏不了？</li>
  </ol>
</div>

## 参考论文

- [NVIDIA H100 Tensor Core GPU Architecture 白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)：H100 SXM5 的最终规格，dense 与 sparse 峰值、显存带宽
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)：2022 年的架构博客，含初步规格、NVLink 与 PCIe 带宽、FP8
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)：TF32、BF16 与 2:4 结构化稀疏
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)：线程层级、内存空间、occupancy 与各 compute capability 的上限
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：访存合并、shared memory 分块、occupancy、拷贝与计算重叠
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)：Williams、Waterman、Patterson，CACM 2009，屋脊模型
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)：Pope 等，2022，小 batch 时读权重为主，batch 和序列变长后读 KV 为主
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)：§4.1 与附录 B 定义 MFU 和 HFU
- [Bandwidth optimal all-reduce algorithms for clusters of workstations](https://doi.org/10.1016/j.jpdc.2008.09.002)：Patarasuk、Yuan，JPDC 2009，ring all-reduce 的通信下界
- [PyTorch 2.14 文档：torch.compiler](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html)：TorchDynamo、AOTAutograd 与 TorchInductor
- [PyTorch 2.14 文档：CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)：异步执行、stream、pinned memory 与 CUDA Graphs
- [Kubernetes 文档：GPU 调度](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)：device plugin 与 `nvidia.com/gpu`
- [Kubernetes 文档：gang scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/gang-scheduling/)：一组 Pod 全有或全无
- [Kueue](https://kueue.sigs.k8s.io/docs/overview/)：kube-scheduler 之上的作业级排队、配额与准入
- [A Better Way to Measure Progress in Semiconductors](https://spectrum.ieee.org/a-better-way-to-measure-progress-in-semiconductors)：IEEE Spectrum 2020 年的文章，节点名与物理尺寸脱钩，以及 LMC 指标
