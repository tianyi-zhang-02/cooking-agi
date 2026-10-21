# GPU 架构：SM、warp、内存层级与 Tensor Core

**中文** · [English](gpu-architecture.en.md)

> 阅读时间：约 13 分钟 · 最近审阅：2026-10

一张 H100 SXM 做大矩阵乘法时，dense BF16 的 Tensor Core 峰值是 989.4 TFLOPS；可同一张卡给大模型一个一个生成 token 时，大部分时间在等数据。这两件事并不矛盾。**GPU 追求的是总吞吐，不是单个线程跑得快：它同时挂着很多 warp，谁在等显存就换谁；而 Tensor Core 能不能吃饱，最后取决于数据能多快搬进 SM。**

硬件数字以 H100 SXM5 为主，最终规格取自 [H100 架构白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)（它注明收录了最终的频率和 TFLOPS），其余来自 NVIDIA 的产品页、架构博客和 CUDA 文档；手算例子里的周期数和矩阵元素是教学假设，会单独标出。怎么写 kernel 见 [CUDA](cuda.md)，roofline 见[性能](performance.md)，各代参数对比见 [NVIDIA 各代 GPU](nvidia-generations.md)。

## 两种设计目标：少等一会儿，还是同时多做

[CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)的说法很直接：CPU 要把一串串行操作尽快做完，同时跑几十个线程；GPU 要同时跑成千上万个线程，用较低的单线程性能换更高的总吞吐。所以 CPU 把更多晶体管花在缓存和控制逻辑上，GPU 花在计算单元上。CPU 怎样缩短一条线程的等待，见 [CPU 架构](cpu-architecture.md)。

| | 延迟优先（CPU） | 吞吐优先（GPU） |
| --- | --- | --- |
| 晶体管主要花在 | 缓存与控制逻辑 | 计算单元和大寄存器堆 |
| 一条线程要等内存时 | 尽量让它少等，或在同一线程里找别的事做 | 换一个已经就绪的 warp 继续发射 |
| 擅长的工作 | 分支多、前后依赖强、要求低延迟 | 同一操作作用在大量数据上，例如矩阵乘 |

这是设计倾向，不是硬分界：CPU 也有 SIMD 向量单元，GPU 线程也能跑分支，只是代价不同。

## SM：一块 GPU 由一百多个处理器组成

在 CUDA 的硬件模型里，GPU 是一组 Streaming Multiprocessor（SM），几个 SM 组成一个 GPC；每个 SM 有自己的寄存器堆、一块可在 L1 和 shared memory 之间划分的片上存储，以及计算单元（[编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)）。H100 SXM5 有 132 个 SM（[Hopper 架构博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)），每个 SM 里有：

| 部件 | H100 SXM5 每个 SM | 负责什么 |
| --- | --- | --- |
| FP32 CUDA core | 128 个 | 普通浮点运算，例如逐元素加法、激活函数 |
| Tensor Core（第 4 代） | 4 个 | 小块矩阵乘加，大部分 FLOPs 在这里 |
| 寄存器堆 | 256 KB，即 65,536 个 32 位寄存器 | 每个线程的私有变量，最快的一层 |
| L1 / shared memory | 合计 256 KB，shared 最多 228 KB | 同一 block 的线程共用、由程序员管理的暂存区 |
| Warp scheduler | 4 个处理块，各一个 | 每个周期挑一个就绪的 warp 发射指令 |
| 驻留上限 | 64 个 warp、2,048 个线程、32 个 block | 决定能同时挂多少工作 |

“4 个处理块、每块一个 warp scheduler”的结构是 [Volta 白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)描述的；[Ampere 白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)说明 A100 每个分区一个 Tensor Core，H100 每个 SM 也是 4 个。驻留上限来自[计算能力表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)，H100 是 compute capability 9.0。

软件这一侧：一次 kernel launch 产生一个 grid，grid 由 thread block 组成。**一个 block 的所有线程都在同一个 SM 上跑**，所以能用 shared memory 交换数据、互相同步；不同 block 之间不保证顺序。Compute capability 9.0 起还有 thread block cluster：同一 cluster 的 block 被调度到同一个 GPC，可以读写彼此的 shared memory。线程层级的细节见 [CUDA](cuda.md#thread-hierarchy)。

## Warp 与 SIMT：32 个线程共用一条指令

SM 按 32 个线程一组来调度，这一组叫 warp。[CUDA 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)把这种方式叫 SIMT：一个 warp 的线程执行同一段代码，但每个线程可以走不同分支。分支不一致时，不走当前路径的线程被 mask 掉，等另一条路径执行时再轮到它们，这叫 warp divergence。

手算一次代价。一个 warp 遇到 `if`：条件为真走路径 A（10 条指令），否则走路径 B（20 条指令）。指令数是教学假设。

| 情况 | warp 实际占用的指令槽 | 有效的“线程 × 指令” | 利用率 |
| --- | ---: | ---: | ---: |
| 32 个线程都走 A | 10 | 32 × 10 = 320 | 100% |
| 32 个线程都走 B | 20 | 32 × 20 = 640 | 100% |
| 16 个走 A，16 个走 B | 10 + 20 = 30 | 16 × 10 + 16 × 20 = 480 | 480 / (32 × 30) = 50% |

**分支本身不贵，同一个 warp 里分道扬镳才贵**；不同 warp 走不同分支没有这个代价。

两点容易误会。第一，SIMT 不等于“32 个线程永远步调一致”。从 compute capability 7.0（Volta）起有 independent thread scheduling，每个线程有自己的程序计数器，warp 内可以更细地分叉、汇合；依赖隐式同步的老代码要用 `__syncwarp()` 显式同步（[高级 kernel 编程](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#independent-thread-scheduling)）。第二，SIMT 和 SIMD 的差别在接口：SIMD 把向量宽度暴露给软件，SIMT 让你按单个线程写代码，由硬件把线程编成 warp。

## 延迟隐藏：靠同时挂着很多 warp

CUDA 指南说明，每个 warp 的执行上下文（程序计数器、寄存器）在整个生命周期里都留在片上，所以 warp 之间切换没有开销；每个发射周期，warp scheduler 从就绪的 warp 里挑一个（[硬件多线程](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading)）。一个 warp 在等显存，就换另一个。

用一个简化模型算要挂多少 warp。设每个 warp 反复做两件事：发射 $W$ 条互不依赖的指令（占 $W$ 个发射周期），然后等一次显存读取，花 $L$ 个周期。一个 scheduler 上驻留 $N$ 个 warp 时，发射槽的利用率约为：

$$
U \approx \min\left(1,\ \frac{N\,W}{W+L}\right).
$$

取 $L=300$、$W=30$（教学假设，不是 H100 的实测延迟）：一个 warp 每 330 个周期只用 30 个发射周期，要 $N\ge 330/30=11$ 个 warp 才能填满。

<!-- widget:step-latency-hiding -->

能挂多少 warp，常常由寄存器决定。H100 每个 SM 有 65,536 个 32 位寄存器，最多驻留 64 个 warp；按 4 个 scheduler 平均分，每个最多 16 个。

| 每线程寄存器 | 每个 warp 占用 | 每 SM 可驻留 warp | 每个 scheduler | 本例 $U$ |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1,024 | 64（到上限） | 16 | 100% |
| 64 | 2,048 | 32 | 8 | 8 × 30 / 330 ≈ 73% |
| 128 | 4,096 | 16 | 4 | 4 × 30 / 330 ≈ 36% |

这里忽略了寄存器的分配粒度和 shared memory 的限制；完整的 occupancy 计算见 [CUDA](cuda.md#occupancy)。

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 把 $W$ 从 30 提到 60（每个线程多算几个独立元素） | 同样的 $L$，6 个 warp 就能填满 | occupancy 越低越好 |
| 每线程寄存器从 64 加到 128 | 驻留 warp 减半，本例 $U$ 从约 73% 掉到约 36% | 寄存器用得多的 kernel 一定慢 |
| 改用异步拷贝（Ampere 起的硬件异步拷贝、Hopper 的 TMA） | 搬运和计算重叠，较少的 warp 也能保持忙碌 | 异步拷贝能突破带宽上限 |

最后一行要单独说。高性能矩阵乘 kernel 常常每个线程用很多寄存器、occupancy 不高，靠指令级并行和异步搬运把等待藏起来；CUDA 指南也把异步拷贝描述为在 warp 切换之外，进一步让搬运和计算重叠（[异步执行](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#asynchronous-execution-features)）。**延迟可以藏，带宽上限藏不掉**，这是下一节的话题。

## 内存层级：越近越小，越远越慢

H100 SXM5 各层的容量与官方带宽（白皮书 V1.04、[Hopper 博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)、[H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)、计算能力表）：

| 层级 | 容量（H100 SXM5） | 谁能访问 | 带宽（官方数字） |
| --- | --- | --- | --- |
| 寄存器 | 每 SM 256 KB，132 个 SM 共 33,792 KB | 单个线程 | 未公布 |
| Shared memory / L1 | 每 SM 合计 256 KB，shared 最多 228 KB | 同一 block；cluster 内可跨 SM | 32 个 bank，每个 bank 每周期 32 bit |
| L2 cache | 50 MB | 全部 SM | H100 未公布 |
| HBM3 显存 | 80 GB | 全部 SM；kernel 之间也靠它交换数据 | 3,352 GB/s |
| NVLink（第 4 代） | — | 其他 GPU | 每 GPU 合计 900 GB/s |
| PCIe Gen5 | — | CPU 与主机内存 | 128 GB/s |

注意出处之间的差异：2022 年的 Hopper 架构博客把 HBM3 带宽写成 3,000 GB/s，标的是初步规格；白皮书 V1.04 给出 3,352 GB/s，与产品页的 3.35 TB/s 一致。V1.04 规格表的显存带宽一行，行名后还留着“Not Finalized for H100”的旧注，但封面写明已含最终的显存频率，本文按 3,352 GB/s 用。

片上带宽 NVIDIA 没给 H100 的绝对值，但 A100 的公开资料够算一次数量级。A100 白皮书给出 L2 读带宽每周期 5,120 字节、boost 频率 1,410 MHz；[CUDA 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-bank-conflicts)给出 shared memory 有 32 个 bank、每个 bank 每周期 32 bit，也就是每个 SM 每周期 128 字节。A100 有 108 个 SM：

$$
\begin{aligned}
\text{shared memory 合计} &\approx 108\times128\ \text{B}\times1.41\ \text{GHz}\approx19.5\ \text{TB/s},\\
\text{L2} &\approx 5120\ \text{B}\times1.41\ \text{GHz}\approx7.2\ \text{TB/s},\\
\text{HBM2e（A100 80GB SXM）} &= 2.039\ \text{TB/s}.
\end{aligned}
$$

前两行是由规格换算的理论峰值，不是实测；bank conflict 和 cache 未命中都会让实际值更低。HBM 数字来自 [A100 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)。比例已经说明问题：越靠近计算单元，带宽越高、容量越小。所以高性能 kernel 的基本套路是 tiling：从 HBM 读一块进 shared memory 或寄存器，在里面反复用很多次，再写回。怎样写见 [CUDA 的 tiling 一节](cuda.md#tiling)；[FlashAttention](../00-foundations/deep-dives/attention-kernels.md) 是同一思路用在 attention 上。

Blackwell 又多了一层。第 5 代 Tensor Core 有专用的 Tensor Memory，[PTX ISA §9.7.18.1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 描述它在 sm_100a 上是每个 CTA 128 行 × 512 列的 32 位单元，即 256 KB；矩阵乘的累加结果 $D$ 放在这里，不再占寄存器。

## Tensor Core：一条指令算一小块矩阵乘加

CUDA core 的一条 FMA 指令做一次乘加。Tensor Core 一次做一小块：$D=A\times B+C$。在 Volta 白皮书里，每个 Tensor Core 处理 4×4 矩阵，每个时钟做 64 次 FMA。

### 手算一个 4×4 乘加

取下面的小矩阵（教学数字），$C$ 是全 1 矩阵：

$$
A=\begin{bmatrix}1&2&0&1\\0&1&1&0\\2&0&1&0\\1&1&1&1\end{bmatrix},\qquad
B=\begin{bmatrix}1&0&2&0\\0&1&0&1\\1&1&0&0\\0&2&1&1\end{bmatrix}.
$$

$D$ 的第 1 行第 2 列，是 $A$ 第 1 行和 $B$ 第 2 列的点积，再加上 $C$ 的对应元素：$d_{12}=1\cdot0+2\cdot1+0\cdot1+1\cdot2+1=5$。16 个元素都这样算，得到：

$$
D=AB+C=\begin{bmatrix}2&5&4&4\\2&3&1&2\\4&2&5&1\\3&5&4&3\end{bmatrix}.
$$

每个输出元素 4 次乘加，即 4 次 FMA；16 个元素共 $16\times4=64$ 次 FMA，正好是 Volta 一个 Tensor Core 一个时钟的量。一次 FMA 按 2 个 FLOP 计，这一小块是 128 FLOP。

精度上，Volta 的 $A$、$B$ 是 FP16，乘积保留全精度，再用 FP32 累加；$C$、$D$ 可以是 FP16 或 FP32。“输入低精度、累加高精度”一直延续到后来的 FP8 和 FP4。

### 从一小块到整张卡的峰值

峰值 = Tensor Core 个数 × 每个每周期的 FMA × 2 × 频率。拿 NVIDIA 公布的数字核对：

| GPU | Tensor Core 数 | 每个每周期 dense FMA | 频率 | 算出来 | 官方数字 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V100 SXM2 | 640 | 64（FP16） | 1,530 MHz | 125.3 TFLOPS | 125 TFLOPS |
| A100 SXM | 432 | 256（FP16/BF16） | 1,410 MHz | 311.9 TFLOPS | 312 TFLOPS |
| H100 SXM | 528 | 512（FP16/BF16） | 1,830 MHz | 989.4 TFLOPS | 989.4 TFLOPS |

V100 一行来自 Volta 白皮书，A100 一行来自 [Ampere 架构博客](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。H100 一行的频率和官方数字来自白皮书 V1.04：Tensor 运算的 boost 频率 1,830 MHz，dense FP16/BF16（FP32 累加）989.4 TFLOPS，稀疏 1,978.9 TFLOPS。每个 Tensor Core 每周期 512 次 FMA 是本文推算的：白皮书说每个 SM 同频下的 MMA 速率是 A100 的 2 倍；代入 $528\times512\times2\times1.83\ \text{GHz}$，正好得到 989.4 TFLOPS。

数据表上的“稀疏”数字来自 Ampere 引入的 2:4 结构化稀疏：每 4 个值里最多 2 个非零时，Tensor Core 吞吐翻倍。普通 dense 模型拿不到这个倍数，所以比较前要先换算成 dense。H100 产品页的 Tensor 行全都注明了“with sparsity”。

### 指令越来越大，发起的线程越来越少

程序里不会直接写 4×4。在 PTX 层面能看到每一代接口怎样变（[PTX ISA §9.7.16–9.7.18](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)）：

| 指令 | 由谁发起 | 一条的形状（FP16/BF16） | 一条的 FLOP | 需要 |
| --- | --- | --- | ---: | --- |
| `mma.sync.m16n8k16` | 一个 warp（32 个线程） | A 16×16，B 16×8 | 2 × 16 × 8 × 16 = 4,096 | sm_80 起 |
| `wgmma.mma_async.m64n256k16` | 一个 warpgroup（4 个连续 warp） | A 64×16，B 16×256 | 2 × 64 × 256 × 16 = 524,288 | sm_90a |
| `tcgen05.mma` | CTA 里的单个线程 | 累加器 $D$ 在 Tensor Memory | 随形状而定 | sm_100a 等 |

一条 `mma.sync` 相当于 32 个上面的 4×4×4 小块（4,096 / 128）。趋势是：一条指令做的活越来越多，越来越异步，发起它的线程越来越少。喂 Tensor Core 占用的线程和寄存器少了，其余 warp 就能去搬数据或算 softmax；[FlashAttention-3](https://arxiv.org/abs/2407.08608) 在 Hopper 上就用 warp specialization 让搬运和计算重叠，并把分块矩阵乘和 softmax 交错执行（另见 [attention kernel](../00-foundations/deep-dives/attention-kernels.md)）。实际写法交给 [CUDA](cuda.md) 和 CUTLASS 这类库。

<!-- paper-figure:fa3-pingpong -->

## 数据类型：每一代 Tensor Core 加了什么

| 格式 | 符号 / 指数 / 尾数位 | 首次有 Tensor Core 支持 | 备注 |
| --- | --- | --- | --- |
| FP16 | 1 / 5 / 10 | Volta（V100） | 可用 FP32 累加 |
| INT8、INT4 | 整数 | 整数 `wmma` 从 sm_72 起，INT4 从 sm_75 起 | 主要用于推理量化 |
| TF32 | 1 / 8 / 10 | Ampere（A100） | 读入 FP32 数据，按 FP32 的范围、较低的内部精度计算，输出 FP32 |
| BF16 | 1 / 8 / 7 | Ampere（A100） | 范围与 FP32 相同 |
| FP8（E4M3、E5M2） | 1 / 4 / 3；1 / 5 / 2 | Hopper（H100）；Ada（compute capability 8.9）也支持 | E4M3 精度高，E5M2 范围大 |
| FP6（E3M2、E2M3）、FP4（E2M1） | 1 / 3 / 2，1 / 2 / 3；1 / 2 / 1 | Blackwell（B200） | 要和块缩放因子一起用 |

出处：FP16 见 Volta 白皮书；TF32、BF16 见 Ampere 博客；FP8 的两种格式见 Hopper 博客；各 compute capability 支持哪些输入类型见计算能力表，FP8 `mma` 需要 sm_89、FP6/FP4 格式名见 PTX ISA。FP16 与 BF16 的取舍见[精度与显存](../00-foundations/deep-dives/precision-and-memory.md)，这里不重复。

格式越窄，越依赖缩放。FP4 的 E2M1 只能表示 0、0.5、1、1.5、2、3、4、6 及其负数。[NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) 每 16 个值共用一个 FP8（E4M3）缩放因子，整个张量再乘一个 FP32 缩放因子，平均每个值 4.5 bit；对照的 MXFP4 是每 32 个值共用一个只能取 2 的幂的 E8M0 缩放因子。

手算一个简化版（只看 4 个值，忽略张量级缩放和缩放因子本身的舍入）。块是 $[0.3,-1.2,2.4,3.0]$，绝对值最大 3.0，缩放因子取 $3.0/6=0.5$。除以 0.5 得 $[0.6,-2.4,4.8,6.0]$，就近取可表示值 $[0.5,-2,4,6]$，乘回去是 $[0.25,-1.0,2.0,3.0]$。如果同一块混进一个 30，缩放因子变成 5，0.3 和 −1.2 除以 5 后都舍成 0：一个离群值抹掉了整块的小数。块越小，一个离群值连累的值越少，这是 NVFP4 选 16 而不是 32 的直觉。

## NVLink 与 NVSwitch：卡与卡之间

H100 SXM5 有 18 条第 4 代 NVLink，每条每方向 25 GB/s，合计 900 GB/s；PCIe Gen5 是 128 GB/s（Hopper 博客、H100 产品页）。NVLink 是点对点链路；NVSwitch 是交换芯片，让一组 GPU 两两之间都能用满 NVLink 带宽。能这样直接互访的一组 GPU，叫一个 NVLink 域。

| 互联 | 带宽 | 连接范围 |
| --- | ---: | --- |
| PCIe Gen5（H100） | 128 GB/s | GPU 到 CPU |
| NVLink 4（H100） | 每 GPU 900 GB/s | 一台 8 卡 HGX 机器内 |
| NVLink 5（GB200 NVL72） | 每 GPU 1.8 TB/s，72 卡合计 130 TB/s | 一个机架 |
| 网卡（HGX B200） | 8 卡合计 0.8 TB/s，约每卡 100 GB/s | 机器之间 |

后两行出自 [GB200 NVL72 页面](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)和 [HGX 页面](https://www.nvidia.com/en-us/data-center/hgx/)。两个数都是两个方向相加：HGX B200 配 8 张 400 Gb/s 网卡（[DGX B200 页面](https://www.nvidia.com/en-us/data-center/dgx-b200/)列的是 8 张单口 ConnectX-7），每个方向合计 400 GB/s，两个方向 0.8 TB/s。同口径比，同一代 Blackwell GPU 的 NVLink 带宽约是机器外网络的 18 倍（每卡 1.8 TB/s 对约 0.1 TB/s）。所以通信最密的 tensor parallel 一般放在 NVLink 域里，跨机器只放通信少一些的切分，见[张量并行](../06-systems/tensor-parallel.md)和[多卡训练](../06-systems/distributed-training.md)。NVL72 把 NVLink 域从一台 8 卡机器扩到了一整个机架。

## 这一篇没讲的

- 怎么写 kernel、怎么用 Nsight 看 occupancy 和带宽：见 [CUDA](cuda.md)。
- Roofline 只提一个数：H100 SXM 的 dense BF16 是 989.4 TFLOPS，HBM 3,352 GB/s，相除约 295 FLOP/字节。一个 kernel 从显存每读 1 字节做不到这么多运算，就先被带宽卡住。判断方法见[性能](performance.md)。
- 各代参数、70B 模型要几张卡的算例：见 [NVIDIA 各代 GPU](nvidia-generations.md)。制程、HBM 和封装：见[芯片怎么造出来](chips-and-fabs.md)。
