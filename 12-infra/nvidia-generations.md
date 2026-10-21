# NVIDIA 各代 GPU：每一代改了什么、为什么

**中文** · [English](nvidia-generations.en.md)

> 阅读时间：约 13 分钟 · 最近审阅：2026-10

读到“新一代比上一代快 N 倍”，先问三件事：哪种精度？dense 还是 sparse？哪个 SKU、按每张 GPU 还是按整机算？**三件事对齐以后再看，过去几代一直在涨的是低精度 Tensor 吞吐、HBM 容量和 NVLink 域的大小；显存带宽也在涨，但比算力慢。每读 1 字节要配多少次运算才能用满算力，从 V100 的约 140 涨到 GB300 用 NVFP4 时的约 1,900，所以大模型 decode 这类读得多、算得少的工作，代际提速主要跟着带宽走。**

SM、warp、Tensor Core 这些概念见 [GPU 架构](gpu-architecture.md)。本文所有硬件数字都来自 NVIDIA 的白皮书、数据表、产品页和技术博客，核对日期 2026-10-15；已经出货和仅宣布的产品分开标。

## 读数据表前，先对齐这几件事

| 要对齐什么 | 例子 | 本文怎么处理 |
| --- | --- | --- |
| Dense 还是 sparse | [H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)的 BF16 写 1,979 TFLOPS，脚注是 with sparsity；dense 是一半。但 [GB300 NVL72 页面](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)的 FP4 是 1,440（sparse）与 1,080（dense），不是 2 倍 | 只比 dense。页面只给 sparse 且脚注写明“dense 是一半”时，才除以 2 |
| 哪个 SKU、什么形态 | 同叫 H100：SXM 版 80 GB、3.35 TB/s；NVL（PCIe）版 94 GB、3.9 TB/s，Tensor 吞吐更低 | 统一用 SXM 或机架版，并写明 |
| 按什么计数 | [GB200 superchip](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) 是 1 颗 Grace CPU 加 2 颗 GPU；[HGX 页面](https://www.nvidia.com/en-us/data-center/hgx/)给的是 8 卡合计 | 一律折算成每 GPU |
| 初步规格还是最终规格 | 2022 年 [Hopper 博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)写 dense BF16 1,000 TFLOPS、显存 3,000 GB/s，并标为初步；[H100 架构白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)给出最终值 989.4 TFLOPS（dense）、3,352 GB/s，与产品页的 1,979（sparse）、3.35 TB/s 一致 | 用最终规格：白皮书 V1.04 或当前产品页 |

官方页面之间也会对不上。Blackwell 每张卡的显存，[Blackwell Ultra 技术博客](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)的对照表写最高 192 GB，GB200 页面折算是 186 GB，[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) 的 1,440 GB 除以 8 是 180 GB。Rubin 的显存带宽、NVLink 和吞吐，不同页面给出的数字也不一样（见下文 Rubin 一节）。遇到这种情况，本文把几种数字都列出来，并写明出处。

## 对齐口径以后：放进一张表

所有吞吐都是 dense、每 GPU。B 系列的数字由 HGX 或 NVL72 页面的整机数字折算。

| 产品（形态） | 工艺 · 晶体管 | SM | Tensor Core（新增精度） | 显存 · 带宽 | NVLink（每 GPU） | 功耗上限 | BF16 | FP8 | FP4 |
| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| V100 SXM2（Volta） | TSMC 12 nm FFN · 211 亿 | 80 | 第 1 代（FP16） | 32 GB HBM2 · 900 GB/s | NVLink 2：300 GB/s | 300 W | 125 TFLOPS（FP16） | — | — |
| A100 80GB SXM（Ampere） | TSMC N7 · 542 亿 | 108 | 第 3 代（TF32、BF16、2:4 稀疏） | 80 GB HBM2e · 2,039 GB/s | NVLink 3：600 GB/s | 400 W | 312 TFLOPS | — | — |
| H100 SXM（Hopper） | TSMC 4N · 800 亿 | 132 | 第 4 代（FP8） | 80 GB HBM3 · 3,352 GB/s | NVLink 4：900 GB/s | 700 W | 989.4 TFLOPS | 1,978.9 TFLOPS | — |
| H200 SXM（Hopper） | TSMC 4N · 800 亿 | — | 第 4 代 | 141 GB HBM3e · 4.8 TB/s | NVLink 4：900 GB/s | 700 W | 989.4 TFLOPS | 1,978.9 TFLOPS | — |
| B200（HGX B200） | TSMC 4NP · 2,080 亿（2 颗 die） | — | 第 5 代（FP6、FP4） | 180 GB HBM3e · 8 TB/s | NVLink 5：1.8 TB/s | 最高 1,200 W* | 2.25 PFLOPS | 4.5 PFLOPS | 9 PFLOPS |
| GB200（NVL72 中每颗 GPU） | 同上 | — | 第 5 代 | 186 GB HBM3e · 8 TB/s | NVLink 5：1.8 TB/s | 最高 1,200 W* | 2.5 PFLOPS | 5 PFLOPS | 10 PFLOPS |
| B300（HGX B300，Blackwell Ultra） | TSMC 4NP · 2,080 亿（2 颗 die） | 最多 160 | 第 5 代（NVFP4 提速） | 约 262 GB HBM3e · 8 TB/s | NVLink 5：1.8 TB/s | 最高 1,400 W* | 2.25 PFLOPS | 4.5 PFLOPS | 13.5 PFLOPS |
| GB300（NVL72 中每颗 GPU） | 同上 | 最多 160 | 同上 | 约 278 GB HBM3e · 8 TB/s | NVLink 5：1.8 TB/s | 最高 1,400 W* | 2.5 PFLOPS | 5 PFLOPS | 15 PFLOPS |
| Rubin（已投产，见下文） | 工艺未公布 · 3,360 亿（2 颗 die） | 224 | 第 5 代（平台博客的说法）；第 3 代 Transformer Engine | 288 GB HBM4 · 22 或 19.2 TB/s | NVLink 6：3.6 或 3 TB/s | 未公布 | 4 PFLOPS | 17.5 PFLOPS | 35 PFLOPS（NVFP4 训练） |

出处，按行：V100 来自 [Volta 白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)和 [V100 数据表](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf)；A100 来自 [Ampere 博客](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)和 [A100 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)；H100 来自白皮书 V1.04 和产品页；H200 来自 [H200 产品页](https://www.nvidia.com/en-us/data-center/h200/)，它的 Tensor 各行与 H100 SXM 产品页完全相同，所以沿用 989.4 / 1,978.9；B200 的显存与带宽由 DGX B200 的 8 卡合计 1,440 GB、64 TB/s 折算，吞吐由 HGX 页面折算；GB200 由 NVL72 页面的 superchip 数字除以 2；B300 的显存由 [DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/) 的 2.1 TB 除以 8，GB300 由 NVL72 的 20 TB 除以 72；H200 和 B 系列的工艺、晶体管，以及 B300 的带宽和 SM 数（“最多 160，随 SKU 不同”），来自 Blackwell Ultra 博客的代际对照表；Rubin 见下文。

\* 功耗一栏的 Blackwell 与 Blackwell Ultra 数字来自 Blackwell Ultra 博客的代际对照表（“up to”，TGP），没说对应哪个 SKU；HGX、DGX 页面只给整机功耗（DGX B200 约 14.3 kW）。另外，按 GB200 新闻稿，GB200 里是两颗 B200 GPU，但 HGX B200 与 GB200 的每卡 BF16 是 2.25 对 2.5 PFLOPS：同一款 GPU 在不同产品里规格不同。

## 每一代改了什么

### Volta（V100，2017）：先有 Tensor Core

Volta 白皮书（2017 年 8 月版）介绍了第一代 Tensor Core：每个 Tensor Core 每个时钟做一次 4×4×4 的 FP16 乘加、用 FP32 累加，V100 共 640 个。同一张卡上，普通 FP32 峰值 15.7 TFLOPS，Tensor 峰值 125 TFLOPS。白皮书的理由是：矩阵乘是训练和推理的核心运算，专用单元能以很高的能效大幅提高吞吐。同一代还有第二代 NVLink（6 条链路共 300 GB/s）和 independent thread scheduling。

### Ampere（A100，2020）：让 FP32 代码也能用 Tensor Core

A100 在 2020 年 5 月发布（Ampere 博客）。第 3 代 Tensor Core 加了三样东西。TF32 读入 FP32 数据，以 FP32 的范围、较低的内部精度计算，原本用 FP32 写的训练代码不改格式也能走 Tensor Core；BF16 与 FP16 同速，但指数位和 FP32 一样多，训练时不容易溢出（取舍见[精度与显存](../00-foundations/deep-dives/precision-and-memory.md)）；2:4 结构化稀疏让满足条件的矩阵吞吐翻倍，从这一代起，数据表的 Tensor 数字开始分 dense 和 sparse 两栏。另外，L2 增大到 40 MB，NVLink 升到第 3 代（600 GB/s），MIG 可以把一张卡切成最多 7 个隔离实例。

### Hopper（H100 2022，H200 2023）：FP8 与 Transformer Engine

H100 在 2022 年 3 月发布（Hopper 博客）。第 4 代 Tensor Core 每个 SM 的 16 位 MMA 速率是 A100 的 2 倍，FP8 是 4 倍。FP8 有 E4M3 和 E5M2 两种；只有 8 位，范围比 16 位格式窄得多，所以同时推出 Transformer Engine：按层统计输出数值，决定用 FP8 还是 16 位，并动态缩放到可表示的范围。这一代还加了 TMA、thread block cluster 和 distributed shared memory，方向都是让数据搬运异步化（见 [GPU 架构](gpu-architecture.md)）。显存换成 HBM3（白皮书 V1.04 的最终值是 3,352 GB/s），NVLink 升到第 4 代（900 GB/s），功耗上限到 700 W。

H200 在 2023 年 11 月 13 日的 SC23 上宣布，NVIDIA 称它是第一款用 HBM3e 的 GPU（[新闻稿](https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform)）。产品页上 H200 SXM 的 Tensor 各行与 H100 SXM 完全相同，变的只有显存：141 GB、4.8 TB/s。**算力不变，只加显存容量和带宽**；下面的算例会看到，对 decode 来说这样的升级为什么有用。

### Blackwell（B200 / GB200，2024）：两颗 die、FP4 和 NVL72

Blackwell 在 2024 年 3 月 18 日的 GTC 上发布（[新闻稿](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing)）。[架构页](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)写明：2,080 亿晶体管，TSMC 4NP，两颗“受光罩尺寸限制”的 die 用 10 TB/s 的片间互联连成一个 GPU。看前三代的 die 面积就能理解为什么：V100 815 mm²、A100 826 mm²、H100 814 mm²（各代白皮书与 Hopper 博客），单颗 die 已经做不大了，晶体管再多，就只能在一个封装里放两颗。制程与封装见[芯片怎么造出来](chips-and-fabs.md)。

第二代 Transformer Engine 加入 FP4 和 micro-tensor scaling，即更细粒度的缩放。2025 年 6 月 NVIDIA 介绍的 NVFP4 是这类按小块缩放的格式：每 16 个值一个 FP8 缩放因子（[NVFP4 博客](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)，手算例子见 [GPU 架构](gpu-architecture.md)）。第五代 NVLink 每 GPU 1.8 TB/s，最多可连 576 颗 GPU。GB200 NVL72 把 36 颗 Grace CPU 和 72 颗 Blackwell GPU 放进一个液冷机架，组成一个 NVLink 域，合计 130 TB/s。

### Blackwell Ultra（B300 / GB300，2025）：提升集中在 NVFP4 和 attention

Blackwell Ultra 在 2025 年 3 月 18 日的 GTC 上发布，合作伙伴产品预计 2025 年下半年起供货（[新闻稿](https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning)）。截至 2026 年 10 月，HGX 页面脚注写着 HGX B300 与 B200 都在出货，GB300 NVL72 页面标“Available Now”。

晶体管、工艺和 die 数与 Blackwell 相同，变化在分配上（Blackwell Ultra 博客）：NVFP4 dense 从 10 提到 15 PFLOPS；attention 里 softmax 要用的指数运算吞吐翻倍；显存最高 288 GB；功耗上限 1,400 W。代价也看得见。HGX 页面上，8 卡 HGX B300 与 HGX B200 的 BF16、FP8 一样（sparse 36 与 72 PFLOPS），INT8 却从 72 POPS 降到 3 POPS（同为 sparse 口径），FP64 从 296 TFLOPS 降到 10 TFLOPS；[计算能力表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)里，B300 的 compute capability 10.3 也没有 FP64 Tensor Core 输入类型。**这一代没有提高 BF16 和 FP8，提升集中在 NVFP4、显存容量和 softmax 的指数运算上，同时砍掉了大部分 FP64 和 INT8 吞吐。**

显存数字有三种写法：博客写最高 288 GB；DGX、HGX B300 页面写 8 卡共 2.1 TB，约每卡 262 GB；GB300 NVL72 页面写 72 卡共 20 TB，约每卡 278 GB。页面没有解释差异。

### Rubin / Vera Rubin（已投产）：HBM4 与 NVLink 6

时间线：

- 2025 年 3 月 GTC（[NVIDIA 博客汇总](https://blogs.nvidia.com/blog/nvidia-keynote-at-gtc-2025-ai-news-live-updates/)）：Vera Rubin NVL144 定在 2026 年下半年，Rubin Ultra 系统在 2027 年下半年。
- 2026 年 1 月 5 日 CES（[新闻稿](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)）：NVIDIA 在其中的 Rubin Readiness 一节写道“NVIDIA Rubin is in full production …”（Rubin 已全面投产），合作伙伴产品下半年供货；机架改称 Vera Rubin NVL72（72 颗 Rubin GPU、36 颗 Vera CPU）。3 月 16 日 GTC 的[新闻稿](https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform)加入第七颗芯片 Groq 3 LPU，[POD 博客](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/)介绍了可扩到 576 颗 GPU 的 Rubin Ultra 和每柜 144 颗的 Kyber 机架，没给日期。
- 2026 年 5 月 31 日 GTC Taipei（[新闻稿](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory)）：量产出货“今年秋天”开始；8 月 27 日更新的 [NVIDIA 博客](https://blogs.nvidia.com/blog/vera-cpu-delivery/)说 AWS 收到了第一台 Vera CPU 服务器和 Vera Rubin GPU。
- 2026 年 8 月 26 日（[第二财季财报新闻稿](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2027)）：黄仁勋说 Vera Rubin“now in full production”（已全面投产），机架已在 CoreWeave、Google Cloud、Microsoft Azure、Oracle Cloud Infrastructure 和 Nebius 等合作伙伴处运行。9 月 30 日的 [NVIDIA 博客](https://blogs.nvidia.com/blog/coreweave-agentic-ai-vera-rubin/)说 CoreWeave 当月收到了第一批 Vera Rubin NVL72 量产机架。

截至 2026 年 10 月，[Vera Rubin NVL72 产品页](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)标着“Available Now”，并写道服务器厂商正在“shipping Vera Rubin-based systems”，所以本文把 Rubin 标为“已投产”。云上的全面开放还在铺开：10 月 21 日的 [NVIDIA 博客](https://blogs.nvidia.com/blog/google-cloud-vera-rubin-infrastructure/)说，Google Cloud 基于 Vera Rubin NVL72 的 A5X 实例下个月才全面开放。下面的数字都是 NVIDIA 公布的规格。

[Rubin GPU 架构博客](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)（2026 年 7 月 21 日）给出 3,360 亿晶体管、两颗计算 die、224 个 SM、最高 288 GB HBM4，没写工艺；1 月的[平台博客](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)说 SM 里是第五代 Tensor Core。表中吞吐取自 [HGX 页面](https://www.nvidia.com/en-us/data-center/hgx/)的单卡表和 NVL72 页面：NVFP4 训练 35、FP8 17.5、BF16 4 PFLOPS，都标 dense。常被引用的“NVFP4 推理 50 PFLOPS”，这两页标 sparse；平台博客则说它来自第三代 Transformer Engine 的“自适应压缩”，没写 dense 或 sparse。

官方页面之间的数字对不上，至少有三组（2026-10-15 读取）：

| 出处 | 每卡显存 · 带宽 | 每卡 NVLink | NVFP4 训练（dense，每卡） |
| --- | --- | --- | ---: |
| CES 新闻稿（2026-01） | 未给 | 3.6 TB/s，NVL72 合计 260 TB/s | 未给 |
| 平台博客（2026-01）、GPU 博客（2026-07）、HGX 页面的单卡表 | 288 GB · 22 TB/s | 3.6 TB/s | 35 PFLOPS |
| Vera Rubin NVL72 产品页、[Vera Rubin 数据表](https://dam-cdn.nvd.orangelogic.com/AssetLink/56p68o47y6f1yucpubl0ump1c0aif422.pdf)（2026-10） | 288 GB · 19.2 TB/s（整柜 20.7 TB、1,400 TB/s） | 3 TB/s，整柜 216 TB/s | 35 PFLOPS |
| 同一 HGX 页面与数据表里的 HGX Rubin NVL8 整机，除以 8 | 2 TB、130 TB/s，约 250 GB · 16 TB/s | 交换带宽 24 TB/s，约 3 TB/s | 270 PFLOPS，约 33.75 |

最新的数据表与 NVL72 产品页一致，但 HGX 页面自己的单卡表和整机表也对不上，页面都没有解释。本文后面的表用 22 和 19.2 两个数给出区间；正式供货后，应以对应 SKU 的数据表为准。

## 设计逻辑：几代下来，什么一直在变

把每代最低精度的 dense 峰值除以显存带宽，就得到“每读 1 字节要配多少次运算，才能用满算力”。这个比值在 [性能](performance.md)里叫屋脊点：

| 产品 | 最低精度的 dense 峰值 | 显存带宽 | FLOP / 字节 |
| --- | ---: | ---: | ---: |
| V100 SXM2 | FP16 125 TFLOPS | 0.9 TB/s | 139 |
| A100 80GB SXM | BF16 312 TFLOPS | 2.039 TB/s | 153 |
| H100 SXM | FP8 1,978.9 TFLOPS | 3.352 TB/s | 590 |
| H200 SXM | FP8 1,978.9 TFLOPS | 4.8 TB/s | 412 |
| GB200（每 GPU） | NVFP4 10 PFLOPS | 8 TB/s | 1,250 |
| GB300（每 GPU） | NVFP4 15 PFLOPS | 8 TB/s | 1,875 |
| Rubin（已投产） | NVFP4 训练 35 PFLOPS | 22 或 19.2 TB/s | 1,591 或 1,823 |

从表里能读出几条设计逻辑：

1. **低精度是最大的杠杆。** 从 V100 的 FP16 到 GB300 的 NVFP4，dense 峰值涨了 120 倍，其中同为 16 位（V100 FP16 125 对 GB300 BF16 2,500 TFLOPS）只涨了 20 倍，其余约 6 倍来自更窄的格式。窄格式要靠 Transformer Engine 的缩放保住精度：第一代按张量统计，第二代缩到微块，Rubin 又加了“自适应压缩”。
2. **带宽涨得比算力慢。** 同期显存带宽从 0.9 涨到 8 TB/s，约 9 倍，所以屋脊点越来越高，每个读进来的字节要被复用得越来越多。H200 反过来：算力与 H100 相同，带宽多 43%，屋脊点从 590 降到 412。
3. **单颗 die 做不大了，就在封装里放两颗。** 三代 die 面积都在 814–826 mm² 之间，Blackwell 和 Rubin 都改成两颗 die 加片间高速互联。
4. **NVLink 域越来越大。** 每 GPU 的 NVLink 带宽每代都涨：300、600、900、1,800 GB/s，Rubin 是 3,600 或 3,000 GB/s（见上文）；域从一台 8 卡机器扩到 72 卡机架，Kyber 机架宣布了每柜 144 颗。Tensor parallel 每层都要通信，MoE 的专家并行要做 all-to-all，都希望留在这个域里，见[张量并行](../06-systems/tensor-parallel.md)。
5. **越来越专用，也越来越耗电。** Blackwell Ultra 砍了 FP64 和 INT8；功耗上限从 300 W 涨到 1,400 W，GB200 NVL72 是液冷机架。

## 算一遍：70B 模型要几张卡，每个 token 最快多快

[导读](README.md)用 8B 模型在一张卡上算过一次 decode；这里换成 70B 参数的 dense 模型，看不同代、不同精度要几张卡。简化假设：

- 只算权重：BF16 每参数 2 字节，共 140 GB；FP8 每参数 1 字节，共 70 GB。容量按数据表上的数字直接相除，不区分 GB 和 GiB，也不扣驱动和 CUDA context 占用。
- Batch 为 1、逐个 token decode，每个 token 把全部权重从 HBM 读一遍；不算 KV cache 的读取和计算时间。
- 多卡时权重平均切开，各卡同时读，通信时间算作 0；带宽按峰值 100% 用满。

于是：

$$
n_{\min}=\left\lceil\frac{P\,b}{M}\right\rceil,\qquad
t_{\text{token}}\ \ge\ \frac{P\,b}{n\,B}.
$$

$P=70\times10^9$ 是参数量，$b$ 是每参数字节数，$M$ 是每卡显存，$B$ 是每卡带宽，$n$ 是卡数。例如两张 H100 跑 BF16：$140\ \text{GB}/(2\times3.352\ \text{TB/s})\approx20.9$ ms，即每秒最多约 48 个 token。

| 产品 | 每卡显存 · 带宽 | BF16（140 GB）：最少几张 · 每 token 下限 | FP8（70 GB）：最少几张 · 每 token 下限 |
| --- | --- | --- | --- |
| V100 SXM2 | 32 GB · 0.9 TB/s | 5 张（FP16）· 31.1 ms | 没有 FP8 Tensor Core |
| A100 80GB SXM | 80 GB · 2.039 TB/s | 2 张 · 34.3 ms | 没有 FP8 Tensor Core |
| H100 SXM | 80 GB · 3.352 TB/s | 2 张 · 20.9 ms | 1 张（只剩 10 GB）· 20.9 ms |
| H200 SXM | 141 GB · 4.8 TB/s | 1 张（几乎没有余量）· 29.2 ms；用 2 张是 14.6 ms | 1 张 · 14.6 ms |
| B200（HGX） | 180 GB · 8 TB/s | 1 张 · 17.5 ms | 1 张 · 8.8 ms |
| B300（HGX） | 约 262 GB · 8 TB/s | 1 张 · 17.5 ms | 1 张 · 8.8 ms |
| Rubin（已投产） | 288 GB · 19.2–22 TB/s | 1 张 · 6.4–7.3 ms | 1 张 · 3.2–3.6 ms |
| RTX PRO 6000 工作站版 | 96 GB · 1.792 TB/s | 2 张 · 39.1 ms | 1 张 · 39.1 ms |
| DGX Spark（GB10） | 128 GB 统一内存 · 0.273 TB/s | 放不下 | 1 台 · 256 ms |

<!-- widget:step-decode-floor -->

最后两行出自 [RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) 和 [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) 产品页。逐行看能读出几件事：

- **在 H100 上，FP8 一张卡和 BF16 两张卡的下限一样**，都是 20.9 ms：字节数减半，相当于带宽翻倍，卡却少用一张。
- **两张 A100（34.3 ms）比五张 V100（31.1 ms）还慢。** 在这个模型里，下限只取决于这组卡的总带宽：5 × 0.9 = 4.5 TB/s，比 2 × 2.039 ≈ 4.1 TB/s 多。卡多也有代价，就是被忽略的通信。
- **H200 一张卡就放得下 BF16 权重，但比两张 H100 慢**（29.2 对 20.9 ms）；它省的是卡数，不是时间。
- **B300 和 B200 在 batch 1 时一样快**，因为带宽相同；多出来的显存用来放 KV cache、组更大的 batch。
- **工作站产品放得下，不等于跑得快。** RTX PRO 6000 一张卡放得下 FP8 的 70B，带宽约为 H100 SXM5 的一半，下限 39.1 ms；DGX Spark 也放得下，但每个 token 至少 256 ms，约每秒 4 个 token。

读工作站卡的规格页另有两个坑。“AI TOPS”常不注明精度和稀疏（RTX PRO 6000 的 4,000 AI TOPS 脚注写明是 FP4 加稀疏），不能和上面的 dense 数字比；同叫 Blackwell，RTX PRO 6000 是 compute capability 12.0、DGX Spark 是 12.1（[CUDA GPU 列表](https://developer.nvidia.com/cuda-gpus)），每 SM 最多 100 KB shared memory，数据中心的 10.0 是 228 KB，按 sm_100a 这类架构专属目标编译的代码也不能在 12.x 上运行。

再核对一下算力确实不是瓶颈：每个 token 约 $2P=1.4\times10^{11}$ 次运算，H100 SXM 的 dense BF16 989.4 TFLOPS 只要 0.14 ms，比读权重的 20.9 ms 小两个数量级以上。batch 为 1 的 decode 深陷带宽限制，同时服务多个请求才能把算力用起来，见[一次请求怎样跑完](../06-systems/llm-serving.md)和 [KV cache](../00-foundations/deep-dives/kv-cache-and-inference.md)。

| 改一个条件 | 应该看到什么 | 不能据此推出什么 |
| --- | --- | --- |
| Batch 从 1 加到 32 | 权重每步只读一次，32 个请求分摊；但每步多读 32 份 KV | 每个用户的 token 一样快 |
| 换成 MoE，总参数仍是 70B，每个 token 只激活一部分 | 每步只读被选中的专家，下限按激活参数算 | 显存也跟着省：所有专家仍要放得下 |
| H100 上 tensor parallel 从 2 张加到 8 张 | 本模型的下限降为四分之一 | 实际也快 4 倍：每层都要在 NVLink 上做 all-reduce，卡越多，这部分占比越大 |
| B200 上 FP8 换 NVFP4 | 权重约 $70\times10^9\times4.5/8\approx39$ GB，下限约 4.9 ms | 模型质量不变，要看量化后的评测 |

## 接下来读

- 这些规格背后的 SM、warp、内存层级和 Tensor Core：[GPU 架构](gpu-architecture.md)。
- 屋脊点怎样用来判断一个 kernel 被算力还是带宽卡住：[性能](performance.md)。
- 多卡切分与通信：[张量并行](../06-systems/tensor-parallel.md)、[多卡训练](../06-systems/distributed-training.md)。
