# 芯片怎么造出来：晶圆、制程、封装与 HBM

**中文** · [English](chips-and-fabs.en.md)

> 阅读时间：约 10 分钟 · 最近审阅：2026-10

新闻里常说“AI 芯片缺货”。缺的到底是什么，是晶圆厂造不出 GPU 吗？问题还涉及良率、封装和存储器。裸片（die）越大，一片晶圆上切得越少，坏的比例也越高；单个 die 的面积又有曝光上限。

因此，AI 芯片越来越多地把几块 die 和 HBM 封装在一起。先进封装和 HBM 本身的产能也有限，可能成为新的瓶颈。这一篇从生产链讲起，用一张晶圆算一遍良率，再看封装和 HBM。

## 从设计图到服务器：一颗 GPU 经过哪些环节

| 环节 | 拿到什么，交出什么 | 这一步可能卡住什么 |
| --- | --- | --- |
| 设计 | 架构和电路 → 交给工厂的版图（tape-out） | 设计周期长，改一次要重新做掩模 |
| 晶圆制造 | 300 mm 硅片 → 经过光刻、沉积、刻蚀等大量工序，做出一整片 die | 先进制程的产能，缺陷密度 |
| 晶圆测试 | 整片晶圆 → 标出哪些 die 能用 | 测不出来的坏 die 会流到后面，浪费封装 |
| 切割与封装 | 好 die（加上 HBM 等）→ 一个封装好的芯片 | 先进封装产能，封装良率 |
| 最终测试 | 封装好的芯片 → 合格品，按规格分档 | 测试时间和设备 |
| 板卡与系统 | 芯片 → 模块、服务器、机柜 | 供电、散热、网络和整机组装 |

设计公司不一定自己有工厂。NVIDIA 自己的说法是，H100 用的 GH100 “fabricated using the TSMC 4N process customized for NVIDIA …”，即用为 NVIDIA 定制的 TSMC 4N 工艺制造（[Hopper 架构博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#nvidia_h100_gpu_architecture_in-depth)讲 H100 架构的一节，2022-03）。HBM 来自存储厂，例如 Micron 公开销售 HBM3E（[Micron 产品页](https://www.micron.com/products/memory/hbm/hbm3e)）。一颗 AI 芯片要凑齐好的逻辑 die、HBM 和封装三样，缺哪样都出不了货。

## 制程名字说了什么，没说什么

早年的节点名大致对应晶体管的一个物理尺寸，今天它更像一代工艺的名字。IEEE Spectrum 2020 年的文章指出，一个“7 nm”晶体管的大部分关键尺寸其实远大于 7 nm，名字和实际尺寸脱钩已有大约二十年；文章还介绍了 H.-S. Philip Wong 等人提出的替代指标 LMC：逻辑密度、内存密度和两者之间的连接密度（[IEEE Spectrum，2020-07](https://spectrum.ieee.org/a-better-way-to-measure-progress-in-semiconductors)）。

所以读制程，看厂商在跟什么比、比的是什么。TSMC 自己的说法（[TSMC 智能手机平台页](https://www.tsmc.com/english/dedicatedfoundry/technology/platform_smartphone_tech_advancedTech)、[HPC 平台页](https://www.tsmc.com/english/dedicatedFoundry/technology/platform_HPC_tech_advancedTech)，2026-10 读取）：

| 名字 | TSMC 怎么描述 | 读的时候注意 |
| --- | --- | --- |
| N4 | N5 的增强版，2022 年量产 | 名字从 5 变成 4，官方说法是“增强版”，不是全新一代 |
| N4P | 比 N5 性能高 11%，2023 年量产 | 比较基准是 N5，不是 N4 |
| N3E | 比 N5 速度约高 20%，功耗省 30% 以上，逻辑密度约 1.6 倍 | TSMC 写 N2P 时把“同功耗下更快”和“同速度下更省电”分成两种比法；这类数字一般不能同时拿到 |
| N2P | 比 N3E 同功耗快 18%，或同速度功耗低 36%；逻辑密度 1.2 倍，芯片密度 1.15 倍 | 逻辑密度和整颗芯片的密度不是一回事，TSMC 分开给了两个数 |

NVIDIA 的产品只用它自己的说法：H100 的 GH100 用定制的 TSMC 4N，800 亿个晶体管，die 面积 814 mm²（[Hopper 架构博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)）；Blackwell 架构的 GPU 有 2,080 亿个晶体管，用定制的 TSMC 4NP，所有 Blackwell 产品都由两块“达到曝光上限”的 die 组成，之间用 10 TB/s 的片间互联连接（[Blackwell 架构页](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)）。“4N”“4NP”和 TSMC 公开的 N4 系列具体差在哪里，本文没有找到双方的公开说明。

比较两颗芯片，节点名只是起点。更可靠的是：同一个 workload 下实测的性能、功耗和成本。

## 为什么大 die 特别贵：一片晶圆的账

光刻机一次能曝光的区域有上限。ASML 的 NXT:2000i 支持的最大视场是 26 mm × 33 mm，即 858 mm²（[ASML 产品页](https://www.asml.com/en/products/duv-lithography-systems/twinscan-nxt2000i)）。GH100 的 814 mm² 已经是这个面积的 95%。所谓 “reticle limit”，就是单块 die 不能比一次曝光的区域还大。

先算一片 300 mm 晶圆能切多少块。设晶圆直径为 $d$，die 面积为 $A$。一个常用的近似是：

$$
\text{DPW}\approx\frac{\pi(d/2)^2}{A}-\frac{\pi d}{\sqrt{2A}}.
$$

第一项是面积之比；第二项扣掉边缘那一圈放不下完整 die 的位置，大约是周长除以 die 对角线的长度。

再算其中有多少是好的。最简单的模型假设缺陷随机、独立地落在晶圆上，平均每平方厘米 $D_0$ 个，一个 die 只要沾上一个缺陷就报废。落在面积 $A$ 里的缺陷数服从 Poisson 分布，一个都没有的概率就是良率：

$$
Y_{\rm Poisson}=e^{-D_0A},\qquad
Y_{\rm Murphy}=\left(\frac{1-e^{-D_0A}}{D_0A}\right)^2.
$$

真实的缺陷往往扎堆，Poisson 对大 die 偏悲观。[Murphy（1964）](https://doi.org/10.1109/PROC.1964.3442)让缺陷密度本身在 die 之间变化，右边是常说的 Murphy 模型。

**教学例子**：$D_0=0.1$ 个/cm²（自拟，晶圆厂一般不公开真实值），比较 100 mm² 的小 die 和 800 mm² 的大 die（例如 25 mm × 32 mm，放得进 26 × 33 mm 的视场）。

| die 面积 | 每片晶圆（公式） | Poisson 良率 | 好 die 数 | Murphy 良率 |
| ---: | ---: | ---: | ---: | ---: |
| 100 mm²（1 cm²） | 640 | $e^{-0.1}=0.905$ | 约 579 | 0.906 |
| 400 mm²（4 cm²） | 143 | $e^{-0.4}=0.670$ | 约 96 | 0.679 |
| 800 mm²（8 cm²） | 64 | $e^{-0.8}=0.449$ | 约 29 | 0.474 |

<!-- widget:step-good-dies -->

面积大 8 倍，好 die 少了 20 倍：切得少是一层，坏得多又是一层。用 0.5 mm 步长平移网格、直接数一下放得进的完整 die，理想晶圆上是 657、151、71 块，比公式略多；真实晶圆还要留边缘禁区和切割道，会更少。数字会变，“好 die 比面积掉得更快”这个结论不变。

## 切成两块再拼：chiplet 的账

把 800 mm² 的大 die 拆成两块 400 mm²：同一片晶圆约 96 块好 die，配成 48 对，而整块的只有约 29 块。前提是封装前就能测出哪块好（known good die），坏的不拿去封装。

这个结果还取决于缺陷密度，并且没有计入片间接口和封装成本。可以逐项改变条件，看看拆分的优势怎样变化：

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 把 $D_0$ 降到 0.05 | 整块 800 mm² 的好 die 从约 29 涨到约 43，拆分的优势变小 | 工艺成熟后就不需要 chiplet |
| 把 $D_0$ 提到 0.2 | 整块只剩约 13，拆成两块还有约 64 块、32 对 | 拆得越碎一定越省 |
| 给每块小 die 加上片间接口的面积 | 每块 die 变大，能切的数量和良率都略降 | 片间接口的功耗和延迟可以忽略 |
| 再乘上封装良率 | 一个封装坏了，里面的好 die 和 HBM 一起报废 | 封装便宜到可以不算 |

Blackwell 的做法说明了另一面：die 已经顶到曝光上限，想要更多晶体管，就只能拼两块，再用高带宽的片间互联把它们连成一个 GPU。拼起来以后，压力转到了封装。

## 先进封装：interposer 在做什么

一个 GPU 和每个 HBM 堆栈之间要连上千根数据线（HBM3 是 1,024 位，HBM4 是 2,048 位）。2.5D 封装的做法是：在 GPU die、HBM 和基板之间垫一层 interposer，用芯片级的细线把它们连起来。普通的有机基板放不下这么密的连接，JEDEC 的一份新标准从反面说明了这点：SPHBM4（JESD330-4，2026-07）沿用 HBM4 的 DRAM die，但把数据线减到四分之一、用 4:1 串行化补回带宽，这样凸点间距才能放宽到有机基板能接受的程度（[JEDEC，2026-07-13](https://www.jedec.org/news/pressreleases/new-jedec%C2%AE-sphbm4-standard-enables-hbm4-class-bandwidth-organic-substrates)）。

TSMC 的 CoWoS 就是这一类。按 TSMC 的说明（[CoWoS 页面](https://3dfabric.tsmc.com/english/dedicatedFoundry/technology/cowos.htm)，2026-10 读取），它把多个 SoC 和 HBM 堆栈集成在一起，2012 年开始量产：

| 类型 | interposer 是什么 | TSMC 给出的规格 |
| --- | --- | --- |
| CoWoS-S | 硅 interposer | 最大约 3.3 倍曝光面积（约 2,700 mm²） |
| CoWoS-R | 聚合物加铜线的 RDL interposer，较柔软 | 布线最小间距（pitch）4 µm，即线宽、线距各 2 µm；2023 年起量产 |
| CoWoS-L | RDL interposer 里嵌入局部硅互联（LSI）小芯片 | 3.5 倍曝光面积的版本 2024 年起量产；TSMC 建议超过 3.3 倍时用 L 或 R |

CoWoS-L 的 LSI 小芯片专门承担 die 与 die、die 与 HBM 之间的高密度走线，其余部分用 RDL。至于 NVIDIA 哪款产品用哪一种，本文查过的 NVIDIA 白皮书、架构博客和产品页都没有写，只有新闻：据路透社 2025-01-16 报道（[Taipei Times 转载](https://www.taipeitimes.com/News/biz/archives/2025/01/17/2003830329)），黄仁勋说 Blackwell 将主要使用 CoWoS-L，仍在生产的 Hopper 使用 CoWoS-S。这是新闻报道转述的 CEO 表态，不是产品规格。

## HBM：把内存堆到 GPU 旁边

HBM 把多层 DRAM die 竖着叠起来，层与层之间用硅通孔（TSV）连通。它不靠单根线跑得快，而靠极宽的接口；标准 HBM 这么宽的接口要靠 interposer 上细密的短线，所以 HBM 和先进封装是绑在一起的。

单个堆栈的带宽可以手算：

$$
\text{每堆栈带宽}=\frac{\text{接口位宽}\times\text{每根线速率}}{8}.
$$

| 代际 | 接口位宽 | 每根线速率 | 每堆栈带宽 | 容量 | 出处 |
| --- | ---: | ---: | ---: | --- | --- |
| HBM3 | 1,024 位 | 最高 6.4 Gb/s | 819 GB/s | 4 GB（4 层 × 8 Gb）到 64 GB（16 层 × 32 Gb） | [JEDEC，2022-01-27](https://www.jedec.org/news/pressreleases/jedec-publishes-hbm3-update-high-bandwidth-memory-hbm-standard) |
| HBM3E（Micron） | 1,024 位 | 超过 9.2 Gb/s | 超过 1.2 TB/s | 8 层 24 GB，12 层 36 GB | [Micron 产品页](https://www.micron.com/products/memory/hbm/hbm3e) |
| HBM4 | 2,048 位 | 最高 8 Gb/s | 最高 2 TB/s | 16 层 × 32 Gb，最高 64 GB | [JEDEC，2025-04-16](https://www.jedec.org/news/pressreleases/jedec%C2%AE-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard-advancing) |

验算一行：$1024\times6.4/8=819.2$ GB/s。HBM4 把位宽翻倍，所以线速率不比 HBM3E 高，带宽照样更大。Micron 的两个“超过”要一起看：按 9.2 Gb/s 正好算是 $1024\times9.2/8\approx1{,}178$ GB/s，比 1.2 TB/s 还差一点；每根线要到约 9.4 Gb/s，单个 placement（堆栈）才超过 1.2 TB/s。

再倒推一个产品：H100 SXM5 有 5 个 HBM3 堆栈、5,120 位接口，[白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)给出显存带宽 3,352 GB/s。每根线约 $3.352\times10^{12}\times8/5120\approx5.24$ Gb/s，和白皮书列的显存数据速率 2,619 MHz DDR（每个时钟传两次，约 5.24 Gb/s）一致，低于 HBM3 标准允许的 6.4 Gb/s。产品带宽不等于“堆栈数 × 标准上限”，要看具体 SKU 的规格。

## 为什么产能卡在封装和 HBM

一个成品 GPU 模块要同时拿到三样东西。用自拟数字算一下（不对应任何具体产品）：某个月有 12,000 块好的逻辑 die（每个模块要 2 块），60,000 个 HBM 堆栈（每个模块要 8 个），封装线能做 6,500 个模块。能出货的是

$$
\min\left(\frac{12000}{2},\ \frac{60000}{8},\ 6500\right)=\min(6000,\ 7500,\ 6500)=6000.
$$

这个月卡在逻辑 die；如果晶圆厂多给 2,000 块，可配成 7,000 对，出货上限便从 6,000 提到封装限制的 6,500。再扩封装，先碰到的是这 7,000 对逻辑 die 的上限；只有逻辑 die 也增至至少 15,000 块、封装能力达到至少 7,500 套时，HBM 的 7,500 套才成为下一道限制。扩当前瓶颈可以增加出货，但扩非瓶颈环节，或在瓶颈转移后继续只扩原环节，都不会继续增加出货。

公开资料里的几条信号，按日期看：

- **TSMC 自己的说法**（CoWoS 页面，2026-10 读取）：2022 年底生成式 AI 兴起以后，CoWoS 的需求更大了，TSMC 在继续扩充先进封装产能。
- **新闻，路透社 2024-05-02**（[ThePrint 转载](https://theprint.in/tech/nvidia-supplier-sk-hynix-says-hbm-chips-almost-sold-out-for-2025/2066101/)）：SK hynix 表示，2024 年的 HBM 已经售罄，2025 年的也几乎卖完。
- **新闻，路透社 2025-01-16**（Taipei Times 转载）：黄仁勋说先进封装产能大概是不到两年前的 4 倍；报道写道，封装因产能限制仍是瓶颈。这后半句是记者的话。

这些是某个时间点的快照，产能每个季度都在变；引用时写上日期，不要把它当成现在仍然成立的结论。

## 回到一次请求

[导读](README.md)里，decode 每一步都要把 16 GB 权重从 HBM 读一遍，速度由 HBM 带宽决定。想让这一步更快，就要更多或更快的 HBM 堆栈，这意味着更大的 interposer、更多的封装产能，以及更多从晶圆上切下来的好 die。下一篇看 CPU 在这台机器里做什么：[CPU 架构](cpu-architecture.md)；GPU die 内部怎样用这些带宽，见 [GPU 架构](gpu-architecture.md)。
