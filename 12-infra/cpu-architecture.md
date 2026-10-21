# CPU 架构：流水线、缓存、SIMD 与 NUMA

**中文** · [English](cpu-architecture.en.md)

> 阅读时间：约 10 分钟 · 最近审阅：2026-10

一台 8 卡 GPU 服务器里也装着两颗 CPU。它们在干什么？GPU 利用率上不去时，为什么常常要回头查 CPU？**CPU 核心的设计目标，是让一条指令流尽快跑完：流水线、乱序执行、分支预测和多级缓存，都是为了掩盖“等”的时间。在 AI 服务器里，CPU 负责读数据、预处理、tokenize、向 GPU 发起工作和搬运数据；它慢一步，GPU 就只能空等。**

## 一个核心想把一条指令流跑多快

执行一条指令要经过取指、译码、读操作数、计算、写回几步。**流水线**让这几步像装配线一样重叠：上一条还在计算，下一条已经在译码。**超标量**让一个周期同时发出好几条指令。**乱序执行**让输入已经就绪的指令先算，最后再按程序顺序提交结果，这样一条指令在等内存时，后面不相关的指令不必干等。

流水线的麻烦在分支：`if` 的结果还没算出来，下一条该取哪里？**分支预测**先猜一个方向接着往下做，猜错了就把这段工作丢掉重来。用每条指令平均花几个周期（CPI）来算这笔账：

$$
\text{CPI}=\text{CPI}_{\rm base}+f_{\rm branch}\times m\times p.
$$

$f_{\rm branch}$ 是分支指令的比例，$m$ 是预测错误率，$p$ 是猜错一次要扔掉多少个周期。**教学数字**：$\text{CPI}_{\rm base}=0.5$（每周期平均完成 2 条），20% 的指令是分支，$p=15$ 周期。

| 预测准确率 | 额外 CPI | 总 CPI | 比理想情况慢 |
| ---: | ---: | ---: | ---: |
| 100% | 0 | 0.5 | 0% |
| 95% | $0.2\times0.05\times15=0.15$ | 0.65 | 30% |
| 90% | $0.2\times0.10\times15=0.30$ | 0.8 | 60% |

准确率只差 5 个百分点，速度差了 30%。这也是为什么处理随机数据的代码（比如按内容分叉的解析逻辑）在 CPU 上会比想象中慢。

## 缓存：大部分时间在等内存

从寄存器到内存是一串越来越大、越来越慢的存储：L1、L2、L3，再到 DRAM。数据按 cache line 整块搬运。Intel 的优化手册里有几个可以对照的数字：Skylake 服务器核心每核 1 MB 的 L2，最短“读出到可用”延迟 14 个周期，cache line 64 字节；Gracemont 核心的 L1 读延迟通常是 4 个周期（[Intel 64 与 IA-32 优化参考手册第 1 卷](https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf)，248966-049，2024-01）。

平均访存时间（AMAT）把各级串起来：

$$
\text{AMAT}=t_1+m_1\big(t_2+m_2\,(t_3+m_3\,t_{\rm mem})\big).
$$

$t_i$ 是第 $i$ 级命中所需的周期，$m_i$ 是在这一级没找到、要往下找的比例（相对于到达这一级的访问）。**教学数字**：$t_1=4$、$t_2=14$、$t_3=50$、$t_{\rm mem}=300$ 周期；L1 命中 95%，L2 接住 L1 缺失里的 80%，L3 接住剩下的一半。

$$
\text{AMAT}=4+0.05\times\big(14+0.2\times(50+0.5\times300)\big)=4+0.05\times54=6.7\ \text{周期}.
$$

| 只改一个条件 | AMAT | 说明 |
| --- | ---: | --- |
| 上面的基准 | 6.7 | 平均比 L1 命中只慢 2.7 个周期 |
| L1 命中率降到 90% | $4+0.10\times54=9.4$ | 缺失率翻倍，平均慢 40% |
| L2 只接住一半 | $4+0.05\times114=9.7$ | 更多访问一路掉到 DRAM |
| 每次都去 DRAM | 300 | 慢 45 倍；cache 让平均值接近 L1 |

结论：缓存命中率的小变化，会被几百个周期的 DRAM 延迟放大。写代码时最该关心的，是数据能不能留在缓存里被反复用。

## 同一个矩阵，按行走还是按列走

一个 $64\times64$ 的 float32 矩阵按行存放（C、NumPy、PyTorch 默认如此）：一行 256 字节，正好 4 个 64 字节的 cache line，整个矩阵 256 个 line。为了能手算，假设一个**只能放 8 个 line 的玩具 cache**，满了就踢掉最久没用的那个（LRU）。

- **按行走**：连续访问 16 个 float 才换一个 line，4,096 次访问只缺失 256 次，缺失率 6.25%。
- **按列走**：相邻两次访问隔了一整行（256 字节），落在不同的 line 上。一列要碰 64 个 line，cache 只放得下 8 个；等回头访问下一列，这些 line 早被踢掉了。4,096 次访问次次缺失。

<!-- widget:step-cache-walk -->

```python
from collections import OrderedDict

def misses(order, n=64, floats_per_line=16, cache_lines=8):
    cache, miss = OrderedDict(), 0
    for i, j in order(n):
        line = (i * n + j) // floats_per_line   # 按行存放：元素 (i, j) 在第 i*n + j 个位置
        if line in cache:
            cache.move_to_end(line)             # LRU：标记为最近用过
        else:
            miss += 1
            cache[line] = True
            if len(cache) > cache_lines:
                cache.popitem(last=False)       # 踢掉最久没用的 line
    return miss

by_row = lambda n: ((i, j) for i in range(n) for j in range(n))
by_col = lambda n: ((i, j) for j in range(n) for i in range(n))
assert misses(by_row) == 256
assert misses(by_col) == 4096
assert misses(by_col, cache_lines=64) == 256
```

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 把 `cache_lines` 改成 64 | 按列走也只缺失 256 次：一列的 64 个 line 正好放得下 | 真实 CPU 上按列走就没问题 |
| 把矩阵改成 $128\times128$，cache 仍是 64 个 line | 按列走又变成次次缺失 | 真实的分界点就在这里；真实 cache 有多级、组相联和硬件预取 |
| 按列走时数一数每个 line 用了几个字节 | 每搬 64 字节只用 4 字节，有效带宽只剩 1/16 | 按行走一定能跑满内存带宽 |

真实机器上，分界点取决于“行数 × 64 字节”和各级 cache 容量的关系；硬件预取器也能接住一部分固定步长的访问（Intel 优化手册建议按固定步长访问，好让预取器识别），所以要实测。放到 AI 代码里：PyTorch 的转置通常只改 stride、不搬数据，后面的算子可能就在按列走；对不连续的张量调用 `.contiguous()`，会复制一份按行重排的数据。GPU 上也有对应的问题，叫访存合并，见 [GPU 架构](gpu-architecture.md)。

## SIMD：一条指令处理一排数

SIMD 让一条指令同时处理一个向量里的多个数。

- **AVX-512（Intel）**：32 个 512 位向量寄存器、8 个掩码寄存器；一个寄存器放 16 个 float32 或 8 个 float64（[Intel AVX-512 介绍](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-avx-512-instructions.html)）。
- **AMX（Intel）**：8 个二维 tile 寄存器，每个最多 16 行 × 64 字节，即 1 KB；由叫 TMUL 的矩阵乘单元在 tile 上做 INT8 和 BF16 矩阵乘；Intel 的示例在第 4 代至强可扩展处理器上运行（[Intel AMX 代码示例](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)）。Intel 给出的每核每周期峰值：AMX 做 2,048 次 INT8 运算，AVX-512 VNNI 是 256 次；AMX 做 1,024 次 BF16 运算，AVX-512 做 FP32 是 64 次（[Intel AMX 简介](https://cdrdv2-public.intel.com/785250/Intel-AMXBrief-Final-3.17.pdf)，按架构峰值、CPU 满载计）。
- **Neon 与 SVE（Arm）**：Neon 的向量固定 128 位。SVE 的向量长度由芯片实现决定，同一个二进制程序可以在不同长度的机器上运行（vector-length agnostic）。Arm 2025-02 修订的 [Introduction to SVE](https://documentation-service.arm.com/static/67ab35a4091bfc3e0a9478b5) 写明长度在 128 到 2048 位之间、必须是 2 的幂；同一文档的修订记录说明，这条限制是这一版新加的，旧材料里“128 位的任意整数倍”的说法已经过时。

粗算一下 CPU 的矩阵乘上限：DGX H100 的两颗 Xeon 8480C 共 112 核，基础频率 2.0 GHz（[DGX H100 用户手册](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)）。按 Intel 的 AMX 峰值，$112\times2.0\times10^9\times1024\approx229$ TFLOP/s BF16。这是纸面上限：AMX 满载时的实际频率、内存带宽都没算进去。同一台机器里 8 张 H100 SXM5，dense BF16 合计 $8\times989.4\approx7{,}915$ TFLOP/s（[H100 架构白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)；产品页每卡写的 1,979 是 2:4 稀疏数字）。CPU 能跑小模型推理，但大矩阵还是交给 GPU。

## 多核与 NUMA：内存也分远近

一台服务器有两颗 CPU 时，每颗 CPU（socket）挂着自己的内存。Linux 内核文档把这样的分组叫 node：CPU 访问本 node 的内存，比访问别的 node 更快、带宽更高；Linux 默认从发起申请的 CPU 所在 node 分配内存，不够时才去别的 node 拿（[Linux 内核 NUMA 文档](https://docs.kernel.org/mm/numa.html)）。

GPU 和网卡也挂在某一颗 CPU 的 PCIe 上。设想 GPU 0 挂在 socket 0，给它喂数据的 worker 却跑在 socket 1、数据也放在 socket 1 的内存里：每个 batch 都要先跨 socket 走一趟，再上 PCIe。单次看不出来，8 张卡一起跑时，跨 socket 的链路可能被挤满。

可以这样查：`nvidia-smi topo -m` 看每张 GPU 离哪颗 CPU 近；`numactl --hardware` 看 node 的编号和距离；用 `numactl --cpunodebind=0 --membind=0` 把进程和内存绑到同一个 node。CUDA C++ Best Practices Guide 的 NUMA 一节还提醒，有些发行版默认开启的自动 NUMA balancing 可能拖慢 GPU 程序，建议手动调整（[CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)）。

## x86 和 Arm：数据中心里的两条路

- **x86**：DGX H100 用 2 颗 Intel Xeon 8480C，每颗 56 核，配 2 TB 内存（DGX H100 用户手册）。
- **Arm**：NVIDIA Grace 是 72 个 Arm Neoverse V2 核心，每核 4×128 位 SVE2，单颗配最多 480 GB LPDDR5X，内存带宽最高 512 GB/s（120 GB、240 GB 配置；480 GB 配置是 384 GB/s）（[Grace CPU 产品页](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/)）。在 Grace Hopper 里，CPU 和 GPU 之间用 900 GB/s 的 NVLink-C2C 相连，NVIDIA 称这是 PCIe Gen5 的 7 倍，并支持 CPU 与 GPU 一致的内存访问（[Grace Hopper 产品页](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)）。GB200 NVL72 一个机柜里是 36 颗 Grace 和 72 个 Blackwell GPU（[GB200 NVL72 产品页](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)）。

对 AI 服务器来说，指令集本身不是重点。更该比较的是：CPU 内存带宽多大，CPU 到 GPU 的链路多快，要用的软件有没有对应架构的版本。

## CPU 在 AI 服务器里干什么

| 工作 | 例子 | 做得不好时的表现 |
| --- | --- | --- |
| 读数据、解码、增强 | 读图片、解码视频、拼 batch | GPU 利用率周期性掉到 0，CPU 核满载 |
| tokenize 和模板 | 聊天请求进来先套模板、切 token | 首 token 时间变长，GPU 却不忙 |
| 发起 GPU 工作 | Python 调框架，框架逐个发射 kernel | 小 batch 时 GPU 上全是空隙 |
| 主机与显存之间搬数据 | 把 batch 拷到 GPU，把结果拷回来 | 拷贝和计算没有重叠 |
| 网络与存储 | RPC、checkpoint 写盘、日志 | 尾延迟抖动，保存 checkpoint 时训练停顿 |

**手算一个数据加载**（教学数字）：8 张 GPU，每张每秒要吃 1,000 个样本，CPU 处理一个样本要 4 ms。需要的 CPU 核数是

$$
8\times1000\ \text{个/秒}\times0.004\ \text{秒/个}=32\ \text{核}.
$$

112 核的机器够用。如果换成重的增强或视频解码，每个样本 20 ms，就要 160 核，超过整台机器，GPU 只能等。这时该做的是把解码挪到别处、预先处理好数据，或者把部分预处理放到 GPU 上，而不是继续加 `num_workers`。

搬数据这一步，要记住带宽差多少。H100 SXM5 的 HBM 是 3,352 GB/s（白皮书 V1.04）；PCIe Gen5 x16 每个方向 64 GB/s（[Hopper 架构博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)），差了约 50 倍。CUDA 最佳实践指南举的老例子是 V100 的 898 GB/s 对 PCIe Gen3 的 16 GB/s，也差 50 多倍，所以它把“尽量少在主机和 GPU 之间搬数据”列为高优先级建议。它还说明：用锁页（pinned）内存传输带宽最高；异步拷贝要求主机内存是锁页的，这样拷贝才能和 kernel 重叠；但锁页内存是稀缺资源，不能滥用。PyTorch 的 DataLoader 设 `pin_memory=True`，就会把取到的 batch 放进锁页内存；`num_workers` 大于 0 时用多个进程加载（[PyTorch 2.14 DataLoader 文档](https://docs.pytorch.org/docs/2.14/data.html)）。

## 为什么还要 GPU

CPU 把大量晶体管花在“让一条线程少等”上：大缓存、乱序窗口、分支预测器。GPU 反过来。CUDA C++ 编程指南的说法是：CPU 擅长把一个线程尽快跑完，同时并行几十个线程；GPU 擅长同时跑几千个，把更多晶体管用在计算上，而不是缓存和控制流，用计算掩盖访存延迟，而不是靠大缓存和复杂的控制逻辑（[CUDA C++ 编程指南 12.8.1，§1.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html)）。再配上 HBM 这样的高带宽显存。数量级上：一颗 Grace 最多 512 GB/s，一张 H100 SXM5 是 3,352 GB/s；上面那台机器两颗 CPU 的 AMX 纸面峰值，不到 8 张 GPU 的 3%。

所以分工是：CPU 管控制流、不规则的小任务和数据准备，GPU 管大块、规则、可以并行的矩阵运算。下一篇讲 GPU 怎样做到这一点：[GPU 架构](gpu-architecture.md)。这台机器的芯片是怎么造出来的，见[芯片怎么造出来](chips-and-fabs.md)；整条链路的地图在[导读](README.md)。
