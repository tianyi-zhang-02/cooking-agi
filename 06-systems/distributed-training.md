# 多卡训练：数据、参数和计算，到底怎么分？

**中文** · [English](distributed-training.en.md)

> 最近审阅：2026-10 · 前置：[一次训练](../00-foundations/deep-dives/training-step.md)、[精度与显存](../00-foundations/deep-dives/precision-and-memory.md)

“用了多张卡”没有告诉我们模型怎样训练。**可能是多份模型各看不同数据，也可能一张卡只拿了模型的一部分；两者都会通信，但原因完全不同。**

先拿 4 层的小模型做示意。这里的尺寸和成本是教学假设，不是硬件 benchmark。

## 先分清两个需求

如果单卡能放下完整模型，只是训练太慢，可以考虑把数据分给更多卡。如果连模型状态都放不下，就要切分状态或计算；多复制几份模型不会解决 OOM。

| 方式 | 主要分什么 | 每卡负责什么 |
| --- | --- | --- |
| Data Parallel / DDP | 数据 | 完整模型上的不同样本 |
| FSDP / ZeRO | 参数、梯度、optimizer 状态的全部或部分 | 保存分片，计算时按需要聚合 |
| Tensor Parallel / TP | 层内矩阵计算 | 一次矩阵运算的一部分 |
| Pipeline Parallel / PP | 模型层 | 一组连续层及经过它们的 microbatches |

它们可以组合。要解释一个配置，最好画出数据流、参数存放位置和通信发生时刻，而不只是说“3D parallel”。

## DDP：各算一份梯度，再同步

```text
GPU 0：样本 A → 完整 4 层模型 → 梯度 g0 ┐
GPU 1：样本 B → 完整 4 层模型 → 梯度 g1 ├→ 平均梯度 → 各自更新
GPU 2：样本 C → 完整 4 层模型 → 梯度 g2 ┤
GPU 3：样本 D → 完整 4 层模型 → 梯度 g3 ┘
```

相同初始参数、相同 optimizer 状态、相同同步后梯度，才能让各副本一起走。[PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html)会按 bucket 同步梯度，并可与反向计算重叠；数据如何分片仍需要调用方安排。

DDP 是训练方式与框架接口，Ring AllReduce 是 collective 的一种实现算法，不是同一个层次的名字。单机多卡也可以用 DDP；具体通信算法由 backend 和配置决定。不要把“DP = 参数服务器、DDP = 多机 Ring”当作定义。

每卡 2 条样本，累积 4 次，4 个 data-parallel ranks，每次更新对应 32 条样本；TP ranks 不应再乘进这个样本数量。重复采样或 sampler 补齐也可能改变实际唯一样本数。

## 最容易漏掉的是全局分母

假设 rank 0 有 2 个有效 tokens，loss 总和为 4；rank 1 有 6 个，loss 总和为 6。先各自求均值再平均，得到 1.5；真正按 token 平均应是 1.25。

设一整个更新窗口的全局有效 token 数为 $T$，data-parallel world size 为 $W$，rank $r$ 的可微 loss 总和为 $S_r$。若同步操作会对梯度取平均，可以在本地反向：

$$
\widetilde{\mathcal L}_r=\frac{W S_r}{T}.
$$

同步后梯度为

$$
\frac1W\sum_r\nabla\widetilde{\mathcal L}_r
=\frac{\sum_r\nabla S_r}{T}.
$$

有效数量本身不求梯度，需要先归并确定。若框架执行的是求和或另有归一化，系数相应调整，不能把这条公式不加判断地套上。

```python
loss_sums = [4.0, 6.0]
valid_counts = [2, 6]
world_size = len(loss_sums)
global_tokens = sum(valid_counts)
local_objectives = [world_size * total / global_tokens for total in loss_sums]
ddp_average = sum(local_objectives) / world_size
assert ddp_average == 1.25
assert sum(total / count for total, count in zip(loss_sums, valid_counts)) / world_size == 1.5
```

这是标量账的演示，梯度结论来自上面的线性求和。累积多个 microbatch 时，分母要覆盖整个窗口；遇到全局零有效 tokens 要跳过或显式报错，不能除零。

## PS 与 Ring：梯度怎样到达更新的位置？

Parameter Server（PS）由 worker 推送梯度、服务器维护参数或参数分片；它可以同步，也可以异步，不能把 PS 直接等同于 stale gradients。同步 AllReduce 则让各参与者取得归约结果。两种组织方式都需要定义一次更新到底包含哪些样本。

Ring AllReduce 常用两个阶段：先 reduce-scatter，让每个 rank 拿到完整归约结果的一块；再 all-gather，让所有 rank 拿齐各块。collective 的定义与实现算法要分开：[NCCL 的介绍](https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/)可帮助理解 ring，但这是历史文章，不是说今天所有通信都固定选择 ring。

对 $P$ 个 ranks、每卡 $S$ 字节的消息，假定均匀分块和理想环路：每轮发送 $S/P$，每阶段 $P-1$ 轮，所以**每卡发送量**为

$$V_{\text{send}}=2\frac{P-1}{P}S.$$

接收量也一样；若把发送和接收加总，需要再乘 2。4 卡归约 120 MB 梯度时，每卡发送 180 MB，不是把每张卡全部梯度都分别发送给其他 3 张卡的 360 MB。

若每轮固定延迟为 $\alpha$、有效带宽为 $B$，串行轮次的简化成本为

$$t_{\text{ring}}\approx2(P-1)\alpha+2\frac{P-1}{P}\frac{S}{B}.$$

这里忽略归约计算、链路争用、分块重叠与实际拓扑。大消息可能受带宽限制，小消息可能受轮次延迟限制；这个公式不能替代 profiler。一次 all-reduce 的耗时怎样估、怎样用 profiler 量出通信等待，见[性能：多卡通信与 profiling](../12-infra/performance.md#communication)。

```python
def ring_traffic(message_bytes, ranks):
    if type(ranks) is not int or ranks < 1 or type(message_bytes) is not int or message_bytes < 0:
        raise ValueError("expected positive integer ranks and nonnegative integer bytes")
    sent = 2 * (ranks - 1) * message_bytes / ranks
    return {"sent_per_rank": sent, "received_per_rank": sent, "rounds": 2 * (ranks - 1)}

assert ring_traffic(120_000_000, 4)["sent_per_rank"] == 180_000_000
assert ring_traffic(120_000_000, 1)["rounds"] == 0
```

### 异步不是只省掉等待

用 $L(w)=w^2/2$、学习率 0.5 做一笔账。初始 $w=2$，第一次更新后为 1。下一次若用当前梯度 1，就得到 0.5；若另一 worker 刚送来在旧参数 2 上算的梯度 2，就得到 0。这里 stale 更新碰巧更接近最优点，但不能据此说它更好——它已经不是同一条优化轨迹。

实际系统应记录梯度基于哪个参数版本、可接受多旧的更新、是否丢弃或重加权，以及恢复时怎么处理在途工作。只有梯度同步更快，不足以证明训练结果相同。

## FSDP / ZeRO：别让每卡都长期保存全部状态

[ZeRO](https://arxiv.org/abs/1910.02054)分阶段切 optimizer states、梯度和参数；[FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)介绍了全分片训练的具体实现。它们并不是同一套 API，但可以放在“减少状态复制”这个问题下理解。

以 full-shard、计算后重新分片的一种路径为例：

```text
本卡持有参数分片
    → all-gather 当前模块需要的参数
    → forward / backward
    → reduce-scatter 梯度
    → 用本地 optimizer 状态更新本地分片
```

何时释放完整参数、反向前是否重新聚合、是否提前 prefetch，都影响峰值和通信量。不能只拿总显存除以卡数。

假设参数、梯度、optimizer 状态分别是 2、2、12 GB，4 卡情况下，仅看理想的常驻状态：

| 策略 | 每卡状态量 |
| --- | ---: |
| 全复制 | $2+2+12=16$ GB |
| ZeRO Stage 1：仅 optimizer 分片 | $2+2+12/4=7$ GB |
| ZeRO Stage 2：optimizer 与梯度分片 | $2+2/4+12/4=5.5$ GB |
| ZeRO Stage 3：三者都分片 | $(2+2+12)/4=4$ GB |

这里没算激活、通信 buffer 和临时聚合。4 GB 不是训练峰值承诺，最大的计算单元也必须能放下。

### Stage 2 通信量相同，为什么不一定一样快？ {#zero-communication}

先看更新后的参数要去哪儿。Stage 2 里，每卡只更新自己负责的那部分参数；下一轮各卡还要跑完整模型，所以必须把更新后的分片重新拼齐。省掉完整梯度的长期存储，不等于省掉了所有同步。

用一个能手算的通信账：4 个 ranks，完整梯度与完整参数各为 8 MiB，按等分 ring 实现，只数**每个 rank 发出的 payload**，不把接收量再加一次。一次 reduce-scatter 或 all-gather 各发送 $8\times3/4=6$ MiB：

| 一个更新周期的简化路径 | 每 rank 发送量 |
| --- | ---: |
| DP：梯度 reduce-scatter → 梯度 all-gather | 6 + 6 = 12 MiB |
| Stage 2：梯度 reduce-scatter → 更新本地分片 → 参数 all-gather | 6 + 6 = 12 MiB |
| Stage 3：前向聚合参数 → 释放 → 反向再聚合 → 梯度 reduce-scatter | 6 + 6 + 6 = 18 MiB |

这是对 [ZeRO §7](https://arxiv.org/html/1910.02054v3#S7)通信分析的具体化，假设参数和梯度传输精度相同、没有梯度累积或额外重聚合。真实 bucket、保留参数和重算策略可能改变这条路径；Stage 3 的 18 MiB 不是所有配置的固定值。

字节相同，耗时仍可能不同：消息拆得多碎、何时发起、能不能与计算重叠，以及最慢的 rank，都在影响等待时间。于是要同时看**发了多少、什么时候发、等了多久**，而不是只凭“通信量没增加”判断性能。这里没有跑多卡 benchmark。

## TP：一次矩阵运算由几张卡合作

对行向量写法 $Y=XW$，可以按 W 的列切：

$$
\begin{aligned}
W&=[W_1\;W_2],\\
Y&=[XW_1\;XW_2].
\end{aligned}
$$

也可以沿输入维度切，得到需要相加的局部结果：

$$
\begin{aligned}
X&=[X_1\;X_2],\\
W&=\begin{bmatrix}W_1\\W_2\end{bmatrix},\\
Y&=X_1W_1+X_2W_2.
\end{aligned}
$$

取 $X=[1,2]$，$W=\begin{bmatrix}1&3\\2&4\end{bmatrix}$，完整输出为 $[5,11]$。按行切产生 $[1,3]$ 和 $[4,8]$，相加恢复结果；不是平均。

[Megatron-LM](https://arxiv.org/abs/1909.08053)展示了怎样配合列切和行切组织 Transformer 层。是否要拼回完整的中间结果，取决于下一步需要什么；每层都 gather，可能只是多传了一遍数据。

TP 的通信频率可能很高，通常更依赖高速互联（NVLink 与 NVSwitch 见 [GPU 架构](../12-infra/gpu-architecture.md)）。小矩阵切得太碎时，通信和调度开销可能超过收益。

如果想把这一步算透，读[张量并行：两张卡怎样算同一层？](tensor-parallel.md)。那里用一个两层 FFN 跟完前向与反向，区分拼接、求和与平均，再解释“一层 Decoder 四次通信”的适用条件。

## PP：分层之后，别让卡一直等着

把层 1–2 放在 GPU 0，层 3–4 放在 GPU 1。一个 batch 串着走时，GPU 1 一开始没活，GPU 0 做完前向又可能等待。把 batch 拆成多个 microbatches，可以让不同阶段同时处理不同数据。

理想的等耗时、无通信开销的 GPipe 式 fill/drain 估算中，$P$ 个 stages、$M$ 个 microbatches，利用率约为：

$$
\frac{M}{M+P-1}.
$$

2 stages、4 microbatches 时为 80%。真实调度的 bubble、前反向比例、负载不均和显存占用都会改变结果；更多 microbatches 也不是无限免费。[GPipe](https://arxiv.org/abs/1811.06965)是理解这个过程的原始资料。

**调度顺序与参数更新时间是两件事。** 1F1B 说的是稳态下交替做前向与反向，不意味着每做完一个 microbatch 就更新权重。同步训练仍可累积完整个 batch 再更新。[PyTorch 的流水线接口](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html)支持多种调度；讨论 [PipeDream](https://arxiv.org/abs/1806.03377)的异步更新与 weight stashing 时，要另外说明权重版本。

### PipeDream 为什么还要保存旧权重？

在允许流水线更新的设计里，同一个 microbatch 的前向和反向之间，参数可能已经改变。weight stashing 保存该 microbatch 前向使用的权重版本，让反向使用匹配的局部导数。[PipeDream](https://arxiv.org/abs/1806.03377)

用两层标量模型 $y=bax$，令 $x=1,a=2,b=3,L=y^2/2$。前向得到 6，正确的 $\partial L/\partial a=6\times3=18$。若反向时直接使用已经改成 4 的 $b$，旧激活配上新权重就给出 24。这个例子说明为何版本重要，不是完整的 PipeDream 调度模拟。

保存旧版本会增加状态；而且局部前后向版本一致，不自动保证所有 stages 使用同一个全局版本，更不使异步训练等同于大 batch 同步 SGD。对比实验要分别检查更新语义和吞吐，不只看 bubble 消失了多少。

## SP、CP、EP：名字接近，切的东西不一样

| 方式 | 先看什么 | 不能省略什么 |
| --- | --- | --- |
| Sequence Parallel / SP | 在 Megatron 的用法中，配合 TP 切 LayerNorm、dropout 等部分的序列激活 | 不能据此说 attention 已沿序列切到多卡上（配合的 TP 切的是 heads） |
| Context Parallel / CP | 沿序列切输入与激活，各卡拿一段 context | 本地 query 仍需要远处 key/value，必须组织交换 |
| Expert Parallel / EP | 将不同专家放到不同设备 | token 派发、结果归并与负载不均 |

这里采用 [Megatron 对 SP / CP 的区分](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)。别的框架可能把序列切分统称为 SP，读文档时要看实际的数据流。

一个小例子：把 8 个 tokens 分给两张卡，GPU 0 拿前 4 个，GPU 1 拿后 4 个。若各自只在本卡算 causal attention，第 8 个 token 就看不到前 4 个，模型已经变了。CP 要解决的是保留这些依赖，同时分摊存储与计算，不是简单拼回输出。

EP 则是另一个问题：一条 token 选中两个专家，若专家在远端，就要发送激活，并把结果送回原位置后加权合并。[Megatron 的 MoE 并行说明](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)展示了它怎样与其他并行方式组合。选中专家的比例不是通信字节比例，更不保证固定倍数加速；热点专家仍可能让别的卡等待。

<span id="dp"></span><span id="pp-kv"></span>

## 推理部署时，卡怎么分？ {#eight-gpus}

训练里的并行方式，到了推理部署要换个问法：同样是八张 GPU，是八张合作跑一份模型（TP8），还是八份模型各接各的请求（DP8）？MoE 的专家怎样跨副本分布？PP 的 KV 存在哪张卡、相邻 stage 每步传多少字节？这几笔账放在 [vLLM 与 SGLang：多张 GPU 的拼法](inference-frameworks.md#deployment-map)，和推理引擎的调度一起讲。

## Offload：把状态移出去，也要算回来要多久

CPU 内存能缓解显存压力，但不是免费扩容。optimizer 更新若移到 CPU，就要考虑梯度传出、更新后的参数传回，以及 CPU 自身的计算时间。小 batch 下，GPU 很快算完，反而可能一直等 CPU。[ZeRO-Offload](https://arxiv.org/abs/2101.06840)讨论的正是这些放置与调度取舍。

因此“理想状态量除以卡数”只能算一部分账。还需要明确临时聚合、激活、prefetch、链路带宽和 optimizer 更新何时结束。参数更多、序列更长或卡更多，都可能改变原先的瓶颈。

例如每次更新顺序传出 2 GB 梯度、传回 2 GB 权重，两个方向的有效带宽都假设为 25 GB/s，仅传输就约需 `2/25 + 2/25 = 0.16` 秒，尚未计 CPU 更新。这里用十进制 GB；它不是任何显卡的实测参数。

想靠 overlap 隐藏这段时间，还得有足够独立的计算，并且下一步不能早于所需权重到齐。真实关键路径取决于依赖、预取粒度、内存与争用，不能简单把所有时间都取一次 `max` 就当作最终延迟。

## 恢复训练：不只是重新读取参数

最容易验证的方案是在同步更新边界保存，先排空流水线，再写 checkpoint。对于下面几类状态，都要决定是恢复还是明确重置：

| 状态 | 漏掉后会发生什么 |
| --- | --- |
| 参数、optimizer、scheduler、step | Adam 动量或学习率轨迹改变 |
| RNG、sampler / 数据游标 | 重复或跳过样本，随机增强改变 |
| 尚未完成的梯度累积与有效 token 数 | 恢复后分母和 batch 不一致 |
| 分片映射、参数版本、在途 microbatch | 加载了不匹配的状态，或重复应用旧梯度 |

不是每个工程都支持任意时刻恢复。先从“保存后多走一步”和“重新加载后多走一步”对照开始，固定随机性，比对 loss、梯度和更新后的参数。支持改卡数恢复时，还要单独验证 reshard；不能只凭文件能读开就宣布恢复正确。

## 先检查正确性，再追吞吐

用小模型、固定样本和无 dropout 设置，比对单卡与多卡的一次更新：loss、有效 token 数、梯度、更新后权重。再加入不同长度、空样本、梯度累积和 checkpoint 恢复。

性能则分别记录有效 tokens/s、每卡峰值、通信时间和慢卡等待。GPU 使用率高不一定代表有用工作多，持续 OOM 重试或大量 padding 也不是有效吞吐。

选法可以很朴素：模型放得下先做 DDP 基线；状态放不下再考虑分片；单层计算过大看 TP；按层拆分看 PP。每加一种并行，都应能说清省下什么，以及新引入了哪次通信。
