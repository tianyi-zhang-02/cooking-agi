# 多卡训练：数据、参数和计算，到底怎么分？

**中文** · [English](distributed-training.en.md)

> 最近审阅：2026-10 · 前置：[一次训练](../00-foundations/deep-dives/training-step.md)、[精度与显存](../00-foundations/deep-dives/precision-and-memory.md)

“用了多张卡”没有告诉我们模型怎样训练。可能是多份模型各看不同数据，也可能一张卡只拿了模型的一部分。两者都会通信，但原因完全不同。

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

$$T\approx2(P-1)\alpha+2\frac{P-1}{P}\frac{S}{B}.$$

这里忽略归约计算、链路争用、分块重叠与实际拓扑。大消息可能受带宽限制，小消息可能受轮次延迟限制；这个公式不能替代 profiler。

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

TP 的通信频率可能很高，通常更依赖高速互联。小矩阵切得太碎时，通信和调度开销可能超过收益。

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
| Sequence Parallel / SP | 在 Megatron 的用法中，配合 TP 切 LayerNorm、dropout 等部分的序列激活 | 不能据此说每层 attention 已在多卡之间拆开 |
| Context Parallel / CP | 沿序列切输入与激活，各卡拿一段 context | 本地 query 仍需要远处 key/value，必须组织交换 |
| Expert Parallel / EP | 将不同专家放到不同设备 | token 派发、结果归并与负载不均 |

这里采用 [Megatron 对 SP / CP 的区分](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)。别的框架可能把序列切分统称为 SP，读文档时要看实际的数据流。

一个小例子：把 8 个 tokens 分给两张卡，GPU 0 拿前 4 个，GPU 1 拿后 4 个。若各自只在本卡算 causal attention，第 8 个 token 就看不到前 4 个，模型已经变了。CP 要解决的是保留这些依赖，同时分摊存储与计算，不是简单拼回输出。

EP 则是另一个问题：一条 token 选中两个专家，若专家在远端，就要发送激活，并把结果送回原位置后加权合并。[Megatron 的 MoE 并行说明](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)展示了它怎样与其他并行方式组合。选中专家的比例不是通信字节比例，更不保证固定倍数加速；热点专家仍可能让别的卡等待。

## 卡数相同，TP8 和 DP8 在忙什么？ {#eight-gpus}

先看 dense 模型，不启用共享专家。**TP8 是 8 张卡合作跑一份模型；DP8 / TP1 是 8 份模型分别接请求。** 一组 TP8 也可以批量处理很多请求，不是“一组只能服务一个人”。差别在于一轮计算由谁合作完成。

假设权重恰好占 16 GiB、每卡 24 GiB，并把全部权重视为可均匀切分。下面只算权重，暂不计 KV、工作区和小型复制参数：

<figure class="worked-update">
<ol>
<li><small>TP8 · 1 个协作组</small><strong>理想每卡 2 GiB 权重</strong><span>同一批请求在 8 卡上协作。省下单卡权重空间，但层内需要频繁交换结果。</span></li>
<li><small>DP8 / TP1 · 8 个副本</small><strong>每卡 16 GiB 权重</strong><span>各副本分担请求，dense 前向不需要副本之间同步。每份还有自己的缓存和队列。</span></li>
<li><small>DP2 / TP4 · 2 个协作组</small><strong>理想每卡 4 GiB 权重</strong><span>每组 4 卡合作，两组分担流量。这不是同时用了 8 份模型。</span></li>
</ol>
<figcaption>纯权重估算，不是可服务并发或吞吐结果。DP8 的单卡剩余 8 GiB，还得容纳缓存和运行时；不能全算成 KV。</figcaption>
</figure>

权重放得下、目标是总吞吐时，可以先测副本方案；单卡容量不足，或者单请求计算太重时，再比较 TP / PP。TP 的矩阵变小不保证延迟线性下降：通信、kernel 效率和调度都可能抵消收益。[vLLM 扩展指南](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)给出了单卡、TP 与 TP+PP 的部署起点；实际选择仍要测自己的请求分布。

比较时固定总卡数、到达流量、输入/输出长度和缓存预热条件。不要只看离线 tokens/s：也看首 token 延迟（TTFT）、输出间隔、尾延迟、被拒请求和每卡显存峰值。一个吞吐更高、但排队更久的配置，不一定更适合交互产品。

## 部署时的 DP，不一定是互不相干的副本

训练里的 DDP 同步梯度；推理里的 data parallelism 通常分配请求。对于 MoE，还要看专家是否跨副本共享。按 2026-10-09 核对的 [vLLM EP 部署说明](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)，启用 EP 时，专家组大小为 TP × DP；attention 的复制或切分与专家分布不是同一件事。这是该框架的组合方式，不是所有系统的定义。

用一个**虚构权重账**：可切分的非专家权重共 6 GiB，路由专家共 42 GiB，部署在 4 张卡上。暂不计复制的小张量、量化元数据、KV 和 workspace：

| 配置 | 每卡非专家权重 | 每卡路由专家 | 每卡合计 |
| --- | --- | --- | --- |
| TP1 / DP4 / EP4 | 6 GiB | 10.5 GiB | 16.5 GiB |
| TP4 / DP1 / EP4 | 1.5 GiB | 10.5 GiB | 12 GiB |

第二种每卡少 4.5 GiB，但单次 attention 计算要在 TP 组内协作。第一种把 attention 留给各 rank；遇到远端专家，仍要交换激活。**不能把第一种当成 4 套完全独立、互不等待的服务。** 副本的缓存、请求队列和共享专家的调度要一起看。

这是容量估算，不是吞吐预测。比较时固定到达流量、输入/输出长度和缓存冷热，记录首 token 延迟、输出间隔、尾延迟与专家负载。若矩阵被切得过小，省显存也可能换来更多等待。

## PP 的 KV 在哪里，边界传多少东西？

先限定为普通 decoder、按层做 PP，不叠加 cache offload 或 prefill/decode 分离。某张卡计算哪些层，就保存那些层的 KV；相邻 stage 传递边界激活，不需要每步把全部历史 KV 来回搬。

但“只传激活”不等于“传得很少”。设 hidden size 为 3072，BF16 每元素 2 字节：

| 单次边界传输 | 张量元素数 | 原始字节数 |
| --- | --- | --- |
| 8 条请求各 decode 1 个 token | 8 × 3072 | 48 KiB |
| 8 条请求各 prefill 4096 个 token | 8 × 4096 × 3072 | 192 MiB |

这里尚未计额外 residual 张量、元数据、协议和同步。长 prefill 可以分块传，代价是调度与流水线行为随之变化；不能拿 decode 的小包去解释所有场景。

[vLLM 的扩展指南](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)建议无 NVLink 的某些配置优先考虑 PP，以减少通信；这不构成“PP 在所有 PCIe 机器上最优”的证明。单请求仍要依次经过各 stage，层间负载不均会让卡空等。若模型单卡已放得下，独立副本、量化后单卡、TP 和 PP 都应在相同流量下比较。分离式 serving 若迁移 KV，则又是不同的通信路径。

再算一个传输下限：假设边界有效带宽为 12 GiB/s，且忽略启动延迟。上表的 48 KiB 至少约需 3.8 微秒，192 MiB 约需 15.6 毫秒。前者很可能由启动和同步开销主导；后者已经不能忽略字节量。这个带宽是算例假设，不是 PCIe 或某款显卡的实测值。

PP 可以减少跨卡协作的频率，但也带来顺序依赖：只有一条请求、一个待生成 token 时，后面的 stage 仍要等前面的结果。增加 microbatch 能让不同请求同时占用不同 stage，却需要足够流量，也可能增加排队和缓存占用。因此“没有 NVLink”只是选型条件之一，还要看链路拓扑、prefill 长度、并发与延迟预算。

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
