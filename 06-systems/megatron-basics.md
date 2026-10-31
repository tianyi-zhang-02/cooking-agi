# 从一次更新到 Megatron：多卡训练的基础

**中文** · [English](megatron-basics.en.md)

先想象只有一个进程、一张 GPU。程序读入一批 token，模型算出预测，再与目标比较得到损失。接着，程序通过反向传播求出梯度，由优化器更新参数。Megatron 要解决的核心问题之一是：**当模型太大、单卡装不下或跑不快时，怎样让多张卡合作完成这些计算？**

我们先手算一次更新，再把同样的计算分到多张卡上。文中的显存估算、进程分组和批大小都是为讲解而设的例子，不是可以直接照用的训练配置，也不是实测结果。资料核对日期：2026-10-30。

## 一次参数更新要做什么 {#one-update}

### 先用一个参数算完一轮

先把 Transformer 放到一边。假设模型只需要学习一个数 $w$，输入是 $x=3$，预测规则是 $\hat y=wx$。目标值为 $y=10$，当前参数为 $w=2$。

**前向计算**得到 $\hat y=2\times3=6$。为了衡量预测与目标相差多少，我们选择平方误差的一半作为损失（loss）：

$$
L=\frac12(\hat y-y)^2=\frac12(6-10)^2=8.
$$

**反向传播**回答的是：参数稍微变化时，损失会怎样变化？按链式法则，这次的梯度为：

$$
\frac{\partial L}{\partial w}=(wx-y)x=(6-10)\times3=-12.
$$

梯度为负，说明在当前位置略微增大 $w$ 可以降低损失。接下来，用学习率为 0.1 的普通梯度下降做一次**优化器更新**：

$$
w_{\rm new}=w-0.1\frac{\partial L}{\partial w}=2-0.1\times(-12)=3.2.
$$

更新后预测为 9.6，损失为 0.08。这是一个能手算的教学例子。大模型的参数更多，损失和优化器也更复杂，但仍然要依次完成“预测 → 衡量误差 → 求梯度 → 更新”。

### 把术语放回刚才的过程

| 词 | 在做什么 | 常见误解 |
| --- | --- | --- |
| 张量 | 带有形状（shape）、数据类型（dtype）和所在设备（device）信息的多维数组 | 形状相同不表示含义相同 |
| 参数 | 训练中需要学习和更新的张量，例如线性层权重 | checkpoint 文件大小不等于训练时的显存峰值 |
| 前向计算 | 用参数把输入变为 logits 等输出 | 训练前向还可能保存反向需要的中间值 |
| 损失函数 | 根据预测和目标计算损失 | padding 位置与回答 token 通常不能按同一种目标计算 |
| 反向传播 | 沿计算图求参数梯度 | 得到梯度时，参数还没自动更新 |
| 优化器更新 | 根据梯度与历史状态修改参数 | Adam 等还要保存额外状态 |

刚才的 $w$ 是参数，$-12$ 是它在这个样本上的梯度。两者要分开看：算出梯度，并不等于改了参数，还得由优化器按规则修改参数。Adam 等优化器还会利用过去梯度的统计量，所以恢复训练时通常也要恢复优化器状态。

在 PyTorch 中，`loss.backward()` 沿计算图累积梯度，`optimizer.step()` 才按优化器规则更新参数；什么时候清零梯度，则由训练循环安排。可以对照[官方 autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)，把刚才的手算写成代码。

### 从一个数，走到语言模型的张量

语言模型通常把形状为 `[B,S]` 的 token 编号变成 `[B,S,H]` 的隐藏状态，再得到词表 logits `[B,S,V]`。这是逻辑形状；如果采用样本拼接（packing）、融合损失计算或分片实现，未必真的会在显存中分配完整的三维 logits。

这里 B 是一批数据中的序列数，S 是序列长度，H 是每个位置的隐藏向量维度，V 是词表大小。Logits 是模型为各个候选 token 给出的未归一化分数，经过 softmax 才得到相应概率。

取 B=2、S=1024、H=4096、V=128000，假设 BF16 每个元素占 2 字节，FP32 每个元素占 4 字节。按“元素数 × 每元素字节数”计算，再除以 `2^20`，就得到以 MiB 为单位的大小：

| 张量 | 字节数计算 | 占用 |
| --- | --- | --- |
| BF16 隐藏状态 `[B,S,H]` | `2 × 1024 × 4096 × 2` | 16 MiB |
| BF16 完整 logits `[B,S,V]` | `2 × 1024 × 128000 × 2` | 500 MiB |
| FP32 完整 logits `[B,S,V]` | `2 × 1024 × 128000 × 4` | 1000 MiB |

只把形状中的 H 换成更大的 V，同样用 BF16，张量就从 16 MiB 增到 500 MiB。这还只是一个张量的大小，接下来还得算上同一时刻保存在显存里的其他状态。

### 显存里还有什么

| 状态 | 为什么需要它 | 估算显存时要问什么 |
| --- | --- | --- |
| 参数 | 执行模型前向 | 是完整副本，还是本地分片？ |
| 激活 | 某些前向中间值供反向使用 | 哪些被保留，哪些会重算？ |
| 梯度 | 告诉优化器如何更新参数 | 用什么数据类型，如何累积和分片？ |
| 优化器状态 | 保存更新规则需要的历史信息 | 是否保存梯度矩（moments）、维护高精度主权重（master weights），或将状态卸载（offload）到 CPU 等位置？ |

除此之外，还有临时张量、通信缓冲区（buffer）、算子工作区（workspace），以及内存分配器保留的空间。**权重文件的大小只覆盖其中一部分。** 推理一般不做反向传播和优化器更新；自回归生成则会保存 KV 缓存等运行状态，因此要单独估算显存占用。

接下来讨论多卡时，可以一直问自己：这次分到不同卡上的，是数据、计算，还是上面的某种状态？

## 一张卡和一个 rank 不是同一个概念 {#processes}

节点（node）在这里指一台机器，GPU 是设备，进程（process）是运行中的程序，rank 则是进程在分布式进程组中的编号。训练时常用一张 GPU 对应一个进程，但这只是部署时的一种选择。同一个进程还可能同时属于 TP、PP、DP 等多个通信组，在每个组里都有自己的编号。`rank=0` 并不一定指整个集群的“第一块物理 GPU”。

先弄清每个进程持有哪些参数、处理哪批数据，再谈并行度。通信时怎样使用 rank、怎样排列数据，见 [NCCL 官方说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

## 多张卡怎样分工 {#parallelism}

### 模型放得下，先看数据并行

数据并行 DP 的做法是：让模型的多份副本同时处理不同数据。两张卡都持有模型，一张读样本 A，另一张读样本 B；更新时同步对应梯度，让副本继续保持一致。PyTorch 的 DistributedDataParallel，简称 DDP，是常用实现之一。

沿用前面的单参数模型，让两个副本都从 $w=2$ 开始，使用同一个平方损失，但各读一个样本：

| 副本 | 样本 | 预测 | 本地梯度 |
| --- | --- | --- | --- |
| A | $x=3,\ y=10$ | $wx=6$ | $(6-10)\times3=-12$ |
| B | $x=2,\ y=2$ | $wx=4$ | $(4-2)\times2=4$ |

如果目标是最小化两条样本损失的平均值，两个副本就用平均梯度和学习率 0.1 更新：

$$
\begin{aligned}
g&=\frac{-12+4}{2}=-4,\\
w_{\rm new}&=2-0.1\times(-4)=2.4.
\end{aligned}
$$

如果各用各的本地梯度，A 的参数会变成 3.2，B 的参数会变成 1.6；同步梯度后，两者都得到 2.4，与单卡优化这两条等权样本的平均损失一致。手算中的“求和后除以 2”是在说明怎样归一化；DDP 默认同步梯度时已经取了平均，不要再不加区分地除一次。见 [DDP 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

这样可以增加单位时间内处理的数据量，但每张卡仍要放得下自己的模型状态。如果参数和优化器状态太大，就需要进一步考虑 FSDP 或 ZeRO 等分片方式。FSDP2 可以把参数等状态分给不同进程，在计算需要某组参数时再把它们聚合起来；参数何时聚合、何时释放，详见[训练框架](training-frameworks.md#fsdp-lifecycle)。

### 一层太大与层数太多，是两种切法

张量并行（tensor parallelism，TP）把一层里的运算分给多张卡。例如，把一个线性层的输出列分给两张卡计算，再根据后续计算的需要通信。两张卡分担的是同一层运算。

流水线并行（pipeline parallelism，PP）则把不同层放到不同阶段（stage）。假设四个 Transformer blocks 分成两段：stage 0 算前两个，再把中间激活交给 stage 1 算后两个；反向传播时，梯度沿相反方向返回。这时还要安排好微批次，让多个阶段尽量都有计算可做。

### 把常见切法放在一起对照

| 方法 | 拆什么 | 各卡的工作与代价 |
| --- | --- | --- |
| DP / DDP | 不同数据 | 各模型副本处理不同数据，更新前同步梯度；各自保存模型状态 |
| FSDP / ZeRO | 部分或全部训练状态 | 按所用策略切分参数、梯度和优化器状态；计算时通过通信取得所需状态 |
| TP | 一层里的张量计算 | 例如各卡计算线性层的不同列；后续可能需要拼接结果，或按指定运算合并结果 |
| PP | 层或计算阶段 | 阶段之间传激活，反向时传对应梯度；流水线需要填充和排空 |
| CP | 长序列的上下文 | 各卡负责序列的一部分，但注意力计算仍需要其他分片的信息 |
| EP | MoE 的专家 | token 按路由送到专家，涉及分发、返回与负载均衡 |

理解前四行后，再看上下文并行（context parallelism，CP）与专家并行（expert parallelism，EP）。CP 用于长序列，切开后的各部分仍需共同完成注意力计算；EP 用于混合专家模型（MoE），把 token 送往相应专家处理。TP 中还有序列并行（sequence parallelism），它有自己的激活切分方式，详见[张量并行](tensor-parallel.md)。

TP 和 FSDP 都可能使用 all-gather，但用了相同的通信操作，不代表切分的是同一种东西：FSDP 通常在需要时聚合参数，用来计算本进程的数据；TP 则让多张卡分担同一层运算。

具体机制可继续查阅 [Megatron 并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、[CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)、[MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) 与 [PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。哪些策略可以组合、组合后有什么限制，要看具体版本；不能据此表判断某个版本支持哪些组合。

## 用八张卡看清进程怎样分组 {#eight-gpus}

这里只讨论稠密模型（dense model）。设 TP=2、PP=2、DP=2，不启用 CP/EP，共有 $2\times2\times2=8$ 个进程。下面的编号是为讲解而指定的，不是 Megatron 默认的 rank 排列：

```text
数据副本 A： stage 0 [0,1]  →  stage 1 [2,3]
数据副本 B： stage 0 [4,5]  →  stage 1 [6,7]

TP：同一个方括号里的两个 ranks 合作算一层
PP：0↔2、1↔3，以及 4↔6、5↔7 传递阶段间数据
DP：0↔4、1↔5、2↔6、3↔7 同步对应参数的梯度
```

读图时，先沿数据副本 A 从左向右看：进程 0、1 合作算前一段，2、3 合作算后一段。这四个进程合起来完成一份模型的计算。副本 B 则用另外四个进程处理另一批数据。图中的 0–7 都是全局编号，方便识别同一个进程；它在某个子组内的 rank 可能不同。

再竖着看：0 与 4 负责两个副本中对应的模型部分，所以需要同步对应梯度。0 与 2 则属于同一个副本的前后两个阶段，传的是阶段间的数据。**八张卡之间，不是每条连线都在传同一种数据。**

### 一次更新到底处理多少条序列

每个 DP 副本一次输入 2 条序列，构成一个微批次，累积 4 个微批次的梯度后再更新。把两份输入记作 A1–A8 和 B1–B8，沿着图看它们怎样参与计算：

| 进度 | 副本 A：`[0,1] → [2,3]` | 副本 B：`[4,5] → [6,7]` |
| --- | --- | --- |
| 第一个微批次 | 处理 A1、A2，向后一段传激活 | 处理 B1、B2，向后一段传激活 |
| 该微批次反向 | 激活梯度由 `[2,3]` 返回 `[0,1]`；各段算参数梯度 | 激活梯度由 `[6,7]` 返回 `[4,5]`；各段算参数梯度 |
| 四个微批次累积完毕 | A1–A8，共 8 条序列 | B1–B8，共 8 条序列 |
| 本次更新 | 与 B 的对应分片同步梯度，归一化并更新 | 与 A 的对应分片同步梯度，按同一规则更新 |

所以，全局批大小（global batch）为 $2\times2\times4=16$ 条序列：每个微批次的序列数 × DP 副本数 × 累积微批次数。TP 和 PP 是让多张卡合作计算这些序列，不能再各乘一次得到 64。如果序列长度可变，同样是 16 条序列，有效 token 数和损失权重也未必相同。表格只说明数据和梯度去了哪里；通信可以与反向计算同时进行，不必等到最后一行才开始。

PP 把数据分成微批次后，stage 0 处理后续数据时，stage 1 可以同时处理前面的数据。例如，stage 1 处理 A1、A2 时，stage 0 可以继续处理 A3、A4。微批次太少，某些阶段可能长时间没有计算可做；太多则会影响内存占用和调度开销。不能仅凭 PP=2 就推断训练快两倍。上面计算进程数和批大小的简单乘法，也不能直接套用到所有 MoE/EP 拓扑上。

## Megatron-LM、Core、Bridge 分别做什么 {#megatron-family}

| 名称 | 主要做什么 |
| --- | --- |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 大规模 Transformer 训练仓库，包含训练入口与相关实现 |
| [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html) | 提供可复用的模型、并行、调度、分布式优化器等训练组件 |
| [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/) | 基于 Core 的训练与模型适配工具，包括 Hugging Face 转换等能力 |
| [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html) | 组织 rollout 采样、奖励计算和训练等后训练流程，可以使用 Megatron 训练 |
| vLLM / SGLang | 这里主要看它们生成采样轨迹时，怎样接收训练端的权重 |

对照八卡图来看，Core 提供并行训练组件，Megatron-LM 组织训练，集群工具负责分配设备。Core 本身不是一种新的语言模型，Megatron-LM 也不是用来替代 Kubernetes 的。训练和推理可以采用不同的分片方式，因此把权重从训练端传给推理端时，可能还需要重新分片、转换和同步，不能只复制一个 Python 对象。模型如何转换，见 [Bridge 转换指南](https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-guide.html)。

Megatron 的 distributed optimizer 也不能直接等同于完整参数分片：它主要把优化器状态等分给不同的 DP 进程，并处理更新时的通信；每张卡上长期保存哪些参数、梯度怎样分布，还要看具体配置，不能看到名称就把它当成 PyTorch FSDP2。见[官方说明](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)。

## 第一轮练习，不需要租八张卡 {#exercise}

重新画出八卡分组，标明每条箭头传的是激活、梯度还是参数，并解释为什么四个进程共同处理 A1、A2，仍然只算两条序列。再用小模型检查张量的 shape、dtype、device，以及前向、反向和更新前后的状态：什么时候得到梯度，什么时候参数才改变？弄清这些之后，再用少量 GPU 验证归一化和同步。没有实际运行的配置，就写明只是根据文档的理解或准备尝试的实验。

继续读：[训练框架与 FSDP2](training-frameworks.md)、[张量并行](tensor-parallel.md)、[GPU 与集群调度](execution-boundaries.md#scheduling)，再接[RL 系统](rl-systems.md)。
