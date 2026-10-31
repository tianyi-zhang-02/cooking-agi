# 训练框架怎么选：从 DDP 到 FSDP2

**中文** · [English](training-frameworks.en.md)

> 核对：2026-10-16 · FSDP2 机制最初对照 PyTorch v2.8.0 源码写成；2026-10 对照 v2.14.1（2026-09-30 发布）源码，分组、DTensor 分片和 `reshard_after_forward` 的行为没有变。本页没有运行多卡性能测试。

换成 FSDP2，训练就会更快吗？先分清两个问题：模型能在一张卡上训练，只是太慢；还是连一张卡都放不下。前者可以让更多卡同时处理不同数据，后者还要想办法减少每张卡保存的状态。选框架之前，先找出自己卡在哪里。

如果刚接触多卡，先读[数据并行、张量并行与流水线](distributed-training.md)。这篇接着解释：同样写着“多卡训练”，不同工具到底接管了哪部分工作。

## 这些名字不在同一层 {#layers}

先跟一小批序列走完一次更新：数据加载器把输入交给模型，模型模块算出预测，损失函数把预测和目标变成一个标量，自动微分据此计算梯度，优化器再用梯度和自身状态更新参数。训练循环把这些步骤串起来；分布式策略则安排状态放在哪里、何时通信。换其中一层，不等于替换整套训练流程。

这几个接口为什么分开，可以对照 [CMU《Neural Network Library Abstractions》第 14–16、19–20 页](https://dlsyscourse.org/slides/7-nn-framework.pdf#page=14)，重点看模块、损失、优化器与数据加载之间的关系。

| 层次 | 常见选择 | 它负责什么 | 不替你决定什么 |
| --- | --- | --- | --- |
| 模型与目标 | PyTorch、Transformers、自定义损失 | 前向计算、可训练参数、反向图 | 标签是否可信、目标是否符合需求 |
| 训练循环 | 自定义循环、Trainer / TRL | 取数据批次、算损失、更新、验证 | 用什么数据，如何分训练与测试 |
| 分布式策略 | DDP、FSDP2、DeepSpeed ZeRO；更大规模还可能用张量并行（TP）/ 流水线并行（PP） | 状态怎么存、计算与通信怎么分 | 更多卡是否值得、结果是否正确 |
| 启动与配置 | torchrun、Accelerate | 进程、设备、后端配置 | 不等于另一种模型架构 |
| 后训练编排 | NeMo RL、verl 等 | 把训练、生成、奖励与权重同步接起来 | 不自动解决奖励或离策略（off-policy）问题 |
| 生成引擎 | vLLM、SGLang | 评估或采样生成（rollout）时高效生成 | 不代替训练器执行优化器更新 |

同一套实验里可以同时有 TRL、FSDP2 和 vLLM：分别管循环、分片与生成。先画清调用关系，再决定要装哪些组件。实践中的选择见 [SFT 与框架](../practice/post-training/training-plan.md)。

## DDP 已经够用时，别急着换 {#ddp}

DDP 是分布式数据并行（Distributed Data Parallel）；FSDP 是全分片数据并行（Fully Sharded Data Parallel）。两者都可以让各卡处理不同数据，关键差别之一是模型状态是否长期完整复制。

在常见的 DDP 路径里，每个训练进程（用 rank 标识）保留完整模型，处理不同数据，再同步梯度。假设单卡上的模型、优化器状态和一个合理的微批次（microbatch）都放得下，DDP 往往是容易检查的起点。

两张卡各处理 4 条序列，不是把同一条序列拆成两半。若累积 8 次再更新，一次更新包含 $2\times4\times8=64$ 条序列；但有效 target token 数还取决于长度、padding 和损失掩码。只对齐“64 条”，不一定对齐了训练目标。

换 FSDP 主要是为了降低复制的状态量。它需要额外调度参数通信，**不是 DDP 的无条件加速版**。如果瓶颈是 CPU 解码数据、远端存储或过多填充，先换分片方式可能几乎没用。

## FSDP2：需要这一层时，再拼出它的参数 {#fsdp-lifecycle}

每张卡只存一份分片，怎么计算完整的一层？想象 4 张卡各保管一份参数分片。算某个 Transformer block 之前，大家先收集并拼齐（all-gather）这个 block 需要的参数；算完后可以释放完整副本。反向时再取得需要的参数，通过归约分片（reduce-scatter）汇总梯度后，各自保留负责的部分。

<figure class="worked-update" lang="zh-CN">
<ol>
<li><small>01 / 前向</small><strong>收集并拼齐参数</strong><span>每个进程算自己的数据批次，但计算当前 block 时要拿到它的完整参数。</span></li>
<li><small>02 / 反向</small><strong>重新收集，再归约梯度</strong><span>若前向后已释放完整参数，反向前再取一次；归约分片后只留下本地梯度分片。</span></li>
<li><small>03 / 更新</small><strong>本地优化器更新分片</strong><span>下一次拼齐的是更新后的参数。样本仍按数据并行分配，并不是每卡只算固定几层。</span></li>
</ol>
<figcaption>示意采用 reshard_after_forward=True。保留参数、预取和混合精度会改变实际内存与通信时间线。</figcaption>
</figure>

[PyTorch v2.14.1 的 fully_shard](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/distributed/fsdp/_fully_shard/_fully_shard.py)把每次调用覆盖的参数组织成通信组，并用 DTensor 表示分片。通常先对各个 block 调用，再处理根模块里剩余的参数。只对最外层调用，会把大量参数放进一个组，不利于降低临时峰值，也更难让通信与计算重叠。

这里的“组”是 **FSDP 的参数通信组**，不是多头注意力的头分组，也不是优化器的超参数分组。

## 为什么不能直接把显存除以卡数 {#memory}

做一个明确的手算：10 亿参数，假设常驻参数和梯度都是 FP32，Adam 的一阶、二阶矩估计也用 FP32，不另存主权重副本。只数这几项，就是 $10^9\times(4+4+8)=16$ GB。理想地分到 4 个进程，各是 4 GB；这里 GB 是十进制，不是 GiB。

这是否意味着 4 GB 显存就够了？还不够：训练峰值还要计入激活、临时完整参数、通信缓冲区、算子工作区和分配器开销。更准确的理解是：

$$M_{\text{peak}}=\max_t\left(M_{\text{persistent}}(t)+M_{\text{activation}}(t)+M_{\text{temporary}}(t)\right).$$

继续用上面的 10 亿参数模型代入计算，以下数值都是假设。模型有 20 个同样大的 block，每个 5,000 万参数。反向算到某个 block 时，每卡常驻 4 GB 分片。这个 block 的完整 FP32 参数刚拼齐，占 $5\times10^7\times4=0.2$ GB；它的完整梯度在归约分片之前又占 0.2 GB。

如果此时还保留着 3 GB 激活和 0.5 GB 工作区，总占用就是 $4+3+0.2+0.2+0.5=7.9$ GB，已经比只数常驻分片的 4 GB 高出将近一倍。$M_{\text{peak}}$ 则要取所有时刻中的最大值。

不要把不同时间各自出现的最大值随便相加，当成精确峰值；也不要只数常驻分片，就认定能训练。先记录时间线，再看哪几项同时存在。混合精度下参数、梯度和通信可能采用不同的 dtype，需要重新列账。

| 想省哪部分 | 可以试什么 | 要付出的代价 |
| --- | --- | --- |
| 参数、梯度、优化器状态的复制 | FSDP / ZeRO | 多了状态收集和通信调度 |
| 前向保留的激活 | 激活重计算（activation checkpointing） | 反向时重算，增加计算 |
| 填充 | 长度分桶、合适的 packing | 要保持样本边界、注意力掩码与目标权重 |
| GPU 上的常驻状态 | CPU 卸载（offload） | 主机内存与传输成本，可能拖慢每步 |
| 单个计算块仍太大 | 更细的分组，或 TP 等模型并行 | 小消息变多，或每层增加通信 |

FSDP 不会自动把长序列的激活除以 GPU 数。增加数据并行进程，也不会自动减少每个进程在微批次大小不变时的激活占用。

还要区分显存容量与访存开销：[CMU《GPU Acceleration》第 10–17 页](https://dlsyscourse.org/slides/12-gpu-acceleration.pdf#page=10)用内存层次和矩阵乘法分块（tiling）讲单卡内的数据复用；这与跨卡分片状态是不同层次的优化。

## 分得太粗、太细，都可能吃亏 {#grouping}

沿用前面的 20 个 block：是一次拼齐整个模型，还是用到哪个 block 才拼哪个？一个大组通信次数少，但临时拼齐时占用大；很多极小组更灵活，却容易花大量时间等待小消息。按 Transformer block 分组是值得测试的起点，不是适合所有模型的定律。

`reshard_after_forward=True` 释放前向后的完整参数，用反向前的再次收集换内存；`False` 保留它们，省去这次收集，但占用更久。**根模块的默认行为可能与内部 block 不同**，所以这里主动写出假设，不把默认值当成统一规律。[官方 FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

预取（prefetch）可以在算当前块时取下一块，前提是还有通信和内存余量。预取过早，可能让更多完整参数同时留在显存；网络已经饱和时，也不会凭空多出带宽。

## DDP、FSDP、ZeRO、Megatron 怎么比较 {#choices}

| 眼前的条件 | 值得先验证的路径 | 要额外检查 |
| --- | --- | --- |
| 每卡放得下，先扩数据吞吐 | DDP | 数据加载、全局 token 权重、梯度同步 |
| 状态复制放不下，想保留 PyTorch 模型写法 | FSDP2 | 分组、数据类型、峰值、保存与恢复 |
| 已有成熟 DeepSpeed 配置或卸载路径 | ZeRO 对应阶段 | 实际分片范围、框架版本与检查点（checkpoint）兼容性 |
| 单层计算或模型规模要求更复杂并行 | TP / PP，及 Megatron 等实现 | 模型支持、拓扑、通信与数据布局 |

[ZeRO 论文](https://arxiv.org/abs/1910.02054)讲状态分片；[Megatron-LM](https://arxiv.org/abs/1909.08053)讲 Transformer 内部的并行组织。它们解决的问题有交集，但不能只按工具名做一张互斥排名表。真实栈可以组合数据、张量和流水线并行。

## 怎样证明换框架没有悄悄换目标 {#verify}

先在固定小样本上比较一次更新，再谈吞吐：输入顺序、有效 token、掩码、梯度归一化、梯度裁剪和优化器配置都要对齐。不同并行归约顺序可能有数值误差，用事先声明的容差比较，不承诺逐比特相同。

然后测预热后的有效 target tokens/s、每步耗时分布、峰值显存和通信等待。GPU 利用率很高，也可能是在重算、填充或通信上忙；MFU 和 HFU 的分子为什么不同，见[性能：MFU 与 HFU](../12-infra/performance.md#mfu)。更有用的问题是：**同样的质量目标，多久能完成一次可靠实验？**

最后故意中断一次。只恢复模型权重，不等于恢复训练；还要检查优化器、学习率调度器（scheduler）、随机数生成器（RNG）、数据游标和更新步。分片检查点能否换卡数加载，需要实际验证所用保存格式与加载器；能加载也不等于重放同一条训练轨迹。接着看[保存与恢复](../practice/post-training/checkpoint-and-resume.md)。

想看这些组件在项目里的位置，读[完整 ML 系统](../practice/ml-system/README.md)；想看同一模型训练完怎么接生成服务，读 [vLLM 与 SGLang](inference-frameworks.md)。
