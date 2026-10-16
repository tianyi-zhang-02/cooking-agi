# 训练框架怎么选：从 DDP 到 FSDP2

**中文** · [English](training-frameworks.en.md)

> 核对：2026-10-16 · FSDP2 机制最初对照 PyTorch v2.8.0 源码写成；2026-10 对照 v2.14.1（2026-09-30 发布）源码，分组、DTensor 分片和 `reshard_after_forward` 的行为没有变。本页没有运行多卡性能测试。

模型能在一张卡上训练，只是太慢；和模型连一张卡都放不下，是两个不同的问题。前者可以让更多卡同时处理不同数据，后者还要想办法减少每张卡保存的状态。选框架之前，先分清自己卡在哪里。

如果刚接触多卡，先读[数据并行、张量并行与流水线](distributed-training.md)。这篇接着解释：同样写着“多卡训练”，不同工具到底接管了哪部分工作。

## 这些名字不在同一层 {#layers}

| 层次 | 常见选择 | 它负责什么 | 不替你决定什么 |
| --- | --- | --- | --- |
| 模型与目标 | PyTorch、Transformers、自定义 loss | 前向计算、可训练参数、反向图 | 标签是否可信、目标是否符合需求 |
| 训练循环 | 自定义 loop、Trainer / TRL | 取 batch、算 loss、更新、验证 | 用什么数据，如何分训练与测试 |
| 分布式策略 | DDP、FSDP2、DeepSpeed ZeRO；更大规模还可能用 TP / PP | 状态怎么存、计算与通信怎么分 | 更多卡是否值得、结果是否正确 |
| 启动与配置 | torchrun、Accelerate | 进程、设备、后端配置 | 不等于另一种模型架构 |
| 后训练编排 | NeMo RL、verl 等 | 把训练、生成、奖励与权重同步接起来 | 不自动修好 reward 或 off-policy 问题 |
| 生成引擎 | vLLM、SGLang | 评估或 rollout 时高效生成 | 不代替训练器做 optimizer step |

同一套实验里可以同时有 TRL、FSDP2 和 vLLM：分别管循环、分片与生成。先画清调用关系，再决定要装哪些组件。实践中的选择见 [SFT 与框架](../practice/post-training/training-plan.md)。

## DDP 已经够用时，别急着换 {#ddp}

DDP 是 Distributed Data Parallel（分布式数据并行）；FSDP 是 Fully Sharded Data Parallel（全分片数据并行）。两者都可以让各卡处理不同数据，关键差别之一是模型状态是否长期完整复制。

在常见的 DDP 路径里，每个 rank 保留完整模型，处理不同数据，再同步梯度。假设单卡上的模型、optimizer 和一个合理的 microbatch 都放得下，DDP 往往是容易检查的起点。

两张卡各处理 4 条序列，不是把同一条序列拆成两半。若累积 8 次再更新，一次更新包含 $2\times4\times8=64$ 条序列；但有效 target token 数还取决于长度、padding 和 loss mask。只对齐“64 条”，不一定对齐了训练目标。

换 FSDP 主要是为了降低复制的状态量。它需要额外调度参数通信，**不是 DDP 的无条件加速版**。如果瓶颈是 CPU 解码数据、远端存储或过多 padding，先换分片方式可能几乎没用。

## FSDP2：需要这一层时，再拼出它的参数 {#fsdp-lifecycle}

想象 4 张卡各保管一份参数分片。算某个 Transformer block 之前，大家把这一块需要的参数拼齐；算完后可以释放完整副本。反向时再取得需要的参数，梯度汇总后各自保留负责的分片。

<figure class="worked-update" lang="zh-CN">
<ol>
<li><small>01 / 前向</small><strong>All-gather 参数</strong><span>每个 rank 算自己的 batch，但计算当前 block 时要拿到它的完整参数。</span></li>
<li><small>02 / 反向</small><strong>重新聚合，再归约梯度</strong><span>若前向后已释放完整参数，反向前再取一次；reduce-scatter 后只留下本地梯度分片。</span></li>
<li><small>03 / 更新</small><strong>本地 optimizer 更新分片</strong><span>下一次聚合得到更新后的参数。样本仍按数据并行分配，并不是每卡只算固定几层。</span></li>
</ol>
<figcaption>示意采用 reshard_after_forward=True。保留参数、预取和混合精度会改变实际内存与通信时间线。</figcaption>
</figure>

[PyTorch v2.14.1 的 fully_shard](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/distributed/fsdp/_fully_shard/_fully_shard.py)把每次调用覆盖的参数组织成通信组，并用 DTensor 表示分片。通常先对各个 block 调用，再处理根模块里剩余的参数。只对最外层调用，会把大量参数放进一个组，不利于降低临时峰值和重叠通信。

这里的“组”是 **FSDP 的参数通信组**，不是多头注意力的 head group，也不是 optimizer 的超参数分组。

## 为什么不能直接把显存除以卡数 {#memory}

做一个明确的手算：10 亿参数，假设常驻参数和梯度都是 FP32，Adam 有两个 FP32 moments，不另存 master weights。只数这几项，就是 $10^9\times(4+4+8)=16$ GB。理想地分到 4 个 ranks，各是 4 GB；这里 GB 是十进制，不是 GiB。

训练峰值还要加上激活、临时完整参数、通信 buffer、算子 workspace 和分配器开销。更准确的理解是：

$$M_{\text{peak}}=\max_t\left(M_{\text{persistent}}(t)+M_{\text{activation}}(t)+M_{\text{temporary}}(t)\right).$$

不要把不同时间各自出现的最大值随便相加，当成精确峰值；也不要只数常驻分片，就认定能训练。先记录时间线，再看哪几项同时存在。混合精度下参数、梯度和通信可能是不同 dtype，需要重新列账。

| 想省哪部分 | 可以试什么 | 交换条件 |
| --- | --- | --- |
| 参数、梯度、optimizer 的复制 | FSDP / ZeRO | 多了状态聚合和通信调度 |
| 前向保留的激活 | Activation checkpointing | 反向时重算，增加计算 |
| padding | 长度分桶、合适的 packing | 要保持样本边界、attention mask 与目标权重 |
| GPU 上的常驻状态 | CPU offload | 主机内存与传输成本，可能拖慢每步 |
| 单个计算块仍太大 | 更细的分组，或 TP 等模型并行 | 小消息变多，或每层增加通信 |

FSDP 不会自动把长序列的激活除以 GPU 数。更多数据并行 ranks 也不会自动降低每个 rank 的 microbatch 激活。

## 分得太粗、太细，都可能吃亏 {#grouping}

一个大组通信次数少，但临时拼齐时占用大；很多极小组更灵活，却容易花大量时间等待小消息。按 Transformer block 分组是值得测试的起点，不是适合所有模型的定律。

`reshard_after_forward=True` 释放前向后的完整参数，用反向前的再次 all-gather 换内存；`False` 保留它们，减少一次重聚合，但占用更久。**根模块的默认行为可能与内部 block 不同**，所以这里主动写出假设，不把默认值当成统一规律。[官方 FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

Prefetch 可以在算当前块时取下一块，前提是还有通信和内存余量。预取过早，可能让更多完整参数同时留在显存；网络已经饱和时，也不会凭空多出带宽。

## DDP、FSDP、ZeRO、Megatron 怎么比较 {#choices}

| 眼前的条件 | 值得先验证的路径 | 要额外检查 |
| --- | --- | --- |
| 每卡放得下，先扩数据吞吐 | DDP | 数据加载、全局 token 权重、梯度同步 |
| 状态复制放不下，想保留 PyTorch 模型写法 | FSDP2 | 分组、dtype、峰值、保存与恢复 |
| 已有成熟 DeepSpeed 配置或 offload 路径 | ZeRO 对应 stage | 实际分片范围、框架版本与 checkpoint 兼容性 |
| 单层计算或模型规模要求更复杂并行 | TP / PP，及 Megatron 等实现 | 模型支持、拓扑、通信与数据布局 |

[ZeRO 论文](https://arxiv.org/abs/1910.02054)讲状态分片；[Megatron-LM](https://arxiv.org/abs/1909.08053)讲 Transformer 内部的并行组织。它们解决的问题有交集，但不能只按工具名做一张互斥排名表。真实栈可以组合数据、张量和流水线并行。

## 怎样证明换框架没有悄悄换目标 {#verify}

先在固定小样本上比较一次更新，再谈吞吐：输入顺序、有效 token、mask、梯度归一化、clipping 和 optimizer 配置都要对齐。不同并行归约顺序可能有数值误差，用事先声明的容差比较，不承诺逐 bit 相同。

然后测 warm-up 后的有效 target tokens/s、每步耗时分布、峰值显存和通信等待。GPU utilization 很高，也可能是在重算、padding 或通信上忙。更有用的问题是：**同样的质量目标，多久能完成一次可靠实验？**

最后故意中断一次。只恢复模型权重，不等于恢复训练；还要检查 optimizer、scheduler、RNG、数据游标和更新步。分片 checkpoint 能否换卡数加载，需要实际验证所用保存格式与 loader；能加载也不等于重放同一条训练轨迹。接着看[保存与恢复](../practice/post-training/checkpoint-and-resume.md)。

想看这些组件在项目里的位置，读[完整 ML 系统](../practice/ml-system/README.md)；想看同一模型训练完怎么接生成服务，读 [vLLM 与 SGLang](inference-frameworks.md)。
