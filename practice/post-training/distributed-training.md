# 多卡怎么选，先看卡在哪里

**中文** · [English](distributed-training.en.md)

> 原创教学项目 · 核对：2026-10-09。显存与 batch 数字用于手算，不是 GPU benchmark；没有执行多机训练。

单卡能启动，不代表适合长跑；单卡 OOM，也不代表必须上张量并行。先把问题拆成两类：**放不下**，还是**跑不完**。前者需要找显存占用，后者要看时间花在哪。

## 先量一个真实 batch

不要只跑一条特别短的样本。用实际长度分布，至少覆盖短、中、长几档，经过 warmup 再记录前向、反向、优化器更新和保存。许多优化器状态第一次 `step` 才分配，刚加载完的显存不能代表训练峰值。

| 观察到的情况 | 先查什么 | 可能有效的改动 |
| --- | --- | --- |
| 模型加载就 OOM | 权重、初始化副本、设备放置 | 量化基座、分片初始化，或更小模型 |
| 长样本反向时 OOM | 激活、attention 实现、临时缓冲 | 小 microbatch、activation checkpointing、合理长度预算 |
| 第一次 optimizer step OOM | 梯度、优化器状态、master weights | LoRA、状态分片，必要时 offload |
| GPU 周期性空闲 | 数据读取、CPU tokenize、同步、checkpoint I/O | 缓存预处理、检查 profile，而不是先加卡 |

activation checkpointing 是少存激活、反向时重算；磁盘 checkpoint 是保存训练状态以便恢复。名字像，解决的不是同一个问题。

## 显存先算下界，再测峰值

一个简化的全参 Adam 预算：BF16 权重 2 bytes、BF16 梯度 2 bytes、FP32 master weights 4 bytes、两个 FP32 动量合计 8 bytes，每参数约 16 bytes。这是明确假设下的估算；具体实现可能保留 FP32 参数或梯度，也可能没有独立 master copy。

若参数量是 0.6B，仅上述状态就是 $0.6\times10^9\times16=9.6\times10^9$ bytes，约 8.94 GiB。**这不包括激活、logits、通信缓冲、allocator 预留和碎片。** LoRA 的可训练状态与此不同，不能直接套全参公式。

因此“小模型为什么用了很多卡”不能只看参数量。长上下文、批量吞吐、迭代时限都会影响资源；但这些也不能反过来证明很多卡就合理。需要拿相同任务的耗时、峰值显存和成本来解释。

## DDP、FSDP、ZeRO、TP、PP 放在一起看

| 方式 | 拆什么 | 适合先考虑的情况 | 主要代价 |
| --- | --- | --- | --- |
| DDP | 数据；每卡保留完整训练副本 | 单卡放得下，希望并行处理更多数据 | 状态仍完整复制，梯度同步有成本 |
| FSDP2 | 参数、梯度、优化器状态 | 状态太大，但分层聚合可承受 | 参数 all-gather 与梯度 reduce-scatter；峰值不等于总状态除卡数 |
| ZeRO 1 / 2 / 3 | 依次增加优化器、梯度、参数的分片 | 需要逐步增加状态节省，且训练栈支持 | 更深分片通常带来更多状态交换；offload 另算传输成本 |
| TP | 一层里的张量计算 | 单层很大，设备互联足够好 | 层内频繁通信，通常更依赖高速互联 |
| PP | 不同层放到不同阶段 | 按层切分模型，配合合适的 microbatch 调度 | 流水线气泡、阶段负载不均、激活传输 |

具体分片行为见 [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) 与 [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)。旧 FSDP1 配置不能当作 FSDP2 接口直接照抄。实际方案可以组合，但先证明一个维度有必要，再增加另一个。

对于这个小型 LoRA 项目，如果单卡已经放得下，先把单卡实验做好；需要更多吞吐时再测 DDP。若全参状态放不下，再比较 FSDP / ZeRO。若主要是长序列激活占用，先查长度、attention 与重算；单靠参数分片可能帮助有限。

## 加卡以后，batch 和 loss 有没有偷偷变

在无 packing、固定样本数的简单设置中：

$$
B_{\mathrm{update}}=B_{\mathrm{micro}}\times A\times D.
$$

$B_{\mathrm{micro}}$ 是每个数据并行副本每次处理的样本数，$A$ 是梯度累积次数，$D$ 是数据并行度。8 张卡做 TP=2、DP=4 时，microbatch=2、累积=8，一次更新对应 $2\times8\times4=64$ 条样本，不是 128。这里没有 PP，也没有重复样本或不完整尾批。

样本数相同还不够。两张数据并行卡，第一张有 1 个有效 target，loss 总和 0.2；第二张有 3 个，总和 3。局部均值再平均是 $(0.2+1)/2=0.6$，全局 token 均值却是 $3.2/4=0.8$。

若底层对梯度做普通的 $D$ 卡平均，整次更新共有 $N$ 个有效 target，局部 loss 总和为 $S_{r,a}$，可让每个累积 microstep 反传 $D S_{r,a}/N$。各卡平均、各 microstep 累加之后，得到全局 token 均值的梯度。

**这个推导假设没有其他自动缩放。** 如果 trainer 已处理有效 token、累积除数或分布式缩放，不要再乘一次。$N$ 必须覆盖完整 accumulation window；最后不足窗口的 batch 也要重算。对照单进程拼接 batch 的梯度，才能确认实现，而不是看到 loss 数字接近就算通过。

## 吞吐看有效工作，不只看利用率

假设同一数据、同一有效 token 总量，单卡是 1,000 个有效 target token/s，4 卡是 2,800。加速比为 2.8，扩展效率为 $2.8/4=70\%$；训练更快，但卡时约是原来的 $4/2.8=1.43$ 倍。这组数是教学假设，不是推荐配置。

同时报告全部非 pad token/s、有效 target token/s、step time、峰值显存与保存停顿。SFT 中 prompt 不计 loss，但仍要计算；改变 prompt/answer 比例会影响两种吞吐的关系，不能靠删掉难例“提升效率”。

| 当前优先级 | 如何比较 |
| --- | --- |
| 尽快拿到结论 | 固定任务与验证频率，比端到端时间 |
| 控制成本 | 比同等质量下的卡时与存储 / I/O 成本 |
| 避免 OOM | 检查长度尾部、首个 optimizer step 和保存峰值 |
| 保持训练语义 | 对照样本 ID、有效 token 分母和一小步梯度 |

项目的[标准库检查](code/training_contracts.py)可复算 batch 与 padding；测试还用解析梯度验证不等长分母。它们不是 NCCL / FSDP 性能测试。下一步是[保存与恢复](checkpoint-and-resume.md)：分布式能跑，不代表中断后能接着跑。
