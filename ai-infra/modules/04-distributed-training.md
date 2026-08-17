# Module 04 · 分布式训练

**中文** · [English](04-distributed-training.en.md) · [返回 AI Infra](../README.md)

> 阅读时间：约 5 分钟 · 难度：Intermediate · 时效性：Fast-moving · 最近审阅：2026-08

## 这一模块解决什么

当模型、activation 或优化器状态超出单卡容量，训练必须被拆分。本模块关注拆什么、怎样同步，以及新增的通信、调度和故障复杂度是否值得。

## 学习目标

- 能分解训练显存的主要组成；
- 理解 DDP、FSDP2/ZeRO、TP、PP、CP 与 EP；
- 解释 AllReduce、AllGather、ReduceScatter 与 AllToAll；
- 能估算一种并行策略的显存与通信量；
- 能从 timeline 中识别通信等待、pipeline bubble 和 straggler。

## 核心笔记

### 训练显存由什么组成

```text
参数
+ 梯度
+ 优化器状态
+ activation
+ 通信和算子临时 buffer
+ allocator fragmentation
```

只用“参数量 × dtype 大小”会严重低估训练显存。Activation 还随 batch size、sequence length、hidden size、层数和 checkpointing 策略变化。

### 并行方式

| 方式 | 切分什么 | 典型通信 | 主要代价 |
| --- | --- | --- | --- |
| DDP | 数据 | gradient AllReduce | 每卡保留完整模型状态 |
| FSDP2 / ZeRO | 参数、梯度、优化器状态 | AllGather、ReduceScatter | 参数频繁 materialize |
| TP | 层内 tensor / matmul | AllReduce、AllGather | 每层通信、拓扑敏感 |
| PP | 模型层 | point-to-point | bubble、负载平衡 |
| CP / SP | 序列 | gather/scatter 类通信 | attention 通信复杂度 |
| EP | MoE experts | AllToAll | token routing 与负载不均 |

并行策略不是互斥选项。大型训练通常把设备组织成多维 mesh，在不同维度应用不同切分。

### FSDP2 与 DTensor

当前 PyTorch FSDP2 用 `DTensor` 表达分片参数，并通过 `fully_shard` 作用于 module。计算前，hook 会 all-gather 需要的参数；计算后再 reshard，避免完整副本持续占用显存。Device mesh 显式描述并行维度，可以把 data sharding 与 tensor、context 等并行组合起来。

这不只是 API 改名。需要理解参数生命周期、placement、mesh dimension 和 distributed checkpoint 格式。部分 tensor-parallel API 仍处于 experimental 状态，因此真正使用前要重新检查支持范围与 migration guidance。

### 集合通信

- **AllReduce**：每个 rank 获得所有输入的归约结果；
- **AllGather**：每个 rank 收集所有 rank 的分片；
- **ReduceScatter**：先归约，再把结果分片给不同 rank；
- **AllToAll**：每个 rank 向所有其他 rank 发送不同分片；
- **Broadcast**：一个 root 把数据发送给所有 rank。

通信性能取决于消息大小、延迟、带宽、拓扑和算法。单机 NVLink 与跨节点网络应该被视为不同成本层级。

### 计算与通信重叠

如果下一段计算不依赖通信结果，系统可以把 NCCL collective 与其他 kernel 放入不同 stream 并行推进。最终 step time 更接近两者的最大值，而不是简单相加。

但 overlap 不是自动成功：依赖、bucket 大小、stream 调度、链路竞争和 kernel 资源都会影响结果。

### Checkpoint 与容错

分片训练的 checkpoint 也可能是分片的。可靠系统需要记录模型、优化器、scheduler、随机状态、数据位置和并行拓扑，并支持拓扑变化后的恢复或显式拒绝不兼容恢复。

## 需要会算

最基础的模型参数存储：

```text
parameter memory = parameter count × bytes per parameter
```

Ring AllReduce 中，每个 rank 的传输量近似为：

```text
2 × (world_size - 1) / world_size × tensor bytes
```

这是理解带宽成本的近似，实际时间还取决于协议、拓扑、分块和竞争。

扩展效率：

```text
scaling efficiency = single-device time / (device count × distributed time)
```

定义吞吐口径时必须明确 global batch、tokens 和 gradient accumulation。

## 动手验证

1. 跑通两张 GPU 的 DDP，并观察 gradient AllReduce。
2. 用相同模型比较 DDP 与 FSDP2 的峰值显存和 step time，并检查参数的 `DTensor` placement。
3. 画出一次训练 step 的 compute/communication timeline。
4. 改变 bucket size 或 micro-batch，观察 overlap 与 bubble。
5. 人为减慢一个 rank，观察 straggler 怎样拖慢全局同步。

## 常见误区

- 增加 GPU 数量不保证线性加速；
- 模型能放下不代表并行策略高效；
- 平均 step time 会掩盖偶发长尾和 straggler；
- 只看 NCCL 带宽无法解释所有通信等待；
- checkpoint 能写出不代表它能正确恢复。

## 掌握检查

- DDP 为什么不能解决模型单卡放不下的问题？
- FSDP 用什么通信换取显存？
- TP 和 PP 对拓扑的要求有什么不同？
- MoE 为什么大量依赖 AllToAll？
- 怎样证明通信与计算真的发生了 overlap？

当前资料：[PyTorch FSDP2 `fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html) · [PyTorch tensor parallelism](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) · [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

下一步：[Module 05 · LLM 推理](05-llm-inference.md)
