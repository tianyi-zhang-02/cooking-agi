# Module 06 · GPU 集群与平台工程

**中文** · [English](06-gpu-platforms.en.md) · [返回 AI Infra](../README.md)

> 阅读时间：约 5 分钟 · 难度：Intermediate · 时效性：Evolving · 最近审阅：2026-08

## 核心问题

单个脚本成功运行只是开始。平台工程要让不同用户、模型和优先级在共享 GPU 集群中可重复、可观测、可恢复地运行，并控制资源浪费和故障影响范围。

## 学习目标

- 理解从驱动到容器再到训练/推理进程的节点软件栈；
- 能解释 Kubernetes device plugin 或 Slurm GPU scheduling；
- 理解 queue、quota、priority、preemption 和 gang scheduling；
- 能设计 job lifecycle、checkpoint 和 retry policy；
- 能定义 GPU 平台的指标、SLO 和告警。

## 核心笔记

### 一个 GPU 节点的栈

```text
GPU firmware / hardware
→ host driver
→ CUDA runtime and libraries
→ container runtime and device exposure
→ framework
→ training or serving process
```

版本兼容问题可能出现在任何相邻层。容器封装了用户空间依赖，但不能完全替代兼容的 host driver。

### 资源发现与调度

调度器需要知道节点有哪些 GPU、显存多大、健康状态、拓扑怎样，以及 workload 需要几张卡。多卡任务通常要求 gang scheduling：资源一起满足后才启动，避免部分 worker 占着 GPU 等其他 worker。

拓扑感知调度需要尽量把高频通信的 GPU 放在更快的互连上，并考虑 CPU affinity、NUMA、NIC 和 GPU 的位置关系。

### 共享和隔离

常见方式包括独占 GPU、MIG、time slicing 或应用级多租户。共享提高利用率，却会增加性能干扰、显存隔离、故障域和安全复杂度。

平台必须明确：什么资源可以超卖、谁可以抢占谁、交互服务和批训练如何隔离，以及过载时优先保护什么。

### Job lifecycle

```text
submit → validate → queue → schedule → initialize
→ run → checkpoint → finish / retry / fail → retain artifacts
```

可靠重试需要区分瞬时基础设施故障、确定性代码错误、数据错误和 OOM。无条件重试可能只是在重复浪费 GPU。

### 可观测性

至少需要三类信号：

- **Metrics**：GPU utilization、显存、功耗、温度、网络、queue time、失败率；
- **Logs**：scheduler、节点、容器、框架和用户进程日志；
- **Traces / timelines**：跨 worker 的计算、通信、I/O 和等待关系。

可观测性要能从“任务慢了”下钻到具体节点、rank、kernel、collective 或存储请求。

## 关键计算

集群使用率至少要区分：

```text
allocation utilization = allocated GPU time / available GPU time
active utilization = useful busy time / allocated GPU time
```

只看 allocation utilization 会把占着 GPU 但没有计算的任务也算作成功利用。

容量规划还应记录：

- queue time 分布；
- 不同 GPU 类型的需求与碎片；
- 失败和重试消耗的 GPU-hours；
- checkpoint 与数据读取带宽；
- 服务的峰值与安全余量。

## 动手验证

1. 画出一个训练 job 从提交到结束的状态机。
2. 为单卡、八卡和跨节点任务设计资源请求。
3. 定义 OOM、节点丢失、网络超时和代码异常的不同重试策略。
4. 为训练和在线推理分别写一组 SLO 与 dashboard 指标。
5. 设计一次 GPU 节点维护：drain、checkpoint、迁移和恢复怎样完成？

## 常见误区

- GPU 被分配不等于 GPU 在做有效工作；
- 容器可运行不等于驱动、CUDA 和框架完全兼容；
- 自动重试不是容错设计的替代品；
- 追求平均利用率可能伤害 P99 延迟或高优先级任务；
- 没有统一 job identity 和版本信息，日志再多也难以关联。

## 学习检查

- 为什么分布式任务需要 gang scheduling？
- device plugin 解决什么，不解决什么？
- 什么情况下适合共享 GPU？
- 怎样区分 allocated、busy 和 useful GPU time？
- 平台怎样让一次训练运行可以被精确复现？

下一步：[Module 07 · 数据、Eval 与学习闭环](07-data-eval-learning-loop.md)
