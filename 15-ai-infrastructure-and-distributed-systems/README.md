# AI Infra：让训练和服务稳定运行

**中文** · [English](README.en.md)

模型能跑一次，和服务能接住一整天的请求，是两回事。输入变长、流量突然上来、一个 worker 掉线，都会让原本正常的程序暴露出新的限制。这条路线从这些变化出发，讲资源、调度、分布式执行和恢复怎样配合。

这里不另造一套 vLLM 或 FSDP 教程。[AI 基础设施](../12-infra/README.md)讲硬件与运行环境，[AI 系统](../06-systems/README.md)讲机制和框架；这一组把它们放进**完整服务与训练作业**里，检查容量、接口和出错后的行为。

## 按需要选一条路 {#route}

| 你想解决的问题 | 建议顺序 | 可以检验理解的结果 |
| --- | --- | --- |
| 不知道服务里谁负责哪一步 | [从请求到 GPU](serving-system.md#stack) → [首 token 与完成时间](serving-system.md#timeline) | 画清网关、副本路由、引擎调度和 worker 的边界 |
| 已经用过框架，想补容量与故障 | [KV 预算](serving-system.md#memory) → [排队](queues-and-failures.md#queue) → [故障](queues-and-failures.md#failure) | 改变输入长度或丢失一个副本，说明瓶颈如何变化 |
| 主要关心训练 | [训练到发布](serving-system.md#training) → [训练框架](../06-systems/training-frameworks.md) → [中断恢复](../practice/post-training/checkpoint-and-resume.md) | 列出恢复一次训练需要的状态，不只列权重文件 |
| 想准备设计讨论或动手验证 | [容量小实验](queues-and-failures.md#exercise) → [源码阅读](../practice/runtime-reading/README.md) | 把一个慢请求追到具体队列、缓存或通信，而不是只说“GPU 不够快” |

不熟悉生成过程，先看 [KV cache](../00-foundations/deep-dives/kv-cache-and-inference.md) 与 [prefill / decode](../practice/runtime-reading/prefill-decode.md)。不必先掌握所有 CUDA 优化，也不需要租机器才能复算本组例子。

## 三个容易混在一起的问题 {#boundaries}

| 问题 | 谁通常负责 | 不应该混成什么 |
| --- | --- | --- |
| 这个请求送给哪个副本？ | 网关与路由 | 引擎这一步选哪些 token 计算 |
| 这一轮哪些请求能进入计算？ | 引擎调度器（scheduler） | 集群给服务增加多少资源 |
| 新副本何时能接流量？ | 部署与资源控制 | 进程启动成功就等于模型已加载、预热完成 |

系统中会有多种“调度”。说清调度的对象和时间尺度，比记住一个组件名字更重要。不同团队可能分别维护这些层，接口要能传递截止时间、取消信号、版本与错误原因。

## 继续深入到已有章节 {#deeper}

| 模块 | 继续读 | 要回答的问题 |
| --- | --- | --- |
| 生成引擎 | [vLLM / SGLang](../06-systems/inference-frameworks.md) | 能力相近时，生态、模型支持和部署约束如何影响选择？ |
| 并行执行 | [张量并行](../06-systems/tensor-parallel.md) · [分布式训练](../06-systems/distributed-training.md) | 拆的是参数、计算还是请求？增加了什么通信？ |
| 训练稳定性 | [稳定性排查](../practice/runtime-reading/training-stability.md) | loss 异常来自目标、精度、归一化，还是数据版本？ |
| 集群与性能 | [Kubernetes](../12-infra/kubernetes.md) · [性能分析](../12-infra/performance.md) | 等待发生在哪一层，扩容是否能消除它？ |
| 产品反馈 | [Production ML](../14-production-ml-and-recs/README.md) | 延迟与降级怎样影响真正的曝光、标签和用户结果？ |

这些不是彼此独立的课程。比如训练提高了模型质量，却把输出长度翻倍，服务的容量和延迟预算就需要重算。反过来，为了速度压短输出，也必须重新评估任务是否完成。

五个方向的整体关系见[系统学习路线](../practice/learning-routes.md)。这里先把服务和训练的边界讲清；优化控制、供电散热和产品协作会在后续补充，先从与你手头问题相关的一条读起就好。
