# Module 02 · GPU 编程与算子优化

**中文** · [English](02-gpu-programming.en.md) · [返回 AI Infra](../README.md)

## 这一模块解决什么

GPU 性能不是“开更多线程”这么简单。本模块研究一个 kernel 怎样映射到 GPU，以及计算、访存、同步和调度如何共同决定性能。

## 学习目标

- 能解释 grid、block、warp、thread 和 SM；
- 能画出 register、shared memory、L2 与 HBM 的数据路径；
- 能识别 uncoalesced access、warp divergence 和同步开销；
- 能解释 tiling、occupancy、arithmetic intensity 和 kernel fusion；
- 能用 profiler 判断算子是计算受限还是带宽受限。

## 核心笔记

### SIMT 执行模型

CUDA 以 thread 表达工作，以 block 组织可以协作的线程，再用 grid 表达整个 kernel。硬件通常以 warp 为单位调度线程；NVIDIA GPU 的 warp 通常包含 32 个线程。

同一个 warp 内的线程如果走不同分支，硬件需要分别执行路径并屏蔽不参与的线程。这叫 warp divergence。不同 warp 之间的分支差异通常不会造成同样的问题。

### SM 与线程驻留

一个 SM 同时容纳多个 block 和 warp。每个 block 消耗一定数量的 register 与 shared memory；资源用量过高会减少可驻留 warp 数量。

Occupancy 是活跃 warp 与硬件上限的比例，但 occupancy 高不等于 kernel 一定快。真正目标是有足够的独立工作隐藏延迟，同时避免不必要的数据移动和指令。

### 内存访问

```text
register → shared memory / L1 → L2 → HBM
```

- register 最快，但数量有限；
- shared memory 由同一 block 内线程显式协作使用；
- HBM 容量最大，但访问代价高；
- 相邻线程访问相邻地址，有利于 memory coalescing；
- tiling 把会重复使用的数据暂存在更近的层级。

### Tensor Core 和矩阵乘法

Tensor Core 针对小块矩阵乘累加。高性能 GEMM 通常把大矩阵拆成 thread-block tile、warp tile 和更小的计算 tile，并在 global memory、shared memory 与 register 之间分阶段搬运数据。

### Streams 与异步执行

Kernel launch、内存复制和部分通信可以异步进入 CUDA stream。不同 stream 可能并发，但依赖必须用 event 或同步原语表达。过度同步会让 CPU、GPU 或通信链路空等。

## 需要会算

Arithmetic intensity：

```text
arithmetic intensity = floating-point operations / bytes moved
```

把它与硬件的计算吞吐和内存带宽放在 roofline model 中，可以判断理论上更可能受计算还是带宽限制。

一个最简单的时间下界是：

```text
time ≥ max(FLOPs / compute throughput, bytes / memory bandwidth)
```

## 动手练习

1. 写 vector addition，并检查连续与跨步访问。
2. 实现 parallel reduction，处理 block 内同步。
3. 实现 tiled matrix multiplication，比较是否使用 shared memory。
4. 用 Triton 写 fused softmax 或 normalization。
5. 用 Nsight 或 PyTorch Profiler 记录 kernel 时间、带宽和 launch gaps。

## 常见误区

- GPU 利用率高不代表 Tensor Core 或 HBM 被有效使用；
- occupancy 不是越高越好；
- 单个 kernel 更快，端到端系统未必更快；
- 小算子可能主要消耗在 launch overhead；
- 不理解 tensor layout 时，很容易优化错误的数据路径。

## 掌握检查

- 为什么同一 warp 内的分支分歧会降低效率？
- tiling 怎样减少 HBM 流量？
- register 使用增加为什么可能降低 occupancy？
- kernel fusion 改善了什么，又可能增加什么成本？
- 怎样用证据区分 compute-bound 与 memory-bound？

下一步：[Module 03 · 数值计算](03-numerical-computing.md)
