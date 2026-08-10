# Module 01 · 计算机系统基础

**中文** · [English](01-computer-systems.en.md) · [返回 AI Infra](../README.md)

## 这一模块解决什么

在优化 GPU、训练或推理之前，先要理解一段程序怎样被 CPU、操作系统和内存系统执行。这个模块建立最基本的性能直觉：**时间究竟花在计算、等待数据，还是等待其他任务？**

## 学习目标

- 能解释进程、线程和一次 context switch；
- 能描述 CPU 从取指到提交的基本过程；
- 能解释 cache、cache line、TLB、虚拟内存和 page fault；
- 能区分 latency、throughput、并发与并行；
- 能判断一个简单程序是 compute-bound 还是 memory-bound。

## 核心笔记

### 程序怎样变成正在运行的进程

源代码先被编译或解释成机器可以执行的指令。操作系统为它创建进程，分配虚拟地址空间，并把代码、堆、栈和动态库映射到物理内存。

线程是操作系统调度的执行流。同一进程里的线程共享地址空间，但各自拥有寄存器状态和栈。线程切换需要保存和恢复执行状态，也可能破坏 cache locality。

### CPU 怎样执行指令

一个简化的乱序 CPU 流水线是：

```text
Fetch → Decode → Rename → Dispatch → Execute → Retire
```

- branch predictor 猜测下一条指令来自哪里；
- register renaming 消除假的寄存器依赖；
- scheduler 让准备好的指令先执行；
- reorder buffer 保证结果按程序顺序提交；
- 预测错误时，推测执行的结果被丢弃。

CPU 核心的价值不是只把一条指令跑快，而是同时挖掘 instruction-level、data-level 和 thread-level parallelism。

### 内存层级

```text
寄存器 → L1 → L2 → L3 → DRAM → SSD / 远程存储
```

越靠近核心，容量越小、成本越高、延迟越低。连续访问会利用 spatial locality；重复访问近期数据会利用 temporal locality。跨 cache line 的随机访问通常比顺序扫描昂贵得多。

虚拟地址需要经过 page table 转换为物理地址，TLB 会缓存最近的地址转换。TLB miss、page fault 和 NUMA 远端访问都会形成额外延迟。

### 并发、并行和吞吐

- **Concurrency**：多个任务在时间上交错推进；
- **Parallelism**：多个任务真的在同一时刻执行；
- **Latency**：完成一个任务需要多久；
- **Throughput**：单位时间能完成多少任务。

增加线程可能提高吞吐，却不一定降低单个请求的延迟。线程过多还会带来调度、同步和 cache contention。

## 需要会算

CPU 时间的基础近似：

```text
CPU time ≈ instruction count × cycles per instruction / clock frequency
```

内存吞吐的基础近似：

```text
time ≥ bytes moved / effective bandwidth
```

这些式子不是精确模拟器，但可以帮助判断“理论上不可能更快”的边界。

## 动手练习

1. 比较同一个大数组的顺序访问和随机访问时间。
2. 改变工作集大小，观察它超过不同 cache 层级时的性能变化。
3. 用 profiler 观察一个 PyTorch CPU workload 的算子时间和线程活动。
4. 分别用一个线程和多个线程完成相同任务，记录 latency 与 throughput。
5. 写一页分析：瓶颈来自指令、内存、同步还是 I/O？

## 常见误区

- CPU 使用率 100% 不代表所有执行单元都被有效利用；
- 更多线程不一定更快；
- Big-O 相同的算法可能因为 locality 不同而有巨大性能差异；
- 内存容量够用不代表内存带宽和延迟没有问题；
- benchmark 如果包含初始化、磁盘读取或 warm-up，结论可能完全失真。

## 掌握检查

- 为什么顺序扫描通常比随机访问快？
- branch misprediction 为什么会浪费周期？
- process 与 thread 共享和不共享什么？
- cache miss、TLB miss 和 page fault 有什么区别？
- latency 变好时，throughput 为什么仍可能变差？

下一步：[Module 02 · GPU 编程](02-gpu-programming.md)
