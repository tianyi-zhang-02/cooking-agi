# AI Infra 实战项目路线

**中文** · [English](README.en.md) · [返回 AI Infra](../README.md)

> 阅读时间：约 5 分钟 · 类型：项目索引 · 时效性：Evolving · 最近审阅：2026-08

## 为什么要单独做项目

读懂概念只能证明“见过”。实战项目要求把正确性、性能、数值稳定性、可观测性和复现放在一起。每个项目都应该产生四类证据：

```text
correctness tests
performance benchmark
profiling evidence
written analysis
```

## 项目路线

| 项目 | 主要技能 | 最终产出 |
| --- | --- | --- |
| [P00 · C/C++ Tensor Core](00-c-cpp-tensor-core.md) | 指针、RAII、layout、测试、benchmark | 小型 CPU tensor library |
| [P01 · 从零实现 Attention](01-attention-from-scratch.md) | matmul、softmax、mask、CUDA、online softmax | CPU/CUDA attention 与分析报告 |
| P02 · Compiler-to-Kernel Lab | `torch.compile`、Triton、CUDA、fusion、profiling | 从 graph 追到 kernel 的算子报告 |
| P03 · Mixed Precision Lab | BF16、FP8/MXFP8/NVFP4、scaling、误差 | 精度—吞吐—显存报告 |
| P04 · Distributed Training Lab | DDP、FSDP2/DTensor、collective、device mesh | 多 GPU scaling report |
| P05 · LLM Serving Benchmark | KV cache、token 调度、TTFT/TPOT、disaggregation | workload-driven serving dashboard |
| [P06 · Self-Improving Service](../modules/08-capstone.md) | trace、eval、SFT、canary、rollback | 完整学习闭环 |

P02–P05 的第一版练习分别放在对应模块中；当实验积累到足够深度时，再拆成独立项目页。

## 统一项目规范

每个项目至少包含：

- `README`：目标、假设、设计和结果；
- `src/`：实现；
- `tests/`：正确性与边界条件；
- `bench/`：固定输入分布和 benchmark harness；
- `results/`：机器、编译器、flags、硬件和原始结果；
- `notes/`：profile 截图、瓶颈解释和下一步。

推荐使用明确的实验表：

| Version | Input shape | Dtype | Correct? | Latency | Throughput | Peak memory | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Benchmark 纪律

1. 先证明结果正确，再比较速度。
2. 把 initialization 与 warm-up 排除在 steady-state 时间之外。
3. 防止编译器删除没有被观察的计算。
4. 固定随机种子和输入分布。
5. 报告 median 以及尾部或分布，而不只报告最好一次。
6. 保存硬件、编译器、library 和 flags。
7. 每次优化只改变少量变量。
8. 用 profiler 解释变化，而不是只记录变化。

## 推荐顺序

```text
C/C++ tensor primitives
→ CPU attention
→ CUDA operators
→ tiled / online-softmax attention
→ mixed precision
→ multi-GPU training
→ LLM serving
→ continual-learning loop
```

不要跳过 CPU reference。它虽然慢，却是后续 CUDA、低精度和融合实现最重要的 correctness oracle。
