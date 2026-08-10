# Project 01 · 手搓 Attention

**中文** · [English](01-attention-from-scratch.en.md) · [项目路线](README.md)

> 阅读时间：约 5 分钟 · 难度：Intermediate → Advanced · 时效性：Evolving · 最近审阅：2026-08

## 目标

从矩阵乘法和 stable softmax 开始，实现可验证的 scaled dot-product attention，再进入 multi-head、GQA、CUDA、tiling、online softmax 和 KV cache。

项目必须保持一条证据链：

```text
数学定义
→ Python oracle
→ CPU C++ reference
→ correctness tests
→ CUDA kernels
→ profiler
→ 数值与性能报告
```

## 数学边界

```text
Q: [B, H, Nq, D]
K: [B, Hkv, Nk, D]
V: [B, Hkv, Nk, Dv]

S = QKᵀ / √D
P = softmax(S + mask)
O = PV
```

第一版使用 `H = Hkv`，`Nq = Nk`，FP32 和无 mask。每增加一个维度或优化，都继续对同一个 reference 验证。

## 四篇短笔记

| 笔记 | 只解决一个问题 |
| --- | --- |
| [01 · Reference、shape 与 mask](../notes/attention/01-reference-shapes-masks.md) | Attention 到底算什么，怎样证明边界正确？ |
| [02 · 从 CPU 到 tiled CUDA](../notes/attention/02-cpu-to-tiled-cuda.md) | 怎样逐阶段优化并保持可调试性？ |
| [03 · Online softmax 与 IO-aware attention](../notes/attention/03-online-softmax.md) | 为什么不 materialize `N×N` matrix？ |
| [04 · Precision、KV cache 与 benchmark](../notes/attention/04-precision-kv-benchmark.md) | 怎样验证低精度和 decode 路径？ |

每篇控制在约五分钟。项目实现可以长期扩展，但概念不会堆在同一个页面。

## 建议目录

```text
attention-from-scratch/
├── CMakeLists.txt
├── include/attention/
├── src/
│   ├── attention_cpu.cpp
│   ├── attention_cuda.cu
│   └── online_softmax.cu
├── reference/attention_reference.py
├── tests/
├── bench/
├── results/
└── README.md
```

可以复用 [P00 C/C++ Tensor Core](00-c-cpp-tensor-core.md) 的 tensor、matmul 和 softmax。

## Milestone 路线

1. **Oracle**：用显式 PyTorch matmul、mask、softmax、matmul 建 reference。
2. **CPU single-head**：materialize 完整 score 和 probability matrix。
3. **Multi-head/GQA**：显式写 shape、stride 和 KV-head mapping。
4. **Masks**：分别测试 causal、padding、additive 和 combined mask。
5. **Optimized CPU**：改善 loop order、tiling 和 parallelism。
6. **Naive CUDA**：QKᵀ、mask/scale、softmax、PV 分开实现和验证。
7. **Tiled CUDA**：复用 shared-memory tile，减少 HBM traffic。
8. **Online softmax**：分块维护 running max、normalizer 和 output。
9. **Precision**：比较 FP32 与 BF16/FP16 input + FP32 accumulation。
10. **KV cache**：分别测 prefill 与 single-token decode。

## 每个版本必须保存

| Evidence | 内容 |
| --- | --- |
| Correctness | absolute/relative error、softmax row sum、NaN/Inf |
| Shape | B、H/Hkv、Nq/Nk、D/Dv、layout、mask |
| Performance | latency、throughput、peak memory、HBM traffic |
| Environment | GPU、CPU、compiler、CUDA、flags、commit |
| Explanation | 为什么更快，瓶颈转移到了哪里？ |

## Definition of done

- CPU reference 通过固定与随机测试；
- mask、GQA 和非 tile 整数倍 shape 有边界测试；
- naive CUDA 各阶段可以独立与 reference 对比；
- 至少一个 tiled/fused 版本由 profile 证明减少瓶颈；
- online softmax 有独立 correctness test；
- 至少一种低精度配置有数值报告；
- prefill 与 decode 分开 benchmark；
- README 解释结果，不只粘贴速度数字。

相关模块：[C/C++](../modules/00-c-cpp-foundations.md) · [GPU](../modules/02-gpu-programming.md) · [Numerics](../modules/03-numerical-computing.md) · [Inference](../modules/05-llm-inference.md)
