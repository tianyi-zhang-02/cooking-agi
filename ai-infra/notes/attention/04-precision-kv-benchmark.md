# Attention 的 Precision、KV Cache 与 Benchmark

**中文** · [English](04-precision-kv-benchmark.en.md) · [返回项目](../../projects/01-attention-from-scratch.md)

> 阅读时间：约 5 分钟 · 难度：Advanced · 时效性：Fast-moving · 最近审阅：2026-08

## 只解决一个问题

怎样证明一个低精度或 decode attention 优化值得使用？

答案必须同时包含 correctness、latency、throughput、memory 和 workload。只报告 TFLOP/s 或单一 shape 不足以支持 serving 决策。

## Precision 路线

依次比较：

1. FP32 input + FP32 accumulation；
2. BF16/FP16 input + FP32 accumulation；
3. 只有硬件和工具链明确支持时，再加入 FP8 或量化路径。

检查：

- output absolute/relative error；
- softmax row sum；
- NaN/Inf；
- large magnitude score；
- long sequence；
- latency、throughput、HBM bytes 和 peak memory。

不要为了宣称支持低精度，手工把所有 intermediate 都降精度。Reduction、normalization 和 accumulation 可能需要更高精度。

## KV cache

Self-attention serving 分为：

- **Prefill**：一次处理 prompt，生成所有历史 K/V；
- **Decode**：每步追加一个 K/V，让新 query 读取历史 cache。

简化 cache 大小：

```text
KV bytes ≈ 2 × layers × tokens × kv_heads × head_dim × bytes
```

实际还受 batch slots、tensor parallel、padding、allocator 和 paged layout 影响。

第一版实现 contiguous cache，明确：shape、append position、valid length、batch slot 和 query-head 到 KV-head mapping。之后再研究 paged allocation。

## 为什么分开 benchmark

Prefill 的矩阵更大，通常更偏 compute-intensive；decode 每步 query 很小，却重复读取权重和历史 KV，通常更受 bandwidth 和 cache layout 影响。

因此 benchmark 至少分为：

| Workload | 变量 |
| --- | --- |
| Prefill | prompt length、batch、heads、dtype |
| Decode | context length、active sequences、KV dtype、GQA |
| End-to-end | TTFT、TPOT、throughput、P95/P99 |

## 动手检查

1. 对同一输入比较 FP32 与 BF16/FP16。
2. 为 cache append、sequence length 和 batch reuse 写边界测试。
3. 分开记录 prefill 和 decode kernel time。
4. 对 context length 做 sweep，观察 bandwidth 与 latency。
5. 保存硬件、CUDA、compiler、flags、commit 和完整 shape。

## 记住

低精度是否成功由质量和系统收益共同决定；KV cache 是否成功由真实 decode workload 决定。离开 workload 的“更快”没有意义。

返回：[Project 01 · 手搓 Attention](../../projects/01-attention-from-scratch.md)
