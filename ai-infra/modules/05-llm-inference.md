# Module 05 · LLM 推理系统

**中文** · [English](05-llm-inference.en.md) · [返回 AI Infra](../README.md)

> 阅读时间：约 5 分钟 · 难度：Intermediate · 时效性：Fast-moving · 最近审阅：2026-08

## 核心问题

训练优化的是得到模型的过程，推理优化的是模型服务真实请求的过程。本模块关注如何在请求长度和到达时间都不规则时，同时控制延迟、吞吐、显存和成本。

## 学习目标

- 能解释 tokenization、queue、prefill、decode 和 streaming；
- 能区分 TTFT、TPOT、end-to-end latency 与 throughput；
- 理解 KV cache、paged attention 和 continuous batching；
- 能解释 prefix caching、chunked prefill、speculative decoding 和 prefill/decode disaggregation；
- 能设计一份包含质量与系统指标的 serving benchmark。

## 核心笔记

### 请求生命周期

```text
到达 → admission control → queue → batching
→ prefill → decode loop → streaming → 完成或取消
```

系统不仅调度请求，也在调度 token。每个请求的 prompt 长度、生成长度和停止时间不同，因此静态 batch 很容易产生 padding 和等待。

### Prefill 与 decode

Prefill 一次处理 prompt 中多个 token，矩阵较大，通常更容易利用 Tensor Core。Decode 每一步通常只为每个请求生成一个新 token，需要反复读取权重和 KV cache，往往更受显存带宽限制。

长 prefill 可能阻塞正在 decode 的请求，降低交互体验；过度优先 decode 又可能让新请求长期等不到首 token。Scheduler 必须管理这两类工作之间的公平性和吞吐权衡。

### KV cache

Attention 需要复用历史 token 的 key 和 value。KV cache 避免每次 decode 重新计算全部历史，但它会随并发、上下文长度和层数增长。

Paged attention 把 KV cache 分成可以非连续分配的 block，减少预留和碎片，并让不同长度请求更灵活地共享显存池。

### Continuous batching

静态 batch 等所有请求一起结束；continuous batching 会在请求完成后立即移除，并把新请求加入后续 decode iteration。它提高利用率，但调度器必须维护每个序列的状态、位置和 cache 映射。

### Prefill 与 decode 何时分离

Chunked prefill 在同一引擎里切分长 prompt；disaggregated serving 则可以把 prefill 和 decode 放到不同 worker pool，让两类阶段采用不同并行、batching 或硬件配置。代价是 KV transfer、额外 queue、routing 和故障协调。

Disaggregation 不会因为更新就自动更快。只有已经测到 phase interference 影响 SLO，或独立扩缩容对真实 workload 的收益足以抵消传输与运维成本时，才应采用。

### 常见优化

- **Prefix caching**：复用相同 prompt 前缀的 KV 状态；
- **Chunked prefill**：把很长的 prefill 拆成片段，减少阻塞；
- **Prefill/decode disaggregation**：在 KV transfer 成本可接受时隔离阶段特有的调度与容量；
- **Speculative decoding**：由较便宜的 draft model 提议多个 token，再由目标模型验证；
- **Quantization**：减少权重或 KV cache 的存储和带宽；
- **Tensor/Pipeline Parallel**：让单卡放不下的模型跨设备运行；
- **Admission control**：在过载前限制进入系统的工作量。

## 关键计算

简化的 KV cache 大小：

```text
KV bytes ≈ 2 × layers × tokens × kv_heads × head_dim × bytes_per_element
```

其中 2 表示 key 和 value。还要乘 batch 中所有活跃序列，并考虑 tensor parallel 的切分方式、padding 与 allocator overhead。

核心指标：

```text
TTFT = first token timestamp - request arrival
TPOT = decode duration / generated tokens
throughput = completed tokens or requests / wall-clock time
```

报告 latency 时必须带 percentile 和 workload 分布，单一平均值几乎没有解释力。

## 动手验证

1. 用 vLLM、SGLang 或 TensorRT-LLM 启动一个小模型。
2. 分别改变 prompt length、output length、并发和 batch token budget。
3. 记录 TTFT、TPOT、throughput、P95/P99 和峰值显存。
4. 比较 prefix caching 开关前后的重复 prompt workload。
5. 用同一数据集比较 BF16 与一种量化配置的质量和成本。
6. 用同一 arrival trace 比较 chunked prefill 单引擎和分离 prefill/decode pool，并计入 KV-transfer 时间。

## 常见误区

- tokens/s 高不代表用户等待时间短；
- 只压测固定长度请求会高估真实服务性能；
- 最大 batch size 通常不是最佳 batch size；
- KV cache OOM 不等于模型权重本身放不下；
- 忽略取消、超时、重试和流式断开会产生错误容量估计。

## 学习检查

- 为什么 prefill 和 decode 的瓶颈不同？
- continuous batching 为什么比静态 batching 更适合 LLM？
- GQA 怎样影响 KV cache 大小？
- speculative decoding 在什么条件下才会加速？
- 怎样防止一个超长 prompt 破坏所有其他请求的尾延迟？

当前资料：[vLLM serving options 与 scheduler controls](https://docs.vllm.ai/en/latest/cli/serve/)

下一步：[Module 06 · GPU 平台](06-gpu-platforms.md)
