# Project 01 · 手搓 Attention：从 CPU Reference 到 CUDA

**中文** · [English](01-attention-from-scratch.en.md) · [项目路线](README.md)

## 项目目标

从基本矩阵和 softmax 开始，实现可验证的 scaled dot-product attention，再逐步加入 multi-head、causal mask、KV cache、CUDA、tiling 和 online softmax。

项目重点不是快速抄出一个 kernel，而是建立一条证据链：

```text
数学定义
→ 简单 reference
→ C++ CPU 实现
→ correctness tests
→ profiler
→ CUDA 实现
→ 数值与性能对照
```

## Attention 数学

给定：

```text
Q: [B, H, Nq, D]
K: [B, Hkv, Nk, D]
V: [B, Hkv, Nk, Dv]
```

标准 scaled dot-product attention：

```text
S = QKᵀ / √D
P = softmax(S + mask)
O = PV
```

其中：

- `B`：batch size；
- `H`：query heads；
- `Hkv`：key/value heads；
- `Nq`、`Nk`：query 和 key sequence length；
- `D`：head dimension；
- causal mask 阻止位置看到未来 token。

第一版让 `H = Hkv`，实现正确后再扩展 MQA/GQA。

## 建议目录

```text
attention-from-scratch/
├── CMakeLists.txt
├── include/attention/
│   ├── attention.h
│   └── tensor.h
├── src/
│   ├── attention_cpu.cpp
│   ├── attention_cuda.cu
│   └── online_softmax.cu
├── reference/
│   └── attention_reference.py
├── tests/
├── bench/
├── results/
└── README.md
```

可以复用 [Project 00](00-c-cpp-tensor-core.md) 的 tensor、matmul 和 softmax，也可以在本项目中保留一个更小的专用实现。

## Milestone 0 · Python correctness oracle

用 PyTorch 写最简单的 reference，但不要调用 high-level attention API：

1. 显式计算 `Q @ K.transpose(-2, -1)`；
2. 除以 `sqrt(D)`；
3. 加 mask；
4. 调用 softmax；
5. 计算 `P @ V`。

保存一组很小的固定输入和输出，供 C++ 测试读取。再加入随机 property tests，避免只对一组 golden values 过拟合。

## Milestone 1 · 单头 CPU C++

实现：

```text
scores[Nq, Nk]
= matmul(Q[Nq, D], transpose(K[Nk, D]))
→ scale
→ mask
→ row-wise stable softmax
→ matmul(probabilities[Nq, Nk], V[Nk, Dv])
```

先 materialize 完整 score 和 probability matrix。它占内存、速度慢，但控制流清楚，最适合作为 C++ reference。

检查：

- `1 / sqrt(D)` 是否在 softmax 前应用；
- causal mask 的边界；
- 每行 probability 是否接近 1；
- 极端 score 是否产生 NaN 或 Inf；
- non-square `Nq != Nk`；
- `Dv != D`。

## Milestone 2 · Multi-Head Attention

加入 batch 和 head 维度：

- 明确使用 `[B, H, N, D]` 还是 `[B, N, H, D]`；
- 写出每个 layout 的 stride；
- 避免在 inner loop 中重复做昂贵的 index 计算；
- 让每个 `(batch, head)` 独立执行单头 attention；
- 测试 head permutation 不会互相污染。

然后加入 MQA/GQA：多个 query heads 映射到较少的 KV heads。必须明确 head mapping，而不是依靠隐式 broadcasting 猜测。

## Milestone 3 · Mask 与 padding

分别支持：

- causal mask；
- padding / valid-length mask；
- arbitrary additive mask；
- causal 与 padding 的组合。

不要用一个有限的大负数就假设所有 dtype 都安全。记录 FP32、FP16/BF16 reference 中 mask value 的行为，并检查整行都被 mask 时的既定输出。

## Milestone 4 · CPU 性能版本

建立清晰 baseline 后再优化：

1. 改善 loop order 和连续访问；
2. 对 `(batch, head)` 或 query row 并行；
3. 使用 tiled matmul；
4. 减少中间 transpose copy；
5. 可选：用 BLAS 代替 matmul，保留自写 softmax；
6. 用 profiler 证明时间花在 QKᵀ、softmax、PV 还是数据布局转换。

报告 CPU reference 与 optimized CPU 版本的差异，但始终保留 reference 用于测试。

## Milestone 5 · Naive CUDA Attention

先拆成多个容易验证的 kernels：

```text
QKᵀ kernel
→ scale and mask kernel
→ row softmax kernel
→ PV kernel
```

每一步都可以从 device copy 回 host 与 CPU reference 比较。这不是最终高性能方案，但能隔离错误。

重点观察：

- thread/block mapping；
- coalesced loads；
- reduction 的同步；
- kernel launch overhead；
- 中间 score matrix 的 HBM 流量；
- register 与 shared-memory 使用。

## Milestone 6 · Tiled Attention

把 Q、K、V 分块搬入 shared memory，复用 tile，并减少 global-memory traffic。先优化 QKᵀ 与 PV，再考虑跨阶段融合。

使用 roofline 思考：

```text
arithmetic intensity = FLOPs / bytes moved
```

比较 naive 与 tiled 版本实际移动的数据量、带宽和 kernel time。不要只因为代码使用了 shared memory 就假设它更快。

## Milestone 7 · Online Softmax 与 FlashAttention 思路

完整 materialize `N × N` score matrix 会带来二次方中间存储和 HBM 流量。Online softmax 通过维护每行的 running maximum 与 normalization sum，按 block 处理 score：

```text
old state: m, l, accumulated output
new block maximum: m_block
new maximum: m_new = max(m, m_block)
rescale old contribution by exp(m - m_new)
add new block contribution scaled by exp(score - m_new)
update normalization l
```

先在 CPU 上验证 online softmax 与普通 stable softmax 一致，再移到 CUDA。然后把 `QKᵀ`、softmax normalization 和 `PV` 的数据流融合，避免把完整 score/probability matrix 写回 HBM。

目标是理解 FlashAttention 的 IO-aware 核心思想，不要求第一版达到生产 kernel 性能。

## Milestone 8 · Precision

依次比较：

- FP32 input + FP32 accumulation；
- BF16/FP16 input + FP32 accumulation；
- 可选的 FP8 或量化实验，前提是硬件和工具链支持。

检查：

- output absolute/relative error；
- softmax row sum；
- NaN/Inf；
- 长序列和大 score；
- latency、throughput 与 peak memory。

不要为了使用某种 dtype 手工把所有中间值都降精度。

## Milestone 9 · KV Cache 与 decode

把 self-attention 分成：

- prefill：一次产生整个 prompt 的 K/V；
- decode：每步追加一个新 K/V，并让单个 query 读取历史 cache。

实现最简单的 contiguous KV cache，再研究 paged layout。记录 cache shape、append 规则、有效长度、batch slot 和 head mapping。

比较 prefill 和 decode 的算术强度与访存行为，解释为什么同一个 attention 算子在两个阶段可能有不同瓶颈。

## Correctness test matrix

至少覆盖：

| 维度 | 测试值 |
| --- | --- |
| Batch | 1、>1 |
| Heads | 1、multiple、GQA |
| Sequence | 1、短、非 tile 整数倍、较长 |
| Head dim | 小、常见、非 tile 整数倍 |
| Mask | none、causal、padding、combined |
| Dtype | FP32、BF16/FP16（若支持） |
| Values | random、constant、large magnitude、all masked edge |

每个 optimized version 都必须与同一 reference 对比，而不是与前一个可能已经出错的优化版本对比。

## Benchmark matrix

记录：

| Version | Shape | Dtype | Causal | Latency | TFLOP/s | HBM traffic | Peak memory | Max error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Benchmark 至少分开：

- end-to-end；
- QKᵀ；
- softmax/mask；
- PV；
- layout conversion；
- host/device transfer，如果它属于真实使用路径。

## Definition of done

- 数学、shape、stride 和 mask 语义写清楚；
- CPU reference 通过固定与随机测试；
- naive CUDA 每个阶段可以独立验证；
- 至少一个 tiled 或 fused 版本由 profile 证明减少了瓶颈；
- online softmax 有独立 correctness test；
- FP32 与至少一种低精度配置有数值报告；
- benchmark 保存硬件、compiler、CUDA、flags 和 workload；
- README 能解释为什么某个版本更快，而不只是报告数字。

相关模块：

- [C/C++ 基础](../modules/00-c-cpp-foundations.md)
- [GPU 编程](../modules/02-gpu-programming.md)
- [数值计算](../modules/03-numerical-computing.md)
- [LLM 推理](../modules/05-llm-inference.md)
