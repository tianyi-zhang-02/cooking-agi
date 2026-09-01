# Online Softmax 与 IO-Aware Attention

**中文** · [English](03-online-softmax.en.md) · [返回项目](../../projects/01-attention-from-scratch.md)

> 阅读时间：约 5 分钟 · 难度：Advanced · 时效性：原理稳定、实现持续变化 · 最近审阅：2026-08

## 核心问题

为什么 attention 不应该总是把完整 `N×N` score matrix 写入 HBM？

普通实现先生成 scores，再读写 probabilities，最后计算 `PV`。Sequence 变长时，中间存储和数据移动按 `N²` 增长。即使 FLOPs 不变，HBM traffic 也可能成为主要瓶颈。

## Online softmax 状态

Stable softmax 需要整行最大值和指数和。Online 版本按 block 读取 scores，并维护：

```text
m = running maximum
l = running normalization sum
o = accumulated weighted output
```

处理新 block 时：

```text
m_new = max(m, max(score_block))
old_scale = exp(m - m_new)
new_weights = exp(score_block - m_new)
l_new = old_scale × l + sum(new_weights)
o_new = old_scale × o + new_weights × V_block
```

最后输出 `o / l`。关键是最大值变化时，旧累计量必须 rescale；遗漏这一步会产生系统性错误。

## IO-aware 数据流

目标不是让 softmax 本身少做一点算术，而是让 Q、K、V tile 在更近的存储层复用，并避免把完整 scores/probabilities 写回 HBM：

```text
Q tile stays local
→ stream K/V tiles
→ update online softmax state
→ emit output tile
```

这就是 FlashAttention 类方法的核心心智模型：重新安排计算顺序，以减少昂贵的数据移动，同时保持数学等价。

## Correctness 顺序

1. 在 CPU 上实现一维 online softmax。
2. 与普通 stable softmax 比较 random 和 extreme inputs。
3. 加入 block boundary 和 non-multiple size。
4. 实现带 `V` 累加的 online attention。
5. 最后移到 CUDA 并融合。

Mask 必须在当前 score block 中正确应用；causal boundary 可能穿过 tile。

## 动手验证

- 让 running maximum 在后一个 block 变大，确认旧状态被 rescale；
- 比较不同 block size 的误差；
- 测量是否真的减少 HBM bytes 和 peak memory；
- 同时记录 latency，避免只减少内存却增加过多计算或同步；
- 对长 sequence 和大 magnitude score 检查 NaN/Inf。

## 关键结论

FlashAttention 不是“一个更快的 softmax”。它是 attention 的 IO-aware 重排。正确性来自 online normalization invariant，性能来自减少高成本存储层之间的数据移动。

下一篇：[Precision、KV cache 与 benchmark](04-precision-kv-benchmark.md)
