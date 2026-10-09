# Attention Residuals：这一层该接哪些前层信息？

**中文** · [English](attention-residuals.en.md)

通常讲 attention，我们问的是“这个 token 看哪些 token”。Attention Residuals 换了一个轴：**同一个 token 走到这一层，要怎样组合前面各层留下的信息？** 没有把序列维度和网络深度分开，很容易把它误读成又一种 sparse attention。

## 从普通 residual 开始

普通残差层写成 $h_{l+1}=h_l+f_l(h_l)$。逐层展开，后面的表示包含 embedding 与各层更新的累加。通路很好走，但这些更新进入总和时的系数都固定为 1。

Attention Residuals 对之前的深度表示做加权。用 $v_i$ 表示 embedding 或某个前层输出，省略 batch 与 token 维度：

$$
s_{l,i}=w_l^\top\operatorname{RMSNorm}(v_i),\qquad
\alpha_{l,i}=\frac{\exp(s_{l,i})}{\sum_j\exp(s_{l,j})},\qquad
h_l=\sum_i\alpha_{l,i}v_i.
$$

$w_l$ 是这一层学习到的 pseudo-query，不是当前 token 另做的一次 $W_Qh$。但权重仍与内容有关，因为 keys 来自当前样本、当前 token 的前层表示。RMSNorm 用于 keys，values 保留原表示。

```text
同一个 token：embedding ─┐
             早期层输出 ├─→ 深度方向的 softmax ─→ 加权和 ─→ 当前子层
             中间层输出 ┤
             最近层输出 ┘
```

## 两路信息怎样合并？

只看标量 value，假设两路是 2 和 10，得到的 scores 是 0 与 $\log 3$。softmax 权重是 1/4 和 3/4，结果为 8。普通求和是 12；简单平均是 6。

```python
import math

def depth_mix(scores, values):
    if not scores or len(scores) != len(values):
        raise ValueError("Expected matching nonempty scores and values")
    if any(not math.isfinite(value) for value in scores + values):
        raise ValueError("Expected finite inputs")
    shift = max(scores)
    weights = [math.exp(score - shift) for score in scores]
    total = sum(weights)
    return sum(weight * value for weight, value in zip(weights, values)) / total

assert math.isclose(depth_mix([0, math.log(3)], [2, 10]), 8)
assert depth_mix([0, 0], [2, 10]) == 6
assert math.isclose(depth_mix([1000, 1000 + math.log(3)], [2, 10]), 8)
```

这个函数只检查加权和，不实现模型里的 RMSNorm 或 query 学习。它也揭示了初始化区别：pseudo-query 为零时，各路均匀平均，并不还原原来的 residual 求和。修改 residual 结构需要训练适配，不是任意 checkpoint 上的无损补丁。

## 为什么还要分 Block？

每层都保存、读取全部前层输出，会增加深度方向的访存。Block AttnRes 把多个连续子层分组：组内仍累加更新，跨组才做 attention。可见的条目包括 embedding、已完成的 block 表示，以及当前 block 已累积的部分；不是等一个 block 全结束，前面的更新才生效。

这里的 block 是**深度分组**，不是一段连续文本。论文把 attention 与 MLP 当作不同子层计数；比较层数时要对齐口径。

| 简化成本，$L$ 个子层、$N$ 个 block | Full AttnRes | Block AttnRes |
| --- | --- | --- |
| 每个 token 保留的深度状态 | $O(Ld)$ | $O(Nd)$ |
| 各层混合的总计算 | $O(L^2d)$ | $O(LNd)$ |
| 选择的粒度 | 单个子层输出 | block 输出与当前部分和 |

这张表只算深度混合，不是整个模型的训练显存或 FLOPs。它没有直接缩小序列 KV cache，也没有让 token attention 变成线性复杂度。

## 分组后，为什么不能平均几个局部输出？

把候选来源分批处理，可以减少中间存储。但每批的 softmax 分母不同，不能平均已经归一化的输出。

例如第一批只有 value 0，第二批有 3 个 value 4，所有 score 都是 0。两批局部输出为 0 与 4，简单平均得 2；正确全局输出是 3，因为第二批的总权重是第一批的 3 倍。

合并时需要保留每批的 log-sum-exp 或等价的最大值与指数和。这与 [FlashAttention 的在线 softmax](attention-kernels.md) 是同一种归一化问题，只是这里沿深度组合来源。

## 应该怎样评估？

先做数值检查：全量混合与分批混合一致；改变未来 token 不影响前缀；block 边界前后没有漏加、重加当前部分和。关闭随机性后再比较，别把 dropout 当成实现误差。

训练对照固定 token 数、数据和总预算，比较普通 residual、Full 与不同 block 大小。除了 loss，还要测训练吞吐、峰值显存、decode 延迟与长序列任务。更细的深度选择不必然值得额外访存，最优分组也不一定适用于另一种硬件。

核对日期：2026-10-08。依据 [Attention Residuals](https://arxiv.org/abs/2603.15031) 与[官方实现](https://github.com/MoonshotAI/Attention-Residuals)。本页算例用于理解归一化与计算量，不声称复现训练收益。
