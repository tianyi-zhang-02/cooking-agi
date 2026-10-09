# MoE：负载均衡

**中文** · [English](load-balancing.en.md)

> 阅读时间：约 7 分钟 · 难度：进阶 · 最近审阅：2026-10-09

## 为什么少数 expert 可能越来越忙 {#_1}

某个 expert 碰巧多分到一些 token，训练机会更多，之后又更容易被选中；其他 expert 很少被用到，就可能越来越难追上。这是一种可能出现的正反馈，不是每个 MoE 都必然塌缩。没被选中的 expert 参数通常拿不到这个 token 的任务梯度，但 router 的梯度还取决于门值归一化和辅助目标，不能把两者混为一谈。

下面是个玩具模拟，切换三种做法，看负载怎么变：

<!-- widget:tx-moe-balance -->

## Capacity factor：每个 expert 最多收多少 token

在 Switch 的 top-1 路由中，capacity factor 用于确定每个 expert 的 token 槽位上限，方便安排计算与缓冲区；它不保证各卡耗时相等：

$$\text{capacity} = \frac{\text{一批里的 token 数}}{N} \times \text{capacity factor}$$

Switch 的溢出处理跳过对应 expert 分支，token 本身仍走残差路径。小 CF 节省槽位但可能多丢分支，大 CF 减少溢出却可能留更多空位；合理取值需要结合路由分布与吞吐测试。Dropless 实现不丢 expert 分配，也仍要处理负载和缓冲区。不要把 top-1 公式直接当成所有 top-k 实现的容量定义。

## 辅助 loss：把不均衡直接写进损失

Switch Transformer 的辅助 loss：

$$\mathcal L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i$$

- $f_i$：这一批里 argmax 落在 expert $i$ 上的 token 比例，本质是个计数，不可导；
- $P_i$：这一批 token 分给 expert $i$ 的平均 router 概率，可导；
- $\alpha = 10^{-2}$。

两个乘在一起：梯度顺着 $P_i$ 传回 router，推多狠则由真实负载 $f_i$ 说了算。完全均匀时 $f_i = P_i = 1/N$，损失等于 $\alpha$；这是均匀状态的参考值，不是这个乘积式在所有路由分布下的严格下界。[GShard](https://arxiv.org/abs/2006.16668) 也用分配比例乘平均门值：把原本不可导的比例平方中的一个因子换成可导近似，而不是直接对离散计数求导。最早的 sparsely-gated MoE 则用了 importance 和 load 两个变异系数损失。

## Router z-loss：管数值，不管均衡

ST-MoE 另加 router z-loss，把每个 token 的 log-sum-exp 往 0 拉：

$$L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^2$$

论文设置系数 $c_z=0.001$。这里 B 是参与计算的 token 数，不是序列数。它约束的是 log-normalizer，不是每个 logit 的绝对值：概率 `[0.9, 0.1]` 的对数作为 logits 时，z-loss 就为 0，但路由仍可能很偏。因此它不能替代负载均衡目标。见 [ST-MoE](https://arxiv.org/abs/2202.08906)。

## 不加辅助 loss：只调 bias

辅助 loss 的梯度和语言模型 loss 落在同一组 router 分数上，两边会互相拉扯。DeepSeek-V3 换了个办法：

1. 每个 expert 有一个 bias $b_i$，**只在挑 top-k 时**加到分数上；
2. 选中以后，门控权重用原始 affinity score 算；bias 不直接进入门值公式，也不靠反向传播更新，但它改变了选中集合，因此会间接改变输出与任务梯度；
3. 每步结束，超载的 expert 把 $b_i$ 调低 $\gamma$，空闲的调高 $\gamma$：前 14.3T token 用 $\gamma = 0.001$，最后 500B token 设成 0。

说它完全没有 balance loss 并不准确：DeepSeek-V3 还留了一项很小的序列级 balance loss（$\alpha = 0.0001$），防止单条序列内部失衡得太离谱。

Qwen3 走的是另一条路：global-batch balance loss，在全局 batch 上算均衡，而不是每个 micro-batch 都算，这样 expert 在局部可以更偏科。

## 不进门值公式，为什么还会改变输出？

看一个只用于说明的 top-2 例子，忽略分组路由和额外缩放：

| Expert | 原始 affinity | 均衡 bias | 用来选人的分数 |
| --- | --- | --- | --- |
| 0 | 0.8 | -0.2 | 0.6 |
| 1 | 0.7 | 0 | 0.7 |
| 2 | 0.2 | 0.5 | 0.7 |

没有 bias 时选 0、1；加上以后选 1、2。后者的权重仍来自 0.7 和 0.2，归一化后约为 0.778、0.222，而不是根据两个 0.7 得到各 0.5。选的人都变了，输出当然可能变。这正是 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 中“选择分数”和“门值”要分开看的原因。

```python
def route_with_bias(affinities, biases, top_k):
    if len(affinities) != len(biases) or not 1 <= top_k <= len(affinities):
        raise ValueError("Invalid routing shape or top_k")
    if any(score <= 0 for score in affinities):
        raise ValueError("This example expects positive affinities")
    chosen = sorted(range(len(affinities)),
                    key=lambda expert: (-(affinities[expert] + biases[expert]), expert))[:top_k]
    denominator = sum(affinities[expert] for expert in chosen)
    return chosen, [affinities[expert] / denominator for expert in chosen]

chosen, weights = route_with_bias([0.8, 0.7, 0.2], [-0.2, 0.0, 0.5], 2)
assert chosen == [1, 2]
assert abs(weights[0] - 7 / 9) < 1e-12
```

## Expert 均衡与卡间均衡，是两张表

假设 4 个 experts，前两个在卡 A，后两个在卡 B；这里只数分配次数，暂不考虑每次计算的差异。

| 各 expert 收到的分配数 | 卡 A / 卡 B | 能看出什么 |
| --- | --- | --- |
| 4、0、4、0 | 4 / 4 | 卡间均衡，但一半 experts 没有训练机会 |
| 4、4、0、0 | 8 / 0 | 专家和卡都不均衡 |
| 2、2、2、2 | 4 / 4 | 这一次两者都均衡，不代表每个 step 都如此 |

再看容量：8 个 tokens、top-2、4 个 experts，一共 16 次分配，平均每个 expert 4 次。如果实现按分配数定义 capacity factor 为 1.25，上限就是 5。**超出一次分配，不等于丢掉一个训练样本**：token 可能还有另一条 expert 路径，也有 residual 路径；到底丢哪条、是否重路由，要看实现。

读监控时同时看 expert 直方图、每卡计算和通信耗时、溢出率、任务 loss。只把第一张图拉平，未必就解决了慢卡，更不能证明模型学得更好。
