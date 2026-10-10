# MoE：负载均衡

**中文** · [English](load-balancing.en.md)

> 阅读时间：约 12 分钟 · 难度：进阶 · 最近审阅：2026-10-09

Top-k 规定的是“一个 token 去几个专家”，没有规定“每个专家收到多少 token”。假如 8 个 token 都选同两个专家，其他专家还是没活干。负载均衡要解决的就是后一个问题。

这里分开看三件事：容量够不够、训练怎样鼓励均衡，以及在哪个范围里统计。只看全 batch 的直方图，有时会漏掉单条序列里的偏斜。

## 为什么少数 expert 可能越来越忙 {#_1}

某个 expert 碰巧多分到一些 token，训练机会更多，之后又更容易被选中；其他 expert 很少被用到，就可能越来越难追上。这是一种可能出现的正反馈，不是每个 MoE 都必然塌缩。没被选中的 expert 参数通常拿不到这个 token 的任务梯度，但 router 的梯度还取决于门值归一化和辅助目标，不能把两者混为一谈。

下面是个玩具模拟，切换三种做法，看负载怎么变：

<!-- widget:tx-moe-balance -->

## Capacity factor：每个 expert 最多收多少 token

在 Switch 的 top-1 路由中，capacity factor 用于确定每个 expert 的 token 槽位上限，方便安排计算与缓冲区；它不保证各卡耗时相等：

$$C = \frac{T}{N}\cdot\mathrm{CF}$$

$T$ 是这一批参与路由的 token 数，$N$ 是专家数，CF 是 capacity factor，$C$ 是每个专家的槽位预算。实际分配整数槽位时，还要按实现的规则取整。

Switch 的溢出处理跳过对应 expert 分支，token 本身仍走残差路径。小 CF 节省槽位但可能多丢分支，大 CF 减少溢出却可能留更多空位；合理取值需要结合路由分布与吞吐测试。Dropless 实现不丢 expert 分配，也仍要处理负载和缓冲区。不要把 top-1 公式直接当成所有 top-k 实现的容量定义。

## 辅助 loss：把不均衡直接写进损失

Switch Transformer 的辅助 loss：

$$\mathcal L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i$$

- $f_i$：这一批里 argmax 落在 expert $i$ 上的 token 比例，本质是个计数，不可导；
- $P_i$：这一批 token 分给 expert $i$ 的平均 router 概率，可导；
- $\alpha = 10^{-2}$。

两个乘在一起：梯度顺着 $P_i$ 传回 router，推多狠则由真实负载 $f_i$ 说了算。完全均匀时 $f_i = P_i = 1/N$，损失等于 $\alpha$；这是均匀状态的参考值，不是这个乘积式在所有路由分布下的严格下界。[GShard](https://arxiv.org/abs/2006.16668) 也用分配比例乘平均门值：把原本不可导的比例平方中的一个因子换成可导近似，而不是直接对离散计数求导。最早的 sparsely-gated MoE 则用了 importance 和 load 两个变异系数损失。

## 整个 batch 均衡，为什么单条序列还可能不均衡？ {#sequence-balance}

两条序列，各有 4 个有效 token，使用两个专家、top-1 路由。为了便于手算，先把辅助 loss 的系数 $\alpha$ 设为 1：

| 统计范围 | 每个 token 的专家概率 | 分配次数 | 平均概率 |
| --- | --- | --- | --- |
| 序列 A | `[0.9, 0.1]` | `[4, 0]` | `[0.9, 0.1]` |
| 序列 B | `[0.1, 0.9]` | `[0, 4]` | `[0.1, 0.9]` |
| 合并后的 batch | 两种各 4 个 | `[4, 4]` | `[0.5, 0.5]` |

合起来看很均衡，但 A 从头到尾只用了 expert 0，B 只用了 expert 1。按上面的辅助 loss，整个 batch 得到 `1.0`；先分别算两条序列、再取平均，得到 `1.8`。这是两种不同的约束，不是谁把同一个数算错了。

偏斜也不一定就是坏事。如果两条序列属于不同领域，局部偏好可能正是分工。把每条短序列都强行拉平，会压缩这种自由；只看全局，又可能放过单条序列的极端集中。要结合模型质量和系统负载选统计范围，不能见到直方图不平就不断加大系数。

### Top-k 的分母为什么是 $KT$？

一条序列有 $T$ 个有效 token，每个选 $K$ 个不同专家，总共就是 $KT$ 次分配。令 $c_i$ 是 expert $i$ 收到的次数。下面用 $q_i$ 表示分配占比，避免与不同论文中缩放方式不同的 $f_i$ 混淆：

$$\begin{aligned}
c_i &= \sum_{t=1}^{T}\mathbf{1}[i\in S_t],\\
q_i &= \frac{c_i}{KT},\qquad \sum_i q_i=1,\\
P_i &= \frac{1}{T}\sum_{t=1}^{T}p_{t,i},\\
L_{\mathrm{seq}} &= \alpha N\sum_{i=1}^{N}q_iP_i.
\end{aligned}$$

$S_t$ 是 token 的选集；$p_{t,i}$ 在**全部 $N$ 个 routed experts 上归一化**，不是只在入选专家上归一化的输出门值。均匀参考值是 $q_i=P_i=1/N$，此时 loss 为 $\alpha$，不是 0，也不能把它当作所有情况下的严格最小值。

例如 4 个专家、4 个 token、top-2，选集依次为 `{0,1}`、`{0,1}`、`{0,2}`、`{1,3}`。计数是 `[3,3,1,1]`，除以 8 得到 `[0.375,0.375,0.125,0.125]`。若只除以 4，占比之和就变成 2；继续套同一个公式，会多出一个 $K$ 的缩放。

[DeepSeek-V3 §2.1.2，式 17–20](https://arxiv.org/html/2412.19437v1#S2.SS1.SSS2)把 $N$ 吸收进 $f_i=Nq_i$，并将 sigmoid affinity 在全部 routed experts 上归一化后计算 $P_i$。论文的辅助统计按原 affinity 的 top-k 写，带 bias 的实际分发另有公式；接入框架时要核对两处用的选集，不要看到同名 `topk` 就默认相同。下面的教学代码不加 bias，也不限制设备组。

### 为什么不能直接优化计数的平方差？ {#balance-gradient}

`[4,0]` 比 `[2,2]` 偏，这件事可以用平方差衡量。但如果计数来自普通 hard top-k，索引和次数不对 logits 求导。把 $\sum_i(c_i-KT/N)^2$ 直接加进 loss，**不会凭空得到训练 router 的梯度**。它可以是监控指标；要用它训练，需要额外的可导近似或梯度估计。

上面的乘积形式保留真实分配占比 $q_i$ 作为固定系数，让梯度走可导的 $P_i$。对 softmax 概率和单条序列，固定当前选集时可以得到：

$$\begin{aligned}
\frac{\partial L_{\mathrm{seq}}}{\partial z_{t,j}}
&=\frac{\alpha N}{T}p_{t,j}\\
&\quad\cdot\left(q_j-\sum_i q_i p_{t,i}\right).
\end{aligned}$$

回到序列 A，$q=[1,0]$、$p=[0.9,0.1]$、$T=4$。单条序列 loss 对一个 token 的两个 logits 的梯度是 `[0.045,-0.045]`。梯度下降会稍微降低 expert 0 的分数、提高 expert 1 的分数。如果再平均两条序列，每条对应的梯度还要除以 2。

这不是“下一步一定平均分配”的保证。Top-k 在越过边界之前不会换专家，语言模型的主 loss 也在同时更新参数。推理时通常不计算这个训练辅助项，但它已经影响了学到的参数；“推理不算”不等于“对生成质量没有影响”。

<details markdown="1">
<summary>代码里怎样处理 padding、变长序列和归一化？</summary>

[完整 CPU 代码](../code/moe_routing.py)提供 `balance_statistics` 和 `balance_loss`，输入 logits 为 `[batch, time, experts]`，`valid` 是 `[batch, time]` 的布尔 mask：

```python
logits = torch.tensor([[[0.9, 0.1]] * 4, [[0.1, 0.9]] * 4]).double().log()
valid = torch.ones(2, 4, dtype=torch.bool)
sequence_loss = balance_loss(logits, valid, top_k=1, scope="sequence")
batch_loss = balance_loss(logits, valid, top_k=1, scope="batch")
assert abs(sequence_loss.item() - 1.8) < 1e-7
assert abs(batch_loss.item() - 1.0) < 1e-7
```

在完整脚本的函数定义后运行即可。代码默认 `alpha=1` 是为核对算术，**不是推荐训练权重**。`scoring="sigmoid"` 用 `softmax(logsigmoid(logits))` 稳定地计算 sigmoid affinity 的全专家归一化，不是普通 `softmax(logits)`。

- **Padding 不参与次数、概率均值和分母。** 代码先把无效位置的 logits 置零，再做归一化，避免 padding 中的 NaN 污染计算；对应梯度为 0。每条序列至少要有一个有效 token，空序列直接报错。
- **先平均每条序列，和先合并所有 token，不是一个目标。** `sequence` 让每条序列在最终均值中等权；`batch` 汇总有效 token 后才计算两个统计量及其乘积。变长时长序列的权重不同，即使等长，先乘后平均也不等于先平均后乘。
- **Packing 要保留真正的序列边界。** 把两篇文档拼在同一张量行里，不代表它们应该作为一条序列做均衡；需要按文档 ID 分段统计。
- **不要把溢出后的计数当成原始路由需求。** 只统计 capacity 截断后留下的分配，可能掩盖热门专家的过载；至少分别记录尝试分配与实际执行数量。
- **本例的 batch 只是输入张量，不是自动跨卡的 global batch。** 分布式实现还要约定归约范围、有效 token 总数和 loss 缩放，不能把各卡的均值无条件再平均。

</details>

## Router z-loss：管数值，不管均衡

ST-MoE 另加 router z-loss，把每个 token 的 log-sum-exp 往 0 拉：

$$L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^2$$

论文设置系数 $c_z=0.001$。这里 B 是参与计算的 token 数，不是序列数。它约束的是 log-normalizer，不是每个 logit 的绝对值：概率 `[0.9, 0.1]` 的对数作为 logits 时，z-loss 就为 0，但路由仍可能很偏。因此它不能替代负载均衡目标。见 [ST-MoE](https://arxiv.org/abs/2202.08906)。

## 不加辅助 loss：只调 bias

辅助 loss 和语言模型 loss 都会更新 router 分数，想推动的方向却不一定相同。DeepSeek-V3 换了个办法：

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

Kimi K3 用分位数来估计下一步的路由 bias，而不是每次只加减一个固定值。[LatentMoE 的小算例](latent-moe.md#quantile-balancing)解释这个门槛怎么算，也说明为什么它不保证下一批刚好均分。
