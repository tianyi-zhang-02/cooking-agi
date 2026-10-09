# DAPO：采样、长度和 clipping，为什么会改变训练？

**中文** · [English](dapo.en.md)

> 最近审阅：2026-10-08 · 先读：[GRPO 的组相对信号](rlhf/after-rlhf.md)

两份训练脚本都写着 GRPO，效果却差很多。除了学习率和模型大小，还该看什么？这篇拿几个很小的 batch，看看哪些看似“工程细节”的地方，其实已经改变了优化目标。

## DAPO 的 4 个改动

[DAPO 原论文](https://arxiv.org/html/2503.14476v1)把 4 件事组合起来：拆开 clipping 上下界；继续采样以补足有组内奖励差异的 prompt；按有效 token 聚合 loss；处理超长回答的奖励噪声。论文的数学推理设置还移除了 Reference KL 项。这是特定训练方案，不代表所有任务都不需要 KL。

## 先看 clip 到底 clip 了什么

对 token 比值 $\rho$、advantage $A$，最大化：

$$
j(\rho,A)=\min\left(\rho A,\operatorname{clip}(\rho,1-\epsilon_l,1+\epsilon_h)A\right).
$$

设 $A=1$，当前比值为 1.25。上界 1.2 时，目标值是 1.2，沿着继续增大该比值的方向已经没有直接收益；上界 1.3 时，目标值仍是 1.25，可以继续得到这项收益。下界可以保持不变。

若 $A=-1$，同样的 1.25 会给出 -1.25，仍有把错误方向拉回来的梯度。**PPO 并没有把模型概率硬锁在区间里**：参数共享、其他 token 的梯度和多次更新，都可能让某些比值越界。

低概率 token 的绝对变化可以很小。例如从 0.01 到 0.012，已是 1.2 倍；从 0.5 到 0.6 也是 1.2 倍。Clip-Higher 放宽的是乘法幅度，不是给所有 token 加同一个概率，也不是保证熵不会下降。

## 动态采样：保留的是组，不是只留正确答案

同一 prompt 采 4 次，二元奖励可能是：

| prompt | 4 次奖励 | 组相对 policy gradient 有无区分信号 |
| --- | --- | --- |
| A | `[0, 0, 0, 0]` | 没有 |
| B | `[0, 1, 0, 1]` | 有；对和错都要保留 |
| C | `[1, 1, 1, 1]` | 没有 |
| D | `[0, 0, 1, 0]` | 有 |

若希望凑够 4 个有效 prompt group，这一轮只收到了 2 个，还得继续生成。已经为 A、C 花掉的采样时间不能从成本统计里删掉。

对独立、同成功概率 $p$ 的 $G$ 次二元采样，出现混合组的概率为：

$$
P(\text{混合组})=1-p^G-(1-p)^G.
$$

$p=0.5,G=4$ 时是 0.875；$p=0.99$ 时只有约 0.0394。现实采样未必独立，这个公式只是说明：太容易或太难的题，反复补采的成本会很高。

过滤后的 prompt 分布也变了。它偏向当前模型还会时对时错的问题，不能再把保留组上的正确率当成整个任务分布上的正确率。评估应保留固定题集。

## 两种平均方式，可以直接算出不同结果

设短回答 2 个有效 token，每个 token 的 surrogate 都是 1；长回答 6 个，每个都是 3。

- 先每条回答求平均，再对回答平均：$(1+3)/2=2$。
- 把 8 个有效 token 放在一起平均：$(2\times1+6\times3)/8=2.5$。

第一种方式让两条回答总权重相同；第二种让每个 token 的基础权重相同，因此长回答的总权重更大。不是一个“更准确的除法”替换了错误除法，而是你改变了权重分配。

```python
import math

def aggregate_tokens(values, masks):
    if not values or len(values) != len(masks):
        raise ValueError("a nonempty batch with matching masks is required")
    selected_rows = []
    for row, mask in zip(values, masks):
        if len(row) != len(mask) or any(type(flag) is not bool for flag in mask):
            raise ValueError("one Boolean mask per token is required")
        selected = [value for value, active in zip(row, mask) if active]
        if not selected or not all(math.isfinite(value) for value in selected):
            raise ValueError("each response needs finite action tokens")
        selected_rows.append(selected)
    response_mean = sum(sum(row) / len(row) for row in selected_rows) / len(selected_rows)
    token_mean = sum(sum(row) for row in selected_rows) / sum(map(len, selected_rows))
    return response_mean, token_mean

values = [[1, 1, 999], [3, 3, 3, 3, 3, 3]]
masks = [[True, True, False], [True] * 6]
assert aggregate_tokens(values, masks) == (2.0, 2.5)
```

999 代表 padding 或非动作位置的垃圾值，它不该影响结果。全 mask 的样本如何处理，也应明确：这里选择报错；生产代码可以跳过，但不能悄悄把它算作零分样本。

### 多卡上还有一次容易漏掉的加权

rank 0 有 2 个 token，局部均值 1；rank 1 有 6 个 token，局部均值 3。直接平均两个 rank 的局部均值仍是 2，不是全局 token 均值 2.5。

若 DDP 最后平均各 rank 梯度，世界大小为 $W$，全局有效 token 数为 $N$，可以让每个 rank 的可微局部和乘 $W/N$，再由 DDP 平均。具体框架如果执行的是 sum、梯度累积或别的 reduction，系数也要对应调整。分母必须覆盖你声明要平均的那一批数据。

这是一般的分布式求均值问题，不是 DAPO 独有，也不需要真正开多卡就能先写一个算例测出来。

## 生成被截断，不等于推理一定错了

长度上限让训练可控，但一个没写完的正确解法与已经完成的错误答案不同。DAPO 讨论了超长样本过滤与 soft overlong punishment。后者在接近上限的一个区间里逐步扣分，而不是在某个位置突然把所有情况打成同一种错误。

用教学上限 100、缓冲长度 20 演示：80 以内不扣分，90 扣 0.5，100 及以上扣 1。它是加入任务 reward 的长度项，不是替代正确性检查，也没有取消生成长度上限。

```python
def length_penalty(length, limit, buffer):
    if not all(type(value) is int for value in (length, limit, buffer)):
        raise ValueError("lengths must be integers")
    if length < 0 or not 0 < buffer <= limit:
        raise ValueError("invalid lengths")
    return -min(1.0, max(0.0, (length - (limit - buffer)) / buffer))

assert [length_penalty(value, 100, 20) for value in (70, 80, 90, 100, 110)] == [0, 0, -0.5, -1, -1]
```

若只看平均 reward，模型可能通过写短来少扣分，却没有变得更会解题。因此应把正确率、长度惩罚、自然结束率和截断率分别记录。

## 如何做一个看得懂的 ablation

| 实验 | 固定什么 | 多记什么 |
| --- | --- | --- |
| 只改 clipping 上界 | 相同 rollout 与 advantage | 正负 advantage 各自的裁剪率、更新范数 |
| 只改 loss reduction | 相同 token 值与 mask | 单卡 / 分片后梯度是否一致 |
| 开动态采样 | 相同总生成预算与独立测试集 | 丢弃组数、总生成 token、保留题目分布 |
| 加长度项 | 相同最大输出预算 | 正确率与长度项分开，不能只报总 reward |

先把这些局部检查做对，再运行训练对比。本文的小算例只检验目标和数据处理，没有复现论文的模型训练。2026 年继续读 [GSPO / ASPO](policy-ratios.md)与 [SAO](async-policy-learning.md)时，也可以沿用这套问题：到底改了采样、权重，还是数据到达训练器的时间？

实现入口：[DAPO 项目页](https://dapo-sia.github.io/) · [verl DAPO recipe](https://github.com/volcengine/verl/tree/main/recipe/dapo)。实际复现要记录 commit，不把随时变化的 main 当作固定版本。
