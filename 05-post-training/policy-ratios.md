# GSPO 与 ASPO：同一条回答，梯度该怎么分？

**中文** · [English](policy-ratios.en.md)

> 最近审阅：2026-10-08 · 先读：[PPO clipping](rlhf/ppo-clipping.md)与 [GRPO](rlhf/after-rlhf.md)

一段回答整体答对了，其中某个 token 的概率却下降了。训练时，是把整段回答一起推高，还是单独照顾那个 token？GSPO 和 ASPO 从不同方向处理这个问题。先别背缩写，看看它们把权重放在哪里。

## 先分清 3 个策略

| 角色 | 这次用来做什么 | 能不能混用 |
| --- | --- | --- |
| 当前策略 $\pi_\theta$ | 正在求梯度的模型 | 随 optimizer step 改变 |
| 采样 / 旧策略 $\pi_b$ | 生成训练回答；提供分母 | 同步训练常用冻结的 old policy；异步时应记录实际采样版本 |
| Reference $\pi_{\rm ref}$ | KL 或偏好目标的锚点 | 不是因为“也冻结了”就能拿来替代采样策略 |

给定已经采出的回答 $y$，token 概率比为：

$$
\rho_t=\exp\left(\log\pi_\theta(y_t\mid x,y_{<t})-\log\pi_b(y_t\mid x,y_{<t})\right).
$$

两个模型必须看到相同的 prefix。拿当前模型自由生成的另一个回答来相除，没有这个含义。prompt、工具返回和 padding 也不能不加区分地算成模型动作。

## GSPO：先把概率变化汇总到回答级别

[GSPO 论文](https://arxiv.org/html/2507.18071v2)使用长度归一化的序列比值，再做序列级 clipping。一个回答的 token 共享组内 advantage $A$：

$$
s=\exp\left(\frac{1}{T}\sum_{t=1}^{T}\log\rho_t\right),\qquad
J=\min\left(sA,\operatorname{clip}(s,1-\epsilon,1+\epsilon)A\right).
$$

它是 token 比值的**几何平均**，不是算术平均，也不是原始序列概率比 $\prod_t\rho_t$。开 $T$ 次方可以减轻长度对数值范围的影响，但也改变了权重；不能据此宣称它是无偏的整条轨迹 importance sampling。

### 两个 token，变化方向相反

下面是自拟数值，不是模型实测。设 $A=1$，两个 token 的比值为 `[2, 0.5]`：

| 算法 / 量 | 算出来多少 | 说明 |
| --- | --- | --- |
| 原始序列比值 | $2\times0.5=1$ | 整段回答的概率没有变 |
| GSPO 比值 | $\sqrt{2\times0.5}=1$ | 在演示用的 `[0.8, 1.2]` 内，不触发 clipping |
| token-level PPO surrogate 的平均 | $(1.2+0.5)/2=0.85$ | 正 advantage 下，第一个 token 的增益被截住 |

在 GSPO 未裁剪的分支，固定 advantage 后：

$$
\nabla J=\frac{sA}{T}\sum_t\nabla\log\pi_\theta(y_t\mid x,y_{<t}).
$$

这里每个 token 得到相同的外部系数 $sA/T$，**不代表参数梯度相同**：不同位置的网络 Jacobian 仍然不同。这个例子也暴露了代价：序列平均可能掩盖局部的大幅变化。

```python
import math

def clipped_surrogate(ratio, advantage, lower=0.8, upper=1.2):
    if not 0 < lower <= 1 <= upper or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("invalid ratio or clipping interval")
    return min(ratio * advantage, min(upper, max(lower, ratio)) * advantage)

def sequence_ratio(log_ratios, action_mask):
    if len(log_ratios) != len(action_mask) or any(type(flag) is not bool for flag in action_mask):
        raise ValueError("one Boolean action mask per token is required")
    selected = [value for value, active in zip(log_ratios, action_mask) if active]
    if not selected or not all(math.isfinite(value) for value in selected):
        raise ValueError("at least one finite action log-ratio is required")
    return math.exp(sum(selected) / len(selected))

ratios = [2.0, 0.5]
assert math.isclose(sequence_ratio([math.log(value) for value in ratios], [True, True]), 1)
assert math.isclose(sum(clipped_surrogate(value, 1) for value in ratios) / 2, 0.85)
assert clipped_surrogate(0.5, -1) == -0.8
```

这是目标值检查，不是完整 trainer。实际 GSPO 的阈值要按序列比值重新选；这里的 0.2 只是方便手算，不能直接照搬为训练配置。组内全同 reward、错误的 action mask、失真的 reward，也不会因为换成 GSPO 就消失。

## ASPO：正 advantage 的权重，反过来算

[ASPO 论文](https://arxiv.org/html/2510.06062v1)讨论的是 token-level 更新：保留按 advantage 方向的 mask，对正 advantage 使用倒数权重，对负 advantage 保留原比值，并约束极端权重。它的动机是让已经涨得很多与仍落后的 token 不再沿用同一种放大方式。

最容易写错的是梯度。令 $\ell=\log p_\theta$，采样概率为 $p_b$。正 advantage 分支写成：

$$
\widehat\rho=\frac{p_b p_\theta}{\operatorname{sg}(p_\theta^2)},
\qquad
\widehat\rho\text{ 的前向值}=\frac{p_b}{p_\theta},
\qquad
\frac{\partial\widehat\rho}{\partial\ell}=+\frac{p_b}{p_\theta}.
$$

$\operatorname{sg}$ 是 stop-gradient。若直接写可微的 `1 / ratio`，导数反而是负数，就把提高好 token 概率的方向写反了。**数值一样，不等于训练行为一样。**

### 算一次局部更新

假设采样时概率 0.2，现在是 0.1，$A=1$。普通比值是 0.5；ASPO 正分支的权重是 2。在没有触发 mask 或权重上限的情况下，它们乘在 $\nabla\log p_\theta$ 前的系数分别为 0.5 和 2。

下面只验证 stop-gradient 的局部含义：求导时固定分母，不是每次扰动后都重新计算分母。

```python
import math

behavior_probability = 0.2
current_probability = 0.1
frozen_square = current_probability ** 2
current_logp = math.log(current_probability)
step = 1e-6

def detached_denominator_value(log_probability):
    return behavior_probability * math.exp(log_probability) / frozen_square

derivative = (
    detached_denominator_value(current_logp + step)
    - detached_denominator_value(current_logp - step)
) / (2 * step)
assert math.isclose(derivative, 2.0, rel_tol=1e-6)
```

没有 mask、dual clipping 和 loss reduction 的这几行，不能叫完整 ASPO 实现。[作者代码](https://github.com/wizard-III/Archer2.0)适合继续对照。论文 v1 的 token-masking 文字中有一处上界符号与标准 PPO 描述不一致；接入训练前应核对具体实现，而不是把那句话直接复制成条件判断。

## 怎么判断哪种改法值得试

先记录同一批 rollout 的 reward、长度、比值分布和有效 token 数，再换 loss。尽量保持数据、生成预算、初始化与评估相同。

| 观察到的现象 | 值得检查 | 不能仅凭什么下结论 |
| --- | --- | --- |
| 少数 token 比值极端 | 序列聚合是否改善稳定性；是否掩盖局部异常 | 不能只看平均 ratio 接近 1 |
| 正负样本的权重分布很不一样 | 按 advantage 符号画分布，核对 detach 与 clipping | 不能把所有熵下降都叫训练崩溃 |
| 有效梯度越来越少 | 按 token / sequence 分别统计 mask 比例 | 两种粒度的 clip fraction 不能直接横比 |
| 分数涨了、回答也长了很多 | 固定推理预算，再比较正确率与失败类型 | 更长不自动等于更会推理 |

这些方法修改的是更新规则，不会把错误的评分标准变成正确标准。接着读 [DAPO 的采样与归一化](dapo.md)，再看[异步训练中的旧数据](async-policy-learning.md)。
