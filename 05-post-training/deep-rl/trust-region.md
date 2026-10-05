# 策略该改多大：从 Natural Gradient 到 PPO

**中文** · [English](trust-region.en.md)

> 阅读时间：约 8 分钟 · 最近审阅：2026-10

知道梯度往哪里走，还没回答“这一步迈多大”。同样把参数加 0.1，有的动作概率几乎不动，有的却变化很大。**限制参数移动，不一定是在限制行为变化。**

先读 [Policy Gradient](policy-gradients.md) 和 [GAE](actor-critic-gae.md)。这一篇解释步长，不重复推导 advantage。

## 为什么旧数据只在附近好用

固定旧策略收集的状态和动作，定义局部替代目标：

$$
L(\theta)=\mathbb E_{s\sim d_{\rm old},\,a\sim\pi_{\rm old}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)}A^{\pi_{\rm old}}(s,a)\right].
$$

这里 $d_{\rm old}$ 是归一化的折扣状态访问分布，advantage 暂时看作精确值。对无限时域折扣回报，在旧参数处有 $\nabla J=\nabla L/(1-\gamma)$；方向一致。但走远以后，新策略会到不同的状态，固定旧状态分布的目标不再精确代表新回报。

概率比值能校正**这个状态里选动作的概率**，不会顺便把旧状态变成新策略实际会遇到的状态。支持集也必须覆盖：旧策略从不选择的动作，不能靠除一个零概率补出来。

## 换一把尺子：看策略分布变了多少

考虑旧策略到新策略的平均 KL：

$$
\bar D(\theta)=\mathbb E_{s\sim d_{\rm old}}
D_{\rm KL}\!\left(\pi_{\rm old}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\right).
$$

旧参数处 KL 为 0，一阶导数也为 0。光滑、支持集不随参数突变时，二阶展开得到：

$$
\bar D(\theta_{\rm old}+\Delta)\approx\tfrac12\Delta^\top F\Delta,
\qquad F=\mathbb E_{s,a\sim d_{\rm old},\pi_{\rm old}}
\left[\nabla\log\pi_{\rm old}(a\mid s)\nabla\log\pi_{\rm old}(a\mid s)^\top\right].
$$

$F$ 是 Fisher 信息矩阵。它让“会显著改变动作概率的参数方向”付出更高代价。这里的 $1/2$ 来自 Taylor 展开；KL 的方向和取期望的分布都要写清楚，不能只记一个矩阵符号。

## 把方向和步长一起算出来

令 $g=\nabla L(\theta_{\rm old})$，做一个局部近似问题：

$$
\max_\Delta g^\top\Delta
\quad\text{s.t.}\quad \tfrac12\Delta^\top F\Delta\le\delta.
$$

当 $F$ 正定、$g\ne0$ 时，拉格朗日条件给出 $g-\eta F\Delta=0$，所以方向是 $F^{-1}g$。再把约束代回去：

$$
\Delta^*=\sqrt{\frac{2\delta}{g^\top F^{-1}g}}F^{-1}g.
$$

这就是自然梯度的方向加上 KL 预算决定的缩放。实践中不直接建一个巨大的逆矩阵：用 Fisher-vector product 和共轭梯度求解，并常加 damping 改善病态问题。加了 damping、用了有限样本，已经不是上面理想问题的精确解。

## 用一个硬币策略手算

动作 1 得 1 分，动作 0 得 0 分，$p=\pi(1)=\sigma(\theta)$。从 $\theta=0$ 开始：

| 量 | 数值 | 原因 |
| --- | --- | --- |
| 当前 $p$ | 0.5 | $\sigma(0)=0.5$ |
| 回报梯度 $g$ | 0.25 | $J=p$，$dp/d\theta=p(1-p)$ |
| Fisher $F$ | 0.25 | Bernoulli 的 score 是 $a-p$，其二阶矩为 $p(1-p)$ |
| KL 预算 $\delta$ | 0.01 | 例子里人为指定 |
| 建议步长 $\Delta$ | 0.28284 | $\sqrt{2\delta/F}$ |
| 更新后 $p$ | 0.57024 | $\sigma(0.28284)$ |

用真实 Bernoulli KL 复核，约为 0.009967。这次很接近 0.01，但局部二阶近似不保证任何一步都满足真实 KL 预算。

<div class="drl-lab" data-drl-lab="trust-step"><p>调 logit 步长，比较动作概率、真实 KL 和二阶近似。Δ=0.3 时真实 KL 约 0.01121，已超过 δ=0.01；改变预算不会改变这个提议本身。</p></div>

<details markdown="1">
<summary>如果直接把 logit 加 2，会怎样？</summary>

新概率约 0.8808，看起来进步更多，真实 KL 却约 0.4338，远超预算。这个无噪声 bandit 的奖励恰好已知，所以不会因此翻车；复杂环境里 advantage 有误差，新状态也未知，不能从这个小例子推出“总该走大步”。

</details>

## TRPO 与 PPO，不是同一种保证

TRPO 用局部目标和 KL 约束指导更新，再通过回溯搜索检查采样目标是否改善、测得的 KL 是否可接受。实际算法使用平均 KL、有限数据和近似求解，不能把理论的单调改进结论原封不动地当作每次训练的保证。

PPO-Clip 则用更便宜的一阶目标。记概率比为 $\rho_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\rm old}(a_t\mid s_t)$；这里不用 $r_t$，以免和即时奖励混淆：

$$
L_{\rm clip}=\mathbb E\left[\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

它在某些方向上不再奖励过大的比值变化，**不是把每个概率比值硬锁在区间里**，更不是严格 KL 约束。共享参数、其他样本和其他 loss 仍可把比值推远。因此还要观察 KL、clip fraction、entropy 和真实任务表现。图解见 [PPO clipping](../rlhf/ppo-clipping.md)。

## 写实现时，先检查这几项

- Old log-prob 在这批数据的更新期间固定，不随 optimizer step 重算为“最新旧值”。
- Advantage 不参与 Actor 的反传；概率比值用 log-prob 差再取 exp。
- Mask 和 reduction 与实际有效样本一致；多 epoch 更新不等于重新采样。
- 明确报告平均 KL 还是其他估计量；平均值小不代表每个状态都小。

参考：[TRPO 原始论文](https://proceedings.mlr.press/v37/schulman15.html) · [PPO 原始论文](https://arxiv.org/abs/1707.06347)。本文推导局部约束步，不展开完整单调改进界与共轭梯度实现。
