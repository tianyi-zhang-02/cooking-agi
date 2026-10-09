# Policy Gradient：奖励怎样变成梯度？

**中文** · [English](policy-gradients.en.md)

> 阅读时间：约 6–9 分钟 · 最近审阅：2026-10

Q-learning 先学哪个动作值钱，再取最大值。另一条路是直接改变选动作的概率：这次比预期好，下次多选一点。难点在于环境不一定可微——梯度从哪里来？

## 不穿过环境，改走轨迹的概率

先看长度为 $T$ 的有限回合，目标暂时不折扣：$J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)]$。环境转移和初始分布不依赖 $\theta$：

$$
p_\theta(\tau)=\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

在允许交换微分与积分的正则条件下，用 $\nabla p=p\nabla\log p$：

$$
\nabla J
=\mathbb E_\tau[R(\tau)\nabla\log p_\theta(\tau)]
=\mathbb E_\tau\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)R(\tau)\right].
$$

环境项不需要求导。策略梯度改变的是“这类轨迹以后出现的概率”，不是对这一次已经发生的奖励反传。若动作可用重参数化且 Q 可微，后面 SAC 会用另一种梯度路径。

## 为什么可以只看未来、再减 baseline

动作不会改变已经发生的奖励，所以可以用从当前步起的回报（reward-to-go）$G_t=\sum_{k=t}^{T-1}r_k$ 替换整段回报。再减去一个只依赖状态的状态基线 $b(s_t)$：

$$
\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)b(s)]
=b(s)\nabla\sum_a\pi(a\mid s)=0.
$$

因此期望梯度不变；方差却可能更小。$V^\pi(s)$ 是常见 baseline，不保证在所有梯度加权意义下方差最小。若 baseline 直接依赖当前采样动作，这个证明就失效，不能随便减。

折扣也要一致：若目标是 $\mathbb E[\sum_t\gamma^t r_t]$，而 $G_t$ 从当前步重新计折扣，直接按有限轨迹推导时外面还要有 $\gamma^t$。文献有时把它吸收到状态访问分布里，别把两种约定混起来。

## 一个只有两种动作的算例

两个动作的奖励分别为 1、3，初始概率各为 0.5。对 softmax 的两个 logits，期望回报是 2，梯度是：

$$
\frac{\partial J}{\partial z_i}=p_i(r_i-\bar r),\qquad
\nabla_z J=[-0.5,\ 0.5].
$$

梯度上升会降低第一个 logit、提高第二个。注意奖励 1 也是正的，但相对平均值 2，它应该被减少，而不是“只要得分大于 0 就奖励”。

## 最小 PyTorch 更新

下面只写一次 on-policy 更新的 loss，不是完整训练器。`log_prob` 是当前策略对采样动作的对数概率，`advantage` 是固定的评分信号；两者都是长度相同的一维张量。这里沿用前面的不折扣目标，且 batch 中没有 padding。若采用有限轨迹折扣目标，还要加入前面说的 $\gamma^t$ 权重；变长轨迹也要明确按步还是按整条轨迹归约。

```python
def actor_loss(log_prob, advantage):
    if log_prob.ndim != 1 or log_prob.shape != advantage.shape or log_prob.numel() == 0:
        raise ValueError("log_prob and advantage must be matching nonempty vectors")
    return -(log_prob * advantage.detach()).mean()
```

负号把梯度上升变成优化器的梯度下降。detach 表示这一步只改变动作概率，不让 Actor 顺着 advantage 的计算图“改评分”。Critic 用自己的 loss 更新。

## 旧数据为什么不能直接重放

旧策略 $\mu$ 产生的样本分布不等于新策略 $\pi$。理论上可以用重要性采样重新加权：

$$
w(\tau)=\prod_t\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}.
$$

前提是目标策略可能走到的轨迹在行为分布中有支持。长轨迹的乘积容易导致极大方差；只使用当前动作的 ratio 也不自动修复状态分布偏移。PPO 的短窗口数据重用与 clipping 是务实折中，不是任意旧数据都能精确校正。

## 为什么过去的奖励可以从梯度里拿掉？

固定动作发生前的历史 $h_t$，过去奖励 $R_{<t}$ 已经是一个常数。对接下来采样的动作取条件期望：

$$
\mathbb E[\nabla\log\pi(a_t\mid s_t)R_{<t}\mid h_t]
=R_{<t}\sum_a\pi(a\mid s_t)\nabla\log\pi(a\mid s_t)=0.
$$

所以删除过去奖励不改变期望梯度，却去掉了无法由当前动作解释的随机量。这也是信用分配的一个具体例子：最后一步动作不应该因第一步已经拿到的奖励而被表扬。

再看 baseline 的作用。把前面奖励 $[1,3]$ 全加 100，最优动作不变，但直接用 return 时，两个动作采样一次都可能获得很大的正向更新。减去同样平移的期望回报以后，advantage 仍是 $[-1,1]$。我们保留了相对好坏，减少无用尺度对估计的影响；这不等于任意形式的奖励变换都保持最优策略。

## 一次 rollout，多次更新，什么时候开始不对？

采样时用 $\pi_{\rm old}$，第一次更新后参数已经变了。继续把同一批样本当作“当前策略刚采到的”，就不再是原来的 on-policy 估计。

| 操作 | 对应的假设 | 常见错误 |
| --- | --- | --- |
| 收集轨迹 | 保存产生动作的策略版本、log-prob 和边界 | 更新后重算 old log-prob，把概率比弄成 1 |
| 构造 return / advantage | 对齐 reward、状态和采样动作 | 误把下一条轨迹的奖励接进来 |
| 更新 Actor | 评分信号固定，loss 的符号与优化器一致 | 让梯度通过 advantage 去修改评分 |
| 再次使用旧数据 | 明确概率比、近似范围和停止条件 | 以为多跑几个 epoch 就只是监督学习 |

最小调试可以先用已知奖励的两动作 bandit：固定种子和数据，检查期望梯度方向是否与手算一致，再加 baseline、Critic 和多步环境。大环境的回报曲线太复杂，不适合拿来辨认一个负号是否写反。

## 从这一步走向哪里

REINFORCE 不需要学环境，机制直接；代价是 return 方差和样本开销。引入 Critic 是为了更好地估计更新方向，接着读 [Actor–Critic 与 GAE](actor-critic-gae.md)。

步长为什么不能只靠一个 learning rate？接着看 [Natural Gradient 与 TRPO](trust-region.md)，从一个硬币策略推到 KL 约束下的更新。

参考：[策略梯度推导](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) · [PPO 原始论文](https://arxiv.org/abs/1707.06347)。
