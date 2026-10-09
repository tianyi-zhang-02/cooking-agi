# DQN：用神经网络学 Q，为什么容易不稳？

**中文** · [English](dqn.en.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-10

动作只有“左、右、停”时，网络可以一次输出 3 个 Q 值，直接选最大值。看起来像普通回归，但标签不是固定真值：**标签里也有网络自己的估计。**

想先看清“回归 loss 降了，为什么 value 还可能跑偏”，从 [Fitted Q 的两状态反例](fitted-q.md)开始；这一篇接着讲 DQN 如何缓解这些问题。

## 把一条 transition 变成 loss

回放缓冲区存 $(s,a,r,s',d)$。在线网络 $Q_\theta$ 预测当前动作的价值。

慢一些变化的目标网络 $Q_{\bar\theta}$ 构造标签：

$$
y=r+\gamma(1-d)\max_{a'}Q_{\bar\theta}(s',a'),\qquad
L(\theta)=\mathbb E[(Q_\theta(s,a)-\mathrm{stopgrad}(y))^2].
$$

$d$ 是真实终止，不是所有 reset。经验回放减弱连续样本的强相关，也重用交互；目标网络减慢目标漂移。它们缓解问题，不保证神经网络一定收敛。

这叫半梯度：只沿当前 Q 反传，不沿 target 反传。不是对完整 Bellman residual 的所有出现位置同时求导。

## DQN 与 Double DQN，差在选择和估值

最大值会更容易挑中被噪声高估的动作。Double DQN 用 online network 选动作，再让 target network 估值：

$$
a^*=\arg\max_a Q_\theta(s',a),\qquad
y_{\rm DDQN}=r+\gamma(1-d)Q_{\bar\theta}(s',a^*).
$$

取 online Q 为 $[5,4]$，target Q 为 $[2,6]$，$r=1,\gamma=0.9$。DQN 选 target 的 6，得到 6.4；Double DQN 选 online 排第一的动作，再取 target 对它的 2，得到 2.8。

<div class="drl-lab" data-drl-lab="double-q"><p>静态对照：DQN target=6.4，Double DQN target=2.8。交互可改变 online 的排序。</p></div>

这里没有真实 $Q^*$，所以**不能凭 target 小就说更准**。这个例子只展示解耦机制。两个网络依然相关，也可能低估。

## 写代码时看 shape 和梯度

下面沿用相同的 Double DQN target，但把公式里的平方误差换成 Smooth L1（这里等价于阈值为 1 的 Huber loss），减弱很大 TD 误差对梯度的影响。它没有改变选动作和估值的规则。

```python
import torch

def double_dqn_loss(online, target, states, actions, rewards,
                    next_states, terminated, gamma=0.99):
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a nonempty vector")
    if not rewards.is_floating_point():
        raise ValueError("rewards must be floating-point")
    if actions.shape != rewards.shape or terminated.shape != rewards.shape:
        raise ValueError("actions, rewards, and terminated must have shape [batch]")
    if actions.dtype != torch.long or terminated.dtype != torch.bool:
        raise ValueError("actions must be int64 and terminated must be boolean")
    if states.ndim != 2 or states.shape != next_states.shape or states.shape[0] != rewards.numel():
        raise ValueError("states and next_states must have matching [batch, state_dim] shapes")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    prediction = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = torch.zeros_like(rewards)
        continuing = ~terminated
        if gamma > 0 and continuing.any():
            successors = next_states[continuing]
            next_actions = online(successors).argmax(dim=1, keepdim=True)
            next_q[continuing] = target(successors).gather(1, next_actions).squeeze(1)
        expected = rewards + gamma * next_q
    return torch.nn.functional.smooth_l1_loss(prediction, expected)
```

这里 `states` 是 `[batch, state_dim]`，`actions` 是 int64 `[batch]`，`rewards` 是浮点 `[batch]`，`terminated` 是布尔 `[batch]`；示例使用同设备、同浮点精度的普通 MLP。形状检查会拒绝 `[batch, 1]` 的奖励，避免它与 `[batch]` 的 Q 广播出一张 `[batch, batch]` 的错误表。

真实终止的后继状态不进入网络：target 就是 reward。若先算出 NaN 再乘零，仍会得到 NaN。外部截断不是终止，必须留下 reset 前的有效 final observation。网络若含 dropout / BatchNorm，还要另定模式；`no_grad` 不等于 `eval()`，改变有效 batch 也会影响 BatchNorm 统计。这个小实现不替你管理这些训练状态。

## 几个名字别混

| 改动 | 改了什么 | 没改什么 |
| --- | --- | --- |
| Double DQN | 下一动作的选择与估值解耦 | 仍是 Q-learning |
| Dueling network | 以 value 与 action advantage 组合 Q | 不等于两个 target network |
| Prioritized replay | 更常抽取某些 transition | 会改变采样分布，需考虑校正与偏差 |

探索通常用 $\epsilon$-greedy。训练时随机动作是为了收集信息，评估时是否关掉要明确记录。把探索噪声造成的低分和 policy 本身的低分混在一起，曲线会不好解释。

## 为什么 max 会把零均值噪声变成高估？

两个动作的真实价值都是 0，误差各自独立地以相同概率取 +1 或 −1：

| 两个误差 | 最大值 |
| --- | --- |
| −1，−1 | −1 |
| −1，+1 | +1 |
| +1，−1 | +1 |
| +1，+1 | +1 |

每个动作的误差均值为 0，最大值的期望却是 0.5。选择过程留下了看起来最好的噪声。若换一组独立、零均值误差评价被选中的动作，这个例子里的估值期望会回到 0。

Double DQN 的网络并不真正独立，所以这解释的是动机，不是无偏性证明。目标网络太接近在线网络，误差可能高度相关；太久不更新，又可能过时。

## Replay 和 target 的“慢”，分别慢在哪里？

Replay 存历史 transition，不是永久固定的 target。抽出 $(s,a,r,s')$ 后，再按当前约定的目标网络构造标签。随机打散减弱连续样本的相关性，却不保证旧数据覆盖新策略会去的状态。

目标网络可以周期性硬复制，也可以软更新 $\bar\theta\leftarrow(1-\tau)\bar\theta+\tau\theta$。同步周期要说明按环境步还是梯度步计数：每次交互训练 10 次，与训练 1 次时的“1000 步”，不是同样节奏。

| 增加什么 | 可能的帮助 | 额外风险 |
| --- | --- | --- |
| Replay 容量 | 历史更丰富 | 数据变旧，稀有成功被淹没 |
| 每次交互的更新次数 | 更充分利用数据 | 过拟合有限 replay，放大 Q 误差 |
| 同步间隔 | 目标暂时更稳 | 目标过时，信息传播变慢 |

样本效率比较至少同时记录环境交互量和梯度更新量。只说“训练一小时”，分不清多用了数据还是计算。

## 检查这 4 件事，再调参

先测终止 target 是否等于 reward；再确认 target 无梯度；再核对 replay 里的 action 与 state 是否对齐；最后看 target 同步和环境步数的单位。Loss 降低但 return 不涨，优先排查这些语义，不要先堆更大网络。

参考：[Double DQN](https://arxiv.org/abs/1509.06461)。下一篇：[连续动作没法枚举最大值，怎么办？](continuous-control.md)
