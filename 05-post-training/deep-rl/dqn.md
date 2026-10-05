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
    prediction = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_actions = online(next_states).argmax(dim=1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions).squeeze(1)
        expected = rewards + gamma * (~terminated).float() * next_q
    return torch.nn.functional.smooth_l1_loss(prediction, expected)
```

states 是 [batch, state_dim]，actions 是整型 [batch]，terminated 是布尔 [batch]。不要让 [batch, 1] 的 prediction 与 [batch] 的 target 广播成 [batch, batch]。若网络包含 dropout / batch norm，no_grad 并不会自动切到 eval 模式；target 的模式也要明确。

## 几个名字别混

| 改动 | 改了什么 | 没改什么 |
| --- | --- | --- |
| Double DQN | 下一动作的选择与估值解耦 | 仍是 Q-learning |
| Dueling network | 以 value 与 action advantage 组合 Q | 不等于两个 target network |
| Prioritized replay | 更常抽取某些 transition | 会改变采样分布，需考虑校正与偏差 |

探索通常用 $\epsilon$-greedy。训练时随机动作是为了收集信息，评估时是否关掉要明确记录。把探索噪声造成的低分和 policy 本身的低分混在一起，曲线会不好解释。

## 检查这 4 件事，再调参

先测终止 target 是否等于 reward；再确认 target 无梯度；再核对 replay 里的 action 与 state 是否对齐；最后看 target 同步和环境步数的单位。Loss 降低但 return 不涨，优先排查这些语义，不要先堆更大网络。

参考：[Double DQN](https://arxiv.org/abs/1509.06461)。下一篇：[连续动作没法枚举最大值，怎么办？](continuous-control.md)
