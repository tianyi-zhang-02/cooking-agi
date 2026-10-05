# MC 与 TD：等结局，还是先估一步？

**中文** · [English](returns-and-td.en.md)

> 阅读时间：约 6–9 分钟 · 最近审阅：2026-10

Bellman 写得很漂亮，可真实环境往往不会给你转移概率表。手里只有 $(s,a,r,s')$，怎么办？最直接的区别是：**等整段走完再算，还是拿下一状态的估值先更新？**

## Monte Carlo：把这次真的拿到的算清楚

奖励依次是 $[1,0,2]$，$\gamma=0.9$，最后一步后终止：

$$
G_0=1+0.9\times0+0.9^2\times2=2.62.
$$

若当前 $V(s_0)=1$，学习率 $\eta=0.2$，更新后是 $1+0.2(2.62-1)=1.324$。

用被评价的策略跑完一整局，算出的 MC 回报是它的 value 的无偏样本，但波动可能很大，也得等到结束。“无偏”不表示一个样本就准确；截断时漏掉后续回报，或直接混入其他策略的数据，都改变了这个结论的前提。

## TD：一步观察，加一个还不准的估计

时序差分不用等整局结束。若下一状态当前估值是 1.5，一步 TD 的训练目标值与误差为：

$$
y_0=1+0.9\times1.5=2.35,\qquad
\delta_0=y_0-V(s_0)=1.35.
$$

同样步长更新到 1.27。它没有“算错”MC，而是在用另一种估计。自举指用已有估计构造新目标值；能更早更新，但也会把估值误差带回来。

介于两者之间的是 n-step return：

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^nV(s_{t+n}).
$$

到真实终点就停止求和并删去尾项。$n$ 越大，通常越依赖已经发生的奖励，越少依赖下一状态的估值。选多长才合适，要看轨迹噪声和估值质量，并不是越长越好。

## 最容易写错：结束，不一定是终止

| 情况 | 还要不要 bootstrap | 下一状态取哪里 |
| --- | --- | --- |
| 成功或失败，任务真的结束 | 不要 | terminal value 为 0 |
| 外部采样时限到了，任务理论上可继续 | 要 | 截断前最后一个 observation |
| 环境自动 reset | 由结束原因决定 | 不能误用新 episode 的初始 observation |

<div class="drl-lab" data-drl-lab="boundary"><p>静态例子：reward=1、next value=5、γ=0.9；真实终止的 target 是 1，外部截断是 5.5。</p></div>

如果“必须在 20 步内完成”本来就是任务定义，时间到可能是真终止；剩余步数也应进入状态。不要把所有 time limit 都机械地当成同一种情况。

```python
def td_target(reward, next_value, gamma, terminated):
    return reward + gamma * (not terminated) * next_value

assert td_target(1.0, 5.0, 0.9, True) == 1.0
assert td_target(1.0, 5.0, 0.9, False) == 5.5
```

## SARSA 和 Q-learning：下一步听谁的

两者都可用 TD 更新 $Q(s,a)$。SARSA 使用行为策略实际采样的下一动作 $a'$；Q-learning 用下一状态的最大 Q，学习贪心目标策略：

$$
y_{\rm SARSA}=r+\gamma Q(s',a'),\qquad
y_{\rm Q}=r+\gamma\max_{a'}Q(s',a').
$$

这里先写非终止转移；真实终止时，两者的后续项都为 0。

假设 $Q(s',\cdot)=[2,5]$，探索选中了第一个动作。$r=1,\gamma=0.9$ 时，SARSA 的 target 是 2.8，Q-learning 是 5.5。区别不是有没有探索，而是 target 在评价哪个后续行为。

## 检查你的理解

<details markdown="1">
<summary>只把 done 换成 terminated，就把所有边界问题修好了吗？</summary>

不一定。Vector environment 可能已经自动 reset；你还得保存 final observation。GAE 的 bootstrap mask 与跨 episode 递推的 mask 也不一样，下一篇 Actor–Critic 会分开写。先检查你用的环境 API，再决定数组怎么存。

</details>

参考：[Gymnasium 对 termination / truncation 的说明](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)。接下来先看[怎样直接更新 policy](policy-gradients.md)。
