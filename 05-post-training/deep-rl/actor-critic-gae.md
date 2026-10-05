# Actor–Critic 与 GAE：把奖励归因说清楚

**中文** · [English](actor-critic-gae.en.md)

> 阅读时间：约 10–12 分钟 · 最近审阅：2026-10

一整段轨迹得了高分，不代表里面每个动作都好。Actor–Critic 想估计的是：**在当时那个状态，选这个动作的预期回报，比按原策略选择高多少？** Actor 负责选动作，Critic 提供估值。下面先讲状态价值 Critic 配合随机策略的版本；后面的 TD3、SAC 会改用 Q Critic。

## 两个网络分别接收什么

| 部件 | 输入 | 输出 | 更新信号 |
| --- | --- | --- | --- |
| Actor $\pi_\theta$ | 当前状态 | 动作分布 | 停止梯度的 advantage |
| Critic $V_\phi$ | 当前状态 | 预计未来 return | 停止梯度的 return target |
| 环境 | 动作 | 下一状态与 reward | 不由上述 loss 训练 |

可以共享编码器，但此时两个 loss 会通过共享参数互相影响，不能再假装它们完全独立。Value 的 MSE 很小，也不代表策略选得好；先看 target 是否本身就错了。

<div class="drl-paths" aria-label="Actor 与 Critic 的两条更新路径">
<section class="drl-path"><h3>Actor：改变以后怎么选</h3><ol><li>状态 → 动作分布</li><li>采样动作 → 环境反馈</li><li>固定 advantage × log-prob</li><li>只沿 log-prob 更新策略</li></ol><small>评分信号要 detach；不是对环境 reward 直接反传。</small></section>
<section class="drl-path"><h3>Critic：修正原来的预期</h3><ol><li>状态 → 当前 value</li><li>奖励 + 尾部估值 → 固定 target</li><li>比较预测与 target</li><li>沿当前 value 更新估值</li></ol><small>Target 要 detach；共享编码器时仍要考虑梯度相互影响。</small></section>
</div>

## 从 TD residual 到 advantage

优势函数 $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ 比较的是：这个动作比在该状态按策略选择的平均预期好多少。若 Critic 恰好是 $V^\pi$，在 $(s_t,a_t)$ 条件下，一步 TD 误差的期望就是 $A^\pi(s_t,a_t)$：

$$
\delta_t=r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t),\qquad
\mathbb E[\delta_t\mid s_t,a_t]=Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

实际的 $V_\phi$ 只是近似值，单步估计会受到它的误差影响。广义优势估计把后面一串 TD 误差按距离加权：

$$
\hat A_t^{\rm GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}
=\delta_t+\gamma\lambda\hat A_{t+1}.
$$

$\lambda=0$ 只用一步；$\lambda=1$ 时，完整终止轨迹中间的 value 项逐项抵消，最后剩下 $G_t-V(s_t)$。若采样片段被截断，尾部仍保留估值，不是完整 MC。中间值是在权衡估值误差与采样噪声，不存在永远最好的 $\lambda$。

## GAE 为什么会长成这个加权和

先把 $n$ 步 residual 加起来。中间的 value 一正一负抵消，留下：

$$
\hat A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}
=\sum_{l=0}^{n-1}\gamma^l r_{t+l}+\gamma^n V(s_{t+n})-V(s_t).
$$

这就是 $n$ 步回报减去起点的估值。GAE 不只挑某个 $n$，而是混合不同长度。取同一 episode 内剩下的 $N$ 步，不能跨过 reset；如果片段结束时任务还没结束，尾部保留 bootstrap：

$$
\hat A_t^{\rm GAE}
=(1-\lambda)\sum_{n=1}^{N-1}\lambda^{n-1}\hat A_t^{(n)}
+\lambda^{N-1}\hat A_t^{(N)}.
$$

最后一项接住了剩余权重，不能把有限轨迹最后一项也机械乘 $(1-\lambda)$。这些权重相加为 1；收集每个 residual 的系数，就得到 $(\gamma\lambda)^l$。

| 用剩下 3 步举例，λ=0.8 | 混合权重 | 实际看到多远 |
| --- | --- | --- |
| 1-step advantage | 0.2 | 1 个 reward，再信 Critic |
| 2-step advantage | 0.16 | 2 个 reward，再信 Critic |
| 3-step advantage | 0.64 | 到当前片段末尾，再处理终止或 bootstrap |

这样也容易理解两端：$\lambda=0$ 全压在第一步；$\lambda=1$ 全压在最长的那段。不要把“更长”直接翻译成“更准”：回报噪声与 Critic 误差都在变。

## 手算一段，再试试滑块

取 $\delta=[1,2,-1]$，$\gamma=0.9,\lambda=0.8$：

$$
\hat A_0=1+0.72\times2+0.72^2\times(-1)=1.9216.
$$

<div class="drl-lab" data-drl-lab="gae"><p>静态例子：λ=0 时 A₀=1；λ=0.8 时 1.9216；λ=1 时 1.99。滑块展示每个 residual 的贡献。</p></div>

这里的 residual 是固定的教学数据，不会随着滑块重训 Critic。改变 $\lambda$ 只改变这次估计方式。

## 为什么需要两个 mask

一个 mask 决定**是否继续使用下一状态的估值**；另一个决定**后面的 TD 误差是否属于同一段轨迹**。真正终止时两者都关闭。外部截断可以 bootstrap，但 GAE 不能跨过 reset，把下一局的奖励算到这一局里。

| 边界 | 使用下一状态 value？ | 接着加下一条 residual？ | 用哪个观测 |
| --- | --- | --- | --- |
| 正常继续 | 是 | 是 | 下一状态 |
| 任务真实终止 | 否 | 否 | 不再 bootstrap |
| 外部时限截断并 reset | 是 | 否 | 截断前 final observation |
| Buffer 满了，环境未结束 | 是 | 当前片段里停止 | Buffer 尾部的下一状态 |

如果“时间用尽”本来就是任务定义的终止条件，则属于真实终止；有限时域任务的剩余时间通常也应进入状态。表里的外部截断，是任务本可继续、只是采集器暂时停了。

```python
def gae(rewards, values, next_values, terminated, truncated,
        gamma=0.99, trace_decay=0.95):
    lengths = {len(part) for part in
               (rewards, values, next_values, terminated, truncated)}
    if len(lengths) != 1:
        raise ValueError("trajectory arrays must have equal lengths")
    result = [0.0] * len(rewards)
    carry = 0.0
    for step in reversed(range(len(rewards))):
        bootstrap = 0.0 if terminated[step] else next_values[step]
        residual = rewards[step] + gamma * bootstrap - values[step]
        same_episode = not (terminated[step] or truncated[step])
        carry = residual + gamma * trace_decay * same_episode * carry
        result[step] = carry
    return result
```

next_values 必须来自对应 transition 的 final observation，不是 reset 后的 observation。普通 rollout buffer 到尾但任务没结束，也可以用尾部 value bootstrap，递推在 buffer 边界停下。

## Critic 的误差怎样进入 Actor？

写成 $V_\phi(s)=V^\pi(s)+e(s)$，一步 residual 的期望就能拆开：

$$
\mathbb E[\delta_t\mid s,a]=A^\pi(s,a)+\gamma\mathbb E[e(s')\mid s,a]-e(s).
$$

假设当前状态高估 2，下一状态平均低估 3，$\gamma=0.9$，advantage 的误差就是 $0.9(-3)-2=-4.7$。一个本来不错的动作可能因此被压低。这是估值误差进入策略更新的具体路径。

加大 $\lambda$ 可以减少对沿途短期估值的依赖，但会带进更长段实际回报的噪声；片段截断时，尾部 bootstrap 仍有误差。不能简单说“$\lambda=1$ 总是无偏”，先问有没有走到真正终点、样本是否来自被评价的策略。

## Padding 和平均方式也会改变目标

两条轨迹中，A 的有效 token loss 为 $[1,1,1]$，B 只有一个 loss 9。按 token 平均是 3，先按轨迹平均再平均是 5。两种都能定义，但给轨迹的权重不一样。

| 实现细节 | 为什么不能略过 |
| --- | --- |
| Padding mask | 补齐位置不能影响均值、方差和分母 |
| Advantage 标准化 | 改变尺度，减均值还可能改变单个样本的符号 |
| 每卡独立标准化 | 不等于全局有效 batch 标准化 |
| Rollout 时的旧 value | 应在更新前固定，不能随当前 Critic 偷偷改 target |

比如 $[1,2]$ 减均值变成 $[-0.5,0.5]$。这是一种 batch 内的相对处理，标准化后的数不能再直接当成真实 $Q-V$。它可能帮助优化，但不是不需要解释的恒等变换。

## Actor 与 Critic 怎样各改各的

Actor 最小化 $-\log\pi_\theta(a_t\mid s_t)\,\mathrm{stopgrad}(\hat A_t)$；Critic 常拟合 $\mathrm{stopgrad}(\hat A_t+V_{\rm old}(s_t))$。这个 target 要在更新前固定，不能每次随着正在训练的 value 偷偷漂移。

PPO 再在 Actor 上加入新旧概率比与 clipping；不是把 GAE 当成 PPO 本身。移步[四种 PPO clipping 情况](../rlhf/ppo-clipping.md)看已有交互。

参考：[GAE 原始论文](https://arxiv.org/abs/1506.02438)。接着看[策略一次该改多大](trust-region.md)；想换到基于 Q 的方法，再读 [DQN](dqn.md)。
