# 从表格到网络：回归做对了，Q 为什么还会跑偏？

**中文** · [English](fitted-q.en.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-10

[Bellman 那篇](mdp-bellman.md)证明过：在折扣和有界奖励等条件下，精确的表格迭代会收敛。为什么换成神经网络，就不能照搬这个结论？**因为现在每次更新后，还得把答案塞回一个受限的函数族里。**

## Fitted Q-iteration：先造标签，再做回归

固定一批 transition $\mathcal D=\{(s_i,a_i,r_i,s'_i,d_i)\}$，$d_i$ 表示真实终止。每轮先用旧函数造标签：

$$
y_i^{(k)}=r_i+\gamma(1-d_i)\max_{a'}Q_k(s'_i,a'),
\qquad
Q_{k+1}\in\arg\min_{Q\in\mathcal F}\sum_i\left(Q(s_i,a_i)-y_i^{(k)}\right)^2.
$$

这叫拟合 Q 迭代（fitted Q-iteration，FQI）。同一轮内，标签固定；下一轮标签才随 $Q_k$ 改变。DQN 用 replay、若干梯度步和周期性 target 更新来做相关的近似迭代，并不是每轮都把回归精确求解。

<div class="drl-flow" aria-label="拟合 Q 迭代">
<span>旧 Q<br><small>构造固定标签</small></span><b>→</b><span>回归<br><small>限制在函数族内</small></span><b>→</b><span>新 Q<br><small>下一轮再造标签</small></span>
</div>

## 多出来的是投影，不只是一点误差

先忽略有限样本和优化误差，把回归写成投影 $\Pi_\mu$，其中 $\mu$ 是拟合用的采样分布：

$$
Q_{k+1}=\Pi_\mu\mathcal TQ_k.
$$

Bellman 算子 $\mathcal T$ 在最大值范数 $\|\cdot\|_\infty$ 下是 $\gamma$-压缩。但线性函数空间的最小二乘正交投影，只能直接保证在对应的加权 $L_2(\mu)$ 范数下不扩张。**两步用的不是同一把尺子，不能把保证直接乘起来。**

对非凸神经网络函数族，投影还可能不唯一，优化也未必找到最优解。所以这里连理想线性投影的好性质都不能无条件拿来用。另一方面，也不是“只要用了函数逼近就一定发散”；特定函数族、分布和算法仍可以有收敛保证。

## 两个状态，就能看到问题

这是一个自拟的 value-iteration 反例：两个状态都确定性转移到状态 2，奖励全为 0，$\gamma=0.9$，每个状态只有一个动作。因此真实 value 全为 0，Q 与 V 在这里等价。

我们限制表示为 $V_\theta=[\theta,2\theta]$，固定给两个状态相同回归权重。Bellman 更新产生标签 $[1.8\theta_k,1.8\theta_k]$。精确做一次最小二乘：

$$
\theta_{k+1}
=\arg\min_u\left[(u-1.8\theta_k)^2+(2u-1.8\theta_k)^2\right]
=1.08\theta_k.
$$

从 $\theta_0=1$ 开始，每轮乘 1.08，反而离零越来越远。真实答案明明就在函数族里，却还是失败了。

| 第几轮 | $\theta$ | 预测的两个 value |
| --- | --- | --- |
| 0 | 1 | [1, 2] |
| 1 | 1.08 | [1.08, 2.16] |
| 2 | 1.1664 | [1.1664, 2.3328] |

第一次回归中，对**同一份旧标签**的平均平方误差从 0.34 降到 0.324：这轮回归确实更好了。但标签下一轮会变，所以这不等于朝真实 value 靠近。

这个反例的固定均匀采样分布，不是该 Markov 过程集中于状态 2 的平稳分布。它不能用来宣称所有 on-policy TD 都会发散；它反驳的是“不论采样分布和投影怎么选，都继承表格收敛性”。

<div class="drl-lab" data-drl-lab="projection"><p>点击逐轮拟合，对照曲线和每轮 MSE 表。γ=0.9 时 θ 每轮乘 1.08；γ=0.5 时每轮乘 0.6。两种情况下，内层回归都在拟合当轮固定标签。</p></div>

<details markdown="1">
<summary>只改折扣，会发生什么？</summary>

重复计算得到 $\theta_{k+1}=1.2\gamma\theta_k$。若 $\gamma=0.5$，倍率是 0.6，这个例子就会收敛。但随意减小折扣也改变了任务重视多远的未来，不是免费的修复。配套 [Python 检查](code/rl_checks.py) 验证这两个条件。

</details>

## “梯度下降”要说清楚对哪个目标

FQI 每轮可以用梯度下降拟合固定标签；Q-learning 的 semi-gradient 也确实是当前固定 target 的梯度。问题是外层目标随 Q 改变，**它们不等于对一个永远不变的整体 Bellman residual 做完整梯度下降**。

若改为对 $\mathbb E[(Q-\mathcal TQ)^2]$ 求完整梯度，又会遇到随机转移下条件期望乘积的估计问题：同一条 next-state 样本同时用于两部分通常不够。独立双样本问题（double sampling）与 Double DQN 的“选择、估值分开”不是同一回事。

## 回到工程：每个稳定化部件各管一点

| 部件 | 主要缓解什么 | 没有保证什么 |
| --- | --- | --- |
| Replay buffer | 时间相关性、样本浪费 | 数据覆盖充分，或采样没有偏差 |
| Target network | 标签随当前更新立即漂移 | 外层迭代一定收敛 |
| Double Q | max 对估值噪声的选择偏差 | Q 一定更准，或没有低估 |
| 独立 rollout 评估 | 训练 loss 与实际行为脱节 | 有限评估样本足以覆盖所有失败 |

参考：[Fitted Q-iteration 的经典批量方法](https://jmlr.org/papers/v6/ernst05a.html)。上面的两状态计算是独立构造的算例，不是该论文的实验结果。接着读 [DQN 的实现与检查](dqn.md)。
