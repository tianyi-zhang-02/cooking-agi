# RLHF：Reward、Value 与 Advantage

**中文** · [English](value-and-advantage.en.md)

> 阅读时间：约 2 分钟 · 难度：必修 · 最近审阅：2026-09

## Reward、Return 和奖励归因

**Reward** $r_t$ 是某一步拿到的即时反馈；**Return** $G_t$ 是从这一步开始的累计未来奖励：

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.$$

在经典偏好 RLHF 中，主要奖励经常等完整回答结束才由 Reward Model 给出。因此前面每一枚 token 都要共同为最后的分数负责：低分究竟是开头方向错了，还是中途出现事实错误？这就是 **credit assignment**（奖励归因）问题。实现里也常把逐 token 的 KL 惩罚当作较密集的 shaping reward，但它不等于人类偏好本身。

## Value、Q 和 Advantage

Critic 要学的不是「这个完整回答好不好」，而是当前前缀往后生成，预计还能拿到多少累计回报：

$$V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].$$

如果还指定当前先选动作 $a$，则是 action value：

$$Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].$$

二者的差是 **Advantage**（优势）：

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).$$

它问的不是「这次总分高不高」，而是「这个动作相对当前状态下的正常预期，好了多少」。同样拿到 $0.6$ 的回报：如果 Critic 原本预测 $0.8$，它低于预期；如果原本只预测 $0.2$，它就明显高于预期。减去这个 baseline 不会改变期望中的策略梯度，却能显著降低方差。

## Bellman equation：更新的是 value，不是 reward

Bellman equation 把长期价值写成一步递归：

$$
V^\pi(s_t)
=\mathbb E_{a_t\sim\pi,\,s_{t+1}\sim P}
\left[r_t+\gamma V^\pi(s_{t+1})\right].
$$

它不是一种“修改 reward 的规则”。$r_t$ 通常由环境、Reward Model 或 verifier 给出；Bellman relation 用这个观测到的 reward 和下一状态的 value，更新当前的 **value estimate**。展开递归便是

$$
V^\pi(s_t)=\mathbb E[r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots].
$$

当 $0<\gamma<1$ 时，越远的 reward 权重越小；$\gamma$ 同时定义有效 planning horizon，并在无限时域问题中帮助回报保持有限。但“远期必然折扣”不是 RL 的定义：有限长度的 LLM episode 常使用 $\gamma=1$，让 terminal reward 对前面 token 不因距离而衰减。

使用一次实际 transition，可以构造 one-step TD target：

$$
y_t=r_t+\gamma V_\phi(s_{t+1}),
\qquad
\delta_t=y_t-V_\phi(s_t).
$$

$\delta_t$ 是 **temporal-difference error**（时序差分误差）：新观察到的“一步 reward 加未来估值”与旧估值相差多少。Monte Carlo 直接使用完整 $G_t$，偏差低但方差高；TD 会 bootstrap，方差低但引入 value approximation bias。PPO 常用 GAE 在两者之间调节。

策略梯度的核心因此可以写成：

$$\nabla_\theta J(\theta)\approx\mathbb E\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right].$$

$\hat A_t>0$ 时，提高这次采样动作的概率；$\hat A_t<0$ 时，降低它。这个公式只直接更新实际采样到的 token，并通过共享参数影响其他状态下的分布。Critic 则用 return 或 bootstrapped target 回归 $V_\phi(s_t)$；它的主要作用是**降低估计方差**，不是替 Actor 决定下一个 token。
