# RLHF：Reward、Value 与 Advantage

**中文** · [English](value-and-advantage.en.md)

> 阅读时间：约 4 分钟 · 最近审阅：2026-10

## Reward、Return 和奖励归因

**Reward** $r_t$ 是某一步拿到的即时反馈；**Return** $G_t$ 是从这一步开始的累计未来奖励：

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.$$

在经典偏好 RLHF 中，主要奖励经常等完整回答结束才由 Reward Model 给出。因此前面每一枚 token 都要共同为最后的分数负责：低分究竟是开头方向错了，还是中途出现事实错误？这就是 **credit assignment**（奖励归因）问题。实现里也常把逐 token 的 KL 惩罚当作较密集的 shaping reward，但它不等于人类偏好本身。

## Value、Q 和 Advantage

Critic 不负责评判完整回答好不好。它要估的是：从当前前缀接着生成，预计还能拿到多少累计回报：

$$V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].$$

如果还指定当前先选动作 $a$，则是 action value：

$$Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].$$

二者的差是 **Advantage**（优势）：

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).$$

它问的是：这个动作比当前状态下的正常预期好了多少。光看这次总分高不高，回答不了这个问题。同样拿到 $0.6$ 的回报：如果 Critic 原本预测 $0.8$，它低于预期；如果原本只预测 $0.2$，它就明显高于预期。减去这个 baseline 不会改变期望中的策略梯度；baseline 估得准时，通常能明显降低方差。

## 算一条三 token 回答 {#worked-returns}

把这些定义放到一条具体回答上。回答有 3 个 token，$\gamma=1$。Reward Model 只在结尾给 1 分；每个 token 另有一笔 −0.1 的 KL shaping。Critic 对 3 个前缀的估值是 `[0.5, 0.6, 0.8]`，回答结束后没有后续回报。数字都是教学假设。

| 位置 $t$ | $r_t$ | $G_t$（从这里往后加） | $V(s_t)$ | $G_t-V(s_t)$ |
| --- | --- | --- | --- | --- |
| 0 | −0.1 | 0.7 | 0.5 | +0.2 |
| 1 | −0.1 | 0.8 | 0.6 | +0.2 |
| 2 | −0.1 + 1 = 0.9 | 0.9 | 0.8 | +0.1 |

$G_t$ 从后往前算最方便：$G_2=0.9$，$G_1=-0.1+0.9=0.8$，$G_0=-0.1+0.8=0.7$。三个位置的 advantage 都为正，说明这条回答在每个前缀上都比 Critic 的预期好一点；最后一步的预期已经很高（0.8），所以只多出 0.1。Critic 估得不准，这一列也会跟着偏；下面的 TD 和 GAE 处理的就是这种取舍。

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
y_t=r_t+\gamma V_\psi(s_{t+1}),
\qquad
\delta_t=y_t-V_\psi(s_t).
$$

$\delta_t$ 是 **temporal-difference error**（时序差分误差）：新观察到的“一步 reward 加未来估值”与旧估值相差多少。Monte Carlo 直接使用完整 $G_t$，偏差低但方差高；TD 会 bootstrap，方差低但引入 value approximation bias。PPO 常用 GAE 在两者之间调节。

沿用上面那条回答：$\delta_0=-0.1+0.6-0.5=0$，$\delta_1=-0.1+0.8-0.6=0.1$，$\delta_2=0.9+0-0.8=0.1$。三项相加是 0.2，正好等于 $G_0-V(s_0)$：$\gamma=1$ 时，中间的估值逐项抵消。只拿 $\delta_t$ 当 advantage，是 GAE 取 $\lambda=0$ 的一端；把后面的 $\delta$ 全部加起来，是 $\lambda=1$ 的一端，也就是上表的 Monte Carlo 优势。

策略梯度的核心因此可以写成：

$$\nabla_\theta J(\theta)\approx\mathbb E\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right].$$

$\hat A_t>0$ 时，提高这次采样动作的概率；$\hat A_t<0$ 时，降低它。这个公式只对实际采样到的位置求和；但在同一状态下，softmax 的各项共享归一化：采样 log-prob 对 logits 的梯度是 $e_a-\pi$。advantage 为正时，直接沿这个梯度更新 logits，会抬高采样 token 的 logit、压低其他 logits。但概率还要一起归一化，不能据此说其他概率都会下降。例如，从 `[0.2, 0.7, 0.1]` 出发，采样第一个 token，取 advantage 为 1、logit 更新步长为 0.1，得到的概率约为 `[0.22374, 0.67402, 0.10224]`。第三个概率也上升了；在这个直接更新 logits 的例子里，下降的是未采样 token 的概率总和。共享参数还会把影响带到其他状态。Critic 则用 return 或 bootstrapped target 回归 $V_\psi(s_t)$；它的主要作用是**降低估计方差**，不是替 Actor 决定下一个 token。
