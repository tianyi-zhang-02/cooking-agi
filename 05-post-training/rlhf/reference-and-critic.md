# RLHF：为什么必须有 Reference 和 Critic

**中文** · [English](reference-and-critic.en.md)

> 阅读时间：约 2 分钟 · 难度：必修 · 最近审阅：2026-09

## 为什么必须有 Reference

因为奖励模型是可以被钻空子的。它只是一个在有限偏好数据上拟合出来的函数，在训练分布之外的行为完全没有约束。策略如果放开了优化，会找到那些**奖励模型给高分、但人根本不认可**的回答——这就是 reward hacking。

所以真正被优化的目标里带一个 KL 惩罚：

$$r_{\text{total}}(x, y) = r_\phi(x, y) - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)$$

意思是：你可以往奖励高的方向走，但不许离出发点太远。$\beta$ 就是这根绳子的松紧。

$\beta$ 太大，模型学不动，输出和 SFT 没区别；$\beta$ 太小，跑几百步之后开始输出人类看不懂但奖励模型很喜欢的东西。**这不是可选的正则项，是这套方法能不能用的前提。**

## 为什么还要一个 Critic

策略梯度需要知道「这个动作比平均水平好多少」，也就是优势 $A_t$。只用回报 $R_t$ 的话方差极大，训练会抖得没法用。Critic 学一个基线 $V_t$，用

$$A_t = R_t - V_t$$

来降方差。PPO 进一步用 GAE 做多步平滑，并把更新裁剪在信任域内：

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\big)\Big], \qquad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

裁剪是为了防止一步走太远——策略一旦跑出旧策略的支撑集，重要性比 $\rho_t$ 就会爆炸。

这里有两把不同的尺子：**PPO clipping** 比较当前 Actor 与采样数据时的旧 Actor，限制一次优化更新；**Reference KL** 比较 Actor 与冻结的 SFT Reference，限制整个训练过程的累计漂移。前者不能替代后者。
