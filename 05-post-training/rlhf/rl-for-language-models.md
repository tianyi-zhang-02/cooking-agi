# RLHF：先把普通 RL 映射到语言模型

**中文** · [English](rl-for-language-models.en.md)

> 阅读时间：约 3 分钟 · 难度：必修 · 最近审阅：2026-09

理解 RL 最清楚的方式，是先把它拆成两层：**环境层**描述世界怎样变化，**学习层**描述 agent 怎样从经验中改进 policy。监督学习通常直接给出目标答案；强化学习（Reinforcement Learning, **RL**）只评价一段行为的结果。模型知道这次得了多少分，却未必知道是哪一步造成的，也没有一个逐步的标准答案可以照抄。

## 环境层：世界如何运行

| Component | 它回答什么问题 | 自动驾驶例子 |
| --- | --- | --- |
| **State**（状态）$s_t$ | 世界此刻的完整情况是什么？ | 车辆位置、速度、周围车辆和路况 |
| **Observation**（观测）$o_t$ | agent 实际看到了什么？ | camera、LiDAR、speedometer |
| **Policy**（策略）$\pi_\theta(a\mid o)$ | 看到这些信息后，动作怎样分布？ | 红灯前给刹车更高概率 |
| **Action**（动作）$a_t$ | agent 真正做了什么？ | 转向、加速或刹车 |
| **Dynamics**（转移规律）$P$ | 动作会怎样改变世界？ | 刹车后速度和位置如何变化 |
| **Reward**（奖励）$r_t$ | 这一步或最终结果得到什么评分？ | 安全行驶为正，碰撞为负 |

一次 transition / experience 常写成

$$
(s_t,a_t,r_t,s_{t+1}),
$$

把许多步连起来便得到 trajectory：

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots).
$$

**State 不一定等于 observation。** State 是决定未来演化所需的完整世界信息；observation 是 agent 能取得的部分信号。教科书 MDP 常假设 $o_t=s_t$，现实中的机器人、游戏和对话 agent 往往只能部分观测，此时更接近 POMDP。Policy 实际只能根据 observation 或历史摘要做决定，而不是读取不可见的真实 state。

## 学习层：怎样判断 policy 的行为

| Quantity | 它回答的问题 |
| --- | --- |
| Immediate reward $r_t$ | “我现在拿到了什么反馈？” |
| Return $G_t$ | “这次 rollout 从现在往后实际拿了多少？” |
| Value $V^\pi(s_t)$ | “从这里按当前 policy 继续，平均预计拿多少？” |
| Q-value $Q^\pi(s_t,a_t)$ | “在这里先做这个 action，平均预计拿多少？” |
| Advantage $A_t$ | “这个 action 比这里的正常选择好还是差？” |

前一层产生 experience，后一层把 experience 转成训练信号。Reward 是环境或 evaluator 给出的反馈；return、value、Q 和 advantage 是为了学习而构造或估计的量。

## 映射到语言模型

语言模型生成一句话，正好可以写成一段序列决策：

| RL 概念 | 在语言模型里是什么 |
| --- | --- |
| **Agent**（智能体） | 正在训练的语言模型 |
| **State**（状态）$s_t$ | prompt 加上已经生成的前缀 $(x, y_{<t})$ |
| **Action**（动作）$a_t$ | 下一枚 token $y_t$ |
| **Policy**（策略）$\pi_\theta(a_t\mid s_t)$ | 模型 softmax 给出的 next-token 分布 |
| **Trajectory**（轨迹）$\tau$ | 从回答开始到结束的整段 token 序列 |
| **Reward**（奖励）$r_t$ | Reward Model、verifier 或真实环境给出的标量反馈 |

这里的 Actor 不是包在语言模型外面的另一个决策器：**语言模型本身就是 policy**。在状态 $s_t=(x,y_{<t})$ 下从词表里采样下一枚 token，所用的概率就是

$$a_t=y_t\sim\pi_\theta(\cdot\mid x,y_{<t}).$$

token 被接到前缀后，便形成下一个状态。单纯文本生成时，这个状态转移几乎就是确定性的字符串拼接；到了工具调用或交互式 agent，环境还会返回搜索结果、执行结果或新的 observation。
