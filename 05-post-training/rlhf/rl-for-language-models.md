# RLHF：先把普通 RL 映射到语言模型

**中文** · [English](rl-for-language-models.en.md)

> 阅读时间：约 2 分钟 · 难度：必修 · 最近审阅：2026-10

语言模型每次只是在预测下一个 token，为什么能用强化学习来训练？因为一次生成可以看成一串决策：每一步看到当前前缀，选一个 token，最后由奖励模型、测试或环境给整段结果打分。监督学习通常直接给出目标答案；强化学习（Reinforcement Learning, **RL**）只评价一段行为的结果。模型知道这次得了多少分，却未必知道是哪一步造成的，也没有一个逐步的标准答案可以照抄。

## 先借用 RL 的几个词 {#rl-vocabulary}

在 RL 里，agent 处在**状态**（state）$s_t$，按**策略**（policy）$\pi_\theta$ 选一个**动作**（action）$a_t$；环境按转移规律给出下一状态 $s_{t+1}$ 和**奖励**（reward）$r_t$。一次 transition 写成 $(s_t,a_t,r_t,s_{t+1})$，许多步连起来就是一条轨迹（trajectory）：

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots).
$$

agent 实际看到的是观测（observation），不一定是完整状态；状态、观测、转移和 Bellman 递推的细节见 [MDP 与 Bellman](../deep-rl/mdp-bellman.md)。Return、value、Q 和 advantage 是为了学习而构造或估计的量，不是环境直接给的反馈，[下一篇](value-and-advantage.md)逐个定义。

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

token 被接到前缀后，便形成下一个状态。单纯文本生成时，这个状态转移就是确定性的 token 序列拼接；到了工具调用或交互式 agent，环境还会返回搜索结果、执行结果或新的 observation。
