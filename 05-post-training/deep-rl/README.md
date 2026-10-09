# Deep RL：从一次选择，到一套训练循环

**中文** · [English](README.en.md)

> 阅读时间：约 6–9 分钟 · 最近审阅：2026-10

眼前有 2 分可以拿，多等一步可能拿到 4 分。该怎么选？**强化学习关心的不只是这一步得了多少分，还包括这个选择会把你带到哪里。** 我们从这样的小问题开始，再慢慢走到估值、策略更新和完整的训练循环。

每篇先用例子把问题讲清楚，再看公式和代码。第一次读可以跟着例子走，复习时再用表格对照算法、检查实现。算例都是自拟的小环境，不依赖课程作业，也不转载课件。

## 怎么读，才不容易绕晕

| 路线 | 读什么 | 读完应该能做什么 |
| --- | --- | --- |
| 先把问题写对 | [MDP 与 Bellman](mdp-bellman.md) → [MC、TD 与终止条件](returns-and-td.md) | 从一段轨迹算出 return 和训练 target |
| 直接改策略 | [Policy Gradient](policy-gradients.md) → [Actor–Critic 与 GAE](actor-critic-gae.md) → [Natural Gradient 与 TRPO](trust-region.md) | 解释梯度方向、baseline、两个 mask，以及步长为什么需要约束 |
| 通过价值选动作 | [Fitted Q 与收敛边界](fitted-q.md) → [DQN](dqn.md) → [DDPG / TD3](continuous-control.md) → [SAC](soft-actor-critic.md) | 分清回归 loss 与策略表现，画出各个网络的梯度路径 |
| 用模型想几步 | [Model-based RL](model-based.md) → [规划与 LQR](planning-and-control.md) | 分清学 dynamics、规划动作和训练 policy |
| 数据与奖励从哪里来 | [Offline RL 与 OPE](offline-and-ope.md) · [探索与层级](exploration-and-hierarchy.md) · [模仿与奖励](imitation-and-rewards.md) | 说出各自增加的假设，而不只是记名字 |
| 真正跑一个实验 | [实验与排错](experiments.md) → [接到 LLM](llm-bridge.md) | 检查实现，再判断 reward 是否代表任务进步 |

这 16 篇不只介绍算法名字。我们会一起算 Bellman 残差界、TD target、梯度方向、PPO 裁剪和控制轨迹，再看 CQL、IQL 与 FQE 如何处理离线数据。层级、IRL 和多智能体目前只作引入，还不是完整的理论综述。

## 不同读法，不需要不同一套笔记

| 今天想读到哪一步 | 可以怎么读 | 应该留下什么 |
| --- | --- | --- |
| 先弄懂直觉 | 顺着开头的任务、图和手算表走，暂时跳过推导 | 能说出算法为什么需要这个部件 |
| 想把原理弄扎实 | 回到公式，逐个确认条件期望、符号和假设 | 能算一个例子，也能指出保证何时不成立 |
| 想动手实现 | 看数据来源、target、梯度路径、mask 和更新时机 | 用小环境查错，而不是只盯训练曲线 |

所有关键算例都直接放在正文里，不要求点滑块或展开交互才知道答案。少量流程图会短暂强调阅读顺序；动画结束后内容仍完整，减少动态效果的系统设置也会被尊重。原有交互留给想改参数的人，不充当正文的替代品。

## 算法再多，先找这 3 件事

<div class="drl-flow drl-sequence" aria-label="强化学习训练循环">
<span>采样<br><small>谁产生轨迹？</small></span><b>→</b><span>估计<br><small>用什么 target？</small></span><b>→</b><span>更新<br><small>梯度流向谁？</small></span>
</div>

| 方法 | 主要学什么 | 动作怎么来 | 容易忽略的代价 |
| --- | --- | --- | --- |
| REINFORCE | policy | 从策略分布采样 | 回报方差大，旧数据不能直接混用 |
| DQN | Q-value | 离散动作中取最大值，训练时加探索 | bootstrap、函数逼近和 off-policy 叠加不稳定 |
| 状态价值 Actor–Critic | policy + V | 从策略分布采样 | Critic 的误差会影响 Actor |
| TD3 / SAC | policy + 两个 Q | 连续动作 policy | 重用数据省交互，不代表省计算 |
| Model-based | dynamics，可能再学 policy / value | 规划或 policy | 模型预测准，不等于规划出的行为可靠 |

On-policy / off-policy 比较的是**收集数据的行为策略，与正在评价或改进的目标策略是否一致**；online / offline 说的是**训练中能不能继续和环境交互**。所以，边交互边用旧数据训练，可以同时是 online 和 off-policy。

## 先补哪几块基础

能读条件期望、链式法则和梯度就可以开始。需要补基础时，可以先看[ML 数学](../../00-foundations/ml-math-interview.md)里的概率与梯度；PyTorch 部分需要知道自动求导和张量形状。

统一记号：$s_t$ 是状态，$a_t$ 是动作，$r_t$ 是执行该动作后收到的奖励，$\gamma$ 是折扣。有限时域写 $t=0,\ldots,T-1$，终止后的 value 为 0。概率策略用 $\pi_\theta$，value 参数用 $\phi$。不要把 reward、return 和 value 当成同一个数。

后面会反复遇到这些词，先用一句话认清它们：

| 术语 | 在这里是什么意思 |
| --- | --- |
| 策略（policy） | 给定状态，决定选哪个动作或怎样分配动作概率 |
| 奖励（reward） | 执行一步后收到的反馈，不是整段表现 |
| 回报（return） | 从某一步起，把后续奖励按约定累加起来 |
| 价值（value） | 在某个策略下，对未来回报的期望 |
| 训练目标值（target） | 这轮拿来拟合的数；可能只是估计，并非真值 |
| 自举（bootstrapping） | 用已有估值，补上还没观察到的未来 |
| 轨迹（trajectory / rollout） | 与环境交互得到的一串状态、动作和奖励；rollout 也可能只是一段 |

## 用法：每次改一个条件

读完算例，试着只改一件事：把终止换成超时、把 $\lambda$ 从 0 调到 1、让两个 Q-network 的排序不同。网页里的小实验就是为这个准备的。它们演示计算机制，不模拟一场真实训练，也不展示虚构的性能提升。

想检查公式有没有落实到代码，运行[配套小程序](code/rl_checks.py)；想看 stop-gradient 是否正确，再运行 [PyTorch 检查](code/torch_updates.py)。两者都不需要 GPU。

## 想先动手，可以从这几张图开始

| 想弄明白 | 去哪一篇 | 自己改什么 |
| --- | --- | --- |
| 奖励怎么一步步传回起点 | [MDP 与 Bellman](mdp-bellman.md) | 调折扣，逐轮更新 |
| 未来哪几步影响这次 advantage | [Actor–Critic 与 GAE](actor-critic-gae.md) | 调 λ，展开每步贡献表 |
| 回归变好为何还会发散 | [Fitted Q](fitted-q.md) | 逐轮拟合，对照 value 曲线与 MSE |
| 一次策略更新到底改了多少 | [Natural Gradient 与 TRPO](trust-region.md) | 调 logit 步长和 KL 预算 |
| 模型预测错了，反馈有什么用 | [Model-based RL](model-based.md) | 调模型误差和规划长度，对照两条真实轨迹 |

调完参数，先猜一下结果为什么变了，再展开计算表核对。重点不是记住曲线的方向，而是看懂改变是怎样一步步传下去的。

## 公开参考

推导与术语可对照 [Spinning Up：RL 基础](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)；各章另列原始论文。PPO 与 RLHF 已有独立系列，读完这里再接过去，不必在两个地方重复背定义。
