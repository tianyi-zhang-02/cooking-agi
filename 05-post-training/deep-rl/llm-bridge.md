# 接到 LLM：哪些 RL 直觉能搬过去？

**中文** · [English](llm-bridge.en.md)

> 阅读时间：约 6 分钟 · 最近审阅：2026-10

把动作换成 token，经典 RL 的很多概念还能用；但不能把连续控制的网络图原封不动搬过去。真正该搬的是：**目标、数据来自谁、估计器、更新约束与验证方式。**

## 先写一张对应表

| 经典 RL | 文本生成中的一种对应 | 要保留的警惕 |
| --- | --- | --- |
| State / observation | prompt 与当前上下文 | 隐藏环境状态不一定在文本里 |
| Action | 一个 token，或一次工具调用 | 两种粒度的时长和成本不同 |
| Policy | 自回归模型 | 采样温度与 log-prob 必须一致 |
| Reward | 偏好、测试、任务结果 | 能优化不代表能代表用户目标 |
| Episode | 一次回答或多轮任务 | EOS、超时、工具失败不可混为一谈 |

单段生成中，拼接 token 的转移近乎确定，但采样与奖励不一定确定。工具 agent 的下一观察还来自外部世界，可能延迟、有权限限制或发生失败。

## PPO 与 GRPO 各估计了什么

PPO 用采样策略收集的轨迹估计 advantage，再通过概率比值和 clipping，抑制某些方向上过大的更新收益；它不把比值硬锁在区间内。参考模型 KL 惩罚与 PPO clipping 也不同：前者惩罚偏离固定基准的行为，后者调整这批数据上的局部优化目标。

GRPO 常用同一 prompt 下多个回答的相对奖励构造组内 advantage，减少对独立 value Critic 的依赖。若组内奖励全相同，标准化后的相对信号可能为零；分组与归一化还会影响 prompt 的相对权重。它不自动解决稀疏奖励、错误 verifier 或跨任务公平比较。

公式与已有交互在 [PPO clipping](../rlhf/ppo-clipping.md) 和 [GRPO / DPO / RLVR](../rlhf/after-rlhf.md)。这里负责把经典 RL 的问题接起来，不另造一套记号。

## 长任务：memory 改变的是什么

当完整环境不可见，可以维护历史或学习记忆状态。记忆不是奖励函数，也不自动让过程满足 Markov；关键是是否保存了未来决策需要的信息。

多步工具调用还要区分任务失败、服务超时、主动取消和预算耗尽。它们可能都让 rollout 停止，但训练 mask、奖励归因与重试策略不应默认一样。先定义 failure semantics，再谈如何增加并发。

## 多个 agent：环境不再固定

其他 agent 也在学习时，你的状态转移与奖励分布会随它们改变。Centralized training / decentralized execution 可以在训练时给 Critic 更多联合信息，执行时各 Actor 只用允许的局部信息。

这只是多智能体入口，不是完整 MARL 课程。通信、博弈目标、非平稳性和协作归因需要另开篇章。参考：[MADDPG](https://arxiv.org/abs/1706.02275)。

## Reward 上涨后，还缺什么证据

检查独立任务成功率、长度与成本、未见场景、工具失败恢复，以及 reward hacking。训练时用同一个 judge 提供奖励，评估又只看它的分数，不能视作独立证据。

可用[评估栈](../../07-evaluation/evaluation-stack.md)安排验证，再用 [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.md)设计评分规则与人工锚点。贯穿这条路线的问题始终一样：模型到底学会了任务，还是只学会让尺子给高分？
