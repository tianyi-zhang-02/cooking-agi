# RLHF：On-policy、off-policy，以及 Reward Model 不是 Critic

**中文** · [English](on-off-policy.en.md)

> 阅读时间：约 3 分钟 · 难度：必修 · 最近审阅：2026-09

## On-policy、off-policy 与 offline RL

判断标准不是“代码是否在线运行”，而是：**生成训练数据的 behavior policy，与正在学习的 target policy 是不是同一个或足够接近。**

| Setting | 数据怎样产生 | 典型特点 |
| --- | --- | --- |
| **On-policy** | 当前或最近的 policy 采样新 trajectory | 分布匹配，但 rollout 贵，旧数据很快失效 |
| **Off-policy** | 另一个 behavior policy 或历史 policy 产生数据 | 能复用 replay buffer / logs，但要处理 distribution mismatch |
| **Offline RL** | 只有一份固定数据集，训练时不能再与环境交互 | 通常是 off-policy 的特殊情形，最受数据覆盖范围限制 |

PPO 用 $\pi_{\text{old}}$ 采一批数据，再让 $\pi_\theta$ 在这批数据上做几次受限更新，看起来同时有 old/new policy；但二者足够接近，而且数据很快会被新 rollout 替换，所以仍属于 on-policy / near-on-policy。Off-policy 则不等于 offline：SAC 可以一边继续收集数据，一边反复学习 replay buffer 中由过去策略生成的经验。

标准 DPO 使用固定 chosen/rejected pairs，具有 offline data 的特征，但没有 Bellman backup、Critic 或环境 rollout；更准确的名字是 **offline preference optimization**，而不是经典 off-policy RL。

<details class="interview" markdown="1">
<summary>工业里什么时候真的需要 RL？</summary>

当一个动作会改变后续状态，而且产品关心的是长期结果时，RL 的抽象最有价值：推荐系统要权衡即时点击与长期留存；广告系统要同时考虑转化、预算和用户体验；物流与机器人要优化一连串相互影响的动作；Conversational AI 要把检索、工具调用、澄清、回答和人工升级看成完整 episode。

真实探索会伤害用户或产生成本，因此工业系统通常组合 logged data、simulation、offline evaluation、action constraints、小流量探索和 A/B test。算法名字不是第一步：先定义 state/action/reward，确认 reward 能否验证、数据由谁生成，以及是否允许安全地收集新 trajectory。

</details>

## Reward Model 不是 Critic

这两个模型都输出标量，所以很容易混：

| | Reward Model | Critic / Value Model |
| --- | --- | --- |
| 输入 | prompt + 完整回答 | 当前 prompt + 生成前缀 |
| 输出 | 学到的偏好代理分数 | 从当前状态出发的预计 return |
| 回答的问题 | 「这个完成的回答看起来有多好？」 | 「从这里按当前策略继续，预计能拿多少分？」 |
| PPO 阶段 | 通常冻结 | 跟随当前 Actor 训练 |

Reward Model 给出的不是「真实人类满意度」，而是从有限偏好数据学到的 **proxy reward**（代理奖励）。它会判断错、偏爱表面风格，也可能被策略钻空子。Critic 学的则是当前 policy 下的条件期望；Actor 一变，它要估计的目标也会跟着变。

从完整回答粒度看，这套训练有一点像 contextual bandit：给一个 prompt，生成一个回答，最后拿一个整体分数。但 token 生成内部仍然是序列决策，状态会随前缀不断变化。两种说法只是抽象粒度不同。
