# RLHF：为什么需要这么绕

**中文** · [English](README.en.md)

> 阅读时间：约 2 分钟 · 难度：必修 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>把「人更喜欢哪个回答」变成可优化的目标</strong></div>
  <div><span>前置知识</span><strong>SFT · 偏好数据 · next-token prediction</strong></div>
  <div><span>核心机制</span><strong>学一个奖励模型，再用 RL 去优化它</strong></div>
  <div><span>常见错误</span><strong>以为四个模型都在训练；以为 KL 项是可选的</strong></div>
</div>

## 快速学习：Actor、RM、Critic、Reference 谁做什么

<details class="interview" markdown="1">
<summary>两分钟讲清 Base → SFT → Preference → RL → Evaluation</summary>

**快速记忆**：Actor 生成，RM 给完整回答打分，Critic 估计 prefix expected return，Reference 约束策略漂移；Evaluation 检查 reward 是否真的对应产品目标。

**面试回答**

> SFT 先提供可用策略；偏好数据训练 Reward Model；PPO 用 reward-derived advantage 更新 Actor，同时通过 clipping 和 reference KL 控制更新。Critic 是 variance-reduction baseline，不是 Reward Model。整个流程必须从当前 policy 重新采样并做独立评估，才构成闭环。

<details markdown="1">
<summary><b>深挖</b>：为什么 Reference 与 PPO clipping 不是同一层约束？</summary>

Clipping 限制一次 optimizer update 相对 rollout policy 的 ratio；reference KL 限制当前策略相对固定 SFT policy 的长期漂移。前者是局部 trust region，后者是行为先验。删掉任一项都不会自动由另一项完全补上。

</details>
</details>

## 为什么需要这么绕

想让模型「回答得更好」，最直接的想法是写个损失函数。但「更好」写不出来——它没有闭式表达，也没有标准答案可以对照。

人却能**比较**：给两个回答，说哪个更好。RLHF 的全部思路就是把这件事变成可优化的目标：

```text
人能比较 → 学一个模型来预测「人会更喜欢哪个」→ 拿这个模型当奖励函数做 RL
```

代价是引入了一层间接：你优化的不再是「人的偏好」，而是「一个模型对人偏好的拟合」。后面所有的麻烦都从这里来。

## 这一组怎么读

1. [先把普通 RL 映射到语言模型](rl-for-language-models.md)：state、action、reward、trajectory 在语言模型里各是什么
2. [Reward、Value 与 Advantage](value-and-advantage.md)：return、value、advantage 和 Bellman equation，为什么要减一个 baseline
3. [On-policy、off-policy](on-off-policy.md)：数据由谁生成；以及 Reward Model 为什么不是 Critic
4. [三个阶段和四个模型](three-stages.md)：SFT → 奖励模型 → RL，第三阶段谁在训谁被冻住（可以一步一步点）
5. [为什么必须有 Reference 和 Critic](reference-and-critic.md)：KL 为什么不是可选项，Critic 为什么只是降方差
6. [后来发生了什么](after-rlhf.md)：GRPO 去掉 Critic，DPO 去掉 RL 循环，RLVR 把奖励换成程序
7. [PPO clipping 的四种更新情况](ppo-clipping.md)：拖动 ρ、切换 A 的正负，看什么时候停止鼓励
8. [评估闭环与复习题](evaluation-and-review.md)：怎样确认真的变好了；面试题和自检
