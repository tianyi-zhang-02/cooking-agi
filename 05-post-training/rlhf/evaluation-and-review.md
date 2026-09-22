# RLHF：评估闭环与复习题

**中文** · [English](evaluation-and-review.en.md)

> 阅读时间：约 4 分钟 · 难度：必修 · 最近审阅：2026-09

## Evaluation 才让训练形成闭环

训练 loss 或平均 reward 变好，不代表模型真的更可用。至少要分别检查：通用能力是否
退化、事实是否有依据、安全与 policy compliance、工具调用与任务完成、多轮一致性，
以及延迟和成本。开放式回答可以结合人工评审、经过校准的 LLM judge 与确定性检查；
数学、代码和结构化任务则优先使用 verifier。

上线前保留冻结的 regression suite，之后再做 shadow evaluation 与受控 A/B test。
线上失败不能直接全部回灌：先按 failure taxonomy 去重、审计与分层采样，再决定它应该
成为 SFT demonstration、chosen/rejected pair、verifier case，还是系统侧规则。

## 面试常见问题

<details class="interview" markdown="1">
<summary>RLHF 第三阶段有几个模型？分别训不训？</summary>

四个：Actor、Critic、Reward、Reference。

**只有 Actor 和 Critic 在更新参数。** Reward 是第二阶段训好后冻结的，Reference 是 SFT 模型的冻结副本。Actor 和 Reference 初始权重相同，训练中 Actor 逐渐偏离，Reference 停在原地作为 KL 的参照。

</details>

<details class="interview" markdown="1">
<summary>为什么需要 Reference model 和 KL 惩罚？去掉会怎样？</summary>

奖励模型只是在有限偏好数据上拟合的函数，训练分布之外没有任何约束。放开优化，策略会找到奖励模型给高分但人不认可的输出，也就是 reward hacking。典型表现是输出变长、堆砌讨好性措辞，或者干脆退化成某种重复模式。

KL 惩罚把策略拴在 SFT 附近。$\beta$ 太大学不动，太小会跑飞——它是这套方法的核心超参，不是可选正则。

</details>

<details class="interview" markdown="1">
<summary>Critic 是干什么的？GRPO 怎么把它省掉的？</summary>

Critic 估计状态价值 $V_t$，用来算优势 $A_t = R_t - V_t$。它的作用是**降方差**：直接用回报做策略梯度，方差大到训不动。

GRPO 的观察是：既然只是要一个基线，那对同一个提示采样一组回答、用组内均值和标准差归一化就够了，不必训练一个全尺寸网络来估。省掉的是一个正在训练的模型，显存和计算都是实打实的。

</details>

<details class="interview" markdown="1">
<summary>DPO 和 PPO 的本质区别是什么？</summary>

DPO 用了一个推导：带 KL 约束的奖励最大化有闭式最优解，把奖励反解成策略的表达式之后，偏好损失可以直接对策略求梯度。于是奖励模型和 RL 循环都不需要了。

标准 DPO 的代价是使用**离线**偏好对。训练中策略在变，数据却不跟着变；它无法主动探索当前 policy 的新失败。PPO 或其他 online RL 可以从当前策略采样，更适合需要探索、环境交互或可验证多步结果的任务，但 rollout 更贵，训练也更不稳定。

所以 DPO 不能完全替代 PPO，但 PPO 也不是所有场景都更好。高质量静态偏好数据可以先用 DPO；需要当前策略不断产生新数据时使用 online RL。Online DPO 等变体进一步说明，关键区别是数据与反馈闭环，而不只是 loss 的名字。

</details>

<details class="interview" markdown="1">
<summary>奖励模型的分数可以直接比较大小吗？</summary>

Bradley–Terry 损失直接约束的是同一 prompt 下 chosen 与 rejected 的**分差**。给所有分数加一个常数，损失完全不变，所以零点没有可识别语义；一个 $2.4$ 不能直接读成「满意度 2.4」。

固定 Reward Model 的原始输出当然可以拿来计算，但跨 prompt、领域或模型版本比较时，必须先确认校准与尺度是否稳定。训练实现也可能做 reward whitening 或 normalization；那是优化选择，不是 Bradley–Terry 必然要求的规则。

</details>

<details class="interview" markdown="1">
<summary>为什么 RLVR 里 reward hacking 变少了？</summary>

因为奖励不再是被拟合出来的。数学题对答案、代码跑测试，检查器是写死的程序，没有「训练分布之外」这回事，也就没有可以钻的空子。

但它换来了另一个限制：只适用于结果可自动验证的任务。写作、对话这类没有检查器的场景，还是得回到学出来的奖励模型。而且验证器本身也可能被绕过——比如代码只过测试用例但逻辑是错的。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>第三阶段四个模型分别从哪来，哪两个在更新参数？</li>
    <li>KL 惩罚去掉会发生什么？$\beta$ 调大调小分别是什么症状？</li>
    <li>Critic 的作用是降方差还是提高准确率？GRPO 用什么替代了它？</li>
    <li>为什么 DPO 不需要奖励模型？它因此付出了什么代价？</li>
    <li>对 PPO-style clipping，$\hat A$ 正负与 $\rho$ 越界分别在哪两种情况下让梯度归零？</li>
  </ol>
</div>

## 继续阅读

- [Post-Training 总览](../README.md)：SFT、偏好学习、RL 各自适合什么问题
- [数据与反馈](../../01-data-and-feedback/)：偏好标签本身的质量问题
- [Evaluation](../../07-evaluation/)：怎么判断对齐之后真的变好了

## 中文导读

- [大模型中的强化学习](https://zhuanlan.zhihu.com/p/693582342) — 知乎 @大家好我是爱因，专栏《机器学习小王子》。
  本章只讲 RLHF 这条主干，刻意没有铺开算法谱系。那一半在这篇里：从 MDP 的要素、
  贝尔曼方程、MC/TD/GAE 的偏差-方差权衡讲起，一路到 PPO 的四模型协同、DPO 与
  IPO/KTO，再到 GRPO、DAPO、Dr. GRPO、RLOO、REINFORCE++ 各自想解决什么问题。
  想看全景，从它开始。

## 参考论文

- [InstructGPT](https://arxiv.org/abs/2203.02155) — 三阶段流程的出处
- [PPO](https://arxiv.org/abs/1707.06347) — 裁剪目标与信任域
- [DPO](https://arxiv.org/abs/2305.18290) — 去掉奖励模型和 RL 循环
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — KL 惩罚与 reward hacking 的早期实证
