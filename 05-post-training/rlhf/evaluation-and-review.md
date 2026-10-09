# RLHF：评估闭环与复习题

**中文** · [English](evaluation-and-review.en.md)

> 阅读时间：约 6 分钟 · 难度：必修 · 最近审阅：2026-10

## Evaluation 才让训练形成闭环

训练 loss 或平均 reward 变好，不代表模型真的更可用。至少要分别检查：通用能力是否
退化、事实是否有依据、安全与 policy compliance、工具调用与任务完成、多轮一致性，
以及延迟和成本。开放式回答可以结合人工评审、经过校准的 LLM judge 与确定性检查；
数学、代码和结构化任务则优先使用 verifier。

上线前保留冻结的 regression suite，之后再做 shadow evaluation 与受控 A/B test。
线上失败不能直接全部回灌：先按 failure taxonomy 去重、审计与分层采样，再决定它应该
成为 SFT demonstration、chosen/rejected pair、verifier case，还是系统侧规则。

## 奖励涨了，先排除哪些解释？ {#independent-evaluation}

假设新模型的训练 reward 更高。可能是能力进步，也可能只是回答更长、尝试更多次，或者更会迎合某个 judge。别急着合成一个总分，先把比较条件固定下来。

| 检查项 | 怎么做 | 避免误判什么 |
|---|---|---|
| 题目与数据划分 | 留出未用于训练、调参的题目；检查近重复与模板泄漏 | 记住训练题不等于会解新题 |
| 生成预算 | 对齐采样次数、token 上限和解码设置；同时报告成本 | 多试几次得到的收益，不全是模型本身的收益 |
| 独立验收 | 训练 verifier 之外，再看边界测试、人工抽查或另一套标准 | 同一套有漏洞的奖励给自己验收 |
| 分组表现 | 按题目难度、语言、长度、任务类别拆分 | 平均涨了，却有一组明显退步 |
| 波动与不确定性 | 保留逐题配对结果，报告样本量、seed 与合适的区间估计 | 把少量题目或一次采样的运气当作稳定收益 |
| 能力与成本回归 | 检查旧任务、拒答、事实性、延迟和输出长度 | 优化一种行为时伤害另一种 |

一个小例子：如果每次独立采样的答对概率都是 $0.5$，尝试四次至少答对一次的概率就是 $1-(1-0.5)^4=0.9375$。**模型没有变，预算变了。**这是假定各次独立、每次成功率固定的教学计算；真实题目难度不同，不能把全体平均成功率直接代入这个式子。评估时要分别说明单次生成和多次尝试的结果；如果需要从多个回答里选出正确答案，还要把选择器的误差与成本算进去。

选择偏好式 judge 时，随机交换 A/B 顺序，并检查它是否只是偏爱长回答。人类标注与 judge 的分歧也别简单删掉：它可能说明 rubric 还不清楚。具体例子见 [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.md)；配对比较与样本混合问题见[指标靠得住吗](../../07-evaluation/metric-robustness.md)。

## 面试常见问题

<details class="interview" markdown="1">
<summary>典型 PPO-style RLHF 有哪些角色？分别训不训？</summary>

常见的四个角色是 Actor、Critic、Reward、Reference。它们不等于「必须同时常驻四份完整模型」。

**只有 Actor 和 Critic 在更新参数。** Reward 是第二阶段训好后冻结的，Reference 是 SFT 模型的冻结副本。Actor 和 Reference 初始权重相同，训练中 Actor 逐渐偏离，Reference 停在原地作为 KL 的参照。

</details>

<details class="interview" markdown="1">
<summary>为什么需要 Reference model 和 KL 惩罚？去掉会怎样？</summary>

奖励模型在有限数据上学到的偏好，未必能泛化到策略后来生成的回答。优化得越来越用力，可能找到高分但不合人意的输出，例如迎合某种表达风格，而不是改善答案本身。

Reference KL 为偏离参照策略增加代价。$\beta$ 越大，这个代价通常越强，但实际效果也取决于 reward 尺度。去掉 KL 不等于训练必然崩溃；保留它也不能保证不钻奖励的空子。应把有无 KL、系数与独立评估一起比较，而不是把某个非零系数当成定律。推导和数值例子见 [Reference 与 Critic](reference-and-critic.md)。

</details>

<details class="interview" markdown="1">
<summary>Critic 是干什么的？GRPO 怎么把它省掉的？</summary>

Critic 估计状态价值 $V_t$，可用来构造类似 $\hat A_t=\hat G_t-V_t$ 的优势估计。合适的 baseline 能降方差；使用 TD 或 GAE 时，还涉及 bootstrapping 的偏差。没有 Critic 也能训练，例如 REINFORCE，并不是「没有就训不动」。

GRPO 从同一 prompt 的一组回答中构造相对信号，不训练独立 Critic。它省下 Critic 的相关成本，但增加了组内采样需求；全对、全错、reward 尺度和长度归一化都需要检查。不能只数少了几个模型，就断言总训练成本更低。

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
<summary>有了 verifier，为什么仍要独立评估？</summary>

因为 verifier 只检查写进去的条件。测试覆盖不足、答案提取错误、边界情况遗漏，都可能让错误答案得高分。程序确定性强，不代表它准确表达了全部任务目标。

RLVR 适合能自动检查结果的任务或子目标；开放式对话也可能用程序检查格式、工具参数，但这些检查覆盖不了全部回答质量。要保留独立题目、不同类型测试和必要的人类评审，确认模型学到的是任务，而不是检查器的漏洞。[具体反例](after-rlhf.md)是一段永远输出同一个列表、却能通过单个排序测试的「解法」。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>典型 PPO-style RLHF 的四个角色分别做什么？为什么不等于四份常驻权重？</li>
    <li>Reference KL 和 old-policy ratio 的参照对象分别是谁？</li>
    <li>Critic 帮助构造什么估计？GRPO 不训练 Critic，换来了哪些代价？</li>
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
