# 偏好从哪来：奖励模型与它学到的东西

**中文** · [English](where-preferences-come-from.en.md)

> 阅读时间：约 11 分钟 · 类型：教学 · 最近审阅：2026-10

<span id="_1"></span>

## 偏好数据提供的是相对顺序 {#_2}

用户想知道怎样修复一个报错。回答 A 给了可以运行的改法，回答 B 解释得很长，却没有解决问题。标注者选 A，记录下来的就是“在这个问题和这两个回答之间，更偏好 A”。

这还不是 A 的客观质量分，也不说明换一位用户仍会这样选。奖励模型尝试从许多这样的比较中学到规律，再给新的回答打分。偏好数据怎么收集，直接影响这个分数的含义。

## 为什么是"比较"而不是"打分" {#_3}

让两个人分别打 1–10 分，他们心里的 8 分可能不一样。给出同一个问题的两个回答，按明确标准比较，往往更容易保持尺度一致。因此，一种常见数据格式是提示 $x$、优选回答 $y_w$ 和另一个回答 $y_l$。

但成对比较不是天然可靠：两个回答都差、好在不同方面，或标注者重视的标准不同，仍然会有分歧。可以记录平局、两者都不合格以及分歧原因，不必强行每次选一个赢家。

## Bradley-Terry：把比较变成分数 {#bradley-terry}

要把"两两比较"变成一个能给任意回答打分的函数，需要一个模型把**序**翻译成**标量**。Bradley-Terry 就是这个翻译器：

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

| 符号 | 意思 |
| --- | --- |
| $r(x,y)$ | 奖励模型给回答 $y$ 的标量分 |
| $\sigma$ | sigmoid，把分数差压到 (0,1) 当概率 |
| 分数**差** | 唯一进入公式的东西 —— 注意这一点 |

于是训练目标就是最大化观测到的偏好的似然：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\big(r(x,y_w) - r(x,y_l)\big)\right]$$

经典做法是在一个预训练模型上接一个标量输出头，读完整个回答给一个分。比较也可以交给模型做，分数也可以由生成式模型写出来，见[后文](#ai-feedback-and-rubrics)。

## 分数的零点与尺度 {#_4}

假设 A、B 的奖励分别是 2 和 1。Bradley–Terry 给 A 胜出的概率是 $\sigma(1)\approx0.731$。把两者都加 10，变成 12 和 11，概率没变。

但把分数都乘 2，分差变成 2，概率就变成 $\sigma(2)\approx0.881$。所以，**平移不影响这个模型，缩放却会影响**；不能从“只有分差进入公式”推出尺度任意。

<!-- widget:step-bt-scale -->

这些分数仍不是跨模型通用的效用单位。接到 RL 后，reward 的尺度会影响 advantage、梯度和它与 KL 惩罚的相对权重。是否做归一化，要连同训练目标和实现一起决定，不是一个无条件必做的修补。

## 偏好数据里的四个坑 {#_5}

可以先抽一小批样本，做四个具体检查：

| 检查 | 怎么做 | 读结果时要小心什么 |
| --- | --- | --- |
| 人与人是否同意 | 让不同标注者独立比较同一对回答 | 分歧可能来自标准不清，也可能来自真实偏好差异 |
| 是否只偏爱长回答 | 在不改变信息的前提下增加寒暄和重复 | 不要把“补充了有用信息”和“只是更长”混在一起 |
| 是否核对事实 | 提供可查询的来源，或让代码实际运行 | 看起来可信和真的正确可能得到不同排序 |
| 是否受顺序影响 | 交换 A/B，隐藏模型名称 | 把答案 ID 对齐后再比较，不要直接比较左右选项 |

标注者的一致率不是奖励模型准确率的通用上界；它取决于比较谁、标签怎样汇总以及数据分布。它更适合帮助我们定位哪些样本需要澄清标准或保留多种偏好。

## 最容易被忽略的一条：奖励模型会过期 {#_6}

奖励模型可能只见过旧策略生成的回答。Actor 更新后，开始写出另一类回答：旧验证集上仍然表现不错的 RM，也可能在这些新样本上打错分。

比如旧数据里详细解释通常更好，新策略却学会了重复相同内容来加长答案。奖励上升，用户并没有得到更多帮助。需要抽查当前策略的输出，而不只重复测旧验证集。

KL 惩罚能限制相对 Reference 的漂移，但它不保证新回答落在 RM 的可靠区域。独立人工复核、任务检查和新分布上的测试仍然需要。

## 标签和分数，也可以由模型给出 {#ai-feedback-and-rubrics}

前面默认了两件事：比较由人来做，分数由接在模型后面的标量头给出。这两件事现在都有常用的替代做法，上面的检查也都要跟着换一个对象。

**AI 来比较（RLAIF）。** [Constitutional AI（Bai 等，2022）](https://arxiv.org/abs/2212.08073)针对无害性：先写一份原则清单，让模型按这些原则判断两个回答哪个更好，用这些 AI 偏好训练偏好模型，再拿它当奖励做 RL；作者把这叫 RL from AI Feedback（RLAIF）。人仍然写原则、做评估，只是不再逐对标注。[Lee 等（ICML 2024）](https://arxiv.org/abs/2309.00267)在摘要、有用对话和无害对话三个任务上报告，RLAIF 与 RLHF 表现相当；他们还试了训练时直接让现成 LLM 打分、不另训奖励模型的做法（d-RLAIF）。换成 AI 标注，省下的是逐对标注的人力；位置偏差、长度偏好这些坑一个都没少，而且同一个评判模型的偏差会在整批数据里重复出现。

**生成式奖励模型。** 奖励模型也可以先写一段文字，再给分。[GenRM（Zhang 等，ICLR 2025）](https://arxiv.org/abs/2408.15240)用 next-token prediction 训练验证器，判断前可以先写推理，还能多采样几次做多数投票。[DeepSeek-GRM（Liu 等，2025）](https://arxiv.org/abs/2504.02495)先为当前问题写出评判原则，再写点评，最后给每个回答一个 1–10 的整数分；推理时并行采样 k 次、把各次分数相加（论文 §4），还可以先用一个 meta RM 筛掉质量差的采样。它逐个回答打分，所以同一个格式能评一条、一对或多条回答。[RM-R1（Chen 等，2025）](https://arxiv.org/abs/2505.02387)把奖励建模当成推理题：先写出这道题的评分要点或参考解，再拿候选回答去对照。代价也直接：评一次要生成一段文字，同一个回答多采样几次，分数也可能不同。

**Rubric 奖励。** 医疗问答这类没有程序检查器的任务，可以把“好回答”拆成一条条能单独判断的要求。[Rubrics as Rewards（Gunjal 等，2025）](https://arxiv.org/abs/2507.17746)让 judge 模型逐条判断回答是否满足（记 1 或 0），按权重加起来再除以总权重（v2 §2.2 式 1）：

$$r(x,\hat y)=\frac{\sum_j w_j\,c_j(x,\hat y)}{\sum_j w_j}$$

也可以把整张 rubric 交给 judge，直接要一个总分（论文叫 implicit aggregation）。论文用 Qwen2.5-7B 做 GRPO、gpt-4o-mini 当 judge，报告与直接让 judge 打 Likert 分的基线相比，HealthBench 上相对提升最多 31%，GPQA-Diamond 上 7%；这是作者自己报告的结果。

### 算一次：一条要求判错，奖励差多少 {#rubric-example}

回到开头修报错的例子。假设用户问“Python 报 `KeyError: 'id'`，怎么修？”，rubric 有 4 条，权重是自拟的教学数字：

| 要求 | 权重 | A：给了能运行的改法 | B：解释很长，没给改法 |
| --- | ---: | :---: | :---: |
| 说清原因：字典里没有 `'id'` 这个键 | 3 | 1 | 1 |
| 给出能运行的修法，例如先检查键或用 `.get` | 3 | 1 | 0 |
| 提醒修法可能掩盖上游缺字段的问题 | 1 | 0 | 0 |
| 没有编造不存在的函数或参数 | 2 | 1 | 1 |

A 得 $(3+3+2)/9\approx0.889$，B 得 $(3+2)/9\approx0.556$。现在假设 judge 把 B 的第二条误判成“满足”，B 也变成 $8/9$，和 A 一样高。如果这两个回答正好是同一道题的一组，GRPO 的组内 advantage 从约 $[1,-1]$ 变成 $[0,0]$：解决了问题的回答和没解决的回答，拿到的训练信号完全相同。

```python
weights = [3, 3, 1, 2]
answer_a = [1, 1, 0, 1]
answer_b = [1, 0, 0, 1]

def rubric_reward(checks):
    return sum(weight * check for weight, check in zip(weights, checks)) / sum(weights)

assert abs(rubric_reward(answer_a) - 8 / 9) < 1e-12
assert abs(rubric_reward(answer_b) - 5 / 9) < 1e-12
misjudged_b = [1, 1, 0, 1]
assert rubric_reward(misjudged_b) == rubric_reward(answer_a)
```

这个例子说明 rubric 要查两样东西：权重由谁定，每一条由谁判。逐条判断比一个笼统的 1–10 分好抽查，可以单独统计 judge 在“是否给出能运行的修法”这一条上的错误率；rubric 本身若也由模型生成，也要抽查它漏了什么。rubric 没写到的要求，奖励照样看不见。无论标签来自人、AI 还是 rubric，上一节的过期问题都一样：策略变了，就拿当前策略的输出重新检查评判。judge 常见的偏差和检查方法见 [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.md)。

## 读偏好训练实验时先找什么 {#_7}

读一个偏好训练实验时，我会先找三个信息：标注者拿什么标准比较，奖励模型是否评过当前策略的输出，最终结论有没有用训练奖励之外的证据核对。如果只报告 RM 分数，暂时还不知道用户是否真的更满意。

## 继续阅读 {#_8}

- [RLHF 的三个阶段](rlhf/three-stages.md)：这个奖励模型接下来被怎么用
- [可验证奖励](verifiable-rewards.md)：当奖励不需要学的时候，上面这些坑消失了哪些
- [对齐税](alignment-tax.md)：偏好里的偏置在下游长成什么样
- [数据与反馈](../01-data-and-feedback/)：标签质量的一般性问题

## 快速学习：Preference Data 与 Reward Model 的边界 {#preference-data-reward-model}

<details class="interview" markdown="1">
<summary>成对比较提供了什么，分数差又代表什么？</summary>

**快速记忆**：Chosen / rejected 记录的是两个回答中更偏好哪一个。Reward Model 拟合这些选择，给出的分数不是通用的效用单位：所有分数同加一个常数，Bradley–Terry 概率不变；同乘一个系数，概率就会变。

**面试回答**

> 偏好数据记录同一 prompt 下的选择。Bradley–Terry 用分数差拟合选择概率；分差大小也影响概率，不只是恢复排序。分数不是客观效用单位，需要检查标注分歧、顺序与长度偏差，以及策略更新后的分布变化。

<details markdown="1">
<summary><b>深挖</b>：为什么 Reward Model 会过期？</summary>

RM 学习的是旧 policy 生成的回答。Actor 更新后，可能生成训练 RM 时没见过的回答，甚至利用 RM 的盲点拿到高分。于是，旧验证集上的成绩还不错，却已经评不好新回答了。需要持续检查当前 policy 的输出，并用人工复核和单独留出的对抗样本验证。

</details>
</details>

## 参考论文 {#_9}

- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — 深度强化学习里用成对偏好训练奖励模型的代表性早期工作
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — KL 惩罚与 reward hacking 的早期实证
- [InstructGPT](https://arxiv.org/abs/2203.02155) — 三阶段流程与标注规范
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — 按原则清单让模型比较回答，即 RLAIF
- [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) — DeepSeek-GRM：先写原则和点评，再给分
- [Rubrics as Rewards](https://arxiv.org/abs/2507.17746) — 没有检查器的任务，用逐条 rubric 当奖励
