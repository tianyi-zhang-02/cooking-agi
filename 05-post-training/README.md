# 后训练：模型预训练之后还要学什么？

**中文** · [English](README.en.md)

> 阅读时间：约 12 分钟 · 类型：总览 · 最近审阅：2026-08

## 会续写文本，还不等于会完成任务

想先弄懂 RL 本身？从独立的[强化学习路线](deep-rl/README.md)开始：MDP、Bellman、Policy Gradient、DQN、SAC 与 Model-based RL，再回到这里看它们怎样用于语言模型。

预训练让模型从大量数据里学习语言、知识和通用能力。后训练则进一步教它怎样完成任务：怎么遵循指令、什么回答更合适，以及怎样根据反馈调整行为。

## 各阶段的直观分工

- **预训练**：从大规模语料中学习语言、知识和通用能力。
- **继续预训练**：进一步适应某个领域的数据和术语。
- **监督微调**：通过高质量示范学习怎样遵循指令、完成任务。
- **偏好学习**：比较不同回答，学习哪一种更符合要求。
- **强化学习**：根据任务结果或奖励信号调整策略。

这些方法各有用途，不能简单地互相替代。选哪一种，要看手上的数据和希望模型学会什么。

## 从基础模型到对齐模型：先分清每一步在学什么

<div class="bilingual-note bilingual-intro">
  <span>中英对照 · CONCEPT-BY-CONCEPT</span>
  <p>下面 4 张卡默认显示中文；点 <strong>English ↻</strong> 就能切换到对应英文，不用离开当前阅读位置。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. 三个阶段，使用三种不同的监督

最常见的主线是：

$$
\boxed{
\text{Pre-Training}
\longrightarrow
\text{SFT}
\longrightarrow
\text{Preference Alignment (RLHF / DPO / RLVR)}
}
$$

| 阶段 | 主要数据 | 训练信号 | 常见结果称呼 |
| --- | --- | --- | --- |
| 预训练 | 海量普通文本、代码等 | 根据上下文预测文本单元 | 基础模型（Base Model） |
| 监督微调（SFT） | 指令与示范回答 | 指定的理想回答 | 指令模型（Instruction Model） |
| 偏好对齐 | 被选中与被拒绝的回答、奖励或验证结果 | 哪种完整行为更好 | 对齐后的模型或策略 |

这是按训练目的划分的阶段，不代表实际训练一定会分别保存 3 份模型状态。主要区别在于：
预训练学习数据分布中的语言、知识与基础能力；SFT 教模型按示范调用这些能力；偏好对齐
再告诉它多个可行回答中哪种更符合目标。

**预训练在经典 RLHF 之前，不是 RLHF 的第一阶段。** RLHF 指利用人类反馈进行强化学习；
讲它的训练流程时，通常从监督微调这一步开始，而不是从零预训练。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. Three stages use three different kinds of supervision</div>

A common high-level path is

$$
\boxed{
\text{Pre-Training}
\longrightarrow
\text{SFT}
\longrightarrow
\text{Preference Alignment (RLHF / DPO / RLVR)}
}
$$

| Stage | Main data | Training signal | Common result label |
| --- | --- | --- | --- |
| Pre-training | large-scale text, code, and related corpora | token targets constructed from the text itself | Base Model |
| SFT | instruction–response demonstrations | selected ideal responses | Instruction Model |
| Preference alignment | chosen/rejected pairs, rewards, or verifiers | which complete behavior is better | Aligned Model / Policy |

These labels describe functional stages; an organization need not save exactly three
separate checkpoints. The useful distinction is that pre-training learns language,
knowledge, and foundational capability from the data distribution; SFT teaches the
model to invoke that capability through demonstrations; preference alignment selects
which of several plausible behaviors better serves the objective.

**Pre-training precedes classic RLHF; it is not RLHF's first stage.** In the full model
lifecycle it is upstream, while the classic RLHF pipeline usually begins from an SFT
policy.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. 同叫 Pre-Training，GPT 与 BERT 的目标并不一样

GPT 类仅解码器模型使用因果语言建模（causal language modeling）：只根据前面的文本预测下一个单元。

$$
\mathcal L_{\text{causal}}
=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

它只看左侧上下文，适合逐个文本单元（token）续写。BERT 类编码器常使用掩码语言建模（masked
language modeling）：遮住部分文本，让模型同时利用左右上下文恢复它们。后者更自然地
服务双向表征、分类和抽取，而不是自回归生成。

预训练结束后的基础模型已经会续写、模仿文本模式并编码大量知识，但不一定会把
用户问题当作必须直接回答的指令。它可能续写问题、模仿网页格式或忽略“三句话”
之类的约束，因为它优化的原始任务是预测文本，不是成为对话助手。

“预训练学能力，SFT 教模型怎样使用能力”是好用的近似，不是绝对定律：SFT 也能
改变知识和能力，只是数据规模与目标通常更偏向调整行为。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. GPT and BERT use different pre-training objectives</div>

GPT-style decoder-only models use causal language modeling:

$$
\mathcal L_{\text{causal}}
=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

They see only the left context and naturally support autoregressive continuation.
BERT-style encoders commonly use masked language modeling: selected tokens are hidden
and recovered using both left and right context. That objective more naturally supports
bidirectional representation, classification, and extraction than generation.

A pretrained Base Model can continue text, imitate textual patterns, and encode a great
deal of knowledge, yet still fail to treat a user prompt as an instruction requiring a
direct answer. It may continue the question, imitate a webpage, or ignore a constraint
such as “use three sentences,” because its original objective predicts text rather than
behaving as a chat assistant.

“Pre-training learns capability; SFT teaches the model how to use it” is a useful
approximation, not a law. SFT can also change knowledge and capability, although its
scale and objective usually emphasize behavioral shaping.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. Post-Training 是范围；LoRA 是更新参数的方法

后训练（Post-training）泛指初始基础预训练之后的训练工作，范围大于 RLHF：

$$
\text{Post-Training}\supset
\{\text{SFT, preference optimization, RL, domain/safety tuning, distillation, tool use}\}.
$$

经典 RLHF 先用成对的偏好数据训练奖励模型，再用 PPO 等强化学习算法调整模型的生成策略。
直接偏好优化（DPO）则直接用“选中哪个回答、拒绝哪个回答”的配对数据训练，不单独训练
奖励模型，也不需要在线采样并计算奖励的强化学习过程。因此 DPO 严格来说不是强化学习，
但两者都能用于偏好对齐。

还要分清两个可以独立选择的问题：

- **用什么数据和目标来教模型**：SFT、DPO、语言建模、蒸馏、RL；
- **允许哪些参数改变**：更新整个模型，或像 LoRA、适配器（adapters）那样只训练少量附加参数。

同一个 SFT 或 DPO 目标都可以全参数更新，也可以用 LoRA。把“LoRA”与“SFT”并列成
训练阶段，会把优化目标和参数更新方式混为一谈。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">3. Post-training is a scope; LoRA is an update mechanism</div>

Post-training broadly covers training performed after initial foundation pre-training
and is larger than RLHF:

$$
\text{Post-Training}\supset
\{\text{SFT, preference optimization, RL, domain/safety tuning, distillation, tool use}\}.
$$

Classic RLHF trains a Reward Model from preference pairs and then optimizes the policy
with an RL method such as PPO. DPO directly optimizes the policy from chosen/rejected
pairs, with neither a separate Reward Model nor an online RL loop. DPO is therefore not
strictly reinforcement learning, although both approaches belong to preference
alignment.

Keep two orthogonal axes separate:

- **what is trained (objective / data):** SFT, DPO, language modeling, distillation, RL;
- **how parameters are updated (parameterization):** full-parameter fine-tuning or
  LoRA / adapters.

The same SFT or DPO objective can use full updates or LoRA. Listing “LoRA” beside “SFT”
as if both were training stages confuses the objective with the update mechanism.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 4. Continued Pre-Training：时间在后，目标仍是语言建模

领域适配常出现下面这条路径：

```text
General Base Model
  → 在大量医学文本上继续 next-token prediction
  → Medical Domain Base Model
  → Medical SFT / preference alignment
```

中间一步叫继续预训练（Continued Pre-Training, CPT），用于领域适配时也叫领域自适应预训练。它发生在
初始预训练之后，但仍使用语言建模目标与无指令领域语料；SFT 则用示范回答塑造行为。

所以 “pre-training / post-training” 的边界有时取决于说话者是在按**时间阶段**还是按
**训练目标**分类。遇到模糊术语时，直接问四件事：数据是什么、目标函数是什么、从哪个
模型检查点（checkpoint）开始、最后想得到什么用途的模型。

英文也要区分：**pre-training** 是训练过程，**pre-trained model** 是完成该过程后的
模型；“pretrained LLM” 在日常语境里有时又宽泛地指一个已经完成对齐的通用模型。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">4. Continued pre-training happens later but retains a language-modeling objective</div>

A common domain-adaptation path is

```text
General Base Model
  → continue next-token prediction on a large medical corpus
  → Medical Domain Base Model
  → Medical SFT / preference alignment
```

The middle step is Continued Pre-Training (CPT) or Domain-Adaptive Pre-Training. It
occurs after initial pre-training in time but still uses a language-modeling objective
and non-instruction domain text; SFT uses demonstrations to shape behavior.

The boundary between “pre-training” and “post-training” can therefore depend on whether
the speaker is classifying by **chronological stage** or by **training objective**. When
terminology is ambiguous, ask four concrete questions: what is the data, what is the
objective, which checkpoint is the starting point, and what kind of model should result?

English terminology also differs: **pre-training** is a process, while a
**pre-trained model** has completed that process. In casual usage, “pretrained LLM” may
even refer broadly to a general-purpose model that has already been aligned.

</div>
</section>

## SFT 在做什么

SFT 用输入—输出示范教模型模仿目标行为。它适合：

- 学习固定任务格式；
- 学会按指令完成任务；
- 把专家过程蒸馏给模型；
- 让模型先学会一个相对稳定的行为起点。

它的限制是只能模仿数据里出现的行为。示范没有覆盖的长尾情况，模型未必知道怎样处理。

## Preference Learning 在做什么

当“最好的答案”很难直接写出来，但人可以比较 A 和 B 时，可以学习偏好。

DPO 等方法把偏好对直接变成策略训练目标，不必另建一套奖励模型训练与评分流程。但仍要检查：偏好标签是否稳定、候选是否有足够差异、训练分布是否接近真实使用场景。

## RL 在做什么

RL 更适合需要多步行动、结果延迟，或者策略必须通过探索学习的任务。

但 RL 并不会自动把弱反馈变强。如果奖励稀疏、含义模糊，或只反映旧策略让用户看到的内容，模型可能只学会利用指标漏洞。

例如推荐场景里，一个点击同时受到曝光位置、标题和用户时间影响。直接把点击当 reward，模型可能学会更强的吸引点击，而不是更高的长期内容价值。

## 为什么数据经常比算法更先成为瓶颈

如果数据缺少纵向关系，同一个用户的行为被切成许多孤立样本，那么模型很难学习长期意图变化。

如果训练数据无法说明样本由什么策略产生，也很难判断模型学到的是用户偏好还是旧系统偏差。

所以需要一个 objective-aware 的数据层，明确：

- 当前训练目标是什么；
- 哪些样本符合这个目标；
- 怎样采样和组合；
- 数据与模型版本怎样追踪；
- 训练和 serving 中的字段语义是否一致；
- 不同实验是否真正可比较。

## 怎样选择方法

可以先问：

| 问题 | 更可能的起点 |
| --- | --- |
| 有高质量标准答案吗？ | SFT |
| 答案难写，但人容易比较吗？ | Preference learning / DPO |
| 需要多步探索，最终结果可验证吗？ | RL |
| 模型缺少领域知识和分布覆盖吗？ | Continued pretraining |
| 反馈本身稀疏、偏置、无法归因吗？ | 先修数据与评估，而不是先换算法 |

## 怎样评估

不能只看训练损失或一个总奖励分。还要检查：

- 新行为是否来自目标机制，而不是数据泄漏；
- 不同用户和任务分组是否都受益；
- 模型是否牺牲多样性、校准或安全性换取一个指标；
- 离线偏好是否转化为真实任务结果；
- 训练和推理时使用的策略、对数概率（log-probability）与数据含义是否一致。

## 这个系列怎么读

先分清两件事：**用什么信号教**，以及**允许哪些参数改变**。如果已经了解 SFT，可以直接去看偏好学习；遇到不熟的概念再回来补，不必从第一页重新读。

**一、打地基：为什么还需要教**

1. 为什么预训练完还不够（本页）—— SFT、偏好学习、RL 各自解决什么学习问题
2. [模型适配：Full Fine-Tuning、LoRA、Prompt Tuning 与蒸馏](model-adaptation.md) —— 区分 learning objective 与 parameterization，并解释分类输出怎样被系统约束
3. [SFT](sft-and-its-ceiling.md) —— 为什么平均 token loss 不能代替任务成功率，以及示范怎样教会回答、澄清或拒答
4. [偏好与奖励模型](where-preferences-come-from.md) —— Bradley–Terry 如何用分数差解释偏好，以及策略变化后奖励模型还是否可靠

想把两种常用方法算清楚，可以接着读 [LoRA / QLoRA](lora-and-qlora.md)和[蒸馏](distillation.md)：前者算参数、梯度和显存，后者算软标签、KL 和 top-k 省略了什么。

想知道 SFT 示范怎么从模型输出里筛出来，接着读[拒绝采样与数据筛选](rejection-sampling.md)：区分“挑一份交给用户”和“挑一批继续训练”，再检查筛选是否悄悄改变了题目比例。

**二、用结果教：RL 那条线**

5. [RLHF 的三个阶段，和后来发生了什么](rlhf/) —— 四个模型，谁在训谁被冻住
6. [PPO 之后](after-ppo.md) —— GRPO / RLOO / REINFORCE++ / DAPO / DPO 怎样使用不同的信号和目标
7. [可验证奖励](verifiable-rewards.md) —— 从排序测试看奖励漏洞，再看稀疏反馈与过程检查

**三、怎么真的跑起来**

8. [后训练的基础设施：采样、数值、上下文](post-training-infrastructure.md) —— 三样都不改目标函数，但决定你能对目标函数做什么。**这一层最少被讨论，而效果常常卡在这里**
9. [同一个基座能走多远](same-base-different-posttraining.md) —— 一次难得的自然实验：基座锁死、只改后训练，看看第 8 篇那三件事值多少

**四、代价**

10. [对齐的取舍](alignment-tax.md) —— 检查具体任务的退化、多样性与分布变化；不能仅凭总体 pass@1 和 pass@k 判断模式坍缩

这些文章可以连着读：先选监督信号和更新方式，再看如何训练、怎样验证，以及有什么代价。

想解决具体问题，也可以从目录直接进入，不必一次读完。

## 继续阅读

- [DAPO：采样、长度和 clipping](dapo.md)：拿小 batch 检查权重与数据筛选。
- [GSPO 与 ASPO](policy-ratios.md)：序列比值、token 权重与 stop-gradient。
- [SAO 与异步训练](async-policy-learning.md)：采样还没完成，策略已经更新了怎么办。
- [数据与反馈](../01-data-and-feedback/)
- [Evaluation](../07-evaluation/)
- [Model Experience](../08-model-experience/)

## 参考论文

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
