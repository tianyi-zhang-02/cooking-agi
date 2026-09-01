# Post-Training：模型预训练之后还要学什么？

**中文** · [English](README.en.md)

> 阅读时间：约 12 分钟 · 类型：总览 · 最近审阅：2026-08

## Post-Training 改变的是模型行为

Pretraining 让模型学会“世界里通常有什么”，Post-Training 则让它学会“面对某类任务时，应该怎样表现和行动”。

## 各阶段的直观分工

- **Pretraining**：从大规模语料中学习语言、知识和通用能力。
- **Continued Pretraining**：继续适应某个领域的数据分布和术语。
- **SFT**：通过高质量示范学习指令遵循和目标行为。
- **Preference Learning**：通过回答之间的比较学习偏好。
- **RL**：根据结果或可验证奖励调整完整的行动策略。

它们不是互相替代的按钮，而是在解决不同学习问题。

## 从 Base Model 到 Aligned Model：先把地图画对

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>下面四张卡默认中文；点 <strong>English ↻</strong> 可在当前位置查看等价英文。</p>
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
| Pre-Training | 海量普通文本、代码等 | 文本自身构造的 token 目标 | Base Model |
| SFT | instruction–response demonstrations | 指定的理想回答 | Instruction Model |
| Preference Alignment | chosen/rejected、reward 或 verifier | 哪种完整行为更好 | Aligned Model / Policy |

这些名称描述的是功能阶段，不保证每家公司一定保存三个独立 checkpoint。最重要的区别是：
预训练学习数据分布中的语言、知识与基础能力；SFT 教模型按示范调用这些能力；偏好对齐
再告诉它多个可行回答中哪种更符合目标。

**Pre-Training 在经典 RLHF 之前，不是 RLHF 的第一阶段。** 若从完整模型生命周期看，
它位于上游；经典 RLHF pipeline 通常从一个 SFT 起点开始讨论。

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

GPT 类 decoder-only 模型使用 causal language modeling：

$$
\mathcal L_{\text{causal}}
=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

它只看左侧上下文，天然适合逐 token 续写。BERT 类 encoder 模型常使用 masked
language modeling：遮住部分 token，让模型同时利用左右上下文恢复它们。后者更自然地
服务双向表征、分类和抽取，而不是自回归生成。

预训练结束后的 Base Model 已经会续写、模仿文本模式并编码大量知识，但不一定会把
用户问题当作必须直接回答的 instruction。它可能续写问题、模仿网页格式或忽略“三句话”
之类的约束，因为它优化的原始任务是预测文本，不是成为对话助手。

“Pre-Training 学能力，SFT 教模型怎样使用能力”是好用的近似，不是绝对定律：SFT 也能
改变知识和能力，只是数据规模与目标通常更偏向行为塑形。

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

Post-Training 泛指初始基础预训练之后的训练工作，范围大于 RLHF：

$$
\text{Post-Training}\supset
\{\text{SFT, preference optimization, RL, domain/safety tuning, distillation, tool use}\}.
$$

经典 RLHF 使用偏好对训练 Reward Model，再用 PPO 等 RL 方法优化 policy；DPO 则直接
从 chosen/rejected pairs 优化 policy，不训练独立 Reward Model，也没有在线 RL loop。
因此 DPO 严格来说不是强化学习，但两者都属于 preference alignment。

还要区分两条正交的轴：

- **训练什么（objective / data）**：SFT、DPO、语言建模、蒸馏、RL；
- **怎样更新参数（parameterization）**：full-parameter fine-tuning、LoRA / adapters。

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

中间一步叫 Continued Pre-Training（CPT）或 Domain-Adaptive Pre-Training。它发生在
初始预训练之后，但仍使用语言建模目标与无指令领域语料；SFT 则用示范回答塑造行为。

所以 “pre-training / post-training” 的边界有时取决于说话者是在按**时间阶段**还是按
**训练目标**分类。遇到模糊术语时，直接问四件事：数据是什么、目标函数是什么、从哪个
checkpoint 开始、最后想得到什么用途的模型。

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
- 建立基本 instruction following；
- 把专家过程蒸馏给模型；
- 让模型先学会一个相对稳定的行为起点。

它的限制是只能模仿数据里出现的行为。示范没有覆盖的长尾情况，模型未必知道怎样处理。

## Preference Learning 在做什么

当“最好的答案”很难直接写出来，但人可以比较 A 和 B 时，可以学习偏好。

DPO 等方法把偏好对直接变成策略训练目标。它比写完整 reward pipeline 简单，但仍然依赖几个假设：偏好标签是否稳定、候选是否有足够差异、训练分布是否接近真实使用场景。

## RL 在做什么

RL 更适合需要多步行动、结果延迟，或者策略必须通过探索学习的任务。

但 RL 并不会自动把弱反馈变强。如果 reward 稀疏、含义模糊或被策略偏置污染，模型可能只学会利用指标漏洞。

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

不能只看训练 loss 或一个总 reward。还要检查：

- 新行为是否来自目标机制，而不是数据泄漏；
- 不同用户和任务 slice 是否一致改善；
- 模型是否牺牲多样性、校准或安全性换取一个指标；
- 离线偏好是否转化为真实任务结果；
- 训练和推理时的 policy、log-probability 与数据语义是否一致。

## 这个系列怎么读

后训练归根到底只有两件事：**用示范教（SFT）**，和**用结果教（RL）**。这个系列按依赖顺序铺开——前面不读，后面读了也用不上。

**一、打地基：为什么还需要教**

1. 为什么预训练完还不够（本页）—— SFT、偏好学习、RL 各自解决什么学习问题
2. [SFT：模仿能到哪儿，到哪儿为止](sft-and-its-ceiling.md) —— 交叉熵为什么对关键 token 不敏感，以及为什么"永远给答案"的示范会把幻觉训进去
3. [偏好从哪来：奖励模型与它学到的东西](where-preferences-come-from.md) —— Bradley-Terry 只学到序不学尺度，以及奖励模型会随策略漂移而**过期**

**二、用结果教：RL 那条线**

4. [RLHF 的三个阶段，和后来发生了什么](rlhf-pipeline.md) —— 四个模型，谁在训谁被冻住
5. [PPO 之后：每个算法都在删掉它的一部分](after-ppo.md) —— GRPO / RLOO / REINFORCE++ / DAPO / DPO 的统一读法
6. [可验证奖励：当奖励不需要学](verifiable-rewards.md) —— 它收窄了 reward hacking，但没有消灭它；可验证性是连续谱不是二分

**三、怎么真的跑起来**

7. [后训练的基础设施：采样、数值、上下文](post-training-infrastructure.md) —— 三样都不改目标函数，但决定你能对目标函数做什么。**这一层最少被讨论，而效果常常卡在这里**
8. [同一个基座能走多远](same-base-different-posttraining.md) —— 一次难得的自然实验：基座锁死、只改后训练，看看第 7 篇那三件事值多少

**四、代价**

9. [对齐税：变听话之后失去了什么](alignment-tax.md) —— 用分布的宽度换分布外的稳健。**pass@1 涨而 pass@k 掉，就是你把分布压成了点**

九篇齐了，这条线是闭合的：**为什么需要教 → 用示范教 → 用结果教 → 怎么真的跑起来 → 代价是什么。**

按顺序读一遍大约一小时。只想解决手头某个具体问题的话，每篇开头那句「先用一句话讲清楚」就是索引。

## 继续阅读

- [数据与反馈](../01-data-and-feedback/)
- [Evaluation](../07-evaluation/)
- [Model Experience](../08-model-experience/)

## 参考论文

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
