# SFT：模仿能到哪儿，到哪儿为止

**中文** · [English](sft-and-its-ceiling.en.md)

> 阅读时间：约 12 分钟 · 类型：教学 · 最近审阅：2026-08

## SFT 学到的是条件分布

SFT 教的是「照着做」，所以它的上限就是示范的上限。但它真正危险的地方不是学不会——**是它会非常自信地学会你没打算教的东西**。

## 它在做什么

给一批「输入 → 理想输出」的示范，最大化模型产出那个输出的似然。逐 token 的交叉熵：

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

就这么简单。没有奖励、没有采样、没有环境。**它是模仿学习，不是强化学习**——模型从来不会看到"如果换一种说法会怎样"。

正因为简单，它在这些事上很好用：固定任务格式、建立基本的指令遵循、把专家的做法蒸馏进模型、以及给后面的 RL 一个合理的起点。

## 一条对话样本究竟怎样进入 SFT

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>下面三张卡默认中文；点 <strong>English ↻</strong> 可在当前位置查看等价英文。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. Pre-Training 与 SFT：公式相似，监督含义不同

预训练文本自己提供下一个 token 标签。给定 $x_1,\ldots,x_T$：

$$
\mathcal L_{\text{pretrain}}
=-\sum_{t=2}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

它不需要人为逐 token 标注，因此称为 self-supervised learning。SFT 的示范则告诉模型：
给定 system 和 user 上下文，理想的 assistant 行为是什么。示范可以由人编写、由模型
生成后筛选，或由多种来源组合；“supervised”指目标行为被外部选定，不等于每个字都由
人手写。

二者都能使用 next-token cross-entropy，但不能说训练“完全相同”。它们的数据来源、
序列结构、loss mask、数据混合与优化目标不同：预训练主要学习语言分布，SFT 主要把
已有能力塑造成指定行为。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. Pre-training and SFT: similar equations, different supervision</div>

Pre-training text supplies its own next-token targets. Given $x_1,\ldots,x_T$:

$$
\mathcal L_{\text{pretrain}}
=-\sum_{t=2}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

No human must label each token, so this is self-supervised learning. An SFT
demonstration instead specifies the desired assistant behavior for a system-and-user
context. Demonstrations may be human-written, model-generated and filtered, or drawn
from mixed sources; “supervised” means the target behavior is externally selected, not
that every character was typed by a person.

Both stages can use next-token cross-entropy, but their training is not “completely the
same.” Data provenance, sequence structure, loss masks, mixtures, and optimization
intent differ: pre-training learns a language distribution, while SFT shapes existing
capability into selected behavior.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. Label Shift 仍然存在，但只有部分目标计入 Loss

对话先经过 [chat template 与 tokenizer](../00-foundations/core/tokenization.md) 得到
序列 $z_1,\ldots,z_T$。输入和标签仍然错开一位：模型根据 $z_{<t}$ 预测 $z_t$。
常见的 assistant-only objective 再加入 mask：

$$
\mathcal L_{\text{SFT}}
=-\sum_{t=2}^{T}m_t\log p_\theta(z_t\mid z_{<t}),
\qquad m_t\in\{0,1\}.
$$

```text
system / user / assistant role marker  → m_t = 0
assistant answer / end-of-message      → m_t = 1
```

这样 system 和 user 仍在左侧上下文里影响预测，但不会因“是否被模型复述得好”贡献损失。
训练代码常把忽略位置的 label 设为 `-100`，让 cross-entropy 跳过它们。

这是一种常见方案，不是唯一方案。多轮数据可能训练所有 assistant turns，也可能只训练
最后一轮；有些训练配置对整段序列计算 loss。最危险的工程错误不是选哪种，而是模板
边界与 mask 错位，让 user 文本或 padding 意外进入目标。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. Label shifting remains, but only selected targets contribute loss</div>

The conversation passes through the
[chat template and tokenizer](../00-foundations/core/tokenization.md) to produce
$z_1,\ldots,z_T$. Inputs and labels are still shifted by one position: the model uses
$z_{<t}$ to predict $z_t$. A common assistant-only objective adds a mask:

$$
\mathcal L_{\text{SFT}}
=-\sum_{t=2}^{T}m_t\log p_\theta(z_t\mid z_{<t}),
\qquad m_t\in\{0,1\}.
$$

```text
system / user / assistant role marker  → m_t = 0
assistant answer / end-of-message      → m_t = 1
```

System and user tokens remain in the left context and affect every prediction, but
their reconstruction does not contribute loss. Training code commonly assigns ignored
labels the value `-100`, which tells cross-entropy to skip those positions.

This is common, not universal. Multi-turn data may train every assistant turn or only
the final one, and some training setups score the full sequence. The most dangerous engineering
failure is not choosing one policy over another; it is misaligning template boundaries
and masks so user text or padding accidentally becomes a target.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. End Token 也是行为监督

如果目标回答只包含“我很好。”却不包含 end-of-message / EOS，模型只学会怎样开始和
延续回答，没有收到“此处应该停”的明确监督。常见 SFT target 会把结束标记也设为
$m_t=1$：

```text
我 → 很好 → 。 → <|im_end|>
```

推理系统把对应 token 放入 stop set，检测到它便终止当前消息。这里必须同时对齐三件事：
训练 target 中的结束标记、tokenizer 的 special-token ID，以及推理引擎的停止配置。

长度上限仍然需要作为安全兜底，但它不等价于让模型学会自然结束；前者是系统强制截断，
后者是模型给“回答完成”分配高概率。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">3. The end token is behavioral supervision too</div>

If a target contains only “I am fine.” but omits an end-of-message or EOS token, the
model learns how to begin and continue the answer but receives no explicit supervision
that it should stop there. A common SFT target therefore gives the end marker
$m_t=1$ as well:

```text
I → am fine → . → <|im_end|>
```

The inference system places the corresponding token in its stop set and ends the
message when it appears. Three pieces must agree: the ending marker used as a training
target, its special-token ID in the tokenizer, and the inference engine's stop
configuration.

A maximum-length limit remains a necessary safety fallback, but it is not the same as
teaching natural termination. One forcibly truncates the system; the other makes the
model assign high probability to “the answer is complete.”

</div>
</section>

## 限制一：只会数据里有的

这条最直白，也最常被低估。示范没覆盖到的情况，模型没有任何依据知道该怎么办。

麻烦在于**它不会告诉你它没覆盖到**。模型面对没见过的情形照样会流利地输出——模仿学到的是"该长什么样"，而不是"这件事我知不知道"。

## 限制二：交叉熵对个别 token 不敏感

损失是所有 token 的和。把一句话里的否定词改掉，总损失几乎不变——一个 token 的贡献被几百个 token 摊薄了。

但在语言里，**一个否定词可以让整句话的意思反过来**。

所以 SFT 的优化目标和你真正关心的东西之间，有一处结构性错配：**它对"整体像不像"很敏感，对"关键处对不对"不敏感。** 这也是 RL 在这里有位置的原因之一——奖励是对整段输出给的，可以对这种局部反转敏感。

## 限制三：它逼模型给答案

这条最隐蔽，后果也最严重。

示范数据里**永远有答案**。没有人会写一条「问：X 是什么？答：我不知道」放进 SFT 集。于是模型从数据的整体形状里学到的是：**面对提问，产出一个答案**。

当模型内部并没有这个知识时，这条习得的规律仍然生效——它会用和有把握时一模一样的语气，说出一件它不知道的事。

**幻觉在这里不是 bug，是训练目标的忠实结果。** 你教它模仿"有答案的样子"，它就学会了在没有答案的时候也做出那个样子。

RL 能处理这件事，是因为奖励函数可以给「放弃回答」一个中等分数、给「编造」一个很低的分数。这个分数结构在 SFT 里没有位置可放——**SFT 只有一个正确答案，没有"选项之间的相对代价"。**

## 一个推论：示范里的知识不在模型里时，SFT 教的是语气

把上面三条合起来会得到一个不太舒服的结论。

如果示范的内容超出了模型在预训练里学到的东西，SFT 并不能把知识"装进去"——梯度只够调整表达方式，装不进事实。模型学到的是**用专家的口吻讲它并不掌握的内容**。

判据：如果一个能力在基座里完全不存在，SFT 通常提升的是**格式和自信度**，不是准确率。这时候该做的是继续预训练或检索，不是加更多示范。

## 那 SFT 什么时候就够了

不要反过来读成"SFT 没用"。它在这些情况下是最优解：

- **有标准答案、格式固定**——比如结构化抽取、格式转换。RL 在这里是杀鸡用牛刀。
- **需要一个稳定的行为起点**——从随机策略开始 RL 探索太贵。SFT 先把策略推进合理区域，RL 再在里面精调。这是 RLHF 三阶段里 SFT 排第一的原因。
- **能力已经在基座里，只是不会被调用**——这时示范起的是"打开开关"的作用。

## SFT 和 RL 的分工

一句话：**SFT 给起点，RL 给方向。**

| | SFT | RL |
| --- | --- | --- |
| 信号粒度 | 每个 token | 整段输出 |
| 需要什么数据 | 理想输出的示范 | 一个能给分的东西（人、模型或程序） |
| 能表达"不知道"吗 | 不能 | 能——给不同选项不同分数 |
| 对局部反转敏感吗 | 不敏感 | 可以敏感 |
| 探索 | 没有 | 有，也因此更贵更难调 |

它们不是两个可以互换的按钮，而是在解决不同的学习问题。

## 设计 SFT 数据时检查什么

1. 我的示范数据覆盖了哪些情况？没覆盖的那些，模型会怎么表现——我测过吗？
2. 我关心的正确性，是"整体像不像"还是"某几个关键 token 对不对"？如果是后者，交叉熵测不出来。
3. 我的 SFT 集里有没有「拒绝回答」的样本？没有的话，我实际上在教模型永远给答案。
4. 我要教的知识，基座里到底有没有？如果没有，我提升的可能只是语气。
5. 这个任务真的需要 RL 吗？有标准答案且格式固定的话，SFT 加好数据通常更划算。

## 继续阅读

- [RLHF 的三个阶段](rlhf-pipeline.md)：SFT 之后那两个阶段在做什么
- [PPO 之后](after-ppo.md)：RL 那条线的算法谱系
- [数据与反馈](../01-data-and-feedback/)：示范和偏好数据本身的质量问题
