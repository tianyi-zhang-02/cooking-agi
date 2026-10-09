# SFT：模仿能到哪儿，到哪儿为止

**中文** · [English](sft-and-its-ceiling.en.md)

> 阅读时间：约 12 分钟 · 类型：教学 · 最近审阅：2026-08

<span id="sft"></span>

## SFT 学到的是条件分布 {#sft_2}

给模型一批客服对话，示范怎样引用退货规则、怎样询问订单号，它可以学会类似的回答方式。但如果示范里总是立即答复，几乎没有“信息不够，先问一句”，模型也可能学到这个习惯。

SFT 的问题不只是有多少样本，更是这些样本示范了什么。它能泛化，也能学习新知识，不能简单说能力上限等于示范上限；只是这些变化都需要在未见过的情境里验证。

## 它在做什么 {#_1}

给一批「输入 → 理想输出」的示范，最大化模型产出那个输出的似然。逐 token 的交叉熵：

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

就这么简单。没有奖励、没有采样、没有环境。**它是模仿学习，不是强化学习**——模型从来不会看到"如果换一种说法会怎样"。

正因为简单，它在这些事上很好用：固定任务格式、建立基本的指令遵循、把专家的做法蒸馏进模型、以及给后面的 RL 一个合理的起点。

## 一条对话样本究竟怎样进入 SFT {#sft_3}

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>下面 3 张卡默认显示中文；点 <strong>English ↻</strong> 就能在原位置切换到对应英文。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. Pre-Training 与 SFT：公式相似，监督含义不同 {#1-pre-training-sft}

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

### 2. Label Shift 仍然存在，但只有部分目标计入 Loss {#2-label-shift-loss}

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

### 3. End Token 也是行为监督 {#3-end-token}

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

## 限制一：覆盖之外，需要另做测试 {#_2}

训练集里只有普通退货，测试时却遇到跨境订单和已拆封商品。模型可能根据预训练知识和相似示范处理好，也可能套用错误规则。没有覆盖不等于必然失败，但我们不能拿训练损失下降来证明它已经会处理这些情况。

把不同条件单独留作测试：未见过的表达、组合和例外。这样才知道它学到了任务规则，还是主要记住了熟悉的回答。

## 限制二：平均 loss 不等于任务质量 {#token}

“可以退款”和“不可以退款”只差一个字，业务含义却完全相反。交叉熵当然会惩罚这个 token 的错误，而且惩罚可以很大；问题是普通逐 token 加权并不知道哪个位置对业务最重要。

假设目标 token 的预测概率从 0.9 降到 0.1，该项负对数似然从约 0.105 增到 2.303。若报告的是整段 200 个 token 的平均 loss，而其他项不变，平均值只增加约 0.011。不能只看平均 loss，还要检查退款判断是否正确。

可以补关键情形的示范、调整样本或 token 权重，也可以增加任务级检查。RL 是一种选择，不是修复这个问题的唯一办法。

## 限制三：澄清与拒答也需要示范 {#_3}

如果所有示范都自信地回答，模型就很少练习澄清、拒答或承认不知道。但这不是 SFT 做不到：我们完全可以加入“请先提供订单号”或“现有资料不足以判断”的示范。

真正难的是示范要和可见信息匹配。同一问题，给足证据时应当回答，缺关键信息时应当追问。只加一批统一的拒答模板，又可能让模型在能回答时也回避。

偏好训练或 RL 可以比较不同选择的代价，但也依赖奖励是否合理。没有哪个训练名字能单独保证诚实。

## 知识是否学到了，怎么检查？ {#sft_4}

SFT 可以把事实学进参数里，但不保证一次出现的事实被可靠记住，也不保证换种问法还能答对。流畅的专业语气更不能代替事实核对。

例如公司退款期限从 30 天改成 14 天。重新微调可以是一种办法；如果规则经常变，检索当前政策通常更方便更新和追溯。两种方案都要测试旧规则是否残留、引用是否正确，以及资料缺失时会不会编造。

选择 SFT、继续预训练还是检索，取决于知识量、更新频率和任务形式，不是“事实一律不能靠 SFT 学”。

## 那 SFT 什么时候就够了 {#sft_5}

不要反过来读成"SFT 没用"。它在这些情况下是最优解：

- **有标准答案、格式固定**——比如结构化抽取、格式转换。RL 在这里是杀鸡用牛刀。
- **需要一个稳定的行为起点**——从随机策略开始 RL 探索太贵。SFT 先把策略推进合理区域，RL 再在里面精调。这是 RLHF 三阶段里 SFT 排第一的原因。
- **能力已经在基座里，只是不会被调用**——这时示范起的是"打开开关"的作用。

## SFT 和 RL 的分工 {#sft-rl}

先问自己拿得到哪种可靠信号。能写出好的示范，就先做 SFT；不容易写出标准过程、却能评估尝试的结果，RL 才可能更合适。实际流程也可以把两者交替使用。

| 比较项 | SFT | RL |
| --- | --- | --- |
| 直接使用的信号 | 选定的目标输出 | 对采样行为或结果的奖励 |
| 常见更新粒度 | token 级交叉熵，可加 mask 和权重 | 奖励经 return、advantage 等分配到动作 |
| 能否学习澄清或拒答 | 可以，提供相应情境的示范 | 可以，奖励必须区分合理澄清与无故回避 |
| 是否评估当前策略的尝试 | 普通离线 SFT 不需要 | 通常需要 rollout，或使用已有轨迹与相应估计 |
| 主要检查 | 覆盖、模板、mask、泛化 | 奖励有效性、采样分布、更新稳定性 |

比较时固定任务和评估预算。更复杂的训练方法不自动等于更好的结果。

## 设计 SFT 数据时检查什么 {#sft_6}

1. 我的示范数据覆盖了哪些情况？没覆盖的那些，模型会怎么表现——我测过吗？
2. 我关心的正确性，是"整体像不像"还是"某几个关键 token 对不对"？如果是后者，除了平均 loss，还应单独测关键判断。
3. 我的 SFT 集里有没有「拒绝回答」的样本？没有的话，需要测试信息不足时是否仍会硬答。
4. 我要教的知识，基座里到底有没有？换一种提问方式后，它还能给出正确的事实和依据吗？
5. 这个任务真的需要 RL 吗？有标准答案且格式固定的话，SFT 加好数据通常更划算。

## 继续阅读 {#_4}

- [RLHF 的三个阶段](rlhf/three-stages.md)：SFT 之后那两个阶段在做什么
- [PPO 之后](after-ppo.md)：RL 那条线的算法谱系
- [数据与反馈](../01-data-and-feedback/)：示范和偏好数据本身的质量问题

## 快速学习：SFT 教会模型什么 {#sft_1}

<details class="interview" markdown="1">
<summary>Assistant-only CE、行为模仿与能力上限</summary>

**快速记忆**：SFT 仍是 next-token CE，但通常只在 assistant tokens 上计 loss；它提高示范行为的概率，不会自动发现数据里没有的策略。

**面试回答**

> 做对话 SFT 时，先把消息按模板拼起来，让模型根据 system 和 user 的内容学习生成 assistant 回答及结束 token。它适合教格式、语气、工具调用和已有解法。这仍然是模仿示范：数据覆盖得不全、质量不好，或者逐 token 的目标没体现任务成败，都会影响结果。

<details markdown="1">
<summary><b>深挖</b>：为什么低 token CE 不等于完整回答更好？</summary>

CE 会惩罚每个目标 token 的错误，但长回答的平均值可能掩盖决定成败的少数位置。需要单独检查这些判断。任务测试、针对性示范、加权以及偏好或奖励信号，都可以从不同角度突出这些位置。

</details>
</details>
