# SFT：一条对话怎样变成训练信号

**中文** · [English](sft-and-its-ceiling.en.md)

> 阅读时间：约 18 分钟 · 类型：教学 · 最近审阅：2026-10

<span id="sft"></span>

## SFT 学到的是条件分布 {#sft_2}

给模型一批客服对话，示范怎样引用退货规则、怎样询问订单号，它可以学会类似的回答方式。但如果示范里总是立即答复，几乎没有“信息不够，先问一句”，模型也可能学到这个习惯。

SFT 的问题不只是有多少样本，更是这些样本示范了什么。它能泛化，也能学习新知识，不能简单说能力上限等于示范上限；只是这些变化都需要在未见过的情境里验证。

先看一条对话是怎么训练的，再讨论数据能教会什么。已经在做实验的话，可以直接跳到[检查 loss mask](#mask-example)或[什么时候尝试 RL](#rl-readiness)。

## 它在做什么 {#_1}

给一批「输入 → 理想输出」的示范，最大化模型产出那个输出的似然。逐 token 的交叉熵：

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

普通离线 SFT 的这一步不需要模型先尝试回答、再根据奖励调整。它用已经选好的示范更新参数，属于模仿学习。示范本身可以来自模型采样和筛选；不要把“这次更新不用 rollout”理解成“整个数据流程都没有采样”。

正因为简单，它在这些事上很好用：固定任务格式、建立基本的指令遵循、把专家的做法蒸馏进模型、以及给后面的 RL 一个合理的起点。

## 一条对话样本究竟怎样进入 SFT {#sft_3}

<div class="bilingual-note bilingual-intro">
  <span>这 3 步也可以对照英文看</span>
  <p>点卡片上的 <strong>English ↻</strong>，会在原位置切换语言。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. 都在预测下一个 token，但数据教的东西不同 {#1-pre-training-sft}

预训练文本自己提供下一个 token 标签。给定 $x_1,\ldots,x_T$：

$$
\begin{aligned}
\mathcal L_{\text{pretrain}}&=\sum_{t=2}^{T}\ell_t,\\
\ell_t&=-\log p_\theta(x_t\mid x_{<t}).
\end{aligned}
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
\begin{aligned}
\mathcal L_{\text{pretrain}}&=\sum_{t=2}^{T}\ell_t,\\
\ell_t&=-\log p_\theta(x_t\mid x_{<t}).
\end{aligned}
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

### 2. 预测下一位，不代表每一位都要算 loss {#2-label-shift-loss}

对话先经过 [chat template 与 tokenizer](../00-foundations/core/tokenization.md) 得到
序列 $z_1,\ldots,z_T$。输入和标签仍然错开一位：模型根据 $z_{<t}$ 预测 $z_t$。
常见的 assistant-only objective 再加入 mask：

$$
\begin{aligned}
\mathcal L_{\text{SFT}}&=\sum_{t=2}^{T}m_t\ell_t,\\
\ell_t&=-\log p_\theta(z_t\mid z_{<t}).
\end{aligned}
$$

其中 $m_t\in\{0,1\}$ 指定哪些目标参与监督。下面是一种简化模板，不代表所有模型的 role marker 都必须忽略：

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
\begin{aligned}
\mathcal L_{\text{SFT}}&=\sum_{t=2}^{T}m_t\ell_t,\\
\ell_t&=-\log p_\theta(z_t\mid z_{<t}).
\end{aligned}
$$

Here $m_t\in\{0,1\}$ selects supervised targets. This simplified template is an example; role-marker supervision depends on the actual template.

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

### 3. 回答结束，也要有一个信号 {#3-end-token}

如果这条目标回答只包含“我很好。”却不包含 end-of-message / EOS，这条样本就没有提供“此处应该停”的明确监督；不代表模型原本完全不会停止。常见 SFT target 会把结束标记也设为
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

If this target contains only “I am fine.” without an end-of-message or EOS token, this example supplies no explicit stopping supervision. The model may already know how to stop from earlier training. A common SFT target gives the end marker
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

## 具体看一次 mask 和 shift {#mask-example}

用户问“2 + 3 等于几？”，示范回答“5”。为方便看对齐关系，假设整条序列只有 6 个符号：

```text
位置        0       1          2          3     4     5
input    <BOS>    问题     <assistant>    5   <EOS> <PAD>
label     -100   -100        -100        5   <EOS> -100
```

真实 tokenizer 会把问题拆成多个 token；这里的“问题”只是一个示意单位。labels 先和 input 保持同位置，**计算 causal loss 时才错开一位**：

| 用哪个位置的输出 | 预测什么 | 这次算 loss 吗？ |
| --- | --- | --- |
| `<BOS>` 后 | 问题 | 不算 |
| 问题后 | `<assistant>` | 本例不算 |
| `<assistant>` 后 | `5` | 算 |
| `5` 后 | `<EOS>` | 算 |
| `<EOS>` 后 | `<PAD>` | 不算 |

所以 `5` 的 label 在位置 3，却使用位置 2 的 logits。直接用位置 3 的输出预测位置 3 的 `5`，就把当前 token 当成了自己的答案。

下面不加载模型，只造一组可手算的 logits。两个有效目标的概率分别设为 0.8 和 0.6，平均 loss 应为 $(-\log0.8-\log0.6)/2\approx0.3670$。

代码把答案文本 `5` 的 token ID 设为 `4`，把 `<EOS>` 的 ID 设为 `5`；文本和词表编号不是一回事。这是人为设定的小词表，不对应真实 tokenizer。

```python
import math
import torch
import torch.nn.functional as functional

probabilities = torch.full((1, 6, 6), 1 / 6, dtype=torch.float64)
probabilities[0, 2] = torch.tensor([.04, .04, .04, .04, .8, .04], dtype=torch.float64)
probabilities[0, 3] = torch.tensor([.08, .08, .08, .08, .08, .6], dtype=torch.float64)
logits = probabilities.log().requires_grad_()
labels = torch.tensor([[-100, -100, -100, 4, 5, -100]])
loss = functional.cross_entropy(
    logits[:, :-1].reshape(-1, 6),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
assert math.isclose(loss.item(), -(math.log(.8) + math.log(.6)) / 2, abs_tol=1e-7)
loss.backward()
active_positions = (logits.grad.abs().sum(-1) > 0).nonzero().tolist()
assert active_positions == [[0, 2], [0, 3]]
```

这里手动 shift，是因为我们直接调用 CE。[Transformers 的 causal LM loss 实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/loss/loss_utils.py)会做这一步；把 labels 交给已处理 shift 的模型时，不要再提前 shift 一遍。

### Mask 了 user，为什么模型还会学着理解问题？

Loss mask 决定“哪些预测要评分”，不是“哪些文字可以看”。预测 `5` 时仍然需要读到前面的 `2 + 3`。梯度也能通过 attention、共享参数等路径传回处理上下文的计算。**不监督问题本身的复述，不等于切断问题对学习的影响。**

Attention mask 是另一件事：它控制可见位置，例如不看未来 token、不看 padding。若把 user 在 attention 里也屏蔽了，模型会失去答题所需的上下文。

### 多轮对话和工具调用怎么选？

| 数据里的内容 | 常见 assistant-only 处理 | 需要确认的地方 |
| --- | --- | --- |
| system / user | 提供上下文，不作为目标 | 这是训练选择，不是所有任务的硬规定 |
| assistant 调用工具 | 可以监督调用名和参数 | 调用格式应由模板准确标出 |
| tool 返回结果 | 通常作为后续回答的上下文 | 不要误把工具消息当成 assistant 回答 |
| assistant 最终回复 | 监督内容与结束标记 | 训练所有轮，还是只训练最后一轮？ |
| padding | 不参与 loss | 若 PAD 与 EOS 共用 ID，要按位置区分 |

[TRL 的版本化说明](https://huggingface.co/docs/trl/v0.23.1/sft_trainer#train-on-assistant-messages-only)分别提供 assistant-only 与 completion-only 配置，不能只凭选项名字判断是否等价。检查真实模板返回的 mask，再解码几条最终 labels，看看训练到底会奖励什么。

长对话被截断时尤其要看：是不是只剩问题、没剩回答？结束标记是否被截掉？全部 labels 都是 `-100` 的样本没有有效监督，应在数据检查阶段处理。Packing 还需要独立的样本边界与 attention 检查，loss mask 本身不能隔开两条对话。

## 没见过的情况，得单独测 {#_2}

训练集里只有普通退货，测试时却遇到跨境订单和已拆封商品。模型可能根据预训练知识和相似示范处理好，也可能套用错误规则。没有覆盖不等于必然失败，但我们不能拿训练损失下降来证明它已经会处理这些情况。

把不同条件单独留作测试：未见过的表达、组合和例外。这样才知道它学到了任务规则，还是主要记住了熟悉的回答。

## Loss 降了，回答真的更好了吗？ {#token}

“可以退款”和“不可以退款”只差一个字，业务含义却完全相反。交叉熵当然会惩罚这个 token 的错误，而且惩罚可以很大；问题是普通逐 token 加权并不知道哪个位置对业务最重要。

假设目标 token 的预测概率从 0.9 降到 0.1，该项负对数似然从约 0.105 增到 2.303。若报告的是整段 200 个 token 的平均 loss，而其他项不变，平均值只增加约 0.011。不能只看平均 loss，还要检查退款判断是否正确。

可以补关键情形的示范、调整样本或 token 权重，也可以增加任务级检查。RL 是一种选择，不是修复这个问题的唯一办法。

## 该追问的时候，数据有没有教过？ {#_3}

如果所有示范都自信地回答，模型就很少练习澄清、拒答或承认不知道。但这不是 SFT 做不到：我们完全可以加入“请先提供订单号”或“现有资料不足以判断”的示范。

真正难的是示范要和可见信息匹配。同一问题，给足证据时应当回答，缺关键信息时应当追问。只加一批统一的拒答模板，又可能让模型在能回答时也回避。

偏好训练或 RL 可以比较不同选择的代价，但也依赖奖励是否合理。没有哪个训练名字能单独保证诚实。

## 知识是否学到了，怎么检查？ {#sft_4}

SFT 可以把事实学进参数里，但不保证一次出现的事实被可靠记住，也不保证换种问法还能答对。流畅的专业语气更不能代替事实核对。

例如公司退款期限从 30 天改成 14 天。重新微调可以是一种办法；如果规则经常变，检索当前政策通常更方便更新和追溯。两种方案都要测试旧规则是否残留、引用是否正确，以及资料缺失时会不会编造。

选择 SFT、继续预训练还是检索，取决于知识量、更新频率和任务形式，不是“事实一律不能靠 SFT 学”。

## 那 SFT 什么时候就够了 {#sft_5}

如果下面这些检查已经过关，就未必需要把训练流程做得更复杂：

- **格式与任务质量都达标**。例如结构化抽取，不只看 JSON 能否解析，还要看字段是否正确、缺失信息是否处理得当。
- **想先建立可靠的行为起点**。示范能教会工具协议和回答习惯；它是否值得作为 RL 的前置步骤，仍取决于基座能力和任务。
- **主要问题是调用已有能力的方式**。先试少量高质量示范，观察未见过的输入是否也改善，而不是默认需要更重的算法。

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

## SFT 训练到什么程度，可以试 RL？ {#rl-readiness}

没有通用的“loss 低于 0.5 就可以开始”。Tokenizer、回答长度、mask 和平均方式都能改变这个数字。更实用的是挑几个 checkpoint，在同一批未参与训练的问题上生成回答，记录它们会怎样失败。

比如要训练一个会调用订单工具的客服助手。下面是**示意实验，不是实测结果，也不是上线阈值**：

| 检查项 | checkpoint A | checkpoint B | 怎么理解 |
| --- | --- | --- | --- |
| 100 次调用中可解析的参数 | 72 | 96 | A 还常卡在协议上，先修模板和示范可能更直接 |
| 100 个任务完成数 | 42 | 61 | B 的格式好了，但离完成任务还有差距 |
| 信息不足的 20 题中合理追问数 | 3 | 12 | 不能让总体成功率掩盖这类错误 |

接下来，不是看到 B 的数字就立刻决定用 RL，而是补 3 个问题：

1. **奖励能不能分清好坏？** 只奖励 JSON 合法，会不会把答非所问也当成功？用人工检查的小切片验证 reward / verifier。
2. **同一道题的不同尝试，有没有可学习的差别？** 对组内相对 advantage 的方法，某题所有回答得分都相同，就没有这项区分信号。全错时先查题目难度、采样和奖励；这不自动等于“必须再做 SFT”。
3. **同样预算下，RL 比简单方案多解决了什么？** 从同一 checkpoint 比较继续 SFT、生成后筛选再训练、小规模 RL。记录质量、长度、成本和原有能力回退，不只看训练 reward。

先确定观察指标和停止条件，再跑小实验。比如成功率没变、输出变长、reward 却上涨，就该检查奖励漏洞，而不是继续加步数。

“先 SFT 再 RL”是一条常见路径，不是定律。[DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948v1)就研究了不以前置 SFT 开始的 RL；这里“不做 SFT”不等于从随机权重开始。是否需要这个阶段，要根据你的模型和任务判断。

<details markdown="1">
<summary>想再深一点：DFT 改的是 token 权重，不是 RL 的启动门槛</summary>

[DFT（2025）](https://arxiv.org/html/2508.05629v1#S3.SS3)用当前目标 token 的概率给 CE 加权，并停止权重的梯度：

$$
\ell_{\text{DFT}}=-\operatorname{sg}(p_y)\log p_y.
$$

这里只讨论这个单项形式；实际批次还要指定 mask 和 reduction。它没有新增环境奖励，也没有规定“概率超过某值才进入 RL”。

一个容易写错的地方是忘记 `detach()`。取二分类目标概率 $p=0.01$，对正类 logit 求导：普通 CE 是 −0.99；带停止梯度权重后是 −0.0099。若直接求 $-p\log p$ 的完整导数，却约为 +0.0357，梯度下降反而会降低这个低概率目标的分数。

这是链式法则的区别，不是数值精度问题。[配套测试](../site/tests/test_loss_contracts.py)检查了两种梯度方向。概率加权也可能减弱困难但重要的样本，因此应在自己的保留集上比较，不把某篇论文的收益当成通用结论。

</details>

## 设计 SFT 数据时检查什么 {#sft_6}

1. 我的示范数据覆盖了哪些情况？没覆盖的那些，模型会怎么表现——我测过吗？
2. 我关心的正确性，是"整体像不像"还是"某几个关键 token 对不对"？如果是后者，除了平均 loss，还应单独测关键判断。
3. 我的 SFT 集里有没有「拒绝回答」的样本？没有的话，需要测试信息不足时是否仍会硬答。
4. 我要教的知识，基座里到底有没有？换一种提问方式后，它还能给出正确的事实和依据吗？
5. 这个任务真的需要 RL 吗？有标准答案且格式固定的话，SFT 加好数据通常更划算。

## 继续阅读 {#_4}

- [交叉熵怎么实现](../00-foundations/pytorch/cross-entropy.md)：稳定计算、soft target、mask 和分母
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
