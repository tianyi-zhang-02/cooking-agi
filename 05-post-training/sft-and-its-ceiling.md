# SFT：一条对话怎样变成训练信号

**中文** · [English](sft-and-its-ceiling.en.md)

> 阅读时间：约 18 分钟 · 类型：教学 · 最近审阅：2026-10

<span id="sft"></span>

## SFT 学到的是条件分布 {#sft_2}

给模型一批客服对话，示范怎样引用退货规则、怎样询问订单号，它可以学会类似的回答方式。但如果示范里总是立即答复，几乎没有“信息不够，先问一句”，模型也可能学到这个习惯。

用这些示范训练模型，就是监督微调（supervised fine-tuning，SFT）。准备数据时，除了样本数量，还要看它们教了哪些行为。模型有可能把学到的做法用到新情境中，也可以学到新知识；具体学会了多少，要用没参与训练的任务来检验。

如果还不清楚为什么训练总要回头改数据，可以先看[数据在教模型什么](../01-data-and-feedback/data-as-supervision.md)：从一条示范开始，看看筛选、采样和标签怎样改变学习方向。

先看一条对话怎样进入训练，再讨论数据能教会什么。已经在做实验的话，可以直接跳到[检查哪些 token 参与损失](#mask-example)，或[什么时候尝试强化学习](#rl-readiness)。

## 它在做什么 {#_1}

给定一批“输入 → 目标回答”的示范，训练时提高目标回答在该输入下的似然。把输入记为 $x$，目标回答的第 $t$ 个 token 记为 $y_t$，$y_{<t}$ 表示它之前的目标文本。逐 token 的交叉熵可以写成：

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

普通离线 SFT 用已经选好的示范更新参数，不必在每次更新前让模型重新生成一批回答、再给它们打分。这属于模仿学习。不过，示范本身可以由模型生成后筛选而来：参数更新使用现成答案，不代表准备数据时没有采样。

因此，SFT 常用于学习任务格式、遵循指令、模仿专家的处理方式，也可以为后续强化学习（reinforcement learning，RL）准备一个起点。是否足够，仍要看任务测试的结果。

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

预训练不需要人为逐个标注这些目标，因此称为自监督学习（self-supervised learning）。SFT 的示范则指定：在系统指令（system）和用户消息（user）给出的上下文里，助手（assistant）应该怎样回应。示范可以由人编写、由模型生成后筛选，也可以混合多种来源。“监督”指我们选定了要学习的行为，不代表每个字都要人手写。

二者都可以用交叉熵训练下一个 token 的预测，但训练数据和用法有区别。预训练通常从大规模文本中学习规律；SFT 用选定的示范调整任务行为。样本如何组织、混合哪些来源、哪些位置参与损失，也都会影响模型学到什么。

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

The text supplies its own targets without separate token-by-token annotation: this is self-supervised learning. An SFT demonstration specifies how the assistant should respond to the system and user messages. People may write those demonstrations, models may generate them for later filtering, or the dataset may combine several sources. “Supervised” means we select the behavior to learn; it does not require every word to be written by a person.

Both can use cross-entropy to train next-token prediction. Pre-training usually learns patterns from a large text collection; SFT uses selected demonstrations to adjust task behavior. How examples are structured, which data sources are mixed, and which positions contribute loss all affect what the model learns.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. 预测下一位，不代表每一位都要算 loss {#2-label-shift-loss}

先用[对话模板与分词器](../00-foundations/core/tokenization.md)把消息转成序列 $z_1,\ldots,z_T$。输入和标签错开一位：模型根据 $z_{<t}$ 预测 $z_t$。如果只想训练助手的回答（assistant-only loss），再用损失掩码（loss mask）选出需要评分的位置：

$$
\begin{aligned}
\mathcal L_{\text{SFT}}&=\sum_{t=2}^{T}m_t\ell_t,\\
\ell_t&=-\log p_\theta(z_t\mid z_{<t}).
\end{aligned}
$$

其中 $m_t\in\{0,1\}$ 指定哪些目标参与监督。下面是一种简化模板；是否监督角色标记，要根据实际模板决定：

```text
system / user / assistant role marker  → m_t = 0
assistant answer / end-of-message      → m_t = 1
```

这样，系统和用户消息仍然影响后面的预测，只是不要求模型练习把它们复述出来。训练代码常把忽略位置的标签设为 `-100`，让交叉熵跳过这些位置。

多轮数据可以训练每一轮助手回答，也可以只训练最后一轮；有些设置会对整段序列计算损失。选好方案后，要检查模板边界和掩码是否对齐，避免本想忽略的用户文本或填充位置意外参与训练。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. Next-token prediction does not require scoring every token</div>

The [chat template and tokenizer](../00-foundations/core/tokenization.en.md) turn the messages into a sequence $z_1,\ldots,z_T$. Inputs and labels are shifted by one position: the model uses $z_{<t}$ to predict $z_t$. To train only on assistant responses, an assistant-only loss uses a loss mask to select the scored positions:

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

System and user messages remain available as context for later predictions, but the model is not trained to reproduce them. Training code commonly sets ignored labels to `-100` so cross-entropy skips those targets.

Multi-turn training may score every assistant answer or only the final one; some setups score the full sequence. Whichever policy you choose, check that template boundaries and masks agree. Otherwise, user text or padding that you meant to ignore can accidentally become training targets.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. 回答结束，也要有一个信号 {#3-end-token}

如果目标回答只有“我很好。”，没有消息结束标记或序列结束符（end of sequence，EOS），这条样本就没有明确教模型“这里该停了”。模型可能在先前训练中学过停止，但这次示范没有提供该位置的信号。常见做法是让结束标记也参与损失，即 $m_t=1$：

```text
我 → 很好 → 。 → <|im_end|>
```

推理系统会在检测到相应的停止 token 时结束当前消息。因此，训练目标中的结束标记、分词器对应的特殊 token 编号，以及推理引擎的停止配置，三者必须一致。

系统仍然需要长度上限，以免回答无限延长。不过，达到上限是强制截断；学会结束则是模型在回答完成时，主动给结束标记较高的概率。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">3. The end token is behavioral supervision too</div>

If the target contains only “I am fine.” without an end-of-message or end-of-sequence (EOS) token, this example does not explicitly teach where to stop. The model may have learned stopping in earlier training, but this demonstration supplies no target for it. A common choice is to include the end marker in the loss, with $m_t=1$:

```text
I → am fine → . → <|im_end|>
```

The inference system places the corresponding token in its stop set and ends the
message when it appears. Three pieces must agree: the ending marker used as a training
target, its special-token ID in the tokenizer, and the inference engine's stop
configuration.

The system still needs a length limit to prevent unbounded generation. Reaching that limit forces a cutoff; learning to stop means assigning the ending token a high probability when the answer is complete.

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

这里手动错开输出与标签，是因为代码直接调用交叉熵。[Transformers 的因果语言模型损失实现](https://github.com/huggingface/transformers/blob/v5.20.0/src/transformers/loss/loss_utils.py)已经包含这一步（2026-10 对照 v5.20.0 的 `ForCausalLMLoss` 核对）。如果调用的模型内部会处理这种对齐，就不要先把 labels 再移动一遍。

<details id="teacher-forcing" markdown="1">
<summary>两个位置都算对了，为什么还要让模型自己生成一次？</summary>

上面预测 `<EOS>` 时，模型看到的前一个 token 是示范里的 `5`，不是它自己刚刚采样的结果。这叫**教师强制（teacher forcing）**：训练时给每个位置提供标准前文，让它学习下一步。

所以 0.6 表示“前面已经是 `5` 时，接着输出 `<EOS>` 的概率”。按这些概率逐步采样，完整生成 `5 → <EOS>` 的概率是 $0.8\times0.6=0.48$，也等于 $\exp(-2\times\text{平均 loss})$。这只是这条两 token 序列的概率，不是整个任务的成功率；其他说法也可能答对。

如果第一步生成了别的 token，下一步会面对不同的前文。这条示范的 loss 没有直接检查那种情况下模型会怎么接。共享参数可能带来泛化，但不能替代实际生成测试：至少看看完整回答、错误前文后的表现，以及何时停止。

</details>

### 用户消息不算损失，模型怎么学着理解问题？ {#mask-user}

损失掩码选出需要评分的预测，用户问题仍然是模型能读到的上下文。预测 `5` 时，模型需要利用前面的 `2 + 3`；梯度也可以通过注意力和共享参数，传回处理问题的计算过程。**忽略问题对应的目标标签，不会切断问题对学习的影响。**

注意力掩码（attention mask）控制的才是“能看见哪些位置”，例如隐藏未来 token 或填充位置。如果在这里也屏蔽用户消息，模型就失去了回答所需的信息。

### 多轮对话和工具调用，哪些内容参与训练？ {#_5}

| 数据里的内容 | 常见 assistant-only 处理 | 需要确认的地方 |
| --- | --- | --- |
| system / user | 提供上下文，不作为目标 | 这是训练选择，不是所有任务的硬规定 |
| assistant 调用工具 | 可以监督调用名和参数 | 调用格式应由模板准确标出 |
| tool 返回结果 | 通常作为后续回答的上下文 | 不要误把工具消息当成 assistant 回答 |
| assistant 最终回复 | 监督内容与结束标记 | 训练所有轮，还是只训练最后一轮？ |
| padding | 不参与 loss | 若 PAD 与 EOS 共用 ID，要按位置区分 |

[TRL v1.16.0 的说明](https://huggingface.co/docs/trl/v1.16.0/en/sft_trainer#train-on-assistant-messages-only)（2026-10 核对）分别提供 assistant-only 与 completion-only 配置，不能只凭选项名字判断是否等价。

assistant-only 要求聊天模板里有 `{% generation %}` 与 `{% endgeneration %}` 标记。文档说，对 Qwen3 等已知模型族，打开 `assistant_only_loss=True` 时 TRL 会自动补上；其他模型要自己确认模板。检查真实模板返回的 mask，再解码几条最终 labels，看看哪些文本实际参与监督。

长对话被截断后，要看看是不是只剩问题、没剩回答，结束标记是否还在。全部 labels 都是 `-100` 的样本没有有效监督，应在数据检查阶段处理。把多条样本拼进同一条训练序列（packing）时，还要检查样本边界和注意力可见范围；损失掩码本身不能阻止两条对话互相影响。

## 没见过的情况，得单独测 {#_2}

训练集里只有普通退货，测试时却遇到跨境订单和已拆封商品。模型可能根据预训练知识和相似示范处理好，也可能套用错误规则。没有覆盖不等于必然失败，但我们不能拿训练损失下降来证明它已经会处理这些情况。

把不同条件单独留作测试：未见过的表达、组合和例外。这样才知道它学到了任务规则，还是主要记住了熟悉的回答。

## Loss 降了，回答真的更好了吗？ {#token}

“可以退款”和“不可以退款”只差一个字，业务含义却完全相反。交叉熵当然会惩罚这个 token 的错误，而且惩罚可以很大；问题是普通逐 token 加权并不知道哪个位置对业务最重要。

假设目标 token 的预测概率从 0.9 降到 0.1，该项负对数似然从约 0.105 增到 2.303。若报告的是整段 200 个 token 的平均 loss，而其他项不变，平均值只增加约 0.011。不能只看平均 loss，还要检查退款判断是否正确。

可以补关键情形的示范、调整样本或 token 权重，也可以增加任务级检查。RL 是一种选择，不是修复这个问题的唯一办法。

## 该追问的时候，数据有没有教过？ {#_3}

如果所有示范都自信地回答，模型就很少练习澄清、拒答或承认不知道。但这不是 SFT 做不到：我们完全可以加入“请先提供订单号”或“现有资料不足以判断”的示范。

示范还要和模型当时能看到的信息匹配。同一问题，给足证据时应当回答，缺关键信息时应当追问。只加一批统一的拒答模板，又可能让模型在能回答时也回避。

偏好训练或 RL 可以比较不同选择的代价，但也依赖奖励是否合理。没有哪种训练方法本身就能保证诚实。

## 知识是否学到了，怎么检查？ {#sft_4}

SFT 可以把事实学进参数里，但不保证一次出现的事实被可靠记住，也不保证换种问法还能答对。流畅的专业语气更不能代替事实核对。

例如公司退款期限从 30 天改成 14 天。重新微调可以是一种办法；如果规则经常变，检索当前政策通常更方便更新和追溯。两种方案都要测试旧规则是否残留、引用是否正确，以及资料缺失时会不会编造。

选择 SFT、继续预训练还是检索，取决于知识量、更新频率和任务形式，不是“事实一律不能靠 SFT 学”。

[Gekhman 等人的闭卷问答研究（2024）](https://arxiv.org/html/2405.05904v3)发现，新事实的拟合比已有知识慢，在其设置下还伴随更多幻觉。这提示我们分开测“新知识学会多少”和“原来的问题是否答错更多”，而不是推出 SFT 永远学不会新事实。该实验也不能替所有模型、数据和训练方法下结论。

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

没有通用的“loss 低于 0.5 就可以开始”。分词方式、回答长度、参与损失的位置和取平均的方式，都能改变这个数字。更实用的是挑几个训练检查点（checkpoint），在同一批未参与训练的问题上生成回答，记录它们会怎样失败。

比如要训练一个会调用订单工具的客服助手。下面是**示意实验，不是实测结果，也不是上线阈值**：

| 检查项 | checkpoint A | checkpoint B | 怎么理解 |
| --- | --- | --- | --- |
| 100 次调用中可解析的参数 | 72 | 96 | A 还常卡在协议上，先修模板和示范可能更直接 |
| 100 个任务完成数 | 42 | 61 | B 的格式好了，但离完成任务还有差距 |
| 信息不足的 20 题中合理追问数 | 3 | 12 | 不能让总体成功率掩盖这类错误 |

B 的表现有所改善，但要不要改用 RL，还需要回答三个问题：

1. **奖励能不能分清好坏？** 只奖励 JSON 合法，会不会把答非所问也当成功？拿一小批人工检查过的样本，验证奖励函数或验证器的判断。
2. **同一道题的不同尝试，有没有可学习的差别？** 如果方法用组内得分计算相对优势（advantage），同题所有回答得分都相同时，这一项就无法区分它们（见 [DAPO 的动态采样](dapo.md#dynamic-sampling)）。全错时先查题目难度、采样和奖励，不一定要直接回去加做 SFT。
3. **同样预算下，RL 比简单方案多解决了什么？** 从同一检查点比较继续 SFT、生成后筛选再训练，以及小规模 RL。记录质量、长度、成本和原有能力回退，不只看训练奖励。

先确定观察指标和停止条件，再跑小实验。比如成功率没变、输出变长，奖励却上涨，就该检查模型是否利用了评分规则的漏洞，别急着继续加训练步数。

“先 SFT 再 RL”是一条常见路径，不是定律。[DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948v1)就研究了不以前置 SFT 开始的 RL；这里“不做 SFT”不等于从随机权重开始。是否需要这个阶段，要根据你的模型和任务判断。

<details markdown="1">
<summary>想再深一点：DFT 改的是 token 权重，不是 RL 的启动门槛</summary>

[DFT（2025）](https://arxiv.org/html/2508.05629v1#S3.SS3)用当前目标 token 的概率给 CE 加权，并停止权重的梯度：

$$
\ell_{\text{DFT}}=-\operatorname{sg}(p_y)\log p_y.
$$

这里只讨论这个单项形式；实际批次还要指定 mask 和 reduction。它没有新增环境奖励，也没有规定“概率超过某值才进入 RL”。截至 2026-10，[TRL v1.16.0 的 SFTTrainer](https://huggingface.co/docs/trl/v1.16.0/en/sft_trainer) 已内置这个损失，在 `SFTConfig` 里设 `loss_type="dft"` 即可；用现成选项前，同样要确认它的 mask 和 reduction 与你的设定一致。

一个容易写错的地方是忘记 `detach()`。取二分类目标概率 $p=0.01$，对正类 logit 求导：普通 CE 是 −0.99；带停止梯度权重后是 −0.0099。若直接求 $-p\log p$ 的完整导数，却约为 +0.0357，梯度下降反而会降低这个低概率目标的分数。

这是链式法则的区别，不是数值精度问题。[配套测试](../site/tests/test_loss_contracts.py)检查了两种梯度方向。概率加权也可能减弱困难但重要的样本，因此应在自己的保留集上比较，不把某篇论文的收益当成通用结论。

</details>

## 设计 SFT 数据时检查什么 {#sft_6}

1. 我的示范数据覆盖了哪些情况？没覆盖的那些，模型会怎么表现——我测过吗？
2. 我关心的正确性，是"整体像不像"还是"某几个关键 token 对不对"？如果是后者，除了平均 loss，还应单独测关键判断。
3. 我的 SFT 集里有没有符合情境的拒答或澄清示范？缺少关键信息时，模型会怎样处理？
4. 要教的知识，基座里是否已有？换一种问法，模型还能给出正确事实和依据吗，还是只学会了自信的语气？
5. 这个任务真的需要 RL 吗？有可靠示范时，先把 SFT 当作基线，再比较同样预算下的任务质量，而不是凭格式固定就认定哪种方法更划算。

## 继续阅读 {#_4}

- [交叉熵怎么实现](../00-foundations/pytorch/cross-entropy.md)：稳定计算、soft target、mask 和分母
- [RLHF 的三个阶段](rlhf/three-stages.md)：SFT 之后那两个阶段在做什么
- [PPO、GRPO 与 DPO](after-ppo.md)：RL 那条线的算法谱系
- [数据与反馈](../01-data-and-feedback/)：示范和偏好数据本身的质量问题

## 快速学习：SFT 教会模型什么 {#sft_1}

<details class="interview" markdown="1">
<summary>Assistant-only CE、行为模仿与能力上限</summary>

**快速记忆**：SFT 的目标是在给定前文下提高示范 token 的似然；对话中常只给 assistant tokens 计 loss。目标不保证每条示范的概率都在每步更新后上涨，更不保证未见过的任务已经会做。

**面试回答**

> 在常见的 assistant-only 设置里，先把对话按模板拼起来，让模型根据 system 和 user 的内容学习生成 assistant 回答及结束 token。示范可以教格式、语气、工具调用和解题方法。数据覆盖和质量很重要；根据标准前文计算的逐 token loss 不直接衡量任务成败，仍要检查模型自己生成的完整回答。

<details markdown="1">
<summary><b>深挖</b>：为什么低 token CE 不等于完整回答更好？</summary>

CE 会惩罚每个目标 token 的错误，但长回答的平均值可能掩盖决定成败的少数位置。需要单独检查这些判断。任务测试、针对性示范、加权以及偏好或奖励信号，都可以从不同角度突出这些位置。

</details>
</details>
