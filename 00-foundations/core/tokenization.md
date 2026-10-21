# Tokenization：从文本到 ID

**中文** · [English](tokenization.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

同样一句话，换一个 tokenizer，得到的 ID 序列可能就不一样长。比如把 `playing` 切成 `play` 和 `ing`，就需要两个编号，而不是给整个单词一个编号。这里的切分只是示意，但它说明了一个实际问题：分词方式会影响序列长度、计算量，也会影响没见过的词怎样表示。下面沿着文本输入的过程，看看分词、查表和对话模板各做了什么。

## 模型接收的是 Token ID {#token-id}

文本不能直接送进神经网络。Tokenizer 先把字符串切成词表里的 token，再映射成整数 ID；模型通过 embedding table 查出对应向量，后续的神经网络计算就在向量上进行。ID 用来查表，并不是模型内部始终使用的表示。

## 第一步：文本怎样被分成 Token {#token}

下面的切分和 ID 仅作示意，不对应某个真实 tokenizer：

```text
"unbelievable!" → ["un", "believ", "able", "!"] → [431, 9821, 612, 5]
```

模型用 embedding table 把每个 ID 查成向量：

$$x_t = E[\text{token\_id}_t], \qquad E \in \mathbb{R}^{|V| \times d}$$

$|V|$ 是词表大小，$d$ 是模型维度。Tokenizer 决定序列有多长，embedding 决定每个离散符号从哪个连续向量开始。

## 第二步：为什么不直接按单词切分 {#_1}

按单词切分很直观，但很难把所有可能出现的词都列出来：新名字、拼写变化、代码、emoji，以及不同语言，都需要处理。反过来，如果每个字符都是一个 token，序列又会变长；没收进基础字符表的字符，仍然可能是未知输入。

Subword tokenizer 在两者之间折中：高频片段保留为整体，低频词拆成更小单元。

| 粒度 | 优点 | 代价 |
| --- | --- | --- |
| word | 序列短、语义直观 | 词表爆炸，未知词严重 |
| character | 能组合已收录字符，不必列出所有词 | 未收录字符仍要处理；序列可能更长 |
| byte | 完整字节表可以表示任意 UTF-8 文本 | 基础单位不等于一个字符，序列可能更长 |
| subword | 词表与长度较平衡 | 切分依赖语料，边界不一定符合人类直觉 |

## 第三步：BPE 怎样合并常见片段 {#bpe}

Byte Pair Encoding 不懂词根，也不知道语法。它只是反复问：**哪两个相邻符号最常一起出现？** 然后把它们粘起来。

```text
l o w </w>
l o w e r </w>

最高频 pair: (l, o)  →  lo
下一轮:      (lo, w) →  low
```

这里假设两个词各出现一次；首轮 `(l, o)` 和 `(o, w)` 都出现两次，示例在并列时先选 `(l, o)`。训练阶段学到一份**有顺序的 merge rules**；编码阶段按同样顺序应用。它没有在查英语词根，只是在压缩当前语料里反复出现的字符串模式。[BPE 原论文 §3.2](https://aclanthology.org/P16-1162/)

想自己算两轮合并，或分清 BPE、WordPiece、Unigram 和 byte fallback，可以接着读[分词算法详解](../deep-dives/tokenizer-algorithms.md)。这页先把输入流程走完。

<details markdown="1">
<summary><b>进阶</b>：为什么 token 边界会影响模型行为</summary>

同一个概念如果在一种语言里是 1 个 token、另一种语言里是 5 个 token，后者会占用更多上下文、更多 attention 计算，也需要更多预测步骤。数字、空格和代码的切分也会改变算术与复制任务的难度。

因此 tokenizer 不是无关紧要的预处理；它定义了模型预测问题的基本单位。

</details>

## 四个容易混淆的对象 {#_2}

1. **Vocabulary**：token 与 ID 的静态映射。
2. **Merge rules / model**：怎样把原始符号组合成 token。
3. **Normalizer / pre-tokenizer**：大小写、Unicode、空格等进入模型前怎样处理。
4. **Special tokens**：BOS、EOS、PAD，以及对话模板里的角色边界。

`decode(encode(text))` 通常应该复原文本，但 normalization 可能使它不是逐字节可逆。PAD 只负责批处理对齐，不应该被模型当作内容；EOS 则是真正的生成终止信号。

## 对话怎样变成模型能读的一串 ID {#id}

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>下面三张卡默认中文；点 <strong>English ↻</strong> 可在当前位置查看等价英文。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. Chat Template：把角色结构序列化 {#1-chat-template}

应用层拿到的对话可能是：

```json
[
  {"role": "system", "content": "You are a helpful assistant"},
  {"role": "user", "content": "你好吗？"}
]
```

Transformer 不会直接处理这份消息列表。Chat template 会先把消息排列成模型训练时使用的格式。以 ChatML 风格为例：

```text
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
你好吗？<|im_end|>
<|im_start|>assistant
```

推理时最后只有 assistant 起始标记，没有答案；模型从这里继续生成。不同模型可能使用
`[INST]...[/INST]`、header tokens 或其他格式，因此 **chat template 必须与 checkpoint
及 tokenizer 匹配**，不能把一个模型的模板随意套给另一个模型。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. Chat templates serialize role structure</div>

An application may represent a conversation as

```json
[
  {"role": "system", "content": "You are a helpful assistant"},
  {"role": "user", "content": "How are you?"}
]
```

The Transformer does not process this message list directly. A chat template first
serializes the messages into the format seen during training. A ChatML-style rendering
might be

```text
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
```

At inference time the final assistant marker has no answer after it; generation begins
there. Other checkpoints may use `[INST]...[/INST]`, header tokens, or entirely
different formats. The **chat template must match the checkpoint and tokenizer** rather
than being copied across model families.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. 特殊 Token 仍然只是词表里的符号 {#2-token}

从模型角度看，system、user、assistant 和消息边界最终都只是 token IDs。若
`<|im_start|>` 被 tokenizer 注册成 special token，它通常整体映射到一个 ID；若没有
注册，就可能被拆成多个普通片段。角色名本身是一个还是多个 token，也取决于具体词表。

模型架构里没有一条写死的规则说“system 权限最高”。它是在训练数据中反复看到角色
标记、内容和目标回答的组合后，学会这些符号通常怎样影响后续 token。

因此不要手抄截图里的数字 ID。特殊 token 的字符串、ID、终止集合和 embedding 都属于
具体 tokenizer/checkpoint 的协议，换模型后必须重新读取配置。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. Special tokens are still vocabulary symbols</div>

To the model, system, user, assistant, and message boundaries eventually become token
IDs. If `<|im_start|>` is registered as a special token, it normally maps to one ID;
otherwise it may be split into ordinary pieces. Whether a role name occupies one token
or several also depends on the vocabulary.

The architecture contains no hard-coded rule that says “system has highest priority.”
The model learns the usual effect of these symbols from repeated combinations of role
markers, message content, and target responses in its training data.

Do not copy numeric IDs from a screenshot. Special-token strings, IDs, stop sets, and
embeddings are part of a specific tokenizer/checkpoint protocol and must be loaded again
when the model changes.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. 完整边界：Messages → Template → IDs → Embeddings {#3-messages-template-ids-embeddings}

整个输入管线是：

$$
\boxed{
\text{structured messages}
\xrightarrow{\text{chat template}}
\text{serialized token text}
\xrightarrow{\text{tokenizer}}
\text{token IDs}
\xrightarrow{\text{embedding table}}
\text{vectors}
}
$$

很多库把“应用模板”和“tokenize”封装进同一个函数，但概念边界仍然重要：template
决定角色和消息边界怎样排列；tokenizer 决定这些字符串切成哪些词表单元；embedding
table 才把整数变成模型维度里的向量。

Chat template 没有改变 attention 公式，它只是给普通序列加了可学习的语法结构。

实际调用时也要避免重复添加特殊 token：在 Transformers v5.20.0 中（2026-10 核对），若先用 `apply_chat_template(tokenize=False)` 得到文本，再单独 tokenize，应设置 `add_special_tokens=False`；直接用 `tokenize=True` 可以合并这两步，v5 的 `return_dict` 默认是 `True`，返回带 `input_ids` 的字典，不是一串 id。[官方说明](https://huggingface.co/docs/transformers/v5.20.0/chat_templating)

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">3. The full boundary: messages → template → IDs → embeddings</div>

The complete input pipeline is

$$
\boxed{
\text{structured messages}
\xrightarrow{\text{chat template}}
\text{serialized token text}
\xrightarrow{\text{tokenizer}}
\text{token IDs}
\xrightarrow{\text{embedding table}}
\text{vectors}
}
$$

Libraries often wrap template rendering and tokenization in one function, but their
conceptual responsibilities remain distinct. The template determines the ordering of
roles and message boundaries; the tokenizer selects vocabulary units; the embedding
table turns integers into model-width vectors.

A chat template does not alter the attention equation. It gives an ordinary token
sequence a learnable grammatical structure.

Avoid duplicating special tokens in actual code. In Transformers v5.20.0 (checked October 2026), if you render with `apply_chat_template(tokenize=False)` and tokenize separately, use `add_special_tokens=False`; `tokenize=True` combines the two steps and, because `return_dict` now defaults to `True` in v5, returns a dict with `input_ids` rather than a bare list of ids. [Official guidance](https://huggingface.co/docs/transformers/v5.20.0/chat_templating)

</div>
</section>

## 把输出形状接到神经网络 {#_3}

假设 batch 中有 4 条文本，padding 后长度 12，模型维度 768：

```text
token_ids       (B, T)    = (4, 12)
attention_mask  (B, T)    = (4, 12)
embeddings      (B, T, d) = (4, 12, 768)
```

Tokenizer 结束于 `(B, T)`；神经网络从 `(B, T, d)` 开始。

## 实验：比较不同分词结果 {#_4}

运行 [`../code/tokenizer_from_scratch.py`](../code/tokenizer_from_scratch.py)。它只用 Python 标准库训练一个迷你 BPE，不需要额外依赖。可以修改训练语料，直接观察同一句话为什么会在另一份语料上被切成完全不同的 token。

这个教学实现会合并空白，未收录的字符会编码为 `<unk>`，没有 byte fallback；`decode_tokens` 还原的是 token 字符串，不是对 ID 的无损往返。可以试试连续空格和一个训练语料里没有的字符，区分这些边界。

## 自检 {#_5}

<div class="taste-check">
  <strong>合上页面前，试着不用术语回答：</strong>
  <ol>
    <li>新名字能怎样拆成已知片段？若连其中的字符都未收录，还需要什么机制？</li>
    <li>为什么同一句话换一个 tokenizer，推理成本也可能变化？</li>
    <li><code>(B, T)</code> 是在哪一步变成 <code>(B, T, d)</code> 的？</li>
  </ol>
</div>

## 继续阅读 {#_6}

查表后，每个位置都有了向量，但还没有结合其他位置的信息。下一页看 [RNN 与 LSTM](recurrent-models.md)：从左往右处理序列时，怎样把前面读到的信息传下去？

## 快速学习：Tokenizer 在模型边界做什么 {#tokenizer}

<details class="interview" markdown="1">
<summary>从 text 到 IDs 的标准回答与一个关键误区</summary>

**快速记忆**：Chat Template 先按约定格式排列不同角色的消息，Tokenizer 再把文本切成 token，转成整数 ID。Special tokens 也是词表中的 token，只是用来标记角色、消息边界等特殊位置。

**面试回答**

> 完整路径是 messages 经 chat template 变成带角色边界的文本，再由 tokenizer 变成 token IDs，最后通过 embedding lookup 得到连续向量。Transformer 从未直接看到字符串，也没有在架构中写死 system、user 或 assistant。

<details markdown="1">
<summary><b>深挖</b>：为什么 tokenizer 与 chat template 不能跨模型乱换？</summary>

特殊字符串是否是单独 token、对应哪个 ID、assistant 起止边界怎样写，都是模型训练分布的一部分。模板与词表不匹配会把结构标记拆碎或映射到错误 ID；即使张量形状没报错，输入的格式也可能已经和训练时不同了。

</details>
</details>
