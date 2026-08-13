# Tokenization：从文本到 ID

**中文** · [English](tokenization.en.md)

> 阅读时间：约 7 分钟 · 难度：必修 · 最近审阅：2026-08

## 先用一句话讲清楚

Tokenizer 把字符串切成有限词表里的 token，再映射成整数 ID；模型从头到尾只看到这些 ID，**从没直接看过“文字”**。

## 最小数据流

```text
"unbelievable!" → ["un", "believ", "able", "!"] → [431, 9821, 612, 5]
```

模型用 embedding table 把每个 ID 查成向量：

$$x_t = E[	ext{token\_id}_t], \qquad E \in \mathbb{R}^{|V| \times d}$$

$|V|$ 是词表大小，$d$ 是模型维度。Tokenizer 决定序列有多长，embedding 决定每个离散符号从哪个连续向量开始。

## 为什么不能直接按词切

按词切会遇到开放词表：新名字、拼写变化、代码、emoji 和不同语言永远列不完。按字符切没有未知词，但序列太长，而且一个字符往往没有稳定语义。

Subword tokenizer 在两者之间折中：高频片段保留为整体，低频词拆成更小单元。

| 粒度 | 优点 | 代价 |
| --- | --- | --- |
| word | 序列短、语义直观 | 词表爆炸，未知词严重 |
| character / byte | 几乎无未知输入 | 序列长，学习局部组合更难 |
| subword | 词表与长度较平衡 | 切分依赖语料，边界不一定符合人类直觉 |

## BPE 的核心动作

Byte Pair Encoding 反复合并训练语料里最常见的相邻符号对：

```text
l o w </w>
l o w e r </w>

最高频 pair: (l, o)  →  lo
下一轮:      (lo, w) →  low
```

训练阶段学到一份**有顺序的 merge rules**；编码阶段按同样顺序应用。它不是在查英语词根，而是在压缩当前语料里反复出现的字符串模式。

<details markdown="1">
<summary><b>进阶</b>：为什么 token 边界会影响模型行为</summary>

同一个概念如果在一种语言里是 1 个 token、另一种语言里是 5 个 token，后者会占用更多上下文、更多 attention 计算，也需要更多预测步骤。数字、空格和代码的切分也会改变算术与复制任务的难度。

因此 tokenizer 不是无关紧要的预处理；它定义了模型预测问题的基本单位。

</details>

## 四个必须分清的对象

1. **Vocabulary**：token 与 ID 的静态映射。
2. **Merge rules / model**：怎样把原始符号组合成 token。
3. **Normalizer / pre-tokenizer**：大小写、Unicode、空格等进入模型前怎样处理。
4. **Special tokens**：BOS、EOS、PAD，以及对话模板里的角色边界。

`decode(encode(text))` 通常应该复原文本，但 normalization 可能使它不是逐字节可逆。PAD 只负责批处理对齐，不应该被模型当作内容；EOS 则是真正的生成终止信号。

## 形状追踪

假设 batch 中有 4 条文本，padding 后长度 12，模型维度 768：

```text
token_ids       (B, T)    = (4, 12)
attention_mask  (B, T)    = (4, 12)
embeddings      (B, T, d) = (4, 12, 768)
```

Tokenizer 结束于 `(B, T)`；神经网络从 `(B, T, d)` 开始。

## 实验

运行 [`../code/tokenizer_from_scratch.py`](../code/tokenizer_from_scratch.py)，它只用 Python 标准库训练一个迷你 BPE，并验证 encode/decode。然后改训练语料，观察 merge rules 和序列长度怎样变化。

## 下一步

Token 已经变成向量，但每个位置仍彼此独立。下一页看 [RNN 与 LSTM](recurrent-models.md) 怎样让第 $t$ 个位置携带前文状态。
