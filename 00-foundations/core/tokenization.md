# Tokenization：从文本到 ID

**中文** · [English](tokenization.en.md)

> 阅读时间：约 7 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>把无限变化的字符串，装进一个有限词表</strong></div>
  <div><span>前置知识</span><strong>原始文本 · vocabulary · merge rules</strong></div>
  <div><span>最后端上桌</span><strong>token IDs · attention mask · embeddings</strong></div>
  <div><span>常见错误</span><strong>把 tokenizer 当成无关紧要的预处理</strong></div>
</div>

## 模型其实没见过文字

我们天天说模型“读”了一句话，但它其实从没看见过文字。Tokenizer 先把字符串切成有限词表里的 token，再映射成整数 ID；模型真正收到的，从头到尾都只是这些数字。

## 第一步：一句话怎样被切碎

```text
"unbelievable!" → ["un", "believ", "able", "!"] → [431, 9821, 612, 5]
```

模型用 embedding table 把每个 ID 查成向量：

$$x_t = E[\text{token\_id}_t], \qquad E \in \mathbb{R}^{|V| \times d}$$

$|V|$ 是词表大小，$d$ 是模型维度。Tokenizer 决定序列有多长，embedding 决定每个离散符号从哪个连续向量开始。

## 第二步：为什么不干脆一个词一个 token

这个直觉很好，但现实里的词表没有尽头：新名字、拼写变化、代码、emoji 和不同语言永远列不完。反过来，如果每个字符都是一个 token，未知词是没了，序列却会长得吓人。

Subword tokenizer 在两者之间折中：高频片段保留为整体，低频词拆成更小单元。

| 粒度 | 优点 | 代价 |
| --- | --- | --- |
| word | 序列短、语义直观 | 词表爆炸，未知词严重 |
| character / byte | 几乎无未知输入 | 序列长，学习局部组合更难 |
| subword | 词表与长度较平衡 | 切分依赖语料，边界不一定符合人类直觉 |

## 第三步：BPE 怎样把常见碎片粘回来

Byte Pair Encoding 不懂词根，也不知道语法。它只是反复问：**哪两个相邻符号最常一起出现？** 然后把它们粘起来。

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

## 四个很容易搅在一起的东西

1. **Vocabulary**：token 与 ID 的静态映射。
2. **Merge rules / model**：怎样把原始符号组合成 token。
3. **Normalizer / pre-tokenizer**：大小写、Unicode、空格等进入模型前怎样处理。
4. **Special tokens**：BOS、EOS、PAD，以及对话模板里的角色边界。

`decode(encode(text))` 通常应该复原文本，但 normalization 可能使它不是逐字节可逆。PAD 只负责批处理对齐，不应该被模型当作内容；EOS 则是真正的生成终止信号。

## 把输出形状接到神经网络

假设 batch 中有 4 条文本，padding 后长度 12，模型维度 768：

```text
token_ids       (B, T)    = (4, 12)
attention_mask  (B, T)    = (4, 12)
embeddings      (B, T, d) = (4, 12, 768)
```

Tokenizer 结束于 `(B, T)`；神经网络从 `(B, T, d)` 开始。

## 动手：自己切一次看看

运行 [`../code/tokenizer_from_scratch.py`](../code/tokenizer_from_scratch.py)。它只用 Python 标准库训练一个迷你 BPE，没有任何神秘依赖。最好顺手改掉训练语料：你会很直观地看到，同一句话为什么会被另一锅数据切成完全不同的样子。

## 自检

<div class="taste-check">
  <strong>合上页面前，试着不用术语回答：</strong>
  <ol>
    <li>为什么一个新名字不会让 subword tokenizer 直接失效？</li>
    <li>为什么同一句话换一个 tokenizer，推理成本也可能变化？</li>
    <li><code>(B, T)</code> 是在哪一步变成 <code>(B, T, d)</code> 的？</li>
  </ol>
</div>

## 继续读

现在文字终于变成了向量，但每个位置还互不认识。下一页看 [RNN 与 LSTM](recurrent-models.md)：如果只能从左往右读，过去到底该装在哪里？
