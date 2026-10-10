# BERT：一句话怎样变成一组表示？

**中文** · [English](bert.en.md)

“水很冷”和“他对人很冷”，用了同一个“冷”，意思却不同。区别就在前后文里。BERT 要学的也是这件事：同一个 token 放进不同句子，应该得到不同的表示，而不只是查出一个固定的词向量。

具体来说，**输入有多少个 token，BERT 就输出多少个向量**。每个向量对应一个位置，但经过编码器（encoder）后，也能包含其他位置的信息。这些向量还不是答案。接上任务输出层（task head），再用对应的标签训练，才可以给评论分类、给词打标签，或者从文章里找答案。

这些表示是怎么学出来的？BERT 的一个预训练任务，是把原文里的部分 token 藏起来，再根据剩下的文字预测它们。下面用 `the tea is cold`（茶是凉的）走一遍：先藏掉 `tea`，看看输入怎么变成向量，又怎么得到预测。

<div class="bert-story" data-bert-story data-lang="zh" id="bert-walkthrough" markdown="1">

**准备输入 → 转成向量 → 双向 encoder → 任务输出**

做 MLM 时，输入是 `[CLS] the [MASK] is cold [SEP]`。Encoder 给每个位置生成带上下文的向量，再用选中位置的输出预测原词 `tea`。左边的 `the`、右边的 `is cold` 都可以提供线索。换成文本分类，则提供完整句子，接上读取 `[CLS]` 的分类 head 做微调。复用的是 encoder，不是 MLM 的词表预测结果。

下面的静态图包含原始预训练的两个任务：MLM 和 NSP。开启 JavaScript 可以逐步查看计算路径；图解只说明结构，不运行模型。

![原始 BERT 的预训练：扰动后的 token 进入同一个双向 encoder；MLM 从选中位置预测原 token，NSP 从 CLS 判断文本对是否相邻。](../assets/bert-flow.svg)

</div>

不必第一次就把全部细节读完：

- **先弄懂它在做什么**：看[输入和向量](#input)、[上下文怎样参与计算](#context-calculation)，再读[评论分类](#classify-example)。想知道它怎么回答问题，接着看[选出答案片段](#answer-span)。
- **准备写代码或查问题**：重点看[三种 mask](#three-masks)、[loss](#loss)和 [Python 检查](#small-lab)。推导与代码可以按需展开。

本文讲的是 2018 年提出的原始 BERT；后来改过的做法会另外说明。

## 1. 为什么要先藏掉一个词？ {#context}

假如原句完整地放在面前，我们却让模型回答“第 2 个位置是什么词”，它抄一个 `tea` 就行了，没必要学会读句子。把 `tea` 换成 `[MASK]` 后，这条捷径就没了：左边的 `the` 提供语法线索，右边的 `is cold` 提供更多语境，模型得把它们用起来。

不过，`the [MASK] is cold` 并不能唯一确定答案。填 `tea` 合理，填 `coffee` 也合理。训练仍然用原文里的 `tea` 作目标，通过交叉熵提高它的概率。模型是从大量例子里学习分布，不是在做每道都只有一个合理答案的填空题。

这也回答了另一个问题：没有人工标注，训练答案从哪儿来？**原文自己提供答案。** 我们先记住 `tea`，再改输入，就得到了一条训练样本。这类从数据自身构造监督的做法，通常叫自监督学习（self-supervised learning）。

到了实际任务，仍然需要任务自己的标签。会恢复原词，不代表已经知道一条评论该分成好评还是差评。BERT 的思路是先用大量文本学表示，再用较少的任务数据去适配，而不是每次从随机参数开始。[BERT 原论文 §3](https://arxiv.org/html/1810.04805v2#S3)

“双向”说的是每层 attention 都能同时利用左边和右边的输入，不是先从左读一遍、再从右读一遍，最后把两个结果拼起来。为什么允许这么读，还得看任务：

| 现在要做什么 | 可用信息 | 模型需要学什么 |
| --- | --- | --- |
| 给已有的一段话分类 | 整段输入已经给定 | 综合前后内容做判断 |
| 从文章里找答案 | 问题和文章已经给定 | 定位答案所在的位置 |
| 继续生成下一句 | 后面的文字尚不存在 | 只根据已有前缀生成 |

前两种任务的完整输入已经在手上，当然可以一起读；续写时，后半句还没生成，就只能根据已有前缀预测。这个区别比“encoder 和 decoder 哪个更好”更有用。BERT 原论文主要研究的是前一类文本理解任务。

<details markdown="1">
<summary>那 BERT、LSTM 和 Decoder-only 到底差在哪儿？</summary>

这几个名字不在同一个层面。LSTM 是递归单元；BERT 是 encoder 架构加一套预训练方法；Decoder-only 说的是架构，常见训练目标是自回归预测。

| 比较哪一件事 | LSTM / BiLSTM | 原始 BERT | 常见自回归 Decoder-only |
| --- | --- | --- | --- |
| 怎样交换信息 | 沿序列更新状态；BiLSTM 分两个方向 | 双向 self-attention | 带 causal mask 的 self-attention |
| 哪些计算可并行 | 可并行样本，但普通递归更新依赖同方向的前一步 | 一层内的全部位置可一起算，层之间仍有依赖 | 训练中已给定目标文本时可一起算；常规生成仍逐 token |
| 原生训练目标 | LSTM 本身不规定 loss | MLM + NSP | Next-token cross-entropy |
| 读取右侧信息 | BiLSTM 可以，单向 LSTM 不可以 | 可读取扰动输入的右侧 | 生成时不可读取未来输出 |

比如分类“服务不算快，但味道很好”：BiLSTM 和 BERT 都能利用后半句，不是只有 attention 才能双向。区别在于计算路径——普通 LSTM 要沿状态链传递信息，full self-attention 允许远处位置在一层内直接交互，但也增加了随长度增长的成对计算。并行更方便，不等于任何长度和硬件上都更快。[Transformer §4](https://arxiv.org/html/1706.03762v7#S4)

再换成续写“服务不算快，但”：此时后半句根本不存在。能用完整句子做分类，和能从前缀生成，是两种任务。可以让 LSTM 训练 next-token prediction，也可以让 encoder 做分类；**不要把训练目标当成某个基础组件自带的属性**。递归计算细节见 [RNN / LSTM](recurrent-models.md)。

在当时，预训练和迁移已经有人在做。2018 年的 [ULMFiT](https://aclanthology.org/P18-1031/) 先训练 LSTM 语言模型，再适配到目标语料和分类任务；同年的 [ELMo](https://aclanthology.org/N18-1202/) 用双向 LSTM 语言模型提供上下文特征。把它们放在一起看，更容易理解 BERT 改了什么：深层双向 attention、适合它的预训练目标，以及同一个 encoder 适配多种任务的方式。

“能读左右文”也不等于“能读任意长的文档”。原始 BERT 的输入上限为 512 个 token，完整 attention 的计算还会随长度平方增长。比如关键信息在被截掉的第 600 个 token，双向 attention 也看不到它。处理长文还得考虑分块、检索或经过验证的长上下文模型。[BERT 附录 A.2](https://arxiv.org/html/1810.04805v2#A1.SS2)

</details>

## 2. 一条输入怎样走过模型？ {#input}

### 先分词，再补上边界

原始 BERT 使用 WordPiece，把文字拆成词表里已有的 token，再转成整数 ID。一个词可能对应多个 subword，所以“句子有 10 个词”不一定等于“模型看到 10 个位置”。分词的细节可以回看 [Tokenization](tokenization.md)。

一段文本会整理成 `[CLS] A [SEP]`，文本对则是 `[CLS] A [SEP] B [SEP]`。`[CLS]` 放在最前面，后面可以用它的表示做分类；`[SEP]` 标出一段文字的结束。它们也有自己的 ID 和 embedding，会和普通 token 一起进入模型。

### 三个 embedding，各补一种信息

光知道当前位置是哪个 token 还不够，还要知道它在哪儿、属于哪段文本：

| 向量 | 回答什么问题 | 在例子里是什么 |
| --- | --- | --- |
| Token embedding | 这里是什么 token？ | 查到 `tea` 的向量；遮盖后则查 `[MASK]` 的向量 |
| Position embedding | 它在第几个位置？ | `[CLS]` 从 0 开始，`tea` 在位置 2 |
| Segment embedding | 它属于哪段文本？ | 单段用 A；文本对用 A、B 区分两段 |

三个向量维度相同，直接逐维相加，再经过 LayerNorm 和 dropout，进入 encoder。比如只看 3 个维度，`[1, 0, 2] + [0, 1, 1] + [1, 1, 0] = [2, 2, 3]`，结果仍是 3 维。这只是说明相加方式的手造数字；BERT-Base 的 hidden size 是 768。原始 BERT 的位置向量是学出来的绝对位置 embedding，不是 RoPE。[原文 §3](https://arxiv.org/html/1810.04805v2#S3)、[Embedding 实现](https://github.com/google-research/bert/blob/master/modeling.py)

例如输入“明天有雨”和“记得带伞”，第一段及其 `[CLS]`、`[SEP]` 用 segment A，第二段及末尾 `[SEP]` 用 segment B。位置编号沿整条输入递增，不在 B 的开头重新从 0 算。两段虽然有边界，仍然能通过 self-attention 互相读取；这里没有另外接一个 decoder 的 cross-attention 子层。

### 输入的形状没变，表示却在变

同一个 token 在 embedding 表里查到的是同一行。进入 encoder 后，它的表示会结合当前句子重新计算。例如把 `the tea is cold` 换成 `the weather is cold`，`cold` 的 token embedding 没变，但它接收到的上下文变了，最后的向量也可以不同。**查表得到的是起点，结合上下文算出的 hidden state 才是后面要用的表示。**

| 计算走到哪儿 | 教学例子：batch 为 2、补齐到 7 个位置、hidden size 为 8 |
| --- | --- |
| Token IDs | `[2, 7]`，整数索引 |
| Embedding 相加 | `[2, 7, 8]`，不因相加变成 24 维 |
| 每层 encoder | 仍是 `[2, 7, 8]`，不同位置交换信息，再做逐位置 FFN |
| 最后一层 hidden states | 每个位置都有一个 8 维表示，尚不是分类概率 |
| 任务输出层 | 按任务变成词表分数、类别分数或答案位置分数 |

表里用小尺寸方便数清楚。BERT-Base 实际会输出 `[batch, length, 768]`，并不会自动把整句话压成一个向量；需要句子向量时，还要选择如何读取或汇总这些位置。原始 block 的 residual、LayerNorm 与 FFN 可以沿着[原版 Transformer](vanilla-transformer.md)继续看。

### “结合上下文”，具体算了什么？ {#context-calculation}

先只看一个注意力头（attention head）、一个输出位置。它会给可见位置分别算一个权重，再把这些位置负责传递内容的 value 向量按权重加起来。这里的权重回答的是“这次从每个位置读多少信息”，不是某个词永远有多重要。

用 3 个位置、每个 value 只有 2 维的小例子，假设 softmax 后的权重已经算好：

<figure class="worked-update" lang="zh-CN" id="context-mixture">
  <figcaption>一次加权汇总 · 数字为教学设定，不是 BERT 的实测权重。</figcaption>
  <ol>
    <li><small>位置 A · 权重 0.2</small><strong>[1, 0] × 0.2</strong><span>传入 [0.2, 0]</span></li>
    <li><small>位置 B · 权重 0.3</small><strong>[0, 2] × 0.3</strong><span>传入 [0, 0.6]</span></li>
    <li><small>位置 C · 权重 0.5</small><strong>[2, 1] × 0.5</strong><span>传入 [1, 0.5]</span></li>
  </ol>
</figure>

把三份相加，这个注意力头在当前位置的输出就是 `[1.2, 1.1]`。先固定权重，只把位置 A 的 value 从 `[1, 0]` 改成 `[3, 0]`，输出就变成 `[1.6, 1.1]`。**别的位置变了，当前位置收到的信息也跟着变。** 真实模型换了上下文，通常不止 value 会变，用来计算权重的向量也可能一起变。

那权重从哪儿来？当前位置用自己的查询向量（query），和各位置的键向量（key）算相似度，再经过缩放和 softmax。多个注意力头分别做这件事，结果再拼接、投影。因此上面算出的 2 维向量只是一个头的结果，还不是整层输出；后面还有残差、归一化和 FFN。机制见 [Transformer §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)。

<details markdown="1">
<summary>展开一层：attention、残差和 FFN 按什么顺序？</summary>

原始 BERT 是 post-LN。设这一层输入为 $H$，先做双向 multi-head attention，再做逐位置 FFN。为方便分步看，下面用 $D$ 表示 dropout，$\operatorname{LN}$ 表示 LayerNorm：

$$
\begin{aligned}
A&=D(\operatorname{MHA}(H)),\\
U&=\operatorname{LN}(H+A).
\end{aligned}
$$

$$
\begin{aligned}
F&=D(\operatorname{FFN}(U)),\\
H'&=\operatorname{LN}(U+F).
\end{aligned}
$$

这里的 MHA 包含各头拼接后的输出投影；FFN 是 dense → GELU → dense。每个子层都有自己的残差相加与 LayerNorm，不是整个 block 最后只加一次。原版 Base 堆 12 层，Large 堆 24 层；token / segment / position embedding 只在进入这串 block 前相加，不是每层重新加一遍。结构可以对照[原始实现的 `transformer_model`](https://github.com/google-research/bert/blob/master/modeling.py)。这也不同于许多现代 decoder 的 pre-norm、RMSNorm 与 SwiGLU，不能直接拿后者的 block 图代替。

上面的加权汇总也可以直接算，不需要下载模型。这里指定的是缩放后的 attention scores，取 `log(2)`、`log(3)`、`log(5)`，softmax 就会得到 `0.2`、`0.3`、`0.5`：

```python
import math

scores = [math.log(2), math.log(3), math.log(5)]
exp_scores = [math.exp(score - max(scores)) for score in scores]
weights = [score / sum(exp_scores) for score in exp_scores]
values = [[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]]

def mix(current_values):
    return [sum(weight * value for weight, value in zip(weights, column))
            for column in zip(*current_values)]

before = mix(values)
values[0] = [3.0, 0.0]
after = mix(values)
assert all(math.isclose(actual, expected)
           for actual, expected in zip(before, [1.2, 1.1]))
assert all(math.isclose(actual, expected)
           for actual, expected in zip(after, [1.6, 1.1]))
print([round(value, 2) for value in before])
print([round(value, 2) for value in after])
```

这段代码只验证一个 head 的加权求和，不包含 Q/K/V 投影、dropout 或训练。

</details>

## 3. 三种“mask”，其实管三件事 {#three-masks}

代码里经常同时出现 `[MASK]`、`attention_mask` 和忽略标签，初看很容易混在一起。可以把它们分别理解为：**输入改了哪里、attention 能看哪里、最后给哪里算分。** 它们不是同一张表。

### 输入遮盖：改了模型看到的内容

MLM（masked language modeling）先选预测位置，再决定如何扰动。原论文使用约 15% 的预测比例；在选中的位置中，80% 换成 `[MASK]`，10% 换随机 token，10% 保持原样。**无论替换成什么，目标仍是原 token。** `[CLS]`、`[SEP]` 和 padding 不作为普通预测位置。[原始数据构造代码](https://github.com/google-research/bert/blob/master/create_pretraining_data.py)

假设一批数据已经选出 150 个目标，按比例约有 120 个走 `[MASK]` 分支、15 个随机替换、15 个保持原样。这是期望数量，不保证每批都恰好满足；随机抽到的 token 也可能碰巧等于原 token。

<details markdown="1">
<summary>读原始代码时，15% 的分母要留意一下</summary>

代码按尚未 padding 的 `len(tokens)` 算预测数量，这个长度包含 `[CLS]` / `[SEP]`；数量会取整、至少取 1，并受 `max_predictions_per_seq` 限制。随后从排除了结构 token 的候选位置中选择。因此它并不等于“每个普通位置独立以 15% 概率入选”。例如 8 个普通 token 加上 `[CLS]` / `[SEP]`，长度是 10；上限允许时，`round(10 × 0.15)` 得到 2 个目标，而不是强制 1.2 个。

仓库后来还加入了 whole-word masking 选项，按 `##` 子词组成的组选择；不要把这个可选分支当作原版所有 checkpoint 的训练方式。实际复现时，应固定版本、采样规则和随机状态。

</details>

保留原样的目标也算 loss。模型并不知道这次哪些未改动的 token 被选中了。混入随机替换和原样输入，是为了缓解预训练里经常见 `[MASK]`、下游却通常见不到它的差别，不意味着这个差别完全消失了。

选哪些位置，也会改变模型需要利用的线索。2019 年百度的 [ERNIE §3.2](https://arxiv.org/html/1904.09223v1#S3.SS2) 尝试按短语、实体整体遮盖。拿“纽约市位于美国”来说，只遮“约”，还能从“纽”和“市”猜；把“纽约市”一起遮住，就更需要外部上下文。这是早期 ERNIE 对训练样本的改动，和后来同名聊天模型的架构是两回事。

### Attention mask：哪些位置可以被读取

双向 encoder 可以读到当前 token 的两边。下面只画有效 query 行，`1` 表示允许读取：

| Query \ Key | `tea` | `[MASK]` | `again` | `[PAD]` |
| --- | --- | --- | --- | --- |
| `tea` | 1 | 1 | 1 | 0 |
| `[MASK]` | 1 | 1 | 1 | 0 |
| `again` | 1 | 1 | 1 | 0 |

`[MASK]` 是一个正常参与计算的输入 token，不是 attention 里的负无穷。Padding key 被排除；padding query 是否仍产生非零 hidden state 是实现问题，不能因此把它计入 loss 或 pooling。独立样本如果 packed 到一起，还需要阻断跨样本读取，不能只放一个分隔符就当作隔离了。

补齐序列和生成 padding mask 是两步：前者添加占位 token，后者告诉 attention 哪些位置是占位。序列也不必总补到模型最大长度，常见做法是补到当前 batch 的最长样本。至于 causal mask，它限制的是前后可见性，见[三处 attention 的对照](vanilla-transformer.md#attention)。

### Loss mask：哪些输出要被评分

用一个手工词表看得更清楚：`PAD=0, CLS=1, SEP=2, MASK=3, the=4, tea=5, is=6, cold=7, hot=8`。这不是实际 WordPiece 词表。特意选第 2、4 个位置（从 0 开始），让 `tea` 保持不变、`cold` 变成 `[MASK]`：

| 字段 | 数值 |
| --- | --- |
| 原始 token IDs | `[1, 4, 5, 6, 7, 2, 0]` |
| 模型输入 | `[1, 4, 5, 6, 3, 2, 0]` |
| Attention mask | `[1, 1, 1, 1, 1, 1, 0]` |
| Labels | `[-100, -100, 5, -100, 7, -100, -100]` |

这里用 `-100` 作为教学实现的忽略标记，这是 PyTorch 交叉熵常见的约定，不是词表中的一个词，也不是原始 TensorFlow 数据格式。只有一个 `[MASK]`，却有两个监督位置。**不能用 `input_ids == mask_token_id` 来代替 loss mask。** 小例子固定选了两个位置，只为说明字段关系，不是在模拟 15% 采样。

## 4. Loss 算在哪儿？ {#loss}

在上一节的例子里，训练只检查两个位置的预测：一个是 `tea`，一个是 `cold`。其余位置提供上下文，但不直接计算 MLM loss。

设本 batch 选中的位置集合为 $M$，原 token 为 $x_i$，扰动后的整段输入为 $\tilde x$。MLM 的一个常用聚合方式是：

$$
L_{\mathrm{MLM}}=-\frac{1}{|M|}\sum_{i\in M}\log p_\theta(x_i\mid\tilde x).
$$

每个待预测的位置，会先给词表中的各个 token 打分，得到 logits，再用交叉熵检查原词的概率。`labels` 和输出位置一一对应：第 2 个位置恢复第 2 个位置的原词，不像 next-token prediction 那样把答案错开一格。

实现上，原始 MLM head 先取出目标位置的 hidden states，经过 dense + activation + LayerNorm，再用与输入 embedding 共享的词表矩阵投影，并加 bias。第一次读可以先记住输入输出：**一个位置的表示，变成词表中每个 token 的分数。**[原始实现](https://github.com/google-research/bert/blob/master/run_pretraining.py)

原 TensorFlow 实现用 `masked_lm_positions`、`masked_lm_ids` 和 `masked_lm_weights` 表示监督；补齐的预测槽位权重为 0，分母是权重和加 `1e-5`。下文手算与教学代码使用严格的有效目标均值，不逐位复刻这个 epsilon。

假设两个目标的正确 token 概率分别为 0.8 和 0.25：

$$
\begin{aligned}
L&=\frac{-\log0.8-\log0.25}{2}\\
 &\approx0.804719.
\end{aligned}
$$

如果误除以包含 padding 的 7 个位置，就会得到约 0.229920：预测没变好，只是分母变大了。这个例子里该数的是两个有效目标，补几个 `[PAD]` 不应该改变答案。教学代码遇到零个目标会报错，方便尽早发现数据问题。

### 一次训练，实际做了哪几步？ {#training-step}

把数据和 loss 接起来，一次更新可以这样看：

1. **留好答案，再改输入。** 保存原 token，选预测位置，构造扰动后的输入和 labels。
2. **整段经过 encoder。** 所有有效位置一起计算 hidden states；选中位置的 MLM head 再给词表打分。
3. **拿原词算 loss。** 对有效目标算交叉熵，再做反向传播，更新 embedding、encoder 和 MLM head。原始 BERT 还会加入下一节的 NSP loss。
4. **下一步用更新后的参数。** 换一个 batch 继续训练；只有实际执行参数更新，模型才会逐渐改变预测。

因此，15% 的预测位置不等于只做 15% 的计算。被选中的词需要其他词提供上下文，encoder 仍要处理整段输入；节省的是只对一部分位置计算词表输出的那部分工作。[原始训练实现](https://github.com/google-research/bert/blob/master/run_pretraining.py)

同一句话也可以出成不同的题：这次藏 `tea`，下次藏 `cold`。原始实现会预先生成若干遮盖版本并保存；RoBERTa 改为在送入模型时生成 masking。两者的区别在于题目何时产生、能怎样变化，不是“一个有随机性、另一个没有”。[RoBERTa §4.1](https://arxiv.org/html/1907.11692v1#S4.SS1)

<details markdown="1">
<summary>动手看看：换输入、加 padding，loss 会怎样变？</summary>

<div class="encoder-lab" data-encoder-lab="mlm" data-lang="zh" id="mlm-lab" markdown="1">

**输入换了，答案没换。**

选中 `cold` 以后，三种扰动方式如下。即使输入保持原样，也仍然要预测 `cold`：

| 分支 | 模型输入 | 目标 |
| --- | --- | --- |
| 遮盖 | `[MASK]` | `cold` |
| 随机替换（假设这次抽到 hot） | `hot` | `cold` |
| 保持原样 | `cold` | `cold` |

手工设定两个正确目标的概率：`tea` 为 0.80，`cold` 为 0.25。按两个目标平均，loss 约为 0.805；增加 padding 不会改变它。开启 JavaScript 可以切换输入、调整概率，看看哪些量会变。这里只计算算式，不运行模型。

</div>

</details>

<details markdown="1">
<summary>没有直接算 loss 的位置，还会学到东西吗？</summary>

会。未选中的位置仍可为目标位置提供上下文；目标的 loss 能沿 attention 路径传回它们。忽略某个位置的输出评分，不等于切断这个位置在整个模型里的梯度。

对目标位置 $i$ 的某个词表 logit $z_{i,v}$，上面这个平均 loss 有

$$
\frac{\partial L}{\partial z_{i,v}}
=\frac{p_{i,v}-\mathbf 1[v=x_i]}{|M|}.
$$

未选中位置的输出 logits 不直接进入该 loss；但共享参数和上下文 hidden states 仍可能有梯度。要区分“这个输出没被评分”和“这段计算没有参与学习”。

</details>

## 5. NSP：两段文字是不是接着写的？ {#nsp}

原始 BERT 还训练 next sentence prediction：给两段文本，判断第二段是否实际接在第一段后面。样本一半是相邻段，一半是随机配对，输出是二分类，而不是下一句的 token。[原文 §3.1](https://arxiv.org/html/1810.04805v2#S3.SS1)

这里的“sentence”可以是一段连续文本，不一定只有一句话。NSP head 读取 `[CLS]` 的 pooled 表示，原实现把 MLM loss 与 NSP loss 相加后反向传播。

比如原文写着“开始下雨了。我收起晾在外面的衣服”，把相邻两段取出来，就是正例。如果第二段换成从别处随机抽到的“矩阵乘法满足结合律”，就是负例。标签看的是它们在语料中是否相邻，而不是我们觉得接得顺不顺：随机抽来的段落，也可能碰巧很搭。

2019 年的 RoBERTa 重新检查了 masking、输入组织、NSP 和训练规模。它采用动态 masking，并在调整后的输入配方里不再使用 NSP。读消融时要注意：输入如何拼接也在变化，不能将全部收益归给“删了一个 loss”。[RoBERTa §4](https://arxiv.org/html/1907.11692v1#S4)

所以，看到一个后来发布的 encoder，可以分别问它用了什么结构、怎么组织数据、用什么目标训练。模型长得像 BERT，不意味着训练时一定做过 NSP。

## 6. 预训练结束后，怎么换成自己的任务？ {#finetuning}

假设我们已经下载了一个预训练 checkpoint，接下来想判断评论是好评还是差评。这时给模型的是完整评论，不需要像 MLM 那样先把词藏起来。训练答案也从“原词是什么”，换成了“这条评论是什么类别”。

### 先做一个评论分类器 {#classify-example}

以“茶凉了，等了很久”为例，我们希望模型输出负面类别。Encoder 仍然给每个位置输出向量，分类层读取 `[CLS]` 的表示，把它转成类别分数。`[CLS]` 不是预先写好的整句平均值；它通过 attention 接收其他位置的信息，再在训练中学会保留任务需要的内容。原始分类代码还会对它做一次 dense + tanh pooling。[分类实现](https://github.com/google-research/bert/blob/master/run_classifier.py)、[Pooler 实现](https://github.com/google-research/bert/blob/master/modeling.py)

记 pooled 表示为 $h_{\mathrm{CLS}}$，分类层的参数为 $W,b$，则

$$
\begin{aligned}
z&=W h_{\mathrm{CLS}}+b,\\
p&=\operatorname{softmax}(z).
\end{aligned}
$$

BERT-Base 的表示是 768 维，二分类的 $W$ 就是 `[2, 768]`，输出两个 logits。假设一次输出是 `[1, 3]`，类别顺序为 `[正面, 负面]`。用 $p_-$ 表示负面类别的概率：

$$
\begin{aligned}
p_-&=\frac{e^3}{e^1+e^3}\\
   &\approx0.880797,\\
L&=-\log p_-\approx0.126928.
\end{aligned}
$$

这是方便手算的假设分数，不是模型实测。注意这次 softmax 比较的是**两个类别**，MLM 比较的则是整个词表。Encoder 可以复用，输出到底意味着什么，取决于新接的 head 和标签。

### 换成问答，读哪些位置？ {#answer-span}

文章写着“会议改到了周五下午。”你问“会议改到什么时候？”答案已经在原文里，模型不用重新写一句话，只需要找出**“周五下午”从哪里开始、到哪里结束**。这就是抽取式问答（extractive QA）。

问题和文章一起进入 BERT，文章的表示因此也会受到问题影响。问答输出层在每个位置给出两个分数：这里像不像答案的**起点（start）**，像不像**终点（end）**。起点和终点各有一次 softmax，比较的是位置，不是词表。

为了看清位置，先把这句原文手工分成 6 格：`会议 / 改到 / 了 / 周五 / 下午 / 。`，从 0 编号。这只是示意分块，不是实际 WordPiece 分词。假设模型给出了下面的分数：

<figure class="worked-update worked-update--pairs" lang="zh-CN" id="answer-boundaries">
  <figcaption>找出“周五下午” · 手工设定分数，演示答案如何取出。</figcaption>
  <ol>
    <li><small>起点 · 位置 3</small><strong>周五 · 分数 4</strong><span>从这里开始读。起点分数比较的是“答案从哪里开始”。</span></li>
    <li><small>终点 · 位置 4</small><strong>下午 · 分数 5</strong><span>读到这里结束，包含这一格。终点分数单独计算。</span></li>
  </ol>
</figure>

这个候选片段的总分是 `4 + 5 = 9`。在下面设定的完整分数里，它是合法片段中分数最高的，所以取出位置 3 到 4，得到“周五下午”。训练时，则用标注的起止位置监督这两个预测。[原论文 §4.2](https://arxiv.org/html/1810.04805v2#S4.SS2)

**不能随手各取一个最高分就结束。** 如果最高起点在位置 4，最高终点却在位置 3，这个范围是倒着的。还要排除问题、特殊 token 和 padding；需要时限制答案长度。下面的小程序只做这一步选择，不运行 BERT。

<details markdown="1">
<summary>用 Python 选片段：别让答案倒着来，也别跨过无效位置</summary>

```python
import math

def best_answer_span(start_scores, end_scores, allowed, max_length):
    length = len(start_scores)
    if length != len(end_scores) or length != len(allowed):
        raise ValueError("Scores and mask must have the same length")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if any(type(value) is not bool for value in allowed):
        raise ValueError("allowed must contain booleans")
    if not all(math.isfinite(score) for score in [*start_scores, *end_scores]):
        raise ValueError("This example expects finite scores")
    best = None
    for start in range(length):
        for end in range(start, min(length, start + max_length)):
            if not allowed[end]:
                break
            score = start_scores[start] + end_scores[end]
            if best is None or score > best[0]:
                best = (score, start, end)
    return best

start_scores = [0, 0, 0, 4, 1, -2]
end_scores = [0, 0, 0, 1, 5, -2]
answer = best_answer_span(start_scores, end_scores, [True] * 6, 3)
assert answer == (9, 3, 4)
print(answer)
```

`allowed` 标出哪些位置可以出现在答案里，`max_length` 是最多几格。循环只考虑终点不早于起点的连续片段，遇到无效位置就停止延伸。同分时保留先遇到的片段；这里没有计算概率，也没判断答案是否真实。若长度为 $L$、最大答案长度为 $A$，枚举成本为 $O(L\min(L,A))$；教学数据很短，不必先做复杂优化。

真实分词不能靠把 token 字符串直接拼回去。应保存每个 token 对应原文的字符范围，再从原文切出答案，这样空格、标点和子词才不会被弄坏。[Transformers 的问答示例](https://huggingface.co/docs/transformers/tasks/question_answering)用 offset mapping 和 sequence IDs 处理这些对齐。

</details>

如果改问“会议在哪个房间？”，这段原文就没有答案。但只要还存在可选片段，上面的程序仍会挑出一个。**能选出最高分，不代表原文真的回答了问题。** 无答案任务需要相应训练标签与拒答判断；原始 BERT 在 SQuAD 2.0 中用 `[CLS]` 表示空答案，并在开发集上选择判断阈值。[原论文 §4.3](https://arxiv.org/html/1810.04805v2#S4.SS3)

也别把“原文没答案”和“答案被截断了”混为一谈。前者需要拒答，后者要先检查文档分块和输入长度。若想把原文改写成一句解释，则已经是生成式问答的需求，不能只靠这个选位置的输出层。

把几种任务并排看，就很容易分清 head 在做什么：

| 任务 | 读取哪里 | 输出与训练数据 |
| --- | --- | --- |
| 文本分类 | `[CLS]` 的聚合表示 | 每条文本的类别 logits 与标签 |
| 命名实体识别 | 每个相关 token 的表示 | token 级标签；先处理词与 subword 对齐 |
| 抽取式问答 | 文中每个候选答案位置 | start / end logits；排除问题和 padding 等非答案位置 |
| 成对相关性打分 | query 与 document 联合编码 | 一对文本的分数与相关性标签 |
| 单向量检索 | 两边分别编码，再 pooling | 还需合适的表示学习目标与检索评估 |

### Encoder 要不要一起训练？ {#adaptation}

如果只想先跑通任务，可以冻结 encoder，只训练分类层。这样不用保存 encoder 反向传播所需的中间激活；输入固定、关闭 dropout 时，还能提前算好特征。但如果原来的表示没有保留你需要区分的信息，一个小分类层能补的也有限。

全参数微调则允许 encoder 跟着任务一起调整，代价是更多训练显存和计算，也更容易在小数据上过拟合。原论文采用端到端微调，并用开发集选择学习率；小数据上的结果还会受分类层初始化和数据顺序影响。[BERT §3.2、§4.1](https://arxiv.org/html/1810.04805v2#S3.SS2)

比较这两种做法时，用同一份数据划分，记录任务分数、训练成本和不同 seed 的变化。不要先假定“全部解冻肯定更好”，再只保留最好的一次结果。

### 为什么检索还要分双塔和 cross-encoder？ {#retrieval-use}

如果将 query 和 document 一起送进 BERT，双方可以在每层交换信息，适合仔细判断一对文本是否相关。但 document 的表示已经依赖这次 query，不能提前算一次供所有 query 复用。假设有 10 万篇候选文档，每次请求都这样做，就要评估 10 万个文本对；可以批处理，却不能省掉这些配对计算。

双塔让 query 和 document 分别编码，文档向量就能提前计算和索引，检索时更省事；代价是两边编码时不能直接交换 token 级信息。常见组合是先用向量召回一小批，再用 cross-encoder 精排。为什么适合余弦相似度的句向量需要专门训练，可以读 [Sentence-BERT（2019）](https://aclanthology.org/D19-1410/)；系统怎么拆，见 [双塔检索](../../04-search/dual-encoder.md)和 [BGE-M3 / Qwen3 Embedding](../../04-search/embedding-models.md)。

## 7. 跑一次，检查最容易弄错的字段 {#small-lab}

[教学脚本](../code/mlm_contracts.py)只依赖 Python 标准库，固定替换结果，不下载模型，也不进行预训练：

```bash
python3 00-foundations/code/mlm_contracts.py
```

```text
Input IDs: [1, 4, 5, 6, 3, 2, 0]
Labels: [-100, -100, 5, -100, 7, -100, -100]
Attention: [1, 1, 1, 1, 1, 1, 0]
Selected-token loss: 0.804719
```

`corruption_action` 用一次均匀随机数演示选中之后的 80/10/10 分支；原始代码使用条件式的两次抽样，概率一致，但相同 seed 不保证相同结果。`build_example` 接收明确的替换位置，保留原标签；`masked_cross_entropy` 只对选中的位置取平均。它不是随机 collator，也没有 attention、任务 head 或反向传播。代码完整保留了“未变但被选中”的情况，方便与只数 `[MASK]` 的错误实现对照。

可以只改一个条件试试：

| 你改了什么 | 应该观察到什么 | 在检查什么 |
| --- | --- | --- |
| 追加 padding，labels 仍为 `-100` | Loss 不变 | 是否只对有效目标取平均 |
| 把忽略位置的 logits 改得很大 | Loss 不变 | 没有标签的位置是否混入计算 |
| 提高正确目标的 logit，其他不变 | Loss 下降 | 目标 ID、维度和交叉熵方向是否正确 |

这些检查通过，只说明数据与 loss 的关系对了。接上真实模型后，还得确认梯度和参数确实在更新，最终再看任务的验证集表现。

## 8. 真正接数据时，还要检查什么？ {#checks}

先做一个足够小、能看懂结果的实验。比如让模型拟合几十条训练样本：如果连它们都学不好，优先检查标签、mask、梯度和优化器；能拟合以后，再看独立验证集。这个小实验是排错工具，不是泛化能力的证明。

| 现象 | 先检查哪里 |
| --- | --- |
| Loss 低得不正常 | 原答案是否泄漏、是否除以了总长度、是否只统计未扰动的位置 |
| 不同 batch 的 loss 忽高忽低 | 有效目标数、长度分布、样本重复和采样是否一致 |
| MLM 学得好，分类却不好 | 标签质量、数据分布、任务 head 与微调方案；预训练 loss 不是任务指标 |
| 离线分数很好，上线掉很多 | 训练与评估是否共享近重复文本、截断是否删了关键内容 |

分类指标也要结合数据看。假设验证集里有 95 条好评、5 条差评，一律预测“好评”也有 **95% 准确率（accuracy），但差评召回率（recall）是 0%**。如果你关心的是找出投诉，这个模型几乎没帮上忙。除了总分，至少看看各类别的精确率和召回率，再读几条分错的评论；按用户、来源或时间划分数据，取决于你真正要泛化到什么场景。

### 分错了一条评论，先别急着换模型 {#inspect-a-mistake}

比如“包装很好看，但茶已经坏了”被分成了好评。先打印**实际送进模型的 token**：如果截断后只剩“包装很好看”，模型根本没读到后半句，先改输入处理，而不是继续调学习率。

如果整句都在，再检查类别编号：训练时 `0` 是差评，展示时却把 `0` 当成好评，也会让结果看起来完全不对。输入和映射都没问题，才继续看模型是否过度依赖“很好看”这样的局部词。

可以补两条小检查：“茶已经坏了”和“茶没有坏”。它们能帮你观察模型对否定词是否敏感，但两条都答对，仍不能说明它处理好了所有否定、转折和反讽。小例子用于定位问题，正式结论还要回到独立测试集。

加载 checkpoint 时，把 tokenizer、特殊 token、label 映射和截断长度一起保存。模型能正常输出，不代表这些配置就匹配。验证 masking 时固定样本和随机种子；比较模型时用独立验证集，记录采样方式，再检查长度、类别和来源不同的样本。

读到这里，可以把 BERT 连成一条完整的路：**从原文构造训练题 → 用上下文学每个位置的表示 → 接上任务输出层 → 用任务数据适配和评估。** 接下来读 [Decoder-only](decoder-only.md)，看看目标变成“继续往下写”以后，哪些地方需要跟着变。

核对日期：2026-10-10。本文的数值是教学算例；代码检查覆盖局部 attention、数据与 loss、答案片段选择，未复现 BERT 预训练或论文分数。
