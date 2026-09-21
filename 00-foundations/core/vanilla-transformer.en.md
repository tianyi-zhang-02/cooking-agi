# Vanilla Transformer

[中文](vanilla-transformer.md) · **English**

> Reading time: ~15 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-en><span>Problem</span><strong>Exchange information across all positions in parallel, without recurrence</strong></div>
    <div class="recipe-face" data-concept-zh><span>解决什么问题</span><strong>让所有位置并行交换信息，不再排队递归</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-en><span>Prerequisites</span><strong>token matrix · position · attention mask</strong></div>
    <div class="recipe-face" data-concept-zh><span>前置知识</span><strong>token matrix · position · attention mask</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-en><span>Core mechanisms</span><strong>self-attention · cross-attention · FFN · residual connection</strong></div>
    <div class="recipe-face" data-concept-zh><span>核心机制</span><strong>自注意力 · 交叉注意力 · 前馈网络 · 残差连接</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-en><span>Common failure</span><strong>Mixing up the three attention sites or masking the wrong direction</strong></div>
    <div class="recipe-face" data-concept-zh><span>常见错误</span><strong>混淆三处注意力，以及把因果掩码遮反</strong></div>
  </div>
</div>

<div class="bilingual-note bilingual-intro">
  <span>CONCEPT-BY-CONCEPT · 逐概念双语</span>
  <p>Read the English by default. On each card, select <strong>中文 ↻</strong> at the top
  right to see the equivalent Chinese for the same concept in place; there is no need
  to return to the top of the page, and you keep your reading position.</p>
</div>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

## Quick learning: one 2017 Transformer layer from end to end

<details class="interview" markdown="1">
<summary>Encoder, decoder, and the boundary with modern decoder-only models</summary>

**Quick memory**: an encoder is self-attention plus an FFN. A decoder adds masked self-attention and cross-attention. Residual paths and normalization wrap every sublayer.

**Interview answer**

> Source tokens enter the encoder after embedding and positional encoding. The decoder applies masked self-attention to the right-shifted target, reads the encoder states through cross-attention, and finally uses a linear vocabulary head plus softmax to predict the next token. Training parallelizes over all target positions; generation still has to be autoregressive.

<details markdown="1">
<summary><b>Deep dive</b>: why are both the target shift and the causal mask necessary?</summary>

The shift decides that each position's input is the previous true token. The causal mask decides that this position cannot read labels farther to the right in self-attention. Shifting without masking still lets deeper attention layers peek at the future; masking without shifting hands the current position the embedding of the very token it has to predict.

</details>
</details>

## Move information, then transform it

Forget the arrow-heavy architecture diagram for a moment. A Transformer layer
repeats only two operations: **attention goes to other positions to fetch
information; the FFN stays at the current position and processes it.** The original
architecture is still an encoder–decoder, but recurrence is removed completely.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh concept-title-h2" role="heading" aria-level="2">先交换位置信息，再逐位置变换</div>

先忘掉那张塞满箭头的大框图。Transformer 一层其实只反复做两件事：**attention 去别的位置拿信息，FFN 留在当前位置加工信息。** 原版仍然是 encoder–decoder，但 recurrence 被彻底拿掉了。

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

## A layer has only two jobs

1. **Attention mixing** exchanges information across token positions.
2. **Channel mixing / FFN** transforms each token's channels independently.

Residual connections and normalization wrap those two operations. Stacking many
layers still means repeating “go out and find information → come back and process
it.” Remembering it this way is much lighter than memorizing the full block diagram.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh concept-title-h2" role="heading" aria-level="2">一层只有两类计算</div>

1. **Attention mixing**：不同 token 之间交换信息。
2. **Channel mixing / FFN**：每个 token 独立变换自己的通道。

外面再套 residual connection 与 normalization。堆很多层，本质上还是反复“出去找信息 → 回来加工信息”。这样记，比背整张结构图轻松很多。

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

## The three attention sites ask different questions

| Site | Query | Key / Value | Mask | Role |
| --- | --- | --- | --- | --- |
| encoder self-attention | source | source | padding | understand the input bidirectionally |
| decoder self-attention | target prefix | target prefix | causal + padding | see only tokens that were already generated |
| decoder cross-attention | decoder state | encoder states | source padding | fetch the evidence needed right now from the input |

All three use the same scaled dot-product attention:

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

$M$ is the mask: it adds 0 at allowed positions and $-\infty$ at forbidden positions.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh concept-title-h2" role="heading" aria-level="2">三处 attention 在问不同的问题</div>

| 位置 | Query | Key / Value | mask | 作用 |
| --- | --- | --- | --- | --- |
| encoder self-attention | source | source | padding | 双向理解输入 |
| decoder self-attention | target prefix | target prefix | causal + padding | 只能看已经生成的 token |
| decoder cross-attention | decoder state | encoder states | source padding | 从输入中取当前所需证据 |

scaled dot-product attention：

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

$M$ 是 mask：允许的位置加 0，禁止的位置加 $-\infty$。

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

## Why the architecture scales

- **Parallel training:** $Q/K/V$ for every position are computed at once.
- **Shorter paths:** any two tokens can interact directly through one attention layer.
- **One uniform structure:** self-attention and cross-attention differ only in where
  the tensors come from.

The cost is a $T\times T$ self-attention score matrix, so the time and memory of a
standard implementation grow approximately quadratically with sequence length.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh concept-title-h2" role="heading" aria-level="2">为什么这套做法突然可以做大</div>

- **训练并行**：所有位置的 $Q/K/V$ 可以一次算出；
- **路径更短**：任意两个 token 一层 attention 就能直接交互；
- **结构统一**：self-attention 与 cross-attention 只是张量来源不同。

代价是 self-attention 的分数矩阵大小为 $T\times T$，标准实现的时间和显存随序列长度近似二次增长。

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

## The missing information: attention does not know order

Attention by itself does not know order. The original Transformer adds a fixed
sinusoidal positional encoding to each token embedding:

$$z_t = E[x_t] + PE_t$$

Without positional encoding, the model sees only a bag of tokens: permuting the input
merely permutes the output in the same way.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh concept-title-h2" role="heading" aria-level="2">缺失的信息：Attention 本身不知道顺序</div>

Attention 本身不知道顺序。原版把固定 sinusoidal position encoding 加到 token embedding：

$$z_t = E[x_t] + PE_t$$

没有位置编码时，模型只能看到一袋 token；调换顺序只会让输出跟着调换。

</div>
</section>

## From token IDs to the next token: the complete path

Use a tiny English-to-Chinese example to connect the entire forward path of the 2017
Transformer. Look only at shapes first, then at each module's job.

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### 1. How the source enters the encoder

The English sentence first goes through the tokenizer:

```text
[I, love, you] -> [21, 135, 48]
```

A token ID is only an integer index into the vocabulary; it carries no notion of
distance or magnitude. The embedding layer uses these IDs to look up vectors in a
matrix $E$. With source length $S=3$ and the original $d_{\text{model}}=512$:

$$X_{\text{emb}}\in\mathbb{R}^{S\times d_{\text{model}}}
=\mathbb{R}^{3\times512}.$$

The paper actually multiplies the embeddings by $\sqrt{d_{\text{model}}}$ and then
adds a fixed sinusoidal positional encoding:

$$X=\sqrt{d_{\text{model}}}\,X_{\text{emb}}+PE,$$

$$PE(pos,2i)=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad
PE(pos,2i+1)=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

There is no need to memorize the constants. What to remember is that $PE$ is **added
to the embedding, not concatenated**, so the shape remains $(S,512)$. Every position
receives a unique but regular set of phases, which is what lets the model tell
`I love you` apart from a sentence with the same tokens in a different order. Dropout
is applied after the sum.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">1. Source 怎样进入 encoder</div>

英文句子先经过 tokenizer：

```text
[I, love, you] -> [21, 135, 48]
```

token ID 只是词表里的整数索引，本身没有距离或大小意义。embedding 层用这些 ID
从矩阵 $E$ 中查出向量。若 source 长度为 $S=3$，原版
$d_{\text{model}}=512$：

$$X_{\text{emb}}\in\mathbb{R}^{S\times d_{\text{model}}}
=\mathbb{R}^{3\times512}.$$

原论文实际把 embedding 乘以 $\sqrt{d_{\text{model}}}$，再加固定的正弦位置编码：

$$X=\sqrt{d_{\text{model}}}\,X_{\text{emb}}+PE,$$

$$PE(pos,2i)=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad
PE(pos,2i+1)=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

不必死记常数；要记住 $PE$ 与 embedding **相加而不是拼接**，所以 shape 仍是
$(S,512)$。每个位置得到一组独特但有规律的相位，模型才能区分 `I love you`
和 token 顺序被调换的句子。相加后还会应用 dropout。

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### Attention dimensions and edge cases

For a single attention head, the more general shapes are

$$Q\in\mathbb R^{L_q\times d_k},\qquad
K\in\mathbb R^{L_{kv}\times d_k},\qquad
V\in\mathbb R^{L_{kv}\times d_v}.$$

Therefore

$$QK^\top\in\mathbb R^{L_q\times L_{kv}},\qquad
\operatorname{softmax}(QK^\top)V\in\mathbb R^{L_q\times d_v}.$$

There are two real shape constraints here. Q and K must both have final dimension
$d_k$ so that their dot product is defined. K and V must both have sequence length
$L_{kv}$, because every key has to correspond to one value that gets read. $d_v$ need
not equal $d_k$; the number of output channels is determined by $d_v$. Setting
$d_k=d_v=64$ per head in the original model was a design choice, not a mathematical
requirement of attention.

Why divide the score by $\sqrt{d_k}$? If $q_i,k_i$ are approximately independent with
zero mean and unit variance, then

$$q^\top k=\sum_{i=1}^{d_k}q_ik_i,\qquad
\operatorname{Var}(q^\top k)\approx d_k.$$

The standard deviation of the dot product therefore grows like $\sqrt{d_k}$. After
scaling,

$$\operatorname{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)\approx1,$$

so the score does not push softmax into its saturated region merely because the head
dimension got larger. Otherwise attention becomes nearly one-hot too early and the
gradients at non-maximum positions are tiny. The dimension used here is the Q/K
matching dimension $d_k$; it has nothing to do with $d_v$.

The individual coordinates of $W_Q,W_K,W_V$ have no fixed human meaning, but the
three projections play different roles: Q expresses “what I am looking for,” K
expresses “how I can be matched,” and V expresses “what content is passed on after a
match.” Hence

$$QK^\top=\text{addressing / routing},\qquad V=\text{retrieved content}.$$

If $Q=K$ is forced, the score before the mask and softmax becomes

$$S=QQ^\top,$$

a symmetric positive-semidefinite matrix with $S_{ij}=S_{ji}$. Row-wise softmax need
not preserve the symmetry, and a causal mask also breaks it, but the underlying
matching function has already been restricted to a symmetric similarity. It cannot
naturally express a directional relation in which “$i$ queries $j$” differs from
“$j$ queries $i$.” Separate $W_Q,W_K$ remove this constraint; a separate $W_V$
decouples “how the information is found” from “what is read once it is found.”

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">Attention 的维度与边界条件</div>

对单个 attention head，更一般的 shape 是：

$$Q\in\mathbb R^{L_q\times d_k},\qquad
K\in\mathbb R^{L_{kv}\times d_k},\qquad
V\in\mathbb R^{L_{kv}\times d_v}.$$

于是

$$QK^\top\in\mathbb R^{L_q\times L_{kv}},\qquad
\operatorname{softmax}(QK^\top)V\in\mathbb R^{L_q\times d_v}.$$

这里有两个真正的 shape 约束：Q 与 K 的最后一维都必须是 $d_k$，才能计算点积；
K 与 V 的序列长度都必须是 $L_{kv}$，因为每个 key 都要对应一份被读取的 value。
$d_v$ 不必等于 $d_k$，输出通道数由 $d_v$ 决定。原版每个头取
$d_k=d_v=64$ 是设计选择，不是 attention 的数学要求。

为什么 score 要除以 $\sqrt{d_k}$？若 $q_i,k_i$ 近似独立、均值为 0、方差为 1，
那么

$$q^\top k=\sum_{i=1}^{d_k}q_ik_i,\qquad
\operatorname{Var}(q^\top k)\approx d_k.$$

点积的标准差因此随 $\sqrt{d_k}$ 增长。缩放后

$$\operatorname{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)\approx1,$$

score 不会仅仅因为 head dimension 变大就把 softmax 推到饱和区。否则注意力会过早接近
one-hot，非最大位置的梯度很小。这里使用的是 Q/K 的匹配维度 $d_k$，与 $d_v$ 无关。

$W_Q,W_K,W_V$ 的单个坐标没有固定的人类语义，但三套投影承担不同角色：Q 表达“我要找
什么”，K 表达“我怎样被匹配”，V 表达“匹配后传递什么内容”。因此

$$QK^\top=\text{addressing / routing},\qquad V=\text{retrieved content}.$$

如果强制 $Q=K$，mask 和 softmax 之前的 score 变成

$$S=QQ^\top,$$

它是对称的 positive-semidefinite matrix，满足 $S_{ij}=S_{ji}$。逐行 softmax 后不一定
仍对称，causal mask 也会破坏对称性，但底层匹配函数已经被限制成对称相似度，难以自然
表达“$i$ 查询 $j$”与“$j$ 查询 $i$”不同的方向关系。分开的 $W_Q,W_K$ 解除这个约束；
分开的 $W_V$ 则把“怎样找到信息”和“找到后读取什么”解耦。

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### 2. What one encoder layer computes

The original model stacks 6 encoder layers with identical structure. Each layer is

```text
X -> multi-head self-attention -> Add & Norm -> FFN -> Add & Norm
```

In self-attention, $Q/K/V$ all come from the current source states, which is why it
is called **self**-attention:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

The original has $h=8$ heads, so each head has

$$d_k=d_v=\frac{512}{8}=64.$$

For one head, the attention formula from the previous section does four things in
order:

$$S=QK^\top,\qquad
\widetilde S=\frac{S}{\sqrt{d_k}},\qquad
A=\operatorname{softmax}_{\text{key}}(\widetilde S),\qquad
O=AV.$$

$S_{ij}$ is the match score between the query at source position $i$ and the key at
position $j$; softmax is computed along the key dimension of each row. For example,
when some head in some layer updates `love`, it might obtain
$A_{\text{love}}=[0.4,\ 0.1,\ 0.5]$, and therefore

$$o_{\text{love}}=0.4v_I+0.1v_{\text{love}}+0.5v_{\text{you}}.$$

These are the read weights of this one head in this one layer. Do not read them as
the model as a whole paying 40% “semantic attention.” Each of the 8 heads has its own
projections; at the end they are concatenated and passed through an output projection:

$$\operatorname{MHA}(X)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_8)W^O.$$

The original is **post-norm**. Each sublayer's output goes through dropout first, is
then added to the input through the residual connection, and LayerNorm comes last:

$$H=\operatorname{LN}\big(X+\operatorname{Dropout}(\operatorname{MHA}(X))\big).$$

Each token then passes **independently** through the same set of FFN parameters:

$$\operatorname{FFN}(h)=\operatorname{ReLU}(hW_1+b_1)W_2+b_2,$$

$$512\rightarrow2048\rightarrow512.$$

Attention is responsible for exchanging information between tokens; the FFN does not
cross tokens and only processes the channel information each position has already
collected. The second Add & Norm is

$$Y=\operatorname{LN}\big(H+\operatorname{Dropout}(\operatorname{FFN}(H))\big).$$

Repeating this for 6 layers gives the encoder memory:

$$C=\operatorname{Encoder}(X)\in\mathbb{R}^{S\times512}.$$

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">2. 一个 encoder layer 算什么</div>

原版堆叠 6 个相同结构的 encoder layer。每层都是：

```text
X -> multi-head self-attention -> Add & Norm -> FFN -> Add & Norm
```

Self-attention 的 $Q/K/V$ 都来自当前 source states，因此称为 **self**-attention：

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

原版有 $h=8$ 个头，所以每个头

$$d_k=d_v=\frac{512}{8}=64.$$

对一个头，上一节的注意力公式依次完成四件事：

$$S=QK^\top,\qquad
\widetilde S=\frac{S}{\sqrt{d_k}},\qquad
A=\operatorname{softmax}_{\text{key}}(\widetilde S),\qquad
O=AV.$$

$S_{ij}$ 是 source 位置 $i$ 的 query 与位置 $j$ 的 key 的匹配分数；softmax
沿每一行的 key 维计算。比如某层某个头在更新 `love` 时，可能得到

$$A_{\text{love}}=[0.4,\ 0.1,\ 0.5],$$

于是

$$o_{\text{love}}=0.4v_I+0.1v_{\text{love}}+0.5v_{\text{you}}.$$

这是这一层、这一个头里的读取权重，不要把它直接解释成模型总体上有 40% 的
“语义关注”。8 个头各有自己的投影，最后拼接并经过输出投影：

$$\operatorname{MHA}(X)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_8)W^O.$$

原版是 **post-norm**。每个 sublayer 的输出先做 dropout，再和输入走残差相加，
最后 LayerNorm：

$$H=\operatorname{LN}\big(X+\operatorname{Dropout}(\operatorname{MHA}(X))\big).$$

随后每个 token **独立**通过同一组 FFN 参数：

$$\operatorname{FFN}(h)=\operatorname{ReLU}(hW_1+b_1)W_2+b_2,$$

$$512\rightarrow2048\rightarrow512.$$

Attention 负责 token 之间交换信息；FFN 不跨 token，只加工每个位置已经收集到的
通道信息。第二个 Add & Norm 是

$$Y=\operatorname{LN}\big(H+\operatorname{Dropout}(\operatorname{FFN}(H))\big).$$

重复 6 层后得到 encoder memory：

$$C=\operatorname{Encoder}(X)\in\mathbb{R}^{S\times512}.$$

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### 3. Why the target is shifted right

If the correct translation is

```text
target labels: [我, 爱, 你, <EOS>]
decoder input: [<BOS>, 我, 爱, 你]
```

the two are offset by one position, which is what lets position $t$ learn to
“predict token $t$ from the tokens before it.” During training the whole correct
target is known and can be fed in parallel in one pass; the causal mask is what
prevents peeking at the future.

The decoder input goes through embedding, positional encoding, and dropout in the
same way. The source embedding, the target embedding, and the output projection
**can be separate as a matter of structure**; the original paper, however, uses a
joint vocabulary and shares the weights of both embeddings and the pre-softmax
projection, while multiplying the embeddings by $\sqrt{d_{\text{model}}}$. So “a
translation model necessarily uses three different sets of weights” is not a fact
about the original Transformer.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">3. Target 为什么要右移</div>

若正确译文是

```text
target labels: [我, 爱, 你, <EOS>]
decoder input: [<BOS>, 我, 爱, 你]
```

二者错开一位，位置 $t$ 才能学习“根据此前 token 预测第 $t$ 个 token”。训练时
整条正确 target 已知，可以一次并行输入；causal mask 负责阻止偷看未来。

Decoder 输入同样经过 embedding、位置编码和 dropout。source embedding、target
embedding 和输出投影在**结构上可以分开**；不过原论文使用联合词表，并共享两套
embedding 与 pre-softmax 投影的权重，同时对 embedding 乘 $\sqrt{d_{\text{model}}}$。
所以“翻译任务必然使用三套不同权重”不是原版 Transformer 的事实。

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### 4. Why one decoder layer has three sublayers

The original also stacks 6 decoder layers; each has one more sublayer than an encoder
layer, the cross-attention:

```text
target states
  -> masked self-attention -> Add & Norm
  -> cross-attention       -> Add & Norm
  -> FFN                   -> Add & Norm
```

The first sublayer is self-attention over the target, but a causal mask is added to
the scores. For 4 positions, the allowed region is the lower triangle:

$$
\begin{bmatrix}
s_{11}&-\infty&-\infty&-\infty\\
s_{21}&s_{22}&-\infty&-\infty\\
s_{31}&s_{32}&s_{33}&-\infty\\
s_{41}&s_{42}&s_{43}&s_{44}
\end{bmatrix}.
$$

Since $e^{-\infty}=0$, masked positions have weight exactly 0 after softmax; position
$i$ can read only target positions 1 through $i$. The first sublayer produces

$$H_1=\operatorname{LN}\big(Y+
\operatorname{Dropout}(\operatorname{MaskedMHA}(Y))\big).$$

The second sublayer, cross-attention, is where the decoder first touches the English
source:

$$Q=H_1W_Q,\qquad K=CW_K,\qquad V=CW_V.$$

That is, **Q comes from the decoder, while K/V come from the encoder memory**. With
target length $T$ and source length $S$, the attention weights have shape $(T,S)$ and
need not be square. For example, when preparing to generate “爱”, the decoder state
can use its query to read the values in the encoder that relate to `love`:

$$H_2=\operatorname{LN}\big(H_1+
\operatorname{Dropout}(\operatorname{CrossAttention}(H_1,C))\big).$$

The third sublayer is the same position-wise FFN as in the encoder:

$$H_3=\operatorname{LN}\big(H_2+
\operatorname{Dropout}(\operatorname{FFN}(H_2))\big).$$

After 6 layers we have

$$D\in\mathbb{R}^{T\times512}.$$

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">4. 一个 decoder layer 为什么有三段</div>

原版也堆 6 个 decoder layer；每层比 encoder 多一个 cross-attention：

```text
target states
  -> masked self-attention -> Add & Norm
  -> cross-attention       -> Add & Norm
  -> FFN                   -> Add & Norm
```

第一段是 target 上的 self-attention，但 score 会加 causal mask。4 个位置的允许区域
是下三角：

$$
\begin{bmatrix}
s_{11}&-\infty&-\infty&-\infty\\
s_{21}&s_{22}&-\infty&-\infty\\
s_{31}&s_{32}&s_{33}&-\infty\\
s_{41}&s_{42}&s_{43}&s_{44}
\end{bmatrix}.
$$

因为 $e^{-\infty}=0$，softmax 后被遮住的位置权重严格为 0；第 $i$ 个位置只能
读取第 1 到 $i$ 个 target position。第一个子层得到

$$H_1=\operatorname{LN}\big(Y+
\operatorname{Dropout}(\operatorname{MaskedMHA}(Y))\big).$$

第二段 cross-attention 才让 decoder 接触英文 source：

$$Q=H_1W_Q,\qquad K=CW_K,\qquad V=CW_V.$$

也就是 **Q 来自 decoder，K/V 来自 encoder memory**。若 target 长度是 $T$、
source 长度是 $S$，attention weight 的 shape 是 $(T,S)$，不必是方阵。比如准备
生成“爱”时，decoder state 可以用 query 去读取 encoder 中与 `love` 相关的 value：

$$H_2=\operatorname{LN}\big(H_1+
\operatorname{Dropout}(\operatorname{CrossAttention}(H_1,C))\big).$$

第三段是和 encoder 相同的逐位置 FFN：

$$H_3=\operatorname{LN}\big(H_2+
\operatorname{Dropout}(\operatorname{FFN}(H_2))\big).$$

6 层之后得到

$$D\in\mathbb{R}^{T\times512}.$$

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### 5. From decoder states to vocabulary probabilities

The 512-dimensional vector at each position is projected onto the target vocabulary.
With vocabulary size $V=30{,}000$:

$$Z=DW_{\text{vocab}}+b,\qquad
W_{\text{vocab}}\in\mathbb{R}^{512\times30{,}000},$$

so $Z\in\mathbb{R}^{T\times30{,}000}$. Each row holds that position's **logits** over
all tokens; only softmax turns them into a conditional probability:

$$p(y_t\mid y_{<t},x)=\operatorname{softmax}(Z_t).$$

Training takes the cross-entropy between this distribution and the target token.
Inference uses greedy decoding, beam search, or sampling to pick one token, then
feeds it back into the decoder and continues generating. Softmax itself only provides
probabilities; it does not require choosing the argmax.

The whole path compresses into one line:

$$
\text{source IDs}\rightarrow C\rightarrow
\underbrace{p(y_1\mid x),\ p(y_2\mid y_1,x),\ldots}_{
\text{masked self-attention reads the prefix; cross-attention reads the source}}
$$

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">5. 从 decoder state 变成词表概率</div>

每个位置的 512 维向量投影到目标词表。若词表大小 $V=30{,}000$：

$$Z=DW_{\text{vocab}}+b,\qquad
W_{\text{vocab}}\in\mathbb{R}^{512\times30{,}000},$$

所以 $Z\in\mathbb{R}^{T\times30{,}000}$。每行是该位置对所有 token 的 **logits**，
softmax 才把它变成条件概率：

$$p(y_t\mid y_{<t},x)=\operatorname{softmax}(Z_t).$$

训练时拿这个分布与目标 token 做 cross-entropy；推理时则用 greedy、beam search
或 sampling 选出一个 token，再把它接回 decoder 继续生成。Softmax 本身只给概率，
并不规定必须选 argmax。

整条路径可以压成一行：

$$
\text{source IDs}\rightarrow C\rightarrow
\underbrace{p(y_1\mid x),\ p(y_2\mid y_1,x),\ldots}_{
\text{masked self-attention 读前缀，cross-attention 读 source}}
$$

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

## Do not conflate the 2017 architecture with a modern decoder-only LLM

| | 2017 vanilla | Modern decoder-only |
| --- | --- | --- |
| Body | encoder + decoder | decoder stack only |
| Norm | post-norm LayerNorm | mostly pre-norm RMSNorm |
| Position | sinusoidal | mostly RoPE |
| Attention | MHA | commonly GQA / MQA |
| FFN | ReLU | commonly SwiGLU |
| Objective | conditional sequence generation | next-token prediction |

The original is the best place to learn what each of the three attention sites is
responsible for. The modern version is the best reference for understanding how
today's language models are trained and served.

<details markdown="1">
<summary><b>Deeper</b>: why training can be parallel while generation still cannot</summary>

During training the true target sequence is known, so it can be shifted right as a whole and fed to the decoder in one pass; the causal mask guarantees that position $t$ cannot see the future. During generation, input $t+1$ is exactly the prediction at step $t$, which does not exist yet, so decoding has to proceed step by step.

</details>

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh concept-title-h2" role="heading" aria-level="2">不要混淆：2017 原版不是现代 Decoder-only</div>

| | 2017 vanilla | 现代 decoder-only |
| --- | --- | --- |
| 主体 | encoder + decoder | decoder stack only |
| norm | post-norm LayerNorm | 多为 pre-norm RMSNorm |
| position | sinusoidal | 多为 RoPE |
| attention | MHA | 常见 GQA / MQA |
| FFN | ReLU | 常见 SwiGLU |
| 目标 | 条件序列生成 | next-token prediction |

原版最适合学习三处 attention 的职责；现代版本最适合理解今天语言模型的训练与推理。

**进阶：为什么训练能并行，生成仍不能并行**

训练时真实目标序列已知，可以把它整体右移后一次喂给 decoder，causal mask 保证位置 $t$ 看不到未来。生成时第 $t+1$ 个输入正是第 $t$ 个预测，尚未产生，因此只能逐步解码。

</div>
</section>

## Verify it: run the complete example

- Quick end-to-end run: [`../code/vanilla_demo.py`](../code/vanilla_demo.py)
- Full math and modern components: [Transformer architecture deep dive](../transformer.en.md)
- Attention forward pass without PyTorch: [`../code/sequence_numpy.py`](../code/sequence_numpy.py)

## Self-check

<div class="taste-check">
  <strong>After drawing the architecture diagram, ask yourself:</strong>
  <ol>
    <li>Where do Q/K/V come from in encoder self-attention, decoder self-attention, and cross-attention?</li>
    <li>Why must Q and K share the same final dimension, while $d_v$ may differ?</li>
    <li>Why divide by $\sqrt{d_k}$? What happens to softmax and the gradients without the scaling?</li>
    <li>If $Q=K$ is forced, what property does the raw score matrix have, and how is expressiveness limited?</li>
    <li>Why can Transformer training be parallel while generation is still token by token?</li>
    <li>If positional encoding is removed, exactly what information does the model lose?</li>
  </ol>
</div>

## Next

Continue to [Decoder-only](decoder-only.en.md) to see how conditional generation, dialogue, code, and many reasoning tasks are unified as autoregressive prediction over a single token stream.
