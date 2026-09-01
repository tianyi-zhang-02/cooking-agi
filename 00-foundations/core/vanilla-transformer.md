# Vanilla Transformer

**中文** · [English](vanilla-transformer.en.md)

> 阅读时间：约 15 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-zh><span>解决什么问题</span><strong>让所有位置并行交换信息，不再排队递归</strong></div>
    <div class="recipe-face" data-concept-en><span>Problem</span><strong>Exchange information across all positions in parallel, without recurrence</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-zh><span>前置知识</span><strong>token matrix · position · attention mask</strong></div>
    <div class="recipe-face" data-concept-en><span>Prerequisites</span><strong>token matrix · position · attention mask</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-zh><span>核心机制</span><strong>自注意力 · 交叉注意力 · 前馈网络 · 残差连接</strong></div>
    <div class="recipe-face" data-concept-en><span>Core mechanisms</span><strong>self-attention · cross-attention · FFN · residual connection</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-zh><span>常见错误</span><strong>混淆三处注意力，以及把因果掩码遮反</strong></div>
    <div class="recipe-face" data-concept-en><span>Common failure</span><strong>Mixing up the three attention sites or masking the wrong direction</strong></div>
  </div>
</div>

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>默认先读中文。每张卡片点右上角 <strong>English ↻</strong>，就能在原位置查看
  同一概念的等价英文；不需要回到页首，也不会丢掉阅读位置。</p>
</div>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

## 先交换位置信息，再逐位置变换

先忘掉那张塞满箭头的大框图。Transformer 一层其实只反复做两件事：**attention 去别的位置拿信息，FFN 留在当前位置加工信息。** 原版仍然是 encoder–decoder，但 recurrence 被彻底拿掉了。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en concept-title-h2" role="heading" aria-level="2">Move information, then transform it</div>

Forget the arrow-heavy architecture diagram for a moment. A Transformer layer
repeats two operations: **attention reads information from other positions; the FFN
transforms that information at the current position.** The original architecture is
still an encoder–decoder, but recurrence is removed completely.

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

## 一层只有两类计算

1. **Attention mixing**：不同 token 之间交换信息。
2. **Channel mixing / FFN**：每个 token 独立变换自己的通道。

外面再套 residual connection 与 normalization。堆很多层，本质上还是反复“出去找信息 → 回来加工信息”。这样记，比背整张结构图轻松很多。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en concept-title-h2" role="heading" aria-level="2">A layer has only two jobs</div>

1. **Attention mixing** exchanges information across token positions.
2. **Channel mixing / FFN** transforms each token's channels independently.

Residual connections and normalization wrap those two operations. Stacking layers
means repeating “go out and read → come back and transform.” This mental model is
more useful than memorizing the full block diagram.

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

## 三处 attention 在问不同的问题

| 位置 | Query | Key / Value | mask | 作用 |
| --- | --- | --- | --- | --- |
| encoder self-attention | source | source | padding | 双向理解输入 |
| decoder self-attention | target prefix | target prefix | causal + padding | 只能看已经生成的 token |
| decoder cross-attention | decoder state | encoder states | source padding | 从输入中取当前所需证据 |

scaled dot-product attention：

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

$M$ 是 mask：允许的位置加 0，禁止的位置加 $-\infty$。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en concept-title-h2" role="heading" aria-level="2">The three attention sites ask different questions</div>

| Site | Query | Key / value | Mask | Role |
| --- | --- | --- | --- | --- |
| encoder self-attention | source | source | padding | bidirectional source understanding |
| decoder self-attention | target prefix | target prefix | causal + padding | prevent future access |
| decoder cross-attention | decoder state | encoder states | source padding | retrieve source evidence |

All three use the same scaled dot-product formula:

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

The mask $M$ adds 0 at allowed positions and $-\infty$ at forbidden positions.

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

## 为什么这套做法突然可以做大

- **训练并行**：所有位置的 $Q/K/V$ 可以一次算出；
- **路径更短**：任意两个 token 一层 attention 就能直接交互；
- **结构统一**：self-attention 与 cross-attention 只是张量来源不同。

代价是 self-attention 的分数矩阵大小为 $T\times T$，标准实现的时间和显存随序列长度近似二次增长。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en concept-title-h2" role="heading" aria-level="2">Why the architecture scales</div>

- **Parallel training:** Q, K, and V for every position are computed at once.
- **Short paths:** any two tokens can interact through one attention layer.
- **One reusable structure:** self-attention and cross-attention differ mainly in
  where Q, K, and V come from.

The cost is a $T\times T$ self-attention score matrix, so standard attention grows
approximately quadratically in time and memory with sequence length.

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

## 缺失的信息：Attention 本身不知道顺序

Attention 本身不知道顺序。原版把固定 sinusoidal position encoding 加到 token embedding：

$$z_t = E[x_t] + PE_t$$

没有位置编码时，模型只能看到一袋 token；调换顺序只会让输出跟着调换。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en concept-title-h2" role="heading" aria-level="2">The missing information: attention does not know order</div>

Attention alone is permutation-equivariant: reordering the input merely reorders the
output. The original Transformer therefore adds fixed sinusoidal positional encoding
to each token embedding:

$$z_t=E[x_t]+PE_t.$$

Without position information, the model sees a bag of tokens rather than an ordered
sequence.

</div>
</section>

## 从 token ID 到下一个 token：完整走一遍

用英译中的小例子，把 2017 原版 Transformer 的整条前向路径串起来。先只看 shape，
再看每个模块的职责。

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. Source 怎样进入 encoder

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
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. How the source enters the encoder</div>

Tokenization first maps text to vocabulary indices:

```text
[I, love, you] -> [21, 135, 48]
```

An ID is only an index. Looking it up in the embedding matrix gives

$$X_{\text{emb}}\in\mathbb{R}^{S\times d_{\text{model}}}.$$

For $S=3$ and the original $d_{\text{model}}=512$, its shape is $(3,512)$. The
paper scales embeddings and adds fixed sinusoidal positions:

$$X=\sqrt{d_{\text{model}}}\,X_{\text{emb}}+PE,$$

$$PE(pos,2i)=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad
PE(pos,2i+1)=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

$PE$ is **added, not concatenated**, so the shape remains $(S,512)$. It gives
attention the order signal needed to distinguish a sequence from a permutation of
the same tokens. Dropout follows the embedding-plus-position sum.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. 一个 encoder layer 算什么

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
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. What one encoder layer computes</div>

The original model stacks six identical encoder layers:

```text
X -> multi-head self-attention -> Add & Norm -> FFN -> Add & Norm
```

Self-attention obtains all three interfaces from the current source states:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

With $h=8$ heads and $d_{\text{model}}=512$, each head has

$$d_k=d_v=512/8=64.$$

For one head, attention performs

$$S=QK^\top,\qquad
\widetilde S=\frac{S}{\sqrt{d_k}},\qquad
A=\operatorname{softmax}_{\text{key}}(\widetilde S),\qquad
O=AV.$$

For example, one head at one layer might give the `love` query
$A_{\text{love}}=[0.4,0.1,0.5]$, producing

$$o_{\text{love}}=0.4v_I+0.1v_{\text{love}}+0.5v_{\text{you}}.$$

These are local read weights for one head, not a literal statement that the whole
model semantically pays “40% attention.” The eight heads are concatenated and
projected:

$$\operatorname{MHA}(X)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_8)W^O.$$

The original is **post-norm**. Each sublayer output is dropped out, added to the
residual stream, and then normalized:

$$H=\operatorname{LN}\big(X+
\operatorname{Dropout}(\operatorname{MHA}(X))\big).$$

The same FFN is then applied independently at every position:

$$\operatorname{FFN}(h)=\operatorname{ReLU}(hW_1+b_1)W_2+b_2,$$

$$512\rightarrow2048\rightarrow512.$$

Attention exchanges information across tokens; the FFN transforms the channels of
each token without exchanging information across positions. The second residual is

$$Y=\operatorname{LN}\big(H+
\operatorname{Dropout}(\operatorname{FFN}(H))\big).$$

After six layers, the encoder produces memory

$$C=\operatorname{Encoder}(X)\in\mathbb{R}^{S\times512}.$$

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. Target 为什么要右移

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
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">3. Why the target is shifted</div>

For the target sequence

```text
labels:        [我, 爱, 你, <EOS>]
decoder input: [<BOS>, 我, 爱, 你]
```

the one-position shift teaches each position to predict its next token from the
prefix. During training, the full target is available in parallel; the causal mask
prevents future leakage.

Decoder inputs also receive embeddings, positions, and dropout. Source embedding,
target embedding, and output projection **can** be separate. The original paper,
however, used a shared vocabulary and tied both embedding matrices to the
pre-softmax projection, while multiplying embedding values by
$\sqrt{d_{\text{model}}}$. Separate weights are therefore not an invariant of the
original Transformer.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 4. 一个 decoder layer 为什么有三段

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
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">4. Why one decoder layer has three sublayers</div>

The decoder also has six layers, each with three sublayers:

```text
target states
  -> masked self-attention -> Add & Norm
  -> cross-attention       -> Add & Norm
  -> FFN                   -> Add & Norm
```

The first sublayer is target self-attention with a causal mask. For four positions,
the allowed score region is lower triangular:

$$
\begin{bmatrix}
s_{11}&-\infty&-\infty&-\infty\\
s_{21}&s_{22}&-\infty&-\infty\\
s_{31}&s_{32}&s_{33}&-\infty\\
s_{41}&s_{42}&s_{43}&s_{44}
\end{bmatrix}.
$$

Since $e^{-\infty}=0$, masked positions receive zero probability after softmax.
Position $i$ can read only positions $1$ through $i$:

$$H_1=\operatorname{LN}\big(Y+
\operatorname{Dropout}(\operatorname{MaskedMHA}(Y))\big).$$

Cross-attention is where target states first read the source:

$$Q=H_1W_Q,\qquad K=CW_K,\qquad V=CW_V.$$

Thus **Q comes from the decoder, while K and V come from encoder memory**. With
target length $T$ and source length $S$, its attention matrix is $(T,S)$ and need
not be square. A decoder state preparing to generate `爱` can retrieve the source
value aligned with `love`:

$$H_2=\operatorname{LN}\big(H_1+
\operatorname{Dropout}(\operatorname{CrossAttention}(H_1,C))\big).$$

The position-wise FFN completes the layer:

$$H_3=\operatorname{LN}\big(H_2+
\operatorname{Dropout}(\operatorname{FFN}(H_2))\big).$$

After six decoder layers,

$$D\in\mathbb{R}^{T\times512}.$$

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 5. 从 decoder state 变成词表概率

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
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">5. From decoder states to vocabulary probabilities</div>

Each 512-dimensional decoder state is projected to the target vocabulary. For
$V=30{,}000$:

$$Z=DW_{\text{vocab}}+b,\qquad
W_{\text{vocab}}\in\mathbb{R}^{512\times30{,}000},$$

so $Z\in\mathbb{R}^{T\times30{,}000}$. Each row contains logits; softmax turns
them into a conditional distribution:

$$p(y_t\mid y_{<t},x)=\operatorname{softmax}(Z_t).$$

Training compares this distribution with the target token using cross-entropy.
Inference uses greedy decoding, beam search, or sampling to select a token, feeds it
back into the decoder, and continues. Softmax provides probabilities; it does not
itself require argmax.

The whole path is

$$
\text{source IDs}\rightarrow C\rightarrow
\underbrace{p(y_1\mid x),\ p(y_2\mid y_1,x),\ldots}_{
\text{masked self-attention reads the prefix; cross-attention reads the source}}.
$$

</div>
</section>

<section class="concept-card concept-card-major" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

## 不要混淆：2017 原版不是现代 Decoder-only

| | 2017 vanilla | 现代 decoder-only |
| --- | --- | --- |
| 主体 | encoder + decoder | decoder stack only |
| norm | post-norm LayerNorm | 多为 pre-norm RMSNorm |
| position | sinusoidal | 多为 RoPE |
| attention | MHA | 常见 GQA / MQA |
| FFN | ReLU | 常见 SwiGLU |
| 目标 | 条件序列生成 | next-token prediction |

原版最适合学习三处 attention 的职责；现代版本最适合理解今天语言模型的训练与推理。

<details markdown="1">
<summary><b>进阶</b>：为什么训练能并行，生成仍不能并行</summary>

训练时真实目标序列已知，可以把它整体右移后一次喂给 decoder，causal mask 保证位置 $t$ 看不到未来。生成时第 $t+1$ 个输入正是第 $t$ 个预测，尚未产生，因此只能逐步解码。

</details>

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en concept-title-h2" role="heading" aria-level="2">Do not conflate the 2017 architecture with a modern decoder-only LLM</div>

| | 2017 vanilla | Modern decoder-only |
| --- | --- | --- |
| Body | encoder + decoder | decoder stack only |
| Normalization | post-norm LayerNorm | usually pre-norm RMSNorm |
| Position | sinusoidal encoding | usually RoPE |
| Attention | MHA | often GQA / MQA |
| FFN | ReLU | often SwiGLU |
| Objective | conditional sequence generation | next-token prediction |

The original architecture is the cleanest place to learn the roles of its three
attention sites. Modern decoder-only models are the right reference for today's LLM
training and inference.

Training remains parallel because the complete target sequence is known and shifted
into the decoder at once; the causal mask blocks future information. Generation is
still sequential because token $t+1$ cannot be used before token $t$ has been
predicted.

</div>
</section>

## 动手验证：运行完整示例

- 快速跑通：[`../code/vanilla_demo.py`](../code/vanilla_demo.py)
- 完整数学与现代组件：[Transformer 架构深拆](../transformer.md)
- 无 PyTorch attention 前向：[`../code/sequence_numpy.py`](../code/sequence_numpy.py)

## 自检

<div class="taste-check">
  <strong>画完结构图后，再问自己：</strong>
  <ol>
    <li>encoder self-attention、decoder self-attention 和 cross-attention 的 Q/K/V 分别来自哪里？</li>
    <li>为什么 Transformer 训练可以并行，生成却仍然逐 token？</li>
    <li>如果拿掉位置编码，模型具体失去了什么信息？</li>
  </ol>
</div>

## 继续读

进入 [Decoder-only](decoder-only.md)，看怎样把条件生成、对话、代码与很多推理任务统一为一条 token stream 上的自回归预测。
