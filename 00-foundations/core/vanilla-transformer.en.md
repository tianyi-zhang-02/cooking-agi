# Vanilla Transformer

[中文](vanilla-transformer.md) · **English**

> Reading time: ~15 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>What we are making</span><strong>Parallel information exchange across sequence positions</strong></div>
  <div><span>Prerequisites</span><strong>token matrix · position · attention mask</strong></div>
  <div><span>Core technique</span><strong>self-attention · cross-attention · FFN · residual</strong></div>
  <div><span>Most common failure</span><strong>mixing up attention sites or masking the wrong direction</strong></div>
</div>

## The original encoder–decoder structure

The original Transformer is an encoder–decoder. Attention moves information across positions, FFNs transform channels independently, and removing recurrence makes full-sequence training parallel.

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

## Three attention sites

| Site | Query | Key / value | Mask | Role |
| --- | --- | --- | --- | --- |
| encoder self-attention | source | source | padding | bidirectional source understanding |
| decoder self-attention | target prefix | target prefix | causal + padding | prevent future access |
| decoder cross-attention | decoder state | encoder states | source padding | retrieve source evidence |

Attention shortens the path between arbitrary positions to one layer and parallelizes training, but the $T\times T$ score matrix makes standard self-attention approximately quadratic in sequence length.

## From token IDs to the next token: the complete path

Use a tiny English-to-Chinese example to connect every block in the 2017 model.

### 1. Source input

Tokenization first maps text to vocabulary indices:

```text
[I, love, you] -> [21, 135, 48]
```

An ID is only an index. Looking it up in the embedding matrix gives

$$X_{\text{emb}}\in\mathbb{R}^{S\times d_{\text{model}}}.
$$

For $S=3$ and the original $d_{\text{model}}=512$, its shape is $(3,512)$. The
paper scales embeddings and adds fixed sinusoidal positions:

$$X=\sqrt{d_{\text{model}}}\,X_{\text{emb}}+PE,$$

$$PE(pos,2i)=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad
PE(pos,2i+1)=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

$PE$ is **added, not concatenated**, so the shape remains $(S,512)$. It gives
attention the order signal needed to distinguish a sequence from a permutation of
the same tokens. Dropout follows the embedding-plus-position sum.

### 2. One encoder layer

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

### 3. Why the target is shifted

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

### 4. One decoder layer

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

### 5. Vocabulary logits and probabilities

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

The 2017 model used sinusoidal positions, post-norm LayerNorm, MHA, and ReLU FFNs. Modern decoder-only models more often use RoPE, pre-norm RMSNorm, GQA, and SwiGLU. Do not collapse these into one architecture.

Run [`../code/vanilla_demo.py`](../code/vanilla_demo.py), then use [the full Transformer deep dive](../transformer.en.md).

## Self-check

<div class="taste-check">
  <strong>After drawing the architecture, answer:</strong>
  <ol>
    <li>Where do Q, K, and V come from at each attention site?</li>
    <li>Why is training parallel while generation remains sequential?</li>
    <li>What information disappears without positional encoding?</li>
  </ol>
</div>

Next: [Decoder-only](decoder-only.en.md).
