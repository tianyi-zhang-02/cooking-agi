# Vanilla Transformer

[中文](vanilla-transformer.md) · **English**

> Reading time: ~15 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>Problem</span><strong>Exchange information across all positions in parallel, without recurrence</strong></div>
  <div><span>Prerequisites</span><strong>token matrix · position · attention mask</strong></div>
  <div><span>Core mechanisms</span><strong>self-attention · cross-attention · FFN · residual connection</strong></div>
  <div><span>Common failure</span><strong>Mixing up the three attention sites or masking the wrong direction</strong></div>
</div>

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

## A layer has only two jobs

1. **Attention mixing** exchanges information across token positions.
2. **Channel mixing / FFN** transforms each token's channels independently.

Residual connections and normalization wrap those two operations. Stacking many
layers still means repeating “go out and find information → come back and process
it.” Remembering it this way is much lighter than memorizing the full block diagram.

## The three attention sites ask different questions

| Site | Query | Key / Value | Mask | Role |
| --- | --- | --- | --- | --- |
| encoder self-attention | source | source | padding | understand the input bidirectionally |
| decoder self-attention | target prefix | target prefix | causal + padding | see only tokens that were already generated |
| decoder cross-attention | decoder state | encoder states | source padding | fetch the evidence needed right now from the input |

All three use the same scaled dot-product attention:

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

$M$ is the mask: it adds 0 at allowed positions and $-\infty$ at forbidden positions.

## Why the architecture scales

- **Parallel training:** $Q/K/V$ for every position are computed at once.
- **Shorter paths:** any two tokens can interact directly through one attention layer.
- **One uniform structure:** self-attention and cross-attention differ only in where
  the tensors come from.

The cost is a $T\times T$ self-attention score matrix, so the time and memory of a
standard implementation grow approximately quadratically with sequence length.

## The missing information: attention does not know order

Attention by itself does not know order. The original Transformer adds a fixed
sinusoidal positional encoding to each token embedding:

$$z_t = E[x_t] + PE_t$$

Without positional encoding, the model sees only a bag of tokens: permuting the input
merely permutes the output in the same way.

## From token IDs to the next token: the complete path

Use a tiny English-to-Chinese example to connect the entire forward path of the 2017
Transformer. Look only at shapes first, then at each module's job.

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
