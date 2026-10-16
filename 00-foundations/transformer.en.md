# The Transformer architecture

[中文](transformer.md) · **English**

> Reading time: ~12 min · Level: intro → advanced · Last reviewed: 2026-10

<div class="lesson-recipe advanced">
  <div><span>What we are dissecting</span><strong>the Transformer, from block diagram back down to matrix operations and properties you can check</strong></div>
  <div><span>Prerequisites</span><strong>matrix multiplication · softmax · residuals · causal LM</strong></div>
  <div><span>Main mechanism</span><strong>Q/K/V · norm · RoPE · GQA · KV cache</strong></div>
  <div><span>What you must be able to prove</span><strong>the implementation satisfies causality, relative position, and cache equivalence</strong></div>
</div>

## Attention gathers context; the FFN transforms each position {#attention-gathers-context-the-ffn-transforms-each-position}

The Transformer is one concrete way to build the "learned coordinate transform $\phi$" from [the previous page](from-linear-to-neural.en.md): **attention moves information across positions; the FFN processes it at a single position.** The two alternate in a stack, and at the end it is still a linear classifier that reads out the answer.

- **Attention**: lets each position read information from other positions according to relevance.
- **FFN**: transforms features independently inside each position.
- **Residual connection**: keeps the original state and adds each sublayer's change on top of it.
- **Stacking N layers**: repeats cross-position communication and per-position transformation, building up layer by layer a representation usable for prediction.

This page connects the parts; each part's derivation has its own note: scaling, masks and Q/K/V in [multi-head attention](core/multi-head-attention.en.md), where the norm goes in [normalization](core/normalization.en.md), and SwiGLU in [FFN and gates](core/ffn-and-gates.en.md). What this page owns is the three uses of Q/K/V, the RoPE code, one Noam-schedule pitfall, and how to prove with tests that a from-scratch implementation is right.

## Scaled dot-product attention {#scaled-dot-product-attention}

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Write out the full shapes first:

$$Q\in\mathbb{R}^{T_q\times d_k},\qquad
K\in\mathbb{R}^{T_k\times d_k},\qquad
V\in\mathbb{R}^{T_k\times d_v}.$$

$Q$ and $K$ must share their last dimension for the dot product; $V$ only needs the same number of tokens $T_k$ as $K$, and its feature dimension $d_v$ may differ. Standard multi-head attention sets $d_v=d_k=d_{\text{model}}/h$ so that concatenation is convenient; that is a common design, not a requirement of the formula. The scale uses each head's $d_k$: with $d_{\text{model}}=768$ and $h=12$, $d_k=64$, and the divisor is $\sqrt{64}$, not $\sqrt{768}$.

Broken down to a single query $\mathbf{q}_i$:

$$\alpha_{ij} = \frac{\exp\!\big(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k}\big)}{\sum_{j'} \exp\!\big(\mathbf{q}_i^\top \mathbf{k}_{j'} / \sqrt{d_k}\big)}, \qquad \mathbf{o}_i = \sum_j \alpha_{ij}\, \mathbf{v}_j$$

That is: **use similarity as the weights and take a weighted average of the values.** Every row of $\alpha_{ij}$ sums to 1.

### Softmax is a differentiable allocation, not an argmax {#softmax-is-a-differentiable-allocation-not-an-argmax}

Softmax maps arbitrary real scores to positive weights that sum to one:

$$\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.$$

For example, $[2,1,0]$ becomes approximately $[0.665,0.245,0.090]$. It does not keep only the top score; it lets one query read several positions at once. Attention applies softmax **row by row over the last dimension** of the score matrix: row $i$ answers "how much weight should query $i$ give to each key $j$". In one line: $QK^\top$ decides **where to read**; multiplying the weights by $V$ decides **in what proportions to combine what was read**.

Three common follow-ups are each worked through in [multi-head attention](core/multi-head-attention.en.md):

- **Why divide by $\sqrt{d_k}$**: under the simplifying assumption of independent, unit-variance components, the dot product has variance $d_k$; dividing by $\sqrt{d_k}$ brings it back to 1, so the softmax does not sharpen early just because the width grew. The variance derivation, the `[8, −8]` example, and why the gradient shrinks are in [that section](core/multi-head-attention.en.md#why-divide-by-sqrtd_k).
- **Where the mask goes**: set blocked positions to `-inf` before the softmax; setting them to 0 still assigns weight, and zeroing after the softmax without renormalizing is not equivalent either. The two-score example is in [the six common mistakes](core/multi-head-attention.en.md#the-six-places-a-from-scratch-implementation-goes-wrong).
- **What $Q=K$ does**: the pre-scaling scores $QQ^\top$ are symmetric, the row-wise softmax is generally not, and attention does not stop working; see the [interview questions](core/multi-head-attention.en.md#common-interview-questions).

### Do the three projection matrices have any "actual meaning"? {#do-the-three-projection-matrices-have-any-actual-meaning}

For the self-attention input $X$, the model learns

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

The answer has two levels:

- **A single parameter value or a single coordinate usually has no fixed human meaning.** With a different random seed, the coordinate axes and weight values can be completely different while the model still implements similar behavior; for any invertible matrix $R$, replacing $Q,K$ with $QR$ and $KR^{-\top}$ leaves the scores unchanged.
- **The computational roles carried by the three matrices are meaningful.** $W_Q$ produces queries, $W_K$ produces the keys used for matching, and $W_V$ decides what content is actually transmitted after a match. Only with separate $W_Q$ and $W_K$ can $W_QW_K^\top$ be asymmetric, so that "$i$ queries $j$" and "$j$ queries $i$" can differ; $W_V$ decouples "why it was found" from "what to take from it".

## One module, three uses: changing where Q/K/V come from {#one-module-three-uses-changing-where-qkv-come-from}

This is the part of the original paper's (2017) encoder-decoder structure most worth keeping an eye on. `self_attn(x, x, x)` and `cross_attn(x, memory, memory)` are the same class; only the three tensors fed in differ:

| Use | Q from | K, V from | mask | Attention shape |
| --- | --- | --- | --- | --- |
| Encoder self-attention | src | src | blocks padding only, **bidirectional** | $(B,h,S,S)$ |
| Decoder self-attention | tgt | tgt | padding **∨** causal | $(B,h,T,T)$ |
| **Cross-attention** | **tgt** | **memory** | blocks src padding | $(B,h,T,S)$ ← not square |

![where Q, K and V come from at each of the three attention sites](assets/attention-sites.en.svg)

Cross-attention is the only place where the encoder and decoder exchange information: at every generation step, each decoder layer uses its current state as the query and looks things up in the encoder's output.

[`code/vanilla_demo.py`](code/vanilla_demo.py) trains a sequence-reversal task and prints cross-attention weights. The following is an intuitive reverse-alignment sketch, not the only valid pattern: encoder positions already mix context, so correct outputs do not require a particular head to form an anti-diagonal.

```
        1  2  3  4  5  6  7  8   <- source (encoder)
  BOS                        @
    1                     @
    2                  @
    3               @
    4            #  :
    5         @
    6      @
    7   @
  ^ decoder step
```

Denser characters mean larger weights: `@` is close to 1, and the `#` and `:` at step 4 show the weight split between two neighboring positions. Use the script's separate generation check to evaluate outputs rather than judging only this picture. Even a perfect small batch does not establish generalization to arbitrary lengths or unseen distributions.

## Multi-head attention: why not one big attention {#multi-head-attention-why-not-one-big-attention}

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O, \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

This equation follows the original paper's interface notation: Q/K/V here are the three **pre-projection inputs**, all equal to X in self-attention. It does not project the already-projected tensors from the opening section a second time.

One head has one set of softmax weights; splitting a fixed width into $h$ heads gives $h$ sets of attention relations at a similar projection cost. The implementation does not need $h$ groups of small matrices: one $(d_{\text{model}}, d_{\text{model}})$ projection reshaped into $h$ heads needs only a single GEMM. The reshape order is the most common from-scratch mistake; it, and why multiple heads are not "a bigger width", are covered in [multi-head attention](core/multi-head-attention.en.md).

## Training stability: why post-norm depends on warmup {#training-stability-why-post-norm-depends-on-warmup}

The paper writes

$$\mathbf{x} \leftarrow \text{LayerNorm}\big(\mathbf{x} + \text{Sublayer}(\mathbf{x})\big)$$

**Layer normalization sits outside the residual addition** (post-norm). Many later models use pre-norm instead:

$$\mathbf{x} \leftarrow \mathbf{x} + \text{Sublayer}\big(\text{Norm}(\mathbf{x})\big)$$

![the residual path under post-norm and pre-norm](assets/transformer-block.en.svg)

The change affects the gradient path, not just notation: Pre-LN keeps an identity term on the residual trunk and is often easier to train, but initialization, depth, and learning rate still matter, and warmup does not automatically become unnecessary. The Jacobian derivation and the warmup result of [Xiong et al.](https://arxiv.org/abs/2002.04745) are in question 3 of [normalization](core/normalization.en.md#four-concepts-answer-first-then-go-deeper).

The original Noam schedule is

$$\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min\big(t^{-0.5},\; t \cdot t_{\text{warmup}}^{-1.5}\big)$$

It increases linearly, then decays with the inverse square root of the step. Neither “six layers inevitably explode” nor “training cannot start without this schedule” follows.

> Field note: `LambdaLR` **multiplies base_lr by** the lambda. Set Adam's `lr` to 0 and attach a Noam schedule, and the learning rate stays 0 forever, while dropout noise makes the loss look like it is still moving. This pitfall is common.

## Positional encoding: from sinusoids to RoPE {#positional-encoding-from-sinusoids-to-rope}

Without positional information or an order-dependent mask, self-attention is **permutation-equivariant**: reordering inputs reorders outputs. A causal mask already constrains earlier versus later positions; positional encodings further express position and distance. The entire causal model should not be described as a bag of words.

### Original: fixed sinusoids {#original-fixed-sinusoids}

$$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

**Added** to the embedding, not concatenated. Wavelengths form a geometric sequence from $2\pi$ to just below $10000\cdot 2\pi$, like clocks running at different speeds. For a fixed offset k, $PE_{pos+k}$ is a linear transform of $PE_{pos}$, making relative offsets representable without guaranteeing that the model learns to use them.

### Now: RoPE {#now-rope}

Nothing is added to the input; instead, **$q$ and $k$ are rotated at every layer**. Pair up the channels of $\mathbf{q}$ two by two as complex numbers, with rotation angle $m\theta_i$ at position $m$:

$$\tilde{\mathbf{q}}_m = R_m \mathbf{q}, \qquad R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \ \ (\text{per channel pair})$$

With fixed q/k content and rotary frequencies, the **explicit positional factor** introduced by RoPE depends only on $n-m$. Actual q/k also depend on context and masks, so the model does not simply operate on distances. Common implementations do not apply this rotation to V; its contextual representation can still carry positional information.

<details markdown="1">
<summary><b>Deeper</b>: where the relative property comes from</summary>

Rotation matrices are orthogonal, and rotations in the same plane add: $R_m^\top = R_{-m}$ and $R_a R_b = R_{a+b}$. So

$$\langle R_m\mathbf{q},\; R_n\mathbf{k}\rangle
= \mathbf{q}^\top R_m^\top R_n \mathbf{k}
= \mathbf{q}^\top R_{n-m}\mathbf{k}$$

$m$ and $n$ appear only as a difference in this rotation identity. That is not a guarantee of long-context generalization: longer inputs introduce unfamiliar phases, distances, and attention distributions, requiring dedicated extension methods and evaluation.

It is also why you cannot rotate only $\mathbf{q}$ and not $\mathbf{k}$: then $R_m^\top$ has no $R_n$ to pair with, and absolute position no longer cancels.

</details>

```python
def apply_rope(x, cos, sin):
    """x: (B, H, T, d)   cos/sin: (T, d/2)"""
    x1, x2 = x.chunk(2, dim=-1)                     # split-half (GPT-NeoX / HF-converted Llama)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)
```

⚠️ There are two pairing conventions: split-half (channel $i$ pairs with $i + d/2$; GPT-NeoX and Llama after Hugging Face conversion) and adjacent pairs (channel $2i$ with $2i+1$; the RoFormer paper and [Meta's original Llama code](https://github.com/meta-llama/llama/blob/main/llama/model.py)). They differ by a channel permutation, so **weights are not interchangeable**: the Hugging Face converter permutes $W_Q$/$W_K$ to fit split-half. This is a classic pitfall when converting models; [RoPE and long context](deep-dives/position-and-context.en.md#rotary-layout) works through the numbers.

## What else the modern decoder-only model changed {#what-else-the-modern-decoder-only-model-changed}

This table describes the choices in the teaching implementation, not a universal configuration for all modern models.

| | vanilla (2017) | This decoder-only implementation |
| --- | --- | --- |
| Structure | encoder + decoder | decoder only |
| Normalization | post-norm LayerNorm | pre-norm RMSNorm |
| Position | sinusoidal, added to the input | RoPE, rotating q/k at every layer |
| Attention | MHA (each of the $h$ heads has its own K/V) | GQA (fewer K/V heads) |
| FFN | ReLU, $4d$ | SwiGLU, $\tfrac{8}{3}d$ |
| Inference | autoregressive; self/cross-attention K/V can be cached | preallocated KV cache |

Newer models change a few more things beyond this table: QK-norm, MLA, and linear attention in some layers; see [decoder-only](core/decoder-only.en.md#1-the-modern-block-at-a-glance).

- **RMSNorm** drops mean subtraction and the bias and only rescales by the root mean square (formula, and whether its success proves centering useless, in [normalization](core/normalization.en.md)). This code computes the mean square in fp32 and casts back to the activation dtype to reduce low-precision reduction error; a mean square is not a variance.
- **SwiGLU** replaces ReLU with a gated projection; the three matrices have about $3dh$ parameters, and choosing $h=\frac83d$ matches a two-matrix FFN of hidden width $4d$. The worked numbers are in the "matched budget" section of [FFN and gates](core/ffn-and-gates.en.md).

**GQA** lets each group of $n_{\text{rep}}$ query heads share one set of K/V heads. For one sequence with the same configuration at every layer, raw KV-cache bytes are

$$2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot T \cdot \text{sizeof(dtype)}$$

With other conditions fixed, reducing $n_{\text{kv}}$ from 32 to 8 makes the **raw KV cache** one quarter as large. It does not divide total model memory or latency by four. See [KV cache and inference cost](deep-dives/kv-cache-and-inference.en.md) for worked accounting.

## Hands-on: implement it from scratch and check the common mistakes {#hands-on-implement-it-from-scratch-and-check-the-common-mistakes}

Neither of the two implementations in [`code/`](code/) calls `nn.MultiheadAttention` or `F.scaled_dot_product_attention`; they use only `nn.Linear` and bare tensor operations:

- [`vanilla.py`](code/vanilla.py) — the original 2017 encoder-decoder, with the Noam schedule and label smoothing
- [`model.py`](code/model.py) — modern decoder-only (RMSNorm + RoPE + GQA + SwiGLU + KV cache)

[`test_model.py`](code/test_model.py) verifies, one by one, the four places a from-scratch implementation most easily gets wrong:

```
  causality:     change token 9 -> logits at the first 8 positions differ by 0.0 (exactly 0)
  kv cache:      incremental decode vs one-shot forward, max error 3.6e-07
  rope relative: score(5,2) = score(20,17) = +5.6092, score(20,10) = +0.4579
  init loss:     4.19  vs  ln(V) = 4.16
```

![what prefill and a single decode step each compute](assets/kv-cache.en.svg)

With a cache, query positions start at `cache.pos + i`, while keys start at zero and end at `cache.pos + T - 1`. An existing prefix therefore gives a non-square mask $(T,S)$; RoPE is sliced from `cache.pos` too. In this implementation `cache.pos` advances once per model forward, after the entire layer loop. Equivalence checks require unchanged parameters, prefix, positions, and visibility, with dropout disabled; changed weights or truncated caches are not directly comparable to full forward passes.

Sources: [original Transformer](https://arxiv.org/abs/1706.03762), [RoFormer](https://arxiv.org/abs/2104.09864), [RMSNorm](https://arxiv.org/abs/1910.07467), [GLU variants](https://arxiv.org/abs/2002.05202).

## Self-check {#self-check}

<div class="taste-check advanced">
  <strong>After finishing this chapter, you should be able to explain at least four key questions:</strong>
  <ol>
    <li>Why is the attention score divided by $\sqrt{d_k}$?</li>
    <li>Once pre-norm moves the norm inside the residual branch, what direct path does the gradient gain?</li>
    <li>Why does RoPE's relative property require rotating both Q and K?</li>
    <li>How do you prove that a KV cache is implemented correctly, rather than the generations merely "not looking broken"?</li>
  </ol>
</div>

## Where to read next {#where-to-read-next}

- [From linear models to neural networks](from-linear-to-neural.en.md) — why the last layer is always a linear classifier
- [Post-training](../05-post-training/README.en.md) — how these parameters keep being changed afterwards
- [Systems overview](../06-systems/README.en.md) — where this sits in the whole system

## Quick learning: what does a Transformer actually do? {#quick-learning-what-does-a-transformer-actually-do}

<details class="interview" markdown="1">
<summary>Learn the spine first, then open the standard answer and the deep dive</summary>

**Quick memory**

- Attention: exchanges information between tokens.
- FFN: applies a nonlinear transformation independently at every token.
- Residual + Norm: lets the two kinds of computation stack deep and stay stable.
- Position information: tells the model about order and relative distance.

**Interview answer**

> A Transformer block alternates token mixing and channel mixing. Self-attention uses QK similarity to aggregate V across sequence positions; the FFN expands, activates, and contracts each position independently. The residual path preserves the old representation and provides a gradient path, Norm controls the scale of sublayer inputs, and positional encoding supplies the ordering information that attention itself lacks.

<details markdown="1">
<summary><b>Deep dive</b>: why is the final layer a linear head, yet the whole model is not linear?</summary>

Attention weights depend on the input:

$$
A(X)=\operatorname{softmax}\left(\frac{XW_Q(XW_K)^\top}{\sqrt{d_k}}\right).
$$

So $A(X)XW_V$ is already an input-dependent nonlinear map, and the FFN activation adds another layer of nonlinearity. The final linear head only reads logits out of the learned hidden state; it does not turn the dozens of layers before it back into a linear model.

</details>

</details>

## Reference papers {#reference-papers}

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — why Post-LN depends on warmup, and why Pre-LN could drop it in the paper's setting
- [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE
- [GQA](https://arxiv.org/abs/2305.13245) — grouped-query attention
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU
