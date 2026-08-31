# The Transformer architecture

[中文](transformer.md) · **English**

> Reading time: ~8 min · Level: intro → advanced · Last reviewed: 2026-08
>
> The main line is enough on its own. Blocks marked **deeper** hold derivations and edge cases; skipping them costs you nothing.

<div class="lesson-recipe advanced">
  <div><span>What we are dissecting</span><strong>Transformer blocks down to matrices and invariants</strong></div>
  <div><span>Prerequisites</span><strong>matrix multiplication · softmax · residuals · causal LM</strong></div>
  <div><span>Main mechanism</span><strong>Q/K/V · normalization · RoPE · GQA · KV cache</strong></div>
  <div><span>Evidence to demand</span><strong>causality, relative position, and cache equivalence</strong></div>
</div>

## In one sentence

The Transformer is one concrete way to build the learned coordinate transform from [the previous page](from-linear-to-neural.en.md). **Attention moves information across positions; the FFN processes each position on its own.** The two alternate for $N$ layers, and a linear classifier reads out the answer at the end.

## Scaled dot-product attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Write the full shapes and the reason for $d_k$ becomes mechanical:

$$Q\in\mathbb{R}^{T_q\times d_k},\qquad
K\in\mathbb{R}^{T_k\times d_k},\qquad
V\in\mathbb{R}^{T_k\times d_v}.$$

Every entry of $QK^\top$ is a dot product over the **$d_k$ components** of one query and
one key, so the scale is $\sqrt{d_k}$. Matrix multiplication only requires Q and K to
share their last dimension. V must share the key sequence length $T_k$, but its feature
dimension $d_v$ may differ. Standard multi-head attention usually chooses
$d_v=d_k=d_{\text{model}}/h$ for convenient concatenation; that is a design convention,
not a mathematical requirement. With $d_{\text{model}}=768$ and $h=12$, each head has
$d_k=64$, so divide by $\sqrt{64}$, not $\sqrt{768}$.

Per query: $\alpha_{ij} = \text{softmax}_j(\mathbf{q}_i^\top\mathbf{k}_j/\sqrt{d_k})$, then $\mathbf{o}_i = \sum_j \alpha_{ij}\mathbf{v}_j$. A weighted average of the value vectors, with similarity as the weights.

### Softmax is a differentiable allocation, not an argmax

Softmax maps arbitrary real scores to positive weights that sum to one:

$$\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.$$

For example, $[2,1,0]$ becomes approximately $[0.665,0.245,0.090]$. It does not keep only
the maximum; one query may read several positions at once. Attention applies softmax
**row by row over the last dimension**: row $i$ answers how query $i$ allocates weight over
keys $j$. Therefore

$$A=\operatorname{softmax}_{j}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right),
\qquad O=AV,\qquad \mathbf{o}_i=\sum_j A_{ij}\mathbf{v}_j.$$

In one line: $QK^\top$ decides **where to read**; $AV$ decides **how to combine what was read**.

### Why $\sqrt{d_k}$

For unit-variance $q, k$, the dot product of $d_k$ terms has variance $d_k$, i.e. a spread of $\pm 8$ at $d_k = 64$. Softmax at that scale is nearly one-hot, and its Jacobian

$$\frac{\partial\,\text{softmax}(z)_i}{\partial z_j} = \alpha_i(\delta_{ij} - \alpha_j)$$

collapses toward the zero matrix — no gradient. Dividing by $\sqrt{d_k}$ pulls the variance back to 1.

<details markdown="1">
<summary><b>deeper</b>: why "close to one-hot" means "no gradient"</summary>

With $\alpha_i \to 1$ and the rest $\to 0$, the diagonal $\alpha_i(1-\alpha_i) \to 0$ and the off-diagonal $-\alpha_i\alpha_j \to 0$: the whole Jacobian goes to zero. The forward pass still produces sensible output while nothing flows backwards.

It is the same saturation as the sigmoid on [the previous page](from-linear-to-neural.en.md), where $\sigma'(z) = \sigma(1-\sigma)$ dies at both ends. Softmax is its multi-class generalisation and inherits the behaviour exactly.

</details>

### Do the three projection matrices have intrinsic meaning?

For self-attention input $X$, the model learns

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

The right answer has two levels:

- **An individual parameter or coordinate usually has no fixed human meaning.** A new random
  seed may produce completely different axes and numbers while implementing similar behavior.
- **The computational roles of the three matrices do matter.** $W_Q$ emits queries, $W_K$
  emits matchable keys, and $W_V$ chooses the content transmitted after a match.

Why separate Q and K? For self-attention over one sequence, forcing $W_Q=W_K=W$ gives

$$S=XWW^\top X^\top,$$

which is symmetric, so the raw compatibility score must satisfy $S_{ij}=S_{ji}$. Separating
them gives

$$S=XW_QW_K^\top X^\top,$$

where $W_QW_K^\top$ need not be symmetric, allowing directional raw compatibility. Three
distinctions matter:

1. symmetry applies to the **raw scores before masks and softmax**;
2. row-wise softmax has a different normalizer per row, so attention weights are generally
   not symmetric;
3. a causal mask also destroys symmetry.

Why separate V? Q/K are the addressing interface; V is the content being read. Two positions
may match on one set of features while a different set should be transmitted. $W_V$ decouples
“why this position was found” from “what to take from it.” The database analogy is useful, but
these are not three manually named semantic fields.

More formally, the internal coordinates are non-identifiable. For any invertible $R$, let

$$Q'=QR,\qquad K'=KR^{-\top}.$$

Then $Q'K'^\top=QK^\top$: the basis can change while the function remains identical. This is
why interpreting an isolated entry such as $W_Q[17,42]$ is usually meaningless. What matters
is the function implemented by the whole projection, its causal effect on outputs, and the
interface constraints among Q, K, and V.

```python
def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))   # before softmax, not after
    attn = scores.softmax(dim=-1)
    return attn @ v, attn
```

<details markdown="1">
<summary><b>follow-up</b>: what if $Q=K$?</summary>

Before scaling, the score matrix is the Gram matrix $QQ^\top$, so it is symmetric and
positive semidefinite. Row-wise softmax generally makes it **non-symmetric**, however,
because each row has a different normalizer.

- If all queries are identical, every score is identical and every attention row is uniform.
- If queries are mutually orthogonal with equal norm, the diagonal wins; attention is close
  to the identity only when that norm is large relative to the softmax temperature.
- In the general case, a position tends to attend more to vectors similar to itself, but it
  is not guaranteed to attend only to itself.

Thus $Q=K$ does not break attention. It changes matching between two projected spaces into
similarity inside one space; row-wise softmax and V still determine the output.

</details>

## One module, three uses

The single most important thing about the 2017 encoder-decoder. `self_attn(x,x,x)` and `cross_attn(x, memory, memory)` are the same class with different tensors:

| Use | Q from | K, V from | Mask | Shape |
| --- | --- | --- | --- | --- |
| Encoder self-attention | src | src | padding only, **bidirectional** | $(B,h,S,S)$ |
| Decoder self-attention | tgt | tgt | padding **∨** causal | $(B,h,T,T)$ |
| **Cross-attention** | **tgt** | **memory** | src padding | $(B,h,T,S)$ — not square |

![where Q, K and V come from at each attention site](assets/attention-sites.svg)

Cross-attention is the only place the two towers touch. [`code/vanilla_demo.py`](code/vanilla_demo.py) trains it to reverse a sequence, where the correct alignment is known in advance, so you can read it straight off the matrix — an anti-diagonal, 64/64 sequences exact.

## Multi-head

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \quad \text{MultiHead} = \text{Concat}(\text{head}_1..\text{head}_h)W^O$$

One attention computes one similarity and returns one average. Splitting into $h$ heads of $d_k = d_\text{model}/h$ lets different heads track different relations at no extra cost. In code it is one $(d_\text{model}, d_\text{model})$ projection reshaped into heads — mathematically identical, a single GEMM.

## Post-norm and the warmup it requires

The paper does $\mathbf{x} \leftarrow \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$ — the norm sits **on the residual highway**. Everything modern does $\mathbf{x} \leftarrow \mathbf{x} + \text{Sublayer}(\text{Norm}(\mathbf{x}))$ instead.

![the residual path under post-norm and pre-norm](assets/transformer-block.svg)

This is why the original recipe needs the Noam schedule

$$\text{lr}(t) = d_\text{model}^{-0.5}\min\big(t^{-0.5},\; t\cdot t_\text{warmup}^{-1.5}\big)$$

Warmup is not a tuning trick under post-norm; it is what keeps the first few thousand steps from diverging. Pre-norm gives the gradient a path that skips every norm, which is what made constant learning rates safe.

> Field note: `LambdaLR` multiplies the **base** lr by your lambda. Set Adam's `lr=0` and attach a Noam schedule and the learning rate stays 0 forever, while dropout noise makes the loss look like it is still moving.

## Positional encoding: sinusoids → RoPE

Attention is permutation-equivariant: shuffle the inputs and the outputs shuffle with them. Position has to be injected.

**Original**: fixed sinusoids $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$, **added** to the embedding (not concatenated), wavelengths in a geometric series — a multi-scale clock.

**Now**: RoPE rotates $\mathbf{q}$ and $\mathbf{k}$ at every layer instead. Because rotation matrices satisfy $R_m^\top R_n = R_{n-m}$,

$$\langle R_m\mathbf{q}, R_n\mathbf{k}\rangle = \mathbf{q}^\top R_{n-m}\mathbf{k}$$

the score depends **only on the relative distance**. $\mathbf{v}$ is not rotated — it should carry no positional information.

<details markdown="1">
<summary><b>deeper</b>: where the relative property comes from</summary>

Rotations are orthogonal and compose additively in a plane: $R_m^\top = R_{-m}$ and $R_aR_b = R_{a+b}$. So $m$ and $n$ can only ever appear as $n-m$.

That is why RoPE extrapolates partially past its training length — it never encoded "token number", only "how far apart". It is also why rotating $\mathbf{q}$ but not $\mathbf{k}$ breaks it: with no $R_n$ to pair against, $R_m^\top$ has nothing to cancel with and absolute position survives.

</details>

```python
def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)          # split-half (Llama convention)
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

⚠️ Two pairing conventions exist — split-half (GPT-NeoX/Llama) and interleaved (the RoPE paper). They differ by a permutation of head channels, so **weights are not interchangeable** across them.

## What else changed

| | vanilla (2017) | today |
| --- | --- | --- |
| Structure | encoder + decoder | decoder-only |
| Norm | post-norm LayerNorm | pre-norm RMSNorm |
| Position | sinusoidal, added to input | RoPE, applied to q/k per layer |
| Attention | MHA | GQA |
| FFN | ReLU, $4d$ | SwiGLU, $\tfrac{8}{3}d$ |
| Inference | recompute each step | KV cache |

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}\odot\boldsymbol{\gamma}, \qquad \text{SwiGLU}(\mathbf{x}) = \big(\text{SiLU}(W_g\mathbf{x})\odot W_u\mathbf{x}\big)W_d$$

RMSNorm drops the mean subtraction and the bias (accumulate the sum of squares in fp32 regardless of activation dtype). SwiGLU has **three** matrices instead of two, hence the $\tfrac{8}{3}d$ hidden width rather than $4d$.

GQA shares one K/V head across $n_\text{rep}$ query heads. The KV cache costs

$$2 \cdot n_\text{layer} \cdot n_\text{kv} \cdot d_\text{head} \cdot T \cdot \text{sizeof(dtype)}$$

so dropping $n_\text{kv}$ from 32 to 8 saves 4× the memory — the dominant constraint in long-context inference.

## Building it, and proving it works

[`code/`](code/) has both, written without `nn.MultiheadAttention` or `F.scaled_dot_product_attention`:

- [`vanilla.py`](code/vanilla.py) — the 2017 encoder-decoder, with the Noam schedule and label smoothing
- [`model.py`](code/model.py) — modern decoder-only (RMSNorm + RoPE + GQA + SwiGLU + KV cache)

[`test_model.py`](code/test_model.py) checks the four things that actually break:

```
  causality:     perturb token 9 -> logits at positions < 9 differ by 0.0 (exactly)
  kv cache:      incremental decode vs one-shot forward, max error 3.6e-07
  rope relative: score(5,2) = score(20,17) = +5.6092;  score(20,10) = +0.4579
  init loss:     4.19 vs ln(V) = 4.16
```

![what prefill computes versus one decode step](assets/kv-cache.svg)

The subtlest bug is the mask under caching: query $i$ sits at absolute position `cache.pos + i` while keys run from 0, so the mask is a **non-square** $(T, S)$, and RoPE's cos/sin must be sliced from `cache.pos`. Also `cache.pos` advances once per forward pass — putting it inside `update()` multiplies it by `n_layer`.

## Self-check

<div class="taste-check advanced">
  <strong>After the full dissection, defend four claims:</strong>
  <ol>
    <li>Why divide attention scores by the square root of head dimension?</li>
    <li>Which gradient path changes under pre-norm?</li>
    <li>Why must RoPE rotate both Q and K?</li>
    <li>How do you prove KV cache correctness beyond plausible generations?</li>
  </ol>
</div>

## Where to read next

- [From linear models to neural networks](from-linear-to-neural.en.md)
- [Post-training](../05-post-training/README.en.md)
- [Systems](../06-systems/README.en.md)
