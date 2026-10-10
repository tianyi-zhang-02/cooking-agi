# Multi-head attention: from equations to implementation

[中文](multi-head-attention.md) · **English**

> Reading time: ~15 min · Level: core · Last reviewed: 2026-10-10

Start with one weighted sum. Two positions supply values `[2, 0]` and `[0, 4]`. With attention weights `0.75` and `0.25`, the output is `[1.5, 1]`. The rest of attention explains where those weights come from, which positions are masked, and how several such computations are combined.

## The core computation: match and aggregate information {#the-core-computation-match-and-aggregate-information}

Each position carries its own **question** (query) and asks every position's **index** (key). The better they match, the more it takes from that position's **content** (value). What comes back is a weighted average.

Each head uses its own projections to compute a weighted result, and the results are concatenated. Some heads may learn patterns such as proximity or coreference, but those roles are neither assigned in advance nor guaranteed to be individually interpretable.

## What the attention matrix represents {#what-the-attention-matrix-represents}

Start with self-attention: $Q$, $K$, and $V$ are three learned linear projections of the same input $X$. Cross-attention can instead obtain K/V from another sequence; we return to that later.

$$
\begin{gathered}
Q=XW_Q\\
K=XW_K\\
V=XW_V
\end{gathered}
$$

They are not three individual dimensions with assigned meanings. This example sets $d_v=d_k$: with $T$ tokens, $Q,K,V$ each have shape $(T,d_k)$, although V can have a different width in general. Their **computational roles** differ: Q/K compute the weights; V supplies the vectors to aggregate.

The full computation has only the following five steps.

### 1. Compute one scalar score for every pair of tokens {#1-compute-one-scalar-score-for-every-pair-of-tokens}

$$
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
S_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_k}}
$$

$S$ has shape $(T,T)$; this is the attention score matrix. Row $i$ means “where the current position $i$ wants to fetch information from”; column $j$ means “candidate source position $j$”. Each cell $S_{ij}$ is just one scalar; **$V$ has not been used yet.**

### 2. A decoder-only model first blocks future positions {#2-a-decoder-only-model-first-blocks-future-positions}

When GPT predicts the next token at position $i$, it may use only position $i$ and everything to its left. The causal mask is written

$$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

For three tokens the score matrix becomes:

$$
S+M=
\begin{bmatrix}
s_{11} & -\infty & -\infty\\
s_{21} & s_{22} & -\infty\\
s_{31} & s_{32} & s_{33}
\end{bmatrix}
$$

The mask does not restrict the model to the single previous token. It lets a position see **itself and every earlier token**, while hiding the future. Training computes all positions in parallel; without the upper-right triangle masked, earlier positions could read the correct answers that come later, and the training objective would leak information. A bidirectional encoder normally has no causal mask, though it may still use a padding mask.

### 3. Apply softmax to every row {#3-apply-softmax-to-every-row}

$$
\begin{gathered}
A=\operatorname{softmax}_{j}(S+M)
\end{gathered}
$$

Softmax runs over the column index $j$, so every row satisfies

$$
\sum_j A_{ij}=1
$$

Because $e^{-\infty}=0$, masked positions receive zero weight. This assumes at least one allowed position per row, with finite scores at allowed positions. An entirely masked row has no valid normalization denominator; ordinary softmax cannot be expected to return zeros automatically. $A$ is the attention weight matrix: row $i$ specifies how much to take from each allowed position.

### 4. Use that row of weights to aggregate all values {#4-use-that-row-of-weights-to-aggregate-all-values}

$$
O=AV,\qquad
\mathbf o_i=\sum_j A_{ij}\mathbf v_j
$$

So $QK^\top$ only decides “what the weights are”; what is actually fetched and mixed is $V$. One cell of the attention matrix is a scalar, while the output $\mathbf o_i$ is a $d_k$-dimensional vector.

### 5. Every head computes its own set, then they merge {#5-every-head-computes-its-own-set-then-they-merge}

Each head has its own projections, score matrix, and attention weights, so heads can learn different ways of matching. The outputs of all heads are concatenated first, then projected through $W_O$ back to the model dimension.

The whole data flow compresses into one line:

$$
\begin{gathered}
S=QK^\top/\sqrt{d_k},\\
A=\operatorname{softmax}_{\rm row}(S+M),\\
O=AV.
\end{gathered}
$$

That is: **three projections produce Q/K/V; Q and K produce pairwise weights between tokens; the mask removes forbidden information paths; row-wise softmax normalizes; and those weights finally take a weighted sum of V.**

> The softmax in MHA does provide nonlinearity, but it mainly decides “whom to communicate with” along the token dimension. The FFN activation instead performs a nonlinear transformation along each token's feature dimension; the two play different roles.

## What a single head computes {#what-a-single-head-computes}

$$
\begin{gathered}
\text{Attention}(Q,K,V)\\
= \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\end{gathered}
$$

Broken down to a single query $\mathbf{q}_i$:

$$
\begin{gathered}
s_{ij}=\mathbf q_i^\top\mathbf k_j/\sqrt{d_k},\\
\alpha_{ij}=\frac{\exp(s_{ij})}{\sum_{j\prime}\exp(s_{ij\prime})},\\
\mathbf o_i=\sum_j\alpha_{ij}\mathbf v_j.
\end{gathered}
$$

Without attention dropout, the weights are nonnegative and sum to one, so **one head's output before output projection** is a convex combination of its values. The opening `[1.5, 1]` example illustrates this. That does not describe the entire Transformer block: dropout, output projection, and residual connections change the result. The weights also depend on the input; this is not a fixed average.

## Why divide by $\sqrt{d_k}$ {#why-divide-by-sqrtd_k}

Start with what scaling changes. Suppose a head has dimension 64 and one query gives dot-product scores `[8, −8]` for two keys:

| Treatment | Input to softmax | Weights on the two positions |
| --- | --- | --- |
| No scaling | `[8, −8]` | About `[0.9999999, 0.0000001]` |
| Divide by $\sqrt{64}=8$ | `[1, −1]` | About `[0.881, 0.119]` |
| Subtract the maximum only | `[0, −16]` | Same as the first row |

Without scaling, almost all weight goes to the first position. After scaling, the second still contributes. **Uniform weights are not the goal. We want to avoid making attention extremely sharp simply by increasing the head dimension.**

Dividing by a positive constant changes the gaps between scores and therefore the weights. Subtracting a common constant preserves the gaps; it makes exponentiation safer without changing the mathematical result. Even `[1000, 1000]` should give `[0.5, 0.5]`. Large absolute values alone do not imply saturation.

<details markdown="1">
<summary>Why a square root? Work through the variance</summary>

Assume $q_1,\ldots,q_{d_k},k_1,\ldots,k_{d_k}$ are **all mutually independent**, each with mean zero and variance one. For one product:

$$
\begin{gathered}
\mathbb E[q_i k_i]=0,\\
\mathbb E[q_i^2]\mathbb E[k_i^2]=1,\\
\operatorname{Var}(q_i k_i)=1.
\end{gathered}
$$

The covariance between distinct products is zero, so:

$$
\begin{aligned}
\operatorname{Var}(q^\top k)&=d_k,\\
\operatorname{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)&=1.
\end{aligned}
$$

At dimension 64, the unscaled dot product has standard deviation 8. Dividing by 8 brings that to one. Dividing by 64 would instead give variance $1/64$. This is a scale adjustment, not an average of the 64 terms.

That is the motivation in [Transformer §3.2.1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1). Learned Q/K need not retain these properties. For a counterexample, set $q=k$: with zero-mean, unit-variance components, $q^\top k$ now has expectation $d_k$, not zero. Unit component variance alone is not enough.

</details>

<details markdown="1">
<summary>What happens to gradients when the weights become sharp?</summary>

For softmax weights $\alpha$ and input scores $z$:

$$
\frac{\partial\alpha_i}{\partial z_j}=\alpha_i(\delta_{ij}-\alpha_j).
$$

With two positions, $\partial\alpha_1/\partial z_1=\alpha_1(1-\alpha_1)$. It is 0.25 when the weight is 0.5 and approaches zero as the weight approaches one. For bounded values and upstream gradients, the gradient reaching the scores through the attention weights becomes small.

This concerns **the gradient path through attention**, not every gradient in the network. It also differs from a classification head with softmax + cross-entropy, whose logit gradient is predicted probability minus the label. A confident wrong prediction can still produce a large gradient there.

</details>

Use the **head dimension $d_k$**, not the full model width $d_\text{model}$. If a model also uses QK normalization or a learned temperature, follow its actual formula. Scaling addresses one source of trouble; it does not guarantee stable training.

## Why use multiple heads: the goal is not more dimensions {#why-use-multiple-heads-the-goal-is-not-more-dimensions}

$$
\begin{gathered}
\text{head}_i=\\
\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V),\\
\operatorname{MultiHead}=\\
\operatorname{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O.
\end{gathered}
$$

One head uses one set of attention weights for all value channels. Splitting a fixed width into $h$ heads gives several such relations at similar projection and matmul cost, although storing separate attention matrices and executing kernels can add overhead. The three points below explain the tradeoff.

### 1. The core of multi-head: multiple sets of attention relations {#1-the-core-of-multi-head-multiple-sets-of-attention-relations}

Suppose $d_{\text{model}}=512$. One full-width attention head computes

$$
\begin{gathered}
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{512}}\right)\\
O=AV.
\end{gathered}
$$

The key limitation is not that “512 dimensions are not enough”. It is that every
value channel shares the same attention matrix $A$. To resolve “she” in “Xiao Ming
gave the book to Xiao Hong because she loves reading”, the model may need to track
coreference, syntactic dependency, semantic role, and local adjacency at the same
time; a single head must compress all of these relations into one distribution.

Multi-head lets head $i$ learn its own projections and weights:

$$
\begin{gathered}
Q_i=XW_i^Q\\
K_i=XW_i^K\\
V_i=XW_i^V,
\end{gathered}
$$

$$
A_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right).
$$

The model therefore obtains $A_1,\ldots,A_h$: several ways to read the sequence.
Some heads may lean toward coreference, others toward locality or syntax, but those
jobs are not assigned by hand and can overlap. The more precise conclusion is that
**different features can use different attention weights instead of all sharing one
distribution.**

### 2. Splitting dimensions controls the budget {#2-splitting-dimensions-controls-the-budget}

The original model uses $d_{\text{model}}=512$ and $h=8$, usually with

$$
d_k=d_v=\frac{512}{8}=64,
$$

so $8\times64=512$. If all eight heads kept the full 512 dimensions, parameters and
compute would grow substantially; splitting the total width is what yields eight
sets of relations at roughly the budget of a single head.

A full-width single-head projection has

$$
W_Q,W_K,W_V\in\mathbb{R}^{512\times512}.
$$

Each multi-head projection is $512\times64$, and eight of them still total

$$
8\times(512\times64)=512\times512.
$$

Standard MHA therefore has about $4d_{\text{model}}^2$ parameters across Q, K, V,
and the output projection, independent of the head count itself. Code also usually
performs one large projection and then reshapes to `(B, H, T, d_head)` rather than
running eight small models in sequence — identical maths, a single GEMM.

### 3. Expressivity is not a generalization guarantee {#3-expressivity-is-not-a-generalization-guarantee}

What multi-head attention increases first is expressivity: it lets several token
relations, matching functions, and contextual summaries coexist. Better
representations sometimes improve performance on unseen data, but “it uses multiple
heads” does not by itself imply better generalization.

With too many heads, each head may become too narrow, several heads may duplicate
one another, parameters may be used inefficiently, and the model may even overfit.
In practice some heads can often be pruned with almost no loss in quality.
Therefore:

**Multi-head attention obtains multiple relations at similar cost; it does not enlarge width for its own sake.**

“Different representation subspaces” should not be over-interpreted either. Each
head does have independent parameters and can therefore learn a different matching
function; but “one head must handle the company sense and another must handle the
fruit sense” is neither designed in advance nor guaranteed to be interpretable.

## The six places a from-scratch implementation goes wrong {#the-six-places-a-from-scratch-implementation-goes-wrong}

Check these six details before trying to diagnose a mistake from the final loss:

| # | Trap | Correct |
| --- | --- | --- |
| 1 | reshape order | `view(B,T,H,dh).transpose(1,2)`, **not** `view(B,H,T,dh)` |
| 2 | mask timing | usually mask before softmax; zeroing weights afterwards without renormalizing is not equivalent |
| 3 | mask value | set blocked logits to $-\infty$; setting them to zero still assigns weight, and adding zero changes nothing |
| 4 | softmax numerical stability | subtract each row's max first |
| 5 | the divisor | $\sqrt{d_\text{head}}$, not $\sqrt{d_\text{model}}$ |
| 6 | merging heads | `transpose(1,2).contiguous().view(...)`, or `reshape`, which copies if needed; transposed strides may be incompatible with `view` |

**Why #1 must be that way.** The projection's output is `(B, T, d_model)`, where the `d_model` dimension holds the $h$ heads laid **end to end**. So split the last dimension into `(H, dh)` first, then move `H` forward. A direct `view(B,H,T,dh)` slices across the time dimension instead, so every "head" is a pile of fragments from unrelated positions — right shape, wrong numbers, and no error raised.

**Check the mask with two scores.** Softmax of `[0, 0]` is `[0.5, 0.5]`. Allowing only the first position gives `[1, 0]` when masked before softmax. Zeroing the second weight afterwards gives `[0.5, 0]`, shrinking the output. Dividing again by the remaining weight sum is mathematically equivalent to `[1, 0]`, but that sum can underflow in finite precision. Masking before softmax avoids this issue. The teaching code rejects entirely masked rows instead of silently producing NaNs.

## Experiment: verify three equivalent implementations {#experiment-verify-three-equivalent-implementations}

<details class="code-drop" markdown="1">
<summary><b>From scratch</b> · pure NumPy, no framework</summary>

This is the version you should be able to write from memory on a whiteboard.

```python
import numpy as np

def softmax(x, axis=-1):
    if np.any(np.all(np.isneginf(x), axis=axis)):
        raise ValueError("Each query must have at least one allowed key")
    x = x - np.max(x, axis=axis, keepdims=True)   # numerical stability: subtract the max first
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention(q, k, v, mask=None):
    """q,k,v: (B, H, T, d_head)   mask: True = blocked"""
    d = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d)      # (B, H, Tq, Tk)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)      # before the softmax
    w = softmax(scores, axis=-1)
    return w @ v, w

class MultiHeadAttention:
    def __init__(self, d_model, n_head, seed=0):
        assert d_model % n_head == 0
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.n_head, self.d_head = n_head, d_model // n_head
        self.Wq, self.Wk, self.Wv, self.Wo = (
            rng.uniform(-s, s, (d_model, d_model)) for _ in range(4))

    def split(self, x):                               # (B,T,C) -> (B,H,T,dh)
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_head, self.d_head).transpose(0, 2, 1, 3)

    def __call__(self, x, mask=None):
        q, k, v = self.split(x @ self.Wq), self.split(x @ self.Wk), self.split(x @ self.Wv)
        out, w = attention(q, k, v, mask)
        b, h, t, dh = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(b, t, h * dh)   # merge heads
        return out @ self.Wo, w

def causal_mask(t):
    return np.triu(np.ones((t, t), dtype=bool), k=1)  # strict upper triangle = the future
```

Full runnable version, with shape printouts and self-checks: [`../code/attention_numpy.py`](../code/attention_numpy.py)

</details>

<details class="code-drop" markdown="1">
<summary><b>With a framework</b> · PyTorch, what you would actually write in a codebase</summary>

```python
import math, torch, torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, bias=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        split = lambda p: p(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k, v = split(self.wq), split(self.wk), split(self.wv)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            att = att.masked_fill(mask, float("-inf"))
        if torch.isneginf(att).all(dim=-1).any():
            raise ValueError("Each query must have at least one allowed key")
        att = att.softmax(dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)   # merge heads
        return self.wo(y), att

def causal_mask(t, device=None):
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
```

When switching to SDPA, **carry the mask across too**. This example uses boolean `True` for blocked positions; PyTorch 2.8 SDPA uses `True` for allowed positions, the opposite convention. With a valid mask, and when only the output is needed, the core call is:

```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None if mask is None else ~mask,
    dropout_p=0.0,
)
```

It returns per-head outputs, not attention weights; head merging and output projection are still needed. Fused-kernel selection depends on device, dtype, and input support: calling SDPA does not guarantee FlashAttention. The teaching implementation's row checks aid debugging, not performance. See the [PyTorch 2.8 SDPA reference](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html).

</details>

[`../code/attention_torch.py`](../code/attention_torch.py) runs **four implementations** on the same set of weights and checks them against one another:

```
  from-scratch vs F.scaled_dot_product_attention : 1.19e-07
  from-scratch vs nn.MultiheadAttention          : 1.19e-07
  from-scratch torch vs pure NumPy               : 1.19e-07
  every attention row sums to 1                  : 1.19e-07
  weight on any future position                  : 0.00e+00
```

Getting the from-scratch version to match `nn.MultiheadAttention` matters more than merely "it runs". When they disagree, locating the difference is itself an effective debugging exercise. `nn.MultiheadAttention` stores $W_Q, W_K, W_V$ as one concatenated `in_proj_weight`, while `nn.Linear` stores $(out, in)$, so moving weights to NumPy needs a transpose.

## Common interview questions {#common-interview-questions}

<details class="interview" markdown="1">
<summary>How many parameters does multi-head attention have?</summary>

Four $d_\text{model} \times d_\text{model}$ matrices, so $4d^2$ (without biases) — **independent of the number of heads**. Splitting into heads only regroups the same parameters.

</details>

<details class="interview" markdown="1">
<summary>What is the time and memory complexity?</summary>

For batch size B and model width d, time including projections is $O(BTd^2+BT^2d)$; explicitly storing the per-head attention matrices takes $O(BhT^2)$ space. FlashAttention uses tiling to avoid writing those full matrices to GPU memory. It does not remove dense attention's quadratic compute, nor the storage for other activations, KV, and parameters.

</details>

<details class="interview" markdown="1">
<summary>Why do Q and K use two different matrices? Can they be shared?</summary>

They can be shared, but that restricts the matching function. In this self-attention example without extra positional transformations, Q=K makes the raw scores $QQ^\top$ symmetric. Row-wise softmax uses different denominators, so the final weights need not be symmetric. Separate projections let even the raw scores differ by direction; symmetric scores do not imply symmetric attention weights.

</details>

<details class="interview" markdown="1">
<summary>Why doesn't V take part in the scoring?</summary>

Separate projections allow matching features to differ from the content being transmitted. Matching might emphasize entity type while values retain particular attributes. Q/K/V all come from the input, so content already affects scores. Involving V in scoring does not inherently cause collapse; the standard design simply separates these computational roles.

</details>

<details class="interview" markdown="1">
<summary>Can multiple heads collapse into one?</summary>

Some heads can be redundant. [Head-pruning research](https://arxiv.org/abs/1905.10650) removed subsets of heads with little performance change in some models and tasks. That does not imply that most heads can safely be removed from any model: head selection, fine-tuning, and evaluation data all matter.

</details>

<details class="interview" markdown="1">
<summary>Why must the mask be added before the softmax? Would filling with 0 work?</summary>

Adding zero to a blocked logit does not mask it. Setting it to zero still leaves an unnormalized weight of $e^0=1$. Setting it to $-\infty$ before softmax removes it from the denominator. Zeroing weights after softmax and renormalizing is mathematically equivalent on a nonempty allowed support, but zeroing alone is not, and numerical stability is harder to handle.

</details>

## Self-check {#self-check}

<div class="taste-check">
  <strong>If you really understand this, you should be able to explain:</strong>
  <ol>
    <li>Write scaled dot-product attention without looking at code, then use two scores to compare masking first with only zeroing weights after softmax.</li>
    <li>Why is the divisor $\sqrt{d_\text{head}}$ rather than $\sqrt{d_\text{model}}$? What happens without it?</li>
    <li>How do the results of <code>view(B,T,H,dh).transpose(1,2)</code> and <code>view(B,H,T,dh)</code> differ? Why does the latter raise no error yet get everything wrong?</li>
    <li>By what factor does multi-head attention multiply the parameter count?</li>
  </ol>
</div>

## Next {#next}

Without positional information or an order-dependent mask, self-attention is **permutation-equivariant**: reorder the inputs and the outputs follow that order, rather than staying unchanged. A causal mask already distinguishes earlier from later positions; positional encodings provide more direct position information. Next, see how [residual connections](residual-connections.en.md) and [normalization](normalization.en.md) help training, then assemble the block in [the vanilla Transformer](vanilla-transformer.en.md).

The equations start from [Attention Is All You Need §3.2](https://arxiv.org/abs/1706.03762); the tiled memory optimization is explained in [FlashAttention](https://arxiv.org/abs/2205.14135).

## Quick learning: multi-head attention in one sentence and its boundary conditions {#quick-learning-multi-head-attention-in-one-sentence-and-its-boundary-conditions}

<details class="interview" markdown="1">
<summary>Match, normalize, aggregate, and why multiple heads exist</summary>

**Quick memory**: Q and K decide where to read; V decides what content is read. Each head learns a separate matching and transport subspace; multiple heads do not magically increase total model width.

**Interview answer**

> Every query takes scaled dot products with all keys, applies a mask and row-wise softmax, then uses those weights to aggregate values. Q and K must share their last dimension for the dot product; V only determines output width. Multiple heads learn different relations in parallel under a fixed compute budget, and $W_O$ mixes their concatenated outputs.

<details markdown="1">
<summary><b>Deep dive</b>: what happens if Q equals K?</summary>

Before softmax, the Gram matrix $QQ^\top$ is symmetric and positive semidefinite. Row-wise softmax uses a different denominator per row, so the final attention matrix is generally not symmetric. Forcing $W_Q=W_K$ also removes some directional matching freedom; separate projections let “who queries whom” score differently from the reverse direction.

</details>
</details>
