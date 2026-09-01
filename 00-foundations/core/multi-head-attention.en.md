# Multi-head attention: from equations to implementation

[中文](multi-head-attention.md) · **English**

> Reading time: ~9 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>The problem</span><strong>every position fetching what it needs from the sequence</strong></div>
  <div><span>Prerequisites</span><strong>three projections W_Q, W_K, W_V · one output projection W_O</strong></div>
  <div><span>Core mechanism</span><strong>scaled dot product · split into heads · mask before softmax</strong></div>
  <div><span>Common mistakes</span><strong>reshape order, mask timing, dividing by the wrong dimension</strong></div>
</div>

## Quick learning: multi-head attention in one sentence and its boundary conditions

<details class="interview" markdown="1">
<summary>Match, normalize, aggregate, and why multiple heads exist</summary>

**Quick memory**: Q and K decide where to read; V decides what content is read. Each head learns a separate matching and transport subspace; multiple heads do not magically increase total model width.

**Interview answer**

> Every query takes scaled dot products with all keys, applies a mask and row-wise softmax, then uses those weights to aggregate values. Q and K must share their last dimension for the dot product; V only determines output width. Multiple heads learn different relations under a fixed compute budget and $W_O$ mixes their concatenated outputs.

<details markdown="1">
<summary><b>Deep dive</b>: what happens if Q equals K?</summary>

Before softmax, the Gram matrix $QQ^\top$ is symmetric and positive semidefinite. Row-wise softmax uses different denominators, so the final attention matrix is generally not symmetric. Forcing $W_Q=W_K$ also removes some directional matching freedom; separate projections let “who queries whom” differ from the reverse direction.

</details>
</details>

## The core computation: match and aggregate information

Each position carries a **question** (query) and asks every position's **index** (key). The better they match, the more it takes from that position's **content** (value). What comes back is a weighted average.

Multi-head means asking several different questions at once — one head tracking syntax, another coreference, another simple adjacency — then concatenating the answers.

## What the attention matrix represents

First set aside the mnemonic. $Q$, $K$, and $V$ are simply three learned linear projections of the same input $X$:

$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$

They are not three individual dimensions, and their coordinates have no human-assigned meanings. For a sequence of $T$ tokens and head dimension $d_k$, each of $Q,K,V$ has shape $(T,d_k)$. The formula gives them different **computational roles**: $Q$ and $K$ determine the weights; $V$ supplies the vectors that are later aggregated.

The full computation has five steps.

### 1. Compute one scalar for every pair of tokens

$
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
S_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_k}}
$

$S$ has shape $(T,T)$ and is the attention score matrix. Row $i$ asks where position $i$ should fetch information from; column $j$ is candidate source position $j$. Each cell $S_{ij}$ is one scalar. **$V$ has not been used yet.**

### 2. A decoder-only model blocks future positions

When GPT predicts from position $i$, it may use position $i$ and everything to its left, but nothing to its right. The causal mask is

$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$

For three tokens:

$
S+M=
\begin{bmatrix}
s_{11} & -\infty & -\infty\\
s_{21} & s_{22} & -\infty\\
s_{31} & s_{32} & s_{33}
\end{bmatrix}
$

The mask does not restrict a token to only its immediate predecessor. It can see **itself and every earlier token**, but no future token. Training evaluates all positions in parallel; without the upper triangle masked, earlier positions could read later ground-truth tokens and leak the answer. A bidirectional encoder normally has no causal mask, though it may still use a padding mask.

### 3. Apply softmax row by row

$
A=\operatorname{softmax}_{j}(S+M)
$

Softmax runs over source index $j$, so every row satisfies

$
\sum_j A_{ij}=1
$

Because $e^{-\infty}=0$, masked positions receive exactly zero weight. $A$ is the attention weight matrix: row $i$ says how much position $i$ takes from every allowed source position.

### 4. Use that row to aggregate the values

$
O=AV,\qquad
\mathbf o_i=\sum_j A_{ij}\mathbf v_j
$

Thus $QK^\top$ only determines the weights; the vectors actually fetched and mixed come from $V$. One cell of the attention matrix is a scalar, while output $\mathbf o_i$ is a $d_k$-dimensional vector.

### 5. Let every head do this separately, then merge

Each head has its own projections, score matrix, and attention weights, so heads can learn different matching rules. Their outputs are concatenated and projected through $W_O$ back to the model dimension.

The whole data flow fits in one line:

$
X\xrightarrow{W_Q,W_K,W_V}(Q,K,V)
\xrightarrow{QK^\top/\sqrt{d_k}}S
\xrightarrow{+M,\,\text{row-softmax}}A
\xrightarrow{AV}O
$

In words: **three projections produce Q/K/V; Q and K produce pairwise token weights; the mask removes forbidden information paths; row-wise softmax normalizes the weights; those weights finally average V.**

> Softmax does make MHA nonlinear, primarily by choosing communication across the token dimension. The FFN activation instead performs a nonlinear transformation across each token's feature dimensions; the two serve different roles.

## A single head

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Per query: $\alpha_{ij} = \text{softmax}_j(\mathbf{q}_i^\top\mathbf{k}_j/\sqrt{d_k})$, then $\mathbf{o}_i = \sum_j \alpha_{ij}\mathbf{v}_j$.

Every row of $\alpha$ sums to 1, so the output is always a convex combination of value vectors. **Attention does not create information — it decides where to move it from.**

## Why $\sqrt{d_k}$

For unit-variance $q, k$,

$$\text{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_{i=1}^{d_k}\text{Var}(q_i k_i) = d_k$$

so the spread is $\pm 8$ at $d_k = 64$. Softmax at that scale is nearly one-hot, and its Jacobian

$$\frac{\partial\,\text{softmax}(z)_i}{\partial z_j} = \alpha_i(\delta_{ij}-\alpha_j)$$

collapses toward the zero matrix — no gradient. Dividing by $\sqrt{d_k}$ restores unit variance.

⚠️ It is $\sqrt{d_\text{head}}$, **not** $\sqrt{d_\text{model}}$. Easy to write the wrong one from memory.

## Why use multiple heads

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \quad \text{MultiHead} = \text{Concat}(\text{head}_1..\text{head}_h)W^O$$

One attention computes one similarity and returns one average. Splitting into $h$ heads of $d_k = d_\text{model}/h$ lets different heads track different relations **at the same total cost** — it divides the budget, it does not add to it.

In code it is one $(d_\text{model}, d_\text{model})$ projection reshaped into heads: identical maths, a single GEMM.

## The six places this goes wrong

| # | Trap | Correct |
| --- | --- | --- |
| 1 | reshape order | `view(B,T,H,dh).transpose(1,2)`, **not** `view(B,H,T,dh)` |
| 2 | mask timing | add it **before** the softmax, not zero things after |
| 3 | mask value | fill with $-\infty$; filling 0 means "equally likely" |
| 4 | softmax stability | subtract the row max first |
| 5 | the divisor | $\sqrt{d_\text{head}}$, not $\sqrt{d_\text{model}}$ |
| 6 | merging heads | `transpose(1,2).contiguous().view(...)` — without `contiguous()` it raises |

**Why #1 must be that way.** The projection's output is `(B, T, d_model)` where `d_model` holds the $h$ heads laid **end to end**. So split the last dimension into `(H, dh)` first, then move `H` forward. `view(B,H,T,dh)` slices across the time dimension instead, giving "heads" made of unrelated positions — right shape, wrong numbers, no error raised.

**The intuition for #3.** Zeroing entries after the softmax leaves the row no longer summing to 1, and the masked positions have already taken probability mass. $-\infty$ is what "this route does not exist" actually means.

## Experiment: verify three equivalent implementations

<details class="code-drop" markdown="1">
<summary><b>From scratch</b> · pure NumPy, no framework</summary>

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # stability: subtract the max
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention(q, k, v, mask=None):
    """q,k,v: (B, H, T, d_head)   mask: True = blocked"""
    d = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d)      # (B, H, Tq, Tk)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)      # BEFORE the softmax
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
```

Runnable, with shape printouts and self-checks: [`../code/attention_numpy.py`](../code/attention_numpy.py)

</details>

<details class="code-drop" markdown="1">
<summary><b>With a framework</b> · PyTorch, what you would actually write</summary>

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
        att = att.softmax(dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)   # merge heads
        return self.wo(y), att
```

In production those middle lines collapse to `F.scaled_dot_product_attention(q, k, v)`, which picks a fused kernel and never materialises the $T \times T$ matrix.

</details>

[`../code/attention_torch.py`](../code/attention_torch.py) runs **four** implementations on identical weights and compares them:

```
  from-scratch vs F.scaled_dot_product_attention : 1.19e-07
  from-scratch vs nn.MultiheadAttention          : 1.19e-07
  from-scratch torch vs pure NumPy               : 1.19e-07
  every attention row sums to 1                  : 1.19e-07
  weight on any future position                  : 0.00e+00
```

Matching `nn.MultiheadAttention` is a far stronger claim than merely running. Where they disagree is the best debugging exercise available — and note that `nn.MultiheadAttention` stacks $W_Q, W_K, W_V$ into one `in_proj_weight`, while `nn.Linear` stores $(out, in)$, so moving weights to NumPy needs a transpose.

## Common interview questions

<details class="interview" markdown="1">
<summary>How many parameters does multi-head attention have?</summary>

Four $d_\text{model} \times d_\text{model}$ matrices, so $4d^2$ without biases — **independent of the number of heads**. Splitting into heads redistributes the same parameters.

</details>

<details class="interview" markdown="1">
<summary>What is the time and memory complexity?</summary>

Time $O(T^2 d)$; memory $O(hT^2)$ for the attention matrix. Long context is bound by the second, which is exactly what FlashAttention removes by tiling the computation and never writing the $T\times T$ matrix to memory.

</details>

<details class="interview" markdown="1">
<summary>Why separate $W_Q$ and $W_K$ instead of sharing one matrix?</summary>

Sharing makes $\mathbf{q}_i^\top\mathbf{k}_j$ symmetric, forcing "A should attend to B" to equal "B should attend to A". Most linguistic relations are asymmetric — an adjective modifies a noun, not the reverse.

</details>

<details class="interview" markdown="1">
<summary>Why doesn't V take part in the scoring?</summary>

The score decides *how much* to take, V decides *what* is taken. Letting content influence its own retrieval probability tends to degenerate.

</details>

<details class="interview" markdown="1">
<summary>Can multiple heads collapse into one?</summary>

Yes, and they often partly do. Heads converging to similar attention patterns is a known phenomenon, and much of the head count can be pruned at inference with little loss. So head *count* isn't the virtue — whether the heads are **complementary** is.

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can:</strong>
  <ol>
    <li>Write scaled dot-product attention from memory, and say why the mask must precede the softmax.</li>
    <li>Explain why the divisor is $\sqrt{d_\text{head}}$ and what happens without it.</li>
    <li>Say how <code>view(B,T,H,dh).transpose(1,2)</code> differs from <code>view(B,H,T,dh)</code>, and why the second fails silently.</li>
    <li>State how many parameters the module has as a function of head count.</li>
  </ol>
</div>

## Next

Attention is order-blind and hard to train deep. See [residual connections](residual-connections.en.md) and [normalisation](normalization.en.md) for what makes depth possible, then assemble everything in [the vanilla Transformer](vanilla-transformer.en.md).
