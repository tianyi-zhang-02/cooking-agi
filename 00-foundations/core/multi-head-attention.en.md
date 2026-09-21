# Multi-head attention: from equations to implementation

[中文](multi-head-attention.md) · **English**

> Reading time: ~9 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-en><span>Problem · 问题</span><strong>Let each position retrieve the information it needs from the sequence</strong></div>
    <div class="recipe-face" data-concept-zh><span>解决什么问题 · PROBLEM</span><strong>让每个位置去序列里取它需要的信息</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-en><span>Prerequisites · 前置知识</span><strong>Three projections W_Q, W_K, W_V · one output projection W_O</strong></div>
    <div class="recipe-face" data-concept-zh><span>前置知识 · PREREQUISITES</span><strong>三个投影 W_Q, W_K, W_V · 一个输出投影 W_O</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-en><span>Core mechanism · 核心机制</span><strong>Scaled dot product · split heads · mask before softmax</strong></div>
    <div class="recipe-face" data-concept-zh><span>核心机制 · CORE MECHANISM</span><strong>缩放点积 · 切头 · mask 在 softmax 之前</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-en><span>Common mistakes · 常见错误</span><strong>Reshape order, mask timing, and scaling by the wrong dimension</strong></div>
    <div class="recipe-face" data-concept-zh><span>常见错误 · COMMON MISTAKES</span><strong>reshape 的顺序、mask 的时机、除错了维度</strong></div>
  </div>
</div>

## Quick learning: multi-head attention in one sentence and its boundary conditions

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

## The core computation: match and aggregate information

Each position carries its own **question** (query) and asks every position's **index** (key). The better they match, the more it takes from that position's **content** (value). What comes back is a weighted average.

Multi-head means asking several different questions of the same sentence at once — one head watching syntactic pairing, one watching coreference, one watching positional adjacency. Each head fetches its own answer, and the answers are concatenated.

## What the attention matrix represents

First set aside the analogy that is easy to misread. $Q$, $K$, and $V$ are simply three different learned linear projections of the same input $X$:

$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$

They are not three individual dimensions, and nobody assigns them a meaning in advance. For a sequence of $T$ tokens and head dimension $d_k$, each of $Q,K,V$ has shape $(T,d_k)$. The formula gives them different **computational roles**: $Q$ and $K$ are used to compute the weights; $V$ supplies the vectors that are actually weighted and summed afterwards.

The full computation has only the following five steps.

### 1. Compute one scalar score for every pair of tokens

$
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
S_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_k}}
$

$S$ has shape $(T,T)$; this is the attention score matrix. Row $i$ means “where the current position $i$ wants to fetch information from”; column $j$ means “candidate source position $j$”. Each cell $S_{ij}$ is just one scalar; **$V$ has not been used yet.**

### 2. A decoder-only model first blocks future positions

When GPT predicts the next token at position $i$, it may use only position $i$ and everything to its left. The causal mask is written

$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$

For three tokens the score matrix becomes:

$
S+M=
\begin{bmatrix}
s_{11} & -\infty & -\infty\\
s_{21} & s_{22} & -\infty\\
s_{31} & s_{32} & s_{33}
\end{bmatrix}
$

The mask does not restrict the model to the single previous token. It lets a position see **itself and every earlier token**, while hiding the future. Training computes all positions in parallel; without the upper-right triangle masked, earlier positions could read the correct answers that come later, and the training objective would leak information. A bidirectional encoder normally has no causal mask, though it may still use a padding mask.

### 3. Apply softmax to every row

$
A=\operatorname{softmax}_{j}(S+M)
$

Softmax runs over the column index $j$, so every row satisfies

$
\sum_j A_{ij}=1
$

Because $e^{-\infty}=0$, masked positions receive exactly zero weight. $A$ is what people usually mean by the attention weight matrix: row $i$ says how much the current position takes from each allowed position.

### 4. Use that row of weights to aggregate all values

$
O=AV,\qquad
\mathbf o_i=\sum_j A_{ij}\mathbf v_j
$

So $QK^\top$ only decides “what the weights are”; what is actually fetched and mixed is $V$. One cell of the attention matrix is a scalar, while the output $\mathbf o_i$ is a $d_k$-dimensional vector.

### 5. Every head computes its own set, then they merge

Each head has its own projections, score matrix, and attention weights, so heads can learn different ways of matching. The outputs of all heads are concatenated first, then projected through $W_O$ back to the model dimension.

The whole data flow compresses into one line:

$
X\xrightarrow{W_Q,W_K,W_V}(Q,K,V)
\xrightarrow{QK^\top/\sqrt{d_k}}S
\xrightarrow{+M,\,\text{row-softmax}}A
\xrightarrow{AV}O
$

That is: **three projections produce Q/K/V; Q and K produce pairwise weights between tokens; the mask removes forbidden information paths; row-wise softmax normalizes; and those weights finally take a weighted sum of V.**

> The softmax in MHA does provide nonlinearity, but it mainly decides “whom to communicate with” along the token dimension. The FFN activation instead performs a nonlinear transformation along each token's feature dimension; the two play different roles.

## What a single head computes

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Broken down to a single query $\mathbf{q}_i$:

$$\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k})}{\sum_{j'}\exp(\mathbf{q}_i^\top \mathbf{k}_{j'}/\sqrt{d_k})}, \qquad \mathbf{o}_i = \sum_j \alpha_{ij}\mathbf{v}_j$$

Every row $\alpha_{i\cdot}$ sums to 1, so the output is always a convex combination of the values. **Attention does not create new information — it only decides where to move it from.**

## Why divide by $\sqrt{d_k}$

Interviewers love this question, and the answer is not “it is an empirical value”.

Let the components of $q, k$ be independent with mean 0 and variance 1. Then

$$\text{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_{i=1}^{d_k}\text{Var}(q_i k_i) = d_k$$

The standard deviation is $\sqrt{d_k}$. At $d_k = 64$ the typical magnitude of a dot product is already $\pm 8$, and softmax at that scale is close to one-hot. The softmax Jacobian is

$$\frac{\partial\,\text{softmax}(z)_i}{\partial z_j} = \alpha_i(\delta_{ij}-\alpha_j)$$

When one $\alpha_i \to 1$ and the rest $\to 0$, the whole Jacobian approaches the zero matrix: **the gradient vanishes**. Dividing by $\sqrt{d_k}$ pulls the variance back to 1.

⚠️ The divisor is $\sqrt{d_k} = \sqrt{d_\text{head}}$, **not** $\sqrt{d_\text{model}}$. It is easy to write the latter out of habit when coding by hand.

## Why use multiple heads: the goal is not more dimensions

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \quad \text{MultiHead} = \text{Concat}(\text{head}_1..\text{head}_h)W^O$$

One attention computes one similarity and returns one average. Splitting into $h$ heads of $d_k = d_\text{model}/h$ lets different heads track different relations **at the same total cost** — it divides the budget, it does not add to it. The three points below make that precise.

<div class="bilingual-note bilingual-intro">
  <span>Concept-by-concept · 逐概念双语</span>
  <p>The three cards below default to English; select <strong>中文 ↻</strong> to view the equivalent Chinese in place.</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### 1. The core of multi-head: multiple sets of attention relations

Suppose $d_{\text{model}}=512$. One full-width attention head computes

$$A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{512}}\right),\qquad O=AV.$$

The key limitation is not that “512 dimensions are not enough”. It is that every
value channel shares the same attention matrix $A$. To resolve “she” in “Xiao Ming
gave the book to Xiao Hong because she loves reading”, the model may need to track
coreference, syntactic dependency, semantic role, and local adjacency at the same
time; a single head must compress all of these relations into one distribution.

Multi-head lets head $i$ learn its own projections and weights:

$$Q_i=XW_i^Q,\qquad K_i=XW_i^K,\qquad V_i=XW_i^V,$$

$$A_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right).$$

The model therefore obtains $A_1,\ldots,A_h$: several ways to read the sequence.
Some heads may lean toward coreference, others toward locality or syntax, but those
jobs are not assigned by hand and can overlap. The more precise conclusion is that
**different features can use different attention weights instead of all sharing one
distribution.**

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">1. 多头的核心：多套注意力关系</div>

假设 $d_{\text{model}}=512$。一个完整维度的单头会计算

$$A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{512}}\right),\qquad O=AV.$$

关键限制不是“512 维不够”，而是所有 value 通道共享同一套注意力矩阵 $A$。处理
“小明把书送给小红，因为她很喜欢阅读”中的“她”时，模型可能同时需要追踪指代、
语法依赖、语义角色和局部邻近；单头必须把这些关系压进一套分布。

多头让第 $i$ 个头学习自己的投影和权重：

$$Q_i=XW_i^Q,\qquad K_i=XW_i^K,\qquad V_i=XW_i^V,$$

$$A_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right).$$

于是模型得到 $A_1,\ldots,A_h$ 多套读取方式。某些头可能偏向指代，另一些偏向
邻近或语法，但这些职责不是人工指定的，也可能彼此重叠。更准确的结论是：
**不同特征可以使用不同的注意力权重，不必全部共享一套分布。**

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### 2. Splitting dimensions controls the budget

The original model uses $d_{\text{model}}=512$ and $h=8$, usually with

$$d_k=d_v=\frac{512}{8}=64,$$

so $8\times64=512$. If all eight heads kept the full 512 dimensions, parameters and
compute would grow substantially; splitting the total width is what yields eight
sets of relations at roughly the budget of a single head.

A full-width single-head projection has

$$W_Q,W_K,W_V\in\mathbb{R}^{512\times512}.$$

Each multi-head projection is $512\times64$, and eight of them still total

$$8\times(512\times64)=512\times512.$$

Standard MHA therefore has about $4d_{\text{model}}^2$ parameters across Q, K, V,
and the output projection, independent of the head count itself. Code also usually
performs one large projection and then reshapes to `(B, H, T, d_head)` rather than
running eight small models in sequence — identical maths, a single GEMM.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">2. 拆分维度是在控制预算</div>

原版使用 $d_{\text{model}}=512,h=8$，通常令

$$d_k=d_v=\frac{512}{8}=64,$$

所以 $8\times64=512$。如果 8 个头都保留完整 512 维，参数和计算会大幅增长；
把总宽度拆开，才能在接近单头的预算下得到 8 套关系。

单头完整投影有

$$W_Q,W_K,W_V\in\mathbb{R}^{512\times512}.$$

多头每组投影是 $512\times64$，8 组合计仍为

$$8\times(512\times64)=512\times512.$$

因此标准 MHA 的 Q/K/V 和输出投影总参数量约为 $4d_{\text{model}}^2$，与头数本身
无关。代码也通常只做一次大投影，再 reshape 成 `(B, H, T, d_head)`；不是顺序执行
8 次小模型。

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-en" data-concept-en markdown="1">

### 3. Expressivity is not a generalization guarantee

What multi-head attention increases first is expressivity: it lets several token
relations, matching functions, and contextual summaries coexist. Better
representations sometimes improve performance on unseen data, but “it uses multiple
heads” does not by itself imply better generalization.

With too many heads, each head may become too narrow, several heads may duplicate
one another, parameters may be used inefficiently, and the model may even overfit.
In practice some heads can often be pruned with almost no loss in quality.
Therefore:

$$\boxed{\text{Multi-head attention obtains multiple relations at similar cost; it does not enlarge width for its own sake.}}$$

“Different representation subspaces” should not be over-interpreted either. Each
head does have independent parameters and can therefore learn a different matching
function; but “one head must handle the company sense and another must handle the
fruit sense” is neither designed in advance nor guaranteed to be interpretable.

</div>
<div class="concept-face concept-zh" data-concept-zh markdown="1">

<div class="concept-title-zh" role="heading" aria-level="3">3. 表达能力不等于泛化保证</div>

多头首先增加的是表达能力：它允许多种 token 关系、匹配函数和上下文摘要并存。
更好的表示有时会改善未见数据上的表现，但“用了多头”并不自动推出 generalization
更好。

头数过多时可能出现每头维度太小、多个头功能重复、参数利用率低，甚至过拟合。
实践中经常可以剪掉部分头而几乎不损失性能。所以：

$$\boxed{\text{多头不是为了把维度做大，而是在相近成本下获得多套注意力关系。}}$$

“不同表示子空间”也不要过度解释。每个头确实有独立参数，因此可以学习不同匹配
函数；但“某个头一定负责公司语义、另一个一定负责水果语义”并不是预先设计或必然
可解释的事实。

</div>
</section>

## The six places a from-scratch implementation goes wrong

These are the places where a mistake actually gets noticed:

| # | Trap | Correct |
| --- | --- | --- |
| 1 | reshape order | `view(B,T,H,dh).transpose(1,2)`, **not** `view(B,H,T,dh)` |
| 2 | mask timing | add it **before** the softmax, not zero things after |
| 3 | mask value | fill with $-\infty$; filling 0 means "equally likely" |
| 4 | softmax numerical stability | subtract each row's max first |
| 5 | the divisor | $\sqrt{d_\text{head}}$, not $\sqrt{d_\text{model}}$ |
| 6 | merging heads | `transpose(1,2).contiguous().view(...)` — without `contiguous()` it raises |

**Why #1 must be that way.** The projection's output is `(B, T, d_model)`, where the `d_model` dimension holds the $h$ heads laid **end to end**. So split the last dimension into `(H, dh)` first, then move `H` forward. A direct `view(B,H,T,dh)` slices across the time dimension instead, so every "head" is a pile of fragments from unrelated positions — right shape, wrong numbers, and no error raised.

**The intuition for #3.** Zeroing some positions after the softmax leaves the remaining weights no longer summing to 1, and the masked positions have already taken their share of probability mass during the softmax. Filling with $-\infty$ is what "this route does not exist at all" actually means.

## Experiment: verify three equivalent implementations

<details class="code-drop" markdown="1">
<summary><b>From scratch</b> · pure NumPy, no framework</summary>

This is the version you should be able to write from memory on a whiteboard.

```python
import numpy as np

def softmax(x, axis=-1):
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
        att = att.softmax(dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)   # merge heads
        return self.wo(y), att

def causal_mask(t, device=None):
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
```

In production, replace the three middle lines with a single `F.scaled_dot_product_attention(q, k, v)` — it automatically picks a fused kernel such as FlashAttention and saves the memory of the intermediate $T \times T$ matrix.

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

## Common interview questions

<details class="interview" markdown="1">
<summary>How many parameters does multi-head attention have?</summary>

Four $d_\text{model} \times d_\text{model}$ matrices, so $4d^2$ (without biases) — **independent of the number of heads**. Splitting into heads only regroups the same parameters.

</details>

<details class="interview" markdown="1">
<summary>What is the time and memory complexity?</summary>

Time $O(T^2 d)$; memory $O(hT^2)$ for the attention matrix. Long context is bottlenecked by the latter, which is exactly what FlashAttention removes — it computes block by block and never writes the $T\times T$ matrix to GPU memory.

</details>

<details class="interview" markdown="1">
<summary>Why do Q and K use two different matrices? Can they be shared?</summary>

Sharing makes $\mathbf{q}_i^\top\mathbf{k}_j$ symmetric, forcing "A should attend to B" to equal "B should attend to A". Most relations in language are asymmetric — an adjective modifies a noun, and the reverse does not hold.

</details>

<details class="interview" markdown="1">
<summary>Why doesn't V take part in the scoring?</summary>

The score decides *how much* to take; V decides *what* is taken. Mixing them lets content influence its own probability of being retrieved, which tends to degenerate.

</details>

<details class="interview" markdown="1">
<summary>Can multiple heads collapse into one?</summary>

Yes, and they often partly do. Several heads converging to similar attention distributions is a known phenomenon, and pruning most heads at inference has little effect. So "many heads" is not a virtue in itself — whether the heads are **complementary** is.

</details>

<details class="interview" markdown="1">
<summary>Why must the mask be added before the softmax? Would filling with 0 work?</summary>

No. Zeroing entries after the softmax leaves the remaining weights no longer summing to 1, and the masked positions have already taken their share of probability mass during the softmax. Filling with $-\infty$ is what is equivalent to "this route does not exist" — $e^{-\infty}=0$, so the position never enters the denominator during normalization.

</details>

## Self-check

<div class="taste-check">
  <strong>If you really understand this, you should be able to explain:</strong>
  <ol>
    <li>Without looking at code, write scaled dot-product attention and say clearly why the mask must come before the softmax.</li>
    <li>Why is the divisor $\sqrt{d_\text{head}}$ rather than $\sqrt{d_\text{model}}$? What happens without it?</li>
    <li>How do the results of <code>view(B,T,H,dh).transpose(1,2)</code> and <code>view(B,H,T,dh)</code> differ? Why does the latter raise no error yet get everything wrong?</li>
    <li>By what factor does multi-head attention multiply the parameter count?</li>
  </ol>
</div>

## Next

Attention is in place, but it is completely insensitive to order, and it stops training once stacked deep. First see how [residual connections](residual-connections.en.md) and [normalization](normalization.en.md) make depth feasible, then return to [the vanilla Transformer](vanilla-transformer.en.md) to assemble the whole block.
