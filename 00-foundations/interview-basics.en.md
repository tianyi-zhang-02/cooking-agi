# Interview basics: most of them ask the same thing

[中文](interview-basics.md) · **English**

> Reading time: ~10 min · Type: quick reference · Last reviewed: 2026-08

## What these questions have in common

These questions look scattered — losses, masks, normalization, RNNs, CNNs — but they fall into three groups: **does the gradient have an undamped path back**, **are training and inference the same thing**, and **is the invariance structural or paid for**. Recognize the group and you don't have to memorize the answer.

Each question below gets three layers: **what to say first** → **surviving the first follow-up** → **the layer that separates you**.

---

## The skeleton first: how attention is actually computed

Three of the questions below hang on this diagram.

```
X                                   [B, T, d_model]
 ├─ Q = X·Wq ─┐                     [B, h, Tq, d_k]
 ├─ K = X·Wk ─┤  split into h heads  [B, h, Tk, d_k]
 └─ V = X·Wv ─┘                     [B, h, Tk, d_v]
 │
 ① scores = Q·Kᵀ / √d_k             [B, h, Tq, Tk]
 ② scores = scores + mask           ← the causal mask goes here
 ③ A      = softmax(scores, -1)     rows sum to 1
 ④ out    = A·V                     [B, h, Tq, d_v]
 ⑤ concat the heads, through Wo
```

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

**How to read $A$**: row $i$ is the distribution over which positions token $i$ attends to. With the causal mask, row $i$ is nonzero only in columns $\le i$. Row 1 is degenerate — it can only see itself, so softmax gives exactly 1.0.

**Why $\sqrt{d_k}$ rather than $\sqrt{d_v}$ or $\sqrt{d_{\text{model}}}$**: each entry of
$QK^\top$ sums exactly $d_k$ products. Q and K must share their last dimension; V's feature
dimension may differ, although standard MHA usually sets $d_v=d_k$. At model width 768 with
12 heads, a head has $d_k=64$, so the divisor is $\sqrt{64}$. See the
[Transformer deep dive](transformer.en.md#scaled-dot-product-attention) for the derivation and
the $Q=K$ edge case.

Time $O(T^2 d)$, memory $O(T^2)$. That $T \times T$ matrix is the long-context bottleneck and the whole motivation for FlashAttention: **never materialize it**.

Reference implementation in [`00-foundations/code/attention_numpy.py`](code/).

---

## Group one: does the gradient have an undamped path

Three questions, one skeleton. **Whenever a *product* shows up, ask whether it can be an *addition*.**

<details class="interview" markdown="1">
<summary>p = σ(z), y is 0/1. Write MSE and BCE, and say which to use</summary>

$$\mathcal{L}_{\text{MSE}} = (p-y)^2 \qquad \mathcal{L}_{\text{BCE}} = -\big[y\log p + (1-y)\log(1-p)\big]$$

**This question is about the gradient, not about reciting formulas.** Key fact: $\sigma'(z) = p(1-p)$.

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = 2(p-y)\cdot p(1-p) \qquad\qquad \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = p - y$$

For BCE the $\sigma'$ cancels:

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial p} = \frac{p-y}{p(1-p)} \;\Longrightarrow\; \frac{\partial \mathcal{L}}{\partial z} = \frac{p-y}{p(1-p)}\cdot p(1-p) = p-y$$

**The consequence**: when $y=1$ and the model is confidently wrong ($z\to-\infty$, $p\to0$) —

- MSE gradient $\approx 2(0-1)\cdot 0\cdot 1 = 0$. **The gradient vanishes exactly when the error is largest.**
- BCE gradient $= -1$. **Maximum gradient exactly when maximally wrong.**

**One layer deeper**: for logistic regression BCE is convex in the weights; squared error with a sigmoid isn't.

**A common misstatement to avoid**: don't say "MSE isn't a proper scoring rule so it isn't calibrated" — **squared error *is* the Brier score, and it *is* proper**. Both yield calibrated probabilities; the difference is optimization behavior, not properness. Volunteering that correction usually lands well.

**In practice**: always `binary_cross_entropy_with_logits`. Computing $p$ then taking its log underflows to $-\infty$. The stable form:

$$\mathcal{L} = \max(z,0) - zy + \log\big(1+e^{-|z|}\big)$$

</details>

<details class="interview" markdown="1">
<summary>What's the difference between an RNN and an LSTM?</summary>

**Say first**: a vanilla RNN's problem isn't memory, it's that **gradients are a product along the time axis**.

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=1}^{k} W_h^\top \operatorname{diag}\big(\tanh'(\cdot)\big)$$

$\tanh' \le 1$ and $\sigma' \le \tfrac14$ (at $z=0$). Multiply $k$ numbers below one and it decays exponentially in the distance. If the recurrent weight norm exceeds one you get explosion instead.

**Clipping fixes explosion; it can't fix vanishing** — it only caps the upper end. So vanishing is the real disease.

**The fix**: a cell state with an **additive** update.

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)$$

The key derivative is $\dfrac{\partial c_t}{\partial c_{t-1}} = f_t$ — **an elementwise gate, not a matrix multiply through a saturating nonlinearity**. With the forget gate near one, gradient flows back almost undamped: the constant error carousel.

One line: **LSTM replaces "multiply by a matrix at every step" with "gated additive accumulation."**

**Detail**: the three gates use sigmoid because a gate needs a soft 0-to-1 switch; the candidate uses tanh because it's a value and needs a sign. **That's a semantic choice, not a gradient one.**

**Two connections that lift the answer:**

- **The LSTM cell state and the Transformer residual stream are the same idea** — an additive identity path so gradients don't pass through a nonlinearity. One along time, one along depth.
- **Transformers didn't replace RNNs because of gradients** — LSTM had solved that. **They won on parallelism.** An RNN is sequential along time; attention gives $O(1)$ path length between any two positions and computes the whole sequence at once.

</details>

<details class="interview" markdown="1">
<summary>Why LayerNorm? Why not BatchNorm? And where does it go?</summary>

**Why normalize at all**: activation scale accumulates along the residual stream with depth. Without it, deep stacks explode or vanish and can't train; normalization also conditions the loss surface so you can use a sane learning rate.

**Why Layer over Batch** — four reasons, and the third is the one most people can't produce:

1. Variable-length sequences with padding contaminate batch-dimension statistics;
2. BN depends on batch size and composition, while autoregressive decoding runs at batch size one, token by token — you'd fall back on running statistics and get a train/inference mismatch;
3. **BatchNorm would leak the future** — statistics computed across time put later tokens into earlier tokens' normalization, **walking straight around the causal mask you just added**;
4. LayerNorm normalizes over the feature dimension per token — independent of the batch and of every other position, identical at train and inference.

**Where it goes** — be able to write both:

```
Post-LN (original, 2017)      Pre-LN (modern)
x = LN(x + Attn(x))          x = x + Attn(LN(x))
x = LN(x + FFN(x))           x = x + FFN(LN(x))
                             ...
                             x = LN(x)   ← a final LN is required
```

**Why Pre-LN is more stable**: in Post-LN the **LayerNorm sits on the residual trunk**, so every backward pass goes through it at every layer. LN's Jacobian scales gradients down, and stacked dozens deep the signal reaching early layers is tiny — hence the original's warmup requirement. Pre-LN leaves a **clean identity trunk**, so $\partial x_{\text{out}}/\partial x_{\text{in}}$ carries an identity term and gradients flow straight back. No warmup needed, and it scales deeper.

**The cost**: the residual stream is never normalized end to end and its scale grows with depth, **so you must add a final LN after the last block**. People forget this one; producing it scores.

**One deeper**: RMSNorm drops mean-centering and the bias and only rescales by RMS — and it works, which tells you **the re-scaling was doing the work, not the re-centering**.

</details>

---

## Group two: training and inference must be the same thing

<details class="interview" markdown="1">
<summary>Why a causal mask? Which step does it go in, and why there?</summary>

**Say first**: it's what makes training all $T$ positions in one forward pass equivalent to training them one at a time.

The autoregressive objective is $\prod_t p(x_t\mid x_{<t})$. Self-attention is fully visible by default, so position $t$ sees $x_t$ itself — **predicting the next token becomes copying the answer**, training loss collapses toward zero, and at inference the future doesn't exist, so generation falls apart.

**So the mask isn't there to make the model stronger. It's there so training and inference are the same model.** Without it you'd need $T$ separate forward passes.

**It goes at step ② of the skeleton**: after the scaled dot product, before the softmax.

```python
mask   = np.triu(np.ones((T, T), dtype=bool), k=1)   # strictly upper = blocked
scores = np.where(mask, -np.inf, scores)             # add -inf, don't zero
```

**Why it must be before the softmax** (the follow-up): adding $-\infty$ makes $e^{-\infty}=0$, and **the softmax renormalizes over the remaining positions** — mathematically identical to those positions not existing.

Zeroing **after** the softmax **breaks normalization** — rows no longer sum to one, and unevenly: position 1 can only see itself, loses the most mass, and gets scaled down hardest. You've multiplied each position by an arbitrary, meaningless attenuation.

**Engineering detail**: in practice use a large negative number (`-1e9` or `torch.finfo(dtype).min`) rather than true `-inf`, because in fp16 a fully masked row (which happens with padding) gives `0/0 = NaN` out of the softmax.

**Why BERT doesn't need it**: it isn't autoregressive. The objective is MLM, and bidirectional visibility is the design, not a leak.

</details>

---

## Group three: is the invariance free or paid for

<details class="interview" markdown="1">
<summary>Does rotating an image affect a CNN's feature extraction?</summary>

**Yes, substantially.** Start by separating two things people conflate:

**Convolution gives translation *equivariance*, not *invariance*:**

$$f(T_x(I)) = T_x\big(f(I)\big)$$

Shift the input and the feature map shifts by the same amount. **Invariance** — output unchanged — comes from pooling afterwards, and it's approximate and local.

**Rotation is neither.** The kernel has a fixed orientation, so a 45-degree edge and a 135-degree edge activate entirely different filters. **Nothing in the architecture makes them the same object.**

**Why the asymmetry is structural**: translation equivariance falls out of **weight sharing plus locality** — the operator gives it to you free. Rotation equivariance isn't in the operator, so you have exactly two options:

1. **Buy it with data**: rotation augmentation. The network learns redundant filters, one set per orientation. You're **spending model capacity on invariance**, and only over the range you augmented.
2. **Change the operator**: group-equivariant CNNs, steerable CNNs, harmonic networks — or a spatial transformer that learns to canonicalize the pose.

**One line**: translation invariance is free, rotation invariance costs — and you pay in either data or operator.

**To make the conversation interesting**: even translation invariance is weaker than assumed — strided downsampling aliases, so a one-pixel shift can flip the prediction. The fix is anti-aliased downsampling.

</details>

---

## One more: Egg Drop gets easy when you reverse the state

With $k$ eggs and $n$ floors, find the threshold in the worst case. The direct formulation is
indeed a two-dimensional DP:

$$T(k,n)=1+\min_{1\le x\le n}\max\big(T(k-1,x-1),\;T(k,n-x)\big).$$

Drop at floor $x$: if it breaks, search below with one fewer egg; if it survives, search above
with the same eggs. The minimum chooses the floor and the maximum pays for the worse branch.
Correct, but every state still enumerates $x$.

Reverse the question: **with $m$ moves and $k$ eggs, how many floors can I cover?** Let that be
$F(m,k)$:

$$F(m,k)=F(m-1,k-1)+1+F(m-1,k),\qquad F(0,k)=F(m,0)=0.$$

After the first drop, the breaking branch covers $F(m-1,k-1)$ floors below, the current floor
adds one, and the surviving branch covers $F(m-1,k)$ above.

```python
def min_moves(eggs, floors):
    cover = [0] * (eggs + 1)
    moves = 0
    while cover[eggs] < floors:
        moves += 1
        for k in range(eggs, 0, -1):
            cover[k] = cover[k] + cover[k - 1] + 1
    return moves
```

The descending update keeps the right-hand side on the previous move. For 100 floors, two eggs
need 14 moves because $1+\cdots+14=105$. Three eggs need 9 because $F(8,3)=92<100$ while
$F(9,3)=129\ge100$.

**The interview idea**: the original is an eggs-by-floors minimax DP. Reversing it gives a
moves-by-eggs coverage DP that compresses to one dimension. “Shrinking the searchable space on
every action” is exactly what this recurrence counts.

## One more: Binary Tree Maximum Path Sum

The central distinction is between a complete answer whose highest point is the current
node and the state that can be returned to its parent.

Let $G(u)$ be the maximum path sum that must start at $u$ and may extend downward through
only one child. Ignore negative contributions:

$$L=\max(0,G(u.left)),\qquad R=\max(0,G(u.right)).$$

A complete path with $u$ as its highest point can use both sides:

$$\text{candidate}=u.val+L+R.$$

The value returned to the parent cannot branch, so it keeps only one side:

$$G(u)=u.val+\max(L,R).$$

```python
def max_path_sum(root):
    best = float("-inf")

    def gain(node):
        nonlocal best
        if node is None:
            return 0
        left = max(0, gain(node.left))
        right = max(0, gain(node.right))
        best = max(best, node.val + left + right)
        return node.val + max(left, right)

    gain(root)
    return best
```

Time is $O(n)$ and recursion space is $O(h)$. Initialize the global answer to negative
infinity, not zero, so an all-negative tree cannot incorrectly choose an empty path.

If the input is a list, clarify whether it is level-order serialization with `None`
markers or heap-indexed storage with `left=2i+1, right=2i+2`. They are not equivalent
for sparse trees. Defining unfamiliar serialization before coding protects correctness;
it is not stalling.

## Appendix: how to present the Transformer architecture

When asked to "walk through the architecture," don't recite the figure. **Go component → the problem it solves.**

| Component | What it solves |
| --- | --- |
| Positional information | attention is permutation-equivariant — **on its own it cannot see word order** |
| Residual | an identity path for gradients (same idea as the LSTM cell state) |
| $\sqrt{d_k}$ scaling | the dot product sums $d_k$ terms so variance grows with $d_k$; unscaled, the softmax saturates toward one-hot and **the gradient vanishes** |
| Multi-head | **one softmax expresses one attention pattern**; heads attend to different relations in different subspaces |
| FFN (≈4× expansion) | **most of the parameters live here**, usually read as key-value memory |
| Causal mask | see above — train/inference consistency |

**Then volunteer this**, which separates "read the 2017 paper" from "knows what current models look like":

| 2017 original | Modern LLM | Why |
| --- | --- | --- |
| Post-LN | Pre-LN + RMSNorm | no warmup, scales deeper |
| Sinusoidal / learned absolute positions | RoPE | relative positions, better extrapolation |
| ReLU FFN | SwiGLU | better at equal compute |
| MHA | GQA / MQA | **the KV cache is the inference memory bottleneck** |

## Where to read next

- [Vanilla Transformer](core/vanilla-transformer.en.md) · [Multi-head attention](core/multi-head-attention.en.md) · [Decoder-only](core/decoder-only.en.md)
- [Normalization](core/normalization.en.md) · [Residual connections](core/residual-connections.en.md)
- [The language model objective](deep-dives/language-model-objective.en.md)
- [Reference implementations](code/): `attention_numpy.py` from scratch, `attention_torch.py` alongside
