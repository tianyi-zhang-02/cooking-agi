# Interview basics: most of them ask the same thing

[中文](interview-basics.md) · **English**

> Reading time: ~10 min · Type: quick reference · Last reviewed: 2026-08

## What these questions have in common {#what-these-questions-have-in-common}

These questions look scattered — losses, masks, normalization, RNNs, CNNs — but they fall into three groups: **does the gradient have an undamped path back**, **are training and inference the same thing**, and **is the invariance structural or paid for**. Recognize the group and you don't have to memorize the answer.

Each question below gets three layers: **what to say first** → **surviving the first follow-up** → **the layer that separates you**.

---

## The skeleton first: how attention is actually computed {#the-skeleton-first-how-attention-is-actually-computed}

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

## Group one: does the gradient have an undamped path {#group-one-does-the-gradient-have-an-undamped-path}

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

**Separate optimization from calibration**: squared error on binary probabilities corresponds to the Brier score, which is also a proper scoring rule. Both it and BCE favor the true conditional probability under the ideal population objective; finite data and restricted models do not guarantee calibration. The comparison here concerns gradients, not whether either loss is allowed to estimate probabilities.

**In practice**: always `binary_cross_entropy_with_logits`. Computing $p$ then taking its log underflows to $-\infty$. The stable form:

$$\mathcal{L} = \max(z,0) - zy + \log\big(1+e^{-|z|}\big)$$

</details>

<details class="interview" markdown="1">
<summary>What's the difference between an RNN and an LSTM?</summary>

**Say first**: one reason vanilla RNNs struggle with long-range relationships is repeated gradient propagation through time.

$$\frac{\partial h_t}{\partial h_{t-k}} = J_t J_{t-1}\cdots J_{t-k+1},\qquad J_t=\operatorname{diag}\big(\tanh'(a_t)\big)W_h$$

This is the state-to-state Jacobian; backpropagation uses its transpose. The weights are shared, but activations change, so the Jacobians need not be identical. A common bound $\|J_t\|_2\le c<1$ guarantees decay along this path. A norm greater than one at one step does not guarantee explosion: directions and the remaining product matter too.

**Clipping limits an update; it cannot recover a signal that has already vanished**, or replace diagnosing the source of instability.

**The fix**: a cell state with an **additive** update.

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)$$

Holding the gates fixed, the direct cell-state path has derivative $\partial c_t/\partial c_{t-1}=\operatorname{diag}(f_t)$. This path only rescales elements, making long-range signal easier to preserve when the forget gate is near one. The total derivative also includes the gates' dependence on previous states; LSTMs do not eliminate vanishing gradients.

One line: **LSTM replaces "multiply by a matrix at every step" with "gated additive accumulation."**

**Detail**: sigmoid bounds a gate between zero and one; tanh allows signed candidate content. These choices fit the roles of gates and values and also affect optimization. Saturation has not disappeared.

**Two connections that lift the answer:**

- **Cell states and residual streams are a useful analogy, not identical structures.** Both offer additive paths; an LSTM gates memory through time, while residual connections typically preserve an identity term through depth.
- **Transformers offer several advantages.** They process positions in parallel during training, shorten paths between visible positions, and provide content-based access to history. LSTMs mitigate rather than solve long-range learning problems. Autoregressive Transformer generation still proceeds token by token.

</details>

<details class="interview" markdown="1">
<summary>Why LayerNorm? Why not BatchNorm? And where does it go?</summary>

**Why normalize at all**: it helps control the scale seen by sublayers and often makes deep networks easier to train. It works alongside initialization, residual scaling, and the optimizer; networks without normalization are not universally untrainable.

**Why do Transformers commonly use LayerNorm rather than BatchNorm?**

1. Padding affects BN statistics when it is included in the reduction without being excluded;
2. BN depends on batch size and composition and defaults to running statistics during inference. Autoregressive decoding differs from training, so this requires care;
3. Including the time axis in training BN statistics lets future tokens affect earlier positions. A causal attention mask does not prevent that leak;
4. Token-wise LN over features mixes neither batch examples nor time positions and uses the same statistics rule in training and inference. That does not make every other part of the model identical across modes.

**Where it goes** — be able to write both:

```
Post-LN (original, 2017)      Pre-LN (modern)
x = LN(x + Attn(x))          x = x + Attn(LN(x))
x = LN(x + FFN(x))           x = x + FFN(LN(x))
                             ...
                             x = LN(x)   ← a common final LN
```

**Why Pre-LN is common**: it preserves an identity term along the residual trunk without routing that path through LN at every layer. Xiong et al. also found large expected gradients near the output at initialization in Post-LN; warmup helps prevent overly aggressive early updates. Their Pre-LN experiments trained without warmup, but that is not a guarantee for every model or training recipe.

**Implementation check**: Pre-LN models commonly normalize the final residual output before the output head. This is a common recipe, not a mathematical necessity. Match the particular model when reproducing it.

**One deeper**: RMSNorm omits mean-centering and usually additive bias, rescaling by RMS with a learned gain. Its success shows that some models train well without centering, not that centering is universally irrelevant. See the [LayerNorm analysis](https://arxiv.org/abs/2002.04745) and [RMSNorm paper](https://arxiv.org/abs/1910.07467).

</details>

---

## Group two: training and inference must be the same thing {#group-two-training-and-inference-must-be-the-same-thing}

<details class="interview" markdown="1">
<summary>Why a causal mask? Which step does it go in, and why there?</summary>

**Say first**: it's what makes training all $T$ positions in one forward pass equivalent to training them one at a time.

The autoregressive objective is $\prod_t p(x_t\mid x_{<t})$. In a common implementation, position $t$ receives $x_t$ and predicts $x_{t+1}$. Seeing itself is allowed; seeing the future token $x_{t+1}$ leaks the target. The causal mask allows the current and earlier positions, matching the information available during generation.

**So the mask isn't there to make the model stronger. It's there so training and inference are the same model.** Without it you'd need $T$ separate forward passes.

**It goes at step ② of the skeleton**: after the scaled dot product, before the softmax.

```python
mask   = np.triu(np.ones((T, T), dtype=bool), k=1)   # strictly upper = blocked
scores = np.where(mask, -np.inf, scores)             # add -inf, don't zero
```

**Why it must be before the softmax** (the follow-up): adding $-\infty$ makes $e^{-\infty}=0$, and **the softmax renormalizes over the remaining positions** — mathematically identical to those positions not existing.

Zeroing **after** the softmax **breaks normalization** — rows no longer sum to one, and unevenly: position 1 can only see itself, loses the most mass, and gets scaled down hardest. You've multiplied each position by an arbitrary, meaningless attenuation.

**Engineering detail**: a fully masked row has no valid probability distribution. A plain softmax over all `-inf` produces NaNs; finite negative values may produce a uniform distribution, which is not a valid fix. `-1e9` also exceeds fp16's finite range. Explicitly handle queries without valid keys, check your attention API's all-masked behavior, and exclude padding from the loss. Masking after softmax and correctly renormalizing can be mathematically equivalent; merely zeroing entries is not.

**Why BERT doesn't need it**: it isn't autoregressive. The objective is MLM, and bidirectional visibility is the design, not a leak.

</details>

---

## Group three: is the invariance free or paid for {#group-three-is-the-invariance-free-or-paid-for}

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

## One more: Egg Drop gets easy when you reverse the state {#one-more-egg-drop-gets-easy-when-you-reverse-the-state}

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

## One more: Binary Tree Maximum Path Sum {#one-more-binary-tree-maximum-path-sum}

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

## Appendix: how to present the Transformer architecture {#appendix-how-to-present-the-transformer-architecture}

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

## Where to read next {#where-to-read-next}

- [Vanilla Transformer](core/vanilla-transformer.en.md) · [Multi-head attention](core/multi-head-attention.en.md) · [Decoder-only](core/decoder-only.en.md)
- [Normalization](core/normalization.en.md) · [Residual connections](core/residual-connections.en.md)
- [The language model objective](deep-dives/language-model-objective.en.md)
- [Reference implementations](code/): `attention_numpy.py` from scratch, `attention_torch.py` alongside

## Quick learning: what are the standard questions really testing? {#quick-learning-what-are-the-standard-questions-really-testing}

<details class="interview" markdown="1">
<summary>A one-minute framework and the boundary conditions people miss</summary>

**Quick memory**

Most Transformer fundamentals test four things: **shape closure, causal information flow, gradient flow, and train/inference equivalence.**

**Interview answer**

> I first write the input and output shapes and identify the reduction axis. Then I check causal masks and data boundaries for leakage, inspect residual paths and normalization for gradient flow, and finally verify that training, prefill, and decode implement equivalent computations.

<details markdown="1">
<summary><b>Deep dive</b>: why are invariants safer than memorized conclusions?</summary>

Facts such as “Q and K dimensions must match,” “V may differ,” and “KV-cached decoding must equal a full forward pass” all follow from matrix shapes and semantic invariants. A memorized sentence breaks when notation or implementation changes; shape, causality, and equivalence let you derive the answer again.

</details>

</details>
