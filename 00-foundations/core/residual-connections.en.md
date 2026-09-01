# Residual connections

[中文](residual-connections.md) · **English**

> Reading time: ~7 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>The problem</span><strong>depth that stops being an obstacle to training</strong></div>
  <div><span>Prerequisites</span><strong>a sublayer f · one identity path</strong></div>
  <div><span>Core mechanism</span><strong>y = x + f(x) — the plus sign is the whole idea</strong></div>
  <div><span>Common mistakes</span><strong>believing it prevents overfitting, or that depth is now free</strong></div>
</div>

## The residual path provides an identity map

An ordinary layer replaces its input. A residual layer **edits** it:

$$y = x + f(x)$$

If $f$ learns zero, the layer is the identity and has done nothing. The default behaviour flips from "you must learn a useful transform" to **"when in doubt, leave it alone"** — and that is where all of the benefit comes from.

## Why the gradient gets through

$$\frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}$$

That $I$ is the point. Stack $L$ layers and backprop becomes

$$\frac{\partial y_L}{\partial x_0} = \prod_{l=1}^{L}\left(I + \frac{\partial f_l}{\partial x_{l-1}}\right)$$

Expand it and one term is $I \cdot I \cdots I = I$: **a path exists along which the gradient reaches layer 1 untouched.**

Without the residual it is a bare product $\prod_l \partial f_l / \partial x_{l-1}$. Slightly below 1 per layer and it decays exponentially; slightly above and it explodes. You have to tune the initialisation to sit exactly at the critical point.

## How gradients change without residual paths

A 40-layer MLP, tanh, initialisation set 20% below critical — the ordinary case of "not tuned perfectly." Same weights, same input; the only difference is the plus sign:

![gradient norm by depth, with and without residual](../assets/residual-gradient.svg)

Walking from layer 40 back to layer 1 the plain stack loses about **10³×** of its gradient, arriving at $2\times10^{-9}$. The residual one is flat and arrives at $2.1$. **At the same learning rate, the early layers of the plain stack are not training at all.**

There is no training here — one forward, one backward. The decay is a property of the architecture, not of insufficient optimisation.

## Three common misreadings

**"It prevents overfitting."** No — it fixes **optimisation**, not generalisation. The famous ResNet observation is that a 56-layer plain network has higher **training** error than a 20-layer one. That is not overfitting.

**"So we can go arbitrarily deep."** No. It removes vanishing gradients as the binding constraint; compute, memory, data and diminishing returns are all still there.

**"What if the dimensions don't match?"** They must, or the addition is undefined. CNNs use a $1\times1$ projection when downsampling; Transformers keep $d_\text{model}$ everywhere, so the question never arises.

<details markdown="1">
<summary><b>Deeper</b>: a residual net behaves like an ensemble of shallow ones</summary>

[Residual Networks Behave Like Ensembles](https://arxiv.org/abs/1605.06431) unrolls an $L$-layer residual network into $2^L$ paths of differing length — at each layer you either take $f$ or the identity.

Measured, the *effective* paths are short: most of the gradient comes from paths of length 10–30 even in a 100+ layer network. Delete a few layers at random and a residual network barely degrades; do that to a plain network and it collapses.

Which is why residual depth behaves more like width: it is not doing $L$ sequential steps of reasoning so much as summing many shorter transforms.

</details>

## The full sublayer also has a Dropout

The formulas above were simplified to keep the residual in focus. The 2017 sublayer is really:

$$\text{LayerNorm}\big(x + \text{Dropout}(f(x))\big)$$

Dropout zeroes a fraction $p$ of activations during training and divides the rest by $1-p$ to keep the expectation, then switches off entirely at inference. It stops the model leaning on any fixed set of channels, and what that buys is **generalisation**. The paper uses $p=0.1$ in three places: each sublayer's output, the embedding-plus-positional-encoding sum, and the attention weights.

**Placement matters: dropout applies to the branch output $f(x)$, never to $x$.**

$$\underbrace{x + \text{Dropout}(f(x))}_{\text{identity path intact}} \qquad\text{vs}\qquad \underbrace{\text{Dropout}(x) + f(x)}_{\text{path broken}}$$

Put dropout on $x$ and the clean route derived above gets randomly cut at every layer — the $I$ term stops holding and the residual has bought nothing. It has to stay inside the branch: **the residual stream must stay clean.**

Worth knowing: large-model pretraining now usually sets dropout to 0. With enough data, overfitting is not the binding constraint and dropout only slows convergence. It survives in finetuning, small models, and limited-data settings.

## How it interacts with normalisation

Three different problems, but their **relative placement** matters:

$$\underbrace{\text{Norm}(x + \text{Dropout}(f(x)))}_{\text{post-norm, 2017}} \qquad\text{vs}\qquad \underbrace{x + \text{Dropout}(f(\text{Norm}(x)))}_{\text{pre-norm, now}}$$

post-norm puts the norm *on* the residual path, so the clean identity route above is **interrupted** — every layer crosses a norm. That is exactly why the original Transformer needs warmup.

pre-norm moves the norm into the branch and leaves the identity path intact, at the cost of output scale accumulating with depth — so a final norm is added at the end.

![post-norm versus pre-norm residual paths](../assets/transformer-block.svg)

## Common interview questions

<details class="interview" markdown="1">
<summary>What problem do residual connections solve?</summary>

Optimisation in deep networks, not generalisation. A bare product of Jacobians decays or explodes exponentially; adding the identity gives $\partial y/\partial x = I + \partial f/\partial x$, so one route through the product leaves the gradient intact.

The evidence is ResNet's own observation: a 56-layer plain net has higher **training** error than a 20-layer one. Overfitting would show the opposite.

</details>

<details class="interview" markdown="1">
<summary>Why add instead of concatenate?</summary>

Concatenation (DenseNet) also preserves information, but the width grows with depth, and so do parameters and memory. Addition keeps the dimension fixed, so layers are stackable and identically sized.

Addition also makes "do nothing" a *reachable* solution ($f = 0$). With concatenation, later layers have to actively learn to ignore what was appended.

</details>

<details class="interview" markdown="1">
<summary>What happens to the variance of $x + f(x)$?</summary>

It accumulates: the residual stream's variance grows roughly linearly with depth. Two standard fixes — scale the residual branch's output projection at init by $1/\sqrt{2L}$ (GPT-2's trick), or apply a final norm at the end.

Left alone, deep layers reach a scale that saturates the softmax.

</details>

<details class="interview" markdown="1">
<summary>Pre-norm or post-norm — why has everything moved to pre-norm?</summary>

pre-norm trains more easily: no norm on the identity path, so the gradient has a clean route, warmup can be crude, and depth scales further.

post-norm sometimes ends up slightly better when it trains at all, since every layer's output is normalised. But it is very sensitive to the schedule, and in practice stability won.

</details>

<details class="interview" markdown="1">
<summary>Where does dropout go, and why not on the residual stream?</summary>

On the branch output: $x + \text{Dropout}(f(x))$.

Not on $x$, because that breaks the identity path — cut randomly at every layer, the $I$ in $\partial y/\partial x = I + \partial f/\partial x$ stops being reliable and the residual's contribution to gradient flow is cancelled out. The original paper also applies dropout after the embedding + positional encoding sum, and to the attention weights.

Note that large-model pretraining often sets dropout to 0 now: with enough data, overfitting isn't the binding constraint and it just slows convergence.

</details>

<details class="interview" markdown="1">
<summary>How does this relate to an LSTM's cell state?</summary>

Same trick. $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$ is an additive path through *time* when $f_t \to 1$; a residual connection is an additive path through *depth*.

One fixes "too many timesteps away", the other "too many layers deep".

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>Why is this an optimisation fix rather than an overfitting fix, and what experiment separates the two?</li>
    <li>What does the $I$ in $\partial y/\partial x = I + \partial f/\partial x$ do during backprop?</li>
    <li>Why does post-norm break that path?</li>
    <li>How does residual-stream variance behave with depth, and what are the two usual remedies?</li>
  </ol>
</div>

## Next

Attention, normalisation and residuals are all covered — time to assemble them: [the vanilla Transformer](vanilla-transformer.en.md).
