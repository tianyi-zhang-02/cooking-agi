# Normalisation: BatchNorm, LayerNorm, and RMSNorm

[中文](normalization.md) · **English**

> Reading time: ~12 min · Level: core · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>controlling numerical scale and gradient paths in deep networks</strong></div>
  <div><span>Prerequisites</span><strong>residual connections · mean and variance · Jacobians</strong></div>
  <div><span>Core mechanism</span><strong>the reduction axes and the Norm's position around the residual branch</strong></div>
  <div><span>Common mistakes</span><strong>forgetting final LN or claiming that every BatchNorm necessarily leaks the future</strong></div>
</div>

## The 30-second mental model

| Concept | One-line memory | Interview keywords |
| --- | --- | --- |
| Why normalize? | Give each sublayer predictably scaled inputs and reduce sensitivity to parameter scale | stable activations · conditioning · larger learning rate |
| Why not BatchNorm? | One token should not depend on other batch members or future positions | variable length · padding · train/eval mismatch · causality |
| Pre-LN vs Post-LN | Pre-LN moves Norm off the residual highway, preserving an identity gradient path | $I+J_fJ_{\mathrm{LN}}$ · final LN |
| RMSNorm | Keep re-scaling and drop re-centering | RMS only · no mean subtraction · cheaper reduction |

> **Normalization is not about keeping every representation permanently at mean zero and variance one. It controls what each sublayer receives and makes a deep network easier to optimize.**

## The core difference: normalization axes

**BatchNorm takes its statistics across a batch of examples. LayerNorm takes them across one example's own features.**

Everything else — variable length, batch size 1, autoregressive decoding, whether train and inference agree — follows from that one sentence.

![which axis each norm averages over](../assets/norm-axes.svg)

## The formulas

**BatchNorm**, per feature $j$, over the batch:

$$\mu_j = \frac{1}{N}\sum_{i} x_{ij}, \qquad y_{ij} = \gamma_j\,\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}} + \beta_j$$

**LayerNorm**, per example $i$, over the features:

$$\mu_i = \frac{1}{d}\sum_{j} x_{ij}, \qquad y_{ij} = \gamma_j\,\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}} + \beta_j$$

**RMSNorm** — LayerNorm without the mean subtraction and without $\beta$:

$$y_{ij} = \gamma_j\,\frac{x_{ij}}{\sqrt{\frac{1}{d}\sum_k x_{ik}^2 + \epsilon}}$$

In all three, $\gamma$ and $\beta$ are **per feature**, length $d$. Only the statistics axis changes.

## Four concepts: answer first, then go deeper

<details class="interview" markdown="1">
<summary>1. Why do deep networks need normalization?</summary>

**Quick learning**

A residual network repeatedly applies

$$
x_{\ell+1}=x_\ell+f_\ell(x_\ell).
$$

If branch outputs have uncontrolled scale, residual additions let activation scale drift with depth. During backpropagation, curvature and gradient scale may also differ sharply across layers and directions. This does not mean values must monotonically explode; it means **one learning rate has trouble serving every layer and direction**.

Normalization gives each sublayer more predictably scaled inputs, reducing sensitivity to initialization and parameter scale. It often improves effective conditioning, which makes larger learning rates and deeper optimization practical.

**Interview answer**

> The residual stream accumulates updates from every layer. If sublayer input scales drift, activations and gradients become harder to control and the optimization problem becomes ill-conditioned. LayerNorm or RMSNorm brings each token's sublayer input back to a predictable scale, improving gradient propagation and making deep Transformers easier to train.

<details markdown="1">
<summary><b>Deep dive</b>: residual scale, conditioning, and what “stable” means</summary>

If residual updates are temporarily treated as approximately uncorrelated,

$$
\operatorname{Var}(x_L)
\approx
\operatorname{Var}(x_0)+
\sum_{\ell=0}^{L-1}\operatorname{Var}\big(f_\ell(x_\ell)\big).
$$

This explains why residual-stream scale may grow with depth. Real networks contain correlations and adaptive weights, so it is not a theorem that variance must grow linearly. The more precise statement is: **depth creates scale drift, while normalization gives each branch a controlled input scale.**

From an optimization view, a Hessian with widely separated eigenvalues makes one learning rate unstable in high-curvature directions and slow in low-curvature ones. Normalization does not guarantee a well-conditioned global Hessian, but it reduces scale disparities between layers and often makes gradients smoother under parameter perturbations.

Pre-LN does not keep the residual stream $x_\ell$ itself at unit variance. It normalizes the value sent into attention or the FFN, which is why a final LN is still used before the output head.

</details>

</details>

<details class="interview" markdown="1">
<summary>2. Why do Transformers usually avoid BatchNorm?</summary>

**Quick learning**

1. **Variable length and padding**: padding can contaminate statistics; even with masking, late positions have few valid samples.
2. **Batch-composition dependence**: changing neighboring examples changes this token's output, and small batches produce noisy statistics.
3. **Train/inference mismatch**: training uses current-batch statistics, while inference normally uses running statistics. Autoregressive decoding often has tiny batches or batch size one.
4. **Possible causality violation**: if the reduction includes the time axis, future tokens affect the mean and variance used for past tokens, bypassing the causal attention mask.

LayerNorm reduces only over one token's hidden features. It is independent of the batch, other positions, and train versus inference mode.

**Interview answer**

> Transformers use LayerNorm because it normalizes every token independently over the hidden dimension. It is independent of batch size, sequence length, and other tokens, and it behaves identically in training and inference. BatchNorm introduces batch-composition, padding, and running-statistics problems; if its reduction also spans time, it leaks future tokens into past positions.

<details markdown="1">
<summary><b>Deep dive</b>: when does BatchNorm really leak the future?</summary>

The answer depends on tensor layout and reduction axes; “BatchNorm always leaks” is too broad.

- For $X\in\mathbb R^{B\times T\times d}$, a separate normalization at every position $t$ that reduces only over $B$ does not directly read future tokens from the same sequence. It still depends on other batch members, and each position may have a different valid sample count.
- A common sequence use of <code>BatchNorm1d</code> reduces each channel over both $B$ and $T$. Then $\mu_j$ and $\sigma_j$ include tokens with $t'>t$. Position $t$ already contains future information before attention runs, so a causal mask cannot block this side channel.
- Flattening $B\times T$ before BatchNorm creates the same leak.

The precise conclusion is:

> **BatchNorm breaks autoregressive causality when its statistics include time. Even when time is excluded, batch dependence, padding, small batches, and train/eval mismatch still make it a poor default for language models.**

Batch size one does not mean BatchNorm must fail at inference: eval mode can use running statistics. The deeper issue is that those statistics came from the training distribution rather than the current token, and training and inference execute different functions.

</details>

</details>

<details class="interview" markdown="1">
<summary>3. Where do Pre-LN and Post-LN go, and why is Pre-LN more stable?</summary>

**Write the structure first**

<pre><code>Post-LN (2017 original)          Pre-LN (common today)
x = LN(x + Attn(x))             x = x + Attn(LN(x))
x = LN(x + FFN(x))              x = x + FFN(LN(x))
                                 ...
                                 x = LN(x)   ← final LN</code></pre>

**Interview answer**

> Post-LN places LayerNorm after residual addition, so the main-path gradient crosses an LN in every block. Pre-LN moves the LN into the branch and leaves a direct identity path in the residual stream. Every block Jacobian therefore contains an explicit identity term, making deep gradient propagation more stable. The trade-off is that the residual stream is not normalized after every addition, so standard Pre-LN uses a final LN after all blocks.

<details markdown="1">
<summary><b>Deep dive</b>: the identity path in the Jacobian</summary>

Ignoring the two-branch detail, Post-LN is

$$
x_{\ell+1}=\operatorname{LN}\big(x_\ell+f_\ell(x_\ell)\big),
$$

with Jacobian

$$
\frac{\partial x_{\ell+1}}{\partial x_\ell}
=
J_{\operatorname{LN}}\big(I+J_{f_\ell}\big).
$$

Across many layers, the gradient repeatedly multiplies by $J_{\operatorname{LN}}$. This does not imply that every layer necessarily shrinks the gradient, but it interrupts a pure identity highway and makes training more sensitive to initialization, residual scaling, and learning-rate warmup.

Pre-LN is

$$
x_{\ell+1}=x_\ell+f_\ell\big(\operatorname{LN}(x_\ell)\big),
$$

so

$$
\frac{\partial x_{\ell+1}}{\partial x_\ell}
=
I+J_{f_\ell}J_{\operatorname{LN}}.
$$

Even when the branch contribution is small, $I$ remains a direct gradient path. The careful claim is that Pre-LN **reduces dependence** on warmup; it does not prove every setup needs no warmup.

Pre-LN has its own trade-off. Residual-stream scale can grow, and later-layer updates may become small relative to the main stream, reducing effective depth. Final LN repairs the output scale, not every expressivity issue.

</details>

</details>

<details class="interview" markdown="1">
<summary>4. Why can RMSNorm work with re-scaling alone?</summary>

**Quick learning**

LayerNorm performs both

$$
x\longmapsto x-\mu
\quad\text{(re-centering)},
\qquad
x\longmapsto \frac{x}{\sqrt{\operatorname{Var}(x)+\epsilon}}
\quad\text{(re-scaling)}.
$$

RMSNorm removes mean subtraction, usually omits $\beta$, and keeps

$$
x\longmapsto
\frac{x}{\sqrt{\frac{1}{d}\sum_jx_j^2+\epsilon}}\odot\gamma.
$$

Its success in modern LLMs suggests that controlling vector magnitude and sublayer input scale often matters more than forcing every token's feature mean to zero.

**Interview answer**

> RMSNorm keeps LayerNorm's scale control and learned gain, but removes mean subtraction and bias. It eliminates a mean reduction, simplifying computation and communication. In practice it often retains similar quality, suggesting that re-scaling is usually more important than re-centering.

<details markdown="1">
<summary><b>Deep dive</b>: does RMSNorm prove that the mean never matters?</summary>

No. Its success is strong empirical evidence, not a mathematical proof. LayerNorm divides by the standard deviation,

$$
\sqrt{\frac{1}{d}\sum_j(x_j-\mu)^2+\epsilon},
$$

while RMSNorm divides by the second moment,

$$
\sqrt{\frac{1}{d}\sum_jx_j^2+\epsilon}.
$$

They are close when feature means are near zero and differ when the mean shifts. Surrounding linear layers, bias-free designs, and training can adapt to that difference.

RMSNorm's engineering gain is one fewer mean reduction and less associated synchronization and memory traffic. Implementations normally accumulate sums of squares in fp32 to avoid bf16 or fp16 numerical error.

</details>

</details>

## How to choose a normalization method

| Situation | Use | Why |
| --- | --- | --- |
| CNN classification, large fixed batch | **BatchNorm** | stable batch statistics, useful regularisation, often faster convergence |
| Any Transformer or language model | **LayerNorm / RMSNorm** | variable length, batch can be 1, generation must be deterministic |
| RNN / LSTM | **LayerNorm** | batch statistics aren't comparable across timesteps |
| Small batches (detection, segmentation, large-model finetuning) | **GroupNorm / LayerNorm** | BatchNorm's estimates get too noisy |
| RL and online learning | **LayerNorm** | the distribution moves with the policy; running averages always lag |
| GAN discriminators | often **InstanceNorm / LayerNorm** | stops same-batch examples leaking into each other |

One rule: **if the same token must produce the same output in different batches, do not let normalization use batch statistics.**

Modern LLMs often go one step further to RMSNorm: one fewer mean reduction and simpler compute, synchronization, and memory traffic, usually with similar quality in practice.

<details markdown="1">
<summary><b>Supplement</b>: why internal covariate shift is not the full explanation</summary>

The original paper argued it reduces internal covariate shift. [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) later undercut that: injecting distribution noise *after* BN leaves training just as fast and stable.

The safer account is that normalization reduces sensitivity to parameter scale and often makes the loss landscape and gradients smoother. This is not an unconditional guarantee on the global condition number of every network.

An angle that gets missed: normalisation removes the weights' scale degree of freedom. $\text{Norm}(\alpha Wx) = \text{Norm}(Wx)$, so scaling weights changes nothing but the effective learning rate. That is why norm layers are usually excluded from weight decay.

</details>

## Verify it

[`../code/norm_compare.py`](../code/norm_compare.py) runs all three on the same activations, shows which axis each reduces over, and demonstrates the batch-size-1 failure.

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>For activations shaped $B\times T\times d$, which axis does LayerNorm reduce over?</li>
    <li>Why is “BatchNorm always leaks the future” too broad? Which implementation really leaks?</li>
    <li>Write Pre-LN and Post-LN and locate the final LN.</li>
    <li>Why does the Pre-LN Jacobian contain an identity path?</li>
    <li>What does RMSNorm remove, and does its success prove re-centering never matters?</li>
  </ol>
</div>

## Next

Normalisation controls the scale going into each layer. The other half of what makes depth trainable is the [residual connection](residual-connections.en.md).
