# Normalisation: BatchNorm and LayerNorm

[中文](normalization.md) · **English**

> Reading time: ~8 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>The problem</span><strong>activations held at a stable scale, layer after layer</strong></div>
  <div><span>Prerequisites</span><strong>a batch of activations · two learned vectors γ and β</strong></div>
  <div><span>Core mechanism</span><strong>which axis the mean and variance are taken over</strong></div>
  <div><span>Common mistakes</span><strong>BatchNorm meeting variable-length sequences or decoding</strong></div>
</div>

## The difference is a single sentence

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

## Why BatchNorm cannot be used in a language model

**1. Train and inference are two different functions.** BatchNorm uses the current batch during training and an accumulated running average at inference. It is the one module whose forward pass depends on which mode it is in — forget `model.eval()` and single requests disagree with batched ones.

**2. Variable length leaves nothing to estimate from.** In a batch of different-length sentences, position 500 might have three live examples. A mean and variance from three samples is noise.

**3. Decoding can have batch size 1.** One sample has zero batch variance, so $x - \mu$ is zero and the output carries no information. PyTorch refuses outright rather than returning something meaningless:

```
ValueError: Expected more than 1 value per channel when training,
got input size torch.Size([1, 8])
```

**4. It couples examples to each other.** Swap another sentence into the batch and your sentence's output changes. Harmless regularisation for image classification; unacceptable non-determinism for token-by-token generation.

LayerNorm is immune to all four, because it only ever looks at one token's own $d$ numbers.

## So which one, when

| Situation | Use | Why |
| --- | --- | --- |
| CNN classification, large fixed batch | **BatchNorm** | stable batch statistics, useful regularisation, often faster convergence |
| Any Transformer or language model | **LayerNorm / RMSNorm** | variable length, batch can be 1, generation must be deterministic |
| RNN / LSTM | **LayerNorm** | batch statistics aren't comparable across timesteps |
| Small batches (detection, segmentation, large-model finetuning) | **GroupNorm / LayerNorm** | BatchNorm's estimates get too noisy |
| RL and online learning | **LayerNorm** | the distribution moves with the policy; running averages always lag |
| GAN discriminators | often **InstanceNorm / LayerNorm** | stops same-batch examples leaking into each other |

One rule: **if "the same input must give the same output regardless of what else is in the batch" is a hard requirement, BatchNorm is out.**

Modern LLMs go one step further to RMSNorm: one fewer reduction, slightly less memory traffic, no measured loss in quality.

<details markdown="1">
<summary><b>Deeper</b>: why normalisation actually helps</summary>

The original paper argued it reduces internal covariate shift. [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) later undercut that: injecting distribution noise *after* BN leaves training just as fast and stable.

The better-supported account is that it **smooths the loss surface** — the gradient's Lipschitz constant shrinks, so a larger learning rate stops diverging.

An angle that gets missed: normalisation removes the weights' scale degree of freedom. $\text{Norm}(\alpha Wx) = \text{Norm}(Wx)$, so scaling weights changes nothing but the effective learning rate. That is why norm layers are usually excluded from weight decay.

</details>

## Verify it

[`../code/norm_compare.py`](../code/norm_compare.py) runs all three on the same activations, shows which axis each reduces over, and demonstrates the batch-size-1 failure.

## Interview questions

<details class="interview" markdown="1">
<summary>What is the difference between BatchNorm and LayerNorm?</summary>

The axis. BatchNorm reduces over the batch dimension, one mean per feature; LayerNorm reduces over the feature dimension, one mean per token.

The consequence: BatchNorm's output depends on the other examples in the batch, so it must keep running averages for inference and behaves differently in train and eval. LayerNorm behaves identically in both.

</details>

<details class="interview" markdown="1">
<summary>Why do Transformers use LayerNorm rather than BatchNorm?</summary>

Four reasons, any one of them sufficient: variable-length sequences leave too few samples at late positions; decoding can run at batch size 1, which PyTorch rejects outright; other examples in the batch would change this example's output, making generation non-deterministic; and the train/inference split is an extra failure mode during finetuning.

</details>

<details class="interview" markdown="1">
<summary>What does BatchNorm do at inference time?</summary>

It uses the running $\hat\mu, \hat\sigma^2$ accumulated during training instead of the current batch — which is why `model.eval()` matters. The classic symptom of forgetting it is a single request disagreeing with the same input inside a batch.

Those running averages are also estimated on the training distribution. If serving drifts, they are permanently wrong and nothing alerts.

</details>

<details class="interview" markdown="1">
<summary>What does RMSNorm drop, and why is that safe?</summary>

The mean subtraction and the $\beta$ bias; only the RMS rescaling remains.

Re-centring turns out to contribute little in practice — the rescaling does the work. Dropping a reduction is real bandwidth at scale. One caveat: accumulate the sum of squares in fp32, or bf16 will ruin the variance.

</details>

<details class="interview" markdown="1">
<summary>Pre-norm versus post-norm?</summary>

post-norm is $\text{Norm}(x + f(x))$ — the norm sits on the residual path. pre-norm is $x + f(\text{Norm}(x))$ — the residual path has no norm on it.

So post-norm's gradient crosses a norm at every layer and gets unstable with depth, which is why the original Transformer needs warmup. pre-norm leaves a clean identity path and tolerates simpler schedules, at the cost of output scale growing with depth — hence a final norm at the end.

</details>

<details class="interview" markdown="1">
<summary>What if the batch size has to be small?</summary>

GroupNorm (statistics within groups of channels) or LayerNorm. Neither depends on the batch, which is why GroupNorm is standard in detection and segmentation where batches are 2–4.

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>Without looking: which axis does each of the two reduce over?</li>
    <li>Why does BatchNorm need running averages and LayerNorm not?</li>
    <li>What happens to each at batch size 1?</li>
    <li>How long are $\gamma$ and $\beta$, and is that the same axis as the statistics?</li>
  </ol>
</div>

## Next

Normalisation controls the scale going into each layer. The other half of what makes depth trainable is the [residual connection](residual-connections.en.md).
