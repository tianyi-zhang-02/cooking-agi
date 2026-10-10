# Cross-entropy: from one prediction to a training loss

[中文](cross-entropy.md) · **English**

> Reading time: about 14 minutes · Runtime: PyTorch / CPU · Last reviewed: 2026-10

A model assigning 40% probability to the right answer and one assigning 1% might both count as wrong. Training should distinguish them: the second prediction needs a larger correction. Cross-entropy (CE) gives us a way to do that.

Start with three classes, then connect the calculation to language models. The first three sections cover the intuition. Continue into masks, reductions, and gradients when you want to implement it.

## How much probability did the right class receive? {#one-prediction}

Suppose we classify a support message as “refund,” “delivery,” or “other.” The predicted probabilities are `[0.2, 0.3, 0.5]`, and the label is “other.” Its loss is:

$$
\ell=-\log 0.5\approx0.6931.
$$

We use natural logarithms, so the unit is nats. Raising the target probability to 0.9 gives a loss of about 0.105; lowering it to 0.01 gives about 4.605. **Less probability for the correct answer means a larger penalty.**

<figure class="worked-update">
<ol>
<li><small>01 / MODEL OUTPUT</small><strong>Start with logits</strong><span>[log 2, log 3, log 5]<br>These are scores, not probabilities.</span></li>
<li><small>02 / PROBABILITIES</small><strong>Softmax → [0.2, 0.3, 0.5]</strong><span>The three classes share probability mass totaling 1.</span></li>
<li><small>03 / TARGET</small><strong>The label is “other”</strong><span>Select class 3: −log 0.5 ≈ 0.6931.</span></li>
</ol>
<figcaption>An original example chosen for easy arithmetic. The label selects what the loss measures; it is not supplied as the answer to the model's forward pass.</figcaption>
</figure>

Why not just count mistakes? The correct class might remain in second place while its probability rises from 0.1 to 0.4. Accuracy need not change, but CE reflects that improvement. Conversely, a lower CE does not guarantee better generated answers, calibration, or accuracy for every subgroup.

## Why compute directly from logits? {#stable-ce}

Let $z_c$ be a class score and $y$ the correct class. Substituting Softmax into the negative log gives:

$$
\ell=-\log p_y
=\log\sum_c e^{z_c}-z_y.
$$

Directly computing `exp(1000)` overflows. Subtract the row maximum $m$ first:

$$
\ell=\log\sum_c e^{z_c-m}-(z_y-m).
$$

A shared shift leaves Softmax unchanged. The largest exponential is now $e^0=1$, and all others are at most 1. Here is a small implementation with the class axis last, followed by a library comparison:

```python
import math
import torch
import torch.nn.functional as functional

def ce_from_logits(logits, targets):
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_partition = shifted.exp().sum(dim=-1).log()
    selected = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_partition - selected).mean()

logits = torch.tensor([[2., 3., 5.]], dtype=torch.float64).log()
targets = torch.tensor([2])
loss = ce_from_logits(logits, targets)
assert math.isclose(loss.item(), math.log(2))
torch.testing.assert_close(loss, functional.cross_entropy(logits, targets))
large_logits = torch.tensor([[1000., 999., -1000.]], dtype=torch.float64)
torch.testing.assert_close(
    ce_from_logits(large_logits, targets),
    functional.cross_entropy(large_logits, targets),
)
```

We never materialize probabilities and then take their logarithms. That route can round a tiny probability to zero before producing an infinite loss. Use established stable implementations for training; the handwritten version explains the calculation.

## Hard and soft targets {#soft-targets}

A hard target selects one class, equivalent to the one-hot distribution `[0, 0, 1]`. A soft target might be `[0.1, 0.2, 0.7]`: rather than pushing all probability onto class 3, it asks the prediction to match a distribution.

$$
\ell=-\sum_c q_c\log p_c.
$$

For the same prediction `[0.2, 0.3, 0.5]`, this soft-target CE is about 0.8869. Being larger than the hard-target loss of 0.6931 does not mean the model deteriorated. **The target changed, so those two numbers are not directly comparable.**

```python
soft_targets = torch.tensor([[.1, .2, .7]], dtype=torch.float64)
log_probabilities = functional.log_softmax(logits, dim=-1)
soft_loss = -(soft_targets * log_probabilities).sum(-1).mean()
torch.testing.assert_close(
    soft_loss, functional.cross_entropy(logits, soft_targets)
)
assert math.isclose(soft_loss.item(), 0.8869413785, abs_tol=1e-9)
```

Distillation can use a teacher distribution; label smoothing mixes one-hot targets with a smoothing distribution. Check nonnegativity, normalization, and class order. For fixed $q$, CE and $D_{KL}(q\|p)$ have the same model gradient because they differ only by $H(q)$, which does not depend on the model. Their values need not match. See the worked example in [distillation](../../05-post-training/distillation.en.md).

### Why is the gradient “prediction minus target”?

For one unweighted Softmax CE with a normalized target:

$$
\frac{\partial\ell}{\partial z_c}=p_c-q_c.
$$

The hard-target example gives `[0.2, 0.3, −0.5]`. Gradient descent lowers the first two scores and raises the correct one. The soft target gives `[0.1, 0.1, −0.2]`: a similar direction, but less pressure to drive class 3 toward 100%.

<details markdown="1">
<summary>Derivation: expand log Softmax</summary>

Using $\sum_c q_c=1$:

$$
\ell=\log\sum_j e^{z_j}-\sum_c q_c z_c.
$$

The first term's derivative with respect to $z_c$ is $p_c$; the second contributes $-q_c$. A batch mean adds division by the sample count. Class weighting or unnormalized targets changes this expression.

Differentiating $-\log p_y$ with respect to the probability gives $-1/p_y$, but differentiating with respect to logits also passes through Softmax. The former does not establish an exploding logit gradient at low probability.

</details>

## Binary CE: clamping is not a stability fix {#binary-ce}

Binary classification can output a single logit $z$ and obtain the positive-class probability through sigmoid. For $y\in\{0,1\}$:

$$
\begin{aligned}
\ell={}&-y\log\sigma(z)\\
&-(1-y)\log(1-\sigma(z)).
\end{aligned}
$$

This can also be computed stably from logits. Applying sigmoid and then clamping probabilities to `[1e-7, 1−1e-7]` may avoid `log(0)`, but changes the objective: clamp has zero gradient outside its retained interval.

Take a positive example with logit −100. Stable BCE gives loss about 100 and gradient about −1, strongly encouraging a higher score. Clamping gives loss only about 16.12 and zero gradient back to the logit—no correction for this severe mistake.

```python
wrong_logit = torch.tensor([-100.], dtype=torch.float64, requires_grad=True)
binary_target = torch.ones_like(wrong_logit)
stable_loss = functional.binary_cross_entropy_with_logits(wrong_logit, binary_target)
stable_loss.backward()
assert math.isclose(stable_loss.item(), 100.)
assert math.isclose(wrong_logit.grad.item(), -1.)
wrong_logit.grad = None
clamped_loss = -wrong_logit.sigmoid().clamp(1e-7, 1 - 1e-7).log().mean()
clamped_loss.backward()
assert wrong_logit.grad.item() == 0.
```

[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) combines sigmoid and BCE; do not apply sigmoid first. Multi-label tasks commonly use one such logit per label. If “refund” and “delivery” can both apply, their probabilities should not be forced to sum to 1.

## Language models: which positions count? {#mask-and-reduction}

A language model predicts vocabulary scores at each position, often shaped `[batch, sequence, vocabulary]`. Averaging the entire tensor is not the objective:

- **Align targets first.** A causal LM position predicts the next token; shift exactly once.
- **Select supervised positions.** Padding and unsupervised prompt targets do not count toward loss. Attention masking is a separate requirement.
- **Choose the denominator.** Here we average over valid target tokens, not padded sequence length.

Suppose two examples have valid token losses `[2]` and `[1, 1, 1]`. A token mean is $5/4=1.25$; a mean of per-example means is $(2+1)/2=1.5$. One weights tokens equally, the other weights examples equally. Either can be an intentional objective, but they are not interchangeable.

The same issue applies to gradient accumulation: naively averaging microbatch means gives the wrong token mean when their valid counts differ. Decide the denominator first; distributed training must also account for cross-rank counts and gradient reduction. See the [training loop](training-loop.en.md) and the complete conversation example in [SFT](../../05-post-training/sft-and-its-ceiling.en.md#mask-example).

<details markdown="1">
<summary>Implementation detail: why can a different label format change CE?</summary>

The [PyTorch CE interface](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html) distinguishes integer classes from probability targets. Integer targets support `ignore_index`; probability targets need explicit position selection. With class weights, their default mean denominators also differ.

For two `[0, 0]` logit rows, labels 0 and 1, and class weights `[1, 3]`, the integer-target weighted mean is $\log2$. Replacing labels with float one-hot targets gives $2\log2$ under the default mean. Same classes, **different reduction conventions**.

```python
weighted_logits = torch.zeros(2, 2, dtype=torch.float64)
hard_targets = torch.tensor([0, 1])
class_weights = torch.tensor([1., 3.], dtype=torch.float64)
one_hot_targets = functional.one_hot(hard_targets, 2).double()
hard_mean = functional.cross_entropy(weighted_logits, hard_targets, weight=class_weights)
soft_mean = functional.cross_entropy(weighted_logits, one_hot_targets, weight=class_weights)
assert math.isclose(hard_mean.item(), math.log(2))
assert math.isclose(soft_mean.item(), 2 * math.log(2))
```

PyTorch's multidimensional CE puts the class axis second, not necessarily last. For `[B,T,V]`, flatten to `[B×T,V]` or explicitly move the axis. Coincidentally equal dimensions can hide the mistake.

An entirely masked batch has no targets to average. Raise an error or handle the empty batch through an explicit training protocol. Do not silently turn its NaN into a “successful update” with `nan_to_num`.

</details>

## How do you check the implementation? {#checks}

Use small tensors before training a large model. This chapter's blocks run sequentially on CPU. The fuller [teaching implementation](../code/loss_contracts.py) checks inputs, empty masks, soft targets, and a single causal shift. Its [tests](../../site/tests/test_loss_contracts.py) compare gradients and extreme values too.

| Check | Expected result |
| --- | --- |
| Add a shared constant to all logits | CE stays unchanged |
| Increase the correct-class probability for a fixed target | Single-example hard CE decreases |
| Replace integer targets with one-hot, without weights | Same loss and gradient |
| Add unsupervised padding | Valid-token mean stays unchanged |
| Binary `z = −100, y = 1` | Finite loss and a corrective gradient |
| Ignore every target | Explicit handling, not a silently discarded NaN |

These checks catch implementation errors, not poor model quality. After they pass, return to data coverage, held-out evaluation, and the task itself.
