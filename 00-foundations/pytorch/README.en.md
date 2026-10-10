# PyTorch: from equations to training code

[中文](README.md) · **English**

> Reading time: about 8 minutes · Runtime: CPU · Last reviewed: 2026-10

Knowing the equation is not quite the same as implementing it correctly. Axes can silently misalign, a slice can change its source, gradients can stop unexpectedly, and a logged loss can use the wrong denominator. These chapters work through those gaps.

There is no need to memorize every API first. Follow a small batch from data to a parameter update, and learn each operation when it becomes useful.

Run each chapter's code blocks in order in one Python session, then start a fresh session for the next chapter. Optional TensorBoard integration is marked separately and is not required for earlier experiments.

## Reading order

| Chapter | Starting question | What to check |
| --- | --- | --- |
| [1 · Tensors and storage](tensors-and-storage.en.md) | Why did changing a slice also change the original? | Shape, dtype, indexing, views, and copies |
| [2 · Operations and shapes](operations-and-shapes.en.md) | Why does valid code compute the wrong loss? | Broadcasting, reductions, matrix products, and stability |
| [3 · Cross-entropy and loss](cross-entropy.en.md) | Why do two “mean losses” give different numbers? | LogSumExp, soft targets, masks, and denominators |
| [4 · Autograd](autograd.en.md) | Why is `.grad` None despite a computation graph? | Accumulation, leaves, detach, and VJPs |
| [5 · From batches to training](training-loop.en.md) | How do we check learning rather than just a working forward pass? | Modules, loaders, validation, logging, and recovery |

## Follow one small update

Start with two points, not a large model: `[1, 0]` belongs to class 0 and `[0, 2]` belongs to class 1. A linear classifier without bias produces two logits per point. We deliberately initialize its weights to zero for easy arithmetic; this is not an initialization recommendation for deep networks.

<figure class="worked-update">
<ol>
<li><small>01 / INPUT</small><strong>2 samples × 2 features</strong><span>Data [[1, 0], [0, 2]]<br>Labels [0, 1]</span></li>
<li><small>02 / FORWARD</small><strong>Zero logits for both classes</strong><span>W = 0 → logits = 0<br>Class probabilities are 0.5 each</span></li>
<li><small>03 / LOSS</small><strong>Mean CE ≈ 0.6931</strong><span>Each target has probability 0.5. Take −log, then average.</span></li>
<li><small>04 / BACKWARD</small><strong>Gradients, not an update yet</strong><span>First weight row: [−0.25, 0.5]<br>Second: [0.25, −0.5]</span></li>
<li><small>05 / UPDATE</small><strong>Move against the gradient</strong><span>SGD learning rate 0.2<br>W ← W − 0.2 × grad</span></li>
<li><small>06 / RECOMPUTE</small><strong>Same-batch CE ≈ 0.5787</strong><span>Target-class scores increased. This checks the update, not generalization.</span></li>
</ol>
<figcaption>An original worked example. Forward computes predictions, backward computes gradients, and the optimizer changes parameters. These are separate operations.</figcaption>
</figure>

Why is the first weight row's gradient `[−0.25, 0.5]`? For one example, softmax cross-entropy gives a logit gradient of predicted probabilities minus the one-hot target. Class 0 has errors −0.5 and 0.5 for the two examples. Multiplying by their respective inputs gives `[−0.5, 0]` and `[0, 1]`. Then average:

$$
\begin{aligned}
\nabla W_{0,:}
&=\tfrac12\big([-0.5,0]+[0,1]\big)\\
&=[-0.25,0.5].
\end{aligned}
$$

The updated weight rows are `[0.05, −0.1]` and `[−0.05, 0.1]`. The first point now scores 0.05 and −0.05 for the two classes; the second scores −0.2 and 0.2. **Labels never entered the model's forward computation, but they determined the update through the loss.**

This self-contained block checks the numbers in the diagram:

```python
import math
import torch

features = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
labels = torch.tensor([0, 1])
weights = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
optimizer = torch.optim.SGD([weights], lr=0.2)
optimizer.zero_grad(set_to_none=True)
logits = features @ weights.T
loss_before = torch.nn.functional.cross_entropy(logits, labels)
loss_before.backward()
expected_gradient = torch.tensor([[-0.25, 0.5], [0.25, -0.5]], dtype=torch.float64)
torch.testing.assert_close(weights.grad, expected_gradient)
torch.testing.assert_close(weights.detach(), torch.zeros_like(weights))
assert math.isclose(loss_before.item(), math.log(2))
optimizer.step()
with torch.no_grad():
    torch.testing.assert_close(weights, -0.2 * expected_gradient)
    loss_after = torch.nn.functional.cross_entropy(features @ weights.T, labels)
assert math.isclose(loss_after.item(), 0.5787059562, abs_tol=1e-9)
```

Chapter 1 explains storage; chapter 2 covers operations and shapes; chapter 3 unpacks the loss; chapter 4 traces gradients; chapter 5 adds loading, validation, and recovery. You do not need to know every function yet. First follow what each step produces, then study its implementation.

This example uses mean CE without class weights and SGD without momentum or weight decay. Changing those conditions changes the update. See the [PyTorch 2.8 CE](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html) and [SGD](https://docs.pytorch.org/docs/2.8/generated/torch.optim.SGD.html) contracts.

## Already comfortable with training code?

Start with the counterexamples: some non-contiguous tensors support `view`; subtracting `[B]` from `[B,1]` produces `[B,B]`; `eval()` does not disable gradients; and the final batch loss is not the epoch average. Understanding these cases makes larger models easier to debug.

The examples are original, small experiments using common PyTorch 2.x interfaces. Numerical validation is recorded separately as **PyTorch 2.8.0 / CPU**, not as a claim to use the latest release or measure GPU performance. From the repository root:

```bash
python -m unittest discover -s site/tests -p test_pytorch_notes.py -v
```

Without PyTorch, the numerical tests explicitly skip while document checks still run. A skip is not a successful numerical validation.

## Where to go next

For model implementations, continue with the [from-scratch labs](../code/README.en.md). To adapt the loop to language modeling, read [one training update](../deep-dives/training-step.en.md), which covers token targets, loss masks, and accumulation. If training fails, use [the diagnostic guide](../deep-dives/generalization.en.md) before assuming the model needs to be larger.
