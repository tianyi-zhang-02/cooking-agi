# From one matrix multiply to gradients, memory, and multiple GPUs

[中文](matrix-gradients-to-systems.md) · **English**

> Last reviewed: 2026-11 · Prerequisites: multiplication, sums, and the idea of a derivative

Why does a single line, `Z = X @ W`, lead to saved activations, backward computation, and communication between GPUs during training? Follow a tiny linear layer through one update, then locate the sources of those system costs.

## A forward pass combines input features {#forward}

Let each row be one input. Leave out the bias for now:

$$
X\in\mathbb{R}^{B\times I},\quad W\in\mathbb{R}^{I\times O},\quad Z=XW\in\mathbb{R}^{B\times O}.
$$

B counts input rows, I counts input features, and O counts output features. An individual output is

$$Z_{bo}=\sum_{i=1}^{I}X_{bi}W_{io}.$$

For one input and one output feature, multiply each input feature by its corresponding weight and add the results:

```text
X = [[1, 2]]        W = [[1, 0],       Z = [[1, 2]]
                         [0, 1]]
```

W is the identity matrix, so this example preserves the input. Learned weights generally recombine its features. PyTorch `nn.Linear` stores its weight as `[O,I]` and computes `X @ weight.T + bias`; the mathematical W used here follows the transposed convention. See the [Linear shape documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html).

## The loss specifies what should change {#loss}

Choose a target `T=[[0,1]]` and a teaching loss:

$$L=\frac12\sum_{b,o}(Z_{bo}-T_{bo})^2.$$

This is half the **sum** of squared errors, not the default elementwise mean used by MSELoss. The error is `[1,1]`, giving a loss of 1.

First calculate the loss gradient at the layer's output:

$$G=\frac{\partial L}{\partial Z}=Z-T=[[1,1]].$$

G has the same shape as Z. Each component gives the local rate at which the loss changes when that output changes slightly. It does not prescribe a replacement value for the output. A different loss changes how G is calculated; the linear layer's backward rules below remain the same.

## Why is the weight gradient X transpose times G? {#backward}

Consider `W[i,o]`. It affects output o for every input row. Its effect on row b is multiplied by `X[b,i]`. Add the contributions:

$$
\frac{\partial L}{\partial W_{io}}
=\sum_b\frac{\partial L}{\partial Z_{bo}}\frac{\partial Z_{bo}}{\partial W_{io}}
=\sum_bG_{bo}X_{bi}.
$$

Arrange those individual derivatives into a matrix:

$$\boxed{\frac{\partial L}{\partial W}=X^\top G}.$$

Check the shapes: `[I,B] @ [B,O] → [I,O]`, matching W. For our numbers:

```text
dW = [[1],  @ [[1, 1]] = [[1, 1],
      [2]]                 [2, 2]]
```

The second input feature equals 2, so the same upstream gradient creates twice the weight gradient in that row. Transposing X expresses the sum of parameter contributions across input rows. Backpropagation is **not matrix inversion**.

If another layer precedes this one, propagate the gradient to its output. Each input feature affects several outputs in the same row, so this time sum over output features:

$$
\frac{\partial L}{\partial X_{bi}}
=\sum_o\frac{\partial L}{\partial Z_{bo}}\frac{\partial Z_{bo}}{\partial X_{bi}}
=\sum_oG_{bo}W_{io}.
$$

In matrix form:

$$\boxed{\frac{\partial L}{\partial X}=GW^\top}.$$

The shapes are `[B,O] @ [O,I] → [B,I]`. With a bias in `Z=XW+b`, sum G over input rows to obtain its gradient. If X is raw data that needs no gradient, this dX computation may be unnecessary. When X is an earlier layer's output, dX enters that layer's chain rule.

Here W is the identity, so dX is also `[[1,1]]`. It measures the loss's sensitivity to the input; it does not instruct the trainer to modify the dataset. Use the weights from the corresponding forward pass to compute dX, not weights that have already been updated.

## Computing a gradient does not update a parameter {#update}

Take a simple SGD step with learning rate 0.1:

$$W_{\mathrm{new}}=W-0.1\frac{\partial L}{\partial W}.$$

This gives:

```text
W_new = [[ 0.9, -0.1],      X @ W_new = [[0.5, 1.5]]
         [-0.2,  0.8]]      New loss = 0.25
```

The loss falls from 1 to 0.25 for this input, parameter value, and learning rate. This one step does not establish convergence for arbitrary rates or demonstrate LLM training quality.

Compare the hand calculation with PyTorch autograd:

```python
import torch

inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
weights = torch.eye(2, dtype=torch.float64, requires_grad=True)
target = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

loss = 0.5 * ((inputs @ weights - target) ** 2).sum()
loss.backward()
torch.testing.assert_close(weights.grad, torch.tensor(
    [[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64))
torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
torch.testing.assert_close(weights, torch.eye(2, dtype=torch.float64))

with torch.no_grad():
    weights -= 0.1 * weights.grad
new_loss = 0.5 * ((inputs @ weights - target) ** 2).sum()
torch.testing.assert_close(new_loss, torch.tensor(0.25, dtype=torch.float64))
```

`backward()` computes the gradient. The line inside `no_grad` performs the SGD update; a training loop normally delegates that work to `optimizer.step()`. Repeated steps also require deliberate gradient clearing or accumulation. Continue with [autograd](../pytorch/autograd.en.md) and the [training loop](../pytorch/training-loop.en.md). This exact example was checked on CPU with PyTorch 2.8.0.

## Where does memory enter the formula? {#memory}

In an intermediate layer, X is usually an activation produced by earlier computation. Since `dW=XᵀG` needs the forward input X during backward, the system must arrange for those values to be available. It can:

- Keep them until backward needs them, using memory capacity.
- Recompute parts of the forward pass, spending more computation.
- Move them elsewhere and retrieve them later, adding transfer and management work.

Which tensors you need to keep depends on the operator, the gradients you want, and how the computation runs. Not every intermediate result has to remain on the GPU. PyTorch autograd saves the data each operation needs; see [Saved tensors](https://docs.pytorch.org/docs/2.14/notes/autograd.html#saved-tensors).

Activations are only one part of training memory. Parameters, gradients, optimizer state, and temporary buffers also matter. Inference usually avoids this training backward graph and optimizer state, but autoregressive generation keeps KV caches and engine buffers. Account for the two workloads separately; continue with [precision and memory](precision-and-memory.en.md).

## Why do multiple GPUs combine gradients? {#distributed}

Split the input rows between two GPUs, initially using the same W:

$$
\frac{\partial L_{\mathrm{sum}}}{\partial W}
=X_1^\top G_1+X_2^\top G_2.
$$

This is the algebra behind aggregating data-parallel gradients. Each GPU knows its local contribution. Combining them must match the intended objective and the backend's reduction convention.

Add a second sample to our example and give one to each GPU, both starting from identity weights:

| Process | Input X | Target T | Output gradient G | Local weight gradient |
| --- | --- | --- | --- | --- |
| Rank 0 | [1, 2] | [0, 1] | [1, 1] | [[1, 1], [2, 2]] |
| Rank 1 | [0, 1] | [1, 0] | [−1, 1] | [[0, 0], [−1, 1]] |

Adding the gradients gives dW = `[[1,1],[1,3]]`. Contributions partly cancel in the first column and reinforce each other in the second. With the summed loss and learning rate 0.1, the update reduces total loss from 2 to 1.14. The accompanying script reproduces this as its second case.

For the average over two samples, divide the gradient by 2. PyTorch DDP normally averages gradients across processes. With one sample per process, that matches the global sample mean, not our earlier global sum. See the [DDP note on gradient averaging](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html). Also check the MSE definition: dividing our loss by B gives a sample mean, whereas default MSE divides squared errors by B×O with no factor of 1/2. Those definitions happen to agree only because O=2 here.

If the objective is a global mean and the GPUs have different valid-token counts, averaging their local means usually differs from the global token mean. A backend may sum or average gradients, so loss denominators must match its behavior. Do not copy an unexplained scaling factor. [Training stability](../../practice/runtime-reading/training-stability.en.md#normalization) works through batches with 100 and 10 valid tokens.

Tensor parallelism can split dimensions of a matrix multiply, changing when partial results communicate. FSDP mainly shards training state and obtains it as needed. Before learning the abbreviations, write down the shapes and ownership of X, W, and G on each GPU. Continue with [one tensor-parallel layer](../../06-systems/tensor-parallel.en.md).

## Continue into an LLM {#next}

This linear layer connects forward computation, gradient propagation, parameter updates, storage, and communication. Apply the same approach to attention next: construct Q, K, and V, then track `QKᵀ`, softmax, and the weighted sum. Attention weights depend on the input, and softmax introduces nonlinearity, so the entire layer is not a fixed W. Autoregressive decoder self-attention also uses a causal mask; not all attention blocks access to later positions. See [the original paper, §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2) and the [Transformer walkthrough](../transformer.en.md).

Change one condition at a time: add a batch row, change the loss denominator, or make W rectangular. Then check the result again. The accompanying [standard-library experiment](../code/matrix_gradient_checks.py) checks every X/W derivative against finite differences in three small cases, including rectangular, non-identity weights. It checks derivatives, not GPU performance:

```bash
python3 00-foundations/code/matrix_gradient_checks.py
```
