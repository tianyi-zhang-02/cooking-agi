# Operations and shapes: valid code can compute the wrong thing

[中文](operations-and-shapes.md) · **English**

> Reading time: about 15 minutes · Prerequisite: [tensors and storage](tensors-and-storage.en.md) · Last reviewed: 2026-10

Two predictions are exactly right, yet mean squared error is not zero. Before changing the optimizer, inspect the shapes: predictions might be `[2,1]` and targets `[2]`. Broadcasting accepts them and silently changes the comparison.

Rather than memorize functions, ask which elements interact, which axes are reduced, and which axes remain.

## Broadcasting aligns trailing axes, not sample identities

Broadcasting aligns axes from the right. Lengths must match or one must be 1; missing leading axes act as singleton axes.

```text
prediction    [B, 1]
target           [B]    → treated as [1, B]
result        [B, B]

Each prediction is compared with every target,
not just the target from the same sample.
```

```python
import torch

prediction = torch.tensor([[1.0], [3.0]])
target = torch.tensor([1.0, 3.0])
pairwise_error = (prediction - target).square()
assert pairwise_error.tolist() == [[0.0, 4.0], [4.0, 0.0]]
assert pairwise_error.mean().item() == 2.0
matched_error = (prediction.squeeze(-1) - target).square()
assert matched_error.mean().item() == 0.0
```

A `[B,B]` result is sometimes intentional, such as all query–candidate scores in contrastive learning. It usually is not the desired regression error. Broadcasting checks sizes, not identities.

Empty axes need care: `[0]` and `[1]` can produce `[0]`, so “take the larger length” is not a universal rule. In-place operations cannot expand their destination's storage shape either. See [Broadcasting semantics](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html).

## Avoiding input copies does not eliminate output memory

`expand` can use zero strides to read the same storage repeatedly; `repeat` copies values. Either way, broadcasting small inputs into a large result can still allocate a large output.

For example, `[B,1,D] - [1,N,D]` produces `[B,N,D]`. If only squared pairwise distances are needed, compare an implementation of `||q||² + ||k||² - 2qkᵀ` that avoids that three-dimensional difference. Consider cancellation error too: it is not automatically more accurate in every dtype.

Several expanded elements can refer to the same memory location. Clone before treating an expanded view as independently writable. Backward sums gradients over broadcast paths; the next chapter illustrates this with a bias.

## Reductions: what is the denominator?

For `[B,T,D]`, `mean(dim=-1)` computes each token's feature mean with shape `[B,T]`. Adding `keepdim=True` gives `[B,T,1]`, which can be subtracted along the feature axis.

| Operation | Result | Check |
| --- | --- | --- |
| `sum` / `mean` / `prod` | Reduced values | Omitting dim commonly reduces all elements |
| `max(dim=...)` / `min(dim=...)` | Values and indices | Use `amax` / `amin` when only values are needed |
| `argmax` / `argmin` | Indices | Neither the selected values nor a differentiable selection rule |
| `topk` | Values and indices | Tied indices do not have a guaranteed stable order |
| `var` / `std` | Variance / standard deviation | `correction` changes the denominator; default is 1 |
| `median` / `quantile` | Median / quantile | With even counts, `median` returns the lower middle value |

Normalization commonly uses population variance with `correction=0`. A single observation with `correction=1` lacks sufficient degrees of freedom; an empty mean is undefined too. Before adding epsilon to a NaN, check whether the statistic is meaningful.

```python
samples = torch.tensor([2.0, 6.0])
assert samples.var(correction=0).item() == 4.0
assert samples.var(correction=1).item() == 8.0
assert samples.median().item() == 2.0
assert samples.quantile(0.5).item() == 4.0

token_loss = torch.tensor([[1.0, 3.0, 99.0], [2.0, 99.0, 99.0]])
valid = torch.tensor([[True, True, False], [True, False, False]])
token_average = token_loss[valid].mean()
sequence_average = torch.stack([row[mask].mean() for row, mask in zip(token_loss, valid)]).mean()
assert token_average.item() == 2.0
assert sequence_average.item() == 2.0
```

These averages happen to agree; they are not equivalent. Change the second sequence's only valid loss from 2 to 8: the token mean becomes 4, while the mean of sequence means becomes 5. One weights tokens equally, the other sequences. They produce different gradients. An empty valid set requires an explicit skip or error policy rather than an accidental NaN.

## Matrix products: what do the last two axes do?

`*` means elementwise multiplication with possible broadcasting. `@` / `matmul` means a matrix product. For `[M,K] @ [K,N] → [M,N]`, K is the summed-over dimension.

| API | Typical inputs | Distinction |
| --- | --- | --- |
| `dot` / `vdot` | Two one-dimensional vectors | `vdot` conjugates the first complex input |
| `mv` | Matrix and vector | `[M,K] × [K] → [M]` |
| `mm` | Two matrices | No batch broadcasting |
| `bmm` | Two three-dimensional tensors | Equal batch counts; no batch broadcasting |
| `matmul` / `@` | Vectors, matrices, or batches | Matrix product on trailing axes, broadcasting on leading batch axes |
| `addmm` | Bias and two matrices | `beta * input + alpha * (left @ right)` |
| `baddbmm` / `addbmm` | Bias and batched matrices | Preserve batch versus sum products over batch |

```python
queries = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
keys = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
scores = queries @ keys.transpose(-2, -1)
assert scores.shape == (2, 3, 5)
torch.testing.assert_close(scores, torch.bmm(queries, keys.transpose(1, 2)))
torch.testing.assert_close(scores, torch.einsum("bqd,bkd->bqk", queries, keys))

offset = torch.zeros(3, 5)
summed = torch.addbmm(offset, queries, keys.transpose(1, 2))
torch.testing.assert_close(summed, scores.sum(dim=0))
```

Each batch has 3 queries, 5 keys, and 4 features. The output scores every query against every key. `einsum` names the axes explicitly for inspection, without promising better performance. [addbmm](https://docs.pytorch.org/docs/2.8/generated/torch.addbmm.html) removes the batch axis; it is not simply `bmm` with a bias.

Useful constructors have concrete roles: `eye` creates an identity matrix; `diag` constructs a diagonal matrix from a vector or extracts a diagonal from a matrix, while `diagonal` also supports batched extraction. `triu` / `tril` keep upper/lower triangles; an upper triangle with `diagonal=1` often marks future positions. `t()` accepts inputs with at most two dimensions. For batched transposition, use `transpose(-2,-1)` rather than treating high-dimensional `.T` as a last-two-axis swap.

## One target value per row: gather

In three-class classification, each row can have a different correct class. `index_select` selects the same columns for all rows; we need a row-specific selection.

```python
logits = torch.tensor([[3.0, 1.0, -2.0], [0.0, 2.0, 4.0]])
labels = torch.tensor([0, 2], dtype=torch.long)
log_probs = logits.log_softmax(dim=-1)
selected = log_probs.gather(1, labels[:, None]).squeeze(1)
loss = -selected.mean()
torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, labels))
```

`gather` requires the index tensor to have the same number of dimensions as the input; ours is `[B,1]`. It does not broadcast the index for you. Add an axis to the labels, select, then remove that axis. Class count belongs to the task definition, not the number of unique labels in each batch.

## Numerical functions: choose the operation, not just more printed digits

`exp`, `log`, and `sqrt` act elementwise. `logsumexp` computes the log of a sum of exponentials, not a more precise ordinary sum. Exponentiating large logits can overflow; taking log after softmax can encounter rounded zero probabilities.

```python
large = torch.tensor([1000.0, 999.0])
assert torch.isinf(large.exp()).all()
stable = torch.logsumexp(large, dim=0)
torch.testing.assert_close(stable, torch.tensor(1000.3132617))
assert torch.isfinite(large.log_softmax(dim=0)).all()
assert torch.exp(torch.tensor([1, 2])).is_floating_point()
assert torch.round(torch.tensor([0.5, 1.5, 2.5])).tolist() == [0.0, 2.0, 2.0]
```

`round` uses ties-to-even; `ceil` rounds toward positive infinity and `floor` toward negative infinity. Near zero, `log1p` is more accurate than first adding one and then taking a log. Real `log` inputs must be positive and real `sqrt` inputs nonnegative. Trace invalid values to their source.

Specify distances too: `linalg.vector_norm(vector, ord=2)` gives Euclidean length, while `ord=1` sums absolute values. `dist(left,right,p=2)` gives the Euclidean distance between tensors. Neither is normalized cosine similarity.

Floating-point operation order matters. `equal` is not approximate and does not enforce identical dtypes; `allclose` uses absolute/relative tolerances and can broadcast shapes. For model tests, explicitly check shapes and use `torch.testing.assert_close` for values and types. Printing more decimal places does not increase computational precision.

These rules are checked against [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html), the [math and reduction API](https://docs.pytorch.org/docs/stable/torch.html), and [CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html). The examples are small tensor calculations, not model-training or performance reproductions.
