# Tensors and storage: which data did you change?

[中文](tensors-and-storage.md) · **English**

> Reading time: about 15 minutes · Prerequisite: Python lists and indexing · Last reviewed: 2026-10

You take a few rows from a batch, change a value, and discover that the original changed too. Often nothing mysterious happened: two tensors refer to the same memory.

Keep three questions separate: where the values live, how their shape maps onto memory, and whether autograd records an operation. This chapter covers the first two; [autograd](autograd.en.md) covers the third.

## Name the axes first

Suppose a batch contains 2 sequences, each with 3 tokens represented by 4 numbers. Its shape is `[2,3,4]`, which we interpret as `[batch, time, feature]`. Those names are our convention, not information PyTorch automatically understands.

| Inspection | Result here | Question answered |
| --- | --- | --- |
| `values.shape` / `values.size()` | `(2,3,4)` | How long is each axis? |
| `values.ndim` | 3 | How many axes? |
| `values.numel()` | 24 | How many elements? |
| `values.dtype` | For example, `float32` | How is each value represented? |
| `values.device` | For example, `cpu` | Where is it stored and computed? |
| `type(values)` | `torch.Tensor` | What is the Python object type? |

`[4]` has one axis: it is neither a `[1,4]` row matrix nor a `[4,1]` column matrix. A scalar has shape `[]` and one element; an empty vector with shape `[0]` has zero elements.

```python
import torch

values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
assert values.shape == (2, 3, 4)
assert values.ndim == 3 and values.numel() == 24
assert torch.tensor(5.0).shape == ()
assert torch.empty(0).numel() == 0
assert values[0].shape == (3, 4)
assert values[0:1].shape == (1, 3, 4)
```

An integer index removes an axis; a slice preserves it. The last two expressions select the same sequence, but only one preserves a batch axis. Choose deliberately.

## Be explicit when creating tensors

`torch.tensor([1,2])` normally infers `int64`; floating-point lists use the current default floating dtype. Specify the training contract: class IDs commonly use `long`, continuous features use floating point, and masks use `bool`. Floating-point model parameters do not imply floating-point class indices.

| Need | Common entry points | Important distinction |
| --- | --- | --- |
| Fixed values | `tensor`, `zeros`, `ones`, `full` | The fill value also affects `full` dtype inference |
| Regular sequence | `arange`, `linspace` | Usually exclude the endpoint versus include both endpoints with a specified count |
| Random values | `rand`, `randn`, `randint` | Uniform [0,1), standard normal, and half-open integer range |
| Specified normal distribution | `normal` | Supply standard deviation, not variance |
| Match an existing tensor | `zeros_like`, `randn_like` | Inherit dtype/device by default; default `randn_like` on an integer tensor fails |
| Allocate storage only | `empty` | Uninitialized values, not zeros or reliably tiny numbers |

Use `empty` for a buffer whose entire contents will be overwritten. Specific determinism settings can fill special values, but that does not make it a useful numerical initialization. The [official documentation](https://docs.pytorch.org/docs/2.8/generated/torch.empty.html) describes the exception.

`.to(dtype=..., device=...)` returns the conversion result and may return the original object when no conversion is needed. Neither it nor `.float()` guarantees a copy. Type promotion is operator-specific: addition can promote types, whereas common matrix products require matching types. PyTorch does not scan every element to discover its type before every operation.

## A storage picture

This picture describes storage, not gradient connections:

```text
storage:       [0, 1, 2, 3, 4, 5]
                └───────┬──────┘
base [2,3]:    [[0,1,2], [3,4,5]]   stride=(3,1)
transpose:     [[0,3], [1,4], [2,5]] stride=(1,3)

clone:         [0, 1, 2, 3, 4, 5]   independent storage
```

For an ordinary strided tensor, the storage offset for an element is `storage_offset + Σ index[axis] × stride[axis]`. Transposition changes how values are addressed without necessarily moving them. Different objects and shapes can share storage.

```python
base = torch.arange(6).reshape(2, 3)
window = base[:, 1:]
snapshot = window.clone()
window[0, 0] = 50
assert base[0, 1].item() == 50
assert snapshot[0, 0].item() == 1
assert window is not base
```

`alias = base` names the same Python object. `window` is another object sharing storage. `clone` allocates independent storage. A new object is not necessarily new data. See [Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html).

## Reshape is not transpose

`reshape` regroups values in their current logical order; `transpose` and `permute` change axis correspondence. To exchange time and features in `[B,T,D] → [B,D,T]`, use `transpose(1,2)` rather than merely requesting that shape.

`view` requires compatible sizes and strides. Contiguous tensors readily support many views, but **non-contiguous does not always mean impossible**. `reshape` shares when possible and copies otherwise; correct code should not depend on which it chooses.

```python
grid = torch.arange(30).reshape(5, 6)
spaced = grid[:, ::2]
flat = spaced.view(-1)
assert not spaced.is_contiguous()
assert flat.stride() == (2,)
assert flat.tolist() == list(range(0, 30, 2))

swapped = grid.transpose(0, 1)
repacked = swapped.reshape(-1)
assert repacked[:5].tolist() == [0, 6, 12, 18, 24]
assert repacked.data_ptr() != swapped.data_ptr()
```

The first example can flatten while retaining a stride of two. The second needs to repack the transposed order; this particular reshape makes a copy. `contiguous()` can also return its input unchanged when the requested layout already holds. See the [stride condition for view](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.view.html).

`flatten(start_dim=1)` commonly preserves the batch axis while combining later axes. `squeeze(dim)` removes a specified singleton axis; unrestricted `squeeze()` can accidentally delete a batch axis when batch size is one. `unsqueeze(dim)` inserts a singleton axis.

## Selecting samples and assembling batches

Basic slicing usually returns a view. Integer-array/list and Boolean advanced indexing generally return copies. However, an assignment such as `values[mask] = ...` modifies the source. Copy behavior when reading does not make indexed assignment harmless.

`values[-1]` selects the last element; negative slice steps such as `values[::-1]` are unsupported, so use `flip`. `matrix[::2,::2]` strides along two axes; `matrix[::2][::2]` selects along the first axis twice.

| Operation needed | Choice | Axis behavior |
| --- | --- | --- |
| Same selected columns for every row | `index_select(dim, indices)` | One-dimensional indices; repeats allowed |
| Different selected columns per row | `gather(dim, indices)` | Output shape follows the index; next chapter has an example |
| Keep elements meeting a condition | Boolean indexing / `masked_select` | A full-shape mask commonly yields a flat selection |
| Choose between values | `where(mask, left, right)` | Elementwise choice with broadcasting |
| Join along an existing axis | `cat` | Other axes must match |
| Add an axis | `stack` | Input shapes must match |
| Split by chunk size | `split` | Integer argument is size, not count |
| Request a chunk count | `chunk` | Can return fewer chunks; `tensor_split` uses the requested count |

```python
first = torch.tensor([2, 4, 6])
second = torch.tensor([8, 10, 12])
assert torch.cat([first, second]).shape == (6,)
assert torch.stack([first, second]).shape == (2, 3)
assert [part.numel() for part in torch.arange(7).split(3)] == [3, 3, 1]
assert len(torch.arange(4).chunk(3)) == 2
assert len(torch.tensor_split(torch.arange(4), 3)) == 3
```

`stack` is not analogous to appending `[3,4]` to `[1,2]`, which creates a ragged Python list. Split outputs form an immutable tuple, but their tensors can still modify shared storage. Check the [chunk contract](https://docs.pytorch.org/docs/2.8/generated/torch.chunk.html) before unpacking a fixed number of outputs.

## Crossing into NumPy or Python

`torch.tensor(array)` copies data; `from_numpy` and compatible `as_tensor` calls can share storage. Sharing avoids a copy but lets changes on either side affect training inputs.

Default `.numpy()` requires a CPU tensor not requiring gradients, with supported dtype/layout and other conditions. It normally shares storage. `.numpy(force=True)` handles detach, CPU transfer, and conjugate-related conversions, but does not guarantee an independent copy. Explicitly copy when you need a snapshot. See the [numpy interface conditions](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.numpy.html).

`.tolist()` produces nested Python numbers; `list(tensor)` iterates over sub-tensors along the first axis. `.item()` needs exactly one element, not necessarily a zero-dimensional tensor. Its Python result carries no gradient connection. Frequent CUDA `item()` calls can also make the CPU wait, so use them for logging rather than constructing a differentiable loss.

When changing one tensor changes another, check shared storage. When shapes look right but meanings differ, check for a reshape used in place of an axis permutation. Next: [valid code that computes the wrong quantity](operations-and-shapes.en.md).
