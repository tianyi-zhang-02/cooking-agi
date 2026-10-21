# Precision and memory: why can BF16 training still run out of memory?

[中文](precision-and-memory.md) · **English**

> Last reviewed: 2026-10 · Prerequisites: [one training step](training-step.en.md), [LoRA / QLoRA](../../05-post-training/lora-and-qlora.en.md)

A few gigabytes of weights do not imply a few gigabytes of training memory. Backpropagation needs intermediate results, optimizers keep state, and kernels need workspace.

Separate **reliable arithmetic** from **fitting the live tensors**. BF16, gradient accumulation, and recomputation address different parts of those problems; no single switch fixes all of them.

## FP16 and BF16 trade range against resolution

These are storage formats, not a claim that every operation accumulates in the same precision. Only FP32, FP16, and BF16 are covered here; for FP8, FP4, and block-scaled formats such as NVFP4, see [the data-types section of GPU architecture](../../12-infra/gpu-architecture.en.md).

| Format | Sign / exponent / fraction bits | Spacing near 1 | Main property |
| --- | --- | --- | --- |
| FP32 | 1 / 8 / 23 | $2^{-23}$ | Broad range, fine resolution |
| FP16 | 1 / 5 / 10 | $2^{-10}$ | Finer than BF16, narrower range |
| BF16 | 1 / 8 / 7 | $2^{-7}$ | FP32-like exponent range, coarser resolution |

For example, rounding 1.001 to the nearest BF16 value gives 1; FP16 gives approximately 1.0009765625. Conversely, $10^5$ exceeds FP16's largest finite value, 65504, but fits BF16's range. **Range and precision are distinct axes.**

Python's built-in half-precision packing illustrates the FP16 rounding below. The BF16 calculation uses spacing near 1 and is only a local illustration.

```python
import struct

half_value = struct.unpack('e', struct.pack('e', 1.001))[0]
bfloat_near_one = 1 + round((1.001 - 1) * 128) / 128
assert half_value == 1.0009765625
assert bfloat_near_one == 1.0
```

Matrix inputs, products, accumulation, and stored outputs may use different precisions. A BF16 tensor does not establish that the entire computation runs in BF16.

### A nonzero gradient can still leave a weight unchanged {#rounded-away-update}

Start with a weight of 1 and subtract 0.001 per step. That update is nonzero in BF16, but storing `1 - 0.001` in BF16 rounds back to 1. The nearest BF16 value below 1 is 0.99609375; 0.999 is closer to 1.

Repeating the update does not automatically accumulate the lost changes. Subtract 0.001 one hundred times, with no schedule, momentum, or training data involved—just storage rounding:

<figure class="worked-update worked-update--pairs" lang="en" id="precision-update-paths">
  <figcaption>The same 100 subtractions, with different precision for retained updates.</figcaption>
  <ol>
    <li><small>Write back to BF16 each step</small><strong>Still 1</strong><span>Each small change rounds away. The next step starts from 1 again.</span></li>
    <li><small>Keep updating in FP32</small><strong>About 0.9</strong><span>The changes accumulate. Casting the final value to BF16 gives 0.8984375.</span></li>
  </ol>
</figure>

<details class="code-drop" markdown="1">
<summary>Check it with PyTorch on CPU</summary>

```python
import torch

def rounded_update_example():
    low_precision = torch.tensor(1.0, dtype=torch.bfloat16)
    full_precision = torch.tensor(1.0, dtype=torch.float32)
    for step in range(100):
        low_precision.sub_(0.001)
        full_precision.sub_(0.001)
    return low_precision.item(), full_precision.item(), full_precision.to(torch.bfloat16).item()

low_result, full_result, rounded_result = rounded_update_example()
assert torch.tensor(0.001, dtype=torch.bfloat16).item() != 0.0
assert low_result == 1.0
assert abs(full_result - 0.9) < 2e-6
assert rounded_result == 0.8984375
```

</details>

This differs from a gradient becoming zero when stored. Loss scaling must be undone before the update; it is not a way to permanently enlarge steps to bypass weight rounding. FP32 parameters or master weights give small updates somewhere to accumulate, but not every low-precision training method needs the same copies. [Mixed Precision Training §3.1–3.2](https://arxiv.org/html/1710.03740v3#S3.SS1) separates update precision from gradient scaling. The BF16 example here is our own illustration, not a reproduction of that paper's FP16 experiments.

## Autocast does not convert the whole model to FP16

`model.half()` changes floating-point parameter and buffer storage. `autocast` chooses execution precision by operation: matrix products and some reductions can use different types within one forward pass. Backward does not universally switch to FP32; its operations follow the types selected for their corresponding forward operations.

Inspect parameter, activation, gradient, and accumulation types separately. Saving a `state_dict()` does not automatically convert AMP-trained weights to FP32. [PyTorch 2.8 AMP examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html) separate autocast from gradient scaling: one selects precision, the other addresses gradient range.

Keep a linear layer's parameters in FP32 and compute `[3, 4] @ [1, 2] = 11` under CPU BF16 autocast, then take squared error against 10. In this run the parameters stay FP32, while the layer output is BF16. Explicitly convert the output to FP32 for the loss; after backward, the parameter `.grad` is FP32 too. **FP32 gradient storage does not undo rounding in earlier low-precision operations.**

<details class="code-drop" markdown="1">
<summary>Inspect the dtypes along the computation</summary>

```python
import torch

def autocast_dtype_example():
    layer = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = layer(torch.tensor([[3.0, 4.0]], dtype=torch.float32))
        loss = (prediction.float() - 10.0).square().mean()
    loss.backward()
    dtypes = (layer.weight.dtype, prediction.dtype, loss.dtype, layer.weight.grad.dtype)
    return dtypes, layer.weight.grad

dtypes, gradient = autocast_dtype_example()
assert dtypes == (torch.float32, torch.bfloat16, torch.float32, torch.float32)
torch.testing.assert_close(gradient, torch.tensor([[6.0, 8.0]], dtype=torch.float32))
print(dtypes)
```

</details>

This checks the PyTorch 2.8.0 CPU interface, not GPU speed. All code on this page ran on 2.8.0; checked in October 2026 against the stable [2.14 documentation](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html), the AMP usage, checkpoint constraints, and memory counters cited below have not changed, but the code has not been rerun on 2.14. The FP32 parameters already retain updates here; do not count a second FP32 master copy serving the same purpose. A scheme with low-precision parameters and separate master weights needs different memory accounting; the 16-byte ledger below assumes exactly that scheme. Consult the [AMP documentation](https://docs.pytorch.org/docs/2.8/amp.html) for operator-specific policies rather than generalizing this layer's dtypes to every operation.

## What does loss scaling protect?

Very small FP16 gradients can round to zero. Multiplying loss by $s$ produces $s\nabla\mathcal L$; dividing gradients by $s$ before the update preserves the ideal-arithmetic gradient while potentially avoiding intermediate underflow.

This is not a universal fix. Scaling the loss does not repair an Inf already produced in the forward pass. BF16 usually does not need scaling in the same way, but it can still overflow, lose updates to rounding, or train unstably.

With a dynamic scaler, the order is:

```text
forward → scale loss → backward through the full accumulation window
        → unscale → gradient clipping → optimizer step → update scale
```

Clipping scaled gradients changes the meaning of the threshold. Changing scale partway through accumulation also mixes incompatible gradient scales. [PyTorch 2.8 AMP examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html) explain the required ordering.

A scalar makes the order visible. A true gradient of 0.25 scaled by 8 becomes 2. With clipping threshold 0.5, the correct order returns 0.25 without clipping. Clipping 2 to 0.5 first, then dividing by 8, returns 0.0625: an unintended fourfold reduction.

```python
def clip_scalar(gradient, limit):
    return max(-limit, min(gradient, limit))

scaled_gradient = 0.25 * 8
correct_gradient = clip_scalar(scaled_gradient / 8, 0.5)
wrong_gradient = clip_scalar(scaled_gradient, 0.5) / 8
assert correct_gradient == 0.25
assert wrong_gradient == 0.0625
```

This scalar illustration is not elementwise clipping advice: `clip_grad_norm_` rescales using a combined gradient norm. The standard dynamic `GradScaler` skips an update with nonfinite gradients and adjusts its scale; it does not automatically guarantee a rerun of the same batch.

## Five parts of the training memory bill

$$
\begin{aligned}
M_{\mathrm{peak}}\approx{}& M_{\mathrm{weights}}+M_{\mathrm{grads}}\\
&+M_{\mathrm{optimizer}}+M_{\mathrm{activations}}\\
&+M_{\mathrm{temporary}}.
\end{aligned}
$$

This is an accounting guide, not an exact claim that every component peaks simultaneously.

Consider a **hypothetical mixed-precision Adam configuration**: 2 bytes per weight, 2 per gradient, a 4-byte FP32 master copy, and 8 bytes for two FP32 moments. That is 16 bytes per parameter, or about 16 GB for a billion parameters **before activations**. Other implementations keep parameters or gradients in FP32 or omit a separate master copy. Inspect actual tensors rather than applying 16 universally.

### Work through one concrete estimate {#state-ledger}

Keep those assumptions and count the resident training state for one billion parameters. List the master copy separately from the low-precision weights:

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>WEIGHTS</small><strong>2 GB</strong><span>One billion BF16 parameters at 2 bytes each.</span></li>
<li><small>GRADIENTS</small><strong>2 GB</strong><span>One BF16 gradient per parameter, also 2 bytes each.</span></li>
<li><small>FP32 MASTER COPY</small><strong>4 GB</strong><span>One extra FP32 copy per parameter at 4 bytes each.</span></li>
<li><small>ADAM MOMENTS</small><strong>8 GB</strong><span>Two FP32 moment estimates per parameter, totaling 8 bytes.</span></li>
</ol>
<figcaption>Total: 16 GB of parameter-related state under these assumptions, before activations and workspace.</figcaption>
</figure>

The moments are Adam’s estimates of the gradient’s first and second raw moments: moving averages of gradients and squared gradients. They remain after the current batch finishes because the next update uses them.

A GB is $10^9$ bytes; a GiB is $2^{30}$ bytes. The same 16 GB is about **14.90 GiB**, not a memory saving.

<details markdown="1">
<summary>Try another parameter count (Python)</summary>

```python
def state_bytes(total_parameters, trainable_parameters,
                weight_bytes=2, gradient_bytes=2, master_bytes=4, moment_bytes=8):
    counts = (total_parameters, trainable_parameters)
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
        raise ValueError("parameter counts must be nonnegative integers")
    if trainable_parameters > total_parameters:
        raise ValueError("trainable parameters exceed total parameters")
    widths = (weight_bytes, gradient_bytes, master_bytes, moment_bytes)
    if any(not isinstance(width, int) or isinstance(width, bool) or width < 0 for width in widths):
        raise ValueError("storage widths must be nonnegative integers")
    return {
        "weights": total_parameters * weight_bytes,
        "gradients": trainable_parameters * gradient_bytes,
        "master": trainable_parameters * master_bytes,
        "moments": trainable_parameters * moment_bytes,
    }

full_state = state_bytes(1_000_000_000, 1_000_000_000)
assert sum(full_state.values()) == 16_000_000_000
assert round(sum(full_state.values()) / 2**30, 2) == 14.90
adapter_state = state_bytes(1_010_000_000, 10_000_000)
assert sum(adapter_state.values()) == 2_160_000_000
```

</details>

The final calculation assumes a billion-parameter base plus 10 million adapter parameters. All weights use 2 bytes, but only the adapter has gradients, master copies, and moments: 2.16 GB. **It is not 1% of 16 GB.** The base remains, and backward activations are still absent from the estimate. Real adapters may use another dtype; quantization also needs metadata such as scales.

### Why does the first optimizer step allocate more? {#lazy-optimizer-state}

Consider a CPU check with PyTorch 2.8.0: 100 FP32 parameters, ordinary AdamW with foreach disabled, no AMP, extra master copy, or shared storage. Counting `numel() * element_size()` gives:

| Point in the run | Weights | Gradients | Optimizer tensors | Total |
| --- | ---: | ---: | ---: | ---: |
| Optimizer constructed | 400 B | 0 B | 0 B | 400 B |
| First backward completed | 400 B | 400 B | 0 B | 800 B |
| First step completed | 400 B | 400 B | 804 B | 1604 B |
| After `zero_grad(set_to_none=True)` | 400 B | 0 B | 804 B | 1204 B |

The 804 B consists of two 400 B moments and a 4 B step tensor in this implementation. These are logical tensor bytes—not Python-object overhead, allocator usage, temporary computation storage, or a GPU peak inferred from a CPU experiment. The [test file](../../site/tests/test_training_engineering.py) contains the executable check.

The useful lesson is that **measurement immediately after loading misses state that does not yet exist**. Inspect at least one complete GPU update and representative long sequences. `element_size()` helps check dtypes, but naively summing views, tied weights, or flat-buffer slices can double-count storage.

Changing the optimizer changes the ledger too. For the same 100 FP32 parameters after the first update:

| Optimizer configuration | Additional optimizer tensor storage |
| --- | ---: |
| SGD, `momentum=0` | 0 B |
| SGD, `momentum=0.9` | 400 B for one momentum buffer |
| AdamW, default non-AMSGrad | 804 B for two moments and the step |

These are CPU state counts, excluding weights and gradients, not GPU peaks. **SGD need not use momentum, and Adam's moments do not produce the gradient.** Backward computes the gradient; the optimizer uses it to update the moments and then the parameters. Compare the update order in [SGD](https://docs.pytorch.org/docs/2.8/generated/torch.optim.SGD.html) and [Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html). Different dtypes, AMSGrad, or a sharded optimizer require a new ledger, not a fixed training-to-inference memory multiplier.

### Do eight GPUs divide the estimate by eight? {#sharded-state}

Ordinary data parallelism keeps model replicas, so no. Under the same 16 GB assumptions, an ideal evenly sharded state estimate is:

| Method | Ideal state per rank | GB per rank |
| --- | --- | ---: |
| Ordinary data parallelism | Weights 2 + gradients 2 + master / moments 12 | 16 |
| ZeRO-1 | $2+2+12/8$ | 5.5 |
| ZeRO-2 | $2+(2+12)/8$ | 3.75 |
| ZeRO-3 | $(2+2+12)/8$ | 2 |

Successively sharding those categories is the idea described in the [ZeRO paper](https://arxiv.org/html/1910.02054v3#S5). This table omits execution-time parameter gathering, communication buffers, activations, and fragmentation. The final 2 GB does not mean a 2 GB card can train this model; gathering a layer creates additional peak usage. Continue with [distributed training](../../06-systems/distributed-training.en.md) for partitioning, prefetch, and release choices.

| Symptom | Inspect first |
| --- | --- |
| OOM while loading | Weights, duplicate models, placement |
| Forward OOM as sequences grow | Activations, attention intermediates, logits |
| OOM at the first optimizer step | Lazily initialized optimizer states |
| Memory grows over many steps | Losses or outputs retaining computation graphs |
| Reserved memory far exceeds allocated memory | Allocator behavior and fragmentation |

Memory reported by `nvidia-smi` is not identical to live PyTorch tensor storage. The reserved/allocated gap alone does not diagnose a leak.

Record allocated, reserved, and peak memory separately. After any needed warmup, reset peak counters, run the measured complete steps, and wait for GPU work to finish. Also retain a separate first-update peak: warmup can otherwise hide one-time allocations. [PyTorch's memory documentation](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management) explains these counters. `empty_cache()` releases unused cached memory, not tensors your program still references.

## Accumulation reduces concurrent microbatch activations

Four microbatches of two sequences can contribute to one optimizer update over eight sequences. Releasing each microbatch's graph after backward avoids holding activations for all eight together.

Weights and Adam states do not shrink. Reducing batch size may also be insufficient when sequence length rises from 2K to 32K. BatchNorm, randomness, sequence lengths, and the loss denominator affect equivalence to a larger batch.

Unequal valid-token counts require token-weighted normalization, not blindly dividing every loss by the accumulation count. Distributed gradient averaging adds another factor; see [distributed training](../../06-systems/distributed-training.en.md).

## Checkpointing recomputes what it did not retain

**Activation checkpointing**, also called gradient checkpointing, is different from [saving training progress to disk](../../practice/post-training/checkpoint-and-resume.en.md). Q, K, and V are intermediate tensors. They do not automatically become checkpoint boundaries: that depends on which results a recomputation strategy retains.

### Where does the gradient come from without saved intermediates? {#recompute-example}

Take two operations: $h=wx$ and $\ell=h^2$. With $x=2$ and $w=3$, forward produces $h=6$ and $\ell=36$. The weight gradient needs that intermediate value:

$$
\frac{\partial\ell}{\partial w}=2hx=24.
$$

<figure class="worked-update worked-update--pairs" lang="en">
<ol>
<li><small>RETAIN</small><strong>Keep h = 6 for backward</strong><span>Read it when needed without repeating the multiplication.</span></li>
<li><small>RECOMPUTE</small><strong>Recover h from the original x and w</strong><span>2 × 3 still gives 6, so the gradient remains 24.</span></li>
</ol>
<figcaption>The gradient path stays intact; how we obtain the intermediate changes. A single scalar illustrates the mechanism, not a useful memory saving.</figcaption>
</figure>

The word “original” matters. Changing the weight to 4 before backward would recompute $h=8$, no longer the original forward. Mutable globals, updated statistics, and external requests can also make rerunning a region unsafe. The default determinism check does not compare every tensor value; compare outputs and gradients yourself. [PyTorch 2.8 documents these constraints](https://docs.pytorch.org/docs/2.8/checkpoint.html).

<details markdown="1">
<summary>Compare ordinary and checkpointed execution on CPU</summary>

```python
import torch
from torch.utils.checkpoint import checkpoint

def squared_product(features, weight):
    hidden = features * weight
    return hidden.square().sum()

def checkpoint_example(use_checkpoint):
    features = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    if use_checkpoint:
        loss = checkpoint(squared_product, features, weight, use_reentrant=False)
    else:
        loss = squared_product(features, weight)
    loss.backward()
    return loss.item(), features.grad.item(), weight.grad.item()

assert checkpoint_example(False) == (36.0, 36.0, 24.0)
assert checkpoint_example(True) == checkpoint_example(False)
```

This uses PyTorch 2.8 / CPU to check values and gradients, not GPU memory or speed. A Dropout comparison also needs matching RNG state; a separate project test covers it. Exact equality here is not a promise for every operator and device.

</details>

### Which intermediates are worth recomputing? {#recompute-budget}

Why does sequence length make it useful? Count two tensors before estimating the whole model. Use batch size 2, hidden size 1024, 16 attention heads, and 2 bytes per element:

| Sequence length | One $B\times T\times d$ hidden state | One explicit $B\times H\times T\times T$ score tensor |
| --- | ---: | ---: |
| 2048 | 8 MiB | 256 MiB |
| 4096 | 16 MiB | 1024 MiB |

```python
def tensor_mib(shape, bytes_per_value=2):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * bytes_per_value / 2**20

assert tensor_mib((2, 2048, 1024)) == 8
assert tensor_mib((2, 16, 2048, 2048)) == 256
```

Doubling length doubles the first tensor and quadruples the second. Neither is total per-layer memory: retention, lifetimes, and fused kernels matter. [FlashAttention](attention-kernels.en.md) avoids materializing the full score matrix in device memory; checkpointing reduces activations retained for backward. They can work together, but their savings cannot simply be added.

FlashAttention does not eliminate all saved state. [The paper, §3.1 and Appendix B](https://arxiv.org/html/2205.14135v2#S3.SS1), uses Q/K/V, outputs, and row-normalization statistics to reconstruct local attention, with RNG handling for dropout. It avoids full quadratic intermediates, not all training state. A saved activation is also distinct from its `.grad`: keeping a forward value for backward does not mean retaining gradients on every non-leaf tensor. See [autograd](../pytorch/autograd.en.md).

```text
Ordinary: x → block 1 → block 2 → block 3 → loss
              retain internal activations for backward
Recompute: retain boundaries → rerun the needed segment → compute gradients
```

Parameters and optimizer states remain. The saving is in selected activations, paid for with extra computation; non-reentrant implementations may stop recomputation once the required intermediates are available. Randomness, mutable state, and side effects must preserve forward semantics.

[PyTorch 2.8 recommends](https://docs.pytorch.org/docs/2.8/checkpoint.html) setting `use_reentrant=False` explicitly. This path can stop recomputation once the needed tensors are available, so logs or counters need not execute exactly twice. Default RNG handling covers supported devices, not arbitrary mutable program state.

You need not recompute every operation. **Selective activation checkpointing** can retain expensive results while rerunning cheaper work. Suppose A produces 4 MiB and takes 6 ms to recompute; B produces 32 MiB and takes 1 ms. Assume both results would otherwise stay alive until backward and can be recomputed independently:

| Choice in this example | Retained internal results | Extra recomputation |
| --- | ---: | ---: |
| Save both | 36 MiB | 0 ms |
| Save A, recompute B | 4 MiB | 1 ms |
| Recompute both | 0 MiB | 7 ms |

Boundary inputs, outputs, and temporary recomputation storage are excluded: **0 MiB is not zero peak training memory**. These are hypothetical numbers, not measurements. Dependencies, fused kernels, and shared storage affect the real choice. PyTorch's [selective checkpointing article](https://pytorch.org/blog/activation-checkpointing-techniques/#selective-activation-checkpoint-new) explains per-operator policies. Find what is still live at the peak, then measure whether the saving is worth the additional work.

## When is a memory optimization worth using?

Fix model, sequence distribution, valid tokens, and optimizer. Record peak memory, valid tokens per second, numerical differences, and resume behavior. These methods are not alternatives to one another:

| Method | Main saving | Main cost |
| --- | --- | --- |
| Lower precision | Some tensor storage and compute bandwidth | Rounding, range, kernel support |
| Gradient accumulation | Concurrent microbatch activations | More serial steps; weights remain |
| Activation checkpointing | Retained intermediates | Recomputation |
| LoRA | Trainable gradients and optimizer states | Restricted updates; base forward and backward passes remain (without base weight gradients) |
| FSDP / ZeRO | Per-device training state | Communication and temporary gathers |

“Less memory” is not the whole result. A slower step may enable a useful larger batch, so measure the total effect. A run that stops crashing by silently changing the objective is no longer the same comparison.
