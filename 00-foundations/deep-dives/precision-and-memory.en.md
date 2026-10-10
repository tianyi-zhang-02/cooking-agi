# Precision and memory: why can BF16 training still run out of memory?

[中文](precision-and-memory.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [one training step](training-step.en.md), [LoRA / QLoRA](../../05-post-training/lora-and-qlora.en.md)

A few gigabytes of weights do not imply a few gigabytes of training memory. Backpropagation needs intermediate results, optimizers keep state, and kernels need workspace.

Separate **reliable arithmetic** from **fitting the live tensors**. BF16, gradient accumulation, and recomputation address different parts of those problems.

## FP16 and BF16 trade range against resolution

These are storage formats, not a claim that every operation accumulates in the same precision.

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

## Autocast does not convert the whole model to FP16

`model.half()` changes floating-point parameter and buffer storage. `autocast` chooses execution precision by operation: matrix products and some reductions can use different types within one forward pass. Backward does not universally switch to FP32; its operations follow the types selected for their corresponding forward operations.

Inspect parameter, activation, gradient, and accumulation types separately. Saving a `state_dict()` does not automatically convert AMP-trained weights to FP32. [PyTorch's AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) separate autocast from gradient scaling: one selects precision, the other addresses gradient range.

## What does loss scaling protect?

Very small FP16 gradients can round to zero. Multiplying loss by $s$ produces $s\nabla\mathcal L$; dividing gradients by $s$ before the update preserves the ideal-arithmetic gradient while potentially avoiding intermediate underflow.

This is not a universal fix. Scaling the loss does not repair an Inf already produced in the forward pass. BF16 usually does not need scaling in the same way, but still has rounding and stability limits.

With a dynamic scaler, the order is:

```text
forward → scale loss → backward through the full accumulation window
        → unscale → gradient clipping → optimizer step → update scale
```

Clipping scaled gradients changes the meaning of the threshold. Changing scale partway through accumulation also mixes incompatible gradient scales. [PyTorch's AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) explain the required ordering.

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

**Activation checkpointing** is different from saving training progress to disk.

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

```text
Ordinary: x → block 1 → block 2 → block 3 → loss
              retain internal activations for backward
Recompute: retain boundaries → rerun the needed segment → compute gradients
```

Parameters and optimizer states remain. The saving is in selected activations, paid for with extra computation; non-reentrant implementations may stop recomputation once the required intermediates are available. Randomness, mutable state, and side effects must preserve forward semantics.

[PyTorch's checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint.html) recommends explicitly selecting the non-reentrant implementation; verify the API for your version. Compare loss and parameter gradients, not only successful execution. Restoring dropout RNG state matters.

## When is a memory optimization worth using?

Fix model, sequence distribution, valid tokens, and optimizer. Record peak memory, valid tokens per second, numerical differences, and resume behavior.

| Method | Main saving | Main cost |
| --- | --- | --- |
| Lower precision | Some tensor storage and compute bandwidth | Rounding, range, kernel support |
| Gradient accumulation | Concurrent microbatch activations | More serial steps; weights remain |
| Activation checkpointing | Retained intermediates | Recomputation |
| LoRA | Trainable gradients and optimizer states | Restricted updates; base forward remains |
| FSDP / ZeRO | Per-device training state | Communication and temporary gathers |

“Less memory” is not the whole result. A slower step may enable a useful larger batch, so measure the total effect. A run that stops crashing by silently changing the objective is no longer the same comparison.
