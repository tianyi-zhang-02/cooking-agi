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
M_{\mathrm{peak}}\approx M_{\mathrm{weights}}+M_{\mathrm{grads}}
+M_{\mathrm{optimizer}}+M_{\mathrm{activations}}+M_{\mathrm{temporary}}.
$$

This is an accounting guide, not an exact claim that every component peaks simultaneously.

Consider a **hypothetical mixed-precision Adam configuration**: 2 bytes per weight, 2 per gradient, a 4-byte FP32 master copy, and 8 bytes for two FP32 moments. That is 16 bytes per parameter, or about 16 GB for a billion parameters **before activations**. Other implementations keep parameters or gradients in FP32 or omit a separate master copy. Inspect actual tensors rather than applying 16 universally.

| Symptom | Inspect first |
| --- | --- |
| OOM while loading | Weights, duplicate models, placement |
| Forward OOM as sequences grow | Activations, attention intermediates, logits |
| OOM at the first optimizer step | Lazily initialized optimizer states |
| Memory grows over many steps | Losses or outputs retaining computation graphs |
| Reserved memory far exceeds allocated memory | Allocator behavior and fragmentation |

Memory reported by `nvidia-smi` is not identical to live PyTorch tensor storage. The reserved/allocated gap alone does not diagnose a leak.

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
