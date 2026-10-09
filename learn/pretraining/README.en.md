# Pretraining: from data to a reliable update

[中文](README.md) · **English**

A Transformer forward pass is only part of training. How does text become targets? Which tokens count toward the loss? How are gradients aggregated, and how do we know whether an update helped? This chapter follows that sequence.

## First, understand one small step on one device

If you have not written a training loop, run the [two-example PyTorch update](../../00-foundations/pytorch/README.en.md) first. Separate forward, backward, and parameter updates before returning to token targets. Read [initialization](../../00-foundations/deep-dives/activation-and-initialization.en.md), [optimizers](../../00-foundations/deep-dives/optimizers.en.md), and [training diagnosis](../../00-foundations/deep-dives/generalization.en.md) when questions about gradient scale or generalization arise.

Consider two sequences. One has 2 valid targets and a summed loss of 4; the other has 6 valid targets and a summed loss of 6.

The token-weighted mean is $(4+6)/(2+6)=1.25$. Averaging the two sequence means instead gives $(2+1)/2=1.5$. The data did not change, but the optimization weights did. Gradient accumulation, distributed training, and padding all bring this small issue back.

Establish the target, mask, and denominator before asking how to compute it faster.

| Reading order | Question to follow | Check afterward |
| --- | --- | --- |
| 1. [Language-model objectives](../../00-foundations/deep-dives/language-model-objective.en.md) | Which position predicts which token? | Why can training run in parallel while generation usually cannot? |
| 2. [The pretraining pipeline](../../00-foundations/deep-dives/pretraining-pipeline.en.md) | How are texts cleaned, deduplicated, mixed, and packed? | Are EOS, attention masks, and loss masks the same? |
| 3. [One training step](../../00-foundations/deep-dives/training-step.en.md) | How does a batch change parameters? | Does changing batch partitioning preserve the objective? |
| 4. [Multi-token prediction](../../00-foundations/deep-dives/multi-token-prediction.en.md) | What supervision do extra heads receive? | Can targets cross document boundaries or leak answers? |

## Then consider precision, memory, and multiple devices

These interventions act at different levels. BF16 changes number representation; recomputation changes stored intermediates; FSDP changes where state lives; TP partitions a matrix computation. All can affect memory, but they are not interchangeable switches.

| Symptom | Check first | Continue with |
| --- | --- | --- |
| Inf / NaN loss | Numeric range, scaling, and clipping order | [Precision and memory](../../00-foundations/deep-dives/precision-and-memory.en.md) |
| Weights fit but backward runs out of memory | Activations, optimizer state, temporary tensors, sequence length | [Precision and memory](../../00-foundations/deep-dives/precision-and-memory.en.md) |
| Results change with device count | Data partitioning, valid-token counts, synchronization, randomness | [Distributed training](../../06-systems/distributed-training.en.md) |
| High utilization but poor throughput | Useful-token throughput, recomputation, communication, waiting | [Distributed training](../../06-systems/distributed-training.en.md) |

## Chapter contents

<!-- widget:study-atlas -->

## Put it to use

Run forward, backward, and one update on a fixed tiny batch and save the parameter changes. Change only one condition, such as accumulation steps or precision. Compare outputs, gradients, and the update within appropriate tolerances before scaling up.

Worked numbers test mechanisms; they are not hardware benchmarks. Report model version, sequence length, effective batch, dtype, hardware, and timing boundaries for actual memory or speed measurements.

Continue with [SFT and post-training](../../05-post-training/README.en.md), or turn to [generation and inference](../inference/README.en.md).
