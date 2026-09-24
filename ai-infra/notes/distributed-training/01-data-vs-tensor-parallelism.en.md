# Data Parallelism vs. Tensor Parallelism: What Actually Gets Split?

[中文](01-data-vs-tensor-parallelism.md) · **English** · [Back to AI Infra](../../README.en.md)

> Reading time: ~5 minutes · Level: Beginner · Freshness: Stable · Last reviewed: 2026-09

## Why this matters

DP (Data Parallelism) and TP (Tensor Parallelism) get mentioned in the same breath, but they split completely different things in training: DP copies the model and splits the data; TP copies the data and splits the matrix math inside the model. Missing that distinction makes it hard to understand why the two get combined, and hard to reason about which one a given model actually needs first.

## DP: every GPU holds the full model

```text
GPU0: full model + one slice of the batch
GPU1: full model + another slice of the batch
```

Each rank gets a different slice of data, runs forward/backward independently, and produces its own gradients. An AllReduce then averages the gradients across ranks, and every rank updates its parameters with that same averaged gradient. Because every rank started identical and applies an identical update, the parameters stay in sync afterward.

DP needs almost no model code changes — just a way to split the batch and synchronize gradients. The cost is that **every GPU must hold the full parameters, gradients, and optimizer state**. Once a model is too large for a single GPU, plain DP simply fails, which is exactly why FSDP2/ZeRO also shard those three components.

## TP: one matrix multiply is split across GPUs

Take a linear layer `Y = XW`. Split the weight `W` column-wise into `[W1 | W2]` and place each half on a different GPU:

```text
GPU0: Y1 = X · W1
GPU1: Y2 = X · W2
```

Each GPU computes its slice of the output, and the results are stitched back together with an AllGather (for a column split) or AllReduce (for a row split) before the next layer can run. In other words: **the same sample, in the same matrix multiply, is broken into two smaller matrix multiplies**, and the GPUs must communicate frequently to keep the intermediate results aligned.

This is the biggest behavioral difference from DP: DP only communicates once, at the end of a step, to sync gradients. TP communicates **inside every layer**, because the next layer needs the previous layer's complete output. That's why TP is usually restricted to GPUs with high-bandwidth interconnects (like NVLink within a node) — doing TP across nodes is expensive.

## Side by side

| | DP | TP |
| --- | --- | --- |
| What gets split | Data (the batch) | Matrices/tensors inside the model |
| Full parameters per GPU? | Yes | No, only a shard |
| When communication happens | Once per step (gradients) | Many times per layer (activations) |
| Typical primitive | AllReduce | AllReduce / AllGather |
| Sensitivity to topology | Lower | High, usually needs high-bandwidth links |
| Problem it solves | Faster training with more samples | A single layer that doesn't fit on one GPU |

DP and TP are not mutually exclusive: real large-model training usually stacks both on the same device mesh — a few GPUs within a node do TP, while DP scales out across nodes, each handling a different constraint on capacity and bandwidth.

## What to calculate

To check whether plain DP is enough, first see whether the parameters fit on one GPU:

```text
per-GPU memory ≈ parameters + gradients + optimizer state + activations
```

With FP32 Adam, parameters, gradients, and the first/second moment each take one copy, roughly `4 × parameter count × 4 bytes` (excluding activations). Once that number exceeds a single GPU's memory, plain DP can no longer carry the model, and you need TP, FSDP2, or a combination of both.

## Hands-on check

1. Run DDP for a small model on two GPUs, and print each rank's parameters before and after the update — confirm they match exactly once gradients are synced.
2. Hand-write a small example that splits a linear layer's weight column-wise across two GPUs, and verify the stitched-together output matches the single-GPU result.
3. Count how many communication calls happen per step, and how much data each one moves, for the DP version versus the TP version.

## Common failure

- Assuming TP only syncs once at the end of a step like DP does — its communication is actually inside every layer, and far more frequent;
- Assuming TP removes the need for DP — most large-scale training still needs DP (or a variant) to scale throughput across nodes;
- Treating "it's split into N shards" as solving memory on its own — activations and communication buffers still need to be accounted for separately.

Next: [Module 04 · Distributed Training](../../modules/04-distributed-training.en.md)
