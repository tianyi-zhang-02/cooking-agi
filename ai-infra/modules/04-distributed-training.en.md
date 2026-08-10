# Module 04 · Distributed Training

[中文](04-distributed-training.md) · **English** · [Back to AI Infra](../README.en.md)

> Reading time: ~5 minutes · Level: Intermediate · Freshness: Fast-moving · Last reviewed: 2026-08

## What this module solves

When model state, activations, or optimizer state exceed one GPU's capacity, training must be partitioned. This module asks what to partition, how to synchronize it, and whether the added communication, scheduling, and failure complexity is justified.

## Learning goals

- Break training memory into its major components.
- Understand DDP, FSDP/ZeRO, TP, PP, CP, and EP.
- Explain AllReduce, AllGather, ReduceScatter, and AllToAll.
- Estimate memory and communication for a parallel strategy.
- Recognize communication waits, pipeline bubbles, and stragglers in a timeline.

## Core notes

### What consumes training memory

```text
parameters
+ gradients
+ optimizer state
+ activations
+ communication and operator buffers
+ allocator fragmentation
```

“Parameter count × dtype size” severely underestimates training memory. Activations also vary with batch size, sequence length, hidden size, layer count, and checkpointing strategy.

### Parallelism dimensions

| Method | What is partitioned | Typical communication | Main cost |
| --- | --- | --- | --- |
| DDP | data | gradient AllReduce | every GPU keeps full model state |
| FSDP / ZeRO | parameters, gradients, optimizer state | AllGather, ReduceScatter | repeated parameter materialization |
| TP | intra-layer tensors / matmuls | AllReduce, AllGather | per-layer communication, topology sensitivity |
| PP | layers | point-to-point | bubbles and load balance |
| CP / SP | sequence | gather/scatter-like communication | attention communication complexity |
| EP | MoE experts | AllToAll | token routing and load imbalance |

These strategies are not mutually exclusive. Large runs arrange devices into a multidimensional mesh and apply different sharding choices along different dimensions.

### Collective communication

- **AllReduce:** every rank receives the reduction of all inputs.
- **AllGather:** every rank collects every rank's shard.
- **ReduceScatter:** values are reduced and the result is partitioned across ranks.
- **AllToAll:** every rank sends a different shard to every other rank.
- **Broadcast:** one root sends data to every rank.

Communication depends on message size, latency, bandwidth, topology, and algorithm. Intra-node NVLink and inter-node networking should be treated as different cost levels.

### Overlapping computation and communication

When later computation does not depend on a communication result, NCCL collectives and other kernels can progress on different streams. Step time may approach the maximum of compute and communication rather than their sum.

Overlap is not automatic: dependencies, bucket sizes, stream scheduling, link contention, and kernel resources all affect the outcome.

### Checkpointing and recovery

Sharded training may produce sharded checkpoints. Reliable recovery must capture model and optimizer state, schedulers, random state, data position, and parallel topology. It must also support topology changes or explicitly reject incompatible recovery.

## Quantities to calculate

Basic parameter storage is:

```text
parameter memory = parameter count × bytes per parameter
```

For ring AllReduce, approximate bytes transferred per rank are:

```text
2 × (world_size - 1) / world_size × tensor bytes
```

This is a bandwidth intuition, not an exact timing model; protocol, topology, chunking, and contention still matter.

Scaling efficiency is:

```text
scaling efficiency = single-device time / (device count × distributed time)
```

Any throughput comparison must define global batch, token count, and gradient accumulation consistently.

## Hands-on work

1. Run two-GPU DDP and observe gradient AllReduce.
2. Compare peak memory and step time between DDP and FSDP for the same model.
3. Draw a compute/communication timeline for one training step.
4. Change bucket size or micro-batch size and observe overlap and bubbles.
5. Artificially slow one rank and observe how a straggler delays global synchronization.

## Common misconceptions

- More GPUs do not guarantee linear speedup.
- Fitting the model does not prove the parallel strategy is efficient.
- Mean step time hides long-tail stalls and stragglers.
- NCCL bandwidth alone cannot explain every communication wait.
- Successfully writing a checkpoint does not prove correct restoration.

## Mastery check

- Why can DDP not solve a model that does not fit on one GPU?
- What communication does FSDP exchange for memory savings?
- How do TP and PP differ in their topology requirements?
- Why does MoE rely heavily on AllToAll?
- What evidence proves computation and communication actually overlap?

Next: [Module 05 · LLM Inference](05-llm-inference.en.md)
