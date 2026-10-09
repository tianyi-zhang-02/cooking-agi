# Distributed training: how are data, state, and computation divided?

[中文](distributed-training.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [one training step](../00-foundations/deep-dives/training-step.en.md), [precision and memory](../00-foundations/deep-dives/precision-and-memory.en.md)

“Multiple GPUs” does not describe the training system. It could mean complete replicas seeing different data, or each device holding only part of a model. Both communicate, for different reasons.

We will use a four-layer model. Sizes and costs below are teaching assumptions, not hardware benchmarks.

## Separate two needs first

If the full model fits on one device but training is slow, distributing data may help. If model state itself does not fit, it needs sharding or partitioned computation; replicating it does not solve OOM.

| Method | Main partition | Each device handles |
| --- | --- | --- |
| Data parallel / DDP | Data | Different examples through a complete model |
| FSDP / ZeRO | Some or all parameter, gradient, optimizer state | Shards, with materialization when needed |
| Tensor parallel / TP | Within-layer matrices | Part of an operation |
| Pipeline parallel / PP | Layers | A stage and its microbatches |

These can be combined. Explain a configuration by drawing data flow, state placement, and communication timing rather than only naming “3D parallelism.”

## DDP computes local gradients, then synchronizes

```text
GPU 0: example A → all four layers → gradient g0 ┐
GPU 1: example B → all four layers → gradient g1 ├→ average → local updates
GPU 2: example C → all four layers → gradient g2 ┤
GPU 3: example D → all four layers → gradient g3 ┘
```

Replicas remain aligned when initialization, optimizer state, and synchronized gradients agree. [PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html) synchronizes gradient buckets and can overlap communication with backward. The caller still arranges data partitioning.

DDP is a training interface; Ring AllReduce is one collective implementation algorithm. They are different layers of the system. DDP also works within one machine, and the backend and configuration determine communication algorithms. “DP means a parameter server; DDP means multi-node Ring” is not a definition.

Two examples per microbatch, four accumulation steps, and four data-parallel ranks give 32 examples per update. Do not multiply TP ranks into that sample count. Repetition or sampler padding can change the number of unique examples.

## The global denominator is easy to miss

Rank 0 has two valid tokens with summed loss four; rank 1 has six with summed loss six. Averaging their local means gives 1.5, while the token-weighted mean is 1.25.

Let $T$ be valid tokens across the entire update window, $W$ the data-parallel world size, and $S_r$ rank $r$'s differentiable summed loss. If synchronization averages gradients, backpropagate locally through

$$
\widetilde{\mathcal L}_r=\frac{W S_r}{T}.
$$

After synchronization,

$$
\frac1W\sum_r\nabla\widetilde{\mathcal L}_r
=\frac{\sum_r\nabla S_r}{T}.
$$

Counts do not require gradients and must be aggregated first. If a framework sums gradients or adds other normalization, adjust the factor rather than applying it blindly.

```python
loss_sums = [4.0, 6.0]
valid_counts = [2, 6]
world_size = len(loss_sums)
global_tokens = sum(valid_counts)
local_objectives = [world_size * total / global_tokens for total in loss_sums]
ddp_average = sum(local_objectives) / world_size
assert ddp_average == 1.25
assert sum(total / count for total, count in zip(loss_sums, valid_counts)) / world_size == 1.5
```

This illustrates scalar accounting; the gradient result follows from the linear sum above. Accumulated microbatches need a denominator covering the entire window. A globally empty window must be skipped or rejected, not divided by zero.

## PS and ring: how do gradients reach the update?

In a Parameter Server (PS) design, workers send gradients to servers maintaining parameters or shards. It can be synchronous or asynchronous; PS does not itself mean stale gradients. A synchronous AllReduce instead returns the reduced result to every participant. Both need a definition of which samples belong to an update.

Ring AllReduce commonly combines reduce-scatter, which leaves each rank with a fully reduced shard, and all-gather, which distributes those shards. Separate a collective's semantics from its algorithm. [NCCL's introduction](https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/) explains ring communication; it is a historical article, not a claim that every current collective uses a ring.

For $P$ ranks and a message of $S$ bytes per rank, assume equal chunks on an ideal ring. Each round sends $S/P$, with $P-1$ rounds per phase. Thus **bytes sent per rank** are

$$V_{\text{send}}=2\frac{P-1}{P}S.$$

Received bytes are equal; counting both directions doubles this number. With four ranks and 120 MB of gradients, each rank sends 180 MB, rather than the 360 MB required to send its whole gradient separately to the other three ranks.

With per-round latency $\alpha$ and effective bandwidth $B$, a simplified serial-round cost is

$$T\approx2(P-1)\alpha+2\frac{P-1}{P}\frac{S}{B}.$$

This omits reduction computation, contention, chunk overlap, and actual topology. Large messages can be bandwidth-limited while small ones are latency-limited. The equation is not a substitute for profiling.

```python
def ring_traffic(message_bytes, ranks):
    if type(ranks) is not int or ranks < 1 or type(message_bytes) is not int or message_bytes < 0:
        raise ValueError("expected positive integer ranks and nonnegative integer bytes")
    sent = 2 * (ranks - 1) * message_bytes / ranks
    return {"sent_per_rank": sent, "received_per_rank": sent, "rounds": 2 * (ranks - 1)}

assert ring_traffic(120_000_000, 4)["sent_per_rank"] == 180_000_000
assert ring_traffic(120_000_000, 1)["rounds"] == 0
```

### Asynchrony changes more than waiting time

Take $L(w)=w^2/2$ and learning rate 0.5. Starting at $w=2$, the first update gives 1. A current gradient of 1 then gives 0.5; a delayed worker's gradient computed at the old value 2 gives 0 instead. The stale update happens to be closer to the optimum here, but this is no general advantage: it follows a different optimization trajectory.

Record gradient parameter versions, acceptable age, rejection or reweighting rules, and treatment of in-flight work on recovery. Faster synchronization alone does not establish training equivalence.

## FSDP and ZeRO reduce replicated state

[ZeRO](https://arxiv.org/abs/1910.02054) progressively shards optimizer states, gradients, and parameters. The [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) explains a full-sharding implementation. They are not identical APIs, but address duplicated training state.

One full-shard path that reshards after computation is:

```text
local parameter shard
    → all-gather parameters needed by the current module
    → forward / backward
    → reduce-scatter gradients
    → update the local shard with local optimizer state
```

Parameter release, backward regathers, and prefetching affect peak memory and communication. Dividing total training memory by device count is insufficient.

Suppose parameters, gradients, and optimizer states occupy 2, 2, and 12 GB. On four devices, ideal persistent state is:

| Strategy | Per-device state |
| --- | ---: |
| Replicate all | $2+2+12=16$ GB |
| Shard optimizer only | $2+2+12/4=7$ GB |
| Shard optimizer and gradients | $2+2/4+12/4=5.5$ GB |
| Shard all three | $(2+2+12)/4=4$ GB |

This excludes activations, communication buffers, and temporary gathers. Four GB is not a peak-memory promise, and the largest computational unit must still fit.

## TP splits individual matrix operations

Using row vectors, $Y=XW$ can be column-partitioned:

$$
W=[W_1\;W_2],\qquad Y=[XW_1\;XW_2].
$$

Partitioning the input dimension instead gives partial results to sum:

$$
X=[X_1\;X_2],\quad
W=\begin{bmatrix}W_1\\W_2\end{bmatrix},\quad
Y=X_1W_1+X_2W_2.
$$

For $X=[1,2]$ and $W=\begin{bmatrix}1&3\\2&4\end{bmatrix}$, the full output is $[5,11]$. Row partitioning produces $[1,3]$ and $[4,8]$, which must be summed, not averaged.

[Megatron-LM](https://arxiv.org/abs/1909.08053) shows how complementary column and row partitions organize Transformer layers. Whether an intermediate remains sharded depends on the next operation; gathering everything after every layer wastes opportunities.

Frequent TP communication benefits from fast interconnects. Splitting small matrices too finely can cost more in communication and scheduling than it saves.

## PP divides layers and schedules microbatches

Place layers 1–2 on GPU 0 and 3–4 on GPU 1. With one serial batch, the second device starts idle and the first later waits. Microbatches let stages process different data concurrently.

For an ideal equal-time, zero-communication GPipe-style fill/drain schedule, with $P$ stages and $M$ microbatches, utilization is approximately

$$
\frac{M}{M+P-1}.
$$

Two stages and four microbatches give 80%. Real bubbles, forward/backward imbalance, uneven layers, and memory limits alter this estimate. More microbatches are not free. [GPipe](https://arxiv.org/abs/1811.06965) is a primary reference for the mechanism.

**Operation order and weight-update timing are separate choices.** 1F1B alternates forward and backward operations in steady state; it does not require an optimizer update after each microbatch. Synchronous training can accumulate the full batch first. [PyTorch pipelining](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html) supports multiple schedules. Discuss [PipeDream](https://arxiv.org/abs/1806.03377)'s asynchronous updates and weight stashing separately, with explicit weight versions.

### Why does PipeDream retain old weights?

When a pipeline allows updates between a microbatch's forward and backward passes, parameters can change in between. Weight stashing retains the forward version so backward uses matching local derivatives. [PipeDream](https://arxiv.org/abs/1806.03377)

For a two-layer scalar model $y=bax$ with $x=1,a=2,b=3,L=y^2/2$, forward gives 6 and the correct $\partial L/\partial a$ is $6\times3=18$. Reusing the old activation but a newly updated $b=4$ gives 24. This illustrates the version issue, not a complete PipeDream schedule.

Retained versions consume state. Local forward/backward consistency does not automatically mean every stage uses the same global version, or make asynchronous training equivalent to synchronous large-batch SGD. Compare update semantics as well as throughput, not just pipeline bubbles.

## SP, CP, and EP partition different things

| Method | What to inspect | What cannot be omitted |
| --- | --- | --- |
| Sequence parallel / SP | In Megatron, sequence-sharded activations around operations such as LayerNorm and dropout, alongside TP | This alone does not partition every attention layer across the context |
| Context parallel / CP | Sequence-partitioned inputs and activations | Local queries still need remote keys and values |
| Expert parallel / EP | Expert placement across devices | Token dispatch, result combination, and load imbalance |

These names follow [Megatron's SP / CP distinction](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html). Other frameworks may use SP more broadly. Inspect the data flow, not only the name.

Split eight tokens across two GPUs, four each. If each device computes causal attention only locally, token eight loses access to the first four tokens. That changes the model. CP must preserve those dependencies while distributing memory and computation; concatenating local outputs is insufficient.

EP solves a different problem. If a token selects two remote experts, send its activation, then return and combine the results at the original position. [Megatron's MoE parallelism guide](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md) describes composition with other strategies. The fraction of selected experts is not a communication-byte ratio or a guaranteed speedup. Hot experts can still leave other devices waiting.

## Serving DP does not always mean independent replicas

Training DDP synchronizes gradients; serving data parallelism typically distributes requests. With MoE, check whether experts are shared across those replicas. In [vLLM's EP deployment documentation](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/), checked on 2026-10-09, enabling EP forms an expert group of size TP × DP. Attention replication/sharding and expert placement are different decisions. This is a framework convention, not a universal definition.

Take a **hypothetical weight budget**: 6 GiB of shardable non-expert weights and 42 GiB of routed experts on four GPUs. Ignore replicated small tensors, quantization metadata, KV, and workspace:

| Configuration | Non-expert weights per GPU | Routed experts per GPU | Total per GPU |
| --- | --- | --- | --- |
| TP1 / DP4 / EP4 | 6 GiB | 10.5 GiB | 16.5 GiB |
| TP4 / DP1 / EP4 | 1.5 GiB | 10.5 GiB | 12 GiB |

The second saves 4.5 GiB per GPU but requires TP cooperation for attention. The first keeps attention local to each rank while still exchanging activations for remote experts. **It is not four completely independent services.** Consider replica caches and queues together with shared-expert scheduling.

This estimates capacity, not throughput. Hold arrivals, input/output lengths, and cache warmth fixed; measure first-token latency, output intervals, tail latency, and expert load. Matrices sharded too finely may save memory while increasing waiting.

## Where does PP keep KV, and what crosses a boundary?

Assume an ordinary decoder partitioned by layer, without cache offload or prefill/decode disaggregation. Each GPU retains KV for its own layers. Adjacent stages exchange boundary activations rather than repeatedly moving the entire historical KV cache.

“Only activations” does not necessarily mean a small transfer. With hidden size 3072 and two bytes per BF16 element:

| One boundary transfer | Element count | Raw bytes |
| --- | --- | --- |
| Eight requests decoding one token each | 8 × 3072 | 48 KiB |
| Eight requests prefilling 4096 tokens each | 8 × 4096 × 3072 | 192 MiB |

Extra residual tensors, metadata, protocol costs, and synchronization are excluded. Chunking long prefill changes scheduling and pipeline behavior; a small decode message cannot describe every workload.

[vLLM's scaling guide](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) recommends considering PP for certain configurations without NVLink to reduce communication. That is not proof of optimality on every PCIe machine. A single request still traverses stages sequentially; unequal stage work creates idle time. If the model fits on one GPU, compare independent replicas, quantized single-GPU serving, TP, and PP under the same workload. Disaggregated serving that transfers KV introduces a different communication path.

## Offload moves state, but transfers still take time

CPU memory relieves GPU pressure; it is not free expansion. CPU optimizer updates require accounting for outgoing gradients, returning parameters, and CPU compute. With small batches, a fast GPU can spend its time waiting for the CPU. [ZeRO-Offload](https://arxiv.org/abs/2101.06840) studies these placement and scheduling choices.

Dividing ideal state memory by device count covers only part of the budget. Temporary materialization, activations, prefetching, bandwidth, and optimizer completion still matter. Larger models, longer sequences, and more devices can each move the bottleneck.

For an original estimate, transfer 2 GB of gradients out and 2 GB of weights back in sequence, with assumed effective bandwidth of 25 GB/s in each direction. Transfer alone takes `2/25 + 2/25 = 0.16` seconds, before the CPU update. These are decimal GB and hypothetical values, not a measured device specification.

Hiding this time requires independent computation and timely arrival of weights needed by the next step. The critical path depends on dependencies, prefetch granularity, memory, and contention; taking one `max` over every duration is not a complete schedule model.

## Recovery needs more than parameters

The simplest verifiable checkpoint boundary drains the pipeline after a synchronous update. Decide which states to restore or explicitly reset:

| State | What can change if omitted |
| --- | --- |
| Parameters, optimizer, scheduler, step | Adam momentum or the learning-rate trajectory |
| RNG, sampler / data cursor | Repeated or skipped samples and changed augmentations |
| Partial gradient accumulation and valid-token counts | A different batch or denominator |
| Shard mapping, parameter versions, in-flight microbatches | Mismatched state or repeated application of old gradients |

Not every system supports recovery at arbitrary instants. Compare one more step after saving with one more step after reloading, fixing randomness and checking loss, gradients, and updated parameters. If changing world size is supported, test resharding separately. Successfully opening a checkpoint file is not a correctness test.

## Check correctness before chasing throughput

Use a tiny model, fixed samples, and disabled dropout to compare one single-device and distributed update: loss, valid-token count, gradients, and updated weights. Then test variable lengths, empty samples, accumulation, and checkpoint restoration.

For performance, record valid tokens/s, per-device peaks, communication, and straggler waits. High utilization alone does not establish useful work; padding and repeated failed attempts are not productive throughput.

Start simply: a DDP baseline if the model fits; sharding when state does not fit; TP for oversized within-layer computation; PP for layer partitioning. Every added parallelism should explain both the saving and the new communication.
