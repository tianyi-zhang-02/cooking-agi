# MoE: what it costs to train and serve

[中文](systems.md) · **English**

> Reading time: ~6 min · Level: advanced · Last reviewed: 2026-10

If each token runs only 8 experts, why is MoE still hard to deploy? Because computing less is only one item on the bill. The other two are moving tokens between GPUs and storing every expert's weights.

**MoE lowers compute per token and turns the cost into communication and memory; whether it pays off depends on interconnect bandwidth, total memory and batch size.** Below we work out each item at DeepSeek-V3's sizes.

## Expert parallelism and two all-to-alls

In a large model the experts live on different GPUs. The forward pass of one MoE layer then takes three steps (as in GShard):

1. **dispatch**: one all-to-all sends each token to the GPUs that hold the experts it chose;
2. each GPU runs its own experts;
3. **combine**: a second all-to-all sends the results back to where each token came from.

This is a typical logical expert-parallel flow, not a requirement for two cross-device collectives in every deployment. Local experts need no remote dispatch. Traffic also depends on expert placement and coalescing sends to the same node. Profile dispatch, expert computation, return traffic, and overlap separately when diagnosing a slow step. For how long a collective takes and how to estimate it from bandwidth, see [Performance: multi-GPU communication](../../12-infra/performance.en.md#communication).

### Worked example: how many bytes does one token send? {#dispatch-bytes}

Each token sends its hidden state to every expert it chose, so the one-way payload per layer is

$$
\text{bytes} = k \times d \times s .
$$

$k$ is the number of experts per token, $d$ the hidden size, and $s$ the bytes per number. With DeepSeek-V3's $k=8$ and $d=7168$, in BF16 ($s=2$):

$$
8 \times 7168 \times 2 = 114{,}688\ \text{B} = 112\ \text{KiB}.
$$

That is before merging sends to the same node, and combine sends the results back again. A micro-batch of 4,096 tokens moves 448 MiB per layer in each direction. Real systems shrink this: DeepSeek-V3 quantizes activations to FP8 before dispatch ($s=1$, half the bytes) while keeping combine in BF16, and several experts on one node receive a single copy, so the cross-node part carries at most 4 copies, or $4 \times 7168 \times 1$ B = 28 KiB in FP8. [V3 §3.2.2, §3.3.3](https://arxiv.org/html/2412.19437v2#S3.SS2)

## Limit how many machines a token touches

Bandwidth between machines is far lower than inside one: for the V3 report's H800 cluster, 160 GB/s for NVLink within a node and 50 GB/s for cross-node IB, about 3.2× apart. For NVLink domain sizes and bandwidth by GPU generation, see [NVIDIA generations](../../12-infra/nvidia-generations.en.md). DeepSeek-V3 uses node-limited routing: each token may go to at most 4 nodes, chosen by the sum of the highest expert scores on each node. That keeps top-8 routing while capping cross-node traffic.

## Memory: sparsity only helps once it fits

Full GPU residency needs storage for all weights, not just active parameters. DeepSeek-V3's main model has 671B parameters; released weights including MTP add roughly 14B. gpt-oss expert weights use MXFP4 at about 4.25 bits per parameter, but not all tensors use that precision. Offloading can move some weights to CPUs, putting bandwidth and cache hits into the latency calculation. Sources for these models are in the [report list](review.en.md).

## When decoding, batch size changes the arithmetic

Small-batch decoding is often bandwidth-limited (for why a larger batch moves decoding toward compute-bound, see [Performance: decode batching](../../12-infra/performance.en.md#decode-batching)), but long-context KV reads, cross-device communication, or larger matrix operations can dominate too. Confirm the bottleneck with a profiler.

- **With a small batch**, a step may use few experts and read fewer weights, but small-matrix inefficiency and dispatch overhead can offset that gain.
- **With a large batch**, dispersed routing touches more experts, potentially approaching all their weights, spread over more tokens. Skewed routing may still use only a few experts.

### Worked example: how many experts does one decode step touch? {#experts-touched}

A layer has $N$ routed experts and each token picks $k$. Assume uniform routing and independent tokens. One token misses a given expert with probability $1-k/N$, so all $B$ tokens in a step miss it with probability $(1-k/N)^B$. The share of experts the step must read is about

$$
1-\left(1-\frac{k}{N}\right)^{B}.
$$

With V3's $N=256$ and $k=8$:

| Tokens in the step, $B$ | Share of experts touched |
| --- | --- |
| 1 | 3.1% |
| 32 | 63.8% |
| 128 | 98.3% |

```python
def experts_touched(experts, top_k, batch):
    if not 0 < top_k <= experts or batch < 0:
        raise ValueError("need 0 < top_k <= experts and batch >= 0")
    return 1 - (1 - top_k / experts) ** batch

assert round(experts_touched(256, 8, 1), 3) == 0.031
assert round(experts_touched(256, 8, 32), 3) == 0.638
assert round(experts_touched(256, 8, 128), 3) == 0.983
```

A single request reads only about 3% of each layer's expert weights per step; at a batch of 128 almost every expert is read, but each weight is now shared by more tokens. Skewed routing touches fewer experts than this uniform estimate. So MoE's "each token computes only a small part" pays off differently at different batch sizes, and with the all-to-alls on top, real throughput depends on the deployment.

## Numerical precision

The router's softmax is sensitive to numerics. Switch Transformer computes the router in fp32 while the rest stays in low precision; ST-MoE's router z-loss, covered in the [load-balancing note](load-balancing.en.md), addresses the same problem.

## In short

For MoE to be cheaper to deploy than a dense model of similar quality, several conditions usually need to hold together:

- **The weights fit**: memory is provisioned for total parameters, or the transfer latency of offloading is acceptable.
- **Communication is hidden**: dispatch and combine overlap with expert computation, or most experts sit inside one high-bandwidth domain such as an NVLink domain.
- **The batch is large enough**: each step has enough tokens to spread the cost of every expert weight it reads; with a tiny batch the step reads little but cannot amortize dispatch overhead.

Single-GPU, memory-tight, small-batch deployments chasing the lowest latency usually fail most of these, and a dense model is often simpler. For a specific deployment, profile how much time dispatch, expert computation and combine each take before deciding.
