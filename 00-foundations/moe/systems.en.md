# MoE: what it costs to train and serve

[中文](systems.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-10-09

## Expert parallelism and two all-to-alls

In a large model the experts live on different GPUs. The forward pass of one MoE layer then takes three steps (as in GShard):

1. **dispatch**: one all-to-all sends each token to the GPU that holds the experts it chose;
2. each GPU runs its own experts;
3. **combine**: a second all-to-all sends the results back to where each token came from.

This is a typical logical expert-parallel flow, not a requirement for two cross-device collectives in every deployment. Local experts need no remote dispatch. Traffic also depends on expert placement and coalescing sends to the same node. Profile dispatch, expert computation, return traffic, and overlap separately when diagnosing a slow step.

## Limit how many machines a token touches

Bandwidth between machines is far lower than inside one. DeepSeek-V3 uses node-limited routing: each token may go to at most 4 nodes, chosen by the sum of the highest expert scores on each node. That keeps top-8 routing while capping cross-node traffic.

## Memory: sparsity only helps once it fits

Full GPU residency needs storage for all weights, not just active parameters. DeepSeek-V3's main model has 671B parameters; released weights including MTP add roughly 14B. gpt-oss expert weights use MXFP4 at about 4.25 bits per parameter, but not all tensors use that precision. Offloading can move some weights to CPUs, putting bandwidth and cache hits into the latency calculation. Sources for these models are in the [report list](review.en.md).

## When decoding, batch size changes the arithmetic

Small-batch decoding is often bandwidth-limited, but long-context KV reads, cross-device communication, or larger matrix operations can dominate too. Confirm the bottleneck with a profiler.

- **With a small batch**, a step may use few experts and read fewer weights, but small-matrix inefficiency and dispatch overhead can offset that gain.
- **With a large batch**, dispersed routing touches more experts, potentially approaching all their weights, spread over more tokens. Skewed routing may still use only a few experts.

So MoE's "each token computes only a small part" pays off differently at different batch sizes, and with the all-to-alls on top, real throughput depends on the deployment.

## Numerical precision

The router's softmax is sensitive to numerics. Switch Transformer computes the router in fp32 while the rest stays in low precision; ST-MoE's z-loss (previous note) addresses the same problem.

## In short

MoE suits settings with enough memory and GPUs, where you want more parameters but need to control compute per token, such as large-scale serving. For a single GPU, tight memory, and small-batch deployments chasing the lowest latency, a dense model is often simpler.
