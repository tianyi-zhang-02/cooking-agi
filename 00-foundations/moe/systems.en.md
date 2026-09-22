# MoE: what it costs to train and serve

[中文](systems.md) · **English**

> Reading time: ~2 min · Level: advanced · Last reviewed: 2026-09

## Expert parallelism and two all-to-alls

In a large model the experts live on different GPUs. The forward pass of one MoE layer then takes three steps (as in GShard):

1. **dispatch**: one all-to-all sends each token to the GPU that holds the experts it chose;
2. each GPU runs its own experts;
3. **combine**: a second all-to-all sends the results back to where each token came from.

That is two exchanges per layer, and the traffic grows with $k$ and with the number of tokens. Much of MoE's training and serving efficiency comes down to whether those exchanges can overlap with compute.

## Limit how many machines a token touches

Bandwidth between machines is far lower than inside one. DeepSeek-V3 uses node-limited routing: each token may go to at most 4 nodes, chosen by the sum of the highest expert scores on each node. That keeps top-8 routing while capping cross-node traffic.

## Memory: sparsity only helps once it fits

Memory has to hold **all** the parameters. DeepSeek-V3 has 671B parameters (the 685B shown on Hugging Face includes a 14B multi-token prediction module), so the weights alone need many GPUs even though only 37B are active. gpt-oss stores its MoE weights in MXFP4, about 4.25 bits per parameter, precisely to fit the total into less memory.

## When decoding, batch size changes the arithmetic

Decoding is usually memory-bound: every step reads the weights it uses from memory.

- **With a small batch**, only a few experts are used per step, so the weights read scale roughly with the active parameters; this is where MoE is fast;
- **with a large batch**, different tokens pick different experts, almost every expert is used in each step, and the weights read approach the total, spread over more tokens.

So MoE's "each token computes only a small part" pays off differently at different batch sizes, and with the all-to-alls on top, real throughput depends on the deployment.

## Numerical precision

The router's softmax is sensitive to numerics. Switch Transformer computes the router in fp32 while the rest stays in low precision; ST-MoE's z-loss (previous note) addresses the same problem.

## In short

MoE suits settings with enough memory and GPUs, where you want more parameters but need to control compute per token, such as large-scale serving. For a single GPU, tight memory, and small-batch deployments chasing the lowest latency, a dense model is often simpler.
