# Looped Transformers: costs and limits

[中文](costs.md) · **English**

> Reading time: ~5 min · Level: advanced · Last reviewed: 2026-10-09

## Compute and latency: loops are serial

At fixed work per loop, recurrent-block compute grows with depth; boundary stages add fixed costs. Later loops depend on earlier states and cannot simply execute simultaneously. Scheduling independent requests can improve utilization without removing that dependency. End-to-end latency also depends on batching, memory access, and queues, so it need not scale exactly with loop count.

## The KV cache grows with the loop count

Shared weights do not make K/V identical: each loop receives different hidden states. A full depth-indexed cache distinguishes (loop, layer). For 24 layers, four loops, K/V width 2048 each, and BF16:

$$\text{bytes/token}=2\times24\times4\times2048\times2=786{,}432$$

The first 2 counts K and V; the last counts bytes per element. This is 768 KiB, approximately 0.786 MB per token, or 3 GiB for one 4096-token sequence. It excludes weights, temporary activations, allocator overhead, and serving reserves. GQA requires using its actual KV width.

## Can the loops share KV?

Sharing usually changes the computation rather than merely storing it more compactly. Compare checkpoint, prefill policy, decode policy, and output length before comparing results:

- [Ouro](https://arxiv.org/abs/2510.25741) reports small quality changes in some decode-only sharing settings. The reduction concerns that decode KV, not all GPU memory.
- [Continuous Depth Batching](https://arxiv.org/abs/2608.09444) (2026), Table 1: Ouro GSM8K-CoT scores are 77.86 / 0.23 / 71.34 for full / single-shared / first-then-shared caches; Huginn scores 33.74 / 33.97 / 34.80 at the tested full depth. Protocols differ, so this is not an identical-setting refutation. Its throughput experiments replay recorded lengths and exits; task quality is evaluated separately.
- [MELT](https://arxiv.org/abs/2605.07721) (2026), Table 5, shows severe degradation for untrained sharing baselines, **not all-zero performance of MELT itself**. The proposed method adds a cache mechanism and post-training. Generation length and format can affect accumulated cache errors.
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) compares sparse per-depth storage and cross-loop sharing, with routing-dependent outcomes. Skipping tokens also changes which keys remain available.
- [Huginn](https://arxiv.org/abs/2502.05171) explores recurrent cache slots. Empirical tolerance is not exact equivalence or evidence that arbitrary models support the same policy.

Start with fixed prompts and decoding, compare full-cache and shared-cache logits, then test short answers, long reasoning, and peak memory. If outputs differ, describe sharing as an approximation with measured errors and benefits.

## Training is harder

A deeper gradient chain can create stability and activation-memory challenges, but is not necessarily unstable. Full backpropagation accumulates gradients from each use of shared weights. Truncation saves activations but changes the gradient. Input injection maintains access to the input; it is not a universal stability guarantee.

## Knowledge-capacity experiments are not universal bounds {#it-does-not-store-more-knowledge}

Ouro's approximate 2-bit-per-parameter result comes from a particular synthetic dataset and training protocol, not a universal neural-network capacity bound. Looping adds no weights, but training, encoding, external memory, and input information all affect answerable knowledge. Evaluate parametric storage, use of contextual facts, and reasoning over facts separately.

## When it is worth it

- weight-storage constraints justify extra compute, without moving the bottleneck to KV;
- additional loops demonstrably help the target task without unacceptable regressions elsewhere;
- dense, looped, CoT, and retrieval alternatives are compared under a fixed budget, rather than ranked by parameter count alone.
