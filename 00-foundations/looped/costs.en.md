# Looped Transformers: costs and limits

[中文](costs.md) · **English**

> Reading time: ~5 min · Last reviewed: 2026-10

## Compute and latency: loops are serial

At fixed work per loop, recurrent-block compute grows with depth; boundary stages add fixed costs. Later loops depend on earlier states and cannot simply execute simultaneously. Scheduling independent requests can improve utilization without removing that dependency. End-to-end latency also depends on batching, memory access, and queues, so it need not scale exactly with loop count.

## The KV cache grows with the loop count

Shared weights do not make K/V identical: each loop receives different hidden states. A full depth-indexed cache distinguishes (loop, layer). For 24 layers, four loops, K/V width 2048 each, and BF16:

$$\text{bytes/token}=2\times24\times4\times2048\times2=786{,}432$$

The first 2 counts K and V; the last counts bytes per element. This is 768 KiB, approximately 0.786 MB per token, or 3 GiB for one 4096-token sequence. It excludes weights, temporary activations, allocator overhead, and serving reserves. GQA requires using its actual KV width.

A prelude or coda that runs once adds fixed KV storage. For [Huginn](models.en.md)'s 2+4+2 layout, equal KV widths and full depth-indexed caching give $2+4R+2$ layer-level entries at $R$ loops. Four loops versus one therefore cost $20/8=2.5$ times the KV, not four times. This is storage accounting, not a latency or accuracy measurement.

## Can the loops share KV?

Sharing usually changes the computation rather than merely storing it more compactly. Compare checkpoint, prefill policy, decode policy, and output length before comparing results:

- [Ouro](https://arxiv.org/abs/2510.25741) reports small quality changes in some decode-only sharing settings. The reduction concerns that decode KV, not all GPU memory.
- [Continuous Depth Batching (v2, 2026-09-25)](https://arxiv.org/html/2608.09444v2), Table 4: GSM8K-CoT accuracy with greedy decoding and flexible matching is 78.09% / 0.23% / 72.40% for Ouro 1.4B (4 loops), using full / single-shared / first-then-shared caches. Huginn 3.5B (16 loops) scores 30.93% / 32.52% / 32.83%. Protocols differ from Ouro's, so these numbers do not directly refute Ouro's reported small quality changes. Its throughput experiments replay recorded lengths and exits; task quality is evaluated separately.
- [MELT (v3, 2026-09-28)](https://arxiv.org/html/2605.07721v3), Appendix B and Table 6, shows severe degradation for training-free sharing baselines on Ouro-1.4B-Thinking; **the degradation belongs to those baselines, not to MELT itself**. The proposed method adds a cache mechanism and post-training. Generation length and format can affect accumulated cache errors.
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) compares sparse per-depth storage and cross-loop sharing, with routing-dependent outcomes. Skipping tokens also changes which keys remain available.
- [Huginn](https://arxiv.org/abs/2502.05171) explores recurrent cache slots. Empirical tolerance is not exact equivalence or evidence that arbitrary models support the same policy.

Start with fixed prompts and decoding, compare full-cache and shared-cache logits, then test short answers, long reasoning, and peak memory. Check differences against expected floating-point roundoff, then measure the approximation's quality and benefits. Agreement on a few outputs does not prove cache equivalence.

## Training is harder

A deeper gradient chain can create stability and activation-memory challenges, but is not necessarily unstable. Full backpropagation accumulates gradients from each use of shared weights. Truncation saves activations but changes the gradient. Input injection maintains access to the input; it is not a universal stability guarantee.

## Knowledge-capacity experiments are not universal bounds {#it-does-not-store-more-knowledge}

The approximate 2-bit-per-parameter result in [Ouro §6.1 (v5; unchanged in v6)](https://arxiv.org/html/2510.25741v5) measures 1M–40M-parameter GPT-2-style models trained on synthetic biographies, not the released 1.4B / 2.6B checkpoints or a universal neural-network capacity bound. Looping adds no weights, but training, encoding, external memory, and input information all affect answerable knowledge. Evaluate parametric storage, use of contextual facts, and reasoning over facts separately.

## When it is worth it

- weight-storage constraints justify extra compute, without moving the bottleneck to KV;
- additional loops demonstrably help the target task without unacceptable regressions elsewhere;
- dense, looped, CoT, and retrieval alternatives are compared under a fixed budget, rather than ranked by parameter count alone.
