# Autoregressive and diffusion LMs: comparing and combining them

[中文](ar-and-diffusion.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10-23

If diffusion language models (dLLMs) can fill several token positions at once, why haven't they broadly replaced autoregressive (AR) models? Fewer steps need not take less time. And when generation is faster, we still need to check whether answer quality has changed.

This note builds on [decoding and speed](decoding-and-speed.en.md): which tasks might benefit, how blocks make caching possible, and how AR and diffusion can work together. We then examine a few recent results and their limits.

## An advantage at what?

| Goal | What diffusion offers | What the comparison still needs |
| --- | --- | --- |
| Latency | Several positions can be committed per forward pass | Cost per pass, extra commit passes, answer quality |
| Infilling | Missing positions can attend directly to both sides | An AR model trained for FIM, not just ordinary continuation |
| Data reuse | Different masks produce different prediction tasks | A distinction between saving fresh data and saving compute |
| Order | More certain positions can be filled first | Checks for inconsistent parallel choices; whether the sampler can revise committed tokens |

Consider six output tokens. If ordinary AR decoding takes 1 ms per step, it needs 6 ms. Diffusion might need only three steps, but at 3 ms per whole-block pass it takes 9 ms. **Half as many steps, more time.** At 1.5 ms per pass, it would instead take 4.5 ms. These are teaching numbers, not measurements; they omit prefill, EOS and scheduling overhead.

Nor should every AR baseline be limited to one token per forward pass. [Speculative decoding](../../00-foundations/deep-dives/dflash.en.md) can verify several draft tokens together. Deployment decisions should compare usable implementations, not just the simplest decoding loops.

The same care applies to infilling. Given `The parcel arrived on [M] after a two-day delay.`, a bidirectional model can use both sides. But an [AR model trained for fill-in-the-middle (FIM)](https://arxiv.org/abs/2207.14255) reads the prefix and suffix before generating the missing span. Infilling is not exclusive to diffusion; constraint satisfaction, edit quality and latency are more useful comparisons.

Data efficiency is conditional too. [Prabhudesai et al.](https://arxiv.org/abs/2507.15857) study repeated use of a fixed dataset. Their result does not automatically carry over to plentiful data with a tight compute budget. The [training-objective note](masked-diffusion.en.md) covers that setting in more detail.

<a id="block-cache"></a>

## After splitting into blocks, what can we reuse?

[Block Diffusion](https://arxiv.org/html/2503.09573v2) combines two orders: left to right across blocks, bidirectional denoising within the current block. Sequential dependencies remain, but their unit is a block rather than one token.

Take a short sentence. For this example only, each word is one token and a block holds three tokens. Real tokenization may differ. The trace illustrates one possible filling order, not a measured model output.

```text
Target: The cats are | on the mat

First block
[M] [M] [M]
The cats [M]
The cats are       → compute and store KV for this completed block

Second block, with access to the first block's KV throughout
[M] [M] [M]
on [M] mat
on the mat        → compute and store KV for this completed block
```

Attention visibility in this example is:

| Query location | Can attend to block 1 | Can attend to block 2 |
| --- | --- | --- |
| Any position in block 1 | Yes, bidirectionally within the block | No |
| Any position in block 2 | Yes | Yes, bidirectionally within the block |

As the second block changes from `[M] [M] [M]` to `on the mat`, the first block cannot see those changes. With positions, weights and inference computation held fixed, its representations at every layer stay unchanged, so its KV can be reused exactly. **Exact here means equivalent to recomputing the same block-causal model, not equivalent to a fully bidirectional model.**

Two implementation details are easy to miss.

**First, the final prediction pass may still have masks in its input.** When `The cats [M]` produces `are`, the cached third position still comes from the masked input. It is not yet the cache for `The cats are`. BD3-LM runs a forward pass over the completed clean block; that pass belongs in the latency budget. The cache update appears explicitly in [Algorithm 2](https://arxiv.org/html/2503.09573v2#S3.SS2).

**Second, a fixed token need not have fixed representations.** Even after `on` has been committed in the second block, it can attend to the other two positions. Changing them from `[M] mat` to `the mat` can change its representations in later layers. Exact caching of the completed prefix therefore does not justify reusing every committed position's KV inside the active block. That reuse needs additional attention restrictions or an approximation.

Smaller blocks generally make prefix reuse easier but leave fewer positions to decode in parallel. Larger blocks expose more parallel work but may require more denoising rounds. Compare quality, round count and cache overhead together when choosing a block size.

## Let AR plan and diffusion write

Another hybrid chooses the parts of an answer before generating them, rather than cutting the answer into fixed-length blocks.

Suppose the task is to describe two cities and recommend one. The descriptions can be developed separately; the recommendation depends on the comparison and can wait. This is a teaching example, not an output reported in a paper.

```text
AR plan: describe both cities, then compare
                 ├─ City A description ─┐
Diffusion writes:│                      ├─ wait for both
                 └─ City B description ─┘
AR / later generation: use both descriptions to recommend a city
```

Separating the descriptions is a planning assumption, not a guarantee of statistical independence. If they must align on the same budget or dates, those shared constraints need to appear in the plan, or the chunks need more communication. Otherwise, parallel generation can produce incompatible details.

[Planned Diffusion](https://arxiv.org/html/2510.18087v2), first released in October 2025 and revised in March 2026, trains a single Transformer to produce semantic chunks and length plans autoregressively, then denoise in parallel, with synchronization points for later dependencies. It does not require two models, but it does require training—not just a different attention mask on an existing AR checkpoint.

The plan uses next-token supervision; the body learns to recover masked content. Attention must also prevent the plan from seeing future body text. Check the implementation's loss indexing: diffusion does not universally mean unshifted labels. [Fast-dLLM v2, §3.2](https://arxiv.org/html/2509.26328v1#S3.SS2), retains shifted prediction.

Prepending an AR-generated prompt to an existing dLLM is a different experiment. [Think First, Diffuse Fast](https://arxiv.org/abs/2603.13243) explores such a training-free combination. Treat it as a specific preprint result, not evidence that any planner improves any dLLM.

<a id="hybrids"></a>

## Similar names, different mechanisms

| Method | Roles of AR and diffusion | Important distinction |
| --- | --- | --- |
| Block Diffusion | Autoregressive across blocks, denoising within each | Fixed blocks are not semantic plans |
| Planned Diffusion | AR produces a plan; diffusion expands semantic chunks | A trained hybrid, not merely a prompt prepended to a model |
| [DFlash](../../00-foundations/deep-dives/dflash.en.md) | Diffusion drafts; an AR target verifies | Correct acceptance and correction preserve the target distribution; the target is not converted into a dLLM |
| [Block Transformer](https://research.google/pubs/block-transformer-global-to-local-language-modeling-for-fast-inference/) | Global block-level modeling with local AR token decoding | An autoregressive architecture, not masked diffusion |

Blocks organize computation. To identify the actual method, ask which conditional distribution is learned, what gets generated first, and which outputs still need verification.

## Reading the recent results

These works affect the comparisons above; this is not a leaderboard. Numerical results below are reported by the authors and have not been reproduced for this site.

### Fast-dLLM v2: conversion training and two levels of caching

[Fast-dLLM v2](https://arxiv.org/html/2509.26328v1), released in September 2025 and subsequently included in ICLR 2026, converts Qwen2.5 weights and combines block caching, sub-block reuse and parallel decoding. Its reported roughly 1B conversion tokens are additional training, not the total cost of obtaining the pretrained model.

Figure 1 reports about 2.54× throughput against Qwen2.5-7B-Instruct on A100 with comparable GSM8K accuracy. Its separate +5.2 percentage-point result is against Fast-dLLM-LLaDA, **not Qwen**. Section 4.2 also reports speedups at batch 64, so diffusion does not necessarily lose at large batches. Gains still depend on cache approximations, quality requirements and implementation; they do not transfer automatically to another serving stack. [Authors' project page](https://nvlabs.github.io/Fast-dLLM/v2/)

### Planned Diffusion: count the plan, and compare quality too

Include planning in latency and check how a faulty plan affects the answer. On AlpacaEval, using the paper's H200 (141 GB) setup and a 2048-token total sequence cap, Appendix B reports 6.272 seconds for AR versus 4.934 seconds for Planned Diffusion, with length-controlled win rate changing from 50% to 49.17%. The sparse version is faster and scores lower. This is not a lossless speedup, and a model-judged benchmark cannot stand in for correctness on every task. [Experimental setup and results](https://arxiv.org/html/2510.18087v2#S5)

### October 2026: revisiting the 16× training-compute claim

[Revisiting the 16× Pretraining Compute Gap](https://arxiv.org/html/2610.17590v1), an October 13, 2026 preprint, separates changes to scoring from changes to the model. Replacing the likelihood bound with exact likelihood under a fixed deterministic unmasking rule changes the fitted multiple from roughly 16× to 12.8× without changing the weights. That part is not an actual reduction in training time.

With encoder–decoder block models, the authors report about 7.8× at block size 32 and 3.0× at size 4. These are fitted estimates within the paper's compute range, still above AR. Runs use a single seed, and smaller blocks also reduce decoding parallelism. The result challenges a universal 16× claim; it does not establish that diffusion is generally cheaper to train than AR.

<a id="fair-test"></a>

## What to compare in a small experiment

Suppose the goal is reliable answers with less waiting. Before building a large system, compare these setups on a fixed test set:

| Setup | What it helps separate |
| --- | --- |
| AR answers directly | Original quality and latency |
| AR plans, then AR writes | Gains from planning alone |
| The same AR plans, then a dLLM writes | Changes attributable to the generation method |
| Human-checked plans, then a dLLM writes | Whether planning is a bottleneck; diagnostic only, not a deployment score |

Keep model family, task, output limits, hardware and precision as comparable as possible; disclose what cannot be controlled. These setups isolate sources of error, but not every checkpoint supports this composition. A trained hybrid also needs its own appropriate AR baseline.

Record quality, full-response latency, time to the first useful span, throughput and peak memory. Test one request and several concurrency levels. Include planning, prefill, denoising, clean-block commits, verification and retries. With block output, an early first token does not necessarily mean the user has a useful passage to read.

Then inspect failures. Was the plan wrong, or did parallel passages disagree? Did approximate caching introduce errors, or did fewer denoising rounds reduce quality? Separating these cases gives a more useful next experiment than a single speedup number.

Next: [caching in diffusion decoding](context-reuse.en.md).
