# Diffusion language models: decoding steps, caching and real speed

[中文](decoding-and-speed.md) · **English**

> Reading time: ~14 min · Draft, pending review

LLaDA 8B Instruct's GSM8K score was measured with one token committed per forward pass: a 512-token answer took 512 steps ([LLaDA, Appendix B.6](https://arxiv.org/abs/2502.09992)). When a third party, with generation length 128 and block size 32, made the same model commit a fixed 4 tokens per step, GSM8K dropped from 73.6 to 61.7 ([Kim et al., 2025-10](https://arxiv.org/abs/2510.04146)). **How much faster a diffusion model can be depends on two numbers: how many tokens it can safely commit per step on average (call it $\tau$), and how many positions each step recomputes (call it $G$). At low concurrency, decode is held back by memory bandwidth while compute sits idle, so speed can rise by up to about $\tau$ times; once concurrency is high enough to use all the compute, each token costs about $G/\tau$ times more computation, and the advantage disappears or even reverses.**

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

This note follows [the masked diffusion objective](masked-diffusion.en.md) and does its accounting with the notation and H100 numbers from [the roofline note](../../12-infra/performance.en.md#decode-batching).

## What one step does: predict every blank, keep the confident ones

LLaDA's sampler (§2.4) starts from an answer that is entirely blank; generation length and number of steps are both hyperparameters. One step from time $t$ to $s$: feed the prompt and the partially masked answer to the model, predict every blank in one forward pass, then mask about $s/t$ of those predictions again. The default "low-confidence remasking" keeps the most confident predictions and masks the rest back; anything after EOS is discarded at the end.

Walk through `the tea is cold`: generation length 4, two steps, two positions kept per step. The confidences are teaching numbers.

| Step | Input | Most likely word and confidence at each blank | Kept | Output |
| --- | --- | --- | --- | --- |
| 1 | `[M] [M] [M] [M]` | the 0.9, tea 0.3, is 0.6, cold 0.4 | the, is | `the [M] is [M]` |
| 2 | `the [M] is [M]` | tea 0.7, cold 0.8 | all | `the tea is cold` |

In step 1 `tea` has only 0.3 and is masked again; in step 2, with `the` and `is` as context, it rises to 0.7. "Remasking" here applies only to this step's new predictions: the committed `the` and `is` never change afterwards. As pseudocode:

```python
def sample(model, prompt, gen_len, steps):
    x = prompt + [MASK] * gen_len
    for k in range(steps):
        probs = model(x)                          # one full bidirectional forward pass
        masked = [i for i in range(len(x)) if x[i] == MASK]
        n_keep = len(masked) // (steps - k)       # spread remaining blanks over remaining steps
        conf = {i: probs[i].max() for i in masked}
        for i in sorted(masked, key=conf.get, reverse=True)[:n_keep]:
            x[i] = probs[i].argmax()              # commit; never revisited
    return x
```

**Can committed tokens be changed?** By default, masked diffusion cannot, and that separates it from the methods below. [Mask-Predict](https://arxiv.org/abs/1904.09324) (Ghazvininejad et al., 2019-04, EMNLP 2019), for machine translation, repeatedly masks and re-predicts the least confident words among all positions. [ReMDM](https://arxiv.org/abs/2503.00307) (Wang et al., 2025-03, NeurIPS 2025) adds a sampler that remasks already decoded tokens of a pretrained masked diffusion model; more steps bring quality closer to autoregressive (AR) models. [LLaDA2.1](https://arxiv.org/abs/2602.08676) (2026-02) adds "change a word into another word" editing on top of "fill a blank with a word". DiffusionGemma's [model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card) says the tokens not selected at each step are fully renoised, and its [developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/) says a token whose confidence drops can be renoised and replaced.

## How many per step: why parallel decoding writes "high house"

The tokens committed together in one step are each chosen from their own marginal distribution, without seeing one another. [Fast-dLLM](https://arxiv.org/abs/2505.22618) (Wu et al., 2025-05) illustrates this with poker hands. Suppose the answer is `full house` half the time and `high card` half the time (a teaching distribution). Position 1's marginal is full 0.5, high 0.5; position 2's is house 0.5, card 0.5. Sampling both positions in one step produces a valid hand with probability only $0.5\times0.5+0.5\times0.5=0.5$; the other half is `full card` or `high house`. In two steps, committing one first fixes the second, and validity returns to 1. The `hot tea` example from the previous note behaves the same way: committing both tokens in one step generates it with probability $0.2\times0.1=0.02$, against 0.105 when the blanks are filled one at a time.

Fast-dLLM decides how many to commit by confidence: every position whose confidence exceeds a threshold (0.9 by default) is committed, and if none does, only the most confident one. Its Theorem 1 says: if $n$ positions all have marginal confidence above $1-\epsilon$ and $(n+1)\epsilon\le1$, then taking each position's maximum and taking the maximum of the joint distribution select the same sequence. With three positions all above 0.95, $(3+1)\times0.05=0.2\le1$, so they can be committed together; in the poker example $\epsilon=0.5$ and $n=2$, $3\times0.5=1.5>1$, so there is no guarantee, and the threshold rule commits only one first. As a result $\tau$ changes with the content: spans that are easy to pin down can be committed many tokens per step, hard places one at a time; DiffusionGemma's model card also says that simple prompts and structured tasks such as code need fewer denoising steps.

The cost of a fixed $\tau$ has been measured by a third party. Kim et al. (2025-10) used LLaDA-8B-Instruct with generation length 128, block size 32 and Fast-dLLM's approximate cache switched on:

| Tokens committed per step (fixed) | 1 | 2 | 4 | 8 | 32 |
| --- | --- | --- | --- | --- | --- |
| Steps | 128 | 64 | 32 | 16 | 4 |
| GSM8K (4-shot) | 73.6 | 70.6 | 61.7 | 29.7 | 6.1 |
| HumanEval (pass@1) | 35.4 | 32.3 | 17.7 | 9.1 | 3.7 |

The 73.6 is the one-token-per-step baseline in this setting. It is not the same setting as LLaDA's self-reported 69.4 (generation length 512, no blocks), so the two cannot be compared directly; what the table compares is the drop from left to right.

Adapting by confidence does much better, but those numbers come from the method's own authors: Fast-dLLM reports LLaDA on GSM8K (5-shot, generation length 256, A100 80GB) going from 79.3 to 78.5 while throughput rises from 6.7 to 54.4 tokens/s, 8.1 times, from caching and parallel decoding combined. Google's model card self-reports that DiffusionGemma generates 15–20 tokens per forward pass, with at most 48 denoising steps. In theory, [Feng et al. (2025-02)](https://arxiv.org/abs/2502.09622) prove that if only perplexity matters, the number of steps need not grow with sequence length, but if the whole sequence must be correct (a chain of reasoning, say), the number of steps must grow linearly with length, and the efficiency advantage over AR models is gone.

## KV cache: why bidirectional attention cannot be cached

In an AR model, the K and V at position $i$ depend only on $x_{\le i}$, which never change again, so the cache is exact (see [the KV cache note](../../00-foundations/deep-dives/kv-cache-and-inference.en.md) and [why a cache must be equivalent](../../00-foundations/deep-dives/language-model-objective.en.md#the-correctness-floor-a-cache-is-not-fine-as-long-as-it-runs-it-must-be-equivalent)). Under bidirectional attention, from layer 2 onward every position's hidden state has read every position, including the blanks that change at every step, so even the prompt's K and V change. LLaDA §2.2 states outright that it is incompatible with KV caching; every step recomputes the prompt and the entire answer.

There are two fixes:

- **Approximate caching.** Fast-dLLM observes that the KV at adjacent steps within a block is highly similar, so within a block it reuses the KV of the prefix (PrefixCache) or also the suffix (DualCache), refreshing everything after each block. It is an approximation; the authors report accuracy within 1–2 points of the baseline.
- **Change the structure: block diffusion.** Block Diffusion (BD3-LM) runs left to right across blocks and diffusion within a block; earlier blocks never attend to later ones, so the KV of finished blocks can be cached exactly. Block size 1 falls back to AR decoding. On OpenWebText, the smaller the block, the closer perplexity gets to AR: AR 17.54; block sizes 4, 8 and 16 give ≤20.73, ≤21.68 and ≤22.27; whole-sequence MDLM ≤22.98. LLaDA2.0 is evaluated with block size 32; DiffusionGemma denoises one 256-token block at a time, and each finished block is written to the KV cache with one causal-attention forward pass ([developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/)). SGLang's [blog post](https://lmsys.org/blog/2025-12-19-diffusion-llm) of 2025-12-19 says it implements block diffusion on its existing chunked-prefill path.

Blocks also happen to improve quality: Kim et al. measured GSM8K at 63.2 without blocks, 73.6 with block size 32, and 75.7 with 16 (generation length 128). The cost is that blocks are serial again, and the smaller the block, the more it resembles AR decoding.

## The accounting by hand (a teaching model): time on the same GPU

**This section is a teaching model, not a measurement.** It uses only three numbers from [the roofline note](../../12-infra/performance.en.md#decode-batching): a 7B model's BF16 weights take about $1.4\times10^{10}$ bytes, the H100 SXM5 has 3.352 TB/s of bandwidth, and its BF16 dense peak is 989.4 TFLOPS. Reading the weights once takes at least $1.4\times10^{10}/3.352\times10^{12}=4.18$ ms; the compute for one position is about $2N=1.4\times10^{10}$ FLOPs, or 0.01415 ms at peak. Ignoring attention compute, KV reads and all overheads, the lower bound for one forward pass over $n$ positions is

$$
T_{\text{step}}(n)=\max\big(4.18\ \text{ms},\;n\times0.01415\ \text{ms}\big)
$$

Below about 295 positions, reading the weights dominates and extra positions are nearly free; above 295, compute dominates. Below is a teaching estimate for one request with a 512-token prompt and 256 generated tokens:

| Scheme | Positions per step | Lower bound per step | Steps | Total time | Relative to AR |
| --- | --- | --- | --- | --- | --- |
| AR + KV cache | 1 (plus one 512-token prefill, 7.24 ms) | 4.18 ms | 256 | 1,076 ms | 1× |
| Whole-sequence diffusion, no cache, 1 per step | 768 | 10.87 ms | 256 | 2,782 ms | 0.39× |
| Same, 4 per step | 768 | 10.87 ms | 64 | 696 ms | 1.55× |
| Blocks $G=32$ + prefix cache, 1 per step | 32 | 4.18 ms | 256 | 1,076 ms | 1.0× |
| Same, 4 per step | 32 | 4.18 ms | 64 | 275 ms | 3.9× |

Without a cache, every step recomputes 768 positions, past the 295 mark, so a single step takes 2.6 times as long as an AR step; at one token per step, the total time is also 2.6 times AR's. With blocks and a cache, each step computes only 32 positions, still in the shadow of the weight read, so committing 4 per step is almost exactly 4 times faster. If each finished block gets one extra forward pass to write the cache (as DiffusionGemma does), 8 blocks add $8\times4.18=33$ ms, for 308 ms in total, about 3.5 times faster.

Now many concurrent requests (still the teaching model, decode phase only, block diffusion with $G=32$, $\tau=4$):

| Concurrent requests | AR positions per step | AR throughput bound | Diffusion positions per step | Diffusion throughput bound |
| --- | --- | --- | --- | --- |
| 1 | 1 | 239 tok/s | 32 | 958 tok/s |
| 8 | 8 | 1,915 tok/s | 256 | 7,662 tok/s |
| 64 | 64 | 15,323 tok/s | 2,048 | 8,834 tok/s |
| 295 | 295 | 70,631 tok/s | 9,440 | 8,834 tok/s |

Diffusion uses up the compute at about 9 concurrent requests ($9\times32=288$ positions) and caps at 8,834 tok/s from then on, exactly 1/8 of AR's cap of 70,671: each committed token costs $G/\tau=8$ positions of compute. AR catches up at about 37 concurrent requests and is faster beyond that. Both tables can be recomputed with a few lines:

```python
W = 1.4e10 / 3.352e12 * 1e3      # reading the weights once: 4.18 ms
C = 1.4e10 / 989.4e12 * 1e3      # compute per position: 0.01415 ms
step = lambda n: max(W, n * C)   # lower bound for one forward pass over n positions
prefill = step(512)              # 7.24 ms
print(prefill + 256 * step(1), 256 * step(768), 64 * step(768))           # 1076 2782 696
print(prefill + 64 * step(32), prefill + 72 * step(32))                   # 275 308 (the latter with 8 cache writes)
for b in (1, 8, 64, 295):
    print(b, round(b / step(b) * 1e3), round(4 * b / step(32 * b) * 1e3))  # AR and diffusion tok/s at concurrency b
```

Putting the two regimes together:

$$
\frac{T_{\mathrm{AR}}}{T_{\mathrm{diff}}}\Big|_{\text{low batch}}\approx\tau,\qquad
\frac{\Theta_{\mathrm{diff}}}{\Theta_{\mathrm{AR}}}\Big|_{\text{compute-bound}}\approx\frac{\tau}{G}
$$

$\Theta$ is throughput. This accounting ignores attention and the KV cache. Once the roofline note's second table includes KV, AR's bound at context 4096 and batch 64 falls to 4,440 tok/s, so the real crossover has to be measured separately. Qualitatively, diffusion also spreads the KV read over $\tau$ tokens (one read per step, $\tau$ commits), while its attention compute also costs $G/\tau$ times more.

**Measurements point the same way.** Kim et al. (A100 80GB, prompt and generation both 128, block size 32) found that LLaMA-3-8B-Instruct's throughput keeps growing with batch size up to 16, while LLaDA stops improving at around 8; with a 2k prompt, batch size 1 and no blocks, LLaDA's latency is higher even for short generations. Fast-dLLM's appendix (A100, prefill 256, batch 1–32) found that at small batch sizes LLaDA with a prefix cache matches or exceeds LLaMA's throughput but falls behind as the batch grows, because LLaMA moves from bandwidth-bound to compute-bound. Google's [launch post](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/) for DiffusionGemma (2026-06-10) says the same: the advantage is largest on a single accelerator at low to medium batch sizes, returns diminish in high-QPS cloud serving and serving costs may rise, and unified-memory devices such as Apple Silicon may not see the same acceleration.

## How to read vendor numbers

All of the following are self-reported by the companies or the method's authors, under different conditions, and cannot be compared with each other:

| Date | System | Self-reported number | Not stated |
| --- | --- | --- | --- |
| 2025-05-20 | [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/) | Average sampling speed across evaluations 1479 tokens/s, excluding a fixed 0.84 s overhead | Model size, hardware, batch |
| 2025-06 | [Mercury Coder](https://arxiv.org/abs/2506.17298) Mini / Small | 1109 / 737 tokens/s on H100, which the report attributes to an independent evaluation of Inception's API by Artificial Analysis | Model size, batch; the inference engine is described only as in-house |
| 2025-12 | [LLaDA2.0](https://arxiv.org/abs/2512.15745)-flash-CAP | 535 TPS against 256 and 237 TPS for two AR baselines, "up to 2.1x" | Hardware and batch; diffusion runs on dInfer, the baselines on SGLang |
| 2026-02-24 | [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2) | 1009 tokens/s on Blackwell GPUs, "more than 5x faster" | Baseline, workload, who measured |
| 2026-06-10 | DiffusionGemma 26B A4B | [Launch post](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/): up to 4x faster, 1000+ tokens/s on a single H100; [model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card): 1100+ at low batch on H100 with FP8 | Baseline and batch for the 4x |
| 2026-09-08 | [Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) | 1107 tokens/s on "widely-available NVIDIA GPUs" | GPU model, benchmark scores, parameter count |

A few concrete traps. In the Mercury report, the comparison models' speeds are Artificial Analysis's median measurements across each model's cloud providers, so hardware, engines and model sizes all differ; the workload is coding prompts of about 1,000 input and 1,000 output tokens. The LLaDA2.0 paper does not say which of 256 and 237 belongs to Ling-flash-2.0 and which to Qwen3-30B-A3B, and the engines differ; the SGLang blog puts both sides on SGLang (H20, TP8, small batches), where the same CAP model reaches 500 TPS against 258 and 237, about 1.9 times. LLaDA's own authors report in Appendix B.7 that, on an A100 at batch 1 with 256 output tokens, the base model reaches comparable accuracy on GSM8K and MATH at 1.5 and 1.8 times the throughput of LLaMA3 8B with a KV cache. Quality has to be read alongside: in Gemini Diffusion's own table, GPQA Diamond is 40.4% against 56.5% for Gemini 2.0 Flash-Lite, and HumanEval 89.6% against 90.2%.

When you read "N times faster", ask four things first: at what concurrency; is the baseline on the same GPU and the same engine; what is the quality at that speed, with which threshold; and does the time include prefill, the per-block commit pass and fixed overheads.

## The relation to DFlash: diffusion only writes the draft

[DFlash](../../00-foundations/deep-dives/dflash.en.md) (2026-02, ICML 2026) takes a different route: a small block-diffusion model reads the target model's features and writes a block of draft tokens in one forward pass; the AR target model verifies it and keeps only the contiguously accepted prefix. The target model's output distribution is unchanged, so the quality risk does not sit on the diffusion side. The authors self-report lossless speedups above 6 times across models and tasks, and up to 2.5 times the speedup of EAGLE-3. Its accounting has the same shape as above: drafting and verification both compute a whole block, and the more is accepted, the better the trade.

| Route | Who decides the final token | Quality risk | Where the speed comes from | At high concurrency |
| --- | --- | --- | --- | --- |
| Native diffusion LM (LLaDA, DiffusionGemma) | The diffusion model itself | Tokens committed in parallel can be inconsistent; lower quality than the same family's AR model | $\tau$ tokens per step | Each token costs about $G/\tau$ times more compute |
| Diffusion draft + AR verification (DFlash) | The AR target model | The target model's distribution is unchanged | Several draft tokens accepted per round | Drafting and verification both use compute; the gain shrinks too |
| Plain autoregressive | The AR model | No extra risk | No parallelism | Throughput grows with batch until compute saturates |

## What to check when you try it

| Try first | What to observe | What you can NOT conclude |
| --- | --- | --- |
| Fix a quality target, sweep the threshold or step count, and plot quality against average $\tau$ | Where quality starts to drop as $\tau$ grows | The speed at one point represents the model |
| On one GPU and one engine, sweep concurrency from 1 to 64 | Where the AR and diffusion throughput curves cross | A batch-1 speedup represents the gain for a whole server |
| Compare per-question accuracy with the approximate cache on and off | Whether the gap stays within 1–2 points | KV similarity close to 1 means the results are equivalent |
| Include prefill, the per-block commit pass and fixed overheads in the timing | End-to-end latency and time to first token | "Sampling speed" alone tells you how long a user waits |
| Record the distribution of tokens committed per step | How much $\tau$ differs between hard and easy spans | The average $\tau$ represents every question |

Next: [review questions](review.en.md).
