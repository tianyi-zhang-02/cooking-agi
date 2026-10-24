# Diffusion language models: review questions

[中文](review.md) · **English**

> Reading time: ~8 min · Last reviewed: 2026-10

This page collects interview questions and self-checks for the series; every number and conclusion in the answers can be traced to a source in the preceding notes.

## Interview questions

<details class="interview" markdown="1">
<summary>What are the forward process and the training objective of masked diffusion? What does the $1/t$ do?</summary>

Forward process: at time $t\in[0,1]$, each position is independently replaced by `[M]` with probability $t$, and once it becomes `[M]` it never changes back (the absorbing state). Training objective: draw $t\sim U(0,1]$ and the corresponding $x_t$, compute the cross-entropy only at masked positions, and multiply by $1/t$ (LLaDA, Eq. 3). At small $t$, only $tL$ positions contribute on average. Weighting by $1/t$ gives expected supervision weight $L$ at each time, though the loss still depends on prediction difficulty. The complete expected objective upper-bounds negative log-likelihood (NLL); an individual random mask need not give a bound.

For the same ideal denoising conditional, a smooth monotone schedule covering the same endpoints leaves the continuous-time objective unchanged when paired with its corresponding weight $-\alpha_t'/(1-\alpha_t)$. Sampling variance and practical optimization can still differ. Under independent, content-independent masking, the ideal word distribution given the visible context does not depend on $t$, so the prediction network can omit time. The sampler still needs it to determine reveal probabilities. See the [MDLM worked example](mdlm-walkthrough.en.md).

</details>

<details class="interview" markdown="1">
<summary>How does it differ from BERT's masked language modeling (MLM)? Why can't BERT be used to generate directly?</summary>

Both can use bidirectional Transformers; the training problem and generative process differ. BERT usually selects about 15% of positions as prediction targets, only some of which are replaced with `[MASK]`. Its original objective does not also specify a reverse process that generates from all masks. Masked diffusion samples different masking levels and weights losses by $1/t$. MDLM writes this as a weighted average of MLM losses and derives a likelihood lower bound for the corresponding reverse process.

Avoid reducing this to “BERT has no likelihood interpretation.” MLM can provide [pseudo-likelihood scores](https://arxiv.org/abs/1910.14659), but those do not directly give a normalized joint sequence probability. Similarly, removing $1/t$ changes this objective, so the same bound derivation no longer applies. That does not mean an unweighted masked model cannot be given a generative process. See the [two-token derivation](masked-diffusion.en.md) and the [MDLM training–sampling walkthrough](mdlm-walkthrough.en.md).

</details>

<details class="interview" markdown="1">
<summary>What does "any order" mean? Use the two-token example to show what the objective equals.</summary>

In training, the objective averages over all generation orders; at inference, blanks can be filled in different orders. With two tokens, both are masked with probability $t^2$, and each of the two single-mask cases has probability $t(1-t)$. After multiplying by $1/t$ and integrating, both terms get weight $1/2$. The objective is the average of the left-to-right and right-to-left autoregressive (AR) losses.

Using the earlier teaching example, the objective is $(1.833+2.996)/2=2.414$. Choosing randomly between those two orders gives a sequence negative log-likelihood of $-\ln 0.105=2.254$. The first quantity is an upper bound; it is loose when the two orders assign different probabilities.

At any length, RADD's Theorem 2 proves equivalence with the any-order autoregressive objective. Hard orders must be learned too, but that does not imply a fixed compute multiple: the roughly 16 times reported by Nie et al. (2024-10) belongs to their scoring and training setup. [Later comparisons](ar-and-diffusion.en.md) need to distinguish scoring, architecture and data conditions.

</details>

<details class="interview" markdown="1">
<summary>Why does committing several tokens in one step go wrong? How do you decide how many to commit?</summary>

The tokens in one step are each chosen from their marginal distributions without seeing one another. If the answer is `full house` half the time and `high card` the other half, committing both in one step gives a valid hand only half the time. Fast-dLLM decides by confidence: commit every position above a threshold (0.9 by default), or the single most confident one if none passes; its theorem gives a condition: if $n$ positions all have confidence above $1-\epsilon$ and $(n+1)\epsilon\le1$, taking per-position maxima agrees with taking the joint maximum. The cost of a fixed step count has been measured by a third party: with LLaDA-8B-Instruct at generation length 128 and block size 32, a fixed 4 per step drops GSM8K from 73.6 to 61.7 (73.6 being the one-per-step baseline in the same setting). Feng et al.'s Theorem 4.4 gives a linear-step lower bound for a worst-case HMM with a 16-token vocabulary, under their factorized reverse process and small learning-error assumptions. Its sequence error measures probability outside the HMM's support; it is not a universal limit on reasoning speedups. Compare latency at matched task accuracy.

</details>

<details class="interview" markdown="1">
<summary>Why can't a diffusion LM use a KV cache directly? What are the two fixes, and what does each cost?</summary>

In an autoregressive model, the K and V at position $i$ depend only on $x_{\le i}$, which stays fixed. With bidirectional attention, a mask read by the previous layer may become a word in the next round. From layer 2 onward, even the prompt's K and V can therefore change. Original LLaDA recomputes the full span each round; AR's exact caching rule does not carry over directly.

One option is approximate caching, as in Fast-dLLM: exploit similarity between nearby steps, reuse KV within a block, refresh periodically, and check the quality loss. Another is block diffusion: generate blocks left to right and denoise within each block. Block-causal attention isolates completed prefixes from updates in the active block, making their KV exactly reusable. A completed block usually still needs a pass over its final tokens to create the clean representations to cache.

Smaller blocks move closer to AR and reduce the number of positions available for parallel generation. The Block Diffusion paper reports lower perplexity bounds for smaller blocks under its training setup, not a guarantee that shrinking blocks improves every model. See the [block-cache example](ar-and-diffusion.en.md#block-cache); reusing attention outputs is a [separate question](context-reuse.en.md).

</details>

<details class="interview" markdown="1">
<summary>Why isn't parallel decoding automatically faster? Estimate low and high concurrency with the roofline.</summary>

Use the teaching model in [decoding and speed](decoding-and-speed.en.md): linear layers only, with 7B parameters, BF16, and H100 SXM5 peak specifications. The lower bound for one forward pass is $\max(4.18\,\text{ms},\ n\times0.01415\,\text{ms})$; below about 295 positions, reading the weights dominates.

For one request with a 512-token prompt and 256 generated tokens: AR takes about 1,076 ms; whole-sequence diffusion without a cache computes 768 positions per step and takes 2,782 ms at one token per step, slower than AR; blocks with $G=32$ plus a cache, at 4 per step, take about 275 ms, 3.9 times faster. Adding a commit pass after each of the 8 blocks costs about 33 ms more: about 308 ms, or roughly 3.5 times faster.

As concurrency rises, diffusion computes $G/\tau=8$ positions per committed token. This simplified model gives a throughput ceiling of 8,834 tok/s, versus AR's 70,671 tok/s, with AR catching up at about 37 concurrent requests. The low-concurrency speedup is therefore about $\tau$, and the throughput ratio at compute saturation is about $\tau/G$. These are estimates, not measurements. Third-party measurements and Google's discussion of DiffusionGemma also highlight the difference between low-concurrency latency and high-concurrency throughput.

</details>

<details class="interview" markdown="1">
<summary>You read "a diffusion model at 1000+ tokens/s, N times faster than autoregressive". What do you ask?</summary>

Four things: at what concurrency; is the baseline on the same GPU and the same engine; what is the quality at that speed, and with which threshold; and does the timing include prefill, the per-block commit pass and fixed overheads. Examples: Gemini Diffusion's 1479 tokens/s excludes a 0.84 s overhead; the comparison speeds in the Mercury report are Artificial Analysis's median measurements across each model's cloud providers; in the LLaDA2.0 paper the diffusion models and baselines run on different engines (dInfer versus SGLang), and when the SGLang blog puts both on one engine the ratio falls from about 2.1 to about 1.9; Mercury 2.5 does not name the GPU model. Quality has to be read at the same time: on the DiffusionGemma model card, most benchmarks are below the same-size Gemma 4. As of October 2026, all of these figures come from the vendors or the methods' authors.

</details>

<details class="interview" markdown="1">
<summary>How does DFlash differ from a native diffusion LM?</summary>

DFlash uses block diffusion to draft: a small model reads the target model's features and predicts a block in one forward pass. The AR target model then verifies it, retaining the consecutively accepted prefix. The draft does not directly determine the final output. [DFlash paper](https://arxiv.org/html/2602.06036v1#S4)

Preserving the target distribution requires correct exact verification and subsequent sampling. Greedy decoding must preserve the target's choices; stochastic sampling needs the corresponding acceptance, rejection, and correction rules. Keeping high-confidence draft tokens alone is not enough. Under these conditions, poor drafts mainly reduce acceptance length and speed rather than change the target distribution. [Original speculative-decoding derivation](https://proceedings.mlr.press/v202/leviathan23a.html)

A native diffusion LM decides its own final tokens. Its parallel-sampling consistency and task quality need direct evaluation; it cannot inherit another target model's quality guarantee. Both approaches must account for drafting or denoising and verification or commit costs. More tokens per cycle usually help amortize those costs, but extra computation and scheduling at high concurrency can change the result. A single-request speedup is not a serving-throughput guarantee.

</details>

## Self-check

<div class="taste-check">
  <strong>Use these questions to check your understanding:</strong>
  <ol>
    <li>What does the two-token objective become without $1/t$, and why can't the original likelihood-bound derivation be reused directly?</li>
    <li>At how many tokens per step was LLaDA's GSM8K score measured? What does that imply for claims of "both good and fast"?</li>
    <li>In the poker example, why is only half of the one-step outputs a valid hand? How does Fast-dLLM's threshold rule handle it?</li>
    <li>Without a cache, why is whole-sequence diffusion slower than AR even at one token per step? With blocks and a cache, what changes in the accounting?</li>
    <li>In the compute-saturated regime, why is the diffusion-to-AR throughput ratio about $\tau/G$? Given a $\tau$ and a $G$, compute the crossover once.</li>
  </ol>
</div>

## References

- [Mask-Predict](https://arxiv.org/abs/1904.09324): parallel prediction and iterative remasking for translation
- [D3PM](https://arxiv.org/abs/2107.03006): discrete diffusion and the absorbing state
- [Autoregressive Diffusion Models](https://arxiv.org/abs/2110.02037): unifying any-order autoregressive models and absorbing diffusion
- [Diffusion-LM](https://arxiv.org/abs/2205.14217): continuous diffusion over word vectors
- [SEDD](https://arxiv.org/abs/2310.16834): score-entropy discrete diffusion
- [MDLM](https://arxiv.org/abs/2406.07524): a simplified objective, a weighted average of MLM losses
- [RADD](https://arxiv.org/abs/2406.03736): dropping time conditioning; the equivalence theorem with any-order autoregressive models
- [Masked Diffusion Models are Secretly Time-Agnostic Masked Models](https://arxiv.org/abs/2409.02908): precision of categorical sampling and generative perplexity
- [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514): scaling laws and the roughly 16-times compute gap
- [Train for the Worst, Plan for the Best](https://arxiv.org/abs/2502.06768): hard subproblems and adaptive decoding order
- [Theoretical Benefit and Limitation of Diffusion Language Model](https://arxiv.org/abs/2502.09622): step counts and sequence-level correctness
- [LLaDA](https://arxiv.org/abs/2502.09992): 8B trained from scratch, sampling, and the self-reported scalability
- [ReMDM](https://arxiv.org/abs/2503.00307): a sampler that remasks decoded tokens
- [Block Diffusion](https://arxiv.org/abs/2503.09573): block diffusion and an exact KV cache
- [d1](https://arxiv.org/abs/2504.12216): RL for masked diffusion LMs (diffu-GRPO)
- [Fast-dLLM](https://arxiv.org/abs/2505.22618): approximate KV cache and confidence-based parallel decoding
- [Mercury](https://arxiv.org/abs/2506.17298): Inception's technical report (speed figures self-reported)
- [Diffusion Beats Autoregressive in Data-Constrained Settings](https://arxiv.org/abs/2507.15857): the comparison when data is repeated over many epochs
- [Seed Diffusion](https://arxiv.org/abs/2508.02193), [Dream 7B](https://arxiv.org/abs/2508.15487): large diffusion LMs of 2025
- [Beyond Next-Token Prediction](https://arxiv.org/abs/2510.04146): a third-party performance characterization of LLaDA against LLaMA-3
- [LLaDA2.0](https://arxiv.org/abs/2512.15745), [LLaDA2.1](https://arxiv.org/abs/2602.08676): 100B MoE models converted from autoregressive models, and token editing
- [DFlash](https://arxiv.org/abs/2602.06036): a block-diffusion draft model with AR verification
- [Scaling Beyond Masked Diffusion Language Models](https://arxiv.org/abs/2602.15014): scaling laws for uniform-state diffusion
- [Neither Parallel Nor Sequential](https://arxiv.org/abs/2606.14620): the order in which DiffusionGemma commits tokens
- [Gemini Diffusion announcement](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/) and [model page](https://deepmind.google/models/gemini-diffusion/): the experimental demo and self-reported speed
- [DiffusionGemma launch post](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/), [model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card), [developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/): architecture, sampler, self-reported speed and quality
- [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2), [Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5): Inception's self-reported figures for 2026
- [SGLang's dLLM support](https://lmsys.org/blog/2025-12-19-diffusion-llm) and [documentation](https://lmsysorg.mintlify.app/docs/supported-models/diffusion_language_models): block diffusion on top of chunked prefill; FDFO scheduling
