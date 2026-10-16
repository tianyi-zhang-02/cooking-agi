# Diffusion language models: review questions

[中文](review.md) · **English**

> Reading time: ~8 min · Draft, pending review

This page condenses the series into interview questions and a self-check; every number and conclusion in the answers can be traced to a source in the first two notes.

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

## Interview questions

<details class="interview" markdown="1">
<summary>What are the forward process and the training objective of masked diffusion? What does the $1/t$ do?</summary>

Forward process: at time $t\in[0,1]$, each position is independently replaced by `[M]` with probability $t$, and once it becomes `[M]` it never changes back (the absorbing state). Training objective: draw $t\sim U(0,1]$ and the corresponding $x_t$, compute the cross-entropy only at masked positions, and multiply by $1/t$ (LLaDA, Eq. 3). At small $t$ only $tL$ positions contribute on average; after the $1/t$, every $t$ carries an expected weight of $L$ tokens, and the whole objective becomes an upper bound on the negative log-likelihood (NLL). With another masking schedule, the continuous-time weight is $-\alpha_t'/(1-\alpha_t)$; the objective itself does not change, only the gradient variance. The model does not need to know $t$, since the number of blanks is already in the input.

</details>

<details class="interview" markdown="1">
<summary>How does it differ from BERT's masked language modeling (MLM)? Why can't BERT be used to generate directly?</summary>

The network can be the same bidirectional Transformer; the difference lies in the distribution of masking ratios and the weighting. BERT masks a fixed ~15% and averages over selected positions; it never learns to start from all blanks and has no likelihood interpretation. Masked diffusion draws a $t$ per example, so it practises everything from almost no masking to full masking, and the $1/t$ ties it to maximum likelihood; MDLM writes it as a weighted average of MLM losses. LLaDA also notes that MaskGIT drops the $1/t$ and so loses the link to maximum likelihood.

</details>

<details class="interview" markdown="1">
<summary>What does "any order" mean? Use the two-token example to show what the objective equals.</summary>

In training, the objective averages over all generation orders; at inference, blanks can be filled in any order. With two tokens: both are masked with probability $t^2$ and exactly one with probability $t(1-t)$ each; after multiplying by $1/t$ and integrating, both terms get weight $1/2$, so the objective is the average of the left-to-right and right-to-left autoregressive (AR) losses. With the teaching numbers that is $(1.833+2.996)/2=2.414$; the negative log-likelihood of the model actually generating the sequence is $-\ln 0.105=2.254$, so it is an upper bound, loose when the two orders disagree. At any length, RADD's Theorem 2 proves equivalence with the any-order autoregressive objective. The price is that hard orders must be learned too: the same validation loss takes about 16 times the compute of autoregressive models (Nie et al., 2024-10).

</details>

<details class="interview" markdown="1">
<summary>Why does committing several tokens in one step go wrong? How do you decide how many to commit?</summary>

The tokens in one step are each chosen from their marginal distributions without seeing one another. If the answer is `full house` half the time and `high card` the other half, committing both in one step gives a valid hand only half the time. Fast-dLLM decides by confidence: commit every position above a threshold (0.9 by default), or the single most confident one if none passes; its theorem gives a condition: if $n$ positions all have confidence above $1-\epsilon$ and $(n+1)\epsilon\le1$, taking per-position maxima agrees with taking the joint maximum. The cost of a fixed step count has been measured by a third party: with LLaDA-8B-Instruct at generation length 128 and block size 32, a fixed 4 per step drops GSM8K from 73.6 to 61.7 (73.6 being the one-per-step baseline in the same setting). In theory (Feng et al.), getting a whole reasoning chain right needs the step count to grow linearly with length.

</details>

<details class="interview" markdown="1">
<summary>Why can't a diffusion LM use a KV cache directly? What are the two fixes, and what does each cost?</summary>

In an autoregressive model, the K and V at position $i$ depend only on $x_{\le i}$, which never change. Under bidirectional attention, from layer 2 onward every position has read the blanks that change at every step, so even the prompt's K and V change, and LLaDA recomputes the whole span at every step. Fix one is approximate caching (Fast-dLLM): KV is very similar between adjacent steps, so reuse it within a block and refresh after the block, which requires checking accuracy separately. Fix two is block diffusion: left to right across blocks, diffusion within a block, so the KV of finished blocks can be cached exactly; smaller blocks are closer to autoregressive and give better perplexity, but less parallelism.

</details>

<details class="interview" markdown="1">
<summary>Why isn't parallel decoding automatically faster? Estimate low and high concurrency with the roofline.</summary>

With the second note's teaching model (linear layers only, numbers from the roofline note: 7B, BF16, H100 SXM5 peak), the lower bound for one forward pass is $\max(4.18\,\text{ms},\ n\times0.01415\,\text{ms})$; below about 295 positions, the weight read dominates. For one request with a 512-token prompt and 256 generated tokens: AR takes about 1,076 ms; whole-sequence diffusion without a cache computes 768 positions per step and takes 2,782 ms at one token per step, slower than AR; blocks with $G=32$ plus a cache, at 4 per step, take about 275 ms, 3.9 times faster. As concurrency rises, diffusion computes $G/\tau=8$ positions per committed token and caps at 8,834 tok/s against AR's 70,671 tok/s, so AR catches up at about 37 concurrent requests. The low-concurrency speedup is therefore about $\tau$, and the throughput ratio at compute saturation is about $\tau/G$. Third-party measurements and Google's own notes on DiffusionGemma point the same way.

</details>

<details class="interview" markdown="1">
<summary>You read "a diffusion model at 1000+ tokens/s, N times faster than autoregressive". What do you ask?</summary>

Four things: at what concurrency; is the baseline on the same GPU and the same engine; what is the quality at that speed, and with which threshold; and does the timing include prefill, the per-block commit pass and fixed overheads. Examples: Gemini Diffusion's 1479 tokens/s excludes a 0.84 s overhead; the comparison speeds in the Mercury report are Artificial Analysis's median measurements across each model's cloud providers; in the LLaDA2.0 paper the diffusion models and baselines run on different engines (dInfer versus SGLang), and when the SGLang blog puts both on one engine the ratio falls from about 2.1 to about 1.9; Mercury 2.5 does not name the GPU model. Quality has to be read at the same time: on the DiffusionGemma model card, most benchmarks are below the same-size Gemma 4. As of October 2026, all of these figures come from the vendors or the methods' authors.

</details>

<details class="interview" markdown="1">
<summary>How does DFlash differ from a native diffusion LM?</summary>

DFlash uses block diffusion only to draft: a small model reads the target model's features, writes a block in one forward pass, and the autoregressive target model verifies it and keeps only the contiguously accepted prefix, so the output distribution equals the target model's and the quality risk is not on the diffusion side. A native diffusion LM decides the final tokens itself and bears the risk of inconsistent parallel commits and lower same-family quality. The speed accounting has the same shape for both: each forward pass computes a whole block, the more tokens accepted (or committed) the better the trade, and the gain shrinks at high concurrency.

</details>

## Self-check

<div class="taste-check">
  <strong>If you really understand it, you should be able to explain:</strong>
  <ol>
    <li>What does the two-token objective become without the $1/t$, and why does it lose its likelihood interpretation?</li>
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
