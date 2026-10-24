# Diffusion language models: denoising a whole block instead of left to right

[中文](README.md) · **English**

> Reading time: ~8 min · Last reviewed: 2026-10

Autoregressive (AR) generation usually predicts the next token from an existing prefix. A diffusion language model (dLLM) instead starts from a noisy state and progressively recovers text.

This series mainly covers masked diffusion: start with blanks, predict missing words, and commit some according to a sampling rule.

Both can use Transformers. The differences go beyond how many words appear per step: what the training target is, what context generation can see, and which computations can be reused. Start with two ways to generate the same sentence, then work through those differences.

Diffusion language models include Google's Gemini Diffusion demo from May 2025, Inception's Mercury series, and DiffusionGemma, released with open weights in June 2026. Their noise processes and decoding methods differ, and some details remain unpublished. This series starts with masked diffusion methods whose details are public.

The masked diffusion objective can be written as an upper bound on negative log-likelihood (NLL). LLaDA's authors also report that their 8B model, trained from scratch, is comparable to similarly sized AR models on many benchmarks. But parallel prediction does not guarantee faster decoding: training covers different generation orders, decoding repeatedly processes a whole block, and committing too many tokens together can introduce inconsistencies. Vendor comparisons within model families have also reported quality gaps. We examine these trade-offs and why speedups often appear at low concurrency, when memory bandwidth is the bottleneck.

## Two ways to write the same sentence

```text
Autoregressive: one token per step, looking only left
the → the tea → the tea is → the tea is cold

Masked diffusion: predict every blank, reveal some according to the sampler
[M] [M] [M] [M] → the [M] is [M] → the tea is cold
```

| | Autoregressive | Masked diffusion (whole sequence) | Block diffusion |
| --- | --- | --- | --- |
| What a step does | Predicts the next token from the left | Reads the whole span bidirectionally and predicts every blank | Left to right across blocks; within a block, like whole-sequence diffusion |
| Tokens committed per step | 1 with ordinary decoding | Set by the step budget or adaptive sampling rule | One or more within the active block |
| KV cache | Processed prefix can be reused exactly | Changing context prevents direct reuse of the whole old cache | Finished blocks can be reused exactly; the active block needs separate treatment |
| Output length | Until EOS | Fixed in advance | Extends block by block |
| Examples | GPT, LLaMA | LLaDA 8B | BD3-LM, LLaDA2.0, DiffusionGemma |

The derivations start with masked diffusion, where adding noise means replacing tokens with `[M]`. This is one type of discrete diffusion. DiffusionGemma also generates in blocks, but Google describes its noise as random token replacement, or uniform-state diffusion. The masked-diffusion formulas therefore do not all apply unchanged. See [where noise is added](masked-diffusion.en.md) for the distinction.

Continuous image diffusion commonly starts from Gaussian noise; see [flow matching in the Qwen-Image note](../../03-multimodal-learning/image-generation.en.md). Text models can also diffuse over continuous word vectors. The main large models discussed here—LLaDA, Dream, LLaDA2.0, and DiffusionGemma—model discrete tokens instead.

Parallel prediction and committing output are also different. A model may predict many positions at once without the sampler keeping every prediction. Even committing several tokens does not necessarily save computation: the step may still reprocess the whole block.

## Timeline

| Date | Event | In one sentence |
| --- | --- | --- |
| 2019-04 | [Mask-Predict](https://arxiv.org/abs/1904.09324) | For translation, predict all target words in parallel, then repeatedly mask and refill the least confident ones (EMNLP 2019) |
| 2021-07 | [D3PM](https://arxiv.org/abs/2107.03006) | A discrete diffusion framework, including the absorbing state: once `[M]`, never back |
| 2021-10 | [ARDM](https://arxiv.org/abs/2110.02037) | Any-order autoregressive models and absorbing diffusion are special cases of one model family (ICLR 2022) |
| 2023-10 | [SEDD](https://arxiv.org/abs/2310.16834) | Score entropy; perplexity 25–75% lower than earlier diffusion LMs, better than GPT-2 (ICML 2024) |
| 2024-06 | [MDLM](https://arxiv.org/abs/2406.07524), [RADD](https://arxiv.org/abs/2406.03736) | The objective simplifies to a weighted average of masked language modeling (MLM) losses; the model need not know the time |
| 2024-10 | [SMDM scaling law](https://arxiv.org/abs/2410.18514) | About 16 times the AR compute at equal validation loss in this experiment, not a constant across diffusion models |
| 2025-02 | [LLaDA](https://arxiv.org/abs/2502.09992) | 8B trained from scratch on 2.3T tokens; the authors report scaling comparable to same-data AR baselines |
| 2025-03 | [Block Diffusion](https://arxiv.org/abs/2503.09573) | Autoregressive across blocks, diffusion within; finished blocks can be cached exactly (ICLR 2025) |
| 2025-05-20 | [Gemini Diffusion](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/) | Experimental demo behind a waitlist; the model page self-reports an average of 1479 tokens/s, excluding a fixed overhead |
| 2025-05 | [Fast-dLLM](https://arxiv.org/abs/2505.22618) | Approximate KV cache plus confidence-based parallel commits; self-reports up to 27.6 times LLaDA's throughput (generation length 1024) |
| 2025-06 | [Mercury technical report](https://arxiv.org/abs/2506.17298) | Mercury Coder Mini / Small at 1109 / 737 tokens/s on H100; the report says the figures come from independent evaluations by Artificial Analysis |
| 2025-09 | [Fast-dLLM v2](https://arxiv.org/abs/2509.26328) | Converts AR weights to block diffusion with block and sub-block caching; [comparison conditions](ar-and-diffusion.en.md) |
| 2025-10; revised 2026-03 | [Planned Diffusion](https://arxiv.org/abs/2510.18087v2) | One model plans autoregressively, then expands content with diffusion; planning is included in latency |
| 2025-10 | [Performance characterization](https://arxiv.org/abs/2510.04146) | A third party measures LLaDA against LLaMA-3: at large batch sizes, AR throughput is higher |
| 2025-12 | [LLaDA2.0](https://arxiv.org/abs/2512.15745) | 16B and 100B MoE models converted from AR models; SGLang support from 2025-12-19 |
| 2026-02 | [LLaDA2.1](https://arxiv.org/abs/2602.08676); [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2) (02-24) | LLaDA2.1 can edit committed tokens; Mercury 2 self-reports 1009 tokens/s on Blackwell |
| 2026-06-10 | [DiffusionGemma](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/) | Open weights, 26B A4B MoE; self-reports up to 4x faster, with quality below the same-size Gemma 4 |
| 2026-09-08 | [Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) | Self-reports 1107 tokens/s without naming the GPU model |
| 2026-10-13 | [Revisiting the 16× training gap](https://arxiv.org/abs/2610.17590v1) | Preprint: scoring and block architecture affect the multiple; frontier-scale extrapolation is not established |

## What is fairly settled and what is still moving

Each cell corresponds to a dated public record. "Signals to watch" describes the kind of result that would change the assessment, not a prediction that it will happen.

| Fairly settled | Still moving | Signals to watch |
| --- | --- | --- |
| The masked diffusion objective is an upper bound on the negative log-likelihood and equals the any-order autoregressive loss: RADD (2024-06), ARDM (2021-10) | Whether masked or uniform-state diffusion is better: Sahoo et al. (2026-02) found at 1.7B that uniform-state has worse perplexity but better GSM8K; Google's developer guide says DiffusionGemma uses uniform-state diffusion, while the third-party Asaria et al. (2026-06) call it masked diffusion | Larger-scale, same-data comparisons across families; a technical report for DiffusionGemma |
| At equal compute, validation loss trails AR models: about 16 times for SMDM ($6\times10^{18}$–$10^{20}$ FLOPs) | Scaling on downstream tasks: LLaDA self-reports parity with same-data AR models up to $10^{23}$ FLOPs; with data repeated over many epochs, diffusion pulls ahead (Prabhudesai et al., 2025-07) | Comparisons above 10B with the same data, the same compute and public checkpoints |
| Fixed parallel commits can cost quality: Kim et al. (2025-10), at generation length 128 and block size 32, saw GSM8K go 73.6 → 61.7 at 4 per step. Separately, Feng et al. prove a linear-step lower bound for a worst-case HMM under their reverse-process and learning-error assumptions—not for every reasoning task ([scope](decoding-and-speed.en.md)) | Adaptive commits and editing: Fast-dLLM's confidence threshold, LLaDA2.1's edits, DiffusionGemma's self-reported 15–20 tokens per forward pass | Third-party measurements of "average tokens per step at equal quality" |
| Changing bidirectional context generally changes KV too; block-causal attention isolates a completed prefix for exact reuse: [cache example](ar-and-diffusion.en.md) | What block size is right, and how much error approximate caching adds inside the active block | Stable interfaces and public benchmarks in mainstream engines: SGLang (since 2025-12), vLLM (the DiffusionGemma guide gives a `vllm serve` command) |
| At low concurrency, diffusion can use compute left underused by AR decode; Kim et al. measured a shrinking advantage as concurrency grew | This is not universal: Fast-dLLM v2 reports gains at large batches too; [conditions](ar-and-diffusion.en.md) | Third-party comparisons on the same hardware and engine, across several concurrency levels, at equal quality |
| Quality is lower in the vendors' own same-family comparisons: on the DiffusionGemma model card, AIME 2026 is 69.1% against 88.3% for the same-size Gemma 4; Gemini Diffusion's GPQA Diamond is 40.4% against 56.5% for 2.0 Flash-Lite | Whether converting from AR weights closes the gap: LLaDA2.0-flash self-reports an average of 73.18 against 72.15 for the original Ling-flash-2.0 | The gap between converted models and their originals on reasoning and long-context tasks |

"Fairly settled" means there is a primary result that holds in the original paper's setting; it does not mean it holds for other tasks or scales.

The training-cost row retains SMDM's historical result. An October 2026 preprint revisits scoring and architecture; see [AR comparisons and recent progress](ar-and-diffusion.en.md). None of these compute multiples should yet be treated as universal.

## What to watch next

These are open questions, and the kinds of results that would change the current assessment; they are not predictions.

- **Speed comparisons under equal conditions.** As of October 2026, most published speedups are at batch size 1 or self-reported by vendors. Seeing which concurrency range the advantage holds in requires a third party to sweep concurrency on the same hardware and engine at equal quality.
- **The quality gap at frontier scale.** Whether the gap between DiffusionGemma and Gemma 4 can be closed by conversion training and RL: [d1](https://arxiv.org/abs/2504.12216) (2025-04) designed diffu-GRPO for masked diffusion, and LLaDA2.1 (2026-02) added large-scale RL. Watch reasoning and long context in particular.
- **The serving stack.** SGLang's dLLM scheduler uses FDFO by default: a request leaves the batch as soon as its current block is resolved, instead of waiting for slower requests in the same batch ([SGLang documentation](https://lmsysorg.mintlify.app/docs/supported-models/diffusion_language_models), as of October 2026). Its throughput and tail latency at high concurrency, and those of vLLM's support for DiffusionGemma, have not been measured publicly.
- **The advantage when data is limited.** Prabhudesai et al. (2025-07) found diffusion more efficient when data is repeated over many epochs; whether this holds for larger models once high-quality data becomes the bottleneck.
- **Diffusion as a draft model.** DFlash-style "diffusion drafts, AR verifies" leaves the output distribution unchanged; whether it becomes mainstream in deployment before native diffusion LMs do.

## How to read this series

It helps to read these first: [BERT](../../00-foundations/core/bert.en.md) (masking and the bidirectional encoder), [the language model objective](../../00-foundations/deep-dives/language-model-objective.en.md) (autoregressive likelihood and the mismatch between training and generation), [decoding strategies](../../00-foundations/core/decoding.en.md), [the KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.en.md) and [the roofline and decode batching](../../12-infra/performance.en.md#decode-batching). Neighbouring notes: [DFlash](../../00-foundations/deep-dives/dflash.en.md) uses block diffusion as a draft model for an AR model; [MTP](../../00-foundations/deep-dives/multi-token-prediction.en.md) covers training objectives that predict several tokens at once; [test-time compute](../test-time-compute/README.en.md) covers spending more inference compute for quality, and samplers like ReMDM, where more steps help, belong to the same question.

1. [What masked diffusion trains](masked-diffusion.en.md): forward masking, the $1/t$ weighting and how it differs from BERT's MLM; a two-token calculation showing that the objective is the average over all orders; how to read LLaDA's scale and scaling claims
2. [MDLM: from training to generation](mdlm-walkthrough.en.md): calculate a loss and reveal probabilities with two sentences; see why correct marginals can still produce conflicting words
3. [Decoding steps, caching and real speed](decoding-and-speed.en.md): step-by-step filling and remasking, how many tokens per step, why the KV cache breaks and how block diffusion fixes it; a teaching cost model of low and high concurrency on one H100, by hand; vendor numbers and third-party measurements; the relation to DFlash
4. [Autoregressive and diffusion LMs: comparing and combining them](ar-and-diffusion.en.md): conditional advantages, a block-cache walkthrough, AR planning with diffusion generation, and recent results with their limitations
5. [Context reuse and caching](context-reuse.en.md): fixed KV need not give fixed attention outputs; worked examples of the costs and errors behind E2D2, FlashBlock, and MAGE
6. [Review questions](review.en.md): interview questions and a self-check
