# Scaling: why today's models train far longer than Chinchilla

[中文](beyond-chinchilla.md) · **English**

> Reading time: ~10 min · Last reviewed: 2026-10

Llama 3's 8B model saw more than 15T tokens, about 1,900 tokens per parameter, close to a hundred times Chinchilla's 20. The two ratios address different questions. Chinchilla asks how to achieve the lowest loss for a fixed training budget; deployment also requires accounting for inference costs and the amount of unique data available.

In the cost model below, including inference tokens in total FLOPs shifts the optimum toward a smaller model trained on more data. The larger the expected inference volume, the further it shifts. Data-constrained experiments also find that repeating the same data about 4 times is nearly as good as using new data, with returns falling quickly after a dozen or so passes. These conclusions come from fitted relationships too; at a thousand or more tokens per parameter, the curves overestimate the benefit of more data.

## Put inference on the same bill

The LLaMA 1 authors (Touvron et al., 2023-02) say it plainly in their introduction: for a target level of performance, the model to choose is not the fastest to train but the fastest at inference; they also observed that a 7B model kept improving after 1T tokens ([arXiv](https://arxiv.org/abs/2302.13971)).

[Sardana et al. (2023-12, ICML 2024)](https://arxiv.org/abs/2401.00448) turned this into an optimization problem. Fix a target loss $\ell$ and minimize total training plus inference FLOPs:

$$
\min_{N,\,D_{\rm tr}}\;6ND_{\rm tr}+2ND_{\rm inf}\qquad\text{s.t.}\quad L(N,D_{\rm tr})=\ell .
$$

$D_{\rm tr}$ is the number of training tokens and $D_{\rm inf}$ the number of inference tokens the model will process over its service life; each inference token is forward-only and costs $2N$. With $D_{\rm inf}=0$ this reduces to Chinchilla's problem (stated the other way round: fix the loss and find the cheapest compute); the larger $D_{\rm inf}$, the more each extra bit of model size costs on the inference side. There is no closed form, so it is solved numerically.

### Work the previous note's example once

Keep the previous note's teaching surface (Besiroglu et al.'s refit of the Chinchilla data). Set the target loss to that of the compute-optimal model at $10^{22}$ FLOPs (about 9B parameters, 184B tokens), vary the lifetime inference volume, and solve numerically for the $N$ with the smallest total bill in each row. Every number in the table is a teaching number on this surface, not a measurement:

| Inference tokens $D_{\rm inf}$ | Optimal $N$ | Training tokens | $D/N$ | Training FLOPs | Total FLOPs | Total saving vs the Chinchilla-optimal model |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 9.0B | 184B | 20 | $1.00\times10^{22}$ | $1.00\times10^{22}$ | 0 |
| $10^{11}$ | 7.5B | 225B | 30 | $1.01\times10^{22}$ | $1.16\times10^{22}$ | 1.5% |
| $10^{12}$ | 4.6B | 447B | 97 | $1.24\times10^{22}$ | $2.16\times10^{22}$ | 23% |
| $10^{13}$ | 2.8B | 1.49T | 536 | $2.48\times10^{22}$ | $8.03\times10^{22}$ | 58% |
| $10^{14}$ | 2.0B | 6.41T | 3211 | $7.68\times10^{22}$ | $4.76\times10^{23}$ | 74% |

$10^{12}$ inference tokens is not extreme: a million requests a day at 1,000 tokens each for 1,000 days. The optimal model is then half the Chinchilla answer; training costs 24% more compute, and the total bill falls by 23%. The last two rows are already outside the fitted surface: as the next section shows, at hundreds to thousands of tokens per parameter the real loss is higher than this surface predicts, so 58% and 74% are optimistic upper bounds.

With the original Chinchilla coefficients ($\alpha=0.336$, $\beta=0.283$), Sardana et al. report that to match a Chinchilla-style 30B model while serving $10^{13}$ inference tokens, the optimum is 13.6B parameters on 2.84 times the data, with 28% fewer total FLOPs (we recomputed with their coefficients and got the same numbers). They also convert FLOPs to dollars: training and prompt processing at 50% MFU, token-by-token generation at only 1% MFU, which pushes the answer further toward small models. Why generation runs at low utilization is covered in [performance](../../12-infra/performance.en.md#decode-batching).

There is another way to spend more compute at inference: let the model think longer when it answers. In [the test-time compute note](../../11-frontier/test-time-compute/parallel-and-verify.en.md), Snell et al. use the same $6N$ / $2N$ accounting to compare scaling pretraining with sampling more at inference; the answer depends on problem difficulty and inference load, and the two do not trade 1:1.

## Does loss keep falling far past 20?

It does, but more slowly than the formula predicts. Sardana et al. trained 47 models. For a 150M-parameter model, loss continued to fall even at 10,000 tokens per parameter. Larger models were tested up to 1,000 tokens per parameter (the 2.5B model up to 500), without reaching a plateau either. But a curve fitted only on typical ratios (under 100 tokens per parameter) overestimates the gain from extra data at extreme ratios. Including the extreme-ratio points in the fit produces noticeably flatter curves. They also found that none of the parametric curves fit the long runs of the 150M model well.

[Gadre et al. (2024-03)](https://arxiv.org/abs/2403.08540) tested directly whether over-training can be extrapolated. They trained 104 models from 0.011B to 6.9B on three data distributions and, fitting with about 300 times less compute, predicted the validation loss of a 1.4B model trained on 900B tokens (32 times more data than compute-optimal) and a 6.9B model trained on 138B tokens (compute-optimal); they then fitted a power law from perplexity to average downstream error and predicted downstream error with about 20 times less compute.

Taken together: over-training is predictable, but fit close to the ratio you are aiming for rather than extrapolating from 20 tokens per parameter all the way to 2,000.

## Not enough data: how many times can the same data be repeated

Put the previous note's 9B model in a different situation: you have only 100B non-repeated tokens but want to train on 184B. What is the repeated part worth?

[Muennighoff et al. (2023-05)](https://arxiv.org/abs/2305.16264) ran 400 training runs (up to 9B parameters and 900B tokens) and converted repeated data into an "effective data" amount:

$$
D'=U_D+U_DR_D^{*}\left(1-e^{-R_D/R_D^{*}}\right).
$$

$U_D$ is the number of unique tokens, $R_D$ the number of repetitions beyond the first pass ($R_D=3$ for 4 epochs), and $R_D^{*}\approx15.4$ a fitted decay constant. Each extra pass adds effective data that shrinks as $e^{-R_D/R_D^{*}}$; as the number of passes goes to infinity, $D'\to U_D(1+R_D^{*})$, at most about 16.4 times the fresh data.

With $U_D=100$B:

| Epochs | Tokens actually trained | Effective data $D'$ | Effective / actual |
| --- | --- | --- | --- |
| 1 | 100B | 100B | 1.00 |
| 2 | 200B | 196.8B | 0.98 |
| 4 | 400B | 372.6B | 0.93 |
| 16 | 1.6T | 1.06T | 0.66 |
| 40 | 4T | 1.52T | 0.38 |

The 2nd pass is worth about 0.97 of a fresh pass, the 5th about 0.80, and the 16th only about 0.39. The paper's conclusion: at a fixed compute budget, repeating up to about 4 times gives nearly the same loss as all-new data (an 8.7B model trained for 4 epochs ended with a validation loss only about 0.5% higher than with 1 epoch); beyond that, the value of extra compute decays toward zero, and models trained for 44 epochs even saw the loss rise midway through training. Back to the example: 184B tokens is 1.84 passes, worth about 182B of effective data, so almost nothing is lost.

They also fitted a decay constant for "excess parameters", $R_N^{*}\approx5.3$, smaller than $R_D^{*}$: surplus parameters lose value faster than repeated data. So when data is limited, extra compute is better spent on more passes than on a bigger model, and the optimal model is smaller and trained for more epochs than Chinchilla suggests.

Real recipes already repeat data. In Chinchilla's training set, Wikipedia was seen 3.4 times and MassiveWeb 1.24 times (Appendix Table A1); LLaMA 1 saw Wikipedia 2.45 times and Books 2.23 times (Table 1). How to account for oversampling by source is covered in [the pretraining pipeline](../deep-dives/pretraining-pipeline.en.md).

## What current open models actually do

The table lists only numbers from official reports or model cards, ordered by the month the cited report or model card appeared (weights were sometimes released earlier), as of October 2026. "Tokens per parameter" is our own division of the numbers in the table: where the report gives a parameter count we use it (6.7B for LLaMA 1), otherwise the nominal size in the model name (Qwen3-8B is about 8.2B, counted as 8B); MoE models are divided by activated parameters, with DeepSeek-V3 also shown per total parameter.

| Model | Report date | Parameters | Training tokens | Tokens per parameter | Source |
| --- | --- | --- | --- | --- | --- |
| Chinchilla | 2022-03 | 70B | 1.4T | 20 | Paper |
| LLaMA 1 7B | 2023-02 | 6.7B | 1.0T | about 150 | Paper, Table 2 |
| Llama 3 8B / 70B | 2024-04 | 8B / 70B | over 15T | about 1,900 / 210 | [Model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) |
| Llama 3.1 405B | 2024-07 | 405B | 15.6T | about 39 | Report |
| DeepSeek-V3 | 2024-12 | 671B total / 37B active | 14.8T | about 400 (about 22 per total parameter) | [Report](https://arxiv.org/abs/2412.19437) |
| OLMo 2 7B / 13B | 2024-12 | 7B / 13B | 4.05T / 5.6T | about 580 / 430 | [Report](https://arxiv.org/abs/2501.00656) |
| SmolLM2 | 2025-02 | 1.7B | about 11T | about 6,500 | [Report](https://arxiv.org/abs/2502.02737) |
| Gemma 3 27B / 1B | 2025-03 | 27B / 1B | 14T / 2T | about 520 / 2,000 | [Report](https://arxiv.org/abs/2503.19786) |
| Qwen3-8B / 0.6B | 2025-05 | 8B / 0.6B | about 36T (same for every size) | about 4,500 / 60,000 | [Report](https://arxiv.org/abs/2505.09388) |
| Kimi K2 | 2025-07 | 1.04T total / 32B active | 15.5T | about 480 | [Report](https://arxiv.org/abs/2507.20534) |
| DeepSeek-V4-Pro / Flash | 2026-04 | 1.6T total / 49B active; 284B total / 13B active | 33T / 32T | about 670 / 2,500 | [Model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [report §4.2](https://arxiv.org/abs/2606.19348) |

A pattern shows: the largest model in a family sits closest to compute-optimal, and the smaller ones are trained far past it. Llama 3 ran its own scaling-law study: fitting the optimal number of **training tokens** $D^*(C)=0.29\,C^{0.53}$ over $6\times10^{18}$ to $10^{22}$ FLOPs and 40M to 16B parameters (the report writes it as $N^\star$, but it counts tokens, not parameters), then extrapolating to the $3.8\times10^{25}$ FLOPs actually used for 405B, gave 402B parameters on 16.55T tokens; they trained 405B on 15.6T. A side example of extrapolation sensitivity: the $(0.53, 0.29)$ in the text are rounded, and plugging them in gives only about 10.5T; the $(0.537, 0.299)$ in the legend of Figure 3 give about 16.3T. The two pairs differ only in the third significant digit, yet after extrapolating more than three orders of magnitude they differ by a factor of 1.5. The same report says the smaller models were trained "much longer than is compute-optimal" and perform better at the same inference budget (self-reported).

Several cautions when reading this table:

- **Different units.** Gemma 3 used knowledge distillation in pretraining, so it is not learning only the next token in the data; the Qwen3 report says every size goes through the same three stages, about 36T in all (over 30T general data, about 5T higher-quality reasoning data, and hundreds of billions of long-context tokens); OLMo 2's totals count every parallel mid-training anneal (7B: 3.9T of pretraining plus 3 runs of 50B; 13B: 5T plus 3 runs of 100B and 1 of 300B; the branches are then weight-averaged). Tokenizers differ, so one token covers different amounts of text in different models.
- **MoE is not dense.** Estimating training compute as $6N_{\text{active}}D$ is a common approximation, but Chinchilla's loss formula was fitted on dense models. Kimi K2 reports its own sparsity scaling law (sparsity is total experts divided by activated experts): at fixed activated parameters, more experts give lower loss; to reach a validation loss of 1.5, sparsity 48 needs 1/1.69, about 59%, of the FLOPs of sparsity 8 (self-reported). The difference between total and activated parameters is covered in [MoE](../moe/README.en.md), and V4's architecture in [DeepSeek-V4](../model-families/deepseek-v4.en.md).
- **Training volume is not quality.** The table shows only how much each model was trained, not whether that ratio was optimal for it; most reports do not publish the matching IsoFLOP data.

## Where these corrections stop working

| Correction | What it assumes | When it fails |
| --- | --- | --- |
| Shrink the model for the expected inference volume | Lifetime inference tokens can be forecast; the loss surface stays accurate at high ratios | The inference forecast is off by an order of magnitude; at thousands of tokens per parameter the surface overestimates the value of data (Sardana et al.) |
| Convert repeated data into effective data | Fitted within 9B parameters and 900B tokens, on one kind of web text | With code, synthetic or curated data mixed in, one pass is worth something different; with many passes, validation loss can rise |
| Copy an open model's ratio | Their data, tokenizer, distillation and multi-stage recipe | Distilled small models and MoE models cannot be compared directly with dense ones |
| Use loss as a proxy for quality | Downstream metrics change smoothly with loss | Step-like metrics; post-training changes the capability being compared |

Next: [how learning rate, batch size and schedule scale up](hyperparameters.en.md). With $N$ and $D$ chosen, the next step is to set this run's optimization hyperparameters.
