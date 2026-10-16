# Scaling: work out the model, data and learning rate before a run

[中文](README.md) · **English**

> Reading time: ~5 min · Draft, pending review

Before a pretraining run starts, several things must be decided: how big the model is, how many tokens it trains on, what learning rate and batch size to use, and how the learning rate changes over time. Once they are set, days to months of compute get spent and are hard to change midway. Scaling laws let you estimate these numbers first from a set of cheap small runs. **The conclusion of this series: at a fixed training budget, parameters and data grow in roughly equal proportion (Chinchilla's roughly 20 tokens per parameter); models that will be deployed are usually trained far longer because of inference cost; learning rate and batch size also follow regularities with scale, and a parametrization such as µP can transfer a learning rate tuned on a small model to a large one. All of these are empirical curves fitted on particular data and recipes; the further from the fitted range, the more they should be read as ranges.**

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

## The numbers to fix before launch

| Number to fix | Mainly determined by | Which note | Common mistake |
| --- | --- | --- | --- |
| Training compute $C$ | GPU count, peak throughput, MFU, days; $C\approx6ND$ | Note 1 | Budgeting with the sparse peak |
| Parameters $N$ and training tokens $D$ | IsoFLOP fits, expected inference volume, amount of unique data | Notes 1 and 2 | Copying "20" without looking at data or inference volume |
| Batch size | The critical batch size, which grows as loss falls and data grows | Note 3 | Raising the batch to several times the critical value to use more GPUs |
| Learning rate | A formula fitted against scale, or µP transfer | Note 3 | Copying a small model's optimal learning rate to a large model under the standard parametrization |
| Learning-rate schedule | Cosine matched to the run length, or WSD | Note 3 | Setting the cosine cycle to the planned length, then stopping early |

## One teaching example throughout

Suppose you have 64 H100 SXM5 GPUs for about 4.6 days; at a BF16 dense peak of 989.4 TFLOPS and 40% MFU that is about $C=10^{22}$ FLOPs (how MFU is counted is covered in [performance: roofline and MFU](../../12-infra/performance.en.md#mfu)). The three notes answer in turn:

1. Looking only at training compute, the optimum is about 9B parameters and 180B tokens; how that answer is fitted from a few small runs, and how far it can be extrapolated.
2. If the deployed model will process $10^{12}$ inference tokens, the optimum becomes about 4.6B parameters and 450B tokens; and if you have only 100B unique tokens, how much repeated training is still worth.
3. What batch size and learning rate this run should use, which schedule to choose, and how to carry a learning rate swept on a width-512 proxy over to a width-4096 model.

All hand-worked numbers are teaching examples, not measurements of any model.

## Timeline

| Date | Work | In one line |
| --- | --- | --- |
| 2018-12 | [McCandlish et al.](https://arxiv.org/abs/1812.06162) | The gradient noise scale predicts the critical batch size |
| 2020-01 | [Kaplan et al.](https://arxiv.org/abs/2001.08361) | Loss is a power law in parameters, data and compute; $N_{\rm opt}\propto C^{0.73}$ |
| 2022-03 | [Chinchilla](https://arxiv.org/abs/2203.15556) | 400+ models; scale parameters and data equally, about 20 tokens per parameter |
| 2022-03 | [µP / µTransfer](https://arxiv.org/abs/2203.03466) | Trains a 6.7B GPT-3 with hyperparameters from a 40M-parameter proxy |
| 2023-02 | [LLaMA](https://arxiv.org/abs/2302.13971) | Choose the model by inference cost; a 7B model trained on 1T tokens |
| 2023-03 | [GPT-4 technical report](https://arxiv.org/abs/2303.08774) | Self-reports predicting the final loss from models with at most one ten-thousandth of the compute |
| 2023-05 | [Data-constrained scaling](https://arxiv.org/abs/2305.16264) | Repeating about 4 times is nearly as good as new data |
| 2023-12 | [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) | Puts inference FLOPs on the same bill |
| 2024-01 | [DeepSeek LLM](https://arxiv.org/abs/2401.02954) | Fits optimal learning rate and batch size against compute; data quality changes the allocation |
| 2024-04 | [MiniCPM](https://arxiv.org/abs/2404.06395), [Chinchilla replication](https://arxiv.org/abs/2404.10102) | WSD schedule; refitting approach 3 also gives about 20 |
| 2024-06 | [Porian et al.](https://arxiv.org/abs/2406.19146), [Pearce and Song](https://arxiv.org/abs/2406.12907) | Explain the Kaplan–Chinchilla disagreement |
| 2024-07 | [Llama 3](https://arxiv.org/abs/2407.21783) | Fits its own optimal-token law $D^*\propto C^{0.53}$, trains 405B on 15.6T; small models far past compute-optimal |
| 2024-10 | [Zhang et al.](https://arxiv.org/abs/2410.21676) | The critical batch size grows mainly with data size |
| 2025-05 | [Power Lines](https://arxiv.org/abs/2505.13738) | Scaling laws for batch size and weight decay |
| 2026-04 | [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | Two MoE models, both pretrained on more than 32T tokens, with most parameters updated by Muon |

## What has settled and what is still moving

Each cell points to a dated public record. The "signals" column says what kind of result would change the picture; it is not a prediction.

| Settled | Still moving | Signals to watch next |
| --- | --- | --- |
| At a fixed training budget, parameters and data grow in roughly equal proportion: Chinchilla's three approaches give 0.46–0.50, and replication and refitting give about 20 tokens per parameter | The ratio moves with data quality, tokenizer and recipe: DeepSeek LLM got 0.450–0.578 on three datasets | Published raw IsoFLOP data and confidence intervals, not just a fitted line |
| Models meant for deployment are trained far past compute-optimal: Llama 3 8B at about 1,900 tokens per parameter, Qwen3's small models even more | How loss flattens at extreme ratios: Sardana et al. found that fits on typical ratios overestimate the gain | Systematic measurements at thousands of tokens per parameter |
| Under the standard parametrization the optimal learning rate moves with width, and µP makes it transferable across width | Transfer across depth, batch size and training length; Everett et al. found that the standard parametrization with per-layer learning-rate rules also transfers | Large-scale, public comparisons of µP against other parametrizations |
| Cosine must match the run length; a short WSD cooldown matches it | Whether the critical batch size tracks data size or loss: Zhang et al. and Bergsma et al. fit differently from Kaplan and DeepSeek LLM | Comparing, on one recipe, the prediction error of batch fits against $C$ versus against $D$ |

## How to read this series

It helps to read these first: [the pretraining pipeline](../deep-dives/pretraining-pipeline.en.md) ($C\approx6ND$ and data mixtures), [one training step](../deep-dives/training-step.en.md), and [from SGD to AdamW](../deep-dives/optimizers.en.md). Neighbouring notes: [Muon](../deep-dives/muon.en.md) scales updates by matrix shape; [performance: roofline and MFU](../../12-infra/performance.en.md#mfu) turns GPU hours into effective FLOPs; [test-time compute](../../11-frontier/test-time-compute/README.en.md) compares spending more compute at inference with scaling pretraining; for total versus activated parameters see [MoE](../moe/README.en.md).

1. [For a given compute budget, how big a model and how much data](compute-optimal.en.md): $C\approx6ND$, power laws, fitting a line by hand from a few small runs, why Kaplan and Chinchilla differ, the conditions behind 20 tokens per parameter, and how far to trust a small-model fit
2. [Why today's models train far longer than Chinchilla](beyond-chinchilla.en.md): putting inference on the bill, over-training, how many times data can be repeated, what open models actually do, and where these corrections stop working
3. [How learning rate, batch size and schedule scale up](hyperparameters.en.md): the critical batch size, learning-rate conversion and fitting, cosine versus WSD, µP and one learning-rate transfer worked by hand
4. [Review questions](review.en.md): interview questions and a self-check
