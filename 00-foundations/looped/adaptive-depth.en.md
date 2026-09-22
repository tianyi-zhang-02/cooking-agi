# Looped Transformers: how many loops per token

[中文](adaptive-depth.md) · **English**

> Reading time: ~2 min · Level: advanced · Last reviewed: 2026-09

## A fixed number of loops wastes most of them

In a sentence, words like "the" and "on" need one loop; only words that must be resolved, or that need several hops, deserve more. Running every token for the full count spends compute where it is not needed. Adaptive depth lets each token decide for itself how many loops to run.

<!-- widget:tx-loop-exit -->

## ACT and PonderNet

- **ACT**: the Universal Transformer gives each position a halting probability and stops it once the running total passes a threshold; a stopped position simply copies its state forward without further updates.
- **PonderNet** (2021) makes halting a probability distribution. The probability of stopping at step $n$ is

$$p_n = \lambda_n \prod_{j<n} (1 - \lambda_j)$$

The loss weights each step's loss by $p_n$ and adds a KL term pulling $p$ towards a geometric prior: $\sum_n p_n \mathcal L_n + \beta\, \mathrm{KL}\big(p \,\|\, \mathrm{Geometric}(\lambda_p)\big)$. At inference it samples where to stop; unlike ACT, its gradients are unbiased.

## Ouro's exit gate

Ouro attaches an exit gate after every loop: $\lambda_t = \sigma(\mathrm{Linear}(h_t))$. Training has two stages:

1. **Stage one**: the loss is $\sum_t p(t \mid x)\, \mathcal L_t - \beta\, H(p)$. The entropy term is equivalent to an ELBO under a uniform prior, and it keeps the gate from favouring early exits from the start;
2. **Stage two**: the gate is trained on its own, with "how much would one more loop lower the loss" as the label.

An easy misunderstanding: in the Ouro code released on Hugging Face, **all 4 loops always run**; early exit only chooses which loop's hidden state is read out as the output, and saves no compute.

## Mixture-of-Recursions: a router assigns depth

Mixture-of-Recursions (2025) borrows the idea from MoE: a router decides each token's recursion depth (expert-choice or token-choice). A token not chosen for a loop skips it, and no KV is stored for it at that loop. On models from 135M to 1.7B, the paper reports throughput up to 2.06 times higher.
