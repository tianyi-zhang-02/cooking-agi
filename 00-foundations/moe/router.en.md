# MoE: how the router picks experts

[中文](router.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-10-09

<span id="the-router-is-one-linear-layer"></span>

## A common router starts with a linear layer

A common router multiplies hidden state $x$ by an $N \times d$ matrix, giving $N$ scores: $s = W_r x$. It keeps the $k$ highest and normalizes over them for gate weights. Other routers can add grouping, biases, or different scoring rules:

$$g_i = \frac{e^{s_i}}{\sum_{j \in \mathrm{TopK}} e^{s_j}}, \qquad i \in \mathrm{TopK}$$

This is the same as a softmax over all $N$ followed by renormalising over the ones kept. Mixtral and gpt-oss do exactly this; DeepSeek-V3 scores with a sigmoid instead and normalises over the selected experts.

For scores `[2, 1, 0]`, keeping and renormalizing the top two gives weights about `[0.731, 0.269]`. This is not universal: top-1 in [Switch Transformer](https://www.jmlr.org/papers/v23/21-0998.html) keeps the selected probability from the full softmax rather than renormalizing it to 1. That gate preserves a task-gradient path to the router. The figure illustrates renormalization over selected experts.

<!-- widget:tx-moe-router -->

## Top-1, top-2, or top-8

- **Top-1**: Switch Transformer. One expert per token, using less expert compute when other factors are equal.
- **Top-2**: GShard, Mixtral. Two expert outputs can contribute, providing another path at additional cost.
- **Top-4**: gpt-oss.
- **Top-8**: DeepSeek-V3, Qwen3. Their experts are cut finer (next-but-one note), so each token picks more of them.

At fixed expert width, larger $k$ usually means more expert compute; cross-device traffic also depends on placement. Across models, top-8 with small experts need not cost more than top-2 with wide experts, or be more stable.

## Why early MoE added noise

The first sparsely-gated MoE (Shazeer et al., 2017) added Gaussian noise with a learned scale to the scores before taking the top k:

$$H_i = (xW_g)_i + \mathcal N(0,1)\cdot \mathrm{Softplus}\big((xW_{\text{noise}})_i\big)$$

[Noise](https://arxiv.org/abs/1701.06538) gives experts near the selection boundary a chance to run. It supports exploration, but a single perturbation need not change the selection, and balanced training is not guaranteed. The figure uses synthetic scores to show how noise changes decisions near a boundary; it does not identify actual word-to-expert assignments.

## Top-k has no gradient: how does the router learn?

Ordinary top-k implementations do not backpropagate through the selected indices. Gradients reach the router through selected gate weights $g_i$. Unselected expert **parameters** usually receive no task gradient from that token; unselected router logits may still receive gradients through normalization or auxiliary objectives. Full-softmax denominators involve other logits, whereas renormalizing over selected experts changes that relationship.

One possible feedback loop is that experts receiving more tokens early get more training and then get selected more often. The next note covers detecting and mitigating that imbalance, rather than assuming it must occur.

## Tokens choose experts, or experts choose tokens

Everything above has tokens choose experts. [Expert Choice](https://arxiv.org/abs/2202.09368) reverses that: each expert selects a fixed number of tokens from the batch. With enough candidates, assignment counts can balance by construction, though device runtimes need not. A token can be selected several times or not at all. Reported convergence gains apply to the paper's training settings, not arbitrary serving workloads.

For autoregressive tasks, ask whether the candidate set contains future tokens. If later tokens can change whether an earlier token is selected, selection itself can leak future information. Restrict candidate scope during training, and check whether batching changes a single request's behavior at deployment.
