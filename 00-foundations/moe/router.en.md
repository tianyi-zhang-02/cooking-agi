# MoE: how the router picks experts

[中文](router.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-09

## The router is one linear layer

The router multiplies the token's hidden state $x$ by an $N \times d$ matrix to get $N$ scores: $s = W_r x$. It keeps the $k$ highest and normalises over just those $k$ to get the gate weights:

$$g_i = \frac{e^{s_i}}{\sum_{j \in \mathrm{TopK}} e^{s_j}}, \qquad i \in \mathrm{TopK}$$

This is the same as a softmax over all $N$ followed by renormalising over the ones kept. Mixtral and gpt-oss do exactly this; DeepSeek-V3 scores with a sigmoid instead and normalises over the selected experts.

<!-- widget:tx-moe-router -->

## Top-1, top-2, or top-8

- **Top-1**: Switch Transformer. One expert per token, the cheapest in compute and communication.
- **Top-2**: GShard, Mixtral. A pair of experts is steadier than one.
- **Top-4**: gpt-oss.
- **Top-8**: DeepSeek-V3, Qwen3. Their experts are cut finer (next-but-one note), so each token picks more of them.

A larger $k$ means more compute and more cross-device traffic per token, but more combinations of expertise.

## Why early MoE added noise

The first sparsely-gated MoE (Shazeer et al., 2017) added Gaussian noise with a learned scale to the scores before taking the top k:

$$H_i = (xW_g)_i + \mathcal N(0,1)\cdot \mathrm{Softplus}\big((xW_{\text{noise}})_i\big)$$

Experts with close scores take turns being chosen, so load spreads out and every expert gets some training. In the figure above, one dose of noise on a function word like "the" is enough to change which experts run.

## Top-k has no gradient: how does the router learn?

Choosing which experts run is discrete and has no gradient. The gradient flows through the gate weights instead: the output is a sum of $g_i E_i(x)$, so how much a chosen expert helped reaches the router through $g_i$. An expert that was not chosen gets no gradient at all from this token.

That plants a problem: an expert that happens to get a few more tokens early gets trained a bit better, and then gets more tokens. Load balancing, the next note, is about exactly that.

## Tokens choose experts, or experts choose tokens

Everything above has each token choose its own top k. Expert Choice (Zhou et al., 2022) turns it around: each expert picks a fixed number of tokens it wants most from the batch. Load is balanced by construction; a token may be picked by several experts or by none; the paper reports training converging more than twice as fast as Switch and GShard. The cost is that whether a token is picked depends on the other tokens in the batch, which sits awkwardly with token-by-token autoregressive decoding.
