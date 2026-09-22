# MoE: why make a model sparse

[中文](README.md) · **English**

> Reading time: ~2 min · Level: advanced · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>more parameters without paying more compute for every token</strong></div>
  <div><span>Prerequisites</span><strong>Transformer block · FFN · softmax</strong></div>
  <div><span>Core mechanism</span><strong>replace the FFN with N experts; a router sends each token to only k of them</strong></div>
  <div><span>Common mistakes</span><strong>thinking 8×7B means 56B; thinking fewer active parameters means less memory</strong></div>
</div>

## Most of a dense block's parameters sit in the FFN

A Transformer block holds two things: attention and the FFN. The FFN carries about two thirds of a block's parameters, and **every token runs through all of them**. Each parameter costs about 2 floating-point operations per token in the forward pass, so in a dense model the parameter count and the compute per token are tied together: more parameters means more compute for every token.

## MoE replaces the FFN with N experts

A Mixture-of-Experts (MoE) layer replaces that single FFN with $N$ expert FFNs of the same shape, plus a small router. Each token goes only to the $k$ experts the router scores highest, and the output is their weighted sum:

$$y = \sum_{i \in \mathrm{TopK}(x)} g_i(x)\, E_i(x)$$

Attention, embeddings, and norms are unchanged and still shared by every token.

<!-- widget:tx-moe -->

## Total versus active parameters

Total parameters grow with $N$; compute per token grows only with $k$. Official numbers for a few public models:

| Model | Experts per layer | Experts per token | Total | Active |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 8 | 2 | 46.7B | 12.9B |
| DeepSeek-V3 | 256 routed + 1 shared | 8 + 1 | 671B | 37B |
| Qwen3-235B-A22B | 128 | 8 | 235B | 22B |
| gpt-oss-120b | 128 | 4 | 116.8B | 5.1B |

Mixtral 8x7B is not 56B: only the FFNs are copied eight times, while attention and embeddings exist once, so the total is 46.7B.

## Sparsity saves compute, not memory

Any token can be routed to any expert, so **every expert has to be resident** (or sharded across GPUs). Memory follows the total parameter count; compute per token follows the active count. That is the main cost of MoE in deployment, covered in its own note.

## How this series reads

1. [How the router picks experts](router.en.md): scores, top-k, renormalisation, and why early MoE added noise (interactive)
2. [Load balancing](load-balancing.en.md): left alone it collapses; auxiliary losses, capacity, and the bias-only fix (interactive)
3. [Fine-grained and shared experts](fine-grained-and-shared.en.md): DeepSeekMoE's two changes, and what different labs chose
4. [What it costs to train and serve](systems.en.md): expert parallelism, all-to-all, memory, and decoding
5. [Review questions](review.en.md): interview questions and a self-check
