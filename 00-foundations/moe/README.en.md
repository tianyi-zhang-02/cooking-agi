# MoE: why make a model sparse

[中文](README.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-10-09

Suppose a layer stores eight equally sized FFNs but each token uses only two. Different tokens can access different parameters without running all eight each time. The eight sets of weights still need storage, and tokens may travel between devices. To understand MoE, separate what is stored from what is used on one pass.

## Most of a dense block's parameters sit in the FFN

Consider a standard MHA block of width d with FFN hidden width 4d, ignoring biases and norms. Attention projections have about 4d² parameters and the FFN has 8d², two thirds of their combined total. GQA or a different FFN width changes that fraction. Dense FFN matrices normally run for every token, with about one multiply-add per matrix weight, or 2 FLOPs under the common convention. Widening the FFN therefore increases per-token compute.

## MoE replaces the FFN with N experts

A Mixture-of-Experts (MoE) layer replaces that single FFN with $N$ expert FFNs of the same shape, plus a small router. Each token goes only to the $k$ experts the router scores highest, and the output is their weighted sum:

$$y = \sum_{i \in \mathrm{TopK}(x)} g_i(x)\, E_i(x)$$

In this replacement, attention, embeddings, and norms remain shared. Models need not replace every FFN with MoE, and MoE is not restricted to Transformers. This series focuses on the sparse FFNs commonly used in language models.

<!-- widget:tx-moe -->

## Total versus active parameters

At fixed expert width, total expert parameters grow with N and main expert compute grows with k. Router scoring over N experts, attention, shared components, and communication also count. These figures refer to specific public models, not universal family configurations:

| Model | Experts per layer | Experts per token | Total | Active |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 8 | 2 | 46.7B | 12.9B |
| DeepSeek-V3 | 256 routed + 1 shared | 8 + 1 | 671B | 37B |
| Qwen3-235B-A22B | 128 | 8 | 235B | 22B |
| gpt-oss-120b | 128 | 4 | 116.8B | 5.1B |

Mixtral 8x7B is not 56B: only the FFNs are copied eight times, while attention and embeddings exist once, so the total is 46.7B.

## Sparsity saves compute, not memory

Running fewer experts does not mean storing only those experts. All weights still live on GPUs, CPUs, or other storage. Full GPU residency follows total parameters; offloading lowers GPU occupancy at the cost of transfers and waiting. A fair dense comparison also fixes precision, target quality, and batch size rather than treating sparsity as a memory guarantee.

Parameter sources are linked in the [review page's model reports](review.en.md); deployment tradeoffs are covered in [system costs](systems.en.md).

## How this series reads

1. [How the router picks experts](router.en.md): scores, top-k, renormalisation, and why early MoE added noise (interactive)
2. [Load balancing](load-balancing.en.md): why imbalance can develop; auxiliary losses, capacity, and bias adjustment (interactive)
3. [Fine-grained and shared experts](fine-grained-and-shared.en.md): DeepSeekMoE's two changes, and what different labs chose
4. [LatentMoE: must experts be as wide as the backbone?](latent-moe.en.md): distinguish three widths, work out parameter and communication costs, then examine Kimi K3
5. [What it costs to train and serve](systems.en.md): expert parallelism, all-to-all, memory, and decoding
6. [Review questions](review.en.md): interview questions and a self-check
