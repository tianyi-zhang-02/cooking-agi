# MoE: why make a model sparse

[中文](README.md) · **English**

> Reading time: ~5 min · Last reviewed: 2026-10

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
| DeepSeek-V4-Pro (2026) | 384 routed + 1 shared | 6 + 1 | 1.6T | 49B |
| Qwen3.8-2.4T-A95B (2026) | 512 routed + 1 shared | 10 + 1 | 2.4T | 95B |
| Gemma 4 26B A4B (2026) | 128 routed + 1 shared | 8 + 1 | 25.2B | 3.8B |
| Kimi K3 (2026) | 896 routed + 2 shared | 16 + 2 | 2.8T | 104B |

Mixtral 8x7B is not 56B: only the FFNs are copied eight times, while attention and embeddings exist once, so the total is 46.7B.

The four 2026 rows were read in October 2026. Sources: [V4-Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json), [Qwen3.8-2.4T-A95B model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), [Kimi K3 report](https://arxiv.org/abs/2607.24653). All four keep shared experts; V4-Pro's first three MoE layers route by token-ID hash rather than content top-k.

## Sparsity saves compute, not memory

Running fewer experts does not mean storing only those experts. All weights still live on GPUs, CPUs, or other storage. Full GPU residency follows total parameters; offloading lowers GPU occupancy at the cost of transfers and waiting. A fair dense comparison also fixes precision, target quality, and batch size rather than treating sparsity as a memory guarantee.

Parameter sources are linked in the [review page's model reports](review.en.md); deployment tradeoffs are covered in [system costs](systems.en.md).

### Cheaper than which dense model? {#dense-comparison}

For a toy comparison, count only the main FFN matrix products. Fix input width `d=128` and use SwiGLU's three matrices. With intermediate width `h`, parameters and main multiply-accumulates per token both equal `3dh`. Exclude biases, activations, routing, attention, and communication:

<div class="worked-table" markdown="1">

| Configuration | FFN parameters | Main MACs / token |
| --- | --- | --- |
| Dense, h=256 | 98,304 | 98,304 |
| MoE, eight h=256 experts, top-2 | 786,432 | 196,608 |
| Dense, h=512: matched MoE compute | 196,608 | 196,608 |
| Dense, h=2048: matched MoE parameters | 786,432 | 786,432 |

</div>

This MoE does twice the work of a single same-width FFN, but one quarter of the work of a parameter-matched dense FFN. Against the compute-matched dense version, it stores four times as many parameters, with routing and transfers on top. **The baseline changes the conclusion.** These counts establish neither equal quality nor proportional latency savings.

## How this series reads

1. [How the router picks experts](router.en.md): scores, top-k, renormalization, and why early MoE added noise (interactive)
2. [Load balancing](load-balancing.en.md): why imbalance can develop; auxiliary losses, capacity, and bias adjustment (interactive)
3. [Fine-grained and shared experts](fine-grained-and-shared.en.md): DeepSeekMoE's two changes, and what different labs chose
4. [LatentMoE: must experts be as wide as the backbone?](latent-moe.en.md): distinguish three widths, work out parameter and communication costs, then examine Kimi K3
5. [What it costs to train and serve](systems.en.md): expert parallelism, all-to-all, memory, and decoding
6. [Review questions](review.en.md): interview questions and a self-check
