# The Transformer, interactively: structure, optimizations, and model families

[中文](transformer-lab.md) · **English**

> Reading time: ~15 min, longer if you play · Level: intro → advanced · Last reviewed: 2026-10-09
>
> This is the hands-on companion to [The Transformer architecture](transformer.en.md). The formulas and derivations live there; this page does one thing: make every mechanism something you can drag, click, and watch the numbers of. Everything in the figures is computed live in your browser. Model-family configurations change quickly, so treat the papers and released configs at the end as the source of truth.

<div class="lesson-recipe advanced">
  <div><span>What we are dissecting</span><strong>One morphing block diagram, and the seven optimizations that reshaped it</strong></div>
  <div><span>Prerequisites</span><strong>the attention formula · residuals · causal LM</strong></div>
  <div><span>Main mechanism</span><strong>KV cache · GQA / MLA · sliding window · RoPE · MoE · FlashAttention</strong></div>
  <div><span>Evidence to demand</span><strong>what each one saves (memory, memory traffic, or compute) and what it costs</strong></div>
</div>

One colour key runs through the page: **blue** is queries and attention, **orange** is keys, the FFN and experts, **green** is values and embeddings.

## The map: one block, many families

Start with a common autoregressive Transformer: embed tokens, pass them through several blocks, and predict the next token. Repeated structure does not mean shared parameters. In a standard block, attention moves information **between** tokens and the FFN transforms each token **independently**, both through residual branches. Hybrid architectures can replace some attention layers with recurrent state updates, so this diagram is not a universal blueprint for every LLM.

Start with a few visible choices: where the norm goes, how position is encoded, how K/V heads are shared, and whether the FFN is dense or a mixture of experts. Pick a family and watch what moves; rows marked with a dot changed from the previous selection. Data, training objectives, and inference settings are outside the diagram but also affect performance.

<!-- widget:tx-arch -->

Three steps are worth comparing: `Transformer ’17 → GPT-2 / 3` (drop the encoder and cross-attention, move the norm inside the residual branch), `GPT-2 / 3 → Llama 3` (LayerNorm → RMSNorm, absolute positions → RoPE, GELU → SwiGLU, MHA → GQA), and `Llama 3 → DeepSeek-V3` (GQA → MLA, dense FFN → MoE).

## Self-attention, step by step

Each token emits three vectors: a **query** (what am I looking for?), a **key** (what do I offer?) and a **value** (what do I pass on?). Scores are dot products $q\cdot k$, scaled by $\sqrt{d_k}$, masked so that no token sees the future, and normalised with a softmax. The output is a weighted average of values: $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$.

<!-- widget:tx-attention -->

This is a toy: 6 tokens, $d_k=4$, and hand-set rather than trained weights, so the calculation is easy to check. Hover a row to follow one token through it. Removing the causal mask demonstrates bidirectional attention, not a complete BERT model: the objective and other layers also differ. For the $\sqrt{d_k}$ factor and the three projections, see [The Transformer architecture](transformer.en.md).

## Why decoding needs a KV cache

Generation happens one token at a time. Recomputing a complete length-$t$ prefix requires $t$ keys, $t$ values, and $t(t+1)/2$ valid causal scores per layer and head. With unchanged weights, prefix, and position settings, and dropout disabled, past K/V can be reused: calculate only the new token's K/V and one attention row. Changing the prefix, weights, or position handling can invalidate that cache.

<!-- widget:tx-kv-cache -->

Switch to “No cache” and compare the cumulative bars. At fixed depth and dimensions, attention work per step falls from $O(t^2)$ to $O(t)$, while cache storage grows with length. Small-batch decoding is often limited by weight and KV reads; larger batches, different kernels, or other hardware can change the bottleneck. The next section compares KV storage, not guaranteed throughput.

## Shrinking the cache: MHA → GQA → MQA → MLA

Per layer and per token, the cache holds

$$
2 \times n_{\text{kv heads}} \times d_{\text{head}}
$$

numbers, assuming equal K/V dimensions and the same configuration in every layer. Multiply by layers, tokens, and bytes per number for storage in bytes. With KV head count and dimensions fixed, query head count adds no extra factor:

- **MQA** (2019): all query heads share one K/V head. At fixed dimensions, this is the smallest cache within these head-sharing schemes; any quality loss depends on training and the task.
- **GQA** (2023): each group of query heads shares a K/V head, trading storage against representation capacity. The Llama 3, Mistral, Qwen3, and Gemma 3 versions below use it.
- **MLA** (DeepSeek-V2, 2024): cache a compressed KV latent plus a separate RoPE key. Storage per layer and token is $d_c+d_r$, not universally $4.5\,d_{\text{head}}$. DeepSeek-V3 uses $512+64=576$ elements. Weight absorption can also avoid explicitly reconstructing all K/V during decoding; see the [MLA derivation](deep-dives/latent-and-sparse-attention.en.md).

<!-- widget:tx-kv-heads -->

The calculator is a controlled comparison, not a deployment configuration: keep depth, length, and storage precision fixed while replacing the attention scheme. The MLA row fixes a 512-dimensional latent and a 64-dimensional RoPE key. It excludes weights, temporary activations, KV page fragmentation, and cross-device replication, and does not imply that every model supports all four variants.

## Not every token needs to see every token

Full causal attention costs $O(n^2)$. **Sliding-window attention** lets each token see only the last $w$ tokens: $O(n\,w)$ compute and a cache capped at $w$. Information still travels further, because every extra layer extends the reach by another window: about $L\,(w-1)+1$ tokens after $L$ layers.

<!-- widget:tx-windows -->

Drag “Stacked layers” to expand the possible information paths. Reachability does not guarantee that distant content is remembered. Gemma 3 interleaves five local layers with one global layer; gpt-oss alternates windowed and full attention. StreamingLLM retains initial sink tokens and a recent window to reduce instability from discarding the initial KV. Details evicted from the window do not remain fully available.

## RoPE: position as rotation

RoPE rotates pairs of dimensions in $q$ and $k$ at different frequencies instead of adding a position vector to embeddings. **For fixed content vectors**, the positional effect on their dot product depends on relative distance $m-n$. Real attention scores still depend on content, not distance alone.

<!-- widget:tx-rope -->

Press play: shifting both positions leaves the score unchanged. Drag distance $\Delta$ and the curve oscillates; it is **not pointwise monotonically decreasing**. One rotation plane is enough to see why: with both content vectors set to $(1,0)$, the dot product is $\cos\Delta$, rising from -1 to 1 between phase differences $\pi$ and $2\pi$. Increasing the base slows some dimensions, but changing that number alone does not guarantee useful long context; training and evaluation still matter. See [The Transformer architecture · RoPE](transformer.en.md).

## MoE: more parameters without using all of them {#moe-more-parameters-same-compute-per-token}

A mixture-of-experts layer replaces one FFN with several experts and a router that selects $k$ for each token. At fixed expert size and $k$, more total experts do not increase the selected FFNs' arithmetic. Routing, communication, and load imbalance still cost work. Selecting two equally sized experts is not as cheap as the original single dense FFN.

<!-- widget:tx-moe -->

The grids are drawn at true scale (8, 128 or 256 experts), so the sparsity you see is the real sparsity. DeepSeek-V3 adds one shared expert that every token visits. The router in the figure is random and only there to show how load spreads; a real router is a learned linear layer, kept balanced by an auxiliary loss or a bias term. The full series is in [MoE](moe/README.en.md).

## FlashAttention: same math, less memory traffic

Naive attention materializes full $n\times n$ score and probability matrices in HBM, which can make memory traffic expensive. FlashAttention uses tiling and online softmax to accumulate outputs without storing those two full matrices in HBM. It does not approximate attention by sparsifying it, but floating-point order changes mean results need not be bitwise identical. Speedups depend on shapes, hardware, and kernels; attention is not universally memory-bound.

<!-- widget:tx-flash -->

Switch to “Standard attention” and watch the two $n\times n$ matrices S and P being written out in full; switch back and notice the upper-right blocks that the causal mask lets it skip entirely.

## Families at a glance

These are **specific historical versions**, not a ranking of each family's latest release. Numbers come from the corresponding reports and configs; see [model-family readings](model-families/README.en.md) for later versions.

| Family | Layout | Norm | Position | Attention | FFN | Context | Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer (2017, base) | Encoder–decoder, 6 + 6 layers | Post-LN · LayerNorm | Sinusoidal, added to the embedding | MHA · 8 heads, plus cross-attention | Dense · ReLU · $d_{ff}$ 2,048 | Sentence pairs (translation) | 65M · $d_{model}$ 512 |
| GPT-2 / GPT-3 (2019–20) | Decoder-only | Pre-LN · LayerNorm, plus a final one | Learned absolute positions | MHA · 25 heads (GPT-2 XL) · 96 heads (GPT-3) | Dense · GELU · 4× | 1,024 · 2,048 | 1.5B: 48 layers · 175B: 96 layers |
| Llama 3 (2024, 8B) | Decoder-only | Pre-norm · RMSNorm | RoPE · $\theta$ = 500,000 | GQA · 32 query / 8 KV heads | Dense · SwiGLU · $d_{ff}$ 14,336 | 8K → 128K (Llama 3.1) | 8B: 32 layers, $d$ 4,096 |
| Mistral 7B (2023) | Decoder-only | Pre-norm · RMSNorm | RoPE | GQA · 32 / 8 · sliding window 4,096 | Dense · SwiGLU · $d_{ff}$ 14,336 | 8K, through a 4,096 window per layer | 7B: 32 layers, $d$ 4,096 |
| Mixtral 8x7B (2023) | Decoder-only | Pre-norm · RMSNorm | RoPE | GQA · 32 / 8 · full 32K attention | MoE · 8 SwiGLU experts, top-2 | 32K | ≈47B total · ≈13B active |
| Gemma 3 (2025, 27B) | Decoder-only | Pre-norm and post-norm · RMSNorm | RoPE · 10k on local layers, 1M on global | GQA · 32 / 16 · QK-norm · 5 local (window 1,024) : 1 global | Dense · GeGLU | 128K | 27B: 62 layers, $d$ 5,376 |
| Qwen3 (2025, 235B-A22B) | Decoder-only | Pre-norm · RMSNorm | RoPE · YaRN for long context | GQA · 64 / 4 · QK-norm | MoE · 128 experts, top-8, no shared expert | 32K native · 131K with YaRN | 235B total · 22B active · 94 layers |
| DeepSeek-V3 (2024) | Decoder-only | Pre-norm · RMSNorm | RoPE on a separate 64-dim key | MLA · 128 heads · cached latent 512 + 64 | MoE · 1 shared + 256 routed, top-8 | 128K | 671B total · 37B active · 61 layers |
| gpt-oss-120b (2025) | Decoder-only | Pre-norm · RMSNorm | RoPE · YaRN to 131K | GQA · 64 / 8 · window 128 on alternate layers · learned sinks | MoE · 128 experts, top-4 | 131K | 117B total · 5.1B active · 36 layers |

## Self-check

<div class="taste-check advanced">
  <strong>After this page you should be able to explain six things:</strong>
  <ol>
    <li>Which four parts of the block changed between the original Transformer and Llama 3, and what did each fix?</li>
    <li>Under what conditions can KV be reused, and why can memory bandwidth still limit decoding after computation is reduced?</li>
    <li>What exactly do GQA, MQA and MLA cache? Why is the MLA cache independent of the number of heads?</li>
    <li>With a sliding window each layer sees only $w$ tokens. How does information travel further than that?</li>
    <li>Why is FlashAttention exact, and which kind of cost does it remove?</li>
    <li>What do total and active parameters each scale with in an MoE?</li>
  </ol>
</div>

## Keep reading

- [The Transformer architecture](transformer.en.md): formulas, derivations and a from-scratch implementation of everything on this page
- [Multi-head attention](core/multi-head-attention.en.md) and [Decoder-only](core/decoder-only.en.md): the step-by-step core chapters

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the original architecture
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) · [RMSNorm](https://arxiv.org/abs/1910.07467) · [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (MQA) · [GQA](https://arxiv.org/abs/2305.13245) · [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (MLA)
- [RoFormer](https://arxiv.org/abs/2104.09864) (RoPE) · [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [FlashAttention](https://arxiv.org/abs/2205.14135) · [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538) · [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- Model reports: [GPT-3](https://arxiv.org/abs/2005.14165) · [Llama 3](https://arxiv.org/abs/2407.21783) · [Mistral 7B](https://arxiv.org/abs/2310.06825) · [Gemma 3](https://arxiv.org/abs/2503.19786) · [Qwen3](https://arxiv.org/abs/2505.09388) · [DeepSeek-V3](https://arxiv.org/abs/2412.19437) · [gpt-oss](https://arxiv.org/abs/2508.10925)
