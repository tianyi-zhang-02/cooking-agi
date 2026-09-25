# The Transformer, interactively: structure, optimizations, and model families

[中文](transformer-lab.md) · **English**

> Reading time: ~12 min, longer if you play · Level: intro → advanced · Last reviewed: 2026-09
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

Every modern LLM runs the same loop: embed the tokens, push them through $N$ identical blocks, predict the next token. A block has two sub-layers. Attention moves information **between** tokens; the FFN transforms each token **on its own**. Both sit inside residual connections.

What separates one family from another is a short list of choices: where the norm goes, how position is encoded, how K/V heads are shared, and whether the FFN is dense or a mixture of experts. Pick a family and watch what moves; rows marked with a dot on the right are the ones that changed from your previous choice.

<!-- widget:tx-arch -->

Three steps are worth comparing: `Transformer ’17 → GPT-2 / 3` (drop the encoder and cross-attention, move the norm inside the residual branch), `GPT-2 / 3 → Llama 3` (LayerNorm → RMSNorm, absolute positions → RoPE, GELU → SwiGLU, MHA → GQA), and `Llama 3 → DeepSeek-V3` (GQA → MLA, dense FFN → MoE).

## Self-attention, step by step

Each token emits three vectors: a **query** (what am I looking for?), a **key** (what do I offer?) and a **value** (what do I pass on?). Scores are dot products $q\cdot k$, scaled by $\sqrt{d_k}$, masked so that no token sees the future, and normalised with a softmax. The output is a weighted average of values: $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$.

<!-- widget:tx-attention -->

This is a toy: 6 tokens, $d_k=4$, weights set by hand (not trained) so that every number can be checked in your head. Hover a row to follow one token through the computation; turn the causal mask off and you get a BERT-style encoder. Why the $\sqrt{d_k}$, and what the three projections mean, is in [The Transformer architecture](transformer.en.md).

## Why decoding needs a KV cache

Generation happens one token at a time. Without a cache, every step re-runs attention over the whole prefix: step $t$ recomputes $t$ keys, $t$ values and about $t^2/2$ scores. But under a causal mask, past keys and values never change, so store them: step $t$ then computes one new K/V pair and one row of scores.

<!-- widget:tx-kv-cache -->

Switch to “No cache” and watch again, keeping an eye on the two cumulative bars. The price is memory that grows with every token. Per-step compute drops from $O(t^2)$ to $O(t)$, but decoding becomes memory-bound: every step has to read the entire cache. That is exactly what the next section attacks.

## Shrinking the cache: MHA → GQA → MQA → MLA

Per layer and per token, the cache holds

$$
2 \times n_{\text{kv heads}} \times d_{\text{head}}
$$

numbers, times layers, tokens and bytes per number. The number of query heads does not appear, so the savings all come from the K/V side:

- **MQA** (2019): keep every query head, share a single K/V head. Smallest cache, some quality loss.
- **GQA** (2023): groups of query heads share one K/V head, the middle ground between MHA and MQA. Llama 3, Mistral, Qwen and Gemma all use it.
- **MLA** (DeepSeek-V2, 2024): cache one small latent vector per token and re-expand it into per-head keys and values. The cache is $4.5\,d_{\text{head}}$ regardless of the number of heads, what GQA would need with 2.25 KV heads.

<!-- widget:tx-kv-heads -->

The lower half is a calculator: pick a model shape, drag the context length and the number of concurrent sequences, and read off the memory for all four variants. Switch the shape to DeepSeek-V3 (128 heads) and compare the MHA and MLA rows.

## Not every token needs to see every token

Full causal attention costs $O(n^2)$. **Sliding-window attention** lets each token see only the last $w$ tokens: $O(n\,w)$ compute and a cache capped at $w$. Information still travels further, because every extra layer extends the reach by another window: about $L\,(w-1)+1$ tokens after $L$ layers.

<!-- widget:tx-windows -->

Drag “Stacked layers” and watch the pale region grow. The other three modes are real models: Gemma 3 interleaves five local layers with one global layer, and only the global layers keep a full-length cache; gpt-oss alternates windowed and full layers; StreamingLLM keeps the first few sink tokens forever, because softmax has to put its weight somewhere and models learn to dump the excess there.

## RoPE: position as rotation

Instead of adding a position vector to the embedding, RoPE rotates each pair of dimensions in $q$ and $k$ by an angle proportional to the token's position, with every pair turning at its own frequency. The dot product of two rotated vectors depends only on the **difference** of their angles, so the score depends only on the relative distance $m-n$.

<!-- widget:tx-rope -->

Press play: both vectors spin and the score does not move. That is what “relative position” means. Then drag the distance $\Delta$, or scrub along the curve on the right, and watch the score decay with distance. A larger base $\theta$ turns more slowly and decays more slowly, which is how Llama 3 ($\theta=500\text{k}$) and Gemma 3 ($\theta=1\text{M}$ on global layers) stretch to long context. The derivation is in [The Transformer architecture · RoPE](transformer.en.md).

## MoE: more parameters, same compute per token

The FFN holds roughly two thirds of a dense block's parameters. A mixture-of-experts layer replaces it with $N$ expert FFNs and a router that sends each token to only the top $k$. Total parameters scale with $N$; compute scales with $k$.

<!-- widget:tx-moe -->

The grids are drawn at true scale (8, 128 or 256 experts), so the sparsity you see is the real sparsity. DeepSeek-V3 adds one shared expert that every token visits. The router in the figure is random and only there to show how load spreads; a real router is a learned linear layer, kept balanced by an auxiliary loss or a bias term. The full series is in [MoE](moe/README.en.md).

## FlashAttention: same math, less memory traffic

On a GPU, attention is limited by memory traffic rather than arithmetic: the standard implementation writes the full $n\times n$ score matrix to slow HBM and reads it back. FlashAttention cuts Q, K and V into blocks that fit in fast on-chip SRAM, computes attention block by block while keeping only a running max and normaliser per row (the online softmax), and only ever writes the $n\times d$ output. The result is **exact**, not an approximation, with extra memory linear in $n$.

<!-- widget:tx-flash -->

Switch to “Standard attention” and watch the two $n\times n$ matrices S and P being written out in full; switch back and notice the upper-right blocks that the causal mask lets it skip entirely.

## Families at a glance

The same checklist, side by side. Numbers come from each family's paper, tech report and released config.

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
    <li>The KV cache cuts per-step compute from $O(t^2)$ to $O(t)$. Why does decoding become memory-bound as a result?</li>
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
