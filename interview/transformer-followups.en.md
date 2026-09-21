# Transformer follow-ups, quick review: the ones you will be asked

[中文](transformer-followups.md) · **English**

> Reading time: ~8 min · Last reviewed: 2026-09

> **Read this first**: only public fundamentals are collected here. These are not any company's interview questions, and nothing here describes any company's process. Model configurations change quickly; for numbers, trust the papers and released configs at the end of [the interactive Transformer page](../00-foundations/transformer-lab.en.md).

When a Transformer comes up, few interviewers stop at “what is the attention formula”. The gap opens in the follow-ups: the KV cache, MQA / GQA / MLA, RoPE, MoE, FlashAttention.

This page is for a **run-through before an interview**: one question, one answer, compressed to what you can say in two sentences at a whiteboard. To actually learn it, with derivations and figures you can drag, go to the knowledge-block side: [The Transformer, interactively](../00-foundations/transformer-lab.en.md) and [The Transformer architecture](../00-foundations/transformer.en.md). Each section below ends with the matching figure.

## KV cache

<details class="qa" markdown="1">
<summary>Why does decoding need a KV cache? What happens without one?</summary>

Generation happens one token at a time. Under a causal mask the K and V of past tokens never change, so recomputing them every step is wasted work. Without a cache, step $t$ recomputes $t$ keys, $t$ values and about $t^2/2$ scores; with a cache it computes one new K/V pair and one row of scores.

</details>

<details class="qa" markdown="1">
<summary>With a cache, what is the per-step complexity, and what becomes the bottleneck?</summary>

Per-step attention drops from $O(t^2)$ to $O(t)$. The price is memory that grows linearly with tokens, and every step has to read the entire cache: decoding goes from compute-bound to **memory-bound**. Everything that “shrinks the cache” afterwards (GQA, MLA, sliding windows) is aimed at exactly this.

</details>

<details class="qa" markdown="1">
<summary>How do you compute the size of the cache?</summary>

$2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{tokens} \times \text{bytes}$. Note that the number of query heads does not appear. A Llama-2-7B shape (32 layers, 32 KV heads, $d_{\text{head}}=128$, fp16) is 0.5 MB per token, so 16 GB at 32K context.

</details>

<details class="qa" markdown="1">
<summary>How do you prove a cached implementation is correct?</summary>

The logits produced position by position on the cached path must be **numerically equivalent** to a full forward pass without a cache. The easy mistakes are the mask and the positions: a query's absolute position is `cache.pos + i`, the mask is a non-square $(T, S)$, and the RoPE cos/sin must be sliced starting at `cache.pos`.

</details>

Figure: [KV cache: what each decoding step computes](../00-foundations/transformer-lab.en.md#tx-kv-cache)

## MHA → GQA → MQA → MLA

<details class="qa" markdown="1">
<summary>What do MQA and GQA change, and why touch only K/V and not Q?</summary>

Cache size depends only on the number of KV heads, not on the number of query heads, so memory is saved entirely on the K/V side. **MQA**: all query heads share a single K/V head; the smallest cache, with some quality loss. **GQA**: groups of query heads share one K/V head, the middle ground; Llama 3 8B has 32 query heads and 8 KV heads.

</details>

<details class="qa" markdown="1">
<summary>What does MLA cache, and why is its cache independent of the number of heads?</summary>

MLA does not cache per-head K and V. It caches one low-dimensional latent vector per token and expands it, through an up-projection, into each head's own K and V when needed. DeepSeek's setting is a latent of $4\,d_{\text{head}}$ plus a $d_{\text{head}}/2$ RoPE key, $4.5\,d_{\text{head}}$ in total: what GQA would need with 2.25 KV heads, except that every head still gets its own K/V.

</details>

<details class="qa" markdown="1">
<summary>Why is RoPE handled separately in MLA?</summary>

RoPE is a position-dependent rotation, so it cannot be absorbed into a position-independent low-rank projection. MLA therefore carries position on a separate, uncompressed small key (64 dims) and sends the rest of the content through the latent.

</details>

Figure: [MHA → GQA → MQA → MLA, with a memory calculator](../00-foundations/transformer-lab.en.md#tx-kv-heads)

## Sliding windows and attention sinks

<details class="qa" markdown="1">
<summary>With a sliding window each layer sees only w tokens. How does distant information arrive?</summary>

Through depth. Every extra layer extends the reach by another window, roughly $L\,(w-1)+1$ tokens after $L$ layers. Mistral 7B uses $w=4096$ over 32 layers, a theoretical reach of about 131K.

</details>

<details class="qa" markdown="1">
<summary>What does it save?</summary>

Compute goes from $O(n^2)$ to $O(n\,w)$, and the per-layer KV cache is capped at $w$. Gemma 3 interleaves five local layers with one global layer, and only the global layers need a full-length cache.

</details>

<details class="qa" markdown="1">
<summary>What is an attention sink, and why does evicting the first tokens from the cache break things?</summary>

Softmax weights must sum to 1. When a head has “nothing to look at”, the model learns to dump the excess weight on the first few tokens. Evict them and that weight has nowhere to go, so the distribution falls apart. StreamingLLM keeps the first few tokens forever; gpt-oss adds a learned bias to the softmax denominator, which plays the same role.

</details>

Figure: [Who can see whom](../00-foundations/transformer-lab.en.md#tx-windows)

## RoPE

<details class="qa" markdown="1">
<summary>How does RoPE represent position, and why is it called “relative”?</summary>

Each pair of dimensions in $q$ and $k$ is rotated by an angle proportional to the position, every pair at its own frequency. The dot product of two rotated vectors depends only on the difference of their angles, so the score depends only on the relative distance $m-n$.

</details>

<details class="qa" markdown="1">
<summary>Why must both Q and K be rotated? Would rotating one be enough?</summary>

No. The relativity comes from the two rotations cancelling: $\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle$. Rotate only one side and what is left in the dot product is absolute position.

</details>

<details class="qa" markdown="1">
<summary>Why raise the RoPE base θ for long context?</summary>

A larger base makes every pair turn more slowly, so the score decays more slowly with distance and far-away positions stay distinguishable. Llama 3 uses $\theta = 500\text{k}$; Gemma 3 uses $\theta = 1\text{M}$ on its global layers.

</details>

Figure: [RoPE: position as rotation](../00-foundations/transformer-lab.en.md#tx-rope)

## MoE

<details class="qa" markdown="1">
<summary>What do total and active parameters each scale with in an MoE?</summary>

The FFN is replaced by $N$ experts, and a router sends each token to only the top $k$. Total parameters scale with $N$; compute per token scales only with $k$. Mixtral is 8 choose 2 (about 47B total, about 13B active); DeepSeek-V3 routes to 8 of 256 experts, plus one shared expert that every token visits (671B total, 37B active).

</details>

<details class="qa" markdown="1">
<summary>What is the classic problem in MoE training?</summary>

Load imbalance: the router tends to send tokens to a few experts and the rest learn nothing. The usual fix is an auxiliary load-balancing loss; DeepSeek-V3 adjusts a bias term instead and adds no auxiliary loss.

</details>

Figure: [MoE: routing drawn at the true number of experts](../00-foundations/transformer-lab.en.md#tx-moe)

## FlashAttention

<details class="qa" markdown="1">
<summary>Is FlashAttention an approximation? What does it actually save?</summary>

It is not an approximation; the result is **exact**. What it saves is memory traffic, not arithmetic: the standard implementation writes the full $n\times n$ score matrix to slow HBM and reads it back. FlashAttention cuts Q, K and V into blocks that fit in on-chip SRAM, computes block by block, and writes back only the $n\times d$ output, with extra memory linear in $n$.

</details>

<details class="qa" markdown="1">
<summary>How is softmax computed once the row is split into blocks? Does a row not have to be normalised as a whole?</summary>

With an online softmax: every row keeps a running max and a running normaliser, and each new block updates both and corrects the output accumulated so far. Mathematically it is identical to normalising the whole row at once.

</details>

Figure: [FlashAttention: where the attention matrix lives](../00-foundations/transformer-lab.en.md#tx-flash)

## Structure and model families

<details class="qa" markdown="1">
<summary>Which parts of the block changed between the original Transformer and Llama 3?</summary>

Four: the norm went from post-LN LayerNorm to pre-norm RMSNorm (deep stacks train more stably); position went from absolute encodings added to the embedding to RoPE; the FFN activation went from ReLU / GELU to SwiGLU; attention went from MHA to GQA. On top of that, the encoder and cross-attention are gone, and so are the biases.

</details>

<details class="qa" markdown="1">
<summary>What is the difference between pre-norm and post-norm?</summary>

Post-norm puts the norm after the residual add, so the gradient passes through a norm at every layer and deep networks need warmup to train stably. Pre-norm moves the norm inside the residual branch, leaving an unscaled identity path along the trunk.

</details>

Figure: [Architecture map: morphing between model families](../00-foundations/transformer-lab.en.md#tx-arch)

## Keep going

- [Quick review: run through the questions, block by block](questions.en.md)
- [The Transformer, interactively](../00-foundations/transformer-lab.en.md): the same material, in its learning form
