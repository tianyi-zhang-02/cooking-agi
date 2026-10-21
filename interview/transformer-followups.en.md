# Transformer interview questions: a quick review

[中文](transformer-followups.md) · **English**

> Reading time: ~8 min · Last reviewed: 2026-10

> **Read this first**: only public fundamentals are collected here. These are not any company's interview questions, and nothing here describes any company's process. Model configurations change quickly; for numbers, trust the papers and released configs at the end of [the interactive Transformer page](../00-foundations/transformer-lab.en.md).

When reviewing Transformers, go beyond the attention formula. Why does decoding use a KV cache? How do MQA / GQA / MLA change that cache? What problems do RoPE, MoE, and FlashAttention address?

Use this page to **review for an interview**: recall the mechanisms and practice explaining them at a whiteboard. These short answers do not replace the full derivations. For those, along with interactive figures, read [The Transformer, interactively](../00-foundations/transformer-lab.en.md) and [The Transformer architecture](../00-foundations/transformer.en.md). Each section below ends with the matching figure.

## KV cache

<details class="qa" markdown="1">
<summary>Why does decoding need a KV cache? What happens without one?</summary>

Generation happens one token at a time. In a causal decoder, earlier positions cannot attend to tokens added later, so their keys and values can be reused. This assumes the same inference setup: fixed weights, unchanged positional and attention settings, and dropout and other randomness disabled.

Without a cache, step $t$ recomputes $t$ keys, $t$ values and about $t^2/2$ scores; with a cache it computes one new K/V pair and one row of scores.

</details>

<details class="qa" markdown="1">
<summary>With a cache, what is the per-step complexity, and what becomes the bottleneck?</summary>

Per-step attention drops from $O(t^2)$ to $O(t)$. The cost is memory that grows linearly with tokens, and every step has to read the entire cache. Small-batch decoding is usually limited by the bandwidth for reading weights and the cache (**memory-bound**). For why a larger batch moves it toward compute-bound, see the [decode roofline example](../12-infra/performance.en.md#decode-batching). GQA, MLA, and sliding windows, discussed below, all reduce cache storage and read costs.

</details>

<details class="qa" markdown="1">
<summary>How do you compute the size of the cache?</summary>

$2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{tokens} \times \text{bytes}$. Note that the number of query heads does not appear. A Llama-2-7B shape (32 layers, 32 KV heads, $d_{\text{head}}=128$, fp16) is 0.5 MiB per token ($2\times32\times32\times128\times2$ bytes), so 16 GiB at 32K (32,768) tokens.

</details>

<details class="qa" markdown="1">
<summary>How do you prove a cached implementation is correct?</summary>

Fix the inputs, weights and inference settings, disable dropout, then compare the cached path's logits position by position with a full forward pass. They should agree within a tolerance appropriate to the precision; different kernels need not produce bitwise-identical results. The easy mistakes are the mask and the positions: a query's absolute position is `cache.pos + i`, the mask is a non-square $(T, S)$, and the RoPE cos/sin must be sliced starting at `cache.pos`.

</details>

Figure: [KV cache: what each decoding step computes](../00-foundations/transformer-lab.en.md#tx-kv-cache)

## MHA → GQA → MQA → MLA

<details class="qa" markdown="1">
<summary>What do MQA and GQA change, and why touch only K/V and not Q?</summary>

Holding other dimensions fixed, cache size scales with KV heads rather than query heads, so changing K/V alone can save cache memory. **MQA**: all query heads share a single K/V head, giving the smallest cache among these KV-sharing arrangements, potentially at some quality cost. The [original paper](https://arxiv.org/abs/1911.02150) reports a small degradation on its tasks, not an inevitable loss for every model. **GQA**: groups of query heads share one K/V head, the middle ground; Llama 3 8B has 32 query heads and 8 KV heads.

</details>

<details class="qa" markdown="1">
<summary>What does MLA cache, and why is its cache independent of the number of heads?</summary>

MLA caches one low-dimensional latent vector per token rather than per-head K/V. Structurally, an up-projection can recover each head's K/V. During inference, the content-side up-projections can instead be absorbed into the query and output projections, avoiding expansion of all past K/V. [DeepSeek-V2](https://arxiv.org/abs/2405.04434) uses a latent of $4\,d_{\text{head}}$ plus a $d_{\text{head}}/2$ RoPE key, $4.5\,d_{\text{head}}$ in total: what GQA would need with 2.25 KV heads, while each head retains its own K/V projection.

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

Softmax weights must sum to 1. Some heads assign substantial weight to the first few tokens even when their content has little relevance to the current prediction: one form of attention sink. Removing those tokens changes the attention allocation and can severely hurt streaming generation, so StreamingLLM retains them. The [gpt-oss implementation](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py) instead adds a learned sink logit per head to the softmax, without a corresponding value. The denominator receives the exponential of that logit, not a bias added directly.

</details>

Figure: [Who can see whom](../00-foundations/transformer-lab.en.md#tx-windows)

## RoPE

<details class="qa" markdown="1">
<summary>How does RoPE represent position, and why is it called “relative”?</summary>

Each pair of dimensions in $q$ and $k$ is rotated by an angle proportional to position, every pair at its own frequency. For fixed original $q$ and $k$, the positional dependence introduced by these rotations depends on their angle difference, expressible through relative distance $m-n$. The score still depends on the content of $q$ and $k$, not just how far apart the tokens are.

</details>

<details class="qa" markdown="1">
<summary>Why must both Q and K be rotated? Would rotating one be enough?</summary>

No. The relativity comes from the two rotations canceling: $\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle$. Rotate only one side and what is left in the dot product is absolute position.

</details>

<details class="qa" markdown="1">
<summary>Why raise the RoPE base θ for long context?</summary>

A larger base slows every pair except the first (whose frequency is $b^0=1$), so low-frequency pairs do not wrap around within a longer range and distant positions are less likely to alias. Changing the base alone does not guarantee good long-context behavior; training and evaluation still matter. Llama 3 uses $\theta = 500\text{k}$; Gemma 3 uses $\theta = 1\text{M}$ on its global layers.

</details>

Figure: [RoPE: position as rotation](../00-foundations/transformer-lab.en.md#tx-rope)

## MoE

<details class="qa" markdown="1">
<summary>What do total and active parameters each scale with in an MoE?</summary>

The FFN is replaced by $N$ experts, and a router sends each token to only the top $k$. At a fixed expert size, total expert parameters scale with $N$, while routed-expert compute per token scales with $k$. Total cost also includes the router, shared modules and communication, so $k$ alone is not enough. Mixtral is 8 choose 2 (about 47B total, about 13B active); DeepSeek-V3 routes to 8 of 256 experts, plus one shared expert that every token visits (671B total, 37B active).

</details>

<details class="qa" markdown="1">
<summary>What is the classic problem in MoE training?</summary>

Load imbalance: the router tends to send tokens to a few experts and the rest learn nothing. The usual fix is an auxiliary load-balancing loss; DeepSeek-V3 relies mainly on a bias term instead, keeping only a very small sequence-wise auxiliary loss.

</details>

Figure: [MoE: routing drawn at the true number of experts](../00-foundations/transformer-lab.en.md#tx-moe)

## FlashAttention

<details class="qa" markdown="1">
<summary>Is FlashAttention an approximation? What does it actually save?</summary>

Standard dense FlashAttention does not approximate the attention formula. “Exact” means mathematical equivalence, not bitwise agreement between floating-point implementations. Its main saving is memory traffic: the standard implementation writes the full $n\times n$ score matrix to slow HBM and reads it back. FlashAttention processes Q, K and V in blocks in on-chip SRAM, saving the $n\times d$ output and normalization statistics needed for backward, but not the full score matrix. Extra memory is linear in $n$. The [original paper](https://arxiv.org/abs/2205.14135) also trades extra arithmetic in backward recomputation for fewer memory accesses. For how far apart HBM and on-chip SRAM are in capacity and bandwidth, see [the GPU memory hierarchy](../12-infra/gpu-architecture.en.md).

</details>

<details class="qa" markdown="1">
<summary>How is softmax computed once the row is split into blocks? Does a row not have to be normalized as a whole?</summary>

Use online softmax. Each row maintains a running maximum and a normalization factor. Each new block updates both quantities and adjusts the output accumulated so far. Mathematically, this is identical to normalizing the whole row at once.

</details>

Figure: [FlashAttention: where the attention matrix lives](../00-foundations/transformer-lab.en.md#tx-flash)

## Structure and model families

<details class="qa" markdown="1">
<summary>Which parts of the block changed between the original Transformer and Llama 3?</summary>

Four: the norm went from post-LN LayerNorm to pre-norm RMSNorm; position went from absolute encodings added to the embedding to RoPE; the FFN activation went from the original ReLU to SwiGLU (GELU appears in other later models); attention went from MHA to GQA. On top of that, the encoder and cross-attention are gone, and so are the biases.

</details>

<details class="qa" markdown="1">
<summary>What is the difference between pre-norm and post-norm?</summary>

Post-norm puts the norm after the residual add, so the gradient passes through a norm at every layer. Pre-norm moves it inside the residual branch, leaving an unscaled identity path along the trunk. [Xiong et al.](https://arxiv.org/abs/2002.04745) analyze large gradients near the output at initialization in the original post-LN setup and explain why warmup helps. This does not mean every post-norm model requires warmup or every pre-norm model can omit it; initialization, depth and optimization settings matter too.

</details>

Figure: [Architecture map: morphing between model families](../00-foundations/transformer-lab.en.md#tx-arch)

## Keep going

- [Quick review: run through the questions, block by block](questions.en.md)
- [The Transformer, interactively](../00-foundations/transformer-lab.en.md): the same material, in its learning form
