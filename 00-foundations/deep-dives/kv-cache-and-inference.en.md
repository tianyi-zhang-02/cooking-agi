# KV cache: what do we stop computing, and start storing?

[中文](kv-cache-and-inference.md) · **English**

> Last reviewed: 2026-10 · Prerequisites: [Multi-head attention](../core/multi-head-attention.en.md), [Decoder-only](../core/decoder-only.en.md)

After generating one token, why should a model read the whole conversation again? It does not need to recompute everything. Some earlier calculations can be kept: that is the starting point for a KV cache.

“Caching makes it fast” is not enough, though. Long conversations consume more memory, and concurrent requests can move the bottleneck elsewhere. Follow a short generation to separate computation, storage, and the optimizations that address each.

## Start with three input tokens

Suppose a prompt is tokenized as `[A, B, C]`. These letters are placeholders, not actual tokenizer output. The model processes the three positions in parallel and uses the output at C to predict D.

| Phase | Tokens supplied this time | Per-layer cache after processing | Prediction |
| --- | --- | --- | --- |
| Prefill | A, B, C | K/V for A, B, C | C's output predicts D |
| First decode step | D | K/V for A, B, C, D | D's output predicts E |
| Second decode step | E | K/V for A, B, C, D, E | E's output predicts F |

Immediately after sampling D, its K/V have not been computed. They appear when D is fed back through the model. Reversing this order can make an implementation process a token twice or skip a position.

## Why can earlier positions stay unchanged?

In a standard causal decoder, A attends only to A; B attends only to A and B. Appending D does not change what A, B, or C could see. With fixed weights, positional encoding, and prefix, inference can reuse their per-layer K/V. An ordinary bidirectional encoder does not satisfy this condition: appending a token can change previous representations. The [Transformers caching explanation](https://huggingface.co/docs/transformers/cache_explanation) develops this causal argument.

At one layer, for the new position $t$:

$$
\begin{aligned}
K_{\le t}&=[K_{<t};k_t],\\
V_{\le t}&=[V_{<t};v_t].
\end{aligned}
$$

$$
o_t=\operatorname{softmax}\left(\frac{q_tK_{\le t}^{\top}}{\sqrt{d_h}}\right)V_{\le t}.
$$

The query is only the current $q_t$, but keys and values include all visible history. Earlier queries are not needed for this new attention row, so they usually need not be cached.

This is not an answer cache, and it does not skip the current token's FFN or remaining layers. **It trades storing per-layer K/V for avoiding repeated computation at old positions.**

## Cheaper steps do not make long context free

A single query head attending over history of length $t$ still computes roughly $t$ matching scores. It avoids rebuilding the entire $t\times t$ matrix for old queries, but the cost per step is not constant.

Starting from a prompt of length $P$ and processing $N$ more tokens, the cumulative query–key match count per head is approximately:

$$
\sum_{j=1}^{N}(P+j)=NP+\frac{N(N+1)}2.
$$

This excludes prefill, projections, FFNs, and hardware parallelism. It simply shows that history grows during generation: caching does not remove the cost of attending to that history.

## Calculate the memory yourself

For a decoder with identical layer structures, unsharded caches, equal sequence lengths, and no window eviction or cache quantization:

$$
M_{KV}=2\,B\,L\,T\,H_{KV}\,d_h\,b.
$$

| Symbol | Meaning |
| --- | --- |
| 2 | One copy each for keys and values |
| $B$ | Number of cached sequences |
| $L$ | Number of layers |
| $T$ | Cached length per sequence |
| $H_{KV}$ | KV heads per layer, not necessarily query heads |
| $d_h$ | Head dimension |
| $b$ | Bytes per element |

Take a hypothetical model with 32 layers, 32 query heads, 8 KV heads, head dimension 128, 8192 cached tokens, and BF16 at 2 bytes per element.

```python
def kv_bytes(batch, layers, tokens, kv_heads, head_dim, bytes_per_value=2):
    return 2 * batch * layers * tokens * kv_heads * head_dim * bytes_per_value

one_sequence = kv_bytes(1, 32, 8192, 8, 128)
eight_sequences = kv_bytes(8, 32, 8192, 8, 128)
print(one_sequence / 2**30, eight_sequences / 2**30)
```

The results are **1 GiB and 8 GiB**. These are raw KV sizes, excluding weights, workspaces, and allocator headroom. With 32 KV heads, one sequence would require 4 GiB. For unequal lengths, sum over actual sequence lengths; preallocation, padding, sharding, or replication can change physical usage.

## The weights fit. Why does the first batch run out of memory? {#inference-budget}

“A 7B model in BF16 is about 14 GB. Shouldn't a 24 GiB device be enough?” It may be enough to load weights. Requests also need KV state, temporary tensors, and kernel workspaces. Longer inputs and higher concurrency expose that difference.

Write a budget before choosing the GPU count. Suppose the model has exactly seven billion BF16 parameters and the device exposes 24 GiB. For this calculation, **assume** 2 GiB for runtime allocations and another 2 GiB of headroom:

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>Weights</small><strong>About 13.04 GiB</strong><span>Seven billion × 2 bytes = 14 GB. Keep GB and GiB conversions consistent.</span></li>
<li><small>Runtime budget</small><strong>2 GiB</strong><span>A placeholder for activations, workspaces, graph capture, and related allocations. Measure the real value.</span></li>
<li><small>Headroom</small><strong>2 GiB</strong><span>Capacity not promised to requests, rather than necessarily allocated tensors.</span></li>
<li><small>Left for KV</small><strong>About 6.96 GiB</strong><span>24 − 13.04 − 2 − 2. Paging and the model's state layout still matter.</span></li>
</ol>
<figcaption>A hypothetical capacity budget, not a measured footprint for a particular 7B model or GPU.</figcaption>
</figure>

If its KV configuration is the earlier 32-layer, eight-KV-head example, each independent 8192-position cache takes 1 GiB. The budget accommodates at most six, not eight. At 16384 positions, each takes 2 GiB and only three fit. Length counts **prompt and generated positions already processed by the model**; prompt length alone understates the cost of a long answer.

The 2 GiB runtime value is an input to this example, not a constant for every model. Measure with the intended model, batch, prefill chunks, and graph settings. Weight quantization need not quantize KV, and scales, zero-points, and unquantized modules can add storage. The [vLLM memory configuration guide](https://docs.vllm.ai/en/latest/configuration/conserving_memory/) is a useful deployment checklist.

### Unequal lengths, pages, and shared prefixes {#paged-budget}

The equal-length formula is convenient, but real requests differ. Round each cache length up to its allocation block, then sum. With four-token blocks and lengths 5, 3, and 8, allocation needs 2 + 1 + 2 = 5 blocks: 20 slots for 16 computed positions, leaving four unused slots.

<details markdown="1">
<summary>Try different lengths in a small KV calculator</summary>

```python
def paged_kv_bytes(lengths, layers=32, kv_heads=8, head_dim=128,
                   bytes_per_value=2, block_tokens=16):
    lengths = list(lengths)
    dimensions = (layers, kv_heads, head_dim, bytes_per_value, block_tokens)
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("Dimensions must be positive integers")
    if any(type(length) is not int or length < 0 for length in lengths):
        raise ValueError("Lengths must be nonnegative integers")
    blocks = sum((length + block_tokens - 1) // block_tokens for length in lengths)
    per_token = 2 * layers * kv_heads * head_dim * bytes_per_value
    return blocks * block_tokens * per_token

kv_budget = 24 * 2**30 - 7_000_000_000 * 2 - 4 * 2**30
assert paged_kv_bytes([8192] * 6) <= kv_budget
assert paged_kv_bytes([8192] * 7) > kv_budget
assert paged_kv_bytes([16384] * 3) <= kv_budget
assert paged_kv_bytes([16384] * 4) > kv_budget
```

This counts KV for identical full-attention layers without sharing. It is not a complete GPU-memory predictor.

</details>

Sharing requires counting **unique physical blocks**, not adding logical request lengths. Two eight-position caches with four-token blocks can share the first block and have separate second blocks: 16 logical positions occupy three blocks, or 12 physical slots. Retained, unused prefixes also occupy the cache pool; they are not memory already returned to the system.

Full-attention state commonly grows with sequence length. Sliding-window layers can retain only the state their windows require, while hybrid models need per-layer-type accounting. Do not apply one uniform $L\times T$ formula to every architecture. The [hybrid KV-cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) addresses these distinctions.

### Multiple GPUs do not divide every allocation equally {#kv-per-rank}

Consider just the earlier 1 GiB cache. Assume even partitioning and omit communication buffers:

| Placement | What each rank keeps | KV in this example |
| --- | --- | ---: |
| TP=2, evenly split eight KV heads | Four KV heads at every layer | 0.5 GiB per rank |
| PP=2, evenly split 32 layers | All KV heads for 16 layers | 0.5 GiB per stage |
| DP=2, each replica serves one request | Its own full 32-layer, eight-head cache | 1 GiB per replica, 2 GiB total |

A less obvious case is an MQA model with one KV head. In a TP=8 implementation that replicates KV, each rank may retain that head rather than an eighth of it. Check the model implementation and backend; `tensor_parallel_size` alone does not determine cache placement.

These calculations describe KV, not proportional reductions in every weight or workspace allocation. Deployment feasibility depends on the **most constrained rank**, not the sum of free memory across devices.

## MHA, GQA, MQA: what is shared?

```text
MHA: Q1 → K1,V1    Q2 → K2,V2    Q3 → K3,V3    Q4 → K4,V4
GQA: Q1,Q2 → K1,V1              Q3,Q4 → K2,V2
MQA: Q1,Q2,Q3,Q4 → K1,V1
```

This sketch uses four query heads. GQA and MQA reduce the number of KV groups; each query head still has its own query and attention output. Smaller caches come with a change to the parameter sharing. Deleting KV heads from a trained MHA model is not automatically quality-preserving. The [GQA paper](https://arxiv.org/abs/2305.13245) describes and evaluates an uptraining approach from MHA checkpoints.

Return to the 32-query-head example. Moving from 32 to 8 KV heads leaves **8/32 = 1/4** of the raw KV storage, with layers, length, head dimension, and dtype fixed. A fourfold reduction means one quarter remains, not four times as much. It also does not mean total model memory falls by 75%.

KV head count is an architecture choice, not necessarily the GPU count. Tensor-parallel sharding or replication then affects per-device storage. Teaching code may use `repeat_interleave` to expand shared K/V to the query-head count, but explicitly copying them does not demonstrate the memory saving. An efficient implementation preserves shared storage and supports grouped access in the kernel.

## What do FlashAttention and PagedAttention save?

| Method | Main change | What it does not promise |
| --- | --- | --- |
| KV cache | Reuse earlier K/V | No further reads of history |
| GQA / MQA | Share K/V across query heads | Identical quality to MHA |
| FlashAttention | Tile computation to reduce intermediate attention memory traffic | Linear pair count for full attention |
| PagedAttention | Manage KV storage in blocks to improve fragmentation and sharing | Eliminate KV storage or change the learning objective |

[FlashAttention](https://arxiv.org/abs/2205.14135) computes mathematically equivalent attention without first writing and rereading the entire score/probability matrix in GPU memory. Changed floating-point operation order need not be bitwise identical. [PagedAttention](https://arxiv.org/abs/2309.06180) primarily addresses cache management for multi-request serving. These ideas are not mutually exclusive; support and gains depend on implementation, hardware, and workload.

## Different results with caching: what should you check?

1. **Positions:** the fourth position cannot start at zero again. RoPE positions must continue rather than reset at each decode call.
2. **Masks:** three cached positions plus two new inputs require 2 × 5 attention, not 2 × 2. The first new position must not see the second.
3. **Prefixes:** changing the system prompt, image input, or adapter can invalidate the cache. Identical text under different weights is not an identical computational prefix.
4. **Determinism:** start with `eval()` to disable dropout and compare logits before comparing sampled text. Different random samples do not establish a cache bug.
5. **Lifecycle:** release caches when requests end. Reordering batches must reorder sequence identities and their caches together, never attaching one person's history to another's request.

The useful test is not “can it generate a sentence?” Fix an input and compare corresponding logits from a **full forward pass** against **prefill followed by incremental appends**, within numerical tolerance. The site's [decoder tests](../code/test_model.py) already include cache-equivalence checks.

## Measure these separately

| Measurement | Question it gets closer to |
| --- | --- |
| Time to first token, TTFT | How long before an answer starts, including queueing and prefill? |
| Inter-token latency, ITL | How smoothly does the answer stream after it starts? |
| Throughput | How many requests or tokens can the system process over time? |
| Peak memory and concurrency limit | How many users fit at the specified lengths? |

Hold model, input/output length distributions, concurrency, and hardware fixed. A short-prompt, single-request test is not a substitute for concurrent long conversations. Before optimizing, find out whether time goes into compute, weight movement, KV reads, or queueing. A newer kernel may not address the actual bottleneck.

Return to [attention](../core/multi-head-attention.en.md) for the tensor computation, or continue to [LoRA and QLoRA](../../05-post-training/lora-and-qlora.en.md) for training rather than inference memory.
