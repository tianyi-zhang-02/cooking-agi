# FlashAttention and PagedAttention: what does each save?

[中文](attention-kernels.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [multi-head attention](../core/multi-head-attention.en.md), [KV cache](kv-cache-and-inference.en.md)

The names are similar, but the problems differ. FlashAttention addresses data movement during attention computation. PagedAttention addresses KV storage for concurrent requests. One mainly changes computation and IO; the other mainly changes cache organization.

We will work through one small example of each before discussing when they help.

## One attention row need not store all its probabilities

Fix a query with three key scores $[\log1,\log2,\log3]$ and scalar values $[10,20,0]$. These are convenient teaching numbers; real values are vectors.

$$
\begin{aligned}
o&=\frac{1\cdot10+2\cdot20+3\cdot0}{1+2+3}\\
 &=\frac{50}{6}\approx8.3333.
\end{aligned}
$$

A straightforward implementation materializes scores and softmax probabilities before multiplying by values. Across queries, that can produce $T\times T$ intermediates. We need the final weighted output, not necessarily a full probability matrix in device memory.

## Online softmax rescales old statistics when a new block arrives

Let $m$ be the largest score seen so far. Maintain

$$
\begin{aligned}
\ell&=\sum_j e^{s_j-m},\\
u&=\sum_j e^{s_j-m}v_j.
\end{aligned}
$$

The output is $u/\ell$. For a new block with maximum $m_b$, set $m'=\max(m,m_b)$ and rescale the old statistics by $a=e^{m-m'}$:

$$
\begin{aligned}
\ell'&=a\ell+\sum_{j\in b}e^{s_j-m'},\\
u'&=au+\sum_{j\in b}e^{s_j-m'}v_j.
\end{aligned}
$$

This is not a mean of independently normalized block outputs. Different denominators make that incorrect.

| Processed positions | $m$ | $\ell$ | $u$ |
| --- | --- | ---: | ---: |
| First two | $\log2$ | 1.5 | 25 |
| Include the third | $\log3$ | 2 | $50/3$ |

The second step multiplies old statistics by $2/3$ before adding the new block. The result remains $25/3$. Even though the third value is zero, its contribution to the denominator must remain.

## Verify the arithmetic before writing a GPU kernel

This single-row, scalar-value implementation tests the mathematics, not GPU performance. Masked scores use negative infinity. An entirely invalid row raises an error rather than silently returning NaN.

```python
import math

def streaming_attention(scores, values, block_size):
    if len(scores) != len(values) or block_size < 1:
        raise ValueError("Invalid shape or block size")
    running_max = -math.inf
    normalizer = 0.0
    numerator = 0.0
    for start in range(0, len(scores), block_size):
        pairs = [(score, value) for score, value in
                 zip(scores[start:start + block_size], values[start:start + block_size])
                 if score != -math.inf]
        if not pairs:
            continue
        new_max = max(running_max, max(score for score, value in pairs))
        correction = math.exp(running_max - new_max)
        normalizer *= correction
        numerator *= correction
        for score, value in pairs:
            weight = math.exp(score - new_max)
            normalizer += weight
            numerator += weight * value
        running_max = new_max
    if normalizer == 0:
        raise ValueError("No valid keys")
    return numerator / normalizer

scores = [math.log(1), math.log(2), math.log(3)]
assert math.isclose(streaming_attention(scores, [10, 20, 0], 2), 25 / 3)
```

Change block size to 1 or 3, then add 1,000 to every score. Normalized results should agree within floating-point tolerance. Production implementations also handle vector values, batches, masks, dropout, and backward.

## FlashAttention does not remove every quadratic pairing

[FlashAttention](https://arxiv.org/abs/2205.14135) combines tiled computation with these statistics, keeping blocks in faster on-chip memory where possible instead of repeatedly materializing the full intermediates in HBM. Backward can recompute selected quantities.

Dense attention still computes the permitted query–key pairs; the pairing count generally remains quadratic. Mathematically exact attention also does not promise bitwise agreement across operation orders.

Sequence length, masks, head dimensions, and hardware affect the benefit. Compare outputs and gradients, and benchmark prefill and decode separately. Kernel speedup is not automatically end-to-end speedup.

## What remains for FA2 and FA3 to improve?

Avoiding the full matrix does not guarantee a fully utilized GPU. Work partitioning and overlap between operations still matter.

| Version | Main change | Not a claim that… |
| --- | --- | --- |
| FA1 | Tiling and online softmax reduce intermediate HBM traffic | Some keys are ignored |
| FA2 | Less non-matmul work and better thread-block / warp partitioning | The attention objective is approximated |
| FA3 | Hopper-oriented overlap of asynchronous movement, matrix multiplication, and softmax; an additional FP8 path | Every GPU benefits equally, or FP8 is mandatory |

One [FA2](https://arxiv.org/abs/2307.08691) change assigns different query rows to different warps, reducing exchanges of partial results. This inter-warp communication involves **on-chip shared memory**, not HBM.

[FA3](https://arxiv.org/abs/2407.08608) exploits Hopper's asynchronous execution and discusses both FP16 and FP8. FP8 adds quantization error to measure: “exact attention” does not promise error-free arithmetic. Check hardware, shape, and dtype support before benchmarking the end-to-end gain.

## PagedAttention separates logical order from physical placement

Suppose each physical block stores KV for four tokens. Request A has five tokens and B has three:

```text
A logical block 0 → physical block 7: [A0 A1 A2 A3]
A logical block 1 → physical block 2: [A4 -- -- --]
B logical block 0 → physical block 5: [B0 B1 B2 --]
```

Three blocks allocate 12 slots for eight tokens, leaving four unused tail slots. Paging does not eliminate all waste. It avoids reserving a large contiguous maximum-length allocation for every request.

A can process three more tokens and cache their K/V before another position needs a new block. Completed requests release blocks when reference counts permit. Requests sharing a prefix and compatible computation conditions may share its cache; writing into a shared tail requires copy-on-write or equivalent protection.

The [PagedAttention paper](https://arxiv.org/abs/2309.06180) develops this mapping and sharing approach. Our block IDs are illustrative, not a vLLM API. Cache reuse also depends on weights, adapters, positions, and multimodal inputs—not merely identical text.

Unused slots inside allocated tail blocks are **internal fragmentation**. **External fragmentation** occurs when enough free space exists in total but is scattered into gaps too small for a required contiguous allocation. Paging mainly addresses the latter; block tables and partially filled tails still cost memory.

Separate a paper's sharing mechanisms from the features a framework currently implements. [vLLM's prefix-caching documentation](https://docs.vllm.ai/en/latest/design/prefix_caching/) describes reusing full blocks and including prefixes, token IDs, and adapter / multimodal information in cache keys. Do not assume arbitrary partially filled tails are reusable.

<span id="cache-lifecycle"></span>

## Finishing a request does not immediately erase its cache

In the checked vLLM V1 commit `10cc2f6`, active requests hold references. After the last release, a cached block can enter the free queue yet remain reusable; allocating it for new content removes the old mapping. This is capacity-driven eviction, not a fixed ten-minute TTL. The [pinned block-pool implementation](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/block_pool.py#L729) handles cached and uncached blocks differently, so do not describe every free block as part of one undifferentiated LRU queue.

Separate **releasing a request's reference, invalidating a cache hit, and returning GPU memory to the system**. A preallocated pool may retain memory for future requests.

Follow one computed prefix block:

| Event | Active references | Block state |
| --- | ---: | --- |
| Request A holds it | 1 | In use; unrelated content cannot overwrite it |
| B matches the same prefix | 2 | Shared by two requests, not two copies of KV |
| A finishes | 1 | B still uses it |
| B finishes | 0 | Eligible for reclamation, but still a possible cache hit |
| A new hit or allocation reuses it | 1 | A hit keeps old content; reassignment removes the old mapping first |

For a seven-token common prefix and four-token blocks, the full-block path can reuse at most four tokens, not seven. Changing token two breaks the first block; matching text later cannot be spliced back because its KV was computed under a different prefix. This example uses ordinary full attention. Hybrid cache groups, hash granularity, and physical blocks need not map one-to-one; check the actual configuration.

## vLLM versus SGLang is not “finer means faster”

vLLM uses prefix-aware block hashes; SGLang's RadixAttention organizes shared paths as a tree. Splittable nodes do not guarantee tokenwise reuse in every configuration. In [SGLang RadixCache at `436d73d`](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/mem_cache/radix_cache.py#L358), matching aligns to pages when `page_size > 1`. A seven-token common prefix can retain seven positions on the ordinary page-size-one path, but aligns down to four with page size four. Configuration changes reuse granularity; this is not a throughput comparison.

| Check on the same request trace | Why |
| --- | --- |
| Token IDs, templates, adapters, and images | Similar text need not mean identical computation |
| Replica placement and warm-up | Caches need not be shared across replicas |
| Common-prefix length and block/page size | Logical overlap differs from reusable length |
| Weight changes, tenant isolation, and explicit resets | Correct reuse comes before speed |
| TTFT, output intervals, throughput, and tail latency | High hit rates can coexist with long queues |

Avoid blanket rules such as “vLLM for concurrency, SGLang for agents.” Replay your workload with fixed models and budgets; changing a backend within one framework can matter more than switching frameworks. These are mechanism and public-implementation checks, not a serving benchmark we ran.

## Paging, continuous batching, and chunked prefill are different mechanisms

| Mechanism | Controls | Practical question |
| --- | --- | --- |
| PagedAttention | Where KV is stored | Can a growing request avoid relocating its entire cache? |
| Continuous batching | Which requests run at the next step | Can a new request replace one that just finished? |
| Chunked prefill | How much of a long prompt runs at once | Will a new long prompt stall existing decodes? |

Suppose two requests fit at once, and A, B, C need 4, 1, and 2 more decode steps. To isolate scheduling, **ignore prefill, arrival times, and differences in step duration**.

| Decode round | Wait for a fixed batch to finish | Refill slots each round |
| --- | --- | --- |
| 1 | A + B; B finishes | A + B; B finishes |
| 2 | A | A + C |
| 3 | A | A + C; C finishes |
| 4 | A; A finishes | A; A finishes |
| 5–6 | C until completion | Already finished |

Refilling reduces this ideal example from six rounds to four, **not a measured one-third latency reduction**. Real requests need prefill and compete for bandwidth and cache. Chunking a long prefill can protect decode responsiveness at the cost of a later first token. [vLLM's tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization/) discusses these TTFT / ITL tradeoffs. Prefill chunk size is also distinct from KV block size.

## How do these optimizations fit together?

| Problem | Relevant mechanism | Still measure |
| --- | --- | --- |
| Large prefill attention intermediates | FlashAttention | Peak memory, forward/backward error, time |
| Fragmentation across variable-length requests | PagedAttention | Block utilization, throughput, scheduling overhead |
| Large intrinsic per-request KV state | GQA / MQA / MLA | Quality, cache size, actual kernels |
| Reading fewer history positions | Sparse attention | Missed information, selection overhead |

These mechanisms can be combined, but benefits depend on implementation. Report hardware, dtype, lengths, batch size, and timing boundaries. Diagnose the bottleneck before enabling every named optimization.
