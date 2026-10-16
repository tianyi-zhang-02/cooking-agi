# FlashAttention and PagedAttention: what does each save?

[中文](attention-kernels.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [multi-head attention](../core/multi-head-attention.en.md), [KV cache](kv-cache-and-inference.en.md)

The names are similar, but the problems differ. **FlashAttention addresses data movement during attention computation. PagedAttention addresses KV storage for concurrent requests.** One mainly changes computation and IO; the other mainly changes cache organization.

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

### Online does not mean probabilities are final immediately {#online-output}

After the first two positions, their probabilities are $[1/3,2/3]$. Once the third arrives, those entries become $[1/6,2/6]$. The denominator is not final while scores remain unread.

Online softmax can maintain the maximum and normalizer in one pass, **but cannot emit each final probability as soon as it reads the corresponding score**. Algorithm 3 in the [2018 paper](https://arxiv.org/html/1805.02867v2#S3) still has an output stage. Returning every probability requires rereading inputs or retaining the required data. Attention can instead accumulate the weighted numerator $u$ alongside the denominator, then return only $u/\ell$. That is why this example does not need to store a probability row.

<details markdown="1">
<summary>How do we combine two independently processed blocks?</summary>

Blocks A and B hold $(m_A,\ell_A,u_A)$ and $(m_B,\ell_B,u_B)$. Choose a shared maximum $m=\max(m_A,m_B)$ and rescale both:

$$
\begin{aligned}
\ell&=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B,\\
u&=e^{m_A-m}u_A+e^{m_B-m}u_B.
\end{aligned}
$$

The first two positions in our example have weighted output $50/3$; the third alone gives zero. Averaging those outputs happens to give the correct $25/3$, because each block has the same unnormalized weight sum of 3. **That coincidence is not an algorithm.** Change the scores to $[\log1,\log1,\log8]$, keeping values $[10,20,0]$: averaging block outputs gives 7.5, but the correct result is 3. Merging the statistics preserves the blocks' different weights.

In real arithmetic, these statistics can be merged in different orders; floating-point sums can still differ through rounding. Skip entirely masked blocks to avoid $e^{-\infty-(-\infty)}$. An entirely masked row has no distribution to normalize.

</details>

<details markdown="1">
<summary>What does one fewer scan save?</summary>

In the paper's Algorithms 2 and 3, safe softmax scans for the maximum, the denominator, and the outputs: three reads and one write per element. The online normalizer combines the first two scans, giving two reads and one write. These are algorithmic memory accesses, **not CPU–GPU round trips**.

Both algorithms remain $O(n)$. Keeping data in registers or shared memory, or fusing the next operation, changes actual HBM traffic and runtime. Scan counts alone do not determine a speedup.

</details>

## Verify the arithmetic before writing a GPU kernel

This single-row, scalar-value implementation tests the mathematics, not GPU performance. Negative infinity marks masked positions that can be skipped. NaN, positive infinity, and nonfinite values at visible positions raise errors, as does an entirely invalid row, rather than silently returning NaN.

```python
import math

def streaming_attention(scores, values, block_size):
    if (len(scores) != len(values) or not isinstance(block_size, int)
            or isinstance(block_size, bool) or block_size < 1):
        raise ValueError("Invalid shape or block size")
    if any(score != -math.inf and not math.isfinite(score) for score in scores):
        raise ValueError("Scores must be finite or -inf for masked positions")
    if any(not math.isfinite(value) for score, value in zip(scores, values)
           if score != -math.inf):
        raise ValueError("Visible values must be finite")
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

[FlashAttention](https://arxiv.org/abs/2205.14135) combines tiled computation with these statistics, keeping blocks in faster on-chip memory where possible instead of repeatedly materializing the full intermediates in HBM. Backward can recompute selected quantities. For how on-chip memory and HBM differ in capacity and bandwidth, see [the memory hierarchy in GPU architecture](../../12-infra/gpu-architecture.en.md); for loading a tile into shared memory and reusing it, see [CUDA tiling](../../12-infra/cuda.en.md#tiling).

Dense attention still computes the permitted query–key pairs; the pairing count generally remains quadratic. Mathematically exact attention also does not promise bitwise agreement across operation orders.

Sequence length, masks, head dimensions, and hardware affect the benefit. Compare outputs and gradients, and benchmark prefill and decode separately. Kernel speedup is not automatically end-to-end speedup.

## What remains for FA2 through FA4 to improve?

Avoiding the full matrix does not guarantee a fully utilized GPU. Work partitioning and overlap between operations still matter.

| Version | Main change | Not a claim that… |
| --- | --- | --- |
| FA1 | Tiling and online softmax reduce intermediate HBM traffic | Some keys are ignored |
| FA2 | Less non-matmul work and better thread-block / warp partitioning | The attention objective is approximated |
| FA3 | Hopper-oriented overlap of asynchronous movement, matrix multiplication, and softmax; an additional FP8 path | Every GPU benefits equally, or FP8 is mandatory |
| FA4 | Blackwell-oriented (B200 / GB200): fully asynchronous matmul pipelines and larger tiles; software-emulated exp and rescaling only when needed; tensor memory and 2-CTA matmul in backward to cut shared-memory traffic | Hopper users should switch too, or the authors' speedups hold for your shapes |

One [FA2](https://arxiv.org/abs/2307.08691) change assigns different query rows to different warps, reducing exchanges of partial results. This inter-warp communication involves **on-chip shared memory**, not HBM; for where each memory space lives and who can see it, see [CUDA memory spaces](../../12-infra/cuda.en.md#memory-spaces).

[FA3](https://arxiv.org/abs/2407.08608) exploits Hopper's asynchronous execution (asynchronous data movement such as TMA; see [NVIDIA generations](../../12-infra/nvidia-generations.en.md)) and discusses both FP16 and FP8. FP8 adds quantization error to measure: “exact attention” does not promise error-free arithmetic.

[FA4](https://arxiv.org/abs/2603.05451) (March 2026) targets Blackwell. That generation doubles Tensor Core throughput, but shared-memory bandwidth and the exponential units did not keep pace, so the bottleneck moves back to work outside the matmuls; FA4 mainly reduces exp, rescaling, and shared-memory traffic. The authors report up to 1.3× over cuDNN 9.13 and 2.7× over Triton in BF16 on B200. These are the authors' own kernel numbers, not end-to-end results. The kernels are written entirely in CuTe-DSL, embedded in Python.

Before adopting any version, check hardware, shape, and dtype support, then benchmark the end-to-end gain.

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

Unused slots inside allocated tail blocks are **internal fragmentation**. **External fragmentation** occurs when enough free space exists in total but is scattered into gaps too small for a required contiguous allocation. Because all blocks have the same size, paging eliminates external fragmentation; allocating small blocks on demand also confines internal fragmentation to each request's last partially filled block. Block tables and those tail slots still cost memory.

Separate a paper's sharing mechanisms from the features a framework currently implements. [vLLM's prefix-caching documentation](https://docs.vllm.ai/en/latest/design/prefix_caching/) describes reusing full blocks and including prefixes, token IDs, and adapter / multimodal information in cache keys. Do not assume arbitrary partially filled tails are reusable.

<span id="cache-lifecycle"></span>

## Finishing a request does not immediately erase its cache

In the checked vLLM V1 commit `10cc2f6`, active requests hold references. After the last release, a cached block can enter the free queue yet remain reusable; allocating it for new content removes the old mapping. This is capacity-driven eviction, not a fixed ten-minute TTL. The [pinned block-pool implementation](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/block_pool.py#L729) handles cached and uncached blocks differently, so do not describe every free block as part of one undifferentiated LRU queue.

Separate **releasing a request's reference, invalidating a cache hit, and returning GPU memory to the system**. A preallocated pool may retain memory for future requests. For how cancellation releases these references, see the [request lifecycle](../../06-systems/llm-serving.en.md#request-lifecycle).

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

For their index structures (block hashes versus a radix tree), default granularity, and eviction policies, see the comparison in [inference frameworks](../../06-systems/inference-frameworks.en.md#prefix). Splittable tree nodes do not guarantee tokenwise reuse in every configuration. Here we only run the page-alignment arithmetic on the seven-token common prefix above. At SGLang `436d73d`, ordinary models use [UnifiedRadixCache](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/mem_cache/registry.py#L142), and the page size [defaults to 1](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/arg_groups/overrides.py#L1282); only some backends or models raise it. When `page_size > 1`, [matching aligns to pages first](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L840). A seven-token common prefix keeps seven positions with page size one, but aligns down to four with page size four. With vLLM configured at its base block-size default of 16, the same prefix fills no complete block, so nothing is reused. Configuration changes reuse granularity; this is not a throughput comparison.

Avoid blanket rules such as “vLLM for concurrency, SGLang for agents.” Replay your workload with fixed models and budgets; changing a backend within one framework can matter more than switching frameworks. For the conditions to check on one request trace and the latency metrics to report, see the [benchmark section of inference frameworks](../../06-systems/inference-frameworks.en.md#benchmark). These are mechanism and public-implementation checks, not a serving benchmark we ran.

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
