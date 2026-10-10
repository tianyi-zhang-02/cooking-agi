# MLA and sparse attention: storing less or reading less?

[中文](latent-and-sparse-attention.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [KV cache](kv-cache-and-inference.en.md), [RoPE](position-and-context.en.md)

Long histories increase both stored state and the work of reading that state for each generated token. Compressing state and narrowing what is read address different costs.

MLA primarily asks how much state each token leaves behind. Sparse attention asks which positions a query should consult. They may be combined, but are not substitutes.

## Start with a compressed representation

Temporarily omit positions and multiple heads, using column vectors. Compress a hidden state into $c_j=W_{\mathrm{down}}h_j$, then define

$$
k_j=U_Kc_j,\qquad v_j=U_Vc_j.
$$

If $c_j$ is smaller than the combined K/V state, caching it can save space. This is a learned parameterization, not lossless compression of arbitrary pretrained K/V tensors. Latent dimension constrains the expressible transformations.

For $c=[2,-1]$, let

$$
\begin{gathered}
U_K=\begin{bmatrix}1&0\\0&2\\1&1\end{bmatrix},\\
k=[2,-2,1]^\top,\\
q=[1,2,-1]^\top.
\end{gathered}
$$

The direct dot product is $q^\top k=-3$. Alternatively, compute $U_K^\top q=[0,3]^\top$ and dot it with $c$, again obtaining -3. This score does not require reconstructing the three-dimensional key first.

## Matrix absorption reorders linear computation

For one head's content score,

$$
q^\top k_j=q^\top U_Kc_j=(U_K^\top q)^\top c_j.
$$

Weighted values similarly satisfy

$$
\sum_j\alpha_jU_Vc_j
=U_V\left(\sum_j\alpha_jc_j\right).
$$

Aggregate in latent space, then project out. Heads retain their own projections and attention weights; they do not become one shared probability distribution.

```python
latent = [2.0, -1.0]
key_projection = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
query = [1.0, 2.0, -1.0]
key = [sum(weight * value for weight, value in zip(row, latent))
       for row in key_projection]
absorbed_query = [sum(row[column] * query_value
                      for row, query_value in zip(key_projection, query))
                  for column in range(len(latent))]
direct_score = sum(first * second for first, second in zip(query, key))
latent_score = sum(first * second for first, second in zip(absorbed_query, latent))
assert direct_score == latent_score == -3.0
```

This verifies a linear identity, omitting softmax scaling and positional branches; it is not a full MLA implementation. [DeepSeek-V2](https://arxiv.org/abs/2405.04434) presents joint KV compression and the inference projections.

## Why separate the RoPE branch?

<span id="rope-absorption"></span>

MLA can use RoPE. The problem is narrower: **rotating the entire expanded key gets in the way of the cheap reordering above**. Attention still works, but one compressed query may no longer suffice for every history position.

Let $R_t$ rotate the query at position $t$ and $R_j$ rotate the key at history position $j$. Their dot product is

$$
(R_tq_t)^\top R_jU_Kc_j
=q_t^\top R_t^\top R_jU_Kc_j.
$$

Without rotation, we compute $U_K^\top q_t$ once. Moving the full expression onto the query now gives $U_K^\top R_j^\top R_tq_t$, which depends on $j$. We lose the convenience of reusing one projection across history, not the ability to encode position. [DeepSeek-V2 §2.1](https://arxiv.org/html/2405.04434v5) separates content and position to address this.

### Why does the order matter? {#rotation-order}

Start with `[1, 1]`. A projection doubles its second coordinate; a rotation turns the vector 90° counterclockwise.

| Order | First result | Final result |
| --- | --- | --- |
| Project, then rotate | `[1, 2]` | `[-2, 1]` |
| Rotate, then project | `[-1, 1]` | `[-1, 2]` |

These are different vectors. With query `[1, 0]`, even the dot product changes from -2 to -1. This is a counterexample about matrix order, not a claim that RoPE rotates every token by 90°; actual angles depend on position and frequency.

<details markdown="1">
<summary>Check the counterexample in Python</summary>

```python
def project_two(vector):
    first, second = vector
    return [first, 2 * second]

def quarter_turn(vector):
    first, second = vector
    return [-second, first]

rotated_key = quarter_turn(project_two([1, 1]))
swapped_key = project_two(quarter_turn([1, 1]))
assert rotated_key == [-2, 1]
assert swapped_key == [-1, 2]
```

The projection is deliberately square so both orders are defined. Real $U_K$ matrices can be rectangular: latent and key spaces need not have the same width. Special structure can permit some reorderings, but an arbitrary projection does not have that property.

</details>

### What does the separated score look like? {#decoupled-score}

The content key remains a linear projection of the latent, while a smaller RoPE branch supplies position. For one head, with content width $d_C$ and positional width $d_R$,

$$
\text{score}_{t,j}=
\frac{(U_K^\top q_t^{C})^\top c_j+(q_t^{R})^\top k_j^{R}}
{\sqrt{d_C+d_R}}.
$$

Here $q_t^R$ and $k_j^R$ have already been rotated. **Add the two terms before taking one softmax over visible history**, rather than normalizing content and position separately. For example, let two history positions have content scores 1 and 1, positional scores 0 and 2, and a combined width of 4. The scaled scores are 0.5 and 1.5, giving weights about 0.269 and 0.731. The positional branch changes which history entry receives more attention.

The content projection can still be reordered; the cache also retains the smaller positional key. Caching full rotated keys is another mathematically valid choice, but it changes the memory savings. The equations alone do not establish which kernel will be faster.

In a **fictional configuration**, eight heads with 64-dimensional K and V require 1,024 cached scalars per token per layer for MHA. A 128-dimensional latent plus a 32-dimensional shared positional key requires 160. This excludes other caches, alignment, and parallel replication. It compares state sizes, not measured quality or speed.

MLA is not another name for GQA. GQA shares KV heads across queries; MLA changes KV representation and computation. Kernel, batch, and hardware determine actual speed.

## What actually survives a decode step?

The DeepSeek-V2 branches make “compressed KV” more concrete:

```text
Current hidden state h
  ├─ KV down-projection ─→ c_KV ────────────────→ history cache
  │                        └─ linear content K/V projections
  ├─ positional key projection ─→ RoPE ─→ k_R ─→ history cache
  └─ query path ─→ current query ───────────────→ read cache, produce output
```

The positional key uses a separate projection of the hidden state, shared across heads. It does not rotate the entire reconstructed content key. Implementations may fuse the two input projections into one matrix multiplication without changing their distinct purposes.

| Intermediate | Needed by later positions? | Treatment |
| --- | --- | --- |
| Historical $c_j^{KV}$ | Yes, for content scores and value aggregation | Cache |
| Historical positional key $k_j^R$ | Yes, for the positional score | Cache |
| Current query latent / projection | Ordinary autoregressive attention does not need past Q | Transient, not KV cache |
| Expanded per-head content K/V | Mathematically involved, not necessarily materialized | Reorder linear operations |

Query compression and KV savings concern different objects. Also, absorption relies on **linearity**: normalization before forming the final latent can coexist with linear up-projection, but inserting an arbitrary nonlinearity after $U_Vc$ prevents moving that projection outside aggregation.

For a tiny counterexample, average values 2 and -2 equally. ReLU after averaging gives 0; averaging after applying ReLU to each gives 1. An arbitrary MLP is not an absorbable up-projection.

## Sparse attention excludes positions from the computation

Suppose a query could attend to eight history positions. Dense attention normalizes over all eight; a sparse scheme selects $\mathcal S$:

$$
o=\sum_{j\in\mathcal S}
\frac{e^{s_j}}{\sum_{k\in\mathcal S}e^{s_k}}v_j.
$$

This generally changes the result, unlike reorganizing the same dense operation with FlashAttention.

Set every score to zero and only the second value to 8. Dense attention outputs 1. A selected set that omits that position outputs 0. Faster reading does not recover omitted evidence.

| Selection | Potential benefit | Likely blind spot |
| --- | --- | --- |
| Local window | Nearby dependencies | Distant facts |
| Fixed global positions | Cross-segment paths | Evidence outside chosen positions |
| Content-based blocks | Query-dependent selection | Rare important information missed by selector |
| Compression plus local/selected detail | Coarse and fine information | Compression loss and selection overhead |

Computing every dense score before taking top-k does not save that score computation. Selection has its own cost, and irregular small reads may use GPUs poorly.

## How should we read NSA?

[Native Sparse Attention](https://arxiv.org/abs/2502.11089) combines compressed, selected, and local-window paths with training and block-oriented hardware considerations. It is not simply retaining a few largest dense attention weights.

Trace what the selector observes, which blocks retain detail, which retain only compressed information, and whether every path is causal. A selector or compressed block containing future tokens leaks information even when the main attention mask is correct.

This note explains mechanisms; it does not equate NSA with other similarly named sparse architectures.

### Three paths retain different information

| Path | Reads | Retains | Cost or blind spot |
| --- | --- | --- | --- |
| Compression | Compressed history blocks | Coarse global information | Lost detail; block count still grows with length |
| Selection | Original K/V in chosen blocks | Query-relevant detail | Selection overhead and missed blocks |
| Sliding window | Recent K/V | Contiguous local context | No full detail outside the window |

```text
Compressed history ─→ compression attention ─────────→ o_comp
                               └─ scores ─→ select ─→ o_select
Recent history ─────────────────────────────────────→ o_window
                         Gate each output, then sum
```

These are not three consecutive transformations of one output. Selection reuses compression scores, but each attention branch normalizes separately. NSA mixes outputs with independent sigmoid gates, which need not sum to one.

Compression and selection blocks may differ in size or stride. Reusing block scores directly requires aligned partitions; otherwise scores need mapping. Selection for a GQA group also aggregates information across query heads, not arbitrary sums of their K/V tensors.

### Catch leakage without training a model

Use zero-based positions and put the query at 5. With four-position blocks, `[0,1,2,3]` is complete, but `[4,5,6,7]` contains future positions. Even if the final read only uses 4 and 5, **compressing all of 4–7 to guide selection already leaks information**.

A second mistake is softmax over all scores followed by zeroing future weights. The future still participates in the denominator.

Compare the two orders with visible values 2 and 4 and a future value of 999. Initially, every score is zero.

```python
import math

def masked_scalar_attention(scores, values, visible, mask_before=True):
    if not (len(scores) == len(values) == len(visible)) or not any(visible):
        raise ValueError("Expected equal lengths and a visible position")
    included = [score for score, keep in zip(scores, visible) if keep or not mask_before]
    shift = max(included)
    weights = [math.exp(score - shift) if keep or not mask_before else 0.0
               for score, keep in zip(scores, visible)]
    return sum(weight * value for weight, value, keep in zip(weights, values, visible) if keep) / sum(weights)

values, visible = [2.0, 4.0, 999.0], [True, True, False]
assert masked_scalar_attention([0, 0, 0], values, visible) == 3.0
assert masked_scalar_attention([0, 0, 0], values, visible, mask_before=False) == 2.0
assert masked_scalar_attention([0, 0, 20], values, visible) == 3.0
assert masked_scalar_attention([0, 0, 20], values, visible, mask_before=False) < 0.001
```

The correct output stays at 3. In the wrong version, changing even the future **score** changes the result. This tests normalization, not a full NSA implementation. A complete model also needs a future-perturbation test: keep the prefix fixed, change the suffix, disable dropout and other randomness, and check unchanged prefix logits. Compression, selection, positional handling, and attention must all respect causality.

### Work through selection and output mixing

Start with the special case where compression and selection blocks align. Two query heads sharing KV assign $[0.6,0.3,0.1]$ and $[0.05,0.45,0.5]$ to three blocks. Independent top-1 choices read blocks 0 and 2. Summing within the group gives $[0.65,0.75,0.6]$, selecting block 1 for both. Shared selection reduces the union of reads, but may omit a particular head's favorite block.

```python
def shared_block_choice(head_probabilities, count):
    import math

    if not head_probabilities or not head_probabilities[0]:
        raise ValueError("Expected nonempty head distributions")
    width = len(head_probabilities[0])
    if type(count) is not int or not 1 <= count <= width:
        raise ValueError("Invalid selection count")
    for head in head_probabilities:
        if len(head) != width or any(not math.isfinite(value) or value < 0 for value in head):
            raise ValueError("Expected finite nonnegative distributions of equal size")
        if not math.isclose(sum(head), 1.0, abs_tol=1e-9):
            raise ValueError("Each head must sum to one")
    totals = [sum(head[index] for head in head_probabilities) for index in range(width)]
    return sorted(range(width), key=lambda index: (-totals[index], index))[:count]

assert shared_block_choice([[0.6, 0.3, 0.1], [0.05, 0.45, 0.5]], 1) == [1]
assert abs(0.2 * 2 + 0.7 * 6 + 0.4 * 4 - 6.2) < 1e-12
```

The code handles aligned blocks only. With compression length $l$, stride $d$, and selection length $l'$, the paper aggregates related compression scores, when $d$ divides both block sizes:

$$
\begin{gathered}
p^{\mathrm{sel}}_t[j]=\\
\sum_{u=0}^{l'/d-1}\sum_{v=0}^{l/d-1}
p^{\mathrm{cmp}}_t[(l'/d)j+u+v].
\end{gathered}
$$

It then sums over query heads sharing KV and selects blocks. Match the paper's indexing and boundaries: overlapping compressed-block probabilities are not probabilities for disjoint raw tokens. [NSA §3.3](https://arxiv.org/html/2502.11089v1#S3.SS3)

After separate branch attention, the output is $g_c o_c+g_s o_s+g_w o_w$. Invented values $o=[2,6,4],g=[0.2,0.7,0.4]$ produce 6.2, above the largest branch output, 6. Independent sigmoid gates need not sum to one. Renormalizing them changes the model.

**What is trained?** The compression branch contributes directly to the output, providing a differentiable training path for the compressor. Discrete selected indices do not become everywhere differentiable because training is described as end-to-end. Do not confuse NSA's reused compression scores with DSA's separate indexer / KL supervision. Kernel checks should cover layouts, padding, gradients, causality, and separate branch softmaxes, not just forward shapes.

## DSA: use a cheaper indexer to choose positions

DeepSeek-V3.2's [DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556) adds a lightning indexer alongside MLA. It does not compute full MLA attention and discard small weights afterward. A cheaper scorer selects top-k positions before the main attention reads their latent K/V entries.

In simplified notation, query position $t$ scores historical position $s$ as:

$$
\begin{gathered}
I_{t,s}=\sum_h w^I_{t,h}\operatorname{ReLU}\big((q^I_{t,h})^\top k^I_s\big),\\
\mathcal S_t=\operatorname{TopK}_{s\le t}(I_{t,s}).
\end{gathered}
$$

Indexer and main attention have separate projections. Few heads and low-precision computation reduce selection cost; MLA query heads share the selected latent entries. Index scores choose locations, rather than replacing the main attention's softmax weights.

### What trains the indexer?

| Stage | Main model | Indexer supervision |
| --- | --- | --- |
| Dense warm-up | Other parameters frozen; attention remains dense | Match aggregated, normalized main-attention distributions |
| Sparse training | Updated by LM loss | Continue matching attention distributions within the selected set |

The paper detaches indexer inputs and uses a separate KL objective for the indexer. Hard top-k is not an ordinary differentiable softmax through which LM loss directly trains every index score.

### What can a missed selection lose?

Use `masked_scalar_attention` from the preceding example. Main attention strongly favors the third position, but the indexer leaves it outside its top two.

```python
def selected_positions(index_scores, count):
    if type(count) is not int or not 0 < count <= len(index_scores):
        raise ValueError("Expected a valid selection count")
    if any(not math.isfinite(score) for score in index_scores):
        raise ValueError("Expected finite index scores")
    return sorted(range(len(index_scores)), key=lambda position: (-index_scores[position], position))[:count]

chosen = selected_positions([3, 2, 1, 0], 2)
attention_scores = [0, 0, 4, 0]
values = [0, 0, 10, 0]
dense_output = masked_scalar_attention(attention_scores, values, [True] * 4)
sparse_output = masked_scalar_attention(attention_scores, values, [position in chosen for position in range(4)])
assert dense_output > 9
assert sparse_output == 0
```

This is not evidence against DSA. It shows why selector recall is another quality bottleneck. Main-kernel speed alone cannot evaluate the system.

### Count the selector's computation too

For length $L$ and $k$ selected positions per query, main-attention pair counts fall from roughly $L^2$ to $Lk$. But an indexer scanning history for each query still contributes a quadratic prefill term. Calling the entire architecture linear is misleading. At short lengths, selection and irregular reads may also offset savings.

Reading fewer positions does not mean retaining less history. Later queries may select other positions, so historical MLA cache remains, with indexer state added. NSA's block selection and DSA's token selection have different computational and hardware tradeoffs.

Test causal selection, top-k boundaries, cache behavior, and position encodings. The [V3.2-Exp model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) records a correction involving different RoPE layouts in the indexer and MLA: matching shapes do not guarantee correct rotations. Checked 2026-10-08; this note describes V3.2, not an automatic description of subsequent DeepSeek versions.

## Compare costs and failures separately

For MLA, measure latent dimension, quality, cache size, and decode speed. Sparse methods additionally need tests of distant evidence, rare crucial tokens, selection overhead, and end-to-end speed.

Hold context length, effective batch, dtype, and hardware fixed; time prefill and decode separately. Short contexts may not amortize selection overhead, while long contexts can expose quality regressions.

Compression, sparsity, lower precision, and kernel improvements are distinct interventions. Change one at a time to identify the source of both savings and failures.
