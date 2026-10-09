# RoPE and long context: fitting the input is not understanding it

[中文](position-and-context.md) · **English**

> Reviewed: 2026-10 · Prerequisite: [multi-head attention](../core/multi-head-attention.en.md)

Changing a context-length setting from 8K to 128K without crashing establishes that the program can accommodate those positions. Whether the model learned such distances or retrieves evidence from the middle requires separate tests.

RoPE does more than attach a position number. It introduces position relationships into attention dot products.

## Start with a two-dimensional rotation

The rotation matrix is

$$
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}.
$$

A query at position $p$ becomes $R(p\theta)q$ and a key at $r$ becomes $R(r\theta)k$. Therefore

$$
(R(p\theta)q)^\top R(r\theta)k
=q^\top R((r-p)\theta)k.
$$

Relative distance enters the dot product through this identity. For fixed unrotated $q,k$, shifting both positions equally preserves the dot product. In a full model, hidden states depend on context, so this is not a universal translation-invariance guarantee.

For illustration, take $\theta=\pi/2$ and $q=k=[1,0]$. Positions 0 and 1 have dot product zero; positions 3 and 4 also give zero. A separation of two gives -1. Rotation preserves individual norms while changing their relative orientation. This is not a real model's frequency configuration.

## Why use multiple frequencies?

Split the head into two-dimensional pairs with different frequencies. A common base form is

$$
\theta_j=b^{-2j/d},\quad j=0,\ldots,d/2-1.
$$

Here $d$ is the rotated dimension and $b$ the base. With $d=4,b=10000$, the two frequencies are 1 and 0.01. A distance of ten creates angle differences of 10 and 0.1 radians. Different frequencies expose different scales of positional variation.

[RoFormer](https://arxiv.org/abs/2104.09864) develops rotary position embeddings. Implementations must also match pair layout, partial versus full rotation, frequency conventions, and cache position offsets. A layout mismatch can preserve shapes while breaking checkpoint semantics.

```python
import math

def rotate_pair(vector, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return [cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1]]

def dot(left, right):
    return sum(first * second for first, second in zip(left, right))

query, key = [1.0, 2.0], [3.0, -1.0]
frequency = 0.2
original = dot(rotate_pair(query, 2 * frequency), rotate_pair(key, 5 * frequency))
shifted = dot(rotate_pair(query, 9 * frequency), rotate_pair(key, 12 * frequency))
relative = dot(query, rotate_pair(key, 3 * frequency))
assert math.isclose(original, shifted, abs_tol=1e-12)
assert math.isclose(original, relative, abs_tol=1e-12)
```

## Why can extrapolation fail?

Short-distance training does not establish reliable behavior at every larger distance. Rotations are periodic: different distances can resemble one another in some dimensions. Multiple frequencies and learned representations jointly determine distinguishability.

“RoPE uses relative position, so any length works” is too strong. So is “one frequency completed a rotation, so the model fails.” Computing the formula and generalizing learned behavior are different claims.

Test evidence position, distractors, and dependency span as well as total length. A conspicuous keyword near the end is a weak test of whole-document integration.

## Interpolation, frequency scaling, and continued training

Simple position interpolation replaces $p$ with $p/s$. For a fourfold extension, $s=4$ compresses positions, including adjacent positions now separated by 0.25. It reduces extrapolation while changing local resolution.

| Method | What changes | What is not guaranteed |
| --- | --- | --- |
| Directly use larger positions | Input range | Stable behavior at unfamiliar distances |
| Position interpolation | Positions compressed into a smaller range | Unchanged local relationships |
| Frequency-dependent methods such as YaRN | Different frequency-band adjustments and associated attention scaling | Equivalence to an arbitrary base change |
| Long-context continued training | Adaptation to new lengths and position settings | Free compute or preserved short-input quality |

[Position Interpolation](https://arxiv.org/abs/2306.15595) and [YaRN](https://arxiv.org/abs/2309.00071) describe concrete schemes. We are comparing design purposes, not promising their reported gains for another model.

## YaRN: which frequencies should change?

For a fourfold extension, dividing every frequency by four is simple, but also changes high-frequency local relationships. YaRN uses frequency-dependent adjustments rather than learning a new frequency set for every token.

A useful reference is the number of rotations within the original training length $L$:

$$
r_j=\frac{L\theta_j}{2\pi},\qquad
\theta'_j=\left(\frac{1-\gamma_j}{s}+\gamma_j\right)\theta_j.
$$

$s$ is the extension factor; $\gamma_j$ transitions from zero to one with $r_j$. Low frequencies favor interpolation; high frequencies favor keeping the original rate. **Frequency refers to different two-dimensional components, not earlier or later tokens.** Each token uses all those components.

For simple arithmetic, choose $s=4$, set $\gamma=0$ below one rotation and 1 above four, and interpolate linearly in between. These are teaching parameters, not checkpoint settings.

| Rotations $r$ in the original window | $\gamma$ | New / original frequency | Effect |
| --- | --- | --- | --- |
| 0.5 | 0 | 0.25 | Interpolate, rotating more slowly |
| 2.5 | 0.5 | 0.625 | Blend the two ends |
| 8 | 1 | 1 | Preserve the frequency |

```python
def yarn_frequency_multiplier(rotations, scale=4.0, low=1.0, high=4.0):
    if rotations < 0 or scale < 1 or not 0 <= low < high:
        raise ValueError("Invalid scaling configuration")
    blend = min(1.0, max(0.0, (rotations - low) / (high - low)))
    return (1 - blend) / scale + blend

assert yarn_frequency_multiplier(0.5) == 0.25
assert yarn_frequency_multiplier(2.5) == 0.625
assert yarn_frequency_multiplier(8.0) == 1.0
assert yarn_frequency_multiplier(2.5, scale=1.0) == 1.0
```

This illustrates the frequency blend. Implementations may express the transition in dimension indices, with rounding and checkpoint-specific choices. It is not drop-in model code.

YaRN also adjusts attention scaling. Multiplying both Q and K by $a$ multiplies their dot product by $a^2$, changing softmax sharpness. Scores `[0,1]` give probabilities near `[0.269,0.731]`; doubling scores gives `[0.119,0.881]`. This acts inside attention, not at output-token sampling temperature.

Check frequency rules, attention scaling, training length, and cache behavior together. A `rope_theta` setting alone does not specify the full extension method.

### Two implementation details: scaling and caches

[YaRN §3.4](https://arxiv.org/html/2309.00071v2#S3.SS4) uses $a=1+0.1\ln s$ in its Llama experiments, multiplying Q and K amplitudes by $a$, hence logits by $a^2$. At $s=4$ that is about 1.2965; at $s=1$ it is 1. This is an empirical setting, not a universal constant for every newer model.

The derivation is $(aQ)(aK)^\top=a^2QK^\top$. Scaling only rotary dimensions does not scale the **entire dot product** by $a^2$. For $Q=K=[2,3]$, the original dot product is 13; doubling both dimensions gives 52, while doubling only the second gives 40. Check placement when adapting partial RoPE or MLA.

The [authors' implementation](https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py) uses dimension-index transition boundaries and rounding. The continuous explanation, teaching function, and a checkpoint's implementation are not interchangeable without checking.

Dynamic scale raises another issue: historical keys rotated with old frequencies and a new query rotated with new ones no longer share the intended position rule. Keep the request's convention fixed or retain state that can reconstruct / rerotate historical keys; changing only the query is insufficient. Compare full prefill with token-by-token decoding across the scale-change boundary, not only on short inputs before it.

## DCA: what happens across a chunk boundary?

[Dual Chunk Attention](https://arxiv.org/abs/2402.17463) distinguishes positions within the same chunk, the previous chunk, and more distant chunks instead of uniformly compressing every position. This is the paper's scheme, not a claim that all 2026 long-context models use it.

Let the original window be $c$, chunk size $s<c$, query position $i$, and visible key position $j\le i$. Keys use $j\bmod s$; the query position depends on the relationship:

| Relationship | Local query position | Intended preservation |
| --- | --- | --- |
| Same chunk | $i\bmod s$ | Within-chunk distances |
| Previous chunk | $\min(s+i\bmod s,c-1)$ | Continuity near the boundary |
| Earlier chunk | $c-1$ | Avoiding unfamiliar large position differences |

Subtract the local key position to obtain the RoPE distance. Distant positions lose fine-grained separation; the scheme does not recover unlimited exact positions.

Take $c=8,s=5$. Position 5 attending to 4 should see distance 1. Resetting both positions modulo the chunk size gives $0-4=-4$. The adjacent-chunk rule gives $5-4=1$. Later, position 8 attending to 4 gets $7-4=3$, not the true distance 4: restricting the position range has a cost.

```python
def dca_distance(query_position, key_position, window=8, chunk=5):
    values = (query_position, key_position, window, chunk)
    if any(type(value) is not int for value in values):
        raise ValueError("Expected integer positions and sizes")
    if not 0 <= key_position <= query_position or not 0 < chunk < window:
        raise ValueError("Expected causal positions and chunk < window")
    gap = query_position // chunk - key_position // chunk
    if gap == 0:
        query_local = query_position % chunk
    elif gap == 1:
        query_local = min(chunk + query_position % chunk, window - 1)
    else:
        query_local = window - 1
    return query_local - key_position % chunk

assert dca_distance(4, 3) == 1
assert dca_distance(5, 4) == 1
assert dca_distance(6, 4) == 2
assert dca_distance(8, 4) == 3
assert dca_distance(10, 0) == 7
```

This tests one query–key distance, not a DCA kernel. A full implementation must **globally normalize** the regions rather than average locally normalized chunk outputs. One chunk containing value 0 and another containing three values of 4, with all scores zero, should yield 3—not the average local output of 2.

DCA is not restricted to reading a few chunks: it can still read the entire history and does not automatically remove quadratic dense-attention computation. Test evidence crossing chunk boundaries, padding, cached/uncached logits, and actual long-document tasks. Running without additional training does not guarantee unchanged quality on every checkpoint. The [authors' implementation](https://github.com/HKUNLP/ChunkLlama) shows concrete integration.

## Position handling does not remove compute costs

At fixed model and batch size, moving from 8K to 128K multiplies length by 16. Ordinary KV storage grows roughly 16-fold; dense prefill attention pair counts grow roughly 256-fold. Runtime need not follow those exact ratios because FFNs, kernels, parallelism, and memory access also matter.

Access to a long context does not mean every task should include all available text. Retrieval is cheaper but can omit jointly necessary evidence; summarization is shorter but can lose details. The choice depends on the task.

## A useful long-context test

Fix model version, RoPE configuration, tokenizer, and decoding. Move the same evidence between beginning, middle, and end, add similar distractors, and include questions requiring two separated pieces of evidence.

Report correctness, citation support, TTFT, peak memory, and short-input regressions. Cached and uncached logits should also agree within tolerance, especially around padding, position offsets, and multi-token decode masks.

Then distinguish increased input capacity, usable retrieval distance, and genuine long-document task ability.

What if a model does not use explicit positional encoding? Continue with [NoPE and sequence order](nope-and-order.en.md): a small recurrent example, a causal-mask example, and the different roles of KDA and MLA in Kimi K3.
