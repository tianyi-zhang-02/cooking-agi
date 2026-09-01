# Attention Reference, Shapes, and Masks

[中文](01-reference-shapes-masks.md) · **English** · [Back to project](../../projects/01-attention-from-scratch.en.md)

> Reading time: ~5 minutes · Level: Intermediate · Freshness: Stable · Last reviewed: 2026-08

## Core problem

How do we build a slow but trustworthy attention correctness oracle?

```text
S = QKᵀ / √D
P = softmax(S + mask)
O = PV
```

The reference should execute these operations explicitly rather than call a high-level attention API. Intermediate scores, masked values, and probabilities then remain inspectable.

## Shape contract

```text
Q [B, H, Nq, D]
K [B, Hkv, Nk, D]
V [B, Hkv, Nk, Dv]
O [B, H, Nq, Dv]
```

Begin with `H = Hkv`. In GQA, several query heads map onto fewer KV heads, and that mapping must be explicit. Do not let framework broadcasting guess the semantics.

Layout must also be explicit. `[B,H,N,D]` and `[B,N,H,D]` represent similar logical values but have different strides, contiguous access patterns, and kernel mappings.

## Stable softmax

For every row, use:

```text
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

Apply `1/√D` before softmax. Check that each row sums approximately to one and that large-magnitude inputs do not create NaN or Inf.

## Mask contract

Implement and test separately:

- causal masks;
- padding or valid-length masks;
- arbitrary additive masks;
- combined causal and padding masks.

Define behavior for a fully masked row. One finite negative constant may behave differently across FP32, FP16, and BF16.

## Test matrix

- `B=1` and `B>1`;
- single and multiple heads;
- `Nq != Nk`;
- `D != Dv`;
- tiny and non-tile-multiple sequence/head dimensions;
- random, constant, and large-magnitude values;
- no mask, causal, padding, and combined masks.

Fixed golden cases make debugging easy; randomized property tests prevent overfitting one input.

## Hands-on check

1. Save Q, K, V, scores, probabilities, and output.
2. Compare every CPU C++ stage with the oracle.
3. Introduce an off-by-one causal mask and confirm that tests fail.
4. Permute heads and verify that they remain independent.

## Key conclusions

Every optimized implementation compares against the same independent oracle, not the preceding optimization. Otherwise one early bug propagates through the entire project.

Next: [CPU to Tiled CUDA](02-cpu-to-tiled-cuda.en.md)
