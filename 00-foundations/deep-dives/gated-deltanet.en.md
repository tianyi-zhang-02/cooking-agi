# Gated DeltaNet: decay the state, then correct its prediction

[中文](gated-deltanet.md) · **English**

A KV cache retains records for historical tokens. A recurrent model instead updates a fixed-shaped state after every token. Decode no longer has to scan every historical record, but exact item-by-item access is lost. Gated DeltaNet studies how to write, revise, and forget within that limited state.

## Treat the state as a small associative memory

Let $S\in\mathbb R^{d_v\times d_k}$, with key $k$ and query $q$ as $d_k$-dimensional columns and value $v$ as a $d_v$-dimensional column. Reading means computing $Sq$.

An additive write uses $S_t=S_{t-1}+v_tk_t^\top$. Repeated keys accumulate rather than replace their old values. In a scalar example, writing 2 and then 6 returns 8, not the latest value 6.

The delta rule reads the current value and writes the error:

$$
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top.
$$

For $\|k_t\|_2=1$, reading along that key afterward gives:

$$
S_tk_t=(1-\beta_t)S_{t-1}k_t+\beta_tv_t.
$$

This interpolates between old and new values. At $\beta=1$, the value along that direction is replaced. Orthogonal directions receive no delta correction. **Similar, nonorthogonal keys can interfere**; the state is not an exact dictionary.

## Why add forgetting?

Correcting only the current key cannot quickly clear other stale information. Gated DeltaNet first decays the state with $\alpha_t$, then corrects the prediction made by that decayed state:

$$
\bar S_t=\alpha_tS_{t-1},\qquad
S_t=\bar S_t+\beta_t(v_t-\bar S_tk_t)k_t^\top.
$$

Equivalently:

$$
S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top,\qquad o_t=S_tq_t.
$$

Do not omit $\alpha$ inside the error. Once the old state has decayed, the correction must target the **post-decay** prediction.

There is also an optimization interpretation: starting from $\bar S_t$, take one gradient step of size $\beta_t$ on the local squared loss $\frac12\|Sk_t-v_t\|^2$. This updates a fast state within a forward pass, not the model's parameters through an extra optimizer step. Outer-loop training still learns projections and gates.

## Work through a two-slot state

Take $S=[2,10]$, $k=[1,0]^\top$, new value $v=6$, and gates $\alpha=0.5,\beta=0.25$.

| Step | Result |
| --- | --- |
| Decay the complete state | $[1,5]$ |
| Read with the key | $1$ |
| Prediction error | $6-1=5$ |
| Write along the key | $0.25\times5[1,0]=[1.25,0]$ |
| New state | $[2.25,5]$ |

The second slot receives no delta correction but still decays. Using the pre-decay value 2 in the error incorrectly produces `[2,5]`.

```python
import math

def gated_delta_step(state, key, value, decay, write):
    if not key or len(state) != len(value) or not state:
        raise ValueError("Invalid state, key, or value shape")
    if any(len(row) != len(key) for row in state):
        raise ValueError("State rows must match key dimension")
    scalars = [decay, write, *key, *value, *(entry for row in state for entry in row)]
    if any(not math.isfinite(entry) for entry in scalars):
        raise ValueError("Expected finite inputs")
    if not 0 <= decay <= 1 or not 0 <= write <= 1:
        raise ValueError("This example uses gates in [0, 1]")
    if not math.isclose(sum(entry * entry for entry in key), 1.0, abs_tol=1e-9):
        raise ValueError("This example assumes a unit key")
    decayed = [[decay * entry for entry in row] for row in state]
    prediction = [sum(entry * key_entry for entry, key_entry in zip(row, key)) for row in decayed]
    return [[entry + write * (target - old) * key_entry
             for entry, key_entry in zip(row, key)]
            for row, target, old in zip(decayed, value, prediction)]

state = gated_delta_step([[2.0, 10.0]], [1.0, 0.0], [6.0], 0.5, 0.25)
assert state == [[2.25, 5.0]]
assert gated_delta_step([[2.0, 10.0]], [1.0, 0.0], [6.0], 1.0, 1.0) == [[6.0, 10.0]]
```

This is a reproducible recurrence, not input projections, short convolutions, normalization, output gating, or a GPU kernel. Endpoints 0 and 1 illustrate limiting behavior; inspect each model's actual parameterization. Some extensions permit a larger writing range, where this interpolation interpretation no longer applies directly.

## Recurrent decode does not require a Python loop for training

Each step is $S_t=S_{t-1}A_t+B_t$. Combining two steps gives:

$$
S_{t+1}=S_{t-1}A_tA_{t+1}+B_tA_{t+1}+B_{t+1}.
$$

Composition is ordered; the matrices cannot be freely swapped. The paper uses chunking and structured transformations to turn within-chunk work into hardware-friendly matrix operations. It neither removes every dependency nor treats the slow loop above as a high-throughput trainer.

Check agreement among single-step recurrence, chunked prefill, and save-state/resume execution. Reset between independent samples, but retain state when continuing the same request. Incorrect boundaries leak across samples or erase the prefix.

## What does fixed state actually save?

A hypothetical layer with 16 heads, each retaining a $128\times128$ FP32 state, uses 1 MiB regardless of prefix length. Another hypothetical attention layer with 16 KV heads, dimension 128, and 8,192 BF16 K/V entries uses 64 MiB.

This is not an equal-capability performance comparison. It illustrates different storage growth. Recurrent state still occupies memory; training stores or recomputes activations; full-attention layers in a hybrid still retain length-dependent KV caches.

| Desired property | Tradeoff |
| --- | --- |
| Fixed-shaped decode state | Compressed history, without guaranteed exact token recall |
| Targeted delta correction | Correlated-key interference and limited state capacity |
| Global decay | Fast clearing can also erase useful information |
| Hybrid attention | Some direct historical access, with its storage and compute costs |

Checked 2026-10-08 against [Gated Delta Networks](https://arxiv.org/html/2412.06464v2) and the [authors' code](https://github.com/NVlabs/GatedDeltaNet). Full training and GPU speed have not been reproduced. Continue with [Qwen](../model-families/qwen.en.md) to see how recurrence combines with attention and FFN/MoE layers.
