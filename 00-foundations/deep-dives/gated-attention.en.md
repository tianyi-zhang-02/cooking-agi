# Gated Attention: deciding how much of an attention output to use

[中文](gated-attention.md) · **English**

Attention produces information for later layers. Output gating adds a separate decision: how much of that information should pass through? This note covers the 2025 attention-gating paper, not MoE expert selection or Gated DeltaNet's recurrent-state update.

## Where does the gate go?

For input $X$, a head projects queries, keys, and values and computes:

$$
O_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}+M\right)V_h.
$$

$M$ contains causal and padding masks. The placement discussed here is after SDPA and before the output projection:

$$
G_h=\sigma(XW_{g,h}),\qquad
Y=\operatorname{Concat}_h(G_h\odot O_h)W_O.
$$

A headwise gate supplies one number per token and head. An elementwise gate supplies one per head dimension. Headwise does **not** mean a fixed switch for a head across the entire training run: its value still depends on the current token representation.

```text
Input ─→ Q/K/V ─→ SDPA ─→ × gate ─→ concatenate heads ─→ output projection
  └────────────→ sigmoid ──┘
```

The sigmoid gates are independent and need not sum to one. Unlike softmax competition, every head can pass less information, or several can pass more simultaneously.

## Work through two heads

Suppose SDPA outputs are `[4, -2]` and `[1, 3]`, with gate values 0.1 and 0.9.

| Stage | Head 1 | Head 2 |
| --- | --- | --- |
| SDPA output | `[4, -2]` | `[1, 3]` |
| After gating | `[0.4, -0.2]` | `[0.9, 2.7]` |

The first path is attenuated, not removed. The following $W_O$ still mixes heads, so a head's gate value is not a calibrated confidence score for the final answer.

```python
import math

def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)

def gated_heads(head_outputs, gate_logits):
    if not head_outputs or len(head_outputs) != len(gate_logits):
        raise ValueError("One logit per head is required")
    if any(not head or any(not math.isfinite(value) for value in head) for head in head_outputs):
        raise ValueError("Expected nonempty finite head outputs")
    if any(not math.isfinite(value) for value in gate_logits):
        raise ValueError("Expected finite logits")
    return [[sigmoid(logit) * value for value in head]
            for head, logit in zip(head_outputs, gate_logits)]

result = gated_heads([[4, -2], [1, 3]], [math.log(1 / 9), math.log(9)])
assert math.isclose(result[0][0], 0.4)
assert math.isclose(result[1][1], 2.7)
assert gated_heads([[4, -2]], [0]) == [[2.0, -1.0]]
```

The last assertion catches an easy mistake: zero gate logits give **half the output, not an identity mapping**. Inserting zero-initialized gates into an existing checkpoint does not preserve its behavior.

## Why might it help, and what does it not promise?

$G(X)\odot O(X)$ adds input-dependent multiplicative nonlinearity. Reading information and deciding its influence become partly separate functions. But a small gate does not mean that head was cheap: SDPA has already run. This is not sparse-compute acceleration by itself.

The gate also changes optimization. Sigmoid's derivative is $g(1-g)$, largest near 0.5 and small near either endpoint. Very large initial logits approximate an identity gate but can make the gate harder to train.

The [paper](https://arxiv.org/abs/2505.06708) compares placements and gate forms and studies attention sinks and training stability. Those results are evidence for its training settings, not a guarantee that attaching a gate to any checkpoint removes attention sinks. One attention heatmap is insufficient evidence either.

## What makes a useful comparison?

Hold data, training tokens, optimizer, and parameter budget fixed while comparing no gate, headwise gating, and elementwise gating. Track:

- Held-out loss and downstream tasks, not just attractive gate distributions.
- Gate values by layer, head, and token type, including persistent saturation.
- Gradient norms, loss spikes, and changes on short and long inputs.
- Prefill/decode latency and memory. Another gate is not automatically a speedup.

For implementation checks, temporarily force gates to one and compare with the original attention path. Then force them to zero and confirm that only the attention branch disappears, not the residual stream. These checks establish wiring, not model quality.

## Keep different gates separate

| Mechanism | What it controls |
| --- | --- |
| Gated Attention here | How much computed attention output passes through |
| MoE router | Which experts compute for a token |
| Gated DeltaNet | State retention and correction with new information |
| Engram gate | Whether a retrieved local pattern fits the current context |

Checked 2026-10-08. The examples test local forward computations, not full-paper training. Continue with [Engram](engram.en.md) to compare attending over past tokens with consulting a learned pattern table.
