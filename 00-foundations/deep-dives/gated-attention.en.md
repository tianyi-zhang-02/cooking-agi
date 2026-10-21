# Gated Attention: deciding how much of an attention output to use

[中文](gated-attention.md) · **English**

Attention produces information for later layers. **Output gating adds a separate decision: how much of that information should pass through?** This note covers the 2025 attention-gating paper ([Qwen3-Next and Qwen3.5](../model-families/qwen.en.md#qwen3-next-boundary) pair every three Gated DeltaNet layers with one such gated attention layer), not MoE expert selection or Gated DeltaNet's recurrent-state update.

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

Suppose SDPA outputs are `[4, -2]` and `[1, 3]`, with gate values 0.1 and 0.5.

| Stage | Head 1 | Head 2 |
| --- | --- | --- |
| SDPA output | `[4, -2]` | `[1, 3]` |
| After gating | `[0.4, -0.2]` | `[0.5, 1.5]` |

The two gates add up to 0.6, not 1: each head's gate is set on its own, so turning one down does not turn another up. The first path is also attenuated, not removed. The following $W_O$ still mixes heads, so a head's gate value is not a calibrated confidence score for the final answer.

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

result = gated_heads([[4, -2], [1, 3]], [math.log(1 / 9), 0])
assert math.isclose(result[0][0], 0.4)
assert math.isclose(result[1][1], 1.5)
assert gated_heads([[4, -2]], [0]) == [[2.0, -1.0]]
```

The last assertion catches an easy mistake: zero gate logits give **half the output, not an identity mapping**. Inserting zero-initialized gates into an existing checkpoint does not preserve its behavior.

## Why might it help, and what does it not promise?

$G(X)\odot O(X)$ adds input-dependent multiplicative nonlinearity. Reading information and deciding its influence become partly separate functions. But a small gate does not mean that head was cheap: SDPA has already run. This is not sparse-compute acceleration by itself.

The gate also changes optimization. Sigmoid's derivative is $g(1-g)$, largest near 0.5 and small near either endpoint. Very large initial logits approximate an identity gate but can make the gate harder to train.

Now consider attention sinks. Softmax weights must sum to one, so a head with nothing useful to read still has to hand out that one unit of attention. Prior work cited in the paper's introduction explains sinks this way: the model piles the spare weight onto one token, often the first, whose value is near zero. The numbers below are a **teaching illustration of the mechanism, not data from the paper**: three keys have values 0 (the sink), 4, and 2.

| Approach | Scores | Attention weights | SDPA output | Gate | Passed on |
| --- | --- | --- | --- | --- | --- |
| No gate, rely on the sink | `[3, 0, 0]` | $[0.909, 0.045, 0.045]$ | 0.27 | None | 0.27 |
| Gate, attention stays spread out | `[0, 0, 0]` | $[1/3, 1/3, 1/3]$ | 2 | 0.1 | 0.2 |

Both approaches produce a small output. In this example, the ungated head reduces its output by scoring the sink highly, giving it $e^3/(e^3+2)\approx0.909$ of the weight. The gated head can keep uniform attention weights and use the gate to scale the output down to a tenth.

The [paper](https://arxiv.org/abs/2505.06708) compares 30 gating variants on 15B-total (2.54B active) MoE models and 1.7B dense models, trained on 400B to 3.5T tokens. The best was an elementwise, head-specific sigmoid gate that multiplies the SDPA output ([§3.2](https://arxiv.org/html/2505.06708v1#S3.SS2)). It reports that in the baseline, 46.7% of attention goes to the first token on average across layers, and 4.8% with this gate; the authors attribute the drop to query-dependent sparse gating ([§4.3](https://arxiv.org/html/2505.06708v1#S4.SS3)). Those results are evidence for its training settings, not a guarantee that attaching a gate to any checkpoint removes attention sinks. Nor can a single attention heatmap show whether a sink is present.

The gate also has costs: each head gets extra projection parameters and one more multiplication. The paper's dense experiments narrow the FFN to keep the total parameter count fixed ([§3.2.2](https://arxiv.org/html/2505.06708v1#S3.SS2.SSS2)). Whether the gate helps in another architecture or at another scale needs its own ablation.

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
