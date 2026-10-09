# FFNs and SwiGLU: what happens after attention?

[中文](ffn-and-gates.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [multi-head attention](multi-head-attention.en.md), [residuals](residual-connections.en.md)

Transformer diagrams often give attention the prominent position and reduce the FFN to a small box. That box can hold substantial parameters and compute. Rather than reading the context again, it applies a nonlinear transformation to each position's existing representation.

We will use one token and column vectors, omitting batch dimensions, biases, and normalization.

## Why expand and contract the representation?

A two-layer FFN is

$$
z=W_{\mathrm{up}}x,\qquad
y=W_{\mathrm{down}}\phi(z),
$$

with $x\in\mathbb R^d$, $W_{\mathrm{up}}\in\mathbb R^{h\times d}$, and $W_{\mathrm{down}}\in\mathbb R^{d\times h}$. Expansion creates intermediate features, the nonlinearity changes their combination, and the final projection returns to residual-stream width.

Without $\phi$, the matrices compose into one linear map. Expanding the hidden dimension alone does not produce nonlinear expressiveness.

Tokens share FFN weights but do not directly exchange information inside an ordinary dense FFN. Earlier attention has already made their representations contextual. Position-wise does not mean context-free.

## SwiGLU adds an input-dependent multiplicative branch

A bias-free formulation is

$$
g=W_gx,\quad u=W_ux,\quad
h=\operatorname{SiLU}(g)\odot u,\quad
y=W_dh,
$$

$$
\operatorname{SiLU}(z)=z\sigma(z).
$$

Two projections create a gate branch and a content branch, which interact elementwise. The gate is not a probability or necessarily between zero and one: SiLU can be negative or exceed one. It is not discrete expert selection.

[GLU Variants](https://arxiv.org/abs/2002.05202) compares gated FFN forms, including SwiGLU. Results depend on training and budget; replacing an activation's name does not reproduce the paper's findings.

## Work through two intermediate features

Isolate the gate with projected vectors $g=[0,1]$ and $u=[3,2]$:

| Dimension | Gate branch | SiLU | Content branch | Product |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 3 | 0 |
| 2 | 1 | 0.7311 | 2 | 1.4621 |

With $W_d=[1,-1]$, the scalar output is approximately -1.4621. We use output dimension one for this calculation; an actual block returns to $d$.

```python
import math

def silu(value):
    return value / (1 + math.exp(-value))

gate = [0.0, 1.0]
content = [3.0, 2.0]
hidden = [silu(gate_value) * content_value
          for gate_value, content_value in zip(gate, content)]
output = hidden[0] - hidden[1]
assert math.isclose(output, -1.4621171572600098)
```

A zero output in the first dimension does not permanently cut off learning. With upstream gradient $\delta$,

$$
\frac{\partial\mathcal L}{\partial u}
=\delta\odot\operatorname{SiLU}(g),\qquad
\frac{\partial\mathcal L}{\partial g}
=\delta\odot u\odot\operatorname{SiLU}'(g).
$$

Since $\operatorname{SiLU}'(0)=1/2$, setting $u=3,\delta=1$ gives gate gradient 1.5. The content gradient is zero at this point, but the gate can still learn. A permanent on/off-switch metaphor misses that behavior.

## Compare three matrices at a matched budget

Ignoring biases, an ordinary FFN has $2dh$ parameters; SwiGLU has $3dh_g$. Matching an ordinary width of $4d$ gives

$$
2d(4d)=3dh_g\quad\Rightarrow\quad h_g=\frac83d.
$$

For $d=12$, ordinary width 48 and gated width 32 both give 1,152 parameters. Real implementations may round widths for hardware alignment.

Three matrices do not necessarily increase total parameters. Keeping the intermediate width unchanged, however, does cost more. Report width, parameter count, and compute when comparing structures, so extra compute is not mistaken for an architectural advantage.

## What changes again when an FFN becomes MoE?

SwiGLU continuously modulates feature channels. An MoE router chooses which expert FFNs process a token; those experts can themselves use SwiGLU. These are different levels of selection.

| Mechanism | Operation | Main constraint |
| --- | --- | --- |
| Attention | Combine information across positions | Length, cache, and IO |
| Dense FFN / SwiGLU | Transform channels at a position | Intermediate activations and matrix work |
| MoE routing | Assign tokens to selected experts | Load balance, communication, stability |

Continue with the [MoE series](../moe/README.en.md). For an FFN implementation, check shapes and nonlinearity placement, use numerical gradients on tiny inputs, then compare task quality and runtime at a matched budget.
