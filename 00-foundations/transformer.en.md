# The Transformer architecture

[中文](transformer.md) · **English**

## In one sentence

The Transformer is one concrete way to build the learned coordinate transform from [the previous page](from-linear-to-neural.en.md): **attention moves information across positions, the FFN processes each position on its own**, alternating for $N$ layers, with a linear classifier reading out the answer at the end.

## Scaled dot-product attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Per query: $\alpha_{ij} = \text{softmax}_j(\mathbf{q}_i^\top\mathbf{k}_j/\sqrt{d_k})$, then $\mathbf{o}_i = \sum_j \alpha_{ij}\mathbf{v}_j$. A weighted average of the value vectors, with similarity as the weights.

### Why $\sqrt{d_k}$

For unit-variance $q, k$, the dot product of $d_k$ terms has variance $d_k$, i.e. a spread of $\pm 8$ at $d_k = 64$. Softmax at that scale is nearly one-hot, and its Jacobian

$$\frac{\partial\,\text{softmax}(z)_i}{\partial z_j} = \alpha_i(\delta_{ij} - \alpha_j)$$

collapses toward the zero matrix — no gradient. Dividing by $\sqrt{d_k}$ pulls the variance back to 1.

```python
def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))   # before softmax, not after
    attn = scores.softmax(dim=-1)
    return attn @ v, attn
```

## One module, three uses

The single most important thing about the 2017 encoder-decoder. `self_attn(x,x,x)` and `cross_attn(x, memory, memory)` are the same class with different tensors:

| Use | Q from | K, V from | Mask | Shape |
| --- | --- | --- | --- | --- |
| Encoder self-attention | src | src | padding only, **bidirectional** | $(B,h,S,S)$ |
| Decoder self-attention | tgt | tgt | padding **∨** causal | $(B,h,T,T)$ |
| **Cross-attention** | **tgt** | **memory** | src padding | $(B,h,T,S)$ — not square |

Cross-attention is the only place the two towers touch. [`code/vanilla_demo.py`](code/vanilla_demo.py) trains it to reverse a sequence, where the correct alignment is known in advance, so you can read it straight off the matrix — an anti-diagonal, 64/64 sequences exact.

## Multi-head

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \quad \text{MultiHead} = \text{Concat}(\text{head}_1..\text{head}_h)W^O$$

One attention computes one similarity and returns one average. Splitting into $h$ heads of $d_k = d_\text{model}/h$ lets different heads track different relations at no extra cost. In code it is one $(d_\text{model}, d_\text{model})$ projection reshaped into heads — mathematically identical, a single GEMM.

## Post-norm and the warmup it requires

The paper does $\mathbf{x} \leftarrow \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$ — the norm sits **on the residual highway**. Everything modern does $\mathbf{x} \leftarrow \mathbf{x} + \text{Sublayer}(\text{Norm}(\mathbf{x}))$ instead.

This is why the original recipe needs the Noam schedule

$$\text{lr}(t) = d_\text{model}^{-0.5}\min\big(t^{-0.5},\; t\cdot t_\text{warmup}^{-1.5}\big)$$

Warmup is not a tuning trick under post-norm; it is what keeps the first few thousand steps from diverging. Pre-norm gives the gradient a path that skips every norm, which is what made constant learning rates safe.

> Field note: `LambdaLR` multiplies the **base** lr by your lambda. Set Adam's `lr=0` and attach a Noam schedule and the learning rate stays 0 forever, while dropout noise makes the loss look like it is still moving.

## Positional encoding: sinusoids → RoPE

Attention is permutation-equivariant: shuffle the inputs and the outputs shuffle with them. Position has to be injected.

**Original**: fixed sinusoids $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$, **added** to the embedding (not concatenated), wavelengths in a geometric series — a multi-scale clock.

**Now**: RoPE rotates $\mathbf{q}$ and $\mathbf{k}$ at every layer instead. Because rotation matrices satisfy $R_m^\top R_n = R_{n-m}$,

$$\langle R_m\mathbf{q}, R_n\mathbf{k}\rangle = \mathbf{q}^\top R_{n-m}\mathbf{k}$$

the score depends **only on the relative distance**. $\mathbf{v}$ is not rotated — it should carry no positional information.

```python
def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)          # split-half (Llama convention)
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

⚠️ Two pairing conventions exist — split-half (GPT-NeoX/Llama) and interleaved (the RoPE paper). They differ by a permutation of head channels, so **weights are not interchangeable** across them.

## What else changed

| | vanilla (2017) | today |
| --- | --- | --- |
| Structure | encoder + decoder | decoder-only |
| Norm | post-norm LayerNorm | pre-norm RMSNorm |
| Position | sinusoidal, added to input | RoPE, applied to q/k per layer |
| Attention | MHA | GQA |
| FFN | ReLU, $4d$ | SwiGLU, $\tfrac{8}{3}d$ |
| Inference | recompute each step | KV cache |

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}\odot\boldsymbol{\gamma}, \qquad \text{SwiGLU}(\mathbf{x}) = \big(\text{SiLU}(W_g\mathbf{x})\odot W_u\mathbf{x}\big)W_d$$

RMSNorm drops the mean subtraction and the bias (accumulate the sum of squares in fp32 regardless of activation dtype). SwiGLU has **three** matrices instead of two, hence the $\tfrac{8}{3}d$ hidden width rather than $4d$.

GQA shares one K/V head across $n_\text{rep}$ query heads. The KV cache costs

$$2 \cdot n_\text{layer} \cdot n_\text{kv} \cdot d_\text{head} \cdot T \cdot \text{sizeof(dtype)}$$

so dropping $n_\text{kv}$ from 32 to 8 saves 4× the memory — the dominant constraint in long-context inference.

## Building it, and proving it works

[`code/`](code/) has both, written without `nn.MultiheadAttention` or `F.scaled_dot_product_attention`:

- [`vanilla.py`](code/vanilla.py) — the 2017 encoder-decoder, with the Noam schedule and label smoothing
- [`model.py`](code/model.py) — modern decoder-only (RMSNorm + RoPE + GQA + SwiGLU + KV cache)

[`test_model.py`](code/test_model.py) checks the four things that actually break:

```
  causality:     perturb token 9 -> logits at positions < 9 differ by 0.0 (exactly)
  kv cache:      incremental decode vs one-shot forward, max error 3.6e-07
  rope relative: score(5,2) = score(20,17) = +5.6092;  score(20,10) = +0.4579
  init loss:     4.19 vs ln(V) = 4.16
```

The subtlest bug is the mask under caching: query $i$ sits at absolute position `cache.pos + i` while keys run from 0, so the mask is a **non-square** $(T, S)$, and RoPE's cos/sin must be sliced from `cache.pos`. Also `cache.pos` advances once per forward pass — putting it inside `update()` multiplies it by `n_layer`.

## Where to read next

- [From linear models to neural networks](from-linear-to-neural.en.md)
- [Post-training](../05-post-training/README.en.md)
- [Systems](../06-systems/README.en.md)
