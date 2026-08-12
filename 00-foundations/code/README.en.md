# Build it by hand: runnable implementations

[中文](README.md) · **English**

Every claim in this chapter has code behind it. Both Transformers are written **without** `nn.MultiheadAttention` and `F.scaled_dot_product_attention` — only `nn.Linear` / `nn.Embedding` / `nn.Parameter` and raw tensor ops.

PyTorch is the only dependency; everything runs on CPU.

```bash
pip install torch
```

## Files

| File | What it does | Runtime |
| --- | --- | --- |
| [`why_nonlinear.py`](why_nonlinear.py) | Three models on XOR, ASCII decision boundaries and hidden space | ~20 s |
| [`make_figures.py`](make_figures.py) | Trains the same models and regenerates the SVGs in [`../assets/`](../assets/) | ~30 s |
| [`vanilla.py`](vanilla.py) | The 2017 encoder-decoder, faithful to the paper | — |
| [`vanilla_demo.py`](vanilla_demo.py) | Shape trace → train "reverse the sequence" → print cross-attention | ~1 min |
| [`model.py`](model.py) | Modern decoder-only: RMSNorm + RoPE + GQA + SwiGLU + KV cache | — |
| [`test_model.py`](test_model.py) | Correctness checks (causality, cache equivalence, RoPE, GQA) | ~5 s |
| [`train.py`](train.py) | Trains `model.py`; default task needs an induction head | ~2 min |

## Suggested order

```bash
python why_nonlinear.py     # why a nonlinearity is required
python test_model.py        # is the hand-rolled Transformer correct
python vanilla_demo.py      # what encoder-decoder and cross-attention do
python train.py             # does the modern decoder-only actually learn
```

## Why the figures are generated

Nothing in [`make_figures.py`](make_figures.py) is hand-drawn. Every boundary, point and contour comes from a real training run, so the pictures cannot drift away from the text. Change the model, rerun, the figures follow.

## The traps

The four things that actually break in a from-scratch Transformer, each covered by `test_model.py`:

1. **Causality** — perturbing token $t$ must leave logits at positions $< t$ **exactly** unchanged (0.0, not 1e-7)
2. **KV-cache equivalence** — incremental decode must reproduce the one-shot forward
3. **RoPE's relative property** — $\langle R_i q, R_j k\rangle$ may depend only on $i-j$
4. **GQA grouping** — `repeat_kv` must duplicate **contiguously**, or query heads pair with the wrong KV head

Two more that aren't unit-testable but bite just as often:

- `cache.pos` advances once per forward pass (*after* the layer loop). Inside `update()` it advances `n_layer` times.
- `LambdaLR` multiplies the **base** lr by your lambda. Base 0 pins the learning rate at 0 forever, and dropout noise keeps the loss looking alive.
