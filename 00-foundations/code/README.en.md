# Build-it-yourself labs: from symbols to generation

[中文](README.md) · **English**

> Type: lab index · Runtime: CPU is enough · Last reviewed: 2026-08

I like to write the same thing twice. The first time without PyTorch, forcing myself to see where every number comes from; the second time handing it over to tensors, modules, and autograd so that it actually learns.

High-level APIs are of course not a bad thing. But if every core computation is wrapped from the very start, then when a shape, mask, or state goes wrong later, it is hard to locate why training does not converge.

## Stage one: verify the computation with pure Python and NumPy

| File | Dependency | What it verifies |
| --- | --- | --- |
| [`tokenizer_from_scratch.py`](tokenizer_from_scratch.py) | Python standard library | BPE merge rules, vocabulary, encode/decode |
| [`sequence_numpy.py`](sequence_numpy.py) | NumPy | unrolled RNN, LSTM gates, scaled dot-product attention, causal mask |

These first two files do not use autograd. Every hidden state, gate, and attention weight can be traced directly from the equation to the array, which makes them the place to see the mechanism clearly before the training loop drowns it out.

```bash
python tokenizer_from_scratch.py
python sequence_numpy.py
```

## Stage two: train it with PyTorch

| File | What it does | Suggested command |
| --- | --- | --- |
| [`sequence_torch.py`](sequence_torch.py) | hand-implemented RNN/LSTM cells; trains delay-copy or encoder–decoder reversal | `python sequence_torch.py --model lstm --task reverse` |
| [`vanilla.py`](vanilla.py) | the 2017 encoder–decoder Transformer | called by `vanilla_demo.py` |
| [`vanilla_demo.py`](vanilla_demo.py) | shape tracing, sequence reversal, cross-attention alignment | `python vanilla_demo.py` |
| [`model.py`](model.py) | modern decoder-only: RMSNorm + RoPE + GQA + SwiGLU + KV cache | called by the tests and the training script |
| [`test_model.py`](test_model.py) | causality, cache equivalence, RoPE relativity, GQA grouping | `python test_model.py` |
| [`train.py`](train.py) | trains the decoder-only model on an induction / copy task | `python train.py` |

These implementations do not call `nn.MultiheadAttention` or `F.scaled_dot_product_attention`. PyTorch is responsible only for tensors, parameter management, and autograd; the model structure stays explicitly visible.

## Quick verification

```bash
python test_learning_path.py
python test_model.py
```

`test_learning_path.py` checks the tokenizer round-trip, the RNN/LSTM shapes, that the upper triangle of causal attention is strictly 0, and the output contract of the two PyTorch cells.

## Suggested run order

```text
tokenizer_from_scratch.py
        ↓
sequence_numpy.py
        ↓
sequence_torch.py --task delay
        ↓
sequence_torch.py --task reverse
        ↓
vanilla_demo.py
        ↓
test_model.py → train.py
```

## Why keep two implementations

The framework-free version is good for answering "where does every quantity in the equation come from"; the PyTorch version is good for answering "how are parameters registered, how do gradients flow, and how is the training loop organized." Look only at the first and you stop at a forward-pass demo; look only at the second and it is easy to misdiagnose shape, mask, and state bugs as framework problems.

## The four Transformer mistakes that most easily "look like they run"

1. **Causality**: change token $t$, and the logits at positions $<t$ must stay strictly unchanged.
2. **KV cache equivalence**: incremental decoding must reproduce the one-shot forward pass.
3. **RoPE relativity**: the positional part of the attention score depends only on relative distance.
4. **GQA grouping**: query heads must map to the correct KV group.

The figures are still generated from actual computation by [`make_figures.py`](make_figures.py) and [`make_arch_figures.py`](make_arch_figures.py), which keeps the text and the implementation from drifting apart.
