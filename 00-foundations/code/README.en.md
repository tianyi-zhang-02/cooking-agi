# Build-it-yourself labs: from symbols to generation

[中文](README.md) · **English**

> Type: lab index · Runtime: CPU is enough · Last reviewed: 2026-08

The labs have two layers. First avoid PyTorch and expose every computation. Then use PyTorch tensors, modules, and autograd to make the same mechanisms learn.

## Without PyTorch: see the computation

| File | Dependency | What it verifies |
| --- | --- | --- |
| [`tokenizer_from_scratch.py`](tokenizer_from_scratch.py) | standard library | BPE merges, vocabulary, encode/decode |
| [`sequence_numpy.py`](sequence_numpy.py) | NumPy | unrolled RNN, LSTM gates, scaled dot-product attention, causal mask |

```bash
python tokenizer_from_scratch.py
python sequence_numpy.py
```

Neither file uses autograd. Every hidden state, gate, and attention weight is a direct array translation of the equations.

## With PyTorch: make it learn

| File | Role | Suggested command |
| --- | --- | --- |
| [`sequence_torch.py`](sequence_torch.py) | manual RNN/LSTM cells; delay-copy and seq2seq reversal | `python sequence_torch.py --model lstm --task reverse` |
| [`vanilla_demo.py`](vanilla_demo.py) | 2017 encoder–decoder shapes and cross-attention | `python vanilla_demo.py` |
| [`model.py`](model.py) | modern decoder-only with RMSNorm, RoPE, GQA, SwiGLU, and KV cache | used by tests and training |
| [`test_model.py`](test_model.py) | causality, cache equivalence, RoPE relativity, GQA grouping | `python test_model.py` |
| [`train.py`](train.py) | induction / copy-task training | `python train.py` |

The Transformer implementations do not call `nn.MultiheadAttention` or `F.scaled_dot_product_attention`; PyTorch handles tensors and autograd while the architecture stays explicit.

## Fast verification

```bash
python test_learning_path.py
python test_model.py
```

The recommended order is tokenizer → NumPy sequence math → recurrent training → seq2seq → vanilla Transformer → modern decoder-only. Keeping both versions separates “where each number comes from” from “how gradients and modules are organized in a real training loop.”
