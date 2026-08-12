# Foundations: what is the model actually computing?

[中文](README.md) · **English**

## In one sentence

From linear regression to the Transformer, **the last step never changed** — it is always a linear classifier. What changes is how elaborate the coordinate transform in front of it is.

## The cooking version

Say you want to pick out the ingredients that could go into a dessert.

- **Linear / logistic regression**: you may only cut along one straight line — "sugar content > 10". Separates salt from sugar; cannot express "tomatoes work in both a dessert and a stew".
- **Hand-built features**: you invent a new axis yourself, say `sugar × acidity`, and cut a line in that. Works — but you had to guess the right axis.
- **Neural network**: you stop guessing. The network learns which axes to look at, and then still cuts one straight line in the space it learned.
- **Transformer**: same single cut at the end; the machinery that decides what to look at is now dozens of attention layers.

The whole story of deep learning compresses to: **hand feature engineering over to gradient descent**.

## What's here

### [From linear models to neural networks](from-linear-to-neural.en.md)

Why logistic regression still draws a straight boundary. What the sigmoid actually buys (not expressiveness — gradients). Whether interaction terms count as nonlinear. What a hidden layer does geometrically.

With a runnable XOR experiment: logistic regression 50%, a 33-parameter *linear* network still 50%, the same 33-parameter network with one ReLU 100%.

### [The Transformer architecture](transformer.en.md)

The 2017 encoder-decoder original versus today's decoder-only stack. What each of the three attention sites does, why post-norm requires warmup, positional encoding from sinusoids to RoPE, and how the KV cache works.

### [`code/`](code/): build it by hand

Two Transformers written without `nn.MultiheadAttention` or `F.scaled_dot_product_attention` — only `nn.Linear` and raw tensor ops — plus the checks that prove they are correct (causality, KV-cache equivalence, RoPE's relative-position property).

Once the encoder-decoder learns to reverse a sequence, cross-attention grows an anti-diagonal on its own:

```
        1  2  3  4  5  6  7  8   <- source (encoder)
  BOS                        @
    1                     @
    2                  @
    3               @
    4            #  :
    5         @
    6      @
    7   @
  ^ decoder step
```

## Why start here

Every later chapter is about **systems**: how data arrives, how memory is organised, what retrieval looks for, how to evaluate. All of that assumes you know what the internal representation is.

- [Memory](../02-memory/) talks about important signal being "flattened into an average" — the thing being flattened is $h$.
- [Post-training](../05-post-training/) modifies the coordinate transform, not the final linear layer.
- [Evaluation](../07-evaluation/) resists a single headline number partly because the same $h$ is separable to very different degrees across slices.

## Papers to start with

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE
- [GQA](https://arxiv.org/abs/2305.13245)
