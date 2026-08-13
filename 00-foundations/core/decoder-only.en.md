# Decoder-only: autoregressive generation

[中文](decoder-only.md) · **English**

> Reading time: ~9 min · Level: core · Last reviewed: 2026-08

## In one sentence

A decoder-only model consumes one token stream, uses a causal mask so each position sees only the left prefix, and predicts the next token at every position.

$$\mathcal L_{\text{LM}}=-\sum_{t=1}^{T-1}\log p_\theta(x_{t+1}\mid x_{\le t})$$

Input and target are the same sequence shifted by one position. Placing instructions, context, and answers in one stream turns conditional generation into the same autoregressive objective.

## Prefill and decode

- **Prefill:** the full prompt is known, so positions run in parallel and each layer stores K/V.
- **Decode:** each new token queries the historical KV cache, then sampling chooses the next token. This path is sequential.

The final hidden state becomes vocabulary logits through a linear head. Temperature changes sharpness; top-$k$ and top-$p$ truncate the sampling set. Sampling changes inference, not the learned distribution.

## The path to post-training

Pre-training, SFT, preference learning, and RL usually preserve the decoder-only backbone. They change the data distribution, loss, comparison structure, and which tokens receive gradient.

Use [`../code/model.py`](../code/model.py), [`../code/test_model.py`](../code/test_model.py), and [`../code/train.py`](../code/train.py). Then read [language-model objectives and generation](../deep-dives/language-model-objective.en.md).
