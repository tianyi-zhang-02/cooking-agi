# Decoder-only: autoregressive generation

[中文](decoder-only.md) · **English**

> Reading time: ~9 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>What we are making</span><strong>One next-token objective for understanding and generation</strong></div>
  <div><span>Ingredients</span><strong>token stream · causal mask · shifted targets</strong></div>
  <div><span>Core technique</span><strong>LM loss · prefill · KV cache · sampling</strong></div>
  <div><span>Most common mistake</span><strong>Assuming parallel training implies parallel generation</strong></div>
</div>

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

Use [`../code/model.py`](../code/model.py), [`../code/test_model.py`](../code/test_model.py), and [`../code/train.py`](../code/train.py).

## Taste check

<div class="taste-check">
  <strong>Carry these three distinctions forward:</strong>
  <ol>
    <li>Why are inputs and labels shifted by exactly one token?</li>
    <li>Why do prefill and decode have different performance profiles?</li>
    <li>Do temperature, top-k, and top-p change the model or how its distribution is read?</li>
  </ol>
</div>

Then read [language-model objectives and generation](../deep-dives/language-model-objective.en.md).
