# Decoder-only: autoregressive generation

[中文](decoder-only.md) · **English**

> Reading time: ~9 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>What we are making</span><strong>One next-token objective for understanding and generation</strong></div>
  <div><span>Prerequisites</span><strong>token stream · causal mask · shifted targets</strong></div>
  <div><span>Core technique</span><strong>LM loss · prefill · KV cache · sampling</strong></div>
  <div><span>Most common mistake</span><strong>Assuming parallel training implies parallel generation</strong></div>
</div>

## Quick learning: the complete generation path of a modern LLM

<details class="interview" markdown="1">
<summary>Messages to logits, then KV-cached decoding</summary>

**Quick memory**: a chat template places every role in one sequence; a causal Transformer predicts the next token at every position; prefill processes the prompt in parallel and decode adds one token at a time while reusing the KV cache.

**Interview answer**

> A decoder-only model serializes system, user, assistant, and tool messages into one context and uses causal self-attention so every position sees only its left prefix. Training predicts all next-token labels in parallel; inference prefills the prompt and then decodes token by token, caching each layer's historical K and V projections.

<details markdown="1">
<summary><b>Deep dive</b>: what does the KV cache store, and why not Q?</summary>

Future queries repeatedly read historical K and V, so caching avoids recomputing their projections. A query is used only by the current token; the next step creates a new query, so there is no cross-step reuse. Caching changes computation cost, not attention semantics, and incremental output must equal a full causal forward pass.

</details>
</details>

## Autoregressive generation in one shared sequence

A decoder-only model consumes one token stream, uses a causal mask so each position sees only the left prefix, and predicts the next token at every position.

$$\mathcal L_{\text{LM}}=-\sum_{t=1}^{T-1}\log p_\theta(x_{t+1}\mid x_{\le t})$$

Input and target are the same sequence shifted by one position. Placing instructions, context, and answers in one stream turns conditional generation into the same autoregressive objective.

## Prefill and decode

- **Prefill:** the full prompt is known, so positions run in parallel and each layer stores K/V.
- **Decode:** each new token queries the historical KV cache, then sampling chooses the next token. This path is sequential.

The final hidden state becomes vocabulary logits through a linear head. Temperature changes sharpness; top-$k$ and top-$p$ truncate the sampling set. Sampling changes inference, not the learned distribution.

## How post-training acts on the same architecture

Pre-training, SFT, preference learning, and RL usually preserve the decoder-only backbone. They change the data distribution, loss, comparison structure, and which tokens receive gradient.

Use [`../code/model.py`](../code/model.py), [`../code/test_model.py`](../code/test_model.py), and [`../code/train.py`](../code/train.py).

## Self-check

<div class="taste-check">
  <strong>Carry these three distinctions forward:</strong>
  <ol>
    <li>Why are inputs and labels shifted by exactly one token?</li>
    <li>Why do prefill and decode have different performance profiles?</li>
    <li>Do temperature, top-k, and top-p change the model or how its distribution is read?</li>
  </ol>
</div>

Then read [language-model objectives and generation](../deep-dives/language-model-objective.en.md).
