# Looped Transformers: reusing one block

[中文](README.md) · **English**

> Reading time: ~2 min · Level: advanced · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>deeper computation without storing more parameters</strong></div>
  <div><span>Prerequisites</span><strong>Transformer block · residual stream · autoregressive decoding</strong></div>
  <div><span>Core mechanism</span><strong>reuse the same weights L times: depth and compute grow with L, parameters do not</strong></div>
  <div><span>Common mistakes</span><strong>thinking more loops store more knowledge; thinking looping leaves the KV cache the same size</strong></div>
</div>

## Depth and parameters normally come as a pair

In an ordinary $L$-layer Transformer every layer has its own weights. To make the model do more serial steps of computation per token, you add layers, and parameters grow with them.

## Loop the block

A looped Transformer stores a single $k$-layer block and applies it $L$ times:

$$h^{(t+1)} = f_\theta\big(h^{(t)},\, e\big), \qquad t = 0, 1, \ldots, L-1$$

$\theta$ is the same in every loop; $e$ is the input embedding, which many designs feed back in at every loop (input injection). The parameters are $k$ layers; the effective depth and the compute are $kL$ layers.

<!-- widget:tx-loop-unroll -->

## An old idea, used in a new way

- **Universal Transformer** (2018): one transition function shared across all positions and all steps, plus ACT, with which each position decides for itself when to stop.
- **ALBERT** (2019) also shares parameters across layers (all of them, by default), cutting BERT-base's 108M parameters to 12M, but inference compute stays the same. Its aim is fewer parameters, not a depth you can turn up.
- **In the last two years**, looping has moved into large-scale pretraining: looped LMs such as Huginn (3.5B) and Ouro (1.4B, 2.6B) treat the number of loops as compute you can add at inference time.

## How this series reads

1. [Why loops help reasoning](why-loops-reason.en.md): serial steps, the theory, and how it relates to CoT (interactive)
2. [How many loops per token](adaptive-depth.en.md): ACT, PonderNet, Ouro's exit gate, Mixture-of-Recursions (interactive)
3. [What today's looped LMs look like](models.en.md): Huginn, Ouro, Relaxed Recursive Transformers
4. [Costs and limits](costs.en.md): latency, the KV cache, training stability, and why it does not store more knowledge
5. [Review questions](review.en.md): interview questions and a self-check
