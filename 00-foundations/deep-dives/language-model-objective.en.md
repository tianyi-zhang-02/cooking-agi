# Language-model objectives, training, and generation

[中文](language-model-objective.md) · **English**

> Reading time: ~10 min · Level: advanced · Last reviewed: 2026-08

<div class="lesson-recipe advanced">
  <div><span>What we are dissecting</span><strong>Why one model has separate training and generation paths</strong></div>
  <div><span>Prerequisites</span><strong>causal LM · cross-entropy · attention mask</strong></div>
  <div><span>Main mechanism</span><strong>loss weighting · teacher forcing · KV cache</strong></div>
  <div><span>Decision to make</span><strong>Fix data, sampling, SFT, or a sequence-level objective?</strong></div>
</div>

Training and generation use the same factorization:

$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$$

The difference is execution. Training and prefill know the whole sequence and run positions in parallel under a causal mask. Autoregressive decode does not know future tokens, queries the KV cache, and remains sequential.

Token cross-entropy means frequent tokens and long samples contribute more training positions. Data mixing, sample weighting, and loss masking therefore alter learned behavior directly. Perplexity cannot be compared naively across tokenizers because the unit of prediction differs.

Teacher forcing trains on true prefixes; generation continues from model-produced prefixes. Before reaching for RL, separate missing behavior coverage, broken templates or masks, unsuitable sampling, and genuinely sequence-level objectives.

Incremental decode should match a full forward pass within numerical tolerance. Correct cache positions, K/V order, causal visibility, per-layer position consistency, and beam reordering are invariants—not optional optimizations. [`../code/test_model.py`](../code/test_model.py) checks this equivalence.

## Self-check

<div class="taste-check advanced">
  <strong>When generation quality fails, ask:</strong>
  <ol>
    <li>What evidence separates missing training coverage from bad sampling?</li>
    <li>Why is perplexity not directly comparable across tokenizers?</li>
    <li>How does full-forward equivalence test KV cache more strongly than plausible text?</li>
  </ol>
</div>

Continue to [Post-training](../../05-post-training/README.en.md).
