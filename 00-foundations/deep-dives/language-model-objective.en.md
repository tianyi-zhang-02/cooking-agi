# Language-model objectives, training, and generation

[中文](language-model-objective.md) · **English**

> Reading time: ~10 min · Level: advanced · Last reviewed: 2026-08

<div class="lesson-recipe advanced">
  <div><span>What we are dissecting</span><strong>Why one model has two execution paths, training and generation</strong></div>
  <div><span>Prerequisites</span><strong>causal LM · cross-entropy · attention mask</strong></div>
  <div><span>Main mechanism</span><strong>loss weighting · teacher forcing · KV cache</strong></div>
  <div><span>Decision to make</span><strong>Whether the problem calls for fixing data, sampling, SFT, or a sequence-level objective</strong></div>
</div>

## Quick learning: one factorization, two execution paths

<details class="interview" markdown="1">
<summary>Training, prefill, decode, and loss weighting</summary>

**Quick memory**: all three predict the same conditional distribution. Training and prefill know the complete input and can run in parallel; decode does not know the future and can only generate step by step. Token CE also implicitly weights by token frequency and sample length.

**Interview answer**

> A causal LM always models $p(x_t\mid x_{<t})$. Training computes CE for all positions of a known sequence in parallel, prefill builds the KV cache for the prompt in parallel, and decode computes once per step, only for the new token. For a quality problem, first separate data coverage, loss weighting, sampling, and cache correctness, and only then decide whether a sequence-level objective is needed.

<details markdown="1">
<summary><b>Deep dive</b>: why does plausible text not prove that the KV cache is correct?</summary>

A slight error in position, mask, or K/V ordering can still generate fluent text while quietly changing the logits. The correct invariant is: for the same prefix, the per-step logits of incremental decode and of a full causal forward agree within numerical tolerance; after a beam reorder, each sequence's cache must also correspond to the correct history.

</details>
</details>

## The core question: why training and generation run at different rhythms

Training and generation use the same parameterized distribution:

$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$$

The only difference is whether the future is known. In training the whole ground-truth sequence is already laid out on the table; in generation the next token does not exist yet, so the model can only write one, look at it, and then keep writing.

| | training / prefill | autoregressive decode |
| --- | --- | --- |
| Input | a whole span of known tokens | the one or few most recently generated tokens |
| attention | parallel under a causal mask | queries the historical KV cache |
| Main bottleneck | compute, activation memory | memory bandwidth, cache, number of serial steps |
| Sources of error | data and objective | additionally, sampling and error accumulation |

## Dissection one: cross-entropy does not decide “what matters” for you

The loss at a single position:

$$\ell_t=-\log p_\theta(y_t\mid x_{\le t})$$

The gradient with respect to the logits is still $p-y$; the formula is clean. But once all positions are summed, frequent tokens and long samples naturally hold more “votes”. So data mixing, sample weighting, and loss masking are not leftovers in the corner of a training script — they are what defines whom the model should actually care about.

Perplexity is the exponential of the average token negative log-likelihood:

$$\text{PPL}=\exp\!\left(\frac{1}{N}\sum_t \ell_t\right)$$

Different tokenizers have different token units, so perplexity cannot be compared directly across them.

## Dissection two: training uses the true prefix; generation depends on the model's own output

In training the model always predicts on top of the true prefix; in generation it must continue from its own output. A low-probability error at one step can carry the subsequent context into a region that the training data rarely covers.

It is very easy here to say, by reflex, “then use RL”. I hold off first and check where the problem actually comes from:

1. the training data does not cover the target behavior;
2. the loss mask or the template is wrong;
3. the sampling strategy is unsuitable;
4. a long-horizon objective cannot be expressed by per-token likelihood.

Only the fourth category truly points to a sequence-level preference or an RL objective.

## The correctness floor: a cache is not “fine as long as it runs”; it must be equivalent

Incremental decoding must be equivalent to a one-shot forward pass within numerical tolerance. All of the following must hold at once:

- the new query uses the correct absolute position;
- the token order of K/V in the cache is unchanged;
- the causal mask lets the query see all of the past and itself;
- every layer reads the same cache position;
- after a batch reorder, the cache is reordered in sync.

[`../code/test_model.py`](../code/test_model.py) aligns the full forward pass with token-by-token decode, which is stronger correctness evidence than “the generation looks normal”.

## What post-training actually changes

SFT, preference learning, and RL are not a separate theory of models. On the same autoregressive distribution they change: where the training samples come from, which tokens count toward the loss, how different outputs are compared, and how the return is assigned across a whole trajectory.

Continue with [Post-Training](../../05-post-training/README.en.md).

## Self-check

<div class="taste-check advanced">
  <strong>When you hit a generation-quality problem, first ask:</strong>
  <ol>
    <li>In terms of evidence, how do you tell a training objective that never covered the behavior apart from sampling that chose badly?</li>
    <li>Why can perplexity not be compared directly across different tokenizers?</li>
    <li>How do you verify a KV cache with full forward versus incremental decode, rather than only checking whether the text looks plausible?</li>
  </ol>
</div>
