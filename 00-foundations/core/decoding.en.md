# Decoding strategies: temperature, top-k, top-p

[中文](decoding.md) · **English**

> Reading time: ~12 min · Level: core · Last reviewed: 2026-10

Suppose the next-token probabilities are `0.6, 0.3, 0.1`. Greedy always chooses the first. Top-k with k=2 keeps the first two; top-p with p=0.8 also needs the first two to reach its probability-mass threshold. Sampling then uses the renormalized distribution. The model has not changed, but its selection rule has.

## The model outputs a probability distribution; the decoding strategy does the choosing {#the-model-outputs-a-probability-distribution-the-decoding-strategy-does-the-choosing}

After each forward step, the model outputs a probability distribution over the vocabulary, for example:

```text
" the"   0.42
" a"     0.18
" my"    0.09
" one"   0.05
...      (the other 50k tokens share the remaining 0.26)
```

With the weights fixed, the decoder can still use different rules to select from this distribution. Reproducing a generation therefore requires the model version and sampling settings; the answer text alone rarely explains why behavior changed.

## Temperature: adjusting how steep the distribution is {#temperature-adjusting-how-steep-the-distribution-is}

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**Usually, divide logits by $T$ before softmax.** Multiplying probabilities by a common factor and renormalizing does nothing; raising them to the power $1/T$ and renormalizing is equivalent to this formula.

| $T$ | Effect |
| --- | --- |
| $T \to 0^+$ | probability concentrates on a unique maximum; tied maxima remain multiple candidates |
| $T < 1$ | the distribution gets steeper, high-probability tokens dominate more, and the output is more conservative |
| $T = 1$ | the model's original distribution |
| $T > 1$ | the distribution gets flatter, long-tail tokens get a chance, and the output is more divergent |

Temperature preserves the **ranking** but changes concentration. Lowering it does not add knowledge or guarantee a more correct answer.

> In an implementation $T=0$ would divide by zero, so `temperature=0` in most APIs actually takes the greedy branch rather than really performing the division.

## Top-k: keep only the k most probable {#top-k-keep-only-the-k-most-probable}

Sort the distribution, keep the top $k$, set the rest to zero, renormalize, and then sample.

The problem is that $k$ is fixed, while the shape of the distribution keeps changing:

- After “The United States of Ame” there is almost exactly one reasonable choice, and the distribution is extremely steep. Here $k=50$ also puts 49 nearly impossible words into the candidate pool.
- After “He thought this movie” there are hundreds of reasonable continuations, and the distribution is very flat. Here $k=50$ cuts too hard and removes options that were perfectly reasonable.

**No single $k$ can fit both cases at once.**

## Top-p (nucleus sampling): truncate by cumulative probability {#top-p-nucleus-sampling-truncate-by-cumulative-probability}

Change the rule: accumulate probabilities from highest to lowest, take **the smallest set whose cumulative probability just reaches ≥ p**, discard the rest, and renormalize.

$$\text{keep the smallest set } V^{(p)} \text{ such that} \sum_{i \in V^{(p)}} p_i \ge p$$

With the same $p = 0.9$:

- when the distribution is steep, the first token alone may already have 0.93, and the candidate pool holds just **1** token;
- when the distribution is flat, it may take accumulating down to the 80th token to reach 0.9, and the candidate pool holds **80** tokens.

**The candidate count adapts to the distribution**, rather than being chosen in advance. But 0.9 means retained model probability mass, not a 90% chance of a correct answer. Whether top-p is preferable still depends on the task and observed outputs.

The two can be stacked: top-k first to cap the upper bound, then top-p to tighten dynamically. Many implementations chain them this way by default.

## The usual processing order {#the-usual-processing-order}

Check the framework instead of treating one order as universal. This comparison explicitly chooses:

```text
Temperature → TopK → TopP
```

Temperature comes **first**. This is not an inconsequential detail, because it changes how large the nucleus cut out by top-p is.

For top-k it makes no difference — temperature is a monotone transformation that does not change the ranking, and top-k looks only at rank. But **top-p looks at cumulative probability**, and once the temperature changes, the rate of accumulation changes with it.

Use probabilities `0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.025, 0.015`, convert them to logits, and set `top_p=0.9`:

| temperature | temperature first → nucleus size | truncation first → nucleus size |
| --- | --- | --- |
| 0.7 | **4** | 5 |
| 1.0 | 5 | 5 |
| 1.5 | **6** | 5 |

With the order reversed, the same parameters give a different candidate pool. At $T=1$ the two happen to be equal, so testing only at the default temperature will never reveal this difference.

Record temperature and top-p together. Fix the processing order before tuning so you can distinguish flattening the distribution from truncating its support.

## Frequency penalty and presence penalty {#frequency-penalty-and-presence-penalty}

Another family of methods uses generation history to **lower logits for tokens that have appeared**, also changing the final distribution:

$$z_i \leftarrow z_i - \alpha_{\text{presence}}\cdot\mathbb{1}[c_i > 0] - \alpha_{\text{frequency}}\cdot c_i$$

where $c_i$ is the number of times token $i$ has appeared in the text generated so far. The difference between the two is right there in the formula:

- **Presence penalty**: subtract a **fixed amount** as soon as the token has appeared at all; 1 occurrence and 10 occurrences cost the same. Its effect is to push the model toward new topics.
- **Frequency penalty**: subtract in proportion to the **accumulated count**; the more often a token has appeared, the harder it is pushed down. Its effect is to suppress repetition.

These penalties track tokens, not topics. Suppressing previously used tokens does not guarantee new ideas; paraphrasing the same idea may evade the penalty.

⚠️ Both are blunt instruments, because **some repetition is supposed to happen**. Punctuation, pronouns, indentation and brackets in code, and a person's name that the text keeps mentioning are all meant to occur frequently. Turn the penalties up too far and the model starts avoiding these necessary tokens; the output becomes awkward or even ungrammatical. Code generation is especially intolerant of this.

Keep an unpenalized baseline and compare outputs on the same prompts. Check the framework's parameter ranges and whether counts include the prompt; there is no universal step size or useful maximum across models.

## Beam search: why generative LLMs rarely use it {#beam-search-why-generative-llms-rarely-use-it}

Beam search expands candidates and retains $k$ by score at each step. The simplest score is **cumulative log-probability**; implementations may add length penalties. The final choice is the best surviving candidate, not necessarily the global optimum.

Consider two steps: A has probability 0.6 and B has 0.4. The best continuation after A has probability 0.5; after B it has 0.9. Greedy commits to A and reaches 0.30. A width-2 beam can retain B and find 0.36. Keeping several prefixes helps, but a pruned prefix is usually gone for good.

Translation and speech recognition often use beam search, not because answers are unique, but because searching by model score can be useful. Score, length handling, and task quality still need to agree. In open-ended generation, likelihood maximization can produce repetition; see [Neural Text Degeneration](https://arxiv.org/abs/1904.09751).

Math does not require greedy decoding either. [Self-consistency](https://arxiv.org/abs/2203.11171) samples multiple reasoning paths and aggregates answers. Choose a strategy together with its candidate budget, verification method, and final metric.

## How to set the parameters in practice {#how-to-set-the-parameters-in-practice}

| Scenario | temperature | top-p | Notes |
| --- | --- | --- | --- |
| Single-answer extraction or classification | greedy baseline | — | Measure correctness and format; consider constrained decoding |
| Code or math with candidate verification | compare low temperature and sampling | experiment-dependent | Report candidate count and verification cost; pass@k is not pass@1 |
| Conversation or writing | sweep a small range | tune jointly | Measure quality and repetition, not diversity alone |
| RL rollout | objective-dependent | objective-dependent | Align sampling, log-probs, and the training objective |
| Self-consistency voting | diverse sampling | truncation is possible | Inference aggregation is not a policy-gradient update |

A few easy traps:

**Do not set both temperature and top-p very low at the same time.** Both are tightening; stacked together they often degenerate straight into greedy. Diversity disappears entirely, while you still think you are sampling.

**Distinguish the target policy from the actual rollout distribution.** Temperature and truncation can change that distribution, so specify which probabilities are recorded and whether correction is applied. Truncation also removes support: adding an importance ratio cannot recover actions that were never sampleable.

**To reproduce a result, fix the random seed and record every decoding parameter.** Recording only the model version is not enough — the same weights with `T=0.7` and with `T=1.0` are two different behaviors.

<details markdown="1">
<summary><b>deeper</b>: what other truncation methods are there</summary>

**min-p**: keep tokens whose probability is ≥ `min_p × highest probability`. This is a threshold relative to the peak, not cumulative mass. Whether it improves high-temperature outputs requires a task-level comparison.

**Repetition penalty and no-repeat n-gram are different operations**. The former changes scores of previously used tokens; the latter bans a next token that would complete a repeated n-gram. After `A B A`, a bigram ban excludes `B`, not every previously seen token. See the [Transformers 4.57.1 generation processors](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.NoRepeatNGramLogitsProcessor).

**Typical sampling**: compare token surprisal $-\log p_i$ with conditional entropy rather than ranking solely by probability. It favors information content near the current expectation; see the assumptions and experiments in [Locally Typical Sampling](https://arxiv.org/abs/2202.00666).

All of these operate at the same place: **modify the logits or modify the candidate set, then sample**. Once you understand temperature and top-p, the rest are variants of the same kind of operation.

</details>

## A top-p implementation with testable boundaries {#top-p-example}

Let vocabulary IDs `0, 1, 2` have probabilities `0.1, 0.6, 0.3`, with threshold 0.8. Sorting selects ID 1, then ID 2: cumulative mass rises from 0.6 to 0.9. **Keep the token that crosses the threshold.** Renormalizing gives probabilities 2/3 and 1/3.

This function computes the final distribution without drawing a random sample, making the result easy to check. It accepts finite logits and is a teaching implementation, not a GPU kernel.

```python
import math

def nucleus_distribution(logits, threshold=0.9, temperature=1.0):
    if not logits or not 0 < threshold <= 1 or temperature <= 0:
        raise ValueError("Expected logits, 0 < threshold <= 1, and temperature > 0")
    if not math.isfinite(temperature) or not all(math.isfinite(value) for value in logits):
        raise ValueError("Expected finite inputs")
    maximum = max(logits)
    weights = [math.exp((value - maximum) / temperature) for value in logits]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    ordered_ids = sorted(range(len(logits)), key=lambda token_id: probabilities[token_id], reverse=True)
    selected_ids = []
    cumulative = 0.0
    for token_id in ordered_ids:
        selected_ids.append(token_id)
        cumulative += probabilities[token_id]
        if threshold < 1 and cumulative >= threshold:
            break
    return {token_id: probabilities[token_id] / cumulative for token_id in selected_ids}

example = nucleus_distribution([math.log(.1), math.log(.6), math.log(.3)], .8)
assert list(example) == [1, 2]
assert math.isclose(example[1], 2 / 3)
```

Return original vocabulary IDs, not sorted positions. Also test `threshold=1` (no truncation), a single candidate, tied maxima, and large positive or negative logits. One convenient example is not enough.

## Common interview questions {#common-interview-questions}

<details class="interview" markdown="1">
<summary>Does temperature act on the logits or on the probabilities? Why?</summary>

On the logits, **before** the softmax: $p_i = \text{softmax}(z_i/T)$.

Given the original softmax probabilities, $p_i^{1/T}/\sum_j p_j^{1/T}$ is **mathematically equivalent** to $\operatorname{softmax}(z/T)$: the common normalization factor cancels. Implementations usually keep logits to avoid probability underflow. Multiplying all probabilities by the same coefficient and renormalizing changes nothing.

</details>

<details class="interview" markdown="1">
<summary>What is the difference between top-k and top-p? Why is top-p more common?</summary>

Top-k keeps a fixed $k$ candidates; top-p keeps the smallest set whose cumulative probability just reaches $p$.

The difference is **whether the candidate pool changes with the distribution**. Top-p can retain few tokens when it is concentrated and many when it is flat. Adaptation does not guarantee quality: a confidently wrong model can still put high probability on the wrong token.

</details>

<details class="interview" markdown="1">
<summary>Why do large models not use beam search for open-ended generation?</summary>

Model likelihood is not task quality; open-ended generation needs checks for repetition and diversity. Beam search is finite-width approximate search, not a global optimizer. Translation and summarization also admit multiple valid answers. Compare with sampling, reranking, or verification under the same budget.

</details>

<details class="interview" markdown="1">
<summary>Are temperature=0 and greedy the same thing?</summary>

With a unique maximum logit, probability concentrates on that token as $T \to 0^+$. Tied maxima share the limiting probability, while greedy implementations typically use a fixed tie-breaking rule.

In an implementation, though, $T=0$ divides by zero, so `temperature=0` in frameworks and APIs usually goes straight to the greedy branch. Note also that greedy is not necessarily fully deterministic either — the floating-point reduction order under batching and different kernel implementations can both make the same input produce different results.

</details>

<details class="interview" markdown="1">
<summary>Which acts first: temperature, top-k, or top-p? Does the order affect the result?</summary>

This article uses temperature → top-k → top-p. Check the framework; interchanging arbitrary steps does not preserve the configuration.

**The order does not matter for top-k**: temperature is a monotone transformation that does not change the ranking, and top-k looks only at rank.

**It does matter for top-p**: top-p looks at cumulative probability; temperature changes each token's relative share, so the number of tokens needed to accumulate to $p$ changes. Measured with the same `top_p=0.9`: at $T=0.7$, temperature first gives 4 candidates and temperature last gives 5; at $T=1.5$ it is 6 versus 5. At $T=1$ the two are equal — so testing only at the default temperature will not reveal it.

</details>

<details class="interview" markdown="1">
<summary>What is the difference between the frequency penalty and the presence penalty?</summary>

Both act on the logits: $z_i \leftarrow z_i - \alpha_p\mathbb{1}[c_i>0] - \alpha_f c_i$.

The **presence penalty** looks only at “has it appeared or not”: 1 occurrence and 10 occurrences cost the same, which pushes the model to change topic. The **frequency penalty** accumulates with the count: the more often a token appears, the harder it is pushed down, which specifically targets repetition.

Both also hit tokens that are supposed to repeat — punctuation, pronouns, code indentation and brackets, a person's name that recurs in the text. Turned up too high, the output becomes awkward, and code generation is especially easy to break.

</details>

<details class="interview" markdown="1">
<summary>How should the sampling parameters be set for rollouts in RL training?</summary>

First identify the actual rollout distribution $\mu$. An untransformed old policy is an easy baseline to audit. Other sampling settings can be deliberate, but the target policy, recorded probabilities, and correction method must be explicit.

In ordinary PPO, the denominator is the old policy that collected the batch, not the current updating policy. Treating samples from $\mu$ as samples from an untransformed old policy changes the estimator. Also check support: actions with zero probability under $\mu$ cannot be recovered from this batch.

</details>

## Self-check {#self-check}

<div class="taste-check">
  <strong>If you really understand this, you should be able to explain:</strong>
  <ol>
    <li>At which step does temperature act? Why is multiplying the probabilities by a coefficient useless?</li>
    <li>With the same 0.9, how large is the top-p candidate pool on a steep distribution and on a flat one?</li>
    <li>On which tasks is beam search appropriate, and on which does it tend to fail? Why?</li>
<li>After changing rollout sampling, what must agree between log-probs and the training objective?</li>
  </ol>
</div>

## Where to read next {#where-to-read-next}

- [Decoder-only: autoregressive generation](decoder-only.en.md): how these distributions are produced step by step
- [The three stages of RLHF](../../05-post-training/rlhf/README.en.md): why rollout sampling parameters affect the gradient

## Quick learning: model probabilities and the decoding policy are two layers {#quick-learning-model-probabilities-and-the-decoding-policy-are-two-layers}

<details class="interview" markdown="1">
<summary>The spine of temperature, top-k, and top-p, and the order they compose in</summary>

**Quick memory**: the model produces logits; temperature changes their relative sharpness; top-k/top-p truncate the candidate set; sampling is the step that actually makes the random choice.

**Interview answer**

> Decoding does not change the model's parameters; it only turns the same logits into different selection policies. The usual order is to apply penalties first, then divide by the temperature, then do top-k or top-p filtering, renormalize, and sample. As temperature approaches 0 the behavior approaches argmax, but temperature in itself is not greedy decoding.

<details markdown="1">
<summary><b>Deep dive</b>: why does top-p adapt to context better than a fixed top-k?</summary>

A concentrated distribution reaches the mass threshold with few tokens; a flat one needs more. Fixed $k$ does not adapt this way. Sampling overhead also depends on sorting, truncation kernels, and batching; a variable candidate count alone does not establish lower serving throughput.

</details>
</details>
