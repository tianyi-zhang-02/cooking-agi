# Decoding strategies: temperature, top-k, top-p

[中文](decoding.md) · **English**

> Reading time: ~8 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>The problem</span><strong>The model gives you a probability distribution; how do you pick one token from it</strong></div>
  <div><span>Prerequisites</span><strong>softmax · autoregressive generation</strong></div>
  <div><span>Core mechanism</span><strong>Reshape the distribution first (temperature), then truncate (top-k / top-p), and finally sample</strong></div>
  <div><span>Common mistakes</span><strong>Thinking temperature acts on probabilities; using beam search for open-ended generation</strong></div>
</div>

## Quick learning: model probabilities and the decoding policy are two layers

<details class="interview" markdown="1">
<summary>The spine of temperature, top-k, and top-p, and the order they compose in</summary>

**Quick memory**: the model produces logits; temperature changes their relative sharpness; top-k/top-p truncate the candidate set; sampling is the step that actually makes the random choice.

**Interview answer**

> Decoding does not change the model's parameters; it only turns the same logits into different selection policies. The usual order is to apply penalties first, then divide by the temperature, then do top-k or top-p filtering, renormalize, and sample. As temperature approaches 0 the behavior approaches argmax, but temperature in itself is not greedy decoding.

<details markdown="1">
<summary><b>Deep dive</b>: why does top-p adapt to context better than a fixed top-k?</summary>

When the distribution is confident, a few tokens already cover the probability mass, and top-p automatically keeps a very small set; when the distribution is flat, it keeps more candidates. A fixed $k$ cannot fit both entropy levels at once. The price is that the number of candidates varies with context, so throughput and reproducibility are harder to control.

</details>
</details>

## The model outputs a probability distribution; the decoding strategy does the choosing

After each forward step, the model outputs a probability distribution over the vocabulary, for example:

```text
" the"   0.42
" a"     0.18
" my"    0.09
" one"   0.05
...      (the other 50k tokens share the remaining 0.26)
```

**The model's job ends here.** Which token gets picked is the decoding strategy's business and has nothing to do with the model weights. The same model with different decoding parameters can behave like two different models — which is also why, when you reproduce someone else's results, the decoding parameters matter as much as the weights.

## Temperature: adjusting how steep the distribution is

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**The key point: $T$ divides the logits, before the softmax; it does not rescale the probabilities afterwards.** The two are not equivalent — scaling afterwards and renormalizing only gives back the original distribution.

| $T$ | Effect |
| --- | --- |
| $T \to 0$ | the distribution approaches one-hot; equivalent to greedy |
| $T < 1$ | the distribution gets steeper, high-probability tokens dominate more, and the output is more conservative |
| $T = 1$ | the model's original distribution |
| $T > 1$ | the distribution gets flatter, long-tail tokens get a chance, and the output is more divergent |

Intuitively, $T$ is the amplification factor on “how confident the model is in itself”. It does not change the **ranking**, only the gaps.

> In an implementation $T=0$ would divide by zero, so `temperature=0` in most APIs actually takes the greedy branch rather than really performing the division.

## Top-k: keep only the k most probable

Sort the distribution, keep the top $k$, set the rest to zero, renormalize, and then sample.

The problem is that $k$ is fixed, while the shape of the distribution keeps changing:

- After “The United States of Ame” there is almost exactly one reasonable choice, and the distribution is extremely steep. Here $k=50$ also puts 49 nearly impossible words into the candidate pool.
- After “He thought this movie” there are hundreds of reasonable continuations, and the distribution is very flat. Here $k=50$ cuts too hard and removes options that were perfectly reasonable.

**No single $k$ can fit both cases at once.**

## Top-p (nucleus sampling): truncate by cumulative probability

Change the rule: accumulate probabilities from highest to lowest, take **the smallest set whose cumulative probability just reaches ≥ p**, discard the rest, and renormalize.

$$\text{keep the smallest set } V^{(p)} \text{ such that} \sum_{i \in V^{(p)}} p_i \ge p$$

With the same $p = 0.9$:

- when the distribution is steep, the first token alone may already have 0.93, and the candidate pool holds just **1** token;
- when the distribution is flat, it may take accumulating down to the 80th token to reach 0.9, and the candidate pool holds **80** tokens.

**The size of the candidate pool scales automatically with the shape of the distribution**, which is exactly why top-p works better than top-k. What it tunes is not “how many to keep” but “how much confidence to keep”.

The two can be stacked: top-k first to cap the upper bound, then top-p to tighten dynamically. Many implementations chain them this way by default.

## The usual processing order

Plenty of material online says “top-k first, then top-p, and temperature last”. **Actual implementations do it the other way round.**

In `transformers 4.44.1` the actual order of the warpers is:

```text
Temperature → TopK → TopP → MinP
```

Temperature comes **first**. This is not an inconsequential detail, because it changes how large the nucleus cut out by top-p is.

For top-k it makes no difference — temperature is a monotone transformation that does not change the ranking, and top-k looks only at rank. But **top-p looks at cumulative probability**, and once the temperature changes, the rate of accumulation changes with it.

Measure it on a concrete distribution (8 tokens, `top_p=0.9`):

| temperature | temperature first → nucleus size | truncation first → nucleus size |
| --- | --- | --- |
| 0.7 | **4** | 5 |
| 1.0 | 5 | 5 |
| 1.5 | **6** | 5 |

With the order reversed, the same parameters give a different candidate pool. At $T=1$ the two happen to be equal, so testing only at the default temperature will never reveal this difference.

**Conclusion: tune temperature and top-p as a pair**, because temperature really does change the behavior of top-p; the reverse is not true.

## Frequency penalty and presence penalty

Another family of methods leaves the shape of the distribution alone and instead **directly pushes down the logits of tokens that have already appeared**:

$$z_i \leftarrow z_i - \alpha_{\text{presence}}\cdot\mathbb{1}[c_i > 0] - \alpha_{\text{frequency}}\cdot c_i$$

where $c_i$ is the number of times token $i$ has appeared in the text generated so far. The difference between the two is right there in the formula:

- **Presence penalty**: subtract a **fixed amount** as soon as the token has appeared at all; 1 occurrence and 10 occurrences cost the same. Its effect is to push the model toward new topics.
- **Frequency penalty**: subtract in proportion to the **accumulated count**; the more often a token has appeared, the harder it is pushed down. Its effect is to suppress repetition.

Use the presence penalty when you “want the content to touch more topics”, and the frequency penalty when you “want the same word to stop recurring”.

⚠️ Both are blunt instruments, because **some repetition is supposed to happen**. Punctuation, pronouns, indentation and brackets in code, and a person's name that the text keeps mentioning are all meant to occur frequently. Turn the penalties up too far and the model starts avoiding these necessary tokens; the output becomes awkward or even ungrammatical. Code generation is especially intolerant of this.

A typical starting point: set both to 0, raise them only once obvious repetition appears, by 0.1–0.3 at a time; you rarely need more than 1.0.

## Beam search: why generative LLMs rarely use it

Beam search maintains $k$ candidate sequences at once. After each expansion step it keeps the best $k$ by **cumulative log-probability**, and at the end it outputs the one with the highest overall probability.

It is standard practice in tasks such as machine translation and summarization — because there **a correct answer exists**, and the highest-probability sequence is exactly what you want.

In open-ended generation, however, it fails, and fails in a characteristic way: **the output is dry and repetitive**.

The reason is that “the most likely sentence” and “a good sentence” are simply not the same thing. Human speech carries uncertainty by nature; a sequence that picks the maximum probability at every step reads like a bland template. This phenomenon is called the likelihood trap — studies have found that in human-written text the token probabilities **rise and fall**, while the probability curve of beam-search output sits smoothly at the high end.

So: **use beam search for tasks with a single correct answer, and sampling for open-ended generation.**

## How to set the parameters in practice

| Scenario | temperature | top-p | Notes |
| --- | --- | --- | --- |
| code, math, structured extraction | 0 | — | you want determinism and reproducibility; sampling only introduces errors |
| factual QA, classification | 0–0.3 | 1.0 | leave a little room, but do not encourage divergence |
| general conversation | 0.7 | 0.9 | the most common default combination |
| creative writing, brainstorming | 0.9–1.1 | 0.95 | diversity is explicitly wanted |
| RL rollout / self-consistency voting | 1.0 | 1.0 | the full distribution **must** be kept, otherwise the samples are biased |

A few easy traps:

**Do not set both temperature and top-p very low at the same time.** Both are tightening; stacked together they often degenerate straight into greedy. Diversity disappears entirely, while you still think you are sampling.

**Sampling parameters during RL training affect the gradient.** If the distribution used for rollouts has been truncated by top-p, it is no longer the policy $\pi_\theta$ itself, and the importance ratio comes out wrong. This is a quiet source of bugs in RLHF implementations.

**To reproduce a result, fix the random seed and record every decoding parameter.** Recording only the model version is not enough — the same weights with `T=0.7` and with `T=1.0` are two different behaviors.

<details markdown="1">
<summary><b>deeper</b>: what other truncation methods are there</summary>

**min-p**: keep every token whose probability is ≥ `min_p × highest probability`. Compared with top-p it follows the peak height of the distribution more directly, and it is more stable at high temperature.

**repetition penalty / no-repeat n-gram**: directly push down the logits of tokens that have already appeared. Effective but blunt — it also penalizes words that are supposed to repeat (pronouns, punctuation, indentation in code).

**typical sampling**: filter tokens by how far their information content deviates from the conditional entropy, on the grounds that human language tends to keep its information density stable.

All of these operate at the same place: **modify the logits or modify the candidate set, then sample**. Once you understand temperature and top-p, the rest are variants of the same kind of operation.

</details>

## Common interview questions

<details class="interview" markdown="1">
<summary>Does temperature act on the logits or on the probabilities? Why?</summary>

On the logits, **before** the softmax: $p_i = \text{softmax}(z_i/T)$.

You cannot rescale the probabilities afterwards and then renormalize — that amounts to normalizing $p_i^{1/T}$, which does change the steepness but is not the same transformation, and is numerically less stable. The more common mistake is to multiply the probabilities by a coefficient and then renormalize; that operation **changes nothing**, because normalization cancels the coefficient.

</details>

<details class="interview" markdown="1">
<summary>What is the difference between top-k and top-p? Why is top-p more common?</summary>

Top-k keeps a fixed $k$ candidates; top-p keeps the smallest set whose cumulative probability just reaches $p$.

The difference is **whether the candidate pool changes with the shape of the distribution**. When the distribution is very steep (the next word is almost unique), top-k forces in a pile of impossible candidates; when it is very flat (many reasonable continuations), top-k cuts too hard. Top-p gives a suitable pool size automatically in both cases.

</details>

<details class="interview" markdown="1">
<summary>Why do large models not use beam search for open-ended generation?</summary>

Because “the highest-probability sequence” is not “a good answer”. Beam search looks for the global maximum likelihood, and the result is flat, repetitive text — token probabilities in human language rise and fall by nature, and picking the maximum all the way yields a template that nobody actually talks like.

Tasks **with a single correct answer**, such as machine translation and summarization, still use it; there the most likely sequence really is what you want.

</details>

<details class="interview" markdown="1">
<summary>Are temperature=0 and greedy the same thing?</summary>

Mathematically they are equivalent in the limit: as $T \to 0$ the softmax approaches one-hot, and sampling necessarily returns the argmax.

In an implementation, though, $T=0$ divides by zero, so `temperature=0` in frameworks and APIs usually goes straight to the greedy branch. Note also that greedy is not necessarily fully deterministic either — the floating-point reduction order under batching and different kernel implementations can both make the same input produce different results.

</details>

<details class="interview" markdown="1">
<summary>Which acts first: temperature, top-k, or top-p? Does the order affect the result?</summary>

In actual implementations **temperature comes first** (the order in `transformers` is Temperature → TopK → TopP → MinP). Plenty of material says “temperature is applied last”, which is backwards.

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

Usually `temperature=1.0, top_p=1.0`, that is, no truncation.

Because the importance ratio in the policy gradient, $\rho_t = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$, assumes that the actions were sampled from $\pi_\theta$. Once top-p truncation is applied, the actual sampling distribution is no longer $\pi_\theta$, while the log-probs are still computed from the full distribution, so the ratio comes out wrong.

This is a bug that raises no error and only makes training drift off course slowly.

</details>

## Self-check

<div class="taste-check">
  <strong>If you really understand this, you should be able to explain:</strong>
  <ol>
    <li>At which step does temperature act? Why is multiplying the probabilities by a coefficient useless?</li>
    <li>With the same 0.9, how large is the top-p candidate pool on a steep distribution and on a flat one?</li>
    <li>On which tasks is beam search appropriate, and on which does it tend to fail? Why?</li>
    <li>Why must top-p truncation be off for RL rollouts?</li>
  </ol>
</div>

## Where to read next

- [Decoder-only: autoregressive generation](decoder-only.en.md): how these distributions are produced step by step
- [The three stages of RLHF](../../05-post-training/rlhf/README.en.md): why rollout sampling parameters affect the gradient
