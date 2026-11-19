# What a reward model predicts: preferences, outcomes, steps, and value

[中文](reward-signals.md) · **English**

> Reading time: ~12–15 min · Last reviewed: 2026-11 · Prerequisite: [Preferences and reward models](where-preferences-come-from.en.md)

Two models assign a response a score of 0.8. One might use that score to rank responses by preference; the other might estimate the chance of eventually answering correctly from the current prefix. Even with the same network, changing the labels and objective changes what its output means.

When reading reward-model code, first locate **the source of the labels, the positions included in the loss, and the scoring rule used at inference**. Connecting these three is more useful than starting with the name ORM, PRM, or Critic.

## Start with the supervision, not the output head {#labels}

| Question being answered | Common name | What a training example supplies | What the score does not establish |
| --- | --- | --- | --- |
| Which of two same-prompt responses is preferred? | Preference reward model | Responses and a comparison label | An individual score is not a correctness probability; both responses may be wrong |
| Did this response reach the correct result? | Outcome reward model (ORM) | A response and its final-outcome label | A correct final answer does not validate every intermediate step |
| Is this step, or the reasoning so far, valid? | Process reward model (PRM) | Step boundaries and process labels, whose definitions vary across papers | A score at each step does not imply human step-by-step annotation |
| What return remains if a particular policy continues from here? | Value function | Returns from continuations, or bootstrapped targets | Low value does not necessarily mean the last step was wrong |

**These names do not uniquely specify an architecture.** An ORM may score only the completed response or predict its eventual outcome at multiple prefixes. A PRM may use binary classes, three classes, or continuous rollout-success targets. Value learning is not restricted to on-policy data either; offline evaluation and off-policy methods also estimate values, as discussed in [OPE](deep-rl/offline-and-ope.en.md).

Some concrete implementations from the original papers:

- [Cobbe et al. (2021), §4.3](https://arxiv.org/html/2110.14168v2#S4.SS3) compare predictions at the end of a solution with predictions after every token. Both use outcome supervision.
- [Uesato et al. (2022), §2.4](https://arxiv.org/html/2211.14275v1#S2.SS4) label whether the steps so far are correct, rather than independently judging each step.
- [Lightman et al. (2023), §§2.4–2.6 and Appendix F](https://arxiv.org/html/2305.20050v1) use positive, negative, and neutral step feedback.
- [Snell et al. (2024), §5.1 and Appendix D](https://arxiv.org/html/2408.03314v1#A4) train a scorer on success rates from continuations sampled at step prefixes. They call it a PRM, but its score is closer to a value estimate under a specified continuation policy.

When someone says an ORM must score every token, or a PRM must have three classes, check whether they are describing one implementation rather than a definition.

## A correct answer can still contain invalid reasoning {#same-answer}

The task is to calculate $6\times7$. Both solutions finish with 42:

| Solution A | Solution B |
| --- | --- |
| $6\times5=30$ | $6\times5=32$ |
| Add $6\times2=12$ | Add 10 |
| The answer is 42 | The answer is 42 |

A final-number checker gives both a positive label. A reasoning check catches B's incorrect first step; its later numbers happen to recover the right answer. This does not make outcome supervision useless. It shows that an outcome label does not supply a label asserting the validity of every step.

The converse matters too: a wrong final calculation should not automatically turn all preceding valid steps into negative process labels. If annotation stops at the first error, later steps remain unlabelled, rather than silently becoming correct, incorrect, or neutral.

“This step is valid” also differs from “every step so far is valid.” Under the latter rule, labels may remain negative after the first error; under the former, later steps require judgment under the stated rubric. Mixing these rules can produce conflicting supervision for the same kind of text.

## From labels to loss: work out the mask first {#masked-loss}

Start with a binary implementation. At an included position $i$, the model produces logit $z_i$, prediction $p_i=\sigma(z_i)$, and target $y_i\in[0,1]$. Binary cross-entropy has the stable form:

$$
\begin{aligned}
\ell_i&=\operatorname{softplus}(z_i)-y_i z_i,\\
\frac{\partial\ell_i}{\partial z_i}&=p_i-y_i.
\end{aligned}
$$

Suppose three steps have labels `[1, 0, unlabelled]`. The first two positions both predict 0.75, so $z=\log3$. Average over the two labelled positions only:

| Position | Label | Individual loss | Logit gradient of the mean loss |
| --- | --- | --- | --- |
| End of step 1 | 1 | $-\log0.75\approx0.2877$ | $(0.75-1)/2=-0.125$ |
| End of step 2 | 0 | $-\log0.25\approx1.3863$ | $(0.75-0)/2=0.375$ |
| End of step 3 | Unlabelled | Excluded | 0 |

The mean loss is about 0.8370. The third position contributes to neither numerator nor denominator. Filling its label with zero and averaging over all three positions would train a different objective. **A missing label is not a negative label.**

If the dataset has a genuine neutral class, three-class cross-entropy is one option. Class indices must still be distinct from ignored positions. A neutral judgment is observed information; a step nobody annotated is not.

Check the path from model tensors to supervision in this order:

<div class="drl-flow" aria-label="Aligning reward-model supervision">
<span>Response<br><small>Task and generator version</small></span><b>→</b><span>Step boundaries<br><small>Map to token positions</small></span><b>→</b><span>Labels and mask<br><small>Exclude unknown labels</small></span><b>→</b><span>Loss reduction<br><small>Per step or response?</small></span>
</div>

A classifier can use the hidden state after reading the current step; do not mechanically apply the shift used for next-token prediction. An implementation that generates label tokens needs alignment appropriate to that formulation. Truncated boundary tokens, padding mistaken for step endings, and averaging over every token can all change the intended objective.

## Copying outcome labels to prefixes is not process supervision {#prefix-value}

Let $h$ be a prefix, let policy $\mu$ generate its continuation, and let $Y$ indicate final-answer correctness. If the outcome label supervises each prefix, then under ideal data and model-fitting conditions the cross-entropy optimum is:

$$
p^*(h)=\mathbb E_{\mu}[Y\mid h]
=P_{\mu}(\text{final answer correct}\mid h).
$$

To see why, write $q=P_\mu(Y=1\mid h)$. The expected loss is $\mathbb E[\ell(z,Y)\mid h]=\operatorname{softplus}(z)-qz$. Its derivative is $\sigma(z)-q$, giving $\sigma(z)=q$ at the optimum for $0<q<1$. The cases $q=0$ and $q=1$ are boundary limits as the logit tends to negative or positive infinity.

This asks whether a continuation drawn from the data policy will succeed, not whether the last step is logically valid. With a single terminal binary reward, $\gamma=1$, and no length penalty, KL cost, or other rewards, it coincides with $V^\mu(h)$. Change the reward or discount and value need not remain a success probability.

For a minimal example, take one valid prefix. An older policy succeeds once in four continuations; an improved policy succeeds three times in four. The observed rates are 0.25 and 0.75, but the prefix itself has not changed from wrong to right. The continuation policy improved. Four samples are only teaching counts, not a precise estimate of the underlying rates, but they illustrate why **a value must be read together with its policy**.

This is one reason to reevaluate a scorer trained on old rollouts before applying it to a new policy. A changed generation distribution can invalidate its earlier calibration. Updating a model-version field does not make the old predictions ground truth.

## How do step scores become a response ranking? {#aggregation}

Consider two candidates:

| Candidate | Two step scores | Product | Minimum |
| --- | --- | --- | --- |
| A | 0.70, 0.99 | 0.6930 | 0.70 |
| B | 0.82, 0.82 | 0.6724 | 0.82 |

The product selects A; the minimum selects B. These are different rankings, not interchangeable implementations of “choose the best response with a PRM.”

Interpreting a product as the probability that all steps are correct requires the factors to represent the appropriate conditional probabilities. Scores lying between zero and one are not sufficient. Length and segmentation matter too: a score of 0.9 at every step gives 0.81 for two steps and 0.6561 for four. Splitting one step into two sentences can change the ranking.

If the model predicts eventual success from each prefix, multiplying those predictions generally has no such joint-probability interpretation. They predict the same final event rather than separate independent checks. Snell et al. compare aggregation rules under rollout-derived supervision and find that the final-step prediction works better in their setting. That is a result about their model, targets, and evaluation, not a rule that every PRM should use its last score.

Record all three choices: **how steps are separated, what a score means, and how scores are combined**. Changing any one of them calls for reevaluating the correctness of the selected answers.

## Checks for review and implementation {#audit}

| Observation | First question | A cheap check |
| --- | --- | --- |
| Low step loss, poor answer selection | Are most labels positive, making the majority class easy? | Hold out questions, inspect erroneous steps separately, then measure selected-answer correctness |
| More candidates, lower actual accuracy | Is search finding weaknesses in the scorer? | On fixed questions, plot candidate count, model score, and actual correctness |
| Performance drops with a new generator | Did topics, length, style, or error types change? | Recheck new-generator samples with humans or an independent program |
| Short answers always rank higher | Are products, length, and segmentation coupled? | Rewrite the same reasoning with different step boundaries and check for unjustified ranking reversals |
| ORM and PRM results are hard to compare | Do labels, question counts, generators, and compute budgets match? | Report annotation cost, scoring cost, and answer quality separately |

Keep question IDs, response IDs, generator versions, step boundaries, label provenance, rubric versions, and unknown-label masks. Split by question or a broader task family so that different answers to one question do not leak across train and test. If collection deliberately targets high-scoring wrong answers, record that selection rather than treating the sample proportions as ordinary traffic rates.

For review, explain three distinctions: **final correctness versus valid reasoning; process labels versus rollout value; a better selector versus a better policy**. They lead directly to decisions about what to label, which loss to use, and what the next experiment should establish.

The accompanying [small checks](deep-rl/code/feedback_exploration_checks.py) reproduce masked gradients, aggregation choices, and the continuation-policy example without a GPU. Continue with [reward-model ranking losses](reward-ranking.en.md) or [sampling, voting, and verifiers](../11-frontier/test-time-compute/parallel-and-verify.en.md).
