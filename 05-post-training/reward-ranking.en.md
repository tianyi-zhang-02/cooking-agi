# Reward models: training on several responses to one prompt

[中文](reward-ranking.md) · **English**

> Last reviewed: 2026-11 · Prerequisite: [Preferences and reward models](where-preferences-come-from.en.md)

Rank four responses to one prompt and you can extract six preference pairs. The dataset appears six times larger, but these are not six independent judgments. They reuse the same responses to the same question.

We will follow one ranking through three decisions: how to compute the loss, how much weight each prompt receives, and what the resulting scores can compare. Worked examples come first, followed by data and implementation checks. This is not a universal recipe for the best reward model.

## More comparisons can change a prompt's weight {#prompt-weighting}

Suppose prompt A has two candidates, producing one pair. Prompt B has four candidates, producing $\binom42=6$ pairs. Flatten the seven pairs and average their losses, and B gets $6/7$ of the weight.

Let A's pair loss be 0.2 and all six of B's pair losses be 0.8:

| Aggregation | Calculation | Result |
| --- | --- | ---: |
| Equal weight per pair | $(0.2+6\times0.8)/7$ | 0.714286 |
| Average within each prompt, then across prompts | $(0.2+0.8)/2$ | 0.5 |

The appropriate choice depends on the objective. Equal prompt weights are straightforward when each prompt represents an equally important user need. Explicit weighting can favor certain tasks or more reliable labels. What matters is not letting extra candidate sampling silently determine the training distribution.

For prompt $i$ with labeled pair set $\mathcal P_i$, one equal-prompt objective is

$$
L_{\mathrm{pair}}=
\frac1N\sum_{i=1}^N\frac1{|\mathcal P_i|}
\sum_{(w,l)\in\mathcal P_i}\operatorname{softplus}(r_{i,l}-r_{i,w}).
$$

Here $\operatorname{softplus}(z)=\log(1+e^z)$ expresses the Bradley–Terry negative log-likelihood. A prompt with no labeled pairs cannot contribute through a zero denominator; skip and count it, or use it in another supervised task.

[InstructGPT (2022), §3.5](https://arxiv.org/html/2203.02155v1#S3.SS5) reports overfitting when related comparisons were shuffled apart, and groups all comparisons from a prompt into one batch element. That does not mean an entire batch must contain only one prompt, or that the whole dataset is processed at once. Grouping also lets several comparisons reuse one reward forward pass per candidate. The reported improvement belongs to that experiment, not a guarantee for every reward model.

## Calculating a probability for a complete ranking {#listwise}

A debugging prompt has three responses, labeled $A\succ B\succ C$. Suppose the reward model currently outputs

$$
r_A=\ln4,\qquad r_B=\ln2,\qquad r_C=0.
$$

These are convenient illustrative outputs, not prescribed target scores. The Plackett–Luce (PL) model selects the next ranked item successively: first from three candidates, then from the two remaining.

<div class="drl-flow" aria-label="Probability of a three-response ranking">
<span>Select A from A, B, C<br><small>4 / (4 + 2 + 1)</small></span><b>×</b><span>Select B from B, C<br><small>2 / (2 + 1)</small></span><b>×</b><span>Only C remains<br><small>Probability 1</small></span>
</div>

Therefore,

$$
P(A\succ B\succ C)=\frac47\frac23=\frac8{21}\approx0.380952,
\qquad L_{\mathrm{PL}}=-\log\frac8{21}\approx0.965081.
$$

For a complete ranking $\tau$ without ties:

$$
P(\tau\mid x,y_{1:K})
=\prod_{k=1}^{K}
\frac{e^{r(x,y_{\tau(k)})}}
{\sum_{j=k}^{K}e^{r(x,y_{\tau(j)})}}.
$$

This is a **ranking probability conditional on the candidate set**, not the probability of generating those responses. At $K=2$, it reduces to Bradley–Terry. PL also assumes that the relative choice odds of two candidates depend on their scores, not on other candidates. Strong context effects, multidimensional tradeoffs, and heterogeneous preferences may violate that assumption. [DPO Appendix A.3](https://arxiv.org/html/2305.18290v2#A1.SS3) uses this definition to derive an objective for ranking supervision.

## All pairs are not the same objective as a full ranking {#pairs-versus-ranking}

The same ranking supplies A over B, A over C, and B over C. Their pairwise probabilities are

$$
\frac46,\quad \frac45,\quad \frac23.
$$

Their product is $16/45\approx0.355556$, with negative log 1.034074—not the PL loss 0.965081. Averaging over the three pairs gives 0.344691, changing the normalization again.

| Training choice | Labels used | Loss here | Interpretation |
| --- | --- | ---: | --- |
| Mean pairwise loss | Three pairs | 0.344691 | Average negative log-likelihood per pair |
| Sum of pairwise losses | Three pairs | 1.034074 | Combined objective for correlated comparisons |
| PL full-ranking loss | One complete rank | 0.965081 | Negative log-likelihood of that ranking |

The first value being smaller does not make the method better. These losses use different units.

PL probabilities sum to one over all six possible rankings. Multiplying the three pairwise probabilities for each ranking and summing those products generally does not. Pairwise training remains a reasonable choice; its product of correlated comparison probabilities is simply not automatically a normalized probability model over complete rankings.

## How one ranking moves three scores {#gradient}

Write the PL loss in a numerically stable form:

$$
L_{\mathrm{PL}}=\sum_{k=1}^{K-1}
\left[
\operatorname{logsumexp}_{j\ge k}r_{\tau(j)}-r_{\tau(k)}
\right].
$$

At each stage, the gradient is the softmax over remaining candidates minus the one-hot vector of the selected item:

| Score | First-stage contribution | Second-stage contribution | Total gradient |
| --- | ---: | ---: | ---: |
| $r_A$ | $4/7-1$ | 0 | $-3/7$ |
| $r_B$ | $2/7$ | $2/3-1$ | $-1/21$ |
| $r_C$ | $1/7$ | $1/3$ | $10/21$ |

Gradient descent on three independently adjustable scores raises A, raises B slightly, and lowers C. B loses to A but beats C; not every response below first place must be pushed down.

The gradients sum to zero, reflecting invariance to a shared offset. A real reward network has shared parameters, so score movements also depend on its Jacobian. Independent-score updates do not guarantee those same movements in a neural network.

Here is the core loss calculation. Input validation, gradients, and tests are in the [companion program](deep-rl/code/reward_learning_checks.py). Subtracting the maximum avoids exponentiating large positive scores.

```python
import math

def ranking_loss(ordered_scores):
    loss = 0.0
    for position in range(len(ordered_scores) - 1):
        remaining = ordered_scores[position:]
        maximum = max(remaining)
        shifted = [score - maximum for score in remaining]
        loss += math.log(sum(math.exp(score) for score in shifted)) - shifted[0]
    return loss

print(ranking_loss([math.log(4), math.log(2), 0]))
```

## Do not invent labels the annotation never supplied {#partial-rankings}

| Actual annotation | Available supervision | Avoid |
| --- | --- | --- |
| Only A is selected as best | A over the other displayed candidates, or PL's first-choice term | Inventing an order between B and C |
| A and B both outrank C; A/B unobserved | A over C and B over C, or a sum over compatible full rankings | Randomly ordering A/B and treating it as certain |
| A and B genuinely tie | Preserve the tie and choose a tie-aware model or explicit loss convention | Treating “unknown” and “equally good” as the same label |
| Both A and B are unacceptable | Preserve absolute rejection information, and relative preference if useful | Treating the selected response as an acceptable demonstration |

For example, if A and B are both above C but their own order is unknown, PL assigns the event probability

$$
P(A\succ B\succ C)+P(B\succ A\succ C)
=\frac8{21}+\frac27\frac45=\frac{64}{105}.
$$

This marginalizes an unobserved order; it does not model the process that generates ties. Standard tie-free PL cannot directly represent genuine ties without an extension or a stated alternative supervision rule.

## Why raw rewards need not compare across prompts {#score-offsets}

Two responses to prompt A score 2 and 1. Two responses to prompt B score 102 and 101. Both prompts predict a $\sigma(1)\approx0.731$ chance that their preferred response wins. Yet global top-score selection among the four responses would retain only B's responses.

Adding any prompt-specific offset leaves BT and PL probabilities unchanged:

$$
r'(x,y)=r(x,y)+c(x).
$$

The labels therefore do not determine a cross-prompt score origin. Shared parameters may produce a scoring convention, but that convention has not been validated by cross-prompt quality labels. Global filtering, common thresholds, or comparisons across tasks require separate calibration and slice checks; correct within-prompt ranking is insufficient.

Likewise, if one prompt only has A/B and C/D comparisons, its labels do not establish whether A beats C. Connecting previously disconnected candidate groups can be more informative than repeating an existing comparison.

## Store annotations before deriving training pairs {#data-and-eval}

Preserve the original ranking task, then derive pairs from it. For example:

```json
{
  "prompt_id": "debug-017",
  "candidate_ids": ["answer-a", "answer-b", "answer-c"],
  "ordered_groups": [["answer-a"], ["answer-b"], ["answer-c"]],
  "rubric_version": "debug-v2",
  "generator_snapshot": "policy-demo-v1",
  "annotation_batch": "review-03"
}
```

Store response text and generation settings by candidate ID. With multiple annotators, also retain anonymous annotator identifiers, disagreements, and label status. `ordered_groups` is a convention for this teaching example, not a universal framework schema. Do not rely on incidental array order to encode preferences.

| Check | Concrete procedure | What it can reveal |
| --- | --- | --- |
| Leakage | Split by prompt or near-duplicate task group before deriving pairs | The same response reaches validation through a different pair |
| Weighting | Print candidate counts, pair counts, and final per-prompt loss weights | Prompts with many candidates accidentally dominate gradients |
| Candidate reuse | Score each candidate once in the current computation, then form comparisons | Redundant forward passes and inconsistent stochastic scores |
| Annotation consistency | Swap display order; inspect disagreements and missing labels | Position bias, forced rankings, unclear rubrics |
| Distribution change | Also evaluate responses from the current policy | Accuracy on old candidates fails to transfer |
| Actual outcomes | Use task tests, independent human review, and failure slices | Preference prediction need not establish factual or code correctness |

Define the training unit explicitly: one pair, one ranking task, or multiple annotations aggregated for a prompt. Track candidate counts, annotator counts, and prompt weights separately so that changes to the dataset remain interpretable.

## Checks and next steps {#checks}

```bash
python3 05-post-training/deep-rl/code/reward_learning_checks.py
python3 -m unittest discover -s site/tests -p test_reward_learning_checks.py
```

Tests cover normalization over six rankings, the $K=2$ reduction to BT, finite-difference gradients, large scores, shared offsets, and prompt weighting. Tiny probabilities can underflow to zero; training should keep the stable log loss rather than compute probabilities and then take a logarithm.

Continue to [DPO's reward-to-policy-ratio derivation](after-ppo.en.md#dpo-derivation), or return to [preference data and reward failure modes](where-preferences-come-from.en.md). Data organization, objective, and evaluation need to agree; changing a loss name alone does not resolve their mismatches.
