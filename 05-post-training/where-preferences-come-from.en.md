# Where preferences come from, and what a reward model actually learns

[中文](where-preferences-come-from.md) · **English**

> Reading time: ~7 min · Type: chapter · Last reviewed: 2026-08

## Preference data provides relative order {#preference-data-provides-relative-order}

A user asks how to fix an error. Response A gives a working fix; B gives a lengthy explanation without resolving it. Choosing A records a preference between these two answers to this prompt.

It does not give A an objective quality score or establish what a different user would prefer. A reward model learns patterns across many comparisons and uses them to score new answers. How the comparisons were collected determines what that score can mean.

## Why comparisons instead of scores {#why-comparisons-instead-of-scores}

Two annotators may mean different things by a score of 8 out of 10. Comparing answers to the same prompt under a stated criterion can make the task easier to calibrate. A common record is a prompt $x$, preferred response $y_w$, and alternative $y_l$.

Pairwise comparisons are not automatically reliable. Both answers may be poor, good in different ways, or judged under different priorities. Ties, both-bad labels, and reasons for disagreement can be useful instead of forcing a winner every time.

## Bradley-Terry: turning comparisons into a score {#bradley-terry-turning-comparisons-into-a-score}

To get from pairwise comparisons to a function that scores any response, you need a model that translates **order** into a **scalar**. Bradley-Terry is that translator:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

| Symbol | Meaning |
| --- | --- |
| $r(x,y)$ | the reward model's scalar score for response $y$ |
| $\sigma$ | sigmoid, squashing the score gap into a probability |
| the **difference** | the only thing that enters the formula — note this |

Training maximizes the likelihood of the observed preferences:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\big(r(x,y_w) - r(x,y_l)\big)\right]$$

In practice this is a pretrained model with a scalar head that reads the whole response and emits one number.

## Score offsets and scale {#a-consequence-the-reward-model-learns-order-not-scale}

Suppose rewards for A and B are 2 and 1. Bradley–Terry gives A a preference probability of $\sigma(1)\approx0.731$. Adding 10 to both rewards gives 12 and 11 without changing the probability.

Multiplying both by 2 is different: the gap becomes 2 and the probability becomes $\sigma(2)\approx0.881$. **A shared offset is invariant; scaling is not.** Depending only on differences does not make the scale arbitrary.

Scores still are not universal utility units shared across reward models. In RL, their scale affects advantages, gradients, and the relative weight of a KL penalty. Normalization is a design choice to examine with the objective and implementation, not an unconditional repair.

## Four traps in preference data {#four-traps-in-preference-data}

Start with a small sample and four concrete checks:

| Check | How to test it | What to distinguish |
| --- | --- | --- |
| Annotator agreement | Have people independently compare the same pairs | Unclear criteria versus genuinely different preferences |
| Length preference | Add repetition without changing useful information | Useful detail versus length alone |
| Factual checking | Supply verifiable sources or execute the code | Credible-looking versus correct |
| Presentation order | Swap A/B and hide model names | Compare answer IDs, not left/right choices |

Inter-annotator agreement is not a universal upper bound on reward-model accuracy. It depends on the annotators, aggregation rule, and data distribution. It is more useful for identifying cases that need clearer criteria or multiple preference profiles.

## The one people miss: reward models expire {#the-one-people-miss-reward-models-expire}

A reward model may have seen only outputs from an older policy. After the actor changes, it may produce a different kind of answer. An RM that still performs well on its old validation set can misjudge these new outputs.

For example, detailed explanations may have been useful in the old data, while the new policy learns to repeat itself for a higher score. Reward increases without helping the user. Inspect current-policy samples rather than relying only on the old validation set.

A KL penalty limits drift relative to a reference policy; it does not certify that new answers remain in the RM's reliable region. Independent human review, task checks, and evaluation on new outputs are still needed.

## Down to a checklist {#down-to-a-checklist}

When reading a preference-training experiment, I look for the annotation criterion, checks on current-policy outputs, and final evidence independent of the training reward. Reporting only the RM score leaves user benefit unresolved.

## Where to read next {#where-to-read-next}

- [The three stages of RLHF](rlhf/three-stages.en.md): what happens to this reward model next
- [Verifiable rewards](verifiable-rewards.en.md): which of these traps disappear when the reward isn't learned
- [The alignment tax](alignment-tax.en.md): what the bias in preferences grows into downstream
- [Data and feedback](../01-data-and-feedback/): label quality in general

## Starting papers {#starting-papers}

- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — the origin of training a reward model from pairwise preferences
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL penalties and reward hacking
- [InstructGPT](https://arxiv.org/abs/2203.02155) — the three-stage pipeline and annotation guidelines

## Quick learning: the boundary between preference data and a reward model {#quick-learning-the-boundary-between-preference-data-and-a-reward-model}

<details class="interview" markdown="1">
<summary>What does a comparison provide, and what does a score difference mean?</summary>

**Quick memory**: a chosen/rejected pair records a preference. A reward model fits such choices; its scores are not universal utility units. A shared offset leaves Bradley–Terry probabilities unchanged, while scaling the differences changes them.

**Interview answer**

> Preference data records choices for the same prompt. Bradley–Terry fits choice probabilities using score differences, whose magnitudes matter as well as their signs. Scores are not objective utility units: check annotator disagreement, position and length bias, and distribution shifts after policy updates.

<details markdown="1">
<summary><b>Deep dive</b>: why does a reward model become stale?</summary>

The RM learns to discriminate outputs from an older policy distribution. As the actor changes, it explores new regions and may find blind spots that inflate proxy reward. Offline accuracy can remain high while online ranking fails, requiring current-policy sampling, human audits, and adversarial held-out evaluation.

</details>
</details>
