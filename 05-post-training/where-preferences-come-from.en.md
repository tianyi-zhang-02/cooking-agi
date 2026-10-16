# Where preferences come from, and what a reward model actually learns

[中文](where-preferences-come-from.md) · **English**

> Reading time: ~11 min · Type: chapter · Last reviewed: 2026-10

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

The classic implementation is a pretrained model with a scalar head that reads the whole response and emits one number. A model can also make the comparisons, and a generative model can write out the score; see [below](#ai-feedback-and-rubrics).

## Score offsets and scale {#a-consequence-the-reward-model-learns-order-not-scale}

Suppose rewards for A and B are 2 and 1. Bradley–Terry gives A a preference probability of $\sigma(1)\approx0.731$. Adding 10 to both rewards gives 12 and 11 without changing the probability.

Multiplying both by 2 is different: the gap becomes 2 and the probability becomes $\sigma(2)\approx0.881$. **A shared offset is invariant; scaling is not.** Depending only on differences does not make the scale arbitrary.

<!-- widget:step-bt-scale -->

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

## Labels and scores can also come from models {#ai-feedback-and-rubrics}

So far we assumed two things: people make the comparisons, and a scalar head on top of a model produces the score. Both now have common alternatives, and every check above has to be aimed at the new source.

**AI comparisons (RLAIF).** [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) targets harmlessness. People write a list of principles; a model uses them to judge which of two responses is better; those AI preferences train a preference model, which then serves as the reward for RL. The authors call this RL from AI Feedback (RLAIF). People still write the principles and run the evaluation; they no longer label each pair. [Lee et al. (ICML 2024)](https://arxiv.org/abs/2309.00267) report that RLAIF performs comparably to RLHF on summarization, helpful dialogue, and harmless dialogue. They also test asking an off-the-shelf LLM for rewards directly during RL, without training a separate reward model (d-RLAIF). AI labels save the human effort of labeling each pair. They remove none of the traps above, such as position bias and length preference, and one judge's bias repeats across the whole dataset.

**Generative reward models.** A reward model can write text before it gives a score. [GenRM (Zhang et al., ICLR 2025)](https://arxiv.org/abs/2408.15240) trains verifiers with next-token prediction, so they can reason before judging and use majority voting over several samples. [DeepSeek-GRM (Liu et al., 2025)](https://arxiv.org/abs/2504.02495) first writes judging principles for the current question, then a critique, and finally an integer score from 1 to 10 for each response. At inference it samples k times in parallel and sums the scores (§4 of the paper); a meta RM can first filter out poor samples. It produces a separate score for each response while accepting one or multiple responses jointly, so one format covers one, two, or many candidates; separate scores do not imply independent judging calls. [RM-R1 (Chen et al., 2025)](https://arxiv.org/abs/2505.02387) treats reward modeling as a reasoning problem: it first writes the grading points or a reference solution for the question, then checks candidates against them. The cost is direct: every judgment generates text, and sampling the same response several times can give different scores.

**Rubric rewards.** For tasks without a program checker, such as medical questions, a good answer can be broken into requirements that are judged one at a time. [Rubrics as Rewards (Gunjal et al., 2025)](https://arxiv.org/abs/2507.17746) has a judge model mark whether each requirement is met (1 or 0), then sums the weights of the met requirements and divides by the total weight (v2 §2.2, Eq. 1):

$$r(x,\hat y)=\frac{\sum_j w_j\,c_j(x,\hat y)}{\sum_j w_j}$$

Alternatively the whole rubric goes to the judge, which returns one overall score (the paper calls this implicit aggregation). Training Qwen2.5-7B with GRPO and gpt-4o-mini as the judge, the paper reports relative gains of up to 31% on HealthBench and 7% on GPQA-Diamond over baselines where the judge gives a direct Likert score; these are the authors' own results.

### Work it through: one misjudged requirement {#rubric-example}

Return to the error-fixing example at the top. Suppose the user asks, “Python raises `KeyError: 'id'`. How do I fix it?” The rubric has four requirements; the weights are illustrative:

| Requirement | Weight | A: gives a working fix | B: long explanation, no fix |
| --- | ---: | :---: | :---: |
| Explains the cause: the dict has no `'id'` key | 3 | 1 | 1 |
| Gives a fix that runs, such as checking the key or using `.get` | 3 | 1 | 0 |
| Warns that the fix can hide a missing field upstream | 1 | 0 | 0 |
| Invents no nonexistent functions or arguments | 2 | 1 | 1 |

A scores $(3+3+2)/9\approx0.889$ and B scores $(3+2)/9\approx0.556$. Now suppose the judge wrongly marks B's second requirement as met. B rises to $8/9$, the same as A. If these two responses form one GRPO group for the same question, the group advantages go from about $[1,-1]$ to $[0,0]$: the response that solved the problem and the one that did not get exactly the same training signal.

```python
weights = [3, 3, 1, 2]
answer_a = [1, 1, 0, 1]
answer_b = [1, 0, 0, 1]

def rubric_reward(checks):
    return sum(weight * check for weight, check in zip(weights, checks)) / sum(weights)

assert abs(rubric_reward(answer_a) - 8 / 9) < 1e-12
assert abs(rubric_reward(answer_b) - 5 / 9) < 1e-12
misjudged_b = [1, 1, 0, 1]
assert rubric_reward(misjudged_b) == rubric_reward(answer_a)
```

So a rubric needs two checks: who sets the weights, and who judges each requirement. Requirement-by-requirement judgments are easier to audit than one overall 1–10 score; you can measure the judge's error rate on “gives a fix that runs” alone. If a model also wrote the rubric, check what it left out. A requirement the rubric does not state stays invisible to the reward. Whether labels come from people, an AI, or a rubric, the expiry problem from the previous section still applies: when the policy changes, recheck the judge on current-policy outputs. For common judge biases and how to test for them, see [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.en.md).

## Down to a checklist {#down-to-a-checklist}

When reading a preference-training experiment, I look for the annotation criterion, checks on current-policy outputs, and final evidence independent of the training reward. Reporting only the RM score leaves user benefit unresolved.

## Where to read next {#where-to-read-next}

- [The three stages of RLHF](rlhf/three-stages.en.md): what happens to this reward model next
- [Verifiable rewards](verifiable-rewards.en.md): which of these traps disappear when the reward isn't learned
- [The alignment tax](alignment-tax.en.md): what the bias in preferences grows into downstream
- [Data and feedback](../01-data-and-feedback/): label quality in general

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

## Starting papers {#starting-papers}

- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — an influential early deep-RL example of training a reward model from pairwise preferences
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL penalties and reward hacking
- [InstructGPT](https://arxiv.org/abs/2203.02155) — the three-stage pipeline and annotation guidelines
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — a model compares responses against a list of principles: RLAIF
- [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) — DeepSeek-GRM: write principles and a critique, then score
- [Rubrics as Rewards](https://arxiv.org/abs/2507.17746) — requirement-by-requirement rubrics as rewards for tasks without a checker
