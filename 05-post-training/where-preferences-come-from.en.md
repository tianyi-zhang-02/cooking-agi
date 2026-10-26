# Where preferences come from, and what a reward model actually learns

[中文](where-preferences-come-from.md) · **English**

> Reading time: ~11 min · Type: chapter · Last reviewed: 2026-10

## Preference data provides relative order {#preference-data-provides-relative-order}

A user asks how to fix an error. Response A gives a working fix; B gives a lengthy explanation without resolving it. Choosing A records a preference between these two answers to this prompt.

That choice does not give A an objective quality score or establish what a different user would prefer. A reward model (RM) learns patterns across many comparisons and uses them to score new answers. The way those comparisons were collected shapes what the score means.

## Why comparisons instead of scores {#why-comparisons-instead-of-scores}

Two annotators may mean different things by a score of 8 out of 10. Asking them to compare answers to the same prompt against a stated criterion can make their judgments easier to align. A common record contains a prompt $x$, the preferred response $y_w$, and the response not selected, $y_l$.

Pairwise comparisons are not automatically reliable. Both answers may be poor, each may have different strengths, or annotators may have different priorities. Rather than forcing a winner, it can help to record ties, mark both answers as unacceptable, and retain reasons for disagreement.

## Bradley-Terry: turning comparisons into a score {#bradley-terry-turning-comparisons-into-a-score}

Learning a scoring function from pairwise comparisons requires a relationship between **scalar scores** and **preference order**. Bradley–Terry uses the difference between two scores to model the probability that one response is preferred:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

| Symbol | Meaning |
| --- | --- |
| $r(x,y)$ | the reward model's scalar score for response $y$ |
| $\sigma$ | the sigmoid function, $\sigma(z)=1/(1+e^{-z})$, which maps the score difference to (0,1) |
| the **difference** | the probability depends on how far apart the scores are, not on their absolute values |

Training maximizes the likelihood of the observed preferences, or equivalently minimizes this negative log-likelihood:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\big(r(x,y_w) - r(x,y_l)\big)\right]$$

A common implementation adds a scalar output head to a pretrained model, producing one score from the prompt and complete response. Models can also supply the comparisons or generate their judgments as text; see [below](#ai-feedback-and-rubrics).

## Score offsets and scale {#a-consequence-the-reward-model-learns-order-not-scale}

Suppose rewards for A and B are 2 and 1. Bradley–Terry gives A a preference probability of $\sigma(1)\approx0.731$. Adding 10 to both rewards gives 12 and 11 without changing the probability.

Multiplying both by 2 is different: the gap becomes 2 and the probability becomes $\sigma(2)\approx0.881$. **Adding the same constant leaves the probability unchanged; scaling generally does not.** Depending only on differences does not make the scale arbitrary.

<!-- widget:step-bt-scale -->

Scores still are not universal units of utility shared across reward models. In reinforcement learning (RL), reward scale can affect advantages—how much better a response is than a baseline—along with gradients and the balance between the reward term and a Kullback–Leibler (KL) divergence penalty. That penalty discourages deviation from a reference policy. Advantage standardization can cancel some effects of rescaling, so whether to normalize depends on the objective and implementation.

## Four traps in preference data {#four-traps-in-preference-data}

Start with a small sample and four concrete checks:

| Check | How to test it | What to distinguish |
| --- | --- | --- |
| Annotator agreement | Have people independently compare the same pairs | Unclear criteria versus genuinely different preferences |
| Length preference | Add repetition without changing useful information | Useful detail versus length alone |
| Factual checking | Supply verifiable sources or execute the code | Credible-looking versus correct |
| Presentation order | Swap A/B and hide model names | Compare answer IDs, not left/right choices |

Inter-annotator agreement is not a universal upper bound on reward-model accuracy. It depends on the annotators, aggregation rule, and data distribution. It is more useful for identifying cases that need clearer criteria or multiple preference profiles.

## Reward models can become stale {#the-one-people-miss-reward-models-expire}

A reward model may have seen only outputs from an older policy. Updating the policy—the model that generates answers—can change the kinds of responses it produces. An RM that still performs well on its old validation set can misjudge these new outputs.

For example, detailed explanations may have been useful in the old data, while the new policy learns to repeat itself for a higher score. Reward increases without helping the user. Inspect current-policy samples rather than relying only on the old validation set.

A KL penalty discourages drift from a reference policy; it does not guarantee that the RM can still judge new answers reliably. Independent human review, task checks, and evaluation on new outputs are still needed.

## Labels and scores can also come from models {#ai-feedback-and-rubrics}

So far, people have supplied the comparisons and a scalar output head has produced the scores. Either part can be replaced. The question is what each replacement changes, and what still needs checking.

**AI-generated preferences.** In [Constitutional AI (Bai et al., 2022, §4.1)](https://arxiv.org/abs/2212.08073), a model compares responses for harmlessness using human-written principles. Those preferences train a reward model, which then guides policy updates. The authors call this RL from AI Feedback (RLAIF). Their method replaces human pairwise labels for harmlessness, not all human feedback: helpfulness labels remain human-provided, as do the principles and final evaluation.

[Lee et al. (ICML 2024, §4.1)](https://arxiv.org/abs/2309.00267) compare RLAIF with RL from Human Feedback (RLHF). In their experiments, human preference results are similar for summarization and helpful dialogue, while RLAIF achieves a higher harmlessness rate on the harmless-dialogue task. They also test direct RLAIF (d-RLAIF), which obtains rewards from an off-the-shelf large language model (LLM) during training rather than fitting a separate reward model. AI labels can reduce human annotation work, but do not automatically remove position or length bias. A judge's systematic errors may recur throughout the dataset.

**Generative reward models.** A reward model can make its judgment by generating text. [GenRM (Zhang et al., ICLR 2025, v3 §3)](https://arxiv.org/abs/2408.15240) trains verifiers through next-token prediction and uses the probability of `Yes` as a correctness score. Its GenRM-CoT variant generates a reasoning trace before judging. What the paper calls “majority voting” averages the `Yes` probabilities across sampled traces; it does not simply count binary votes.

[DeepSeek-GRM (Liu et al., 2025, v3 §4)](https://arxiv.org/abs/2504.02495) generates principles for the question, writes a critique, and assigns each response an integer score from 1 to 10. At inference, it samples $k$ judgments in parallel and sums each response's scores. An optional meta reward model evaluates the principles and critiques, retaining only the higher-scoring judgments. The input can contain one, two, or many responses, each receiving its own score; separate scores do not imply separate judging calls.

[RM-R1 (Chen et al., 2025, v4 §2.3.1)](https://arxiv.org/abs/2505.02387) organizes its reasoning by task type. For chat, it develops grading criteria; for reasoning tasks such as math or code, it first attempts its own solution, then compares the candidates against it. Generating these judgments costs additional computation, and repeated sampling for the same candidates can produce different judgments.

**Rubric-based rewards.** When code alone cannot adequately judge quality, as in open-ended medical questions, a rubric can break a good answer into individual requirements. In [Rubrics as Rewards (Gunjal et al., 2025)](https://arxiv.org/abs/2507.17746), a judge marks each requirement as met or unmet, giving $c_j$ a value of 1 or 0. The explicit aggregation method weights these judgments by $w_j$ and divides their sum by the total weight (v2 §2.2, Eq. 1):

$$r(x,\hat y)=\frac{\sum_j w_j\,c_j(x,\hat y)}{\sum_j w_j}$$

The example below uses nonnegative weights with a positive total, keeping the reward between 0 and 1. Alternatively, the judge can receive the whole rubric and return an overall score, an approach the paper calls implicit aggregation.

The authors train Qwen2.5-7B with Group Relative Policy Optimization (GRPO), using gpt-4o-mini as the judge, 16 responses per prompt at temperature 1.0, and eight H100 GPUs. They report that implicit aggregation improves on the Direct-Likert baseline, which assigns a single 1–10 rating, by up to 31% on HealthBench and 7% on GPQA-Diamond (v2 §§4–5). These are relative gains under that setup, not percentage-point gains or results attributable to the explicit weighted sum above.

### Work it through: one misjudged requirement {#rubric-example}

Return to the error-fixing example. Suppose the user asks, “Python raises `KeyError: 'id'`. How do I fix it?” The rubric has four requirements with illustrative weights. Assume that a missing field is allowed to trigger a default path. If the field is required, the fix should repair the data source or raise an explicit error, not merely suppress the exception with `.get`.

| Requirement | Weight | A: gives a working fix | B: long explanation, no fix |
| --- | ---: | :---: | :---: |
| Explains the cause: the dict has no `'id'` key | 3 | 1 | 1 |
| Gives a fix that runs, such as checking the key or using `.get` | 3 | 1 | 0 |
| Warns that the fix can hide a missing field upstream | 1 | 0 | 0 |
| Invents no nonexistent functions or arguments | 2 | 1 | 1 |

A scores $(3+3+2)/9\approx0.889$ and B scores $(3+2)/9\approx0.556$. Now suppose the judge wrongly marks B's second requirement as met. B rises to $8/9$, the same as A.

If these two responses form one GRPO group, their original rewards have mean $13/18$ and population standard deviation $1/6$, giving standardized advantages of approximately $[1,-1]$. After the error, both rewards are equal. With a small positive constant in the denominator to prevent division by zero, the advantage vector becomes $[0,0]$. Using the sample standard deviation changes the original magnitudes, but not the loss of distinction when rewards tie. Only this outcome-reward advantage has vanished; other terms, such as a KL penalty, can still produce different updates for the two responses.

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

A rubric therefore needs two checks: who sets the weights, and who judges each requirement. Item-level judgments are easier to audit than one overall 1–10 score. For example, you can measure errors on “gives a fix that runs” separately. If a model also wrote the rubric, check for missing requirements: the reward cannot account for what the rubric omits.

Whether people or models supply the labels, with or without a rubric, the same problem remains: as the policy changes, check that its new outputs are still judged reliably. For common judge biases and ways to test for them, see [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.en.md).

## What to look for in a preference-training experiment {#down-to-a-checklist}

Look for three things: the annotation criteria, an assessment of the reward model's judgments on current-policy outputs, and evidence independent of the training reward. An RM score alone does not establish that users prefer the resulting model.

## Where to read next {#where-to-read-next}

- [The three stages of RLHF](rlhf/three-stages.en.md): what happens to this reward model next
- [Verifiable rewards](verifiable-rewards.en.md): which of these traps disappear when the reward isn't learned
- [The alignment tax](alignment-tax.en.md): how preference biases affect downstream behavior
- [Data and feedback](../01-data-and-feedback/): label quality in general

## Quick learning: the boundary between preference data and a reward model {#quick-learning-the-boundary-between-preference-data-and-a-reward-model}

<details class="interview" markdown="1">
<summary>What does a comparison provide, and what does a score difference mean?</summary>

**Remember**: a chosen/rejected pair records a relative preference, not proof that one answer is acceptable and the other is not. A reward model fits these choices; its scores are not universal units of utility. Adding a shared constant leaves Bradley–Terry probabilities unchanged, while scaling the differences generally changes them.

**Interview answer**

> Preference data records choices for the same prompt. Bradley–Terry fits choice probabilities using score differences, whose magnitudes matter as well as their signs. Scores are not objective utility units: check annotator disagreement, position and length bias, and distribution shifts after policy updates.

<details markdown="1">
<summary><b>Deep dive</b>: why does a reward model become stale?</summary>

The RM may have learned only from an older policy's outputs. As the policy changes, it can generate unfamiliar answers or exploit blind spots that earn high scores. Accuracy on the old validation set can remain high even when the RM misjudges new answers. Check current-policy samples with human review and held-out adversarial examples.

</details>
</details>

## Starting papers {#starting-papers}

- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — an influential early deep-RL example of training a reward model from pairwise preferences
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL penalties and reward hacking
- [InstructGPT](https://arxiv.org/abs/2203.02155) — the three-stage pipeline and annotation guidelines
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — a model compares responses against a list of principles: RLAIF
- [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) — DeepSeek-GRM: write principles and a critique, then score
- [Rubrics as Rewards](https://arxiv.org/abs/2507.17746) — rubric-based rewards when code alone cannot adequately judge quality
