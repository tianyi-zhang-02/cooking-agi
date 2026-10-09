# The score improved. What does that actually establish?

[中文](metric-robustness.md) · **English**

A new model gains three Recall points. Before calling it an improvement: was the candidate pool identical? Were both models evaluated on the same queries? Did a subgroup regress? Would another seed produce the same result?

Start with small calculations rather than memorizing statistical tests. All numbers here are teaching data, not experimental results. For intuition, read [population mixtures](#population-mix); for runnable code, jump to [paired comparison](#paired-comparison).

## 1. The same Recall name can hide a different task

Let $P$ be the labeled positive set for a query and $R_K$ its top-$K$ results. Standard Recall@K is:

$$
\mathrm{Recall@K}=\frac{|P\cap R_K|}{|P|},\qquad |P|>0.
$$

With eight known positives and three positives in the top three, Recall@3 is $3/8$, not 1. Some implementations replace the denominator with $\min(|P|,K)$ and return 1. That can be a separately named normalized measure, but should not be compared silently with standard Recall.

| Check first | Why it changes the conclusion |
| --- | --- |
| How are queries with no positives handled? | Exclusion, zero-filling, and separate reporting produce different averages; report coverage |
| Average per query or pool hit counts? | Macro gives equal weight to queries; micro favors queries with more positives |
| How was the pool built? | Full-corpus, randomly sampled, and hard-negative pools are different tasks |
| Where did positives come from? | Exposed clicks, human judgments, and teacher labels measure different things |

With one positive and sampled comparisons, Recall@K becomes a hit indicator. Under random ranking with $n$ comparisons, its expectation is $\min(K,n+1)/(n+1)$. At $K=10$, 19 comparisons give 50%; 999 give 1%. This says nothing about model quality. Pool construction alone changes difficulty.

Some comparisons may also be relevant but unlabeled. A model that promotes them may lose measured Recall. Report the result as Recall under the specified pool and labels, not as a direct measure of real satisfaction.

## 2. Whose proportions define the average? {#population-mix}

A fictional evaluation contains 90% established users and 10% new users:

| Population | Old model | New model | Change |
| --- | --- | --- | --- |
| Established users | 80% | 85% | +5 percentage points |
| New users | 40% | 30% | −10 percentage points |
| Weighted 90% / 10% | 76% | 79.5% | +3.5 percentage points |

Overall performance improves while new users regress. Both statements are true. For a target population split equally between the groups, the old model scores 60% and the new one 57.5%: the ordering reverses.

```text
10% new users: old 76.0% → new 79.5%  (+3.5 percentage points)
50% new users: old 60.0% → new 57.5%  (−2.5 percentage points)
Within-group performance stays identical. The target population changes.
```

Do not select weights because they make the conclusion attractive. Define the observed population, target deployment population, and protected slices in advance. The weights $w_g$ in $\bar m=\sum_g w_gm_g$ are part of the question.

Matched sampling can balance measured conditions without turning populations into randomized controls. Matching positive counts may leave task difficulty unmatched. A high score in a small slice is not, by itself, a causal effect of a design choice.

## 3. Could the difference be noise? Start with paired examples {#paired-comparison}

Run both models on the same queries and record $d_i=m_i^{\mathrm{new}}-m_i^{\mathrm{old}}$. Inspect the distribution of differences before its average. Comparing means from unrelated samples also mixes in changes in query difficulty.

This minimal paired bootstrap resamples queries while preserving each old/new pair. It estimates **uncertainty from test-sample composition, conditional on these two sets of model outputs**.

```python
import random
from statistics import mean

old_scores = [0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3]
new_scores = [0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6]
differences = [new - old for old, new in zip(old_scores, new_scores)]
generator = random.Random(7)
estimates = sorted(mean(generator.choices(differences, k=len(differences)))
                   for _ in range(4000))
lower, upper = estimates[100], estimates[3899]
assert abs(mean(differences) - 0.05) < 1e-12
assert lower <= 0 <= upper
print(round(mean(differences), 3), round(lower, 3), round(upper, 3))
```

The mean gains 0.05, but this eight-example interval still includes zero; it does not establish a stable improvement. Nor does it establish equivalence: the data may be insufficient to distinguish the models. This is an approximate percentile-interval demonstration. See [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) for pairing and interval methods, and inspect small-sample, skewed, or degenerate cases.

When one user contributes many correlated queries, resample an appropriate cluster such as the user, then compute the preselected query-weighted or user-weighted statistic. Resampling log rows independently can mistake correlated observations for additional independent evidence.

Bootstrap does not measure training variance for you. Comparing training methods also needs independently trained seeds. Rerunning one checkpoint is not retraining it. If the judge is stochastic, inspect scoring repeatability separately.

## 4. Inspect twenty metrics and some will probably rise

Slices help find problems, but selecting the largest gain from many comparisons can overstate effects. Prespecify a primary metric, regression guardrails, and a small set of motivated slices. Treat other discoveries as exploratory and confirm them on independent held-out data.

Not every result belongs in a significant/not-significant binary. Report effect size, interval, sample size, and practical cost. A tiny statistically significant gain may not justify double the latency; failure to reach significance does not prove a method has no value.

## 5. What if the evaluator changed?

Judge versions, prompts, reference answers, and task distributions affect LLM scores. Longer new answers may be more complete, or the judge may simply prefer length. Recheck fixed human-reviewed anchors and inspect errors by task rather than relying only on overall agreement.

Do not use an evaluation set indefinitely as development data. Record which examples informed rubric, threshold, or model changes, and keep a final subset out of those decisions. Sharing a training teacher and evaluator is not automatically invalid, but their agreement is not independent evidence; external references are needed to test shared blind spots.

See [LLM-as-a-Judge](llm-as-a-judge/README.en.md) for concrete biases. If the failure could be in retrieval, context selection, or generation, use the [evaluation stack](evaluation-stack.en.md) to locate it first.
