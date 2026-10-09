# 03 · Ranking: good posts, dull list?

[中文](03-ranking-and-diversity.md) · **English**

> Public design first; the lab uses entirely invented candidates, scores, and vectors.

## Predicting behavior is not defining value

The candidates are already here. Now separate two questions: **is this post worth seeing, and is it still worth seeing beside the others?** The first concerns an item; the second concerns a list. The lab lets you change the latter without retraining a model.

Twitter's 2023 Heavy Ranker documentation describes multiple behavior predictions combined by configured weights. The useful distinction is **what the model predicts** versus **what the product rewards**, not memorizing historical coefficients. [Pinned Heavy Ranker documentation](https://github.com/twitter/the-algorithm-ml/blob/b85210863f7a94efded0ef5c5ccf4ff42767876c/projects/home/recap/README.md)

A teaching expression is:

$$s(u,i)=\sum_{a}w_a\,\hat p(a\mid u,i).$$

$u$ is the user, $i$ the candidate, and $a$ an action task. $\hat p$ predicts that action's probability; $w_a$ determines its weight in ranking. Are probabilities calibrated? Is the behavior a useful satisfaction proxy? Could correlated actions double-count the same benefit? This isn't an objective measurement of “liking.”

### Different ordering, unchanged model {#serving-weight-example}

Keep only predicted probabilities of liking and rejection. The last two columns use “like − penalty × rejection”; all values are invented.

<div class="worked-table" markdown="1">

| Item | Like | Reject | Penalty 2 | Penalty 5 |
| --- | --- | --- | --- | --- |
| A | 0.20 | 0.04 | 0.12 | 0.00 |
| B | 0.12 | 0.01 | 0.10 | 0.07 |

</div>

Changing the penalty from 2 to 5 puts B ahead of A without changing either prediction. The selection preference changed; predictive accuracy did not. The combined score is not a probability: it can be negative and need not sum to 1.

The weighted quantities are **this user's predicted action probabilities**, not the counts of likes or reports the post has already received. Adding a continuous prediction such as reading time also requires explicit units and scaling; it isn't another action probability.

## Training losses and serving objectives differ

A multitask model can learn separate labels for clicks, saves, and rejection. Their observation and missingness processes differ. Training loss weights control optimization; serving weights $w_a$ combine predictions. They need not match.

A missing behavior isn't automatically a zero label. Check exposure, sampling, labeling windows, and task masks before adding more heads. Competing objectives aren't a newly discovered issue; [YouTube's 2019 multitask-ranking paper](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/) discusses objective conflicts and selection bias.

## A list is more than independent scores

Three nearly identical tutorials can each look useful yet make a repetitive feed. A selection rule can penalize similarity to previously selected items. This is an **illustrative greedy reranker**, not Twitter's or X's serving formula:

$$g(i\mid S)=s(i)-\lambda\max_{j\in S}\max(0,\cos(v_i,v_j)).$$

The penalty is zero for empty $S$ and recomputed after each selection. This considers only the most similar selected item and does not guarantee a globally optimal list.

<div data-recsys-lab></div>

Without JavaScript, the starting result is still clear: at $\lambda=0$ without an author limit, the example chooses A, B, C, all labeled ML. A stronger penalty may introduce other topics while reducing the average original score.

## Don't equate spread with improvement

| Observation | What it establishes | What it doesn't establish |
| --- | --- | --- |
| More topics | More hand-assigned categories represented | Higher user satisfaction |
| Larger embedding distances | Less similarity in this representation | Relevant content or genuine semantic difference |
| Lower average raw score | Departure from the original score objective | A necessarily worse list |
| No repeated author | The author constraint is satisfied | Content quality or viewpoint diversity |

Irrelevant content can also increase distances. Evaluate relevance, coverage, and list experience together, then test in controlled deployment. Sweeping a parameter doesn't establish a universally optimal diversity level.

## More task heads don't necessarily mean more labels

Consider a synthetic table with confirmed exposure, complete logs for both actions, and a finished observation window. An explicit “not interested” action defines rejection in this example.

| Record | Like label / mask | Rejection label / mask | Meaning |
| --- | --- | --- | --- |
| Liked, no rejection action | 1 / 1 | 0 / 1 | Zero means that action did not occur within the window |
| Explicit rejection, no like | 0 / 1 | 1 / 1 | Rejection evidence, unlike a random unexposed item |
| Both actions occurred | 1 / 1 | 1 / 1 | Both labels can be one if the product permits it |
| Candidate only, never exposed | — / 0 | — / 0 | Neither preference nor rejection was observed |

**No like isn't dislike; no rejection isn't liking.** The tasks predict separate actions, not mutually exclusive user classes. Incomplete logs or an unfinished window require masking the affected task; the two masks need not match.

Filling the last row with zeros doesn't create information by adding heads; it trains harder on an assumption. Masking avoids treating unknown labels as known ones, but **does not remove exposure-policy selection bias**.

### What is the loss denominator? {#task-loss-mean}

One explicit teaching convention is to average over each task's valid examples, then combine the task means:

$$N_a=\sum_n m_{na},\qquad
L=\sum_{a:N_a>0}\alpha_a\,
\frac{\sum_n m_{na}\,\ell_{na}}{N_a}.$$

$n$ indexes examples, $m_{na}\in\{0,1\}$ marks an available label, $\ell_{na}$ is its per-example task loss, and $\alpha_a$ is a training weight. If a task has no labels in the batch, omit its term and report a label count of zero—not “zero loss, therefore perfect predictions.”

For instance, valid like losses `[0.2, 0.6, 0.4]` average to 0.4; rejection losses `[0.1, 0.3, 0.5]` average to 0.3. With weights 1 and 0.5, total loss is **0.55**. Adding an example with neither label should change neither numerator nor denominator. Distributed training needs global valid counts too; averaging rank-local means is incorrect when those counts differ.

Task means don't guarantee freedom from task interference. Inspect valid-label counts and gradients together. Training weights $\alpha_a$ differ from serving weights $w_a$: a stronger serving-time rejection penalty might reduce unpleasant content while suppressing exploration, so check both outcomes.

## Self-check

<details><summary>Why doesn't changing the penalty prove the model learned more interests?</summary><p>This lab trains no model and adds no candidates. It changes selection. Representation, retrieval coverage, and list choice are distinct.</p></details>

<details><summary>Can an author limit replace a similarity penalty?</summary><p>Not fully. One author can cover several topics and different authors can repeat the same content. Explain and evaluate each constraint separately.</p></details>

Next: [what changed in newer systems](04-modern-recsys.en.md).
