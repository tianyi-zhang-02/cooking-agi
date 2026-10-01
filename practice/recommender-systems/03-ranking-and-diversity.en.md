# 03 · Ranking: good posts, dull list?

[中文](03-ranking-and-diversity.md) · **English**

> Public design first; the lab uses entirely invented candidates, scores, and vectors.

## Predicting behavior is not defining value

The candidates are already here. Now separate two questions: **is this post worth seeing, and is it still worth seeing beside the others?** The first concerns an item; the second concerns a list. The lab lets you change the latter without retraining a model.

Twitter's early Heavy Ranker documentation describes multiple behavior predictions combined by configured weights. The useful distinction is **what the model predicts** versus **what the product rewards**, not memorizing historical coefficients. [Heavy Ranker documentation](https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md)

A teaching expression is:

$$s(u,i)=\sum_{a}w_a\,\hat p(a\mid u,i).$$

$\hat p$ predicts an action; $w_a$ encodes a value choice. Are probabilities calibrated? Is the behavior a useful satisfaction proxy? Could correlated actions double-count the same benefit? This isn't an objective measurement of “liking.”

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

## Self-check

<details><summary>Why doesn't changing the penalty prove the model learned more interests?</summary><p>This lab trains no model and adds no candidates. It changes selection. Representation, retrieval coverage, and list choice are distinct.</p></details>

<details><summary>Can an author limit replace a similarity penalty?</summary><p>Not fully. One author can cover several topics and different authors can repeat the same content. Explain and evaluate each constraint separately.</p></details>

Next: [what changed in newer systems](04-modern-recsys.en.md).
