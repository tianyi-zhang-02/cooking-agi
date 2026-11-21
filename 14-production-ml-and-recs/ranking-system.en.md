# One recommendation request: candidates, ranking, and feedback

[中文](ranking-system.md) · **English**

> Reading time: ~15 minutes · Last reviewed: 2026-11

Suppose a content feed must select 20 items from 10,000 available items for one refresh. Running the most expensive model on all of them may be too slow. Returning only popular items may miss a user's niche interests. Where should the system spend its computation?

Every count and latency below is a **teaching assumption**, not a production benchmark or description of a company's system. For the model details behind vector retrieval, see [two-tower retrieval](../practice/recommender-systems/06-why-two-towers.en.md).

## Separate requests from updates {#system}

An online request needs a prompt response. Content ingestion and training proceed separately, outside the request's waiting path: a refresh does not wait for the model to retrain.

<div class="system-flow" role="group" aria-label="Request, content, and training paths through a recommender">
<section class="system-flow-lane">
<h3>One refresh</h3>
<ol><li>Receive the request and context</li><li>Retrieve candidate IDs and sources</li><li>Fetch features, filter, and rank</li><li>Return 20 items and a request ID</li></ol>
<p>Actual impressions and later actions feed the training path.</p>
</section>
<section class="system-flow-lane">
<h3>Content updates</h3>
<ol><li>Receive new or revised content</li><li>Encode and update the index</li><li>Make the index available to retrieval</li></ol>
<p>This runs separately from requests; retrieval records which index version it used.</p>
</section>
<section class="system-flow-lane">
<h3>Training and feedback</h3>
<ol><li>Join impression and action logs</li><li>Build examples, train, and evaluate</li><li>Release validated model and feature versions</li></ol>
<p>A release changes future requests, not the interpretation of past results.</p>
</section>
</div>

A request should be traceable to its model, feature, and index versions. That does not justify logging every raw input or a complete user profile. Record only what investigation requires, with access controls, retention limits, and a deletion process.

| Stage | Input | Output | First checks when something fails |
| --- | --- | --- | --- |
| Request entry | Permitted context, request ID, deadline | Validated request | Missing fields, duplicates, exhausted time budget |
| Multiple retrieval sources | Query representation, recent actions, item collection | IDs, sources, retrieval scores | Per-source count, coverage, timeout, index freshness |
| Features and eligibility | Candidate IDs, feature versions, eligibility rules | Rankable candidates and missing-value flags | Stale features or incorrect exclusions |
| Lightweight and heavy ranking | Features and model version | Scores and order | Input consistency and systematic harm to a user segment |
| Reranking and response | Ranked items, deduplication and display constraints | Final list | Duplicates, concentration, short lists, actual latency |
| Logging and feedback | Actual display events and later behavior | Joinable training and evaluation events | Whether an impression occurred, missing or late events |

These are responsibilities, not a requirement for six microservices. A small system can perform most steps in one service. Keep clear inputs, outputs, and measurement points, then split components when independent scaling or failure isolation warrants it.

## Where did the useful items disappear? {#funnel}

Three retrieval sources return 300, 200, and 200 items. That is 700 results, but 100 are duplicates, leaving 600 unique candidates. Adding raw source counts overstates coverage.

Suppose a separate manual check identifies 10 items that are both eligible and relevant to this request:

| Stage | Candidates remaining | Relevant items retained | Coverage of the original 10 |
| --- | ---: | ---: | ---: |
| Available collection | 10,000 | 10 | 100% |
| Retrieval and deduplication | 600 | 8 | 80% |
| Eligibility filtering | 450 | 6 | 60% |
| Lightweight ranking | 100 | 6 | 60% |
| Heavy ranking and reranking | 20 | 5 | 50% |

The two relevant items lost during filtering were eligible under our definition. Investigate the rule or its inputs before tuning the ranker. If the items were actually ineligible, revise the relevant set and denominator instead of recording a model failure.

Without a later stage adding candidates, the final relevant set is a subset of the retrieved set. Therefore:

$$
\mathrm{Recall}_{\rm final}\leq\mathrm{Recall}_{\rm retrieval}.
$$

Even a perfect downstream ranker cannot recover the two items missed during retrieval. Increasing each source's output also increases feature-fetch and scoring costs. Measure **additional unique relevant candidates**, not just the raw count.

### Why use more than one model? {#model-choice}

A two-tower model encodes the user and item separately. Item representations can be indexed in advance, making it useful for candidate retrieval. A heavier interaction model can examine user–item details jointly, but typically repeats that work for each candidate. It can rank a smaller set; a linear model or small MLP can also provide a simpler starting point.

| Choice | Main benefit | Cost or limitation | How to compare |
| --- | --- | --- | --- |
| Popularity or rules baseline | Training and complex indexing | Limited personalization and long-tail coverage | Separate new/returning users and popular/long-tail items |
| Two-tower retrieval | Precomputed item representations | Restricted user–item interaction; index maintenance | Hold downstream stages fixed; compare coverage and index cost |
| Heavy ranking model | Richer interactions on a small candidate set | More latency and compute | Hold candidates fixed; compare ranking gains and runtime |

Not every product needs three ranking stages. Direct scoring may be simpler for a small collection. An additional filter may pay off for a very large candidate set. Measure the cost and losses at each stage before adding complexity.

## Follow dependencies when budgeting latency {#latency}

Our example fetches user features and queries the candidate index in parallel because the query vector is already available. If building that vector requires the feature fetch, this parallelism is not possible.

| Step in this request | Duration | Dependency |
| --- | ---: | --- |
| Entry handling | 5 ms | First |
| User features / candidate query | 30 / 40 ms | Parallel; both must finish |
| Candidate feature join | 8 ms | Waits for candidate IDs |
| Scoring | 22 ms | Waits for features |
| Filtering and reranking | 5 ms | Waits for scores |
| Serialization and response | 5 ms | Last |

$$
T=5+\max(30,40)+8+22+5+5=85\ \mathrm{ms}.
$$

A 120 ms deadline leaves 35 ms of headroom for this request. Serializing the two parallel tasks raises the total to 115 ms, leaving 5 ms. This is **one request's timeline**, not a sum of component p99s. Estimate end-to-end p99 from request traces; correlations, queueing, and retries change the tail.

Decide timeout behavior in advance. A retrieval timeout might fall back to another candidate source with the same eligibility checks. A heavy-ranker timeout might retain the lightweight ranking. Privacy, authorization, and essential safety checks must still apply. Count and evaluate degraded responses separately; HTTP 200 alone does not establish success.

## A release changes more than weights {#release}

Imagine updating a two-tower model. Both the new query vector and old item index have 128 dimensions. Matching dimensions do not establish compatibility: changed training or normalization can leave dot products numerically valid but semantically wrong.

An immutable release record might look like this:

```yaml
release_id: feed-demo-v7
ranker: ranker-v7
feature_schema: features-v3
transform: transform-v4
query_encoder: encoder-v5
item_encoder: encoder-v5
candidate_index: items-v5-snapshot-42
fallback: popular-eligible-v2
```

This is an illustrative manifest, not framework configuration. It identifies the combination being served and lets rollback restore that combination. Feature deletion, content takedowns, and revoked permissions must still take effect independently: rolling back an index must not restore forbidden content.

[Google's ML engineering rules](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew) recommend reusing processing logic and sampling actual served features to check training–serving differences. Shared code removes one source of skew; it does not fix late data, stale indexes, or inconsistent defaults.

A release can progress through fixed-request replay, shadow execution, a small traffic experiment, and broader rollout. Shadow execution checks inputs and runtime without changing what users see. It cannot by itself establish a behavioral improvement; that needs a live randomized experiment or another suitable causal design.

## Make the handoffs explicit {#handoffs}

| Work area | What it delivers | What to verify together |
| --- | --- | --- |
| Data and features | Field meaning, time semantics, update frequency, missing-value rules | Can a historical request recover the features available then? |
| Retrieval and modeling | Candidate sources, model/index versions, quality report | Is the gain real on fixed candidates, and is the added computation worthwhile? |
| Platform and serving | Latency, capacity, release and rollback tools | Does the service maintain its agreed behavior under load and failures? |
| Product and experimentation | Success metrics, guardrails, experiment population, stopping rules | Who benefits, who loses, and is the change worth shipping? |

Team names and organization charts differ. What matters is an owner for each boundary. Not every failure calls for a better model, and not every product-metric decline calls for more replicas.

## Try a small investigation {#exercise}

Copy the candidate table and change one condition: a new source returns 200 items, of which 180 duplicate existing candidates. The remaining 20 add no relevant item to the labeled set. Would you retain the source?

The unique candidate count rises to 620, while coverage remains 80% on this labeled set. A decision still needs evidence about missing labels, diversity, gains on other requests, and added latency. One request is not enough to settle it. Continue with [training examples and online experiments](data-and-experiments.en.md).
