# 04 · Modern recommendation: which layer changed?

[中文](04-modern-recsys.md) · **English**

> Checked 2026-10-09. The X discussion uses the linked `77d431a` snapshot. Public code and papers only—not a claim to know complete production configurations.

## Newer doesn't mean replacing everything

The earlier episodes use a 2023 snapshot. The newer [X repository](https://github.com/xai-org/x-algorithm/blob/77d431aabf409ca1c1eed9bec7e2183f7c914e23/README.md) lists sources including Thunder, Phoenix, and SimClusters and separates ranking from visibility decisions. An old component map cannot stand in for it.

[Phoenix's documentation](https://github.com/xai-org/x-algorithm/blob/77d431aabf409ca1c1eed9bec7e2183f7c914e23/phoenix/README.md) still distinguishes dual-encoder retrieval from richer Transformer ranking. This release includes training and serving implementations with a synthetic-data local path; that doesn't supply real training data or reproduce production impact.

## Ask what the model replaces

For each new name, write down **its inputs, its outputs, and the old stage it replaces**. That tells you more than whether it uses an LLM. Return to the [architecture guide](00-architecture.en.md) when you need to place it in the system.

```mermaid
flowchart LR
 A["History and content"] --> B["Representation / sequences"]
 B --> C["Candidate retrieval"]
 C --> D["Multi-objective scoring"]
 D --> E["List choice and eligibility"]
 E --> F["Exposure and feedback"]
 F -. "Update data and objectives" .-> A
```

A Transformer may encode history, score candidates, or autoregressively generate item IDs. Its name alone doesn't tell you which stage changed.

| Direction | What changes | Questions to ask |
| --- | --- | --- |
| Long-history modeling | Representation of actions and order | Is history available? Where is recomputation done? |
| Richer candidate interaction | How context affects a prediction | Can candidates attend to each other? Does batch composition change scores? |
| Multimodal representation | Information from text and images | Is the information incremental and actually used? |
| Generative retrieval | Produce identifiers rather than only vector neighbors | What about valid IDs, fresh items, and decoding cost? |

## Three sources, three different questions

1. **X Phoenix: can other candidates contaminate a score?** Candidate isolation lets candidates attend to user context rather than one another. My interpretation is that it makes individual scoring and list diversity easier to analyze separately—not that predictions can be cached across arbitrary users and requests. [Architecture](https://github.com/xai-org/x-algorithm/blob/77d431aabf409ca1c1eed9bec7e2183f7c914e23/phoenix/README.md)
2. **Meta's multi-stage sequence model: where does expensive work belong?** Its 2026 article separates heavier user modeling from lightweight online ranking. The transferable question is compute reuse versus freshness, not whether another system can copy its scale or reported gains. [Official article](https://engineering.fb.com/2026/08/05/ml-applications/from-user-sequences-to-scaling-laws-a-multi-stage-architecture-for-metas-ads-ranking/)
3. **TIGER: can retrieval generate identifiers?** The paper uses Semantic IDs and a sequence model to predict the next item's identifier. This is a research direction, not proof that every platform replaced ANN, and not a chatbot writing recommendation explanations. [Paper](https://arxiv.org/abs/2305.05065v3)

Another distinction: **using Semantic IDs does not by itself make retrieval generative.** The Phoenix snapshot uses SIDs as item-side inputs and still compares vectors. TIGER generates ID sequences as its target. Look at whether the model outputs a vector or decodes an identifier—not simply whether it uses SIDs.

## What hasn't disappeared

Feedback remains exposure-dependent, new content lacks interactions, and serving has latency and cost limits. A larger model may improve representation but doesn't reveal preferences for unexposed content.

Start with a testable hypothesis about missing information or a bottleneck. “Generative is the trend” isn't one.

## Test generative retrieval as one source first

You need not replace the whole system on day one. Keep the existing retriever and ranker, decode generated IDs into candidates, and apply the same visibility, deduplication, and ranking path. This incremental experiment asks whether another source supplies missing useful candidates—not whether end-to-end generation is universally superior.

Suppose a decoder emits `[A, B, A, D, ?]`: A and B are currently visible, D resolves but was deleted, and `?` cannot be resolved. Counting emissions, 4/5 resolve and 3/5 are currently eligible. After deduplication, only A and B can reach the next stage. **Valid-ID rate and distinct candidate count are different metrics.**

Then ask whether those two items are relevant, whether existing sources already found them, and how much decoding time they cost. Five emissions aren't five recovered candidates.

| Held fixed | Deliberately varied | Report separately |
| --- | --- | --- |
| Queries, labels, visible-catalog snapshot | Add a generated candidate source | Valid IDs and uniquely relevant candidates |
| Total merged budget and ranker | Source quotas | Relevance, coverage, and where candidates are removed |
| Latency and compute accounting | Decoding length or search width | Timeouts, added compute, value of the increment |

Assigning new item IDs and invalidating old ones remain data-maintenance problems. Constrained decoding can exclude certain invalid outputs without guaranteeing current visibility. The same reasoning applies to multimodality: establish what information images add, then compare a controlled image-masked condition. A newer model name cannot replace that comparison.

## Self-check

<details><summary>Does using a Transformer make a recommender generative?</summary><p>No. An encoder may output vectors and a ranker may predict action probabilities. Inspect the objective, output, and whether it generates candidate identifiers.</p></details>

<details><summary>Why isn't a synthetic-data run a production result?</summary><p>It checks interfaces, numerical behavior, and execution. Real impact also depends on distributions, candidate sets, configuration, experiments, and users.</p></details>

Next: [a small evaluation experiment](05-evaluation-lab.en.md).
