# Inside recommender systems: from one post to the whole architecture

[中文](README.md) · **English**

A good recommendation can look like a model simply guessed your interests. But before scoring, the system has chosen where to search and which posts are eligible. After scoring, it still needs to assemble a useful list.

Follow a feed refresh: retrieve candidates, score them, and assemble a list. Then return to the design choices—why split the work this way, and what changes with another model or index?

<span id="start-with-the-whole-systemthen-select-a-stage"></span>

## Separate requests, updates, and training

This is a teaching architecture, not Twitter / X’s production topology. Switch among 3 paths to separate a refresh, a content update, and a training run. On the request path, change one condition and trace where candidates disappear.

<div class="widget" data-recsys-architecture><p>Serving: request → retrieval → merging and features → scoring → list → feedback. Updates: content changes → encoding → index → compatible release. Learning: exposure and feedback → examples → training → evaluation → compatible release.</p></div>

Remember: **a box is not necessarily a separate service, and refreshing an item vector is not retraining a model.** The [architecture guide](00-architecture.en.md) explains both.

## Pick the question you came with

<div class="rd-route-grid widget">
<a href="00-architecture.en.md"><small>01 / ORIENT</small><strong>I want to understand the whole system</strong><span>Separate the 3 paths and the responsibilities of models, indexes, and data.</span></a>
<a href="01-feed-pipeline.en.md"><small>02 / TRACE</small><strong>I want to follow a post through a request</strong><span>Read public retrieval, ranking, and selection code. Track where a candidate survives or disappears.</span></a>
<a href="06-why-two-towers.en.md"><small>03 / DESIGN</small><strong>I want to understand the choices</strong><span>Change query vectors yourself, then compare encoders, indexes, caches, and release strategies.</span></a>
<a href="05-evaluation-lab.en.md"><small>04 / VERIFY</small><strong>I want to know whether it actually improved</strong><span>Keep the model fixed and change only the candidate pool or denominator.</span></a>
</div>

For a first pass, try **architecture guide → 01 → 02 → 03 → 05**. Return to 04 for newer approaches and 06–08 for design details. You do not need to read every source file first.

## One question per episode

| Chapter | Question | Try this after reading |
| --- | --- | --- |
| [00 · Architecture guide](00-architecture.en.md) | How do serving, updates, and learning connect? | Label when each component runs and what it produces |
| [01 · One refresh](01-feed-pipeline.en.md) | Why isn’t recommendation one model? | Trace a candidate’s source, features, and outcome |
| [02 · Multiple sources](02-candidate-retrieval.en.md) | What does another source actually add? | Check unique relevant candidates at a fixed budget |
| [03 · Ranking and lists](03-ranking-and-diversity.en.md) | Why can good posts make a dull list? | Adjust the penalty; separate scores from selection |
| [04 · Place the new method](04-modern-recsys.en.md) | Which stage does a Transformer or generative retriever replace? | Put the model back into the architecture |
| [05 · Evaluation lab](05-evaluation-lab.en.md) | Did the model change, or just the score? | Switch the pool, Top-K, and denominator |
| [06 · Why two towers](06-why-two-towers.en.md) | What computation is saved, and what interaction is lost? | Compare a mixed query with separate queries |
| [07 · Component choices](07-component-choices.en.md) | What could implement the same responsibility? | State constraints, alternatives, and stopping conditions |
| [08 · Engineering the path](08-serving-lifecycle.en.md) | Can a successful response still be wrong? | Check compatibility, timeouts, freshness, and rollback |

Each note compares alternatives and works through computation, latency, or candidate budgets. The small teaching datasets explain mechanisms; whether real users prefer the results requires separate validation.

<span id="what-comes-from-public-code-and-what-is-a-teaching-design"></span>

## Public sources used in this series

| Material | How it is used | What it does not establish |
| --- | --- | --- |
| [Twitter 2023 · ee5e7fc](https://github.com/twitter/the-algorithm/blob/ee5e7fc18dc0e971a6c02826b196294048765817/README.md) | Public-component reading in 01–03 | Today’s complete X production architecture |
| [X public snapshot · 77d431a](https://github.com/xai-org/x-algorithm/blob/77d431aabf409ca1c1eed9bec7e2183f7c914e23/README.md) | Version comparison in 04, checked 2026-09-30 | Real training data, all experiment settings, or production gains |
| Official papers and engineering material | Compare the assumptions behind designs | One company’s approach as the only answer |
| Our diagrams, code, and interactions | Examine a mechanism using invented data | Any company’s business results or internal implementation |

Links pin revisions rather than silently combine different generations into a “current system.” No internal projects, private data, or confidential interview material appear here.

Try designing your own [recommendation feed](../../learn/system-design/feed.en.md). If vectors are unfamiliar, start with [embeddings and similarity](../../00-foundations/core/embeddings-and-similarity.en.md).
