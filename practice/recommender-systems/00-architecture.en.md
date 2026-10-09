# 00 · Recommender architecture: requests, content updates, and training

[中文](00-architecture.md) · **English**

> Teaching architecture · Last reviewed: 2026-09. All examples are invented, not descriptions of a company’s production configuration.

A refresh, an edited post, and a completed training run can all change what a reader sees. They are not the same operation. Draw them as one long arrow and it becomes easy to imagine that every request retrains the model and re-encodes the catalog.

Before choosing a model, establish **what runs when, what it produces, and who consumes the result**.

## 1 · Separate the timelines first

| Path | Trigger | Main output | Typical concern |
| --- | --- | --- | --- |
| A refresh · online | Opening a page or requesting the next list | Useful, eligible results within a deadline | Too few candidates, slow dependencies, invalid eligibility |
| Content updates · batch / nearline | Publication, edits, deletions, or scheduled refreshes | Updated features, vectors, and indexes | Old events overwriting new content; delayed deletion |
| Model learning · offline | Data and experiments are ready | A new model and evidence for a release decision | Wrong labels, temporal leakage, aggregate-only evaluation |

Offline means outside this request’s critical path, not that the model never changes. Nearline is not a universal number of seconds either: it reacts asynchronously to updates. The acceptable delay depends on requirements and cost.

<div class="widget" data-recsys-architecture data-lane="update"><p>Content changes can trigger re-encoding with the released model and an index update. Training changes the encoder itself and requires checking query/item/index compatibility. Online requests use these prepared components rather than retraining them each time.</p></div>

## 2 · Connect the paths through one new post

Imagine a small community for photography and programming tutorials. An author publishes an astrophotography guide; a reader then opens their feed.

1. **Publication.** The update path records the content version, produces a reusable representation, and makes it searchable. It does not re-encode the post separately for every reader.
2. **Refresh.** Current context helps find candidates. Retrieval narrows the pool, ranking examines it more closely, and selection forms a list. Eligibility may be checked at several stages.
3. **Actual exposure.** Exposure, position, and subsequent behavior enter the data. Unshown is not disliked, and inactivity is not necessarily rejection.
4. **Another training run.** The team revisits examples or objectives, trains a model, and compares releases. After validation, compatible components are deployed together—not a new checkpoint against an unrelated old index.

Retrieval followed by ranking is a common division of work; the [TensorFlow Recommenders tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) provides a runnable introduction. This fictional community and update path are our teaching design, not the tutorial’s deployment configuration.

## 3 · Similar boxes can represent different decisions

| Decision level | Examples | Do not confuse it with |
| --- | --- | --- |
| Responsibility | Find candidates, predict behavior, limit repetition | One responsibility need not mean one service |
| Model architecture | Dual encoders, sequence encoders, multi-task rankers | “Uses a Transformer” does not identify its place in the system |
| Algorithm or data structure | Dot product, HNSW, IVF-PQ | Accurate neighbor search is not accurate preference prediction |
| Engineering implementation | Batch jobs, caches, RPCs, index services | A framework alone is not a complete reliable system |

For example, **Faiss can implement vector search; it does not automatically handle permissions, event ordering, or release rollback**. Define the boundary before choosing the library. The [official index list](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) describes search implementations; verify update, deletion, and filtering support in the implementation you actually use.

## 4 · Even a vector carries a contract

Two components producing 256 numbers each are not automatically compatible. For every arrow, ask:

- **What does it mean?** An item ID, a query vector, or a behavior probability? A cosine score is not a probability.
- **Which version?** Do query and item representations share a compatible space, preprocessing, and normalization?
- **How long is it valid?** After an edit, how long can old features remain usable? Can deletion take effect promptly?
- **What does failure mean?** A successful empty result, a timeout, and an incompatible release are not the same retrieval outcome.

For example, an old item index and a new user encoder can still produce dot products, but those scores may no longer mean what they did during training. Check version compatibility before retraining to fix an apparent regression.

## 5 · Ask where a change will propagate

Before drawing another box, trace one small change:

| Change | What it affects | First check |
| --- | --- | --- |
| Replace the item encoder | Item vectors, query compatibility, index | Can both sides still be compared using the intended score? |
| Add a retrieval source | Merge, deduplication, budget, downstream scoring cost | Does it add unique relevant candidates at a fixed budget? |
| Increase a diversity penalty | List selection, not necessarily model parameters | How do relevance and repetition change separately? |
| Change training labels | Objective meaning, data coverage, evaluation protocol | Can the old metric detect the added signal? |

Understanding architecture means **tracing the information, cost, and risk that one decision passes downstream**. A bigger diagram is not necessarily a better explanation.

## Diagnose one failure with three timestamps

Consider a fictional trace: a post is published at 10:00, a user refreshes at 10:02, and the updated item vector becomes searchable at 10:05. Replaying the 10:02 request against today's latest features uses a vector unavailable to that request.

| Timestamp | Meaning | This example |
| --- | --- | --- |
| Event time | When content or behavior occurred | Published at 10:00 |
| Processing completion | When encoding completed | 10:04 |
| Search availability | When the index could return it | 10:05 |

“Not retrieved at 10:02” may indicate slow updates rather than weak semantic understanding. Inspect stage delays and retries before changing the encoder. The same applies to training features: use information available before the decision, not future information reconstructed afterward.

One approach retains replayable feature/index snapshots and the version ID used by each request. Storage and maintenance increase, so this need not mean a full snapshot every second; periodic snapshots plus incremental events may suffice. Whatever the implementation, it should establish what the system could see then rather than explain yesterday with today's data.

## Try tracing it yourself

<details><summary>A new post cannot be found. Should you first replace the ranker?</summary><p>Check whether it became searchable, its encoding and index were updated, and eligibility checks passed. The ranker only compares supplied candidates; it cannot recover content upstream never delivered.</p></details>

<details><summary>The new item tower has the same dimensions. Can the old query tower stay?</summary><p>Not on dimensionality alone. Require a validated compatibility relationship, fixed-example comparisons, and a release protocol. Plausible dot products may no longer express the trained relationship.</p></details>

Continue to [01 · One refresh](01-feed-pipeline.en.md) to apply this reading method to public code. For a hands-on start, try [06’s vector experiment](06-why-two-towers.en.md) or [05’s evaluation experiment](05-evaluation-lab.en.md).
