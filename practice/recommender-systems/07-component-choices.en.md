# 07 · What could each box in the diagram be?

[中文](07-component-choices.md) · **English**

> Last checked: 2026-10-09. Teaching choices, not a Twitter / X configuration list or tool benchmark.

## Don't start by choosing a model

Use the fictional community from [the feed exercise](../../learn/system-design/feed.en.md). Is the problem missing candidates, poor ordering, or new content arriving late? Replacing one component won't necessarily fix all three.

Separate **a component's responsibility from its implementation**. Vector retrieval is a responsibility; exhaustive search, HNSW, and IVF-PQ are implementation choices. PyTorch, Faiss, and databases also occupy different abstraction layers.

## 1 · How should users and items be encoded?

Start with two fictional settings:

| Both recommend content | First experiment | Why, and when to change |
| --- | --- | --- |
| Small, slowly changing catalog; requirements still being tested | Simple features and an exact-search baseline | Make quality and failures understandable first; add approximation when measured search cost justifies it |
| Constant new content, little ID history, limited memory | Content representations, event updates, and a compressed-index experiment against exact search | Seek new-item coverage while paying encoding, refresh, and approximation costs; measure whether it pays off |

These are starting points, not prescriptions. A small catalog can have high traffic; limited memory does not automatically imply PQ. Write down your constraints rather than letting a comparison table make the decision.

| Implementation | When to consider it | Cost and check |
| --- | --- | --- |
| ID embeddings + small MLP | Stable repeated interactions and a cheap starting point | Weak support for unseen IDs; inspect cold-start and low-frequency slices |
| Text encoder + structured features | Informative descriptions that might transfer to new items | Cleaning, truncation, encoding, refresh cost; test added semantics with fixed labels and pool |
| Behavior-sequence encoder | Order or recent changes may matter | History reads and computation; compare shuffled order, shorter histories, and simple pooling |
| Image / multimodal encoder | Text genuinely omits visual evidence | Processing, missing modalities, cache invalidation; ablate images to test incremental value |

These can be combined; they aren't a model leaderboard. ID and content features can share a tower, while sequence modeling can live only on the user side. **Name the missing information before choosing the component.**

If item titles are absent, repairing the input may matter more than enlarging the encoder. If shuffling history changes nothing, the case for expensive order-sensitive modeling remains unproven.

## 2 · Choosing an index is more than asking what's fastest

Start with exact retrieval over the same vectors. Keep its top-k as a reference for approximation. This asks whether the index recovers what the model would choose—not whether the model understands users.

| Path | When to try it | Main trade-off |
| --- | --- | --- |
| Flat exact scan | Small sets, implementation checks, or sufficient hardware throughput | No approximation loss; scanning grows with catalog size |
| HNSW | Exploring a graph-search latency–recall trade-off | Extra graph memory; test search settings, build time, and update support together |
| IVF / IVF-PQ | Narrowing search or reducing storage | Partitioning and quantization can lose precision; index training adds maintenance |
| Packaged vector service | Persistence, filtering, sharding, operational management | Another dependency; verify search semantics and version capabilities |

Update and deletion behavior depends on the implementation. “HNSW” alone doesn't specify it. Faiss is a search library, not automatically a complete distributed database. [Faiss index types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) · [Selection guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)

Compare **neighbor recovery, p95/p99 latency, memory, and build/update time** together. A QPS number without dimensions, hardware, and filtering conditions isn't enough.

<details markdown="1">
<summary>A common trap: training with cosine, searching with inner product</summary>

For nonzero vectors, cosine is the normalized dot product:

$$\cos(q,v)=\frac{q^\top v}{\|q\|\|v\|}.$$

Inconsistent normalization can let vector norms change rankings. Shapes match and the request succeeds, but it computes a different score. [Faiss distance definitions](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

With $q=(1,0)$, $A=(2,0)$ and $B=(100,100)$, the dot products are 2 and 100, putting B first. Cosines are 1 and $1/\sqrt{2}\approx0.707$, putting A first. B is longer, not better aligned.

Conversely, omitting normalization doesn't always change the order. With unit item vectors, positive rescaling of **one nonzero query** preserves exact dot-product rankings while scaling scores. That scale still matters for absolute thresholds or comparisons across queries. Normalize both sides when actual cosine values are required.

Hand-calculate a few fixed vectors and compare offline and serving paths. Test zero vectors, NaNs, duplicate IDs, and deterministic tie-breaking too.

</details>

## 3 · What can the ranker be?

| Option | What it can examine | Why not start with the heaviest one? |
| --- | --- | --- |
| Rules / linear model / trees | Defined features and combinations | Useful baselines; test whether they express the required interactions |
| Small MLP / multi-task ranker | User–item interactions and several behavior objectives | Missing labels, task conflicts, calibration, and loss weights need handling |
| Sequence or joint-encoding model | Richer history–candidate or text–text matching | More input access and computation; test incremental value on a small set first |

Lower in this table doesn't mean better. Candidate generation followed by a more expensive ranker is an established division; boundaries depend on the workload and goals. [YouTube 2016 paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)

A ranker cannot recover candidates retrieval never supplied. Conversely, when the pool already contains enough relevant items, expanding it may mostly pass cost downstream.

## 4 · Where should features and representations be computed?

| Placement | Suits | What to watch |
| --- | --- | --- |
| Batch | Stable features and full backfills | New or edited items wait for the next run |
| Event-driven / nearline | Refreshing representations after publication or edits | Duplicate/out-of-order events, retries, and backlog |
| Request-time / online | Information only available for this request | Tail latency, timeouts, missing features, fallbacks |

These are complementary. Build a base index in batch, handle changes through events, and add essential context online. Multiple paths must still agree on the meaning of each field.

## 5 · Don't hide merging, eligibility, and diversity inside one box

- **Merging and budgets:** deduplicate and retain provenance. Don't compare unrelated raw model scores directly; validate common scoring, source quotas, or rank-based fusion.
- **Eligibility and permissions:** apply inexpensive restrictions early and recheck critical eligibility before returning. Deletions cannot wait solely for a vector backfill.
- **List experience:** author caps, repetition, and exploration budgets act on the selected list. They relate to individual relevance but aren't identical to it.

Late filtering can leave too few results; blindly overfetching adds cost. Track removals at each stage, by source and cohort.

## Write a choice as a falsifiable statement

Instead of “use an advanced multimodal recommendation architecture,” try:

“We suspect text misses a particular visual signal. Compare image-enabled and text-only versions with a fixed pool and label protocol. Add the item-side path only if the relevant gains justify refresh and storage costs.”

That gives an observation, intervention, control, and stopping condition. Someone else can challenge it or continue the experiment.

## Work through a choice, not just a table

Suppose a teaching service cannot retrieve new posts. Inspect failed cases: source text exists, encoding succeeded, but the index lacks the matching revision; old-item retrieval and ranking work. A stronger encoder won't repair this diagnosed update delay.

Compare shorter batch intervals with event-driven updates. Batching changes less but repeats scans and retains scheduling delays. Event-driven work responds sooner to edits but must handle duplicates, reordering, backlogs, and recovery. A small team with modest freshness needs may reasonably choose batching; “nearline” isn't inherently the better answer.

Hold content and encoding fixed, then measure publication-to-search delay, missed updates, idempotency after duplicate events, and encoding/index-write costs. Define a freshness target and test which design meets it. Also verify that more frequent writes do not worsen existing query latency.

If the current revision is indexed but exact search ranks it poorly, investigate representation and training signals instead. If exact search ranks it highly but ANN misses it, investigate index settings. **“Can't find it” can require three entirely different changes.**

## Self-check

<details><summary>ANN recovery is high but user relevance is poor. Where next?</summary><p>The index may accurately retrieve the model's wrong neighbors. Check representations, supervision, and task evaluation; approximating exact search isn't approximating true preferences.</p></details>

<details><summary>When a post changes, is updating its database text enough?</summary><p>No. Consider vectors, feature caches, index versions, and when old results become invalid. Define the update path and freshness requirement.</p></details>

Next: [From component diagram to a serving path](08-serving-lifecycle.en.md)
