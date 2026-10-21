# Data to model: the deliverable is more than a checkpoint

[中文](data-to-model.md) · **English**

> Original teaching design, checked 2026-10-11. Continues the [document search system](README.en.md); no new trained-model results are claimed.

When a relevant document is missing, it may never have been ingested, its title may have been lost during chunking, the embedding may be unsuitable, or the evaluation may contain only easy queries. Separate these before deciding whether to repair the pipeline or train a model.

## 1 · Ingestion: retain provenance {#ingestion}

Documents need IDs, revisions, sources, content times, arrival times, and access scope. Images also need processing metadata such as dimensions and processor version. Behavioral events need stable event IDs, request IDs, and occurrence times. Raw content and identity retention must follow permissions, purpose, and retention limits—not a “log everything” policy.

| Situation | Required handling | Failure otherwise |
| --- | --- | --- |
| Retried event delivery | Idempotent deduplication by stable event ID | One click becomes several labels |
| Document update | Revision and replacement relationship | Search and citations reference different versions |
| Deletion or permission withdrawal | Tombstone / access event with propagation checks | Old indexes or caches keep returning removed content |
| Parsing failure | Error classification, retry, quarantine | Silent loss changes the data distribution |

A versioned table can do much of this in a small project. A message queue does not remove the need to define duplicates, ordering, late arrival, and replay. “We use Kafka” is not a data contract.

## 2 · Features: was this known at the time? {#point-in-time}

Suppose a prediction is made at 10:00. An event occurs at 09:58 but enters the system at 10:02. Replaying today's complete logs must not make it available to the 10:00 example.

For hand calculation, use integer minutes. A real pipeline also needs timezones, clock conventions, and late-arrival rules. This example looks up one feature for one entity, requiring both event and availability times to be no later than the query:

```python
def feature_as_of(records, entity, query_time):
    eligible = [
        record for record in records
        if record["entity"] == entity
        and record["event_at"] <= query_time
        and record["available_at"] <= query_time
    ]
    return max(eligible, key=lambda record: (record["event_at"], record["available_at"]), default=None)


records = [
    {"entity": "user-a", "event_at": 590, "available_at": 591, "value": 2},
    {"entity": "user-a", "event_at": 598, "available_at": 602, "value": 3},
]
assert feature_as_of(records, "user-a", 600)["value"] == 2
assert feature_as_of(records, "user-a", 603)["value"] == 3
```

This example assumes upstream deduplication and handles a single feature. Equal-time revisions, TTLs, and missing-value handling require additional rules. [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) illustrate historical feature retrieval, but availability time must be recorded and handled by your pipeline; a feature-store setting cannot reconstruct missing history.

## 3 · Examples: logs are not labels {#labels}

A user may not click because they never saw the result, disliked the title, already knew the answer, or found it irrelevant. Negatives need exposure, position, observation-window, and task definitions. Unseen candidates and explicit rejections should not silently become the same label.

| Evidence | What it offers | Remaining uncertainty |
| --- | --- | --- |
| Clicks and dwell | Responses to displayed content | Resolution of the task; position and exposure bias |
| Human relevance labels | Query–document judgments under a rubric | Agreement and coverage |
| Synthetic queries / teacher scores | Additional candidate supervision | Teacher bias, duplicates, calibration |
| Explicit irrelevant feedback | Clearer rejection | Mistakes and delayed feedback |

Separate training, development, and test by appropriate sources, times, or query families. Randomly splitting neighboring chunks of one PDF may mostly measure document memorization. Each data delivery should report deduplicated size, label balance, missingness, and source changes.

## 4 · Models: build a diagnostic baseline {#training}

Check keyword retrieval before adding dense retrieval; their errors may differ. Screenshot queries can use an image–text encoder such as [SigLIP](../../03-multimodal-learning/siglip.en.md), but its image vectors do not belong in an arbitrary text embedding space. Retrieve candidates through compatible channels, then fuse or rerank explicitly.

A retriever needs positive, negative, and false-negative checks. SFT for citation behavior needs chat-template, target-mask, and valid-token weighting checks. These are different supervision problems. See [dual encoders](../../04-search/dual-encoder.en.md) and [data to batches](../post-training/data-pipeline.en.md).

If training does not fit, account for weights, optimizer, activations, and transient memory before choosing [DDP / FSDP2 / ZeRO](../../06-systems/training-frameworks.en.md). Before scaling, compare loss and one update on a fixed batch; afterward, inspect sample duplication, denominators, and accumulation. A faster implementation of the wrong objective is not progress.

## 5 · Checkpoints: recovery and deployment differ {#checkpoint}

| Purpose | Save or associate | Acceptance test |
| --- | --- | --- |
| Resume training | Model, optimizer, scheduler, RNG, data cursor, update count, configuration, data revision | Interrupt and compare recovered samples and updates |
| Serve inference | Weights / adapter, base model, config, tokenizer / processor, template, precision, dependencies | Fixed-input regression through the actual loader |
| Reproduce an experiment decision | Code revision, environment, splits, metric definitions, checkpoint-selection rule | Re-run the comparison |

A sharded training checkpoint may not load directly into a serving engine; an adapter also requires the correct base. Export, quantization, and merging create artifacts that need their own acceptance checks. Training-time scores do not validate the deployed artifact. See [save and resume](../post-training/checkpoint-and-resume.en.md).

## 6 · Indexes: compatible dimensions are not enough {#index}

Old encoder A and new encoder B may both output 768 dimensions. That matches shape, not coordinate meaning. Encoding queries with B and searching document vectors from A may destroy the intended alignment.

Associate the model and index in a release:

```json
{
  "release": "search-demo-v2",
  "query_encoder": "encoder-b",
  "document_encoder": "encoder-b",
  "processor": "processor-b",
  "index": "index-b-snapshot-17",
  "document_snapshot": "documents-17",
  "ranking_config": "ranking-4",
  "evaluation_report": "eval-search-demo-v2"
}
```

This is a teaching manifest, not a registry's configuration format. Real identifiers should resolve to immutable revisions or content hashes. Some dual encoders intentionally use different query and document models; record the **jointly trained or validated compatible pair**, not a requirement for identical files.

Build the replacement index alongside the current one, check coverage, deletion, permissions, and retrieval quality, then switch read traffic. Roll back the compatible bundle together. Reverting only the query encoder may leave a broken combination.

## 7 · Evaluate components and the product {#evaluation}

Fix queries and candidate eligibility, then ask separately: was the relevant document ingested, retrieved, ranked well, and used correctly in the summary? A final answer score alone cannot locate the failing module.

| Layer | Minimum checks | Interpretation trap |
| --- | --- | --- |
| Data / index | Coverage, freshness, deletion, permissions | More coverage does not imply better relevance |
| Retrieval | Recall@k at fixed budgets, language and query slices | Increasing k is not a free improvement |
| Ranking | NDCG or suitable metrics, failure examples | Changed labels or candidate pools break direct comparisons |
| Generation | Support, citations, abstention, task completion | Valid format does not establish factual support |
| Cost | Latency, memory, encoding, training, index refresh | One forward pass is not the whole cost |

Set release gates before inspecting results and reserve the test set for final comparisons. Small samples require uncertainty and failure details, not claims of stable gains from one additional correct answer. Continue with [serving, monitoring, and feedback](serving-and-feedback.en.md).
