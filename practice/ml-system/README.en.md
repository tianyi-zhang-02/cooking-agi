# An ML system, from data to users

[中文](README.md) · **English**

> Original teaching design, checked 2026-10-11. The product, data, and budgets are hypothetical, not a company's architecture or production measurements.

“I work on the model” can mean the objective, retrieval embeddings, online ranking, or serving performance. Different teams may own each. Understanding the whole system does not mean building everything yourself; it means knowing where your inputs come from and who consumes your outputs.

We follow a **document search product**. A user submits a question and optionally a screenshot. The system returns accessible sources and, when useful, a cited summary. If search results satisfy the need, stop there. An LLM is not required to complete the diagram.

## Choose your entry point {#routes}

| Question | Start here |
| --- | --- |
| What belongs in a complete project? | This page's system map and handoffs |
| How do data become models, indexes, and releases? | [Data to model](data-to-model.en.md) |
| What happens during a request or an incident? | [Serving and feedback](serving-and-feedback.en.md) |
| Where do FSDP, vLLM, and SGLang fit? | FSDP in [data to model](data-to-model.en.md#training), vLLM / SGLang in [after deployment](serving-and-feedback.en.md#generation), then [training stacks](../../06-systems/training-frameworks.en.md) and [inference engines](../../06-systems/inference-frameworks.en.md) |
| How do image and text retrieval connect? | [CLIP](../../03-multimodal-learning/clip.en.md) → [SigLIP](../../03-multimodal-learning/siglip.en.md), then index compatibility |

## Separate three timelines {#three-loops}

<figure class="worked-update" lang="en">
<ol>
<li><small>01 / Content updates</small><strong>Documents → processing → encoding → index</strong><span>Today's document need not wait for another training run. Deletion and permission changes must propagate too.</span></li>
<li><small>02 / Live requests</small><strong>Query → retrieval → ranking → display</strong><span>Add generation if a summary is needed. This path has immediate permission, quality, latency, and failure constraints.</span></li>
<li><small>03 / Model iteration</small><strong>Feedback → dataset → training → evaluation → release</strong><span>Logs are evidence to inspect, not automatically reliable labels. Validate a version before changing user results.</span></li>
</ol>
<figcaption>The timelines share versions and logs but run at different frequencies. Indexing is not training; releasing a model is not merely replacing weights.</figcaption>
</figure>

A simple “data → training → inference” diagram hides index refreshes, client retries, deletion propagation, and the effect of live decisions on future training data. [Hidden Technical Debt in ML Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/) examines such surrounding dependencies and feedback. This series develops an original example of how to handle them.

## What each component delivers {#components}

| Component | Input → output | Contract to establish |
| --- | --- | --- |
| Requirements and metrics | User problem → success criteria and limits | Relevant, accessible, timely; unacceptable errors |
| Ingestion | Documents / events → versioned records | IDs, source, time, permission, deletion events |
| Cleaning and splits | Records → training and evaluation examples | Deduplication, leakage, invalid and missing values |
| Features and encoding | Text / images → features or vectors | Preprocessing, dimensions, revision, missing-value behavior |
| Learning objective | Examples and labels → loss | Positives, negatives, masks, sample / token weights |
| Training execution | Snapshot and configuration → checkpoint | Retry, recovery, update provenance |
| Offline acceptance | Candidate version → slice metrics and failures | More than aggregate scores; no repeated test-set tuning |
| Artifact release | Weights, processor, index → release bundle | Which artifacts must switch together, and how to roll back |
| Serving | Request → candidates, scores, answer | Authorization, deadlines, limits, caches, cancellation, fallback |
| Monitoring and feedback | Results and exposures → diagnosis and future data | What was actually shown, with privacy boundaries |

These responsibilities do not each require a platform. A first version can use a few files, one process, and a fixed dataset. Split components when independent scaling, access control, or failure isolation actually helps.

## Who owns what? {#ownership}

This is **one possible division of work**, not a universal organization chart. A small team may combine roles; a large organization may divide them further. MLE, SWE, and research scientist (RS) titles alone do not establish ownership.

| Boundary | Possible lead | Deliverable | Joint acceptance |
| --- | --- | --- | --- |
| User needs and constraints | Product, application team | Metrics, interactions, failure expectations | ML, data, serving |
| Documents and event streams | Data engineering | Schemas, snapshots, quality checks, deletion notices | Features and training |
| Retrieval / ranking / vision | Applied ML, research | Model, processor, evaluation, limitations | Data, platform, application |
| Training and recovery | ML platform, training infrastructure | Jobs, resources, checkpoint support | Model team verifies the objective |
| Indexes and execution | Search, inference, backend | APIs, budgets, fallback | Model team verifies compatibility |
| Release, reliability, security | Service owner, SRE, security | Rollout controls, alerts, runbooks, authorization | Relevant data and model owners |

If the model team changes embeddings from 768 to 1024 dimensions, “the new model is better” is not a sufficient handoff. Indexes need rebuilding, requests need version identification, rollback needs the old index, and evaluation needs migration coverage. **Interfaces and acceptance are shared work, not a one-way transfer of a weights file.**

## Six questions for a module handoff {#handoff}

For an embedding service:

| Question | Example agreement |
| --- | --- |
| What goes in? | Authorized text or image references with content revisions |
| What comes out? | Fixed-dimension vectors, encoder revision, normalization contract |
| How is correctness checked? | Fixed reference examples and offline/online preprocessing parity |
| When does it return? | A target under normal load, a deadline, timeout error codes |
| What happens on failure? | Explicit error; the caller decides whether keyword fallback is acceptable |
| Who manages changes? | Named owner, compatibility window, migration and rollback plan |

An API schema is only part of this. A request ID should connect input revisions, model version, and index version. Otherwise every service may report success while the combined result is wrong.

## Different products need different paths {#variants}

| Product | Main differences | Checks that remain |
| --- | --- | --- |
| Tabular classification / risk prediction | Feature computation, delayed labels, thresholds; possibly no retrieval or LLM | Point-in-time correctness, calibration, drift, rollback |
| Recommendation feed | Histories, candidates, exposures, ranking, exploration | Selection bias, freshness, slices, feedback loops |
| RAG / knowledge assistant | Document access, retrieval, context, citations | Evidence versions, hallucinations, tools, costs |
| Multimodal search | Image preprocessing, aligned representations, indexes | Pair quality, resolution, encoder compatibility |

The purpose of the map is to identify relevant components, not memorize an architecture containing every available tool.

## A manageable exercise {#exercise}

Start with 20 public documents you may use and ten original questions. Build keyword search first and record returned document IDs and revisions. Add vector retrieval without changing the evaluation questions; inspect which relevant sources it actually adds. Then handle one failure explicitly: a changed document, withdrawn permission, incompatible index, or timeout.

Training can wait. Explaining freshness, result provenance, and recovery already goes beyond wiring APIs together. When training becomes useful, continue to [data to a releasable model](data-to-model.en.md).
