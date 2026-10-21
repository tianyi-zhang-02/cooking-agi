# After deployment: requests, failures, and feedback

[中文](serving-and-feedback.md) · **English**

> Original teaching design, checked 2026-10-11. Continues the document search example; times and traffic are hypothetical, not production measurements.

A model can look good offline and still disappoint users. Answers may be wrong, late, filtered down to too few results, or served from an old index. Make a request traceable before deciding to retrain.

## 1 · Admission: deciding whether to accept work {#entry}

The client supplies a query, optional image, request ID, and related inputs. The server resolves identity and authorization, validates size, type, and length, and sets a deadline. Output-token limits and wall-clock deadlines are different controls.

Queues cannot grow indefinitely. Design rate limits, queue bounds, timeouts, and retries together. Three retries worsen overload if the original request is still running on the GPU. Pure reads may support deduplication or coalescing; tools with external side effects need their own idempotency and confirmation rules.

## 2 · Candidates: eligibility before ranking {#retrieval}

Keyword, text-vector, and optional image–text retrieval can run in parallel. Each returns IDs, scores, provenance, and versions. Deduplicate, filter, fuse, and rerank under explicit rules. BM25, different encoders' cosine scores, and probabilities are not interchangeable numbers to sum blindly.

Where possible, authorization should constrain candidate eligibility. Post-filtering needs handling when fewer than k candidates remain. Recheck permissions and current revisions when fetching documents; an ACL indexed hours ago is not permanent authority.

| Failure | Possible fallback | Required check |
| --- | --- | --- |
| Vector service times out | Keyword results only | Mark degraded mode rather than pretending fusion completed |
| Reranker times out | Initial ranking | Keep access checks, deduplication, and basic eligibility filters |
| New encoder and index are incompatible | Complete previous bundle | Never mix spaces simply to return something |
| Authorization service is unavailable | Predefined rejection or demonstrably public content | Security is not an optional latency optimization |

## 3 · Generation: execution is not evidence validation {#generation}

For a summary, the application selects authorized evidence, retains citation IDs, and builds the prompt. vLLM / SGLang handle generation scheduling and KV state. Instructions inside retrieved documents remain untrusted; entering a prompt does not grant tool permissions.

<figure class="worked-update" lang="en">
<ol>
<li><small>01 / Context</small><strong>Pass revisions and citations</strong><span>Record selected passages and what the context budget removed, not only the query.</span></li>
<li><small>02 / Generation</small><strong>Trace waiting and output</strong><span>Separate queues, prefill, first token, and decode; propagate cancellation to the engine.</span></li>
<li><small>03 / Response</small><strong>Check evidence and retain a fallback</strong><span>Validate citation existence and support. Without evidence, return sources or explain the limitation.</span></li>
</ol>
<figcaption>Structured output helps validate format. A citations field does not prove support, and generation completion does not finish request acceptance.</figcaption>
</figure>

Streaming also requires a decision about what can be shown immediately and what needs checking first. Filtering a completed string cannot retract content already streamed to the user. The boundary depends on product risk; not every response needs full buffering.

## 4 · Latency: optimize what users wait for {#latency}

For search without generation, imagine one serial request: admission 20 ms, query encoding 30 ms, retrieval 80 ms, reranking 60 ms, response 30 ms. That totals 220 ms, leaving 30 ms in a 250 ms budget. This is one illustrative path. **Adding service p95s does not yield end-to-end p95**, and parallel work must be analyzed on the critical path.

Now take a 500 ms request with 300 ms in model computation and 200 ms elsewhere. Making model computation three times faster gives $200+300/3=300$ ms overall: only $500/300\approx1.67$ times faster end to end.

```python
def end_to_end_speedup(total_ms, accelerated_ms, factor):
    import math
    if not all(math.isfinite(value) for value in (total_ms, accelerated_ms, factor)):
        raise ValueError("expected finite values")
    if total_ms <= 0 or not 0 <= accelerated_ms <= total_ms or factor <= 0:
        raise ValueError("invalid timing or speed factor")
    new_ms = total_ms - accelerated_ms + accelerated_ms / factor
    return total_ms / new_ms


assert abs(end_to_end_speedup(500, 300, 3) - 5 / 3) < 1e-12
```

Load matters too. Near capacity, queueing can become substantial; multiplying a lightly loaded single-request time does not predict tail latency under concurrency. See the [engine comparison protocol](../../06-systems/inference-frameworks.en.md#benchmark).

## 5 · Release a compatible bundle {#release}

Validate artifacts and run fixed-input regressions before shadow traffic or a canary. Shadow traffic helps compare outputs and execution, but shadow results did not receive real user reactions. Do not duplicate side-effecting tool calls. A canary changes some live traffic and needs predefined stop or rollback gates.

Models, processors, prompts, ranking configuration, and indexes need not all change together, but the release manifest must capture compatibility. Roll back a compatible combination, isolate cache versions, and confirm the previous version still has capacity.

A small project can start with old artifacts, a switchable pointer, fixed comparison inputs, and a rehearsed rollback. “Supports rollback” in a README is not a recovery test.

## 6 · Monitoring: not every drop is drift {#monitoring}

| Layer | Signals | Example |
| --- | --- | --- |
| Service health | Errors, queues, deadlines, cancellation cleanup | Idle GPU while tokenization stalls |
| Data health | Missing fields, parser failures, freshness, source mix | New PDF parser drops tables |
| Model quality | Relevance, abstention, support, slices | Stable aggregate score hides worse Chinese long-document results |
| Product outcome | Task completion, feedback, actual usage cost | Longer answers make the relevant detail harder to find |

A distribution change may reflect new users. A metric drop may be broken instrumentation. Check measurement and pipelines before choosing retraining. Delayed labels may require early proxies, but final conclusions need mature labels and stable definitions.

## 7 · Feedback: retain what was actually shown {#feedback}

A click should connect to the request, displayed list and positions, actual model / index revisions, exposure time, and experiment assignment. “User clicked document X” alone cannot reconstruct what else the user could choose or establish whether a change in behavior came from the new version.

Where appropriate, sample actual serving features or revisions sufficient to reconstruct them, then compare training and serving. [Google's Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) discusses training-serving skew and feedback loops. Collection must still be minimized and respect authorization and retention limits.

Do not automatically treat previously displayed, clicked items as the only truth in the next dataset. The old model's omissions can become future blind spots. Use task-appropriate labeling, exploration, or audits; more logs alone do not remove selection bias.

## 8 · Diagnose across boundaries {#incident}

| User symptom | First evidence to inspect | Modules involved |
| --- | --- | --- |
| New documents are missing | Revision chain from ingestion to index | Data, encoding, indexing |
| Relevant sources suddenly rank poorly | Query / document encoders and ranking revisions | Models, search, release |
| Retrieval succeeds but summary is wrong | Actual context, citations, generation configuration | Retrieval, application, model |
| Occasional long waits | Queue and stage timings in the same trace | Application, inference, platform |
| Deleted material reappears | Tombstone, index, cache, authorization records | Data, service, security |

Different teams can own different modules, but a service owner needs to coordinate diagnosis across the request. Useful end-to-end understanding means locating your layer's failures and communicating with upstream and downstream owners using inputs, versions, and traces.

For hands-on work, choose [recommendation](../recommender-systems/README.en.md), [RAG](../rag/README.en.md), or [post-training](../post-training/README.en.md). You do not need to build every path at once.
