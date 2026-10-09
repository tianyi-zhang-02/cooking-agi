# From documents to usable evidence

[中文](data-and-retrieval.md) · **English**

> Original teaching example · Reviewed: 2026-10. Documents, rankings, and budgets are synthetic, not service measurements.

“Friday or Wednesday?” is first a data-consistency problem, not a difficult language question. Leave the LLM out for a moment and follow a paragraph into retrieval.

## Keep a path back to the source

A chunk needs a document, revision, and source location. Otherwise, editing or deleting a document leaves no reliable way to identify its obsolete fragments.

| Field | Example | Purpose |
| --- | --- | --- |
| `document_id` / `revision` | `rules` / `2` | Check whether it is current |
| `chunk_id` | `rules-v2-deadline` | Deduplication, citations, regression diagnosis |
| `section` / source location | Registration / paragraph 2 | Verify against the original |
| `visibility` | `members` | Restrict candidates using server-side identity |
| Effective time / deletion state | Effective after publication | Distinguish future, current, and deleted content |

The code simplifies this to integer revisions and visible groups. A real system also handles effective intervals, identity changes, and conservative failure handling for authorization. Access control is not a prompt instruction: enforce it **before evidence reaches rerankers, LLMs, responses, or ordinary logs**. Shared cache keys must include authorization scope and version, not just the question.

## Keep rules and exceptions together

“Registration closes Wednesday” is easy to retrieve alone. But separating it from “for in-person events only” changes its meaning. Start with paragraph or heading boundaries, preserving nearby qualifications where necessary, rather than blindly cutting by character count.

Haystack's [DocumentSplitter](https://docs.haystack.deepset.ai/docs/documentsplitter) preserves source information and supports length and overlap settings. A splitter implements boundaries; your documents determine which qualifications belong together.

Compare two chunking schemes on the same questions: is the supporting passage complete, how much duplicated context is retrieved, and how many tokens are used? Ten relevant-looking chunks from one paragraph may be less useful than two covering a rule and its exception.

## Restrict visibility before selecting top-k

Suppose the top keyword scores are private budget: 9, obsolete policy: 8, current policy: 7. Selecting top-2 and then filtering leaves nothing. That does not mean there is no valid answer.

The teaching implementation filters current revisions and access before ranking. Verify whether the actual index supports pre-filtering; if it only post-filters, measure candidate loss and refill cost. **Never relax authorization to fill k slots.** Recheck current authorization before returning evidence to handle revocation or deletion during a request.

## How do we combine lexical and vector results?

A lexical score of 12 is not necessarily better than cosine similarity 0.8. Reciprocal rank fusion (RRF) offers an explainable baseline:

$$
\operatorname{RRF}(d)=\sum_{r:\,d\in L_r}\frac{1}{c+\operatorname{rank}_{L_r}(d)}.
$$

Ranks start at 1 and `c > 0`. We use `c=10` below to make the arithmetic easy, not as a production recommendation. RRF avoids directly calibrating raw scores, but its candidate windows and constant still need evaluation. [Elasticsearch's RRF definition](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

| Passage | Lexical rank | Vector rank | RRF score |
| --- | --- | --- | --- |
| A | 1 | 2 | `1/11 + 1/12 ≈ 0.1742` |
| B | 2 | Absent | `1/12 ≈ 0.0833` |
| C | Absent | 1 | `1/11 ≈ 0.0909` |

The order is A, C, B. A ranks near the top in both lists and wins the fusion. C appears only in the vector list, but ranks higher there than B does in the lexical list.

Deduplicate IDs within each list before scoring. Appearing in both lists doesn't prove relevance either: RRF combines rankings; relevance still needs evaluation.

## What if the evidence doesn't fit?

Suppose 10 tokens remain and three passages, in priority order, have lengths 6, 6, and 4. Try a simple rule: accept a passage if it fits, otherwise skip it and continue.

| Passage | Length | Space remaining | Decision |
| --- | --- | --- | --- |
| First | 6 | 10 → 4 | Keep |
| Second | 6 | 4 → 4 | Skip: too long |
| Third | 4 | 4 → 0 | Keep |

The first and third passages survive. This greedy method is easy to implement, but need not choose the best evidence: those two passages may repeat each other, and the second may contain a crucial exception. Check retained support as well as the token limit.

A real budget subtracts instructions, the question, citation wrappers, and reserved output, using **the generator's tokenizer**. The example counts whitespace-separated words only to expose packing behavior. It cannot estimate Chinese token usage or enforce an actual model context limit.

## Run it, then break one assumption

From the repository root:

```bash
python3 practice/rag/code/evidence_pipeline.py
python3 -m unittest discover -s site/tests -p 'test_practice_projects.py'
```

The [complete code](code/evidence_pipeline.py) returns `rules-v2` as current visible evidence, merges two illustrative lists into `A, C, B`, packs the first and third passages, and flags an unprovided citation, `rules-v1`.

Remove `rules` from `current_revisions`: without a confirmed current revision, that document is excluded. Then paraphrase the query so it shares no words with the passage and inspect what keyword retrieval misses. Those cases can later test whether embeddings recover the missing evidence.

The code has no ANN index, authorization service, LLM, or automatic synchronization. It checks filtering, fusion, and packing contracts. Service integration still needs timeouts, stale-index handling, edit/delete events, and authorization rechecks.

[Next: A citation isn't proof](evidence-and-evaluation.en.md)
