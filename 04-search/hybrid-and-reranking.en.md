# Hybrid retrieval and reranking: does another source actually help?

**English** · [中文](hybrid-and-reranking.md)

A user searches for “attention mask shape mismatch.” Dense retrieval finds a good introduction to attention. Keyword search finds a debugging note containing the exact error and tensor shapes. Both are reasonable matches, but being on topic is not the same as solving the error.

Following the [dual-encoder note](dual-encoder.en.md), we now combine sources and decide what deserves a closer read. Start with the document example for the design intuition, or jump to [rank fusion](#rank-fusion) for an implementation.

## 1. What does each source miss?

Use this four-document teaching collection:

| Document | Content | Keyword rank | Dense rank |
| --- | --- | --- | --- |
| A | An old-version debugging note with the exact error string | 1 | Outside top 3 |
| B | Mask broadcasting rules and a fix for the current version | 2 | 1 |
| C | A Transformer overview without this particular error | 3 | 2 |
| D | The same dimensionality issue explained with different wording | Outside top 3 | 3 |

Keyword methods such as BM25 score term matches; exact identifiers, error codes, and rare names can matter. Dense retrieval can connect different wording, but may prioritize the topic over detailed constraints. Neither source automatically captures the entire need.

For the keyword score itself, [TF-IDF and BM25](tfidf-and-bm25.en.md) separates frequency, rarity, and length effects.

Inspect unique relevant results before combining sources. If both always return the same content, fusion may add little. If the second source only adds noise, it can make downstream processing more expensive without helping.

## 2. Why not just add the scores? {#rank-fusion}

A BM25 score of 12.4 and a cosine similarity of 0.81 are not on the same scale. Raw addition can let the numerically larger source dominate. Normalization is an option, but its calibration still needs checking as queries change.

Reciprocal Rank Fusion (RRF) is a simple starting point: use ranks instead of requiring comparable raw scores.

$$
\mathrm{RRF}(d)=\sum_{r:d\in L_r}\frac{1}{k+\mathrm{rank}_r(d)}.
$$

Ranks start at 1. $L_r$ is the list actually returned by source $r$; a missing document contributes zero for that source. The constant $k$ smooths rank differences and is distinct from the final top-$K$ output size. See the formula and implementation conventions in [Elasticsearch’s RRF documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion).

With a teaching constant $k=10$, B scores $1/12+1/11\approx0.1742$, C scores $1/13+1/12\approx0.1603$, A scores $1/11\approx0.0909$, and D scores $1/13\approx0.0769$. B and C receive support from both lists. Being first in keyword search does not guarantee that A wins fusion.

```python
def reciprocal_rank_fusion(rankings, constant=10):
    if constant < 0:
        raise ValueError("The constant must be non-negative")
    scores = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("Deduplicate each source before fusion")
        for position, document in enumerate(ranking, start=1):
            scores[document] = scores.get(document, 0) + 1 / (constant + position)
    return sorted(scores, key=lambda document: (-scores[document], document))

assert reciprocal_rank_fusion([["A", "B", "C"], ["B", "C", "D"]]) == ["B", "C", "A", "D"]
```

The code breaks ties by document ID for reproducibility. A real system should specify its own tie convention.

## 3. What changes when a source returns fewer results?

Keep the same four documents and change only the returned lists. The constant stays at 10; no hidden full-collection ranking is used:

| Inputs | Fused result | What changed? |
| --- | --- | --- |
| Top 3 from both sources | B → C → A → D | B and C each receive two contributions |
| Keyword top 3 only | A → B → C | D cannot enter the result |
| Top 1 from both sources | A and B tie | B loses its second-place keyword contribution |

Disable dense retrieval: D has no way into the candidate set. Reduce both lists to top 1: B loses its second-place contribution. **Fusion can use only the results it receives.** A larger smoothing constant cannot recover content that was never retrieved.

RRF discards raw score gaps, so it cannot distinguish a narrow first-place win from an overwhelming one. Duplicate entries within a source must not earn repeated credit. It is a transparent baseline, not a universally optimal fusion method.

## 4. What does reranking add?

After fusion, we still need to determine whether A is outdated and B applies to the current environment. A cross-encoder jointly reads the query and candidate for finer comparisons. An LLM judge with clear criteria is another option, but cost, consistency, and resistance to instructions inside retrieved content need testing.

```text
Documents the user may access
    ├─ Keyword results ─┐
    └─ Dense results ───┴→ Deduplicate and fuse → Fixed pool → Rerank → Evidence context
```

Enforce access controls before sending content to a model or service that is not allowed to see it. “Do not mention it in the answer” is not access control. If the index cannot filter, perform checks in a trusted retrieval layer and account for the smaller pool after filtering.

A reranker can reorder only the candidates it receives. If the correct document was never indexed or retrieved, a more expensive reranker cannot recover it. This concerns pure reranking; an agent allowed to search again is a different design with a separate budget.

## 5. Compare under the same budget

Comparing 100 dense candidates against 100 dense plus 100 keyword candidates changes both the source and the downstream computation. Do not attribute the entire gain to the fusion algorithm.

| Comparison | Hold fixed | What does it test? |
| --- | --- | --- |
| Dense-only vs hybrid | Final pool size, corpus snapshot, evaluation queries | Does the added source recover unique relevant content? |
| RRF vs learned fusion | The same source results | Does the fusion rule improve ordering? |
| No reranking vs reranking | Candidate set and labels | Can reranking promote good content already present? |
| Different budgets | Record per-stage depth, latency, and cost | Is the extra computation worthwhile? |

Suppose there are three known relevant documents. One source finds $\{B\}$, the other $\{B,D\}$. Their union finds two, not three. Duplication and unique positives are useful measures, provided incomplete judgments remain explicit: unlabeled is not automatically irrelevant.

## 6. Do not fill the context with near-duplicates

The final context is another budget. Five similar explanations may be less useful than one explanation and one version-specific note. Preserve sources, dates, and passage boundaries so the answer’s evidence can be inspected.

Longer context does not guarantee more usable evidence. [Lost in the Middle](https://arxiv.org/abs/2307.03172) found sensitivity to relevant information’s position in its tested settings. That does not establish identical behavior for every model, but it is a reason to measure grounded answers rather than stopping at retrieval scores.

Continue with the [evaluation stack](../07-evaluation/evaluation-stack.en.md): separate missing evidence from evidence that was retrieved but used incorrectly before choosing the next fix.
