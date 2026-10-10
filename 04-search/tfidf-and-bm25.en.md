# TF-IDF and BM25: how keyword retrieval scores a document

[中文](tfidf-and-bm25.md) · **English**

> Reading time: about 12 minutes · Prerequisites: counts, logarithms, weighted sums · Reviewed: 2026-10-10

When you search for an error message, an article on a similar topic may not be enough. An error code, function name, or version number can identify the exact problem. Keyword retrieval preserves these literal clues. But how should matching two different words compare with matching one word twice?

Before the formula, consider two changes. Repeating `cache` 20 times should not make a note 20 times more useful than an explanation of the error. And one mention of `error` in a one-line incident record may provide a different clue from one mention in an entire manual. BM25 accounts for diminishing returns from repetition and differences in document length. **It still does not understand which passage solves the problem.**

For a quick understanding, start with that intuition and the [parameter comparison](#parameter-choice). For the arithmetic, follow the 4 short documents below; for integration, continue to [implementation and debugging](#implementation). This is background for [hybrid retrieval](hybrid-and-reranking.en.md), with no embedding prerequisite.

## 1. Occurrences in one document versus documents containing a term

The query is `cache error`. To make every step checkable, use these already-tokenized documents and ignore order, punctuation, and case processing.

| Document | Tokens | Length | `cache` count | `error` count |
| --- | --- | --- | --- | --- |
| A | `cache cache error` | 3 | 2 | 1 |
| B | `cache error` | 2 | 1 | 1 |
| C | `cache guide setup notes` | 4 | 1 | 0 |
| D | `release notes` | 2 | 0 | 0 |

Term frequency (TF) counts within a document: `cache` appears twice in A. Document frequency (DF) counts documents across the collection: 3 documents contain `cache`. Its DF is 3, **not its 4 total occurrences**.

There are $N=4$ documents, with average length $\overline L=(3+2+4+2)/4=2.75$. All statistics come from the same collection snapshot. Changing the documents or chunking changes these statistics too.

## 2. TF-IDF: give less common terms more weight

TF-IDF multiplies term frequency by inverse document frequency (IDF): one measures repetition within the document, the other measures rarity across the collection. Choose a simple convention with natural logarithms:

$$
\begin{aligned}
\operatorname{idf}(t)&=\ln\frac{N}{\operatorname{df}(t)},\\
w(t,d)&=f(t,d)\operatorname{idf}(t).
\end{aligned}
$$

$f(t,d)$ is the count of term $t$ in document $d$. Sum document weights for distinct query terms, once per term. An absent term contributes zero; query terms outside the collection vocabulary are ignored. TF-IDF names a family of weighting conventions, not a unique scoring function. The [IR textbook definition](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html) gives this product form.

The IDF of `cache` is $\ln(4/3)\approx0.2877$; for `error`, it is $\ln(4/2)\approx0.6931$. A therefore scores $2\times0.2877+0.6931\approx1.2685$, compared with B's $0.9808$.

| Document | `cache` contribution | `error` contribution | TF-IDF sum |
| --- | --- | --- | --- |
| A | 0.5754 | 0.6931 | **1.2685** |
| B | 0.2877 | 0.6931 | 0.9808 |
| C | 0.2877 | 0 | 0.2877 |
| D | 0 | 0 | 0 |

A beats B by repeating `cache`. That is not necessarily wrong, but should 20 repetitions count 20 times as much?

One option replaces a positive frequency with $1+\ln f$, keeping absent terms at zero: [sublinear TF](https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html). Another represents both query and document as weighted vectors and uses cosine normalization. **That is not the sum above**: if both vectors include IDF, each matching coordinate contributes an IDF-squared factor to the dot product, followed by division by vector norms. Compare formulas, not just the name “TF-IDF.”

### Does repeating a document really double its score? {#tf-conventions}

Duplicate `cache error` to get `cache error cache error`, holding IDF fixed. Raw counts change from `[1, 1]` to `[2, 2]`, doubling the weighted sum above. Relative frequencies stay at `[1/2, 1/2]`. With L2-normalized TF-IDF, the vectors also have the same direction, so their cosine scores against a fixed query are unchanged.

“TF-IDF ignores length and rewards repetition without a limit” therefore describes only particular implementations. [Scikit-learn 1.7](https://scikit-learn.org/1.7/modules/feature_extraction.html#tfidf-term-weighting) uses L2 normalization by default, with smoothed IDF $1+\ln((N+1)/(\operatorname{df}+1))$, rather than this section's raw-count sum. Holding collection statistics fixed isolates repetition; adding a document and fitting again can change IDF as well.

## 3. BM25: repetition helps, but not proportionally forever {#bm25-score}

Use this BM25 variant with positive IDF:

$$
\begin{aligned}
r_t&=\frac{N-\operatorname{df}(t)+0.5}{\operatorname{df}(t)+0.5},\\
\operatorname{idf}_{+}(t)&=\ln(1+r_t).
\end{aligned}
$$

$$
\begin{aligned}
a_d&=1-b+bL_d/\overline L,\\
g(t,d)&=\frac{f(t,d)(k_1+1)}{f(t,d)+k_1a_d},\\
S(q,d)&=\sum_{t\in\operatorname{unique}(q)}
\operatorname{idf}_{+}(t)g(t,d).
\end{aligned}
$$

Set $k_1=1.2,b=0.75$. The IDF convention and parameter defaults can be checked against [Lucene 10.3.1](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html). This is a teaching formula, not a bit-for-bit reproduction of Lucene's complete scorer, field statistics, or length encoding.

Here $r_t$ is an intermediate ratio for IDF, $a_d$ adjusts for document length, and $g(t,d)$ is the adjusted frequency factor. Sum the query-term contributions to obtain $S(q,d)$. These intermediate values make the calculation easier to inspect; they are not extra parameters to tune.

There are two separate effects:

- **Term-frequency saturation:** hold the length adjustment fixed. Each additional repetition adds less than the previous one.
- **Length normalization:** at the same term frequency, a longer document is generally discounted. One occurrence in a 10-word record and one in a long survey need not carry the same evidence.

Isolate frequency by fixing $L_d/\overline L=1$:

| Frequency $f$ | Raw TF | $f(k_1+1)/(f+k_1)$ with $k_1=1.2$ |
| --- | --- | --- |
| 1 | 1 | 1.0000 |
| 2 | 2 | 1.3750 |
| 4 | 4 | 1.6923 |
| 8 | 8 | 1.9130 |

With length fixed, this factor approaches $k_1+1=2.2$. This is a controlled comparison, not a claim that appending words leaves length unchanged. Actually adding repetitions changes both frequency and length.

Back to our documents: the new IDFs are $0.3567$ for `cache` and $0.6931$ for `error`. A's length adjustment is $1-0.75+0.75\times3/2.75\approx1.0682$. First compute its frequency factor $g(\texttt{cache},A)$, then multiply by IDF:

$$
\begin{aligned}
\frac{2\times2.2}{2+1.2\times1.0682}&\approx1.3408,\\
0.3567\times1.3408&\approx0.4782.
\end{aligned}
$$

| Document | `cache` contribution | `error` contribution | BM25 sum |
| --- | --- | --- | --- |
| A | 0.4782 | 0.6683 | 1.1465 |
| B | 0.4015 | 0.7802 | **1.1817** |
| C | 0.3008 | 0 | 0.3008 |
| D | 0 | 0 | 0 |

B now narrowly beats A: both cover the query, B is shorter, and A's extra occurrence helps less. **This is a formula's ranking, not a human relevance label.** One document might only repeat an error, while another explains the fix. Usefulness needs separate evidence.

<span id="4-parameters-and-conventions-worth-separating"></span>

## 4. Parameters and conventions worth separating {#parameter-choice}

| Change | Direct effect | Caveat |
| --- | --- | --- |
| $k_1=0$ | Each matched term contributes its IDF, independent of frequency | Return zero for absent terms rather than computing $0/0$ |
| Increase $k_1$ | At a fixed length ratio, allow larger differences from repetition | Larger is not automatically better; tune on development data |
| $b=0$ | Disable length normalization | Saturation remains; this is not raw TF |
| $b=1$ | The length adjustment becomes $L_d/\overline L$ | Longer documents do not always lose: counts and matched terms also matter |

Some BM25 definitions omit the outer `1 +` in IDF. Under that convention, terms appearing in more than half the documents can receive negative IDF; `cache` would here. A difference in formula is not necessarily a bug. The [IR textbook's BM25 section](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) discusses these forms.

BM25 scores are not relevance probabilities, and should not be blindly added to embedding cosine scores. [RRF](hybrid-and-reranking.en.md#rank-fusion) is one way to combine rankings without requiring comparable raw scores.

### A higher score or just the first item in a tie? {#tied-scores}

Consider a different collection: 6 documents, exactly 3 containing `cache`. Under the IDF convention without the outer `1 +`:

$$
\operatorname{idf}(\texttt{cache})=\ln\frac{6-3+0.5}{3+0.5}=0.
$$

For a query containing only `cache`, repetitions contribute nothing. An engine may break ties by document ID or input order, but **the first result did not earn a higher score**. Our positive-IDF convention gives $\ln 2$ instead, allowing frequency and length to affect this term's score.

[BM25Okapi in rank_bm25 0.2.2](https://github.com/dorianbrown/rank_bm25/blob/0.2.2/rank_bm25.py#L79-L113) illustrates why conventions matter: it starts without the outer `1 +`, then replaces negative IDFs; a zero IDF is not made positive by that step. Inspect per-term contributions and ties before explaining a ranking. An experiment with $b=0$ also cannot demonstrate length normalization: it has switched that factor off.

## 5. Implement the calculation before optimizing retrieval {#implementation}

A term's contribution is short. The outer `LexicalIndex` validates parameters and computes statistics; this helper assumes valid inputs and positive average length for terms that occur in the collection.

```python
def bm25_term(frequency, length, average_length, inverse_frequency, k1, length_weight):
    if frequency == 0:
        return 0.0
    length_factor = 1 - length_weight + length_weight * length / average_length
    return inverse_frequency * frequency * (k1 + 1) / (frequency + k1 * length_factor)
```

The code names the formula’s $b$ parameter `length_weight`. The complete standard-library implementation is [lexical_retrieval.py](code/lexical_retrieval.py). From the repository root, run:

```bash
python3 04-search/code/lexical_retrieval.py
```

```text
TF-IDF: [1.2685, 0.9808, 0.2877, 0.0]
BM25: [1.1465, 1.1817, 0.3008, 0.0]
```

Inputs are **sequences of already-tokenized strings**. The implementation does not tokenize or lowercase. The example's `.split()` only reproduces this English word list; it is not a Chinese search analyzer. Query terms are deduplicated. Empty documents score zero but count toward document count and average length. An empty collection returns an empty list; an all-empty collection does not divide by zero. Production engines may treat missing and empty fields differently, so check their statistics contract.

Building statistics scans all $T$ tokens in expected $O(T)$ time. A query checks $Q$ distinct query terms against each of $N$ documents in expected $O(NQ)$ time, assuming expected constant-time hash operations. Frequency storage scales with the total number of distinct document–term pairs. This is a transparent reference implementation, **not a large-scale search engine**.

A production query generally reads candidates from an inverted index: `cache → A:2, B:1, C:1`, `error → A:1, B:1`. D need not be scanned for these matches. Frequent terms can still have long postings lists. Top-K selection, filtering, pruning, and updates all have costs; inverted indexing does not make every query $O(1)$.

## 6. What to check on real data

| Symptom | First check | Small diagnostic |
| --- | --- | --- |
| An exact error exists but is not retrieved | Query and document analyzer compatibility | Print actual tokens for `ERR_42`, `C++`, and Chinese text |
| Scores change after chunking | Document count, DF, and average length may all change | Hold the query fixed and compare term contributions |
| Long tutorials consistently lose to short notes | Excessive length penalty or navigation boilerplate | Evaluate length slices and sweep $b$, not just the overall mean |
| Terms match, but the answer is wrong | Matching is not authority, freshness, or answerability | Inspect versions, access controls, evidence spans, then reranking / RAG |
| Paraphrasing produces no results | Too little lexical overlap | Add a dense route and compare unique relevant results at fixed total candidate count |

A rare word may be a typo, not a useful signal. Stop words, synonyms, and field weights depend on the task. Splitting or normalizing an error-code field indiscriminately can destroy its most informative part.

Keep fixed queries and relevance judgments for evaluation. Inspect exact identifiers, paraphrases, Chinese queries, and long-document slices. Select $k_1,b$ on development data and reserve held-out judgments for final comparison. The retrieval collection may include documents to be searched under the task protocol; repeatedly tuning with test relevance labels is a different issue.

## 7. Where to go next

- Different words with similar meaning: [dual encoders](dual-encoder.en.md).
- Combining sources: [hybrid retrieval and reranking](hybrid-and-reranking.en.md).
- Turning retrieval into a checkable answer: [RAG evidence chains](rag-evidence.en.md).
- Check your understanding: change the query to `cache cache error unknown`. This implementation should return the same scores as `cache error`. Set $b=0$ and explain the ranking change, then set $k_1=0$ and confirm that A and B tie.

These are teaching documents, not product results or model benchmarks. Beyond the formulas, check statistical and engineering conventions against your actual engine version.
