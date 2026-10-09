# Too many vectors: searching faster without losing too much

[中文](vector-indexes.md) · **English**

> Reading time: ~10 min · Last reviewed: 2026-10

The dual encoder has turned queries and documents into vectors. The collection now contains 10 million vectors with 768 dimensions each. Must every query compare against all of them?

It can: that is exact search. The questions are whether it fits in memory and finishes within the latency budget. Approximate nearest-neighbor search (ANN) accepts some missed neighbors in exchange for speed or storage savings. Here, “missed” means neighbors under the same scoring function, **not every genuinely relevant answer**. Exact search cannot repair a poor representation.

The [dual-encoder note](dual-encoder.en.md) provides the background. Numbers below are teaching examples, not product benchmarks.

## 1. Start with memory and a consistent comparison

Storing the float32 vectors alone takes:

$$
10^7\times768\times4=30{,}720{,}000{,}000\ \text{bytes}\approx28.61\ \text{GiB}.
$$

That excludes IDs, index structures, metadata, replicas, and query buffers. A “48× compression” claim about vector codes does not mean the whole service uses 48× less memory.

Before comparing indexes, fix the embedding version, collection, filters, and metric. Cosine similarity is not an ordinary dot product. For nonzero vectors normalized to unit length:

$$
\|\hat q-\hat x\|_2^2=2-2\hat q^\top\hat x.
$$

Maximum inner product and minimum squared L2 then give the same ranking. Handle zero vectors separately. If vector norms carry information in the training objective, do not silently remove it to use cosine search. [Faiss metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

## 2. IVF: choose regions before searching their contents

IVF assigns database vectors to clusters, each with its own list. A query selects nearby clusters and scans their lists. `nlist` counts all clusters; `nprobe` controls how many are searched. [Faiss index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

Suppose one million vectors occupy 1,000 equally sized clusters. Probing ten clusters scans about 10,000 vectors rather than a million. Finding the clusters and maintaining top-k still cost time. Real lists are uneven, so `nprobe / nlist` is only a rough scan fraction, not a speedup prediction.

```text
The same query q
  ├─ nprobe = 1 → scan nearest cluster A
  └─ nprobe = 2 → scan clusters A and B
                              ↑
                   the nearest vector may sit at B's edge
```

The closest centroid need not own the closest point. Probing more clusters reduces this kind of miss at a cost. IVF-Flat still stores full vectors: it reduces the search scope, not the vector payload.

## 3. PQ: store IDs, but do not measure distances between IDs

Product quantization (PQ) splits a vector into parts and represents each part by an entry in its own learned codebook. Split 768 dimensions into 64 parts with 256 entries per codebook: each part needs an 8-bit ID, making a 64-byte vector code. The original float32 vector needs 3,072 bytes, so the **code payload alone** is 48× smaller. Codebooks are fitted to training samples; splitting coordinates alone does not train a quantizer.

ADC keeps the query unquantized and estimates distance to the reconstruction of a stored code. Squared L2 distances add across parts. SDC quantizes the query as well. The distinction is which sides are quantized, not whether a lookup table is used. [PQ paper, Sections II–III](https://paper-notes.zhjwpku.com/assets/pdfs/Product_Quantization_for_Nearest_Neighbor_Search.pdf)

Shrink the example to four dimensions. The query is `[1, 2 | 3, 4]`; each subspace has three codewords. This code demonstrates ADC only: it neither trains codebooks nor chooses the nearest encoding. The illustrative IDs are not bit-packed.

```python
def adc_squared_distance(query_parts, codebooks, codes):
    if not (len(query_parts) == len(codebooks) == len(codes)):
        raise ValueError("one query part and code per codebook required")
    total = 0.0
    for query_part, codebook, code in zip(query_parts, codebooks, codes):
        if not isinstance(code, int) or not 0 <= code < len(codebook):
            raise ValueError("invalid code")
        center = codebook[code]
        if len(query_part) != len(center):
            raise ValueError("dimension mismatch")
        total += sum((value - centroid) ** 2
                     for value, centroid in zip(query_part, center))
    return total


query_parts = [[1.0, 2.0], [3.0, 4.0]]
codebooks = [
    [[0.0, 0.0], [1.0, 1.0], [4.0, 4.0]],
    [[0.0, 0.0], [3.0, 3.0], [5.0, 5.0]],
]
assert adc_squared_distance(query_parts, codebooks, [1, 1]) == 2.0
assert adc_squared_distance(query_parts, codebooks, [2, 1]) == 14.0
```

Code `[1, 1]` reconstructs `[1, 1, 3, 3]`, giving `0² + 1² + 0² + 1² = 2`. A real scan builds a query-specific table of distances to each subspace's codewords, then sums the entries selected by each stored code.

**The IDs themselves carry no geometry.** Relabel a codebook and update the stored IDs consistently: reconstructions and ADC distances stay unchanged, while Hamming distances between the binary IDs may change. [Polysemous Codes](https://arxiv.org/abs/1609.01882) specifically learns a code assignment suitable for Hamming prefiltering. Ordinary PQ IDs do not acquire that property automatically.

## 4. IVF-PQ introduces two sources of approximation

IVF narrows the regions searched; PQ compresses stored vectors. A common IVFADC construction encodes each vector's **residual** relative to its coarse centroid, rather than simply chaining two unrelated steps. [PQ paper, Section IV](https://paper-notes.zhjwpku.com/assets/pdfs/Product_Quantization_for_Nearest_Neighbor_Search.pdf)

| Failure | Concrete example | How to isolate it |
| --- | --- | --- |
| Not scanned | A useful document belongs to an unprobed cluster | Hold the codes fixed and increase `nprobe` |
| Distance misestimated | Approximate distances reverse two candidates | Fix the scanned set and compare full-vector scores |
| Still irrelevant | Even exact neighbors are on-topic but unhelpful | Inspect relevance labels and embeddings, not just the index |

Retrieve a larger approximate shortlist and rescore its original vectors to repair ordering **within that shortlist**. This cannot recover a vector that never entered it. The originals also need storage and incur read costs. Exact-distance refinement is not the same as a [cross-encoder reading the query and document jointly](hybrid-and-reranking.en.md).

## 5. Choosing between Flat, HNSW, and IVF-PQ

[HNSW](https://arxiv.org/abs/1603.09320) searches a layered neighbor graph, navigating sparse upper layers before exploring more densely below. It does not require PQ compression, and its graph has its own memory cost.

| Option | Why try it first? | What still needs checking? |
| --- | --- | --- |
| Flat | A modest collection, efficient batching, or an exact baseline | Full-scan latency, throughput, and memory; exact does not always mean slowest in practice |
| HNSW | Low-latency in-memory search | Graph memory, construction, and search width; deletion behavior depends on the implementation |
| IVF-Flat | Isolate the effect of searching fewer regions | List imbalance, `nprobe`, and tail latency |
| IVF-PQ | Vector storage is the main constraint | Code length, quantization error, representative training samples, and refinement costs |

These are starting points for experiments, not mandatory transitions at a particular collection size. Hardware, concurrency, filtering, and update frequency can change the answer.

## 6. ANN recall is not task relevance recall

Let A be the exact top ten and B the approximate top ten. `|A ∩ B| / 10` measures retained exact neighbors. Relevance Recall@10 instead uses labeled relevant documents. A high value for the former shows that the approximation works, not that the representation does.

Fix a query set and collection snapshot, sweep index parameters, and record:

- ANN recall and labeled relevance metrics;
- p50 / p95 / p99 latency, throughput, peak memory, and build time;
- result counts after strict filters, plus rare-topic and short-query performance;
- consistency between the index and metadata after embedding updates, document additions, and deletions.

Access controls also change the eligible collection. Do not send unauthorized text to the model and rely on a prompt to keep it secret. Next: [preserving evidence and evaluating RAG answers](rag-evidence.en.md).
