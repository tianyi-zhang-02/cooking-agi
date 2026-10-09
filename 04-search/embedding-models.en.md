# Choosing between Qwen3 Embedding and BGE-M3

[中文](embedding-models.md) · **English**

> Reading time: about 15 minutes · Prerequisites: vectors and dot products; attention masks in the second half · Reviewed: 2026-10-09

A user searches for “I forgot my password. How do I recover it?” Two passages say:

- A: “Choose Reset password on the login page, verify your identity, then set a new password.”
- B: “Enter your password before changing your profile picture.”

Both mention a password, but only A answers the question. Embedding retrieval aims to learn that distinction: **does this passage meet the query's need, rather than merely contain its words?** This is a relevance judgment we want to test, not a claim about measured model output.

For the main differences, sections 1–3 are enough. For integration, continue with [inputs and indexes](#input-contract); to check the arithmetic, jump to the [small lab](#small-lab). You do not need to learn every model family first.

## 1. The shared idea: encode documents ahead of time

On their dense paths, both models can represent a passage with one vector. Encode and index documents first; when a query arrives, encode it and search for nearby document vectors. This is the [dual-encoder design](dual-encoder.en.md) from the previous note. The two paths can share one set of model weights.

```text
Offline: A, password-reset steps → document vector A ┐
         B, profile-picture rules → document vector B ┴→ document index

Request: forgotten-password question → query vector → search → A, B → rerank
```

Changing the embedding model changes how text becomes vectors. It does not replace retrieval with chat generation. An embedding model is also distinct from its family's reranker, which typically inspects query and candidate together to refine a smaller pool.

## 2. Compare specific checkpoints

This note compares `Qwen/Qwen3-Embedding-0.6B` with `BAAI/bge-m3`, not every Qwen and BGE model. BGE v1.5 and BGE-M3, for example, have different length and input conventions.

| Check | Qwen3-Embedding-0.6B | BGE-M3 |
| --- | --- | --- |
| Backbone | Qwen3 decoder, causal attention | XLM-RoBERTa encoder, bidirectional attention |
| Dense pooling | Final hidden state at the last valid token | Final hidden state at the initial CLS marker |
| Full dense dimension | 1024; MRL dimension reduction supported | 1024; do not assume arbitrary truncation is supported |
| Published input limit | 32K tokens | 8192 tokens |
| Retrieval input convention | Task instruction on queries, not documents | Official M3 usage does not require query instructions |
| Outputs compared here | One dense vector | Dense; sparse and multi-vector paths can be enabled separately |

Check the architecture and dimensions in the pinned [Qwen configuration](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/b92a3823bbfb19966c5779407712d8fcfb5f8a60/config.json) and [BGE-M3 configuration](https://huggingface.co/BAAI/bge-m3/blob/84790c1a606f60d06c6932e4ecdd174b466d84ac/config.json). Input conventions and usage come from the [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3). These are published capabilities of specific models, not measurements from this site.

Why the end for one and the beginning for the other? Under causal attention, the final position can use preceding content; the first cannot see the rest of the sentence. A bidirectional encoder's CLS can aggregate later content. **Pooling turns a sequence of token states into one vector and follows the model's training convention**, not whichever position is convenient. Architecture labels alone do not determine retrieval quality.

A task instruction for the password query might specify retrieving help passages that directly answer the question. It guides the representation; it does not ask the model to generate the answer first. Both models depend on representation-learning training. Extracting a token from an ordinary chat model is not the same retriever.

### Why these two designs? {#paper-context}

BGE-M3's 2024 motivation was broader than one benchmark score: languages, input lengths, and retrieval modes often needed separate adaptations. Its unified model addresses those needs together, which helps explain the three outputs. [Original paper §1, §3](https://arxiv.org/html/2402.03216v3) · [Official code](https://github.com/FlagOpen/FlagEmbedding)

The 2025 Qwen3 Embedding report uses LLMs as both representation backbones and synthetic-data sources, followed by weak supervision, high-quality fine-tuning, and checkpoint merging. This is not merely an encoder-to-decoder swap; data and training recipes matter. [Original paper §3.2–3.3](https://arxiv.org/html/2506.05176v1#S3.SS2) · [Official code](https://github.com/QwenLM/Qwen3-Embedding)

That background explains design choices, not why a newer model must suit your project better. Cross-paper scores also mix data, scale, output modes, and evaluation settings; they do not isolate architecture. Use the matched experiments in section 7 for an actual selection.

## 3. Three M3 outputs are not three free retrieval systems

The password example involves several possible signals: the passage's overall meaning, an exact error identifier, and whether different parts of the request have matching evidence.

| Path | What it retains | What you need |
| --- | --- | --- |
| Dense | One vector per passage | A vector index; fits a conventional ANN system |
| Sparse | Learned weights for tokens in the text | A sparse index; not just zeroing small dense coordinates |
| Multi-vector | Multiple token vectors | More storage and matching work, or scoring only existing candidates |

[BGE-M3 §3.2](https://arxiv.org/html/2402.03216v3#S3.SS2) describes these paths. Sparse weights are learned, rather than computed with BM25's frequency formula. Multi-vector scoring uses late interaction: find each query token's best document-token match, then aggregate. This retains finer matching information but is not joint query–document encoding by a cross-encoder.

Sharing an encoder pass does not eliminate output storage, transfer, or search costs. Calling the dense interface does not automatically build the other indexes or provide three-way fusion.

The first-pass takeaway is simple: **both offer single-vector retrieval; M3 also exposes other matching paths, while this Qwen model supports task instructions and configurable dimensions. Choose for your queries, documents, and resources—not the newer name.** The remaining sections turn those differences into implementation checks.

## 4. Keep inputs, pooling, and indexes compatible {#input-contract}

The easy detail to miss is what actually enters the encoder. Qwen's query format contains `Instruct:` and `Query:`; its retrieval example leaves documents unprompted. BGE-M3 does not use that same convention. Mixing one model's prompt with another's pooling or padding can produce valid tensor shapes but invalid comparisons.

### Padding: the final position may not contain text

Each valid position below already includes the tokenizer's special-token handling. These are positions, not a proposed word-to-token mapping.

| Padded sequence | Attention mask | Position to pool, zero-based |
| --- | --- | --- |
| `[valid1, valid2, valid3, PAD, PAD]` | `[1, 1, 1, 0, 0]` | 2 |
| `[PAD, PAD, valid1, valid2, valid3]` | `[0, 0, 1, 1, 1]` | 4 |

Both have three valid tokens, so `mask.sum() - 1` returns 2 for both: it only works for the right-padded row here. `hidden[-1]` only works for the left-padded row. A general check finds the last 1 in the mask. Reject an all-padding sequence rather than indexing a meaningless vector.

The [Qwen paper](https://arxiv.org/html/2506.05176v1#S2) describes pooling at a terminal EOS. Integration must also check the actual tokenizer, special tokens, and wrapper behavior. The lab below verifies selection of the final valid position; it does not validate real tokenization or recommend manually appending EOS.

### Why can't two 1024-dimensional models share an index?

Consider two dimensions. An old query is $(1,0)$, relevant document A is $(1,0)$, and document B is $(0,1)$. Their dot products are 1 and 0.

Suppose a new model merely rotates every vector by 90 degrees. Its own retrieval is unchanged: new query $(0,1)$ scores 1 against new A $(0,1)$ and 0 against new B $(-1,0)$. But deploying only the new query encoder against the old index produces **0 for A and 1 for B**.

Real model changes are not just rotations. The example isolates one point: **equal dimensions do not imply compatible coordinates.** Re-encode documents and rebuild the index for the new model. Mixing versions requires a trained and validated compatibility method, not just a shape check.

Keep a comparable configuration record beside the index:

```text
Model and tokenizer revisions
Query/document templates and special-token handling
Pooling, output dimension, normalization, distance function
Chunking, truncation, document snapshot
Vector storage precision, index parameters, encoding-library version
```

Fine-tuning, chunking changes, and new document templates can also require re-encoding. Switch the query service and index as a pair, retaining the previous pair for rollback.

## 5. What does a shorter vector save? {#dimension-budget}

[MRL, or Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147), trains for useful prefix representations. A supporting model can use a specified prefix of its vector followed by renormalization. Arbitrarily dropping coordinates from any embedding does not offer the same guarantee.

A normalized vector $(0.6,0,0.8)$ becomes $(0.6,0)$ after retaining two coordinates. Its norm is now 0.6, not 1; renormalizing gives $(1,0)$. Without that step, a dot product is no longer cosine similarity between the retained representations. A zero prefix requires an explicit error or agreed handling, not division by zero.

For one million passages, counting only contiguous vector-array data:

| Storage | Bytes | Approximate GiB |
| --- | --- | --- |
| 1024 dimensions, FP32 | 4,096,000,000 | 3.815 |
| 1024 dimensions, FP16 | 2,048,000,000 | 1.907 |
| 256 dimensions, FP16 | 512,000,000 | 0.477 |

Here $1\ \mathrm{GiB}=2^{30}$ bytes. ANN graphs, IDs, metadata, replicas, and working memory are excluded. The last two rows differ by 4× in **raw vector storage**, not total system cost. Truncating an output usually does not reduce the encoder's forward-pass work by 4× either.

Evaluate whether reduced dimensions lose distinctions the task needs. Measure exact vector rankings before ANN rankings to separate representation loss from approximate-index misses. [Vector indexes](vector-indexes.en.md) continues the storage and search tradeoffs.

## 6. Small checks without downloading a model {#small-lab}

This standard-library lab checks pooling, normalization, mixed coordinate systems, and late-interaction arithmetic. It is not a model-quality evaluation, training run, or GPU benchmark.

```python
from embedding_contracts import last_valid_pool, normalized_prefix, mean_maxsim

states = [[9, 9], [9, 9], [1, 0], [2, 0], [3, 4]]
pooled = last_valid_pool(states, [0, 0, 1, 1, 1])
assert pooled == [3.0, 4.0]
assert normalized_prefix(pooled) == [0.6, 0.8]
assert normalized_prefix([0.6, 0, 0.8], 2) == [1.0, 0.0]

score = mean_maxsim([[1, 0], [0, 1]], [[0.8, 0.6], [0, 1]])
assert abs(score - 0.9) < 1e-12
```

Where does the last result come from? The two query vectors and two document vectors produce:

| Similarity | Document token 1 | Document token 2 | Row maximum |
| --- | --- | --- | --- |
| Query token 1 | 0.8 | 0 | 0.8 |
| Query token 2 | 0.6 | 1 | 1 |

Average the row maxima: $(0.8+1)/2=0.9$. The operation does not require matching word order or a distinct document-token match for every query token. Negation, ordering, and cross-passage relationships can still be missed.

The complete implementation is [embedding_contracts.py](code/embedding_contracts.py). The import above requires `04-search/code` on the Python path. From the repository root, run the complete example and tests with:

```bash
python3 04-search/code/embedding_contracts.py
python3 -m unittest discover -s site/tests -p 'test_embedding_contracts.py'
```

```text
Last valid: [3.0, 4.0]
Mixed spaces: [0.0, 1.0]; rebuilt: [1.0, 0.0]
Mean MaxSim: 0.9
```

This code demonstrates contracts. Actual BGE-M3 token projections, excluded special tokens, and masks still require the matching implementation. `mean_maxsim` takes $O(QDd)$ time for $Q$ query tokens, $D$ document tokens, and dimension $d$. It computes row maxima without retaining a full $Q\times D$ score matrix, but the input representations still occupy memory.

## 7. Choose with an experiment you can explain

Compare **dense-only** first. Add M3's sparse or multi-vector path afterward; otherwise an improvement could come from the encoder or simply an extra retrieval source.

| Question | Experiment | What it establishes |
| --- | --- | --- |
| Is the representation useful? | Same corpus and judgments, exact search, fixed K | Removes ANN approximation as a confounder |
| Does it use important details? | Contrast small changes such as “reset” versus “cannot reset,” alongside real queries | Diagnostic clues, not a benchmark made of a few invented questions |
| Is longer context useful? | Same source text; record each tokenizer's lengths, truncation, and retained evidence | Separates more available evidence from better matching |
| Do extra retrieval paths help? | Fixed final pool size; track unique relevant additions, deduplication, and reranking | Improvement beyond retrieving more candidates |
| Can it be deployed? | Same hardware and traffic conditions; measure query latency, throughput, memory, rebuild time | Costs for a specified workload |

One thousand tokens from different tokenizers need not cover the same source text. First compare common document passages, then compare each model's recommended deployment configuration, reporting these as separate experiments. Select instructions, dimensions, and fusion weights on development data, not by repeatedly tuning on the final test set.

Look below the average: short queries, Chinese, English, cross-language retrieval, code identifiers, and long documents may behave differently. Human relevance judgments and clicks can both be incomplete; unlabeled does not mean confirmed irrelevant. Leaderboards help shortlist models, not replace these checks.

If the existing dense system already finds most useful evidence, inspect omissions, chunking, and reranking before rebuilding everything. For missed exact names or error identifiers, add [BM25](tfidf-and-bm25.en.md) as a baseline. When multiple sources help, continue to [fusion and reranking](hybrid-and-reranking.en.md) and account for both gains and costs.
