# Dual encoders: why encode queries and documents separately?

**English** · [中文](dual-encoder.md)

Imagine searching a million passages of course notes. A query asks why gradients vanish. You want passages about repeated multiplication and RNNs, not just anything containing “gradient.”

You could send the query and every passage through a model together, but a million joint comparisons per request would be expensive. A dual encoder starts with a simpler question: **documents rarely change between requests, so can we compute them ahead of time and encode only the new query?**

A vector dot product is enough background for this note. For a calculation, jump to [one training step](#training-step); if you already use a vector store, start with [model–index compatibility](#index-version). All numbers are teaching examples.

## 1. What does separate encoding buy?

A query encoder produces $\mathbf q=f_\theta(x)$ and a document encoder produces $\mathbf d=g_\phi(y)$. They may share parameters or use different ones. “Dual encoder” describes the computation paths, not necessarily two independently parameterized models.

$$
s(x,y)=\mathbf q^\top\mathbf d.
$$

Normalize both vectors to unit length and the dot product becomes cosine similarity. After normalization, direction determines the score; without it, magnitude matters too. This is a training–serving convention, not a harmless serving-side switch.

```text
Document update: document → chunks → document encoder → vector index
User request: query → query encoder → neighbor search → candidates → reranking
```

The savings come from not re-encoding every document on every request, not from making search free. Brute-force comparison with $N$ vectors of dimension $d$ still costs roughly $O(Nd)$ arithmetic. Approximate nearest-neighbor search (ANN) trades index resources and some missed exact neighbors for faster lookup. Measure latency, memory, and recall rather than assuming every ANN method costs $O(\log N)$.

| Approach | What can be precomputed? | What remains per request? | Cost or limitation |
| --- | --- | --- | --- |
| Dual encoder | Document vectors | Query vector and neighbor search | Query encoding cannot inspect document details |
| Cross-encoder | Usually not the full pair score | Joint encoding of the query and each candidate | Detailed comparison is expensive; usually limited to a smaller pool |
| Late interaction | Multiple document-token vectors | Query–document token matching | More detail, but a larger index and more involved matching |

[DPR](https://arxiv.org/abs/2004.04906) is a classic learned dual encoder. [ColBERT](https://arxiv.org/abs/2004.12832) illustrates late interaction: document tokens are not all collapsed into one vector before a single dot product.

## 2. How does similarity get learned?

Mean-pooling a language model does not automatically produce geometry suited to your task. Training must specify which documents should outrank the current comparison documents for a query.

A common single-positive contrastive loss is:

$$
L=-\log\frac{\exp(s^+/\tau)}{\exp(s^+/\tau)+\sum_{j=1}^{B-1}\exp(s_j^-/\tau)}.
$$

Here $s^+$ is the positive score, $s_j^-$ are comparison scores in this batch, and $\tau>0$ is temperature. The denominator is not every irrelevant document in the world. It is the candidate set used for this training comparison.

### Positive and negative examples do not imply pairwise binary classification {#infonce-and-ce}

Keep the vanishing-gradient query. The 3 training candidates cover repeated multiplication in RNNs, learning-rate selection, and saving checkpoints. The positive is at index 0. **Single-positive InfoNCE asks which of these 3 to select, rather than asking “relevant?” independently 3 times.** It is multiclass cross-entropy over candidate indices. Those classes refer to different documents when the candidate pool changes, not fixed topic categories.

Start with scores that favor none of the candidates:

| All scores are 0 | Output | Supervision |
| --- | --- | --- |
| Candidate softmax | `[1/3, 1/3, 1/3]`, summing to 1 | Positive index `0` |
| Independent pairwise sigmoids | `[1/2, 1/2, 1/2]`, no sum-to-one constraint | Binary labels `[1, 0, 0]` |

The first loss is $\log 3\approx1.099$. Mean BCE over the 3 pairs is $\log 2\approx0.693$; summed BCE is $3\log 2$. Their numerical sizes do not establish which objective is better: the questions and reductions differ. Compare [the InfoNCE paper](https://arxiv.org/abs/1807.03748) and PyTorch 2.8's [CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html) and [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html).

One small change makes the distinction memorable. Keep the original scores fixed and add another zero-score candidate. The positive's softmax probability falls from $1/3$ to $1/4$. Each original pair's sigmoid remains $1/2$. **Candidate softmax depends on the comparison pool; it is not an absolute probability that a user likes a document.**

Pass temperature-scaled logits directly into CE. Applying softmax and then passing its probabilities as logits adds another softmax and changes the objective. This is a useful stopping point if you only need the distinction; next, keep the same 3 candidates and calculate an update.

## 3. Work through one training step {#training-step}

Set temperature to 1. Scores $[2,1,0]$ for one positive and two comparisons give softmax probabilities of approximately $[0.665,0.245,0.090]$. The loss is $-\log(0.665)\approx0.408$.

Writing the temperature-scaled scores as $z_j=s_j/\tau$:

$$
\frac{\partial L}{\partial z_j}=p_j-\mathbb 1[j=+].
$$

The gradients are approximately $[-0.335,0.245,0.090]$. Treating scores as independent variables, gradient descent raises the positive and lowers both comparisons, pushing harder against the higher-scoring comparison. An actual encoder shares parameters across scores, so an update need not move every score in that direction. Gradients with respect to the original scores $s_j$ include another factor of $1/\tau$.

```python
import math

def contrastive_step(scores, temperature=1.0):
    if not scores or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Need scores and a finite positive temperature")
    scaled = [score / temperature for score in scores]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("Scaled scores must be finite")
    maximum = max(scaled)
    shifted = [value - maximum for value in scaled]
    weights = [math.exp(value) for value in shifted]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    loss = math.log(total) - shifted[0]
    gradients = [(probability - int(index == 0)) / temperature
                 for index, probability in enumerate(probabilities)]
    return loss, probabilities, gradients

loss, probabilities, gradients = contrastive_step([2, 1, 0])
assert abs(loss - 0.407606) < 1e-5
assert gradients[0] < 0 < gradients[1]
assert abs(sum(gradients)) < 1e-12
assert math.isclose(contrastive_step([1e20, 1e20])[0], math.log(2))
```

The last check deliberately assigns two candidates the same enormous score. Each receives half the probability, so the loss must be $\log 2$, not zero. Subtract the maximum and compute the loss directly from the shifted scores. Adding the large number back and subtracting it again can erase a small loss through floating-point rounding.

Now imagine the second document also explains vanishing gradients well, but remains labeled negative. The code runs correctly while pushing it away. That is a false negative: **correct numerical computation does not guarantee correct supervision.**

## 4. Multiple positives and hard negatives

In-batch negatives are cheap: reuse other queries’ positives as comparisons. But several documents may answer the same query. Check duplicate documents and known multiple positives; larger batches can also increase opportunities for false negatives.

A random document, another user's positive interaction, or an unexposed item is not automatically confirmed irrelevant to this query. Such items can be training comparisons, but state that assumption. An absent positive label is not evidence that the user rejected the item.

For a known positive set $P$, one design maximizes its total probability:

$$
L_{\mathrm{set}}=-\log\sum_{j\in P}p_j.
$$

This permits most probability to concentrate on one positive. Another design gives each positive its own term: $L_{\mathrm{avg}}=-|P|^{-1}\sum_{j\in P}\log p_j$. They are not equivalent: the first rewards finding at least one; the second asks for probability on every known positive. The task, not the shorter formula, should decide.

Harder negatives are not always better. A high-scoring “negative” might expose a model error or a missing label. Inspect a small set before increasing difficulty. Record where negatives came from, when they were mined, and which model found them; see [feedback and objectives](../01-data-and-feedback/feedback-to-objectives.en.md).

Evaluation also needs a fixed candidate pool and labeling protocol. Adding negatives or changing the sampling method changes task difficulty; the resulting loss or Recall change cannot be attributed entirely to model quality.

## 5. Why can a model upgrade break search? {#index-version}

Suppose training updates both encoders, but deployment updates only the query encoder while retaining old document vectors. Dimensions and APIs still match, but the two spaces may no longer align. Unless compatibility with the old space was explicitly enforced, “both are 768-dimensional” is not enough.

A reproducible retrieval version links encoder checkpoints, tokenizer, text cleaning, chunking, pooling, normalization, similarity function, and index snapshot. Test fixed queries before deciding whether to rebuild the index or retain a compatible path.

Separate two diagnoses before tuning ANN:

1. **Exact search also misses the document:** inspect representation, training data, chunking, and ingestion.
2. **Exact search finds it but ANN does not:** inspect indexing parameters, quantization, filters, and search budget.

ANN recall measures how well approximate search recovers exact neighbors. Relevant-document Recall measures recovery of labeled relevant content. Neither substitutes for the other: perfectly recovering irrelevant nearest neighbors is not task success.

## 6. Before adding more vectors

A query asking for both beginner-friendly explanation and mathematical detail may lose one constraint in a single vector. Try keyword support, query reformulation, or reranking a small pool before adding more vectors. The evidence for multi-vector retrieval is useful content missed by the single-vector baseline, not simply a larger query budget.

Hold the final reranking pool size fixed and track unique relevant additions, duplication, latency, and index size. For checkpoint selection, read [Qwen3 Embedding and BGE-M3](embedding-models.en.md); for combining retrieval paths, continue to [hybrid retrieval and reranking](hybrid-and-reranking.en.md).
