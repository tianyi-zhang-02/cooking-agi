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

## 3. Work through one training step {#training-step}

Set temperature to 1. Scores $[2,1,0]$ for one positive and two comparisons give softmax probabilities of approximately $[0.665,0.245,0.090]$. The loss is $-\log(0.665)\approx0.408$.

Writing the temperature-scaled scores as $z_j=s_j/\tau$:

$$
\frac{\partial L}{\partial z_j}=p_j-\mathbb 1[j=+].
$$

The gradients are approximately $[-0.335,0.245,0.090]$. Gradient descent raises the positive score and lowers both comparisons, pushing harder against the higher-scoring comparison. Gradients with respect to the original scores $s_j$ include another factor of $1/\tau$.

```python
import math

def contrastive_step(scores, temperature=1.0):
    if not scores or temperature <= 0:
        raise ValueError("Need scores and a positive temperature")
    scaled = [score / temperature for score in scores]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    loss = maximum + math.log(total) - scaled[0]
    gradients = [(probability - int(index == 0)) / temperature
                 for index, probability in enumerate(probabilities)]
    return loss, probabilities, gradients

loss, probabilities, gradients = contrastive_step([2, 1, 0])
assert abs(loss - 0.407606) < 1e-5
assert gradients[0] < 0 < gradients[1]
assert abs(sum(gradients)) < 1e-12
```

Now imagine the second document also explains vanishing gradients well, but remains labeled negative. The code runs correctly while pushing it away. That is a false negative: **correct numerical computation does not guarantee correct supervision.**

## 4. Multiple positives and hard negatives

In-batch negatives are cheap: reuse other queries’ positives as comparisons. But several documents may answer the same query. Check duplicate documents and known multiple positives; larger batches can also increase opportunities for false negatives.

For a known positive set $P$, one design maximizes its total probability:

$$
L_{\mathrm{set}}=-\log\sum_{j\in P}p_j.
$$

This permits most probability to concentrate on one positive. Another design gives each positive its own term: $L_{\mathrm{avg}}=-|P|^{-1}\sum_{j\in P}\log p_j$. They are not equivalent: the first rewards finding at least one; the second asks for probability on every known positive. The task, not the shorter formula, should decide.

Harder negatives are not always better. A high-scoring “negative” might expose a model error or a missing label. Inspect a small set before increasing difficulty. Record where negatives came from, when they were mined, and which model found them; see [feedback and objectives](../01-data-and-feedback/feedback-to-objectives.en.md).

## 5. Why can a model upgrade break search? {#index-version}

Suppose training updates both encoders, but deployment updates only the query encoder while retaining old document vectors. Dimensions and APIs still match, but the two spaces may no longer align. Unless compatibility with the old space was explicitly enforced, “both are 768-dimensional” is not enough.

A reproducible retrieval version links encoder checkpoints, tokenizer, text cleaning, chunking, pooling, normalization, similarity function, and index snapshot. Test fixed queries before deciding whether to rebuild the index or retain a compatible path.

Separate two diagnoses before tuning ANN:

1. **Exact search also misses the document:** inspect representation, training data, chunking, and ingestion.
2. **Exact search finds it but ANN does not:** inspect indexing parameters, quantization, filters, and search budget.

ANN recall measures how well approximate search recovers exact neighbors. Relevant-document Recall measures recovery of labeled relevant content. Neither substitutes for the other: perfectly recovering irrelevant nearest neighbors is not task success.

## 6. Before adding more vectors

A query asking for both beginner-friendly explanation and mathematical detail may lose one constraint in a single vector. Try keyword support, query reformulation, or reranking a small pool before adding more vectors. The evidence for multi-vector retrieval is useful content missed by the single-vector baseline, not simply a larger query budget.

Hold the final reranking pool size fixed and track unique relevant additions, duplication, latency, and index size. [Hybrid retrieval and reranking](hybrid-and-reranking.en.md) continues this example with a small document collection.
