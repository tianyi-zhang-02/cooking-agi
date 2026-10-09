# Pretraining: how does a pile of text become a model?

[中文](pretraining-pipeline.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [LM objectives](language-model-objective.en.md), [one training step](training-step.en.md)

Once next-token prediction makes sense, pretraining can look like a loop that feeds text into a loss. Yet identical models and objectives can learn different things because the documents, repetitions, and sequence boundaries differ.

We will follow a fictional small corpus. The numbers illustrate the accounting; they are not a recommended recipe.

## Preserve provenance before transforming text

Suppose the sources are web pages, code, and technical documentation. Preserve source, collection time, content hash, language, and processing version before removing navigation menus, broken encodings, and repeated templates.

Keeping only the final strings makes a faulty parser difficult to audit. It also hides which languages and domains a filter removed. Being able to download text is not permission to train on it: check licensing, privacy, and deletion requirements first.

| Operation | Intended target | Possible collateral damage |
| --- | --- | --- |
| Format and language checks | Broken extraction and out-of-scope text | Mixed languages and unusual symbols |
| Quality filtering | Boilerplate and meaningless concatenation | Dialects, short documents, specialist writing |
| Exact deduplication | Identical documents | Repetition that carries useful context |
| Near deduplication | Slightly edited mirrors | Similar documents with different conclusions |

[RefinedWeb](https://arxiv.org/abs/2306.01116) provides a public study of filtering and deduplication, not a universal filter recipe. Inspect samples from both retained and rejected documents.

Split training and validation at an appropriate source, duplicate-cluster, or temporal level. Randomly placing near-identical documents on opposite sides creates leakage. Benchmark paraphrases and leaked answers also evade simple exact matching.

## Mixture weights determine what gets repeated

Imagine 8 million unique web tokens, 1 million code tokens, and 1 million documentation tokens after deduplication. Proportional sampling gives 80/10/10. Changing the training mixture to 50/30/20 makes code appear more often.

With a 20-million-token training budget:

| Source | Unique tokens | Sampled tokens | Average reuse |
| --- | ---: | ---: | ---: |
| Web | 8 million | 10 million | 1.25 |
| Code | 1 million | 6 million | 6 |
| Documentation | 1 million | 4 million | 4 |

Repeating code is not acquiring five million independent new code tokens. Oversampling can shift capabilities while increasing memorization and overfitting. Compare mixtures at a controlled token or compute budget, and inspect domain-level validation rather than only the mixture average.

These are **token proportions**. Sampling documents instead changes the realized token mixture when document lengths differ.

## Packing is more than reducing padding

Two tokenized documents become:

```text
Document 1: [A, B, EOS]
Document 2: [C, D, EOS]
Packed:     [A, B, EOS, C, D, EOS]
```

May C attend to A and B? That is part of the training definition:

- **Continuous stream:** an ordinary causal mask allows cross-document attention. EOS indicates a boundary but does not enforce one.
- **Independent-document packing:** a block-diagonal causal mask restricts C to document 2. Whether position IDs reset is a separate implementation contract.

To avoid training the cross-document prediction of C after EOS, also mask that **loss term**. Attention masks determine visibility; loss masks determine scored targets. They are not interchangeable. The first token of a document may be unscored or predicted from its own BOS, depending on the format.

| Input position | Next target | Continuous stream may score | Independent documents, no BOS here |
| --- | --- | --- | --- |
| A | B | Yes | Yes |
| B | EOS | Yes | Yes |
| EOS | C | Yes | No |
| C | D | Yes | Yes |
| D | EOS | Yes | Yes |

Padding must not become an ordinary target. Printing tokens, labels, document IDs, and masks for one example is more informative than observing a decreasing loss.

## What is averaged in one update?

For valid positions $\mathcal V$, the token-mean loss is

$$
\mathcal L=-\frac{1}{|\mathcal V|}
\sum_{t\in\mathcal V}\log p_\theta(x_{t+1}\mid x_{\le t}).
$$

Suppose two microbatches contain 2 and 6 valid targets, with summed losses 4 and 6. The global mean is $10/8=1.25$. Averaging their individual means gives $(2+1)/2=1.5$. Gradient accumulation has the same weighting issue.

```python
loss_sums = [4.0, 6.0]
valid_counts = [2, 6]
token_mean = sum(loss_sums) / sum(valid_counts)
batch_mean = sum(total / count for total, count in zip(loss_sums, valid_counts)) / 2
assert token_mean == 1.25
assert batch_mean == 1.5
```

A typical update fixes the valid-token denominator for its accumulation window, accumulates correctly weighted gradients, unscales and clips when needed, then steps the optimizer. See [distributed training](../../06-systems/distributed-training.en.md) for synchronization and normalization across devices.

## Why consider the schedule and budget together?

Warmup gradually increases the learning rate; a specified decay schedule follows. Neither repairs broken masks or poor data. Record whether the schedule advances by optimizer steps or consumed tokens, especially when changing batch sizes or resuming a run.

A rough estimate for dominant dense Transformer training matrix work is $C\approx6ND$, with $N$ parameters and $D$ training tokens. Long-context attention, recomputation, communication, and hardware utilization are outside that simple estimate. FLOPs are not elapsed time.

[Chinchilla](https://arxiv.org/abs/2203.15556) studies allocation between model size and data under a training compute budget. It is not a law prescribing one tokens-per-parameter ratio for every project. Data quality, repetition, availability, and eventual inference volume also matter.

For example, 262,144 valid targets per global update over 1,000 updates gives 262,144,000 targets. A measured one-second step time supports an estimate of the training segment's duration. A guessed step time does not establish a GPU budget.

## What makes a run reproducible?

Save model, optimizer, scheduler, random states, data recipe, and data position together; mixed precision may also require scaler state. Loading weights and restarting at “step 1,001” need not continue the same trajectory.

Fix tokenizer, split, masks, and scoring unit for validation. Inspect code, language, and length slices separately. Token perplexity is not directly comparable across tokenizers. Also evaluate actual tasks: better text prediction does not automatically imply better instruction following.

Three cheap checks come first: can the model fit a tiny batch; does changing padding leave valid-token loss unchanged; and does one resumed update match uninterrupted training within the chosen tolerance? Scale up after these pass.
