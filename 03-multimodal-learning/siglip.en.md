# SigLIP: image–text matching without a softmax competition

[中文](siglip.md) · **English**

> Checked 2026-10-11. We separate the SigLIP objective (2023) from SigLIP 2 extensions (2025). Arithmetic and code are original teaching examples, not pretraining results.

[CLIP](clip.en.md) asks which text in a batch matches an image. SigLIP asks a different question: **does this particular image–text pair match?** Each pair contributes a loss without first normalizing a whole row into probabilities summing to one.

The architecture can still encode images and text separately and compare their vectors. Changing the objective does not create a conversational multimodal LLM. [SigLIP paper](https://arxiv.org/abs/2303.15343)

## Start with two pictures {#two-pairs}

Take a bicycle and a cup of coffee, with matching descriptions. Suppose the model produces these **logits**, already including similarity scaling and bias—not raw cosines:

| | “A bicycle” | “A cup of coffee” |
| --- | ---: | ---: |
| Bicycle image | 2 | −2 |
| Coffee image | −2 | 2 |

Sigmoid maps 2 to about 0.881 and −2 to 0.119. Each entry is a separate decision: high scores on the diagonal, low scores elsewhere.

CLIP's first-row softmax is approximately `[0.982, 0.018]`. Add another suitable description with logit 2 and the first probability falls to about 0.495. SigLIP's original pair score remains 0.881, **holding the model and existing logits fixed**. This does not mean adding negatives during training leaves parameters unchanged.

Neither number is inherently more accurate. They answer different questions, and neither is automatically a calibrated real-world correctness probability.

## The dual encoder remains; the objective changes {#architecture}

<figure class="worked-update" lang="en">
<ol>
<li><small>01 / Encode separately</small><strong>Images and text become vectors</strong><span>Encode, project, and normalize. Candidate vectors can still be precomputed.</span></li>
<li><small>02 / Score pairs</small><strong>Scale similarity and add bias</strong><span>z = scale × cosine + bias. Negatives outnumber positives, so the bias matters.</span></li>
<li><small>03 / Learn from pairs</small><strong>Each pair contributes logistic loss</strong><span>Positive scores should rise; negative scores should fall. All terms update shared encoder parameters.</span></li>
</ol>
<figcaption>Pairwise loss terms do not imply separate parameters or statistically independent training examples.</figcaption>
</figure>

For $B$ paired examples, let $u_i,v_j$ be normalized vectors, $z_{ij}=\exp(a)u_i^\top v_j+b$ the logits, and $y_{ij}\in\{+1,-1\}$ the labels. The parameters $a$ and $b$ are learned; the exponential keeps the scale positive.

With one positive per row and all image–text combinations in the batch, the objective is:

$$L=\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}\operatorname{softplus}(-y_{ij}z_{ij}).$$

The denominator is $B$, not $B^2$. The [official trainer](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/trainers/proj/image_text/siglip.py#L289) sums pair losses within rows and averages rows. A default BCE mean over all matrix elements reduces the gradient scale by another factor of $B$. You can account for that scale at fixed batch size, but it is not an unchanged training setup.

## Compute the loss {#loss-example}

All four terms above equal $\log(1+e^{-2})\approx0.1269$, giving $L\approx4\times0.1269/2=0.2539$. This standard-library example avoids directly computing `exp(1000)`:

```python
import math


def pair_loss(logit, positive):
    if not math.isfinite(logit) or not isinstance(positive, bool):
        raise ValueError("expected a finite logit and a boolean label")
    margin = -logit if positive else logit
    return max(margin, 0.0) + math.log1p(math.exp(-abs(margin)))


def siglip_loss(logits):
    batch_size = len(logits)
    if batch_size == 0 or any(len(row) != batch_size for row in logits):
        raise ValueError("expected a nonempty square score matrix")
    return math.fsum(
        pair_loss(score, row_index == column_index)
        for row_index, row in enumerate(logits)
        for column_index, score in enumerate(row)
    ) / batch_size


scores = [[2.0, -2.0], [-2.0, 2.0]]
assert math.isclose(siglip_loss(scores), 2 * math.log1p(math.exp(-2)))
```

This checks the formula with diagonal positives; it is not a production trainer. Near-duplicate images, repeated captions, or several valid descriptions require explicit multi-positive or sampling rules. Sigmoid does not identify false negatives for you.

<details markdown="1">
<summary>One step deeper: what do the gradient and bias do?</summary>

For a binary label $q\in\{0,1\}$, the derivative of an unnormalized pair loss is $\sigma(z)-q$. Gradient descent raises an under-scored positive and lowers an over-scored negative. The full objective also applies $1/B$.

If all similarities are temporarily zero and only a common bias is learned, one positive and $B-1$ negatives per row give $B\sigma(b)-1=0$. Thus $b=-\log(B-1)$ for $B>1$. This simplified derivation explains class balance; it does not prescribe official initialization or the final trained bias. Changing the sampled negative ratio changes the calculation.

</details>

## Easier chunking does not make negatives free {#distributed}

Without a global softmax denominator, pair terms can be accumulated in chunks. Suppose four GPUs each hold 32 pairs. Local comparisons provide 31 negatives per image. A target pool of 128 pairs also requires information about 96 remote texts. Embedding exchange, computation placement, and gradient flow still need an implementation.

Four accumulated local batches do not automatically create cross-batch pairs. Dropping negatives or changing their ratio changes the objective, not just its memory use. A larger pool can also contain more false negatives.

## What SigLIP 2 adds {#siglip2}

[SigLIP 2](https://arxiv.org/html/2502.14786v1) extends multilingual data and the training recipe with captioning-related tasks, self-distillation, masked prediction, and different image-shape handling. **Not every variant uses every stage**: the paper's NaFlex path omits the standard variant's self-distillation and masked-prediction stage.

<!-- paper-figure:siglip2-recipe -->

| Problem | Useful interpretation | Check when selecting a checkpoint |
| --- | --- | --- |
| Global matching works, local semantics are weak | Supervise local representations as well as global alignment | Pooled vectors versus patch features downstream |
| Languages differ greatly in quality | Coverage depends on data and training | Language- and task-specific evaluations |
| Square resizing loses detail in a long screenshot | Preserve aspect ratio and adjust visual token budgets | Whether it is actually NaFlex and uses the matching processor |

Use the checkpoint's processor rather than carrying over CLIP resize, normalization, or tokenizer settings. Fixed-resolution and NaFlex inputs are not interchangeable switches. Consult the [official release](https://google-research.github.io/big_vision/big_vision/configs/proj/image_text/README_siglip2.html) and [Transformers SigLIP2 interface](https://huggingface.co/docs/transformers/en/model_doc/siglip2).

## Retrieval and VLMs use different outputs {#downstream}

Image–text retrieval usually needs comparable global vectors: encode candidates offline, encode queries online, then use ANN or exact scoring. After an encoder upgrade, old candidate vectors generally cannot be mixed with new query vectors. Plan reindexing and a coordinated switch.

A generative VLM often uses visual token or patch features, followed by a projector or resampler feeding an LLM. It still needs the connector, training data, and a generation objective. Adding a SigLIP encoder alone does not teach visual question answering. See [vision to language](vision-to-language.en.md). Robot VLA models connect it the same way: OpenVLA feeds DINOv2 and SigLIP features together into Llama 2 and outputs actions; see [Robot learning: VLA](../13-embodied/robot-learning/vla.en.md).

Compare CLIP, SigLIP, and SigLIP 2 on the slices you need: languages, small text, similar objects, duplicate descriptions, degraded images, and offline/online encoding costs. A newer name is not evidence of improvement on your task. Fix the data split and retrieval protocol before comparing results.
