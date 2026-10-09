# Good on Training Data, Poor on New Data?

[中文](generalization.md) · **English**

> Reading time: about 12 minutes · Level: foundations to advanced · Reviewed: 2026-10

Training accuracy keeps improving while validation barely moves. Should you add data, change the model, or increase regularization? A curve describes the symptom; it does not identify the cause.

This chapter separates overfitting, underfitting, leakage, and distribution shift, then works through what L1, L2, and Dropout actually change. Start with [linear to neural models](../from-linear-to-neural.en.md) if losses and gradients are new to you.

## First Make the Comparison Meaningful

These classification error rates are teaching examples, not measured results:

| Case | Training error | Validation error | First thing to investigate |
| --- | --- | --- | --- |
| A | 2% | 18% | Split design, leakage, distribution differences, then overfitting |
| B | 24% | 25% | Broken training, weak features, or underfitting |
| C | 1% | 1% | An easy task—or duplicate samples across splits? |
| D | 8% | 5% | Augmentation / Dropout enabled only during training? |

A **generalization gap is not a diagnosis**. Training on daytime images and validating on night scenes does not establish memorization. Nor should you subtract a regularized training objective from an unregularized validation loss and interpret the result as a gap.

Compare task metrics under matched weighting, preprocessing, and model modes; record the optimization objective separately. `model.eval()` changes Dropout / BatchNorm behavior but does not disable autograd.

## Overfitting, Underfitting, or Unsuccessful Optimization?

Overfitting means adaptation to the training sample fails to transfer adequately to new samples from the intended distribution. Underfitting means the model or training setup has not learned the available pattern well. These are not synonyms for large and small models.

A useful debugging order:

1. **Memorize a tiny clean batch.** Temporarily remove augmentation and strong regularization. Inspect labels, loss, gradients, and which parameters belong to the optimizer before increasing width.
2. **Match the split to the claim.** Future prediction calls for time-aware evaluation; new-user generalization requires preventing the same person from appearing in both splits.
3. **Plot training and learning curves.** The former varies update count; the latter varies training-set size. They answer different questions.
4. **Change one major factor at a time.** Compare more updates, more capacity, more data, or different regularization with the evaluation set held fixed.

Memorizing a tiny batch checks the training path, not generalization. More rows need not mean more independent information: ten thousand duplicates may teach less than one thousand representative examples.

## Validation Selects; Test Data Checks the Final Choice

Early stopping selects a checkpoint using validation performance. Cross-validation estimates performance across splits and supports hyperparameter selection. Rotating which fold is held out does not automatically improve a single model.

If every disappointing test result causes another design change, the test set has become part of selection. Reserve independent final evaluation, with group or time splits when samples are dependent.

Preprocessing is also learned. With training values `[0, 2]`, the mean is 1. Include a future observation of `100`, and it becomes 34. No label was accessed, but future distribution information entered training. Fit preprocessing only on training data, then apply that transform to validation and test. Each cross-validation fold needs its own fitted preprocessing. See the [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## L1 and L2 Through One Parameter

Take a data loss of $\frac12(w-a)^2$, where $a$ is the unregularized optimum.

Adding a **squared L2 penalty** $\frac\lambda2w^2$ gives

$$w_{L2}=\frac{a}{1+\lambda}.$$

This shrinks weights toward zero without generally making a nonzero $a$ exactly zero at finite $\lambda$. The square matters: $\lambda\lVert w\rVert_2$ is a different objective.

For an L1 penalty $\lambda|w|$, the subgradient condition at zero gives soft thresholding:

$$w_{L1}=\operatorname{sign}(a)\max(|a|-\lambda,0).$$

With $a=0.4,\lambda=0.5$, L1 gives 0 while squared L2 gives about 0.267. This explains exact zeros; it does not prove a discarded feature is irrelevant. Correlated features may substitute for one another.

```python
import math

def regularized_scalar(target, strength, kind):
    if not math.isfinite(target) or not math.isfinite(strength) or strength < 0:
        raise ValueError("finite target and nonnegative strength required")
    if kind == "l1":
        return math.copysign(max(abs(target) - strength, 0.0), target)
    if kind == "squared_l2":
        return target / (1.0 + strength)
    raise ValueError("unknown penalty")
```

Units affect the penalty: multiplying a feature by 100 allows its coefficient to shrink by 100 while preserving predictions. Equal regularization strengths no longer mean equal constraints. Specify scaling, whether the intercept is penalized, and parameter groups excluded from weight decay.

## Dropout Preserves an Expectation, Not Every Output

Inverted dropout draws $m\sim\operatorname{Bernoulli}(1-p)$ and uses

$$\tilde h=\frac{mh}{1-p},\qquad 0\le p<1.$$

For $h=4,p=0.5$, training outputs 0 or 8 with equal probability, retaining mean 4. Evaluation uses 4 directly, without another retention multiplier. This is the convention in [PyTorch Dropout](https://docs.pytorch.org/docs/main/generated/torch.nn.Dropout.html).

Now apply a square afterward. The training expectation is $(0^2+8^2)/2=32$, while evaluation gives $4^2=16$. Preserving activation expectations does not make a nonlinear evaluation network the exact average of all subnetworks. Dropout is training noise and a constraint, not an exactly equivalent ensemble for free.

Units are not permanently removed; masks are resampled during training. Stronger Dropout is not always better. Its interaction with small-data SFT and the pretrained distribution needs validation.

## Preprocessing Is Not the Same as Network Normalization

| Operation | Source of statistics | Important qualification |
| --- | --- | --- |
| Min–max scaling | Training-feature extrema | New values may leave `[0,1]`; constant features need handling |
| Z-score | Training mean and standard deviation | Divide by standard deviation, not variance; this does not make arbitrary data Gaussian |
| BatchNorm | Batch statistics in training; running statistics by default at evaluation | These statistics are not optimizer-trained gamma / beta |
| LayerNorm / RMSNorm | Selected dimensions of the current input | Not a single fixed linear transformation for every input |

A nondegenerate fixed affine transformation can preserve a one-dimensional distribution's shape, but changes its values, mean, and variance. Calling the distribution unchanged is misleading. Network normalization also uses input-dependent statistics. See [normalization](../core/normalization.en.md) for axes and formulas.

## Turn the Diagnosis into an Experiment

Do not apply every remedy at once. Fix a sensible split and metric, then compare baseline, more data only, different regularization only, and more capacity only. Record training and validation performance, computational cost, and variability.

Failure on one subgroup suggests checking coverage and labels. Failure to reduce training loss calls for checking [activations and initialization](activation-and-initialization.en.md) and [optimizers](optimizers.en.md). Each change should test a hypothesis, rather than treating every gap as a request for more Dropout.
