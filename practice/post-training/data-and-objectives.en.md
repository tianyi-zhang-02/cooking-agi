# Loss is falling. What is the model learning?

[中文](data-and-objectives.md) · **English**

> Original teaching example · Reviewed: 2026-10. Token losses are supplied numbers, not measurements from a training run.

“No errors, and loss went down” establishes that a program optimized something. Open a sample to check whether that objective matches the behavior you intended to teach.

## Group related examples before splitting train and test

One original question can appear in two languages and several paraphrases. Splitting by row can put one phrasing in training and another in testing, making evaluation easier than genuinely new questions. Here we split by original question-family `group_id` so related paraphrases stay together.

That isn't sufficient for every task: hold out document families if testing transfer to new documents; split by time and inspect cross-period near-duplicates if testing future requests. **Your generalization claim determines the split unit, not the random seed.**

| Field | Why retain it |
| --- | --- |
| Sample ID and question-family ID | Traceability, deduplication, paraphrase leakage checks |
| Input-evidence revision | Avoid pairing today's answer with yesterday's evidence |
| Demonstration source and review status | Separate human corrections, generated targets, and unreviewed examples |
| Answerable / missing evidence / conflicting evidence | Detect a dataset that only teaches “always answer” |
| Split and preprocessing versions | Reconstruct training, development, and test sets |

Teacher-generated answers are candidate demonstrations, not automatic ground truth. Copying the same teacher's judgments into both training and evaluation can measure imitation rather than better task performance. Retain an independently human-reviewed slice, especially for missing or conflicting evidence.

## Loss masks: which tokens contribute to the objective?

In decoder-only SFT, questions and evidence remain in the input so the model can condition on them. We may include only assistant-answer target tokens in the loss. A loss mask **does not remove prompt context from attention**.

Suppose two positions belong to the prompt and two to the answer, with these supplied token losses:

```text
token loss:  4.0   3.0   0.8   0.4
loss mask:    0     0     1     1
valid mean: (0.8 + 0.4) / 2 = 0.6
```

Ignoring the mask gives 2.05 and changes the objective. This doesn't prove answer-only loss is universally better; it shows the configurations are different experiments.

Actual implementations must check next-token shifting: each prediction is paired with the next token, and the mask must align with the shifted targets. Check padding, end-of-sequence (EOS), and tool messages too. Our example accepts already-aligned losses and masks; it does not shift them again.

Masked prompt positions contribute no direct prediction loss, but remain context for the answer. Answer-loss gradients can flow through computations involving that context. “No prompt loss” does not mean “no gradients through any prompt-related computation.”

TRL `v0.29.0` [SFTTrainer](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer) supports `assistant_only_loss`, but the chat template must support returning assistant masks. Setting `True` isn't enough: print actual tokens and labels to verify the intended boundaries.

## Averaging by token or by answer

Example A has one valid token with loss 0.2; example B has three, each with loss 1.0:

$$
L_{\text{token}}=\frac{0.2+1+1+1}{4}=0.8,\qquad
L_{\text{sample}}=\frac{0.2+1}{2}=0.6.
$$

Under token averaging, A receives 1/4 of the weight and B receives 3/4. Under example averaging, each receives 1/2. The latter weights answers equally; the former weights valid tokens equally. A longer answer doesn't necessarily dominate the actual gradient, but it receives a larger share in this reduction.

The same issue arises across devices. Averaging per-device means with unequal valid-token counts does not produce the global token mean; the framework's gradient scaling needs a separate check.

We require at least one valid target token per example. A fully truncated answer should not silently become a zero-loss example. Reject and count it so length settings cannot quietly remove the learning target.

## Hand-check one batch before training

The [reference implementation](code/experiment_checks.py) provides `masked_objective` with both token and example means. It rejects mismatched lengths, nonbinary masks, nonfinite losses, and examples with no valid targets.

```bash
python3 practice/post-training/code/experiment_checks.py
python3 -m unittest discover -s site/tests -p 'test_practice_projects.py'
```

The demo first reports `0.8` and `0.6`. Set one mask to all zeros and confirm an explicit failure. With a real trainer, compare per-position losses for the same batch against hand calculations instead of relying on a smooth dashboard curve.

For how losses are produced, read the [PyTorch training loop](../../00-foundations/pytorch/training-loop.en.md). This note checks reductions and data contracts, not optimizer updates.

[Next: Multi-GPU choices](distributed-training.en.md) · [Skip to experiment comparison](experiments-and-release.en.md)
