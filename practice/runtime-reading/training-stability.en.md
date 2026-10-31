# Why training becomes unstable: from samples to an update

[中文](training-stability.md) · **English**

When loss oscillates, it is tempting to blame the learning rate. Sometimes that is right. But the weights assigned to long and short samples may have changed, padding may enter the loss, low-precision intermediates may overflow, or the learner may receive increasingly stale rollouts. Similar curves can have different causes.

We first use `A B C D → E F G` to check which positions the model learns, work through microbatch denominators, then examine numerics, RL data, and recovery. Start with the section closest to your symptom rather than changing the whole training setup at once.

## Define the symptom first {#symptoms}

| Observation | First distinction | First evidence |
| --- | --- | --- |
| NaN or Inf loss | Where did the first non-finite value appear? | Preserve the failing batch; inspect inputs, logits, loss, and gradients |
| Oscillating or suddenly larger loss | Harder data or a different update scale? | Valid-token counts, data slices, learning rate, and pre-clipping gradient norm |
| Falling loss but worse validation | Overfitting, leakage, wrong objective, or changed evaluation? | Fixed validation data, identical generation settings, and failures |
| Rising reward but worse answers | Reward exploitation or a changed output distribution? | Independent evaluation, lengths/truncation, and reward components |
| Training stops progressing | No learning signal or blocked execution? | Worker stages, queues, collective order, and timeouts |

Stochastic gradients are noisy. An uneven curve is not sufficient evidence of failure, and a run without exceptions is not sufficient evidence of useful learning. Compare quality and cost on the same task while retaining concrete samples you can replay.

## Which positions contribute to the loss? {#masks}

Take four prompt tokens followed by three answer tokens. In next-token prediction, position D predicts E, E predicts F, and F predicts G. The logits predicting the three answer targets are therefore shifted by one position relative to those targets.

Count only answer tokens E, F, and G for now, leaving EOS aside. Aligning prediction positions and targets makes a shift error easier to see than checking two equally sized tensors:

<div class="compact-table" markdown="1">

| Logits at | Next target | In loss? |
| --- | --- | --- |
| A | B | No |
| B | C | No |
| C | D | No |
| D | E | Yes |
| E | F | Yes |
| F | G | Yes |

</div>

The first column identifies the position whose logits make the prediction. The first three targets are still in the prompt, so they are excluded from answer loss. Position D predicts E, the first answer token, so that term is included.

These are six aligned prediction/target pairs with loss mask `[0, 0, 0, 1, 1, 1]`. Starting the loss at E's logits would omit the prediction of E given the full prompt. EOS, truncation, and tool responses need their own treatment in a complete example.

The same distinction applies to real training records:

| Content | Available as context? | Included in this policy/answer loss? |
| --- | --- | --- |
| User prompt | Yes | Objective-dependent; commonly excluded from response-only SFT |
| Model-generated answer | Yes | Valid action or target positions are included |
| Tool observation | Yes | Usually not a policy action |
| Padding | Alignment only | Excluded from valid targets |

**Attention masks and loss masks serve different purposes.** Text can be visible as context without contributing to the current loss. Packing also needs to prevent attention across unrelated example boundaries.

Print a real example as four aligned columns: token, target, attention boundary, and loss mask. Shapes alone do not establish correctness. In NeMo RL, trace shifting and masks through [`prepare_loss_input`](https://github.com/NVIDIA-NeMo/RL/blob/1c4020163a5523be4365de7fdeaa411770ecc589/nemo_rl/algorithms/loss/loss_input.py#L199). For SFT data rules, see [data and objectives](../post-training/data-and-objectives.en.md).

## How should microbatch losses be combined? {#normalization}

Suppose the chosen objective is **mean loss over all valid tokens**. Microbatch A contains 100 valid tokens with mean loss 1; B contains 10 with mean loss 3.

$$
\begin{aligned}
L_{\mathrm{token}}&=\frac{100\times1+10\times3}{110}=\frac{13}{11}\approx1.182,\\
L_{\mathrm{batch}}&=\frac{1+3}{2}=2.
\end{aligned}
$$

The second expression gives B equal weight despite its tenfold smaller token count. It is not necessarily an invalid objective, but it **is not the declared token-average objective**. The resulting loss values are not directly comparable either.

Consider each token's weight. Averaging the two microbatch means gives an A token weight `1/200` and a B token weight `1/20`: each B token counts ten times as much. Under the global token mean, both receive `1/110`. This affects the relative gradient contributions, not just the number printed in a log.

For gradient accumulation, each microbatch's loss sum can be divided by the total valid-token count of the logical batch before accumulating gradients. That total must be known or handled by an equivalent method. Dividing each microbatch mean by the accumulation count is insufficient when token counts differ.

<details markdown="1">
<summary>On multiple GPUs, does gradient averaging divide once too often?</summary>

Assume R data-parallel ranks average gradients, with N valid tokens globally. The following local scalar produces the global token-average gradient after reduction:

$$
\begin{aligned}
L_r&=\frac{R}{N}\sum_{i\in\mathcal{D}_r}\ell_i,\\
\frac{1}{R}\sum_r\nabla L_r&=\frac{1}{N}\sum_i\nabla\ell_i.
\end{aligned}
$$

This assumes disjoint data, R ranks in the actual averaging group, and no additional scaling elsewhere. The backend may already implement normalization; do not multiply by R again without checking. A rank with no valid targets cannot simply skip collectives that other ranks await. An entirely empty logical batch needs explicit handling, not division by zero.

</details>

The [small experiment](code/runtime_checks.py) reproduces 1.182 versus 2 and checks two-rank reduction with explicitly supplied gradient sums; it does not infer gradients from mean loss values. See [distributed training](../../06-systems/distributed-training.en.md) for the broader execution layouts.

## Where do NaNs arise, and what does clipping limit? {#numerics}

Take two-class logits `[1000, 1001]`, with the second class correct. Direct exponentiation can overflow, but subtracting the maximum yields a cross-entropy of just:

$$
\log\left(1+e^{-1}\right)\approx0.313262.
$$

Locate the first non-finite value rather than immediately replacing NaNs with zero. Stable `log_softmax`, cross-entropy, and necessary higher-precision reductions implement the intended objective; they do not repair incorrect labels or masks.

Also distinguish **gradient clipping**, which limits a gradient norm, from **PPO clipping**, which modifies probability-ratio terms in a surrogate objective. They are not substitutes.

For example, gradient `[3, 4]` has L2 norm 5. With a norm limit of 1, ideal norm clipping scales the whole vector by `1/5` to `[0.6, 0.8]`, preserving its direction. Clamping each component to 1 would instead produce `[1, 1]`, a different operation. Implementations may add epsilon for numerical safety; see [PyTorch's norm-clipping API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html). If the gradient already contains NaNs, investigate their origin rather than expecting rescaling to repair them.

With FP16 loss scaling, accumulate the logical batch's scaled gradients before unscaling, checking/clipping, stepping, and updating the scaler. Changing the scale during accumulation corrupts the sum's meaning. See the [PyTorch AMP example](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html#gradient-clipping) for ordering. BF16 often does not require the same scaling, but it does not make every operation immune to overflow or precision loss.

## In RL, the model producing the data changes too {#rl}

Supervised training can repeatedly use a fixed dataset. In online RL, changing the policy changes subsequent answers, lengths, and success rates. Asynchronous sampling adds another difference: the model learning now may not be the model that produced its samples.

Consider one action with behavior probability 0.2 and current probability 0.3: its ratio is 1.5. Substituting another policy's 0.25 as the denominator gives 1.2. Everything remains finite and executable, but the comparison has changed. See [asynchronous RL](../../06-systems/async-rl.en.md#objective) for behavior, old, and current policies.

A small example illustrates one [PPO](https://arxiv.org/abs/1707.06347) clipped term: ratio 1.5 and clipping interval `[0.8, 1.2]`.

| Advantage | Unclipped term | Clipped term | Minimum |
| --- | --- | --- | --- |
| +1 | 1.5 | 1.2 | 1.2 |
| −1 | −1.5 | −1.2 | −1.5 |

The positive-advantage example stops rewarding further increases in that ratio; the negative-advantage example retains the penalty for worsening it. This is not a hard bound on all action probabilities, nor a guarantee that global KL or model capabilities remain stable.

GRPO also needs informative groups. An all-zero or all-one reward group has zero mean-centered relative advantage; KL or other loss terms may still contribute gradients. Very small standard deviations require numerical care, but increasing epsilon cannot create useful feedback.

Look beyond mean reward: inspect within-group rewards, the fraction of equal-reward groups, valid action counts, lengths/truncation, policy age, ratios/clip fraction, and independent evaluation. Consuming only the earliest completed answers can favor short tasks. That is a sample-selection issue, not something faster GPUs automatically fix.

## Does lower loss mean a better model? {#evaluation}

Loss can decrease smoothly toward the wrong behavior. Filtering difficult examples can improve the training average. Shorter output limits can accelerate rollouts while harming long-reasoning tasks. A reward parser that accepts malformed answers can create a loophole the policy quickly learns.

Keep a fixed validation snapshot alongside recent traffic slices. Compare candidate checkpoints with the same tokenizer, prompt format, sampling settings, and evaluator, then inspect failures. Beyond the average, retain checks for existing capabilities, the target task, and vulnerable long-tail cases.

Not every use case benefits from fine-tuning or RL. If the target task does not improve and existing capabilities regress, keeping the original model is a useful result. Continue with [data pipelines](../post-training/data-pipeline.en.md) and [experiments and release](../post-training/experiments-and-release.en.md).

## Did the restart actually resume the same training? {#recovery}

Weights, optimizer state, learning-rate progress, random state, and data consumption must agree. RL can add in-flight rollouts, queue payloads, group indexes, and generation versions. Loading only weights may start a legitimate new experiment, but should not silently be called equivalent recovery.

Try a small interruption test. With fixed data and configuration, run two updates. In another run, save after the first, restart, and perform the second. Compare model/optimizer state, consumed sample IDs, step counts, and outputs. Allowed differences depend on nondeterminism and the recovery contract. Asynchronous systems need not promise bitwise replay, but should explain discarded, repeated, or regenerated work.

This is why [NeMo RL's sample lifecycle](nemo-rl.en.md#lifecycle) concerns learning correctness as well as operations. See [checkpoints and recovery](../post-training/checkpoint-and-resume.en.md) for persistence details.

## An order that keeps debugging manageable {#debug-route}

1. Preserve the failing sample and configuration; check tokens, label shifts, masks, and denominators.
2. Compare forward results, loss, and gradients on a small higher-precision path. Overfitting a tiny batch is a check, not evidence of generalization.
3. Restore mixed precision, packing, and accumulation one at a time.
4. Add multiple GPUs and verify reduction, partitioning, and recovery rather than guessing from a large-cluster curve.
5. For RL, check the loss on traceable fixed rollouts before restoring online generation, weight refreshes, and asynchronous queues.
6. Finally compare time, useful samples, GPU-hours, and independent evaluation at the same quality threshold.

Explain differences on the same small dataset before restoring the full configuration. Even before finding the best hyperparameters, you can then tell whether a change affected data, the objective, numerical computation, or execution, rather than trying to infer several changes from one curve.
