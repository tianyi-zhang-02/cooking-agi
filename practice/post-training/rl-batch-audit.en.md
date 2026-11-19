# Audit an RL loss from one batch of responses

[中文](rl-batch-audit.md) · **English**

> Original teaching lab · Checked: 2026-11 · No model download; not a completed RL training run.

Writing the PPO or GRPO formula correctly is only part of the job. Did responses end up in the wrong group? Did EOS survive masking? Were old log-probabilities recorded before the update, or recomputed afterward? Code often runs despite these mistakes.

Rather than build another trainer, we follow a small batch into its loss. We use a [RLOO](../../05-post-training/rloo.en.md) baseline to avoid introducing a Critic and GAE at the same time. Grouping, alignment, and policy-version checks also apply to other sampling-based methods. This lab covers single-turn text responses; see [post-training infrastructure](../../05-post-training/post-training-infrastructure.en.md#rollout-record) for action boundaries in tool interactions.

## Keep enough information to reconstruct the update {#record}

Start with two prompts and three responses per prompt:

<figure class="worked-update">
<ol>
<li><small>01 / SAMPLE</small><strong>Keep the original tokens</strong><span>Record prompt and response IDs, generation version, sampling settings, and stop reason.</span></li>
<li><small>02 / SCORE</small><strong>Separate low scores from missing scores</strong><span>Keep reward components, scorer version, and execution errors.</span></li>
<li><small>03 / GROUP</small><strong>Find the three sibling responses</strong><span>Use identifiers, not asynchronous arrival order, to construct advantages.</span></li>
<li><small>04 / RECOMPUTE</small><strong>Align the current probabilities</strong><span>Run the same tokens through the policy and retain its computation graph.</span></li>
<li><small>05 / REDUCE</small><strong>Check masks and denominators</strong><span>Calculate by hand, then compare a whole batch with split batches.</span></li>
</ol>
<figcaption>Check each stage independently before integrating a model and distributed training.</figcaption>
</figure>

| Record | Question it should answer |
| --- | --- |
| `prompt_id`, `sample_id`, input and response token IDs | Which responses belong together? Was a record consumed twice? |
| Action mask, stop reason, actual length | Which positions were sampled actions? Did generation emit EOS or reach a limit? |
| Policy version, tokenizer/template version, sampling settings | Which distribution and input produced the response? |
| Reward components, scoring status, scorer version | Was zero a measured result or a missing service response? |
| Required old/reference/sampler log-probabilities | Who computed each array, for which tokens, and when? |

Not every method needs all these fields. There is no need to store three probability arrays mechanically, but any array you use needs a precise meaning. Decoded text alone is not enough: tokenizing it again need not recover the original sampled actions.

## One shift changes every target {#alignment}

Consider five positions. `<P>` stands for a prepared prompt token; this is not a particular real tokenizer.

| Input position | Token | Action? | Predicted by logits at |
| --- | --- | ---: | --- |
| 0 | `<P>` | No | Outside this example |
| 1 | `A` | Yes | Position 0 |
| 2 | `B` | Yes | Position 1 |
| 3 | `<EOS>` | Yes | Position 2 |
| 4 | `<PAD>` | No | Excluded from the loss |

At position $t$, a causal model predicts token $t+1$. After shifting, `labels=[A,B,EOS,PAD]` and the action mask is `[1,1,1,0]`. If the three actions have conditional probabilities 0.5, 0.25, and 0.8, their sequence log-probability is $\log(0.5\times0.25\times0.8)=\log0.1\approx-2.302585$.

Once a model has returned `logits`, the following PyTorch helper selects the sampled tokens' probabilities:

```python
import torch


def completion_log_probs(logits, input_ids, action_mask):
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, length, vocabulary]")
    if action_mask.shape != input_ids.shape or action_mask.dtype != torch.bool:
        raise ValueError("Expected a boolean action mask matching input IDs")
    if action_mask[:, 0].any():
        raise ValueError("The first input position has no preceding logits")
    shifted_mask = action_mask[:, 1:]
    if not shifted_mask.any(dim=-1).all():
        raise ValueError("Every response needs at least one sampled action")
    safe_logits = logits[:, :-1].float().masked_fill(~shifted_mask[..., None], 0)
    token_log_probs = safe_logits.log_softmax(dim=-1).gather(
        -1, input_ids[:, 1:, None]
    ).squeeze(-1)
    if not torch.isfinite(token_log_probs[shifted_mask]).all():
        raise ValueError("Action log-probabilities must be finite")
    token_log_probs = token_log_probs.masked_fill(~shifted_mask, 0)
    return token_log_probs, token_log_probs.sum(dim=-1)
```

Use `log_softmax`, rather than `softmax` followed by `log`, for its numerical treatment; see the [PyTorch documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.log_softmax.html). The real forward pass still needs the correct attention mask, positions, and template. An action mask selects policy-loss positions; it is not an attention mask. The prompt conditions the answer even though it has no direct policy-loss term.

If PAD and EOS share an ID, constructing the action mask as `token != pad_id` also deletes a genuinely generated EOS. Use actual generation boundaries. A length-truncated response did not emit EOS: appending one does not make it a sampled action. Scoring and retaining truncated responses require an explicit policy.

## Wrong groups can still produce plausible advantages {#grouping}

Two prompts have these rewards:

| Prompt | Rewards | Correct leave-one-out advantages |
| --- | --- | --- |
| Q1 | `[1, 0, 0.5]` | `[0.75, −0.75, 0]` |
| Q2 | `[0, 0, 0]` | `[0, 0, 0]` |

Suppose results arrive as `Q1-A, Q2-A, Q1-B, Q2-B, Q1-C, Q2-C`. A bare `reshape(2, 3)` produces groups `[1,0,0]` and `[0,0.5,0]`. Each group's advantages still sum to zero, and the tensor shape is correct. The comparisons are nevertheless between different prompts.

Use `prompt_id` to collect the planned responses from the same generation round and policy before computing statistics. Timeouts and scoring errors are not default zero rewards. Resampling, dropping an entire group, or retaining an incomplete group can change the sampling distribution; record the decision. Keeping only high-scoring responses does not preserve the original on-policy sample.

## Do not mix up probability roles {#probabilities}

| Probability | When it changes | Purpose |
| --- | --- | --- |
| Current policy log-probability | Changes with trainable parameters; retains gradients | Current update |
| Old log-probability | Fixed while the batch is reused | Denominator for the corresponding policy ratio |
| Reference log-probability | Belongs to the chosen reference; any refresh is explicit | Reference-relative regularization |

If the inference engine's sampling distribution differs from the learner's old policy, distinguish that behavior distribution too. Temperature, top-p, templates, and precision can all matter; matching weight names is insufficient. See [sampler/learner probability alignment](../../05-post-training/post-training-infrastructure.en.md#two-numerical-agreement-a-silent-crack).

Begin debugging with one scoring backend and fixed weights, inputs, and randomness: before an update, current/old ratios should be close to one. Compare the real generation engine separately, without assuming one universal tolerance across precisions and lengths. If ratios remain mechanically equal to one after an update, check whether old values were overwritten. They should not track every current forward pass.

Rewards, advantages, and old/reference scores are generally fixed targets, while current log-probabilities retain gradients. For one fresh on-policy RLOO reward update, the loss is `-(advantages.detach() * sequence_log_probs).mean()`. Repeated updates on old data require an explicit estimator and update rule rather than repeated use of this minimal expression.

## What does `.mean()` average? {#reduction}

Use two responses' **illustrative per-token losses** to check denominators: a short response `[2,4]` and a long response `[1,1,1,1]`. These are not measured RL losses and do not predict whether a model becomes more or less verbose.

| Reduction | Result | Short: token weight | Long: token weight |
| --- | ---: | ---: | ---: |
| Sum each sequence, then average responses | $(6+4)/2=5$ | $1/2$ | $1/2$ |
| Average each response's tokens first | $(6/2+4/4)/2=2$ | $1/4$ | $1/8$ |
| Average all valid tokens together | $(6+4)/6=5/3$ | $1/6$ | $1/6$ |
| Normalize by fixed length 4 | $(6+4)/(2\times4)=1.25$ | $1/8$ | $1/8$ |

Coefficients are $\partial L/\partial\ell_{i,t}$, **not parameter-gradient magnitudes**. Tokens' own gradients can differ greatly. The first row matches sequence REINFORCE's summation above; other rows may belong to different objectives. Do not substitute them merely to make a loss look smaller. Fixed-length normalization differs from the sequence sum by a constant here; dividing each response by its actual length changes relative weighting.

This table also helps distinguish training recipes. [Dr. GRPO, §3.2](https://arxiv.org/html/2503.20783v2#S3.SS2) removes per-response length and within-group standard-deviation normalization; its example implementation divides by a fixed generation budget. [DAPO, §3.3](https://arxiv.org/html/2503.14476v2#S3.SS3) reduces over the total valid-token count. For one fixed batch, the length denominators may differ only by a common scale. Across batches, however, the actual token count varies while the fixed budget need not. Calling both “token averaging” hides that difference, and the full algorithms differ in more than their denominators.

Keep the chosen denominator when splitting into microbatches or multiple devices. For a valid-token average, count tokens across the whole optimizer step rather than giving short and long microbatches equal weight. See [global denominators in distributed training](distributed-training.en.md) for compensation when gradients are averaged across devices.

## Run the checks before integrating a model {#checks}

From the repository root:

```bash
python3 practice/post-training/code/rl_batch_checks.py
python3 -m unittest discover -s site/tests -p 'test_rl_batch_audit.py'
```

The standard-library program checks leave-one-out advantages, ID-based grouping, token shifting, and four reductions. Tests also enumerate all groups for a small binary policy to verify the expected gradient. When PyTorch is installed, they additionally exercise the helper above and check how action masking affects gradients.

Before real training, add these comparisons:

- Extra padding leaves loss unchanged for the same valid tokens. Changing the prompt, tokenization, or actual actions is not such an invariance.
- Reordering records preserves advantages matched by `sample_id`; duplicates and missing samples are detected.
- Whole-batch and accumulated gradients agree under the same denominator; old/reference parameters receive no unintended gradients.
- Compare the original and updated models on fixed held-out questions. A reward curve is not independent evaluation.

Passing these checks makes later numerical, optimization, and distributed failures easier to isolate. It does not validate the reward or prove the updated model is better. Continue with [experiments and release](experiments-and-release.en.md), or [why RL is hard to train](../../05-post-training/deep-rl/why-rl-is-hard.en.md) when optimization remains unreliable.
