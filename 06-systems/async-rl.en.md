# Async RL and AReaL: less waiting, older data

[中文](async-rl.md) · **English**

Some tasks finish in five seconds; others spend a minute using tools. If training waits for the whole batch, completed answers sit waiting for the slowest task. Training GPUs may have no work to do either.

Async RL lets generation and training progress together: rollout workers keep collecting answers, while the learner updates parameters once an eligible batch is ready. For example, complete groups can enter training while other groups continue generating. An objective that depends on a full group still needs its members. Removing a batch-wide wait does not remove the objective's data requirements.

Overlap also introduces a question: **what if training has moved on while an answer from an older model is still arriving?** The gap between those policies is policy lag. To judge whether async execution helps, connect the waiting time saved to the usable data retained and the resulting learning quality.

Read [one RL update](rl-systems.en.md), then ask: **which wait disappears, and which policy produced the training data?**

The version traces and queue numbers below are hypothetical; this note does not report an async training run or a reproduction of a paper's speedup. Sources were checked on 2026-10-30. The separate [async policy-learning note](../05-post-training/async-policy-learning.en.md) covers SAO, whose objective should not be substituted for AReaL's.

## Four meanings of async {#levels}

| Meaning | Work that may overlap | Does not automatically imply |
| --- | --- | --- |
| Async HTTP / API | Client handles other requests while awaiting results | Learner overlaps rollout |
| Engine async / overlap scheduling | CPU schedules ahead while GPU executes | Model parameters are being updated |
| Concurrent rollout / environments | Multiple prompts, tools or environments run | Training consumes older-version data |
| Rollout–learner overlap | Generation and parameter updates progress together | No policy lag, or an unchanged learning objective |

An `async_engine=true` setting alone does not establish an async RL loop. Inspect its timeline, weight publication and training entry point.

For example, a client might send four requests concurrently and receive results asynchronously, yet wait for all four before training and generate nothing during training. Requests overlap, but rollout and learning do not. The rest of this note concerns the final row: other rollouts keep progressing while the learner updates parameters.

## A version trace makes age visible {#trace}

### Follow one slow task

Label successive published policies v10, v11 and v12. Task A starts on v10. While it is still generating, the learner uses other data to reach v12. A's answer is not necessarily wrong when it arrives, but its choices came from v10, not v12.

Here learner v12 means the parameters currently used to compute loss, and age is $a=v_{\rm learner}-v_{\rm behavior}$. These are teaching conventions, not universal framework step numbering. Add hypothetical times: enough other eligible data are available for two updates and weight synchronizations (refits), completing at seconds 20 and 45. The learner remains at v12 from seconds 45–65. These times do not estimate any framework's performance.

| Event | Learner | Generation | Queue / training decision |
| --- | --- | --- | --- |
| Second 0: start | v10 | Long task A starts on v10 | A is incomplete |
| Second 20: consume other eligible data, update and refit | v11 | B starts on v11; A's continuation depends on the protocol | B has behavior v11 |
| Second 45: another update and refit completes | v12 | C starts on v12 | A and B are still running |
| Second 60: A finishes; apply hypothetical age limit 1 | v12 | A delivers an answer with behavior v10 | Age 2; A is ineligible |
| Second 65: B finishes; retain age limit 1 | v12 | B delivers an answer with behavior v11 | Age 1; other eligibility checks remain |

A is assumed to use v10 throughout, and B uses v11 throughout. Each refit applies within its own publication scope. To realize this trace, older versions must remain available to these tasks while other tasks use newer ones. Alternatively, a worker can finish A before replacing its weights. If every worker has to wait for A, however, the overlap in the table no longer applies.

A runs for 60 seconds but has policy age 2; B runs for 45 seconds and has age 1. Wall-clock time measures elapsed waiting, while version difference counts how far the learner has advanced. If the learner never updates, even a long-running task can retain age 0.

Another option is to pause A and resume it with new weights. That is the partial-rollout case below: different parts of one answer come from different versions, so a single v10 label is no longer sufficient.

### Passing the age limit is only one check

Even if B passes the age check at second 65, it cannot train under a full-group objective while another group member is missing. Task completion, groups, masks, duplicates, rewards and consumption rules must all meet the training entry requirements. Integer age also does not uniquely determine distribution drift: one large update may differ more than several small ones.

## Queues affect throughput and data selection {#queue}

### Faster generation can simply mean a longer queue

Suppose generation delivers 50 training units/s and learning consumes 40. For this calculation, keep both rates constant and assume no throttling or dropping:

| Quantity | Calculation | Meaning |
| --- | --- | --- |
| Added backlog in one minute | `(50 - 40) × 60 = 600` | The queue grows by 600 units per minute |
| Wait with 400 units ahead | `400 / 40 = 10` seconds | Queueing only, before this unit's own training |

A unit might be a complete group, but use the same unit throughout. A rate in groups/s cannot be subtracted from tokens/s. The second row also assumes first-in, first-out service without higher-priority arrivals jumping the queue.

Queueing can also increase policy age. Suppose v10 data enter the queue while the learner is at v11. If the learner completes exactly two more version updates during that 10-second wait, consumption happens at v13 and age rises from 1 to 3. With the same hypothetical limit of 1, an enqueue-only filter misses the change; eligibility needs reevaluation at the consumption point required by the objective. Those two updates are an additional assumption, not a consequence of waiting 10 seconds.

Increasing generation concurrency need not improve training throughput here. Backpressure can slow production when the learner falls behind; other choices include reallocating resources or dropping data under an explicit policy. Capacity limits how much the queue stores, age limits how far data may lag, and timeouts limit task or waiting duration. A larger buffer can hold more backlog but cannot raise consumption above 40 units/s. Output lengths, tool latency and learning also affect production rates, so inspect throughput by length and record sample-age distributions.

### Dropping slow tasks also changes the training data

If long tasks expire more often, retained data favor fast tasks. Suppose two short answers to a question finish while two longer answers are still using tools. Training on just the first two lets completion speed influence sample selection.

Reuse the [four-answer exercise](rl-systems.en.md#exercise): R1 and R3 finish at seconds 5 and 6 with reward 1; R2 and R4 finish at seconds 20 and 60 with reward 0. The full group has rewards `[1,0,1,0]`. With mean subtraction alone, R1 and R3 each receive advantage 0.5. Treating the first two completions as a separate group at second 6 changes the mean to 1 and both centered values to 0. Starting early changes the learning signal in this example, not just the schedule.

Importance ratios address differences in the probabilities policies assign to actions. They do not automatically correct this completion-order bias. When GRPO advantages depend on a group of answers to the same question, first-completing samples are not the complete group.

## Distinguish behavior, old and current {#objective}

### Start with the probability of one token

For exactly the same context, suppose a token had probability 0.2 during generation and probability 0.3 under the current training policy. The current-to-behavior ratio is `0.3 / 0.2 = 1.5`.

Now suppose the generation probability was not retained. Running a different old checkpoint gives 0.25; substituting it produces `0.3 / 0.25 = 1.2`. That ratio compares a different pair of distributions. It does not compare the current policy with the policy that actually generated the data.

A ratio of 1.5 only says the current policy assigns this action higher probability; it does not say the action deserves encouragement. Its effect on the update also depends on the advantage and objective. Rewards and a baseline provide a learning direction, while probability ratios account for which policy produced the data used to estimate the update.

Keep three roles separate:

| Name | Meaning here |
| --- | --- |
| behavior | The policy that actually produced the action, with sampling settings accounted for |
| old | A frozen policy snapshot used by a training objective; not necessarily the behavior policy |
| current | The policy currently being optimized |

The old policy here is also distinct from the reference model used for a KL penalty in RLHF. Training-side logprob recomputation can be a legitimate algorithmic step without recovering missing generation records.

<details markdown="1">
<summary>Expand: how the two probability ratios fit together</summary>

Let $\mu$ be the distribution that actually produced the action, $\pi_{\rm old}$ a frozen training policy snapshot, and $\pi_\theta$ the current policy. At the same context with nonzero denominators:

$$
\frac{\pi_\theta(a\mid h)}{\mu(a\mid h)}
=\frac{\pi_\theta(a\mid h)}{\pi_{\rm old}(a\mid h)}
\frac{\pi_{\rm old}(a\mid h)}{\mu(a\mid h)}.
$$

The first factor compares current with old; the second compares old with behavior. Their product recovers the current-to-behavior ratio.

Using the earlier numbers, the first factor is `0.3 / 0.25 = 1.2` and the second is `0.25 / 0.2 = 1.25`, giving `1.2 × 1.25 = 1.5`. As an arithmetic example, clamping each factor to `[0.8,1.2]` and multiplying gives 1.44. Multiplying first and clamping 1.5 to that interval gives 1.2 instead. This compares clipping orders; it does not yet define PPO's complete surrogate objective.

This is **not a complete decoupled PPO loss**. An implementation must also define its surrogate objective, advantages, stop-gradients, clipping, token/sequence aggregation and weighting. Clipping each factor separately is generally different from clipping their product. PPO clipping does not guarantee that arbitrary old data are safe to use.

</details>

### Identical weight versions need not produce identical sampling distributions

Suppose a toy vocabulary has three tokens with model probabilities `[0.5,0.3,0.2]`. Retaining only the top two and renormalizing gives sampling probabilities `[0.625,0.375,0]`. If the second token is sampled, the original-model-to-behavior ratio is `0.3 / 0.375 = 0.8`, not 1, even though the weights have not changed. The third token originally had nonzero probability but now cannot be sampled.

Temperature, top-p, quantization and kernels can also affect probabilities. State whether logged probabilities describe the distribution from raw logits or the distribution after sampling transformations. A weight-version label alone cannot distinguish the two distributions above.

If sampling assigns zero probability to actions the target distribution still allows, the sampled actions alone cannot provide the usual unbiased importance correction. This is a support problem, not something a smaller queue-age limit fixes.

## Where AReaL fits, and how limits differ {#frameworks}

AReaL separates generation and training so they can overlap. It must also control how far data may lag and how training uses those data. The original [systems paper, first released in 2025](https://arxiv.org/abs/2505.24298), considers both scheduling and objective changes. The examples explain why these belong together: async calls let tasks progress, age controls determine which data remain, and the objective determines how those data affect updates.

Use the comparison below when reading an implementation. These settings all concern staleness, but their units and enforcement points differ.

<details markdown="1">
<summary>Expand: version and configuration differences across AReaL, NeMo RL and verl</summary>

| Path and source | What is controlled | Distinction |
| --- | --- | --- |
| [Pinned AReaL async guide](https://github.com/areal-project/AReaL/blob/d25f2929b2a1e962e9d366717fbf872ab46a4997/docs/en/algorithms/async.md) | `rollout.max_head_offpolicyness` controls allowed policy lag; the guide defines 0 as synchronous | Not seconds or a portable cross-framework parameter |
| [Pinned NeMo RL legacy async GRPO](https://github.com/NVIDIA-NeMo/RL/blob/dd2e251da8ea1b4de6e76394b543e89d1e8cbe9e/docs/guides/async-grpo.md) | `max_trajectory_age_steps`, generation/target versions and buffer age filtering | Separate from beta SC sampler settings |
| [verl fully async recipe](https://verl.readthedocs.io/en/latest/advance/fully_async.html) | Rollout, queue, trainer and synchronization; `staleness_threshold` controls the documented stale-sample allowance | Not a universal count of N old policy versions |

The pinned AReaL guide requires `actor.recompute_logprobs` when enabling `actor.use_decoupled_loss`, and warns that the objective can conflict with other algorithm configurations. A configuration snippet is therefore not a universal recommendation.

The NeMo RL row describes its legacy async path, not the beta Single-Controller path. The verl link points to the latest documentation read for this note, not an immutable implementation. Recheck the configuration and source when upgrading.

</details>

A [2026 follow-up report](https://arxiv.org/abs/2607.01120) proposes an agent-oriented online-learning architecture, including trajectory protocols, a data proxy and update control. This later design discussion, the original paper and the current repository answer different questions. A speedup reported in one paper is not a guarantee for every current model and configuration.

## Changing weights midway through an answer {#partial-rollout}

### Keeping an answer's prefix does not change which model generated it

Suppose v10 generates the first 20 answer tokens. The system pauses generation, changes weights, then uses v11 to generate 30 more. It avoids discarding and regenerating the first 20 tokens, but the answer now combines two policies:

| Answer positions | Generating version | Records to retain |
| --- | --- | --- |
| Tokens 1–20 | v10 | Generation-time logprobs and action masks for these positions |
| Tokens 21–50 | v11 | Logprobs under the new version and the resumption boundary |

A single `behavior_version=v11` loses the first segment's origin. Retain token-aligned behavior logprobs and version boundaries, or equivalent traceability. If tool output appears between model actions, those positions come from the environment, not from policy choices.

For example, at learner v12 the first 20 tokens have version age 2 and the remaining 30 have age 1. Labeling the whole answer “age 1” hides the earlier segment's lag. The chosen algorithm must define trajectory eligibility, segment handling, advantages and loss; the version used at the last resumption cannot decide these for it.

### What happens to the old KV after a weight update?

Old KV was computed from the prefix using old weights. Retaining the first 20 tokens does not validate their v10 KV under v11. Resumption requires the implementation's prefix-recomputation or cache-invalidation protocol, which also has a cost.

During prefix recomputation, v11 reads the 20 existing tokens to build the state needed for subsequent generation. It does not choose those first 20 actions again or replace their original behavior logprobs with v11 probabilities. Cache reconstruction and behavior records therefore need separate handling.

Pause, transfer and resume are control operations. Also check that content, positions, masks and versions remain aligned. Partial rollout can reduce waiting and repeated generation without turning these 50 tokens into single-policy data.

## Evidence for useful acceleration {#evaluation}

### Measure speed, usable data and learning together

| Comparison | Record together |
| --- | --- |
| System efficiency | Fixed total GPU budget, topology, stage allocation, elapsed time, GPU-hours |
| Effective data | Completion, timeout, drop and group-completeness rates; lengths and ages |
| Learning changes | Updates, loss/ratio/clip metrics, fixed evaluation and time to target quality |
| Consistency and recovery | Behavior records, refit acknowledgements, KV handling, consumption, checkpoint |

Consider a hypothetical count before drawing any conclusion about model quality. With the same GPU budget, one-minute window and group definition, setup A produces 100 groups, of which 90 are complete and within the age limit. Setup B produces 120, but only 72 qualify. Assume the learner can consume every eligible group. Raw production increases by 20%, while usable groups fall from 90 to 72. Even if eligible counts also rise, learning still needs fixed evaluation: more updates or a higher training-reward curve cannot by themselves replace a comparison of time to the same target quality.

### An ideal bound is not a measured speedup

Let $T_R$ be the generation time to prepare one eligible batch and $T_L$ the time to train on such a batch. With fixed stage allocations, constant batch times, and no fill/drain or communication overhead, an ideal pipeline period is limited by $\max(T_R,T_L)$, versus $T_R+T_L$ for serial execution.

Suppose the stages take 60 and 25 seconds. Serial processing takes 85 seconds per batch. With separate stage resources and overlap, the first batch still traverses both stages, but steady-state completion can reach one batch every 60 seconds: an ideal throughput improvement of about `85 / 60 = 1.42` times.

Expand those assumptions into three batches. Each cell shows “generation interval → training interval” in seconds. Stage allocations stay fixed, and communication and publication overhead remain excluded.

| Batch | Serial | Overlapping generation and training |
| --- | --- | --- |
| Batch 1 | 0–60 → 60–85 | 0–60 → 60–85 |
| Batch 2 | 85–145 → 145–170 | 60–120 → 120–145 |
| Batch 3 | 170–230 → 230–255 | 120–180 → 180–205 |

The first batch finishes at second 85 in either case; overlap does not shorten its passage through both stages. Only subsequent completion intervals shrink from 85 to 60 seconds. For these three batches alone, the elapsed-time ratio is `255 / 205 ≈ 1.24`, below the steady-state 1.42. Fill and drain matter particularly for short runs.

This schedule also depends on data eligibility. Start at v10, publish the next version after each training batch, and pin each generation batch to its starting version. In the overlapping schedule, batch 2 starts on v10 at second 60 and is still generating when v11 is published at second 85. It enters training at second 120 with learner v11. The schedule therefore requires an objective that accepts these older-version data and a system that retains the version needed for generation. Requiring every batch to wait for the preceding update before sampling restores the serial dependency.

There are no free GPUs in this comparison. If the original phases each used all GPUs in turn, splitting that pool between stages requires measuring both times again. Higher output tokens/s may also accompany fewer usable samples or lower evaluation quality. Compare time and resources to the same target quality, not just raw output rate.

First recompute the trace, 600-unit backlog and probability ratios. Label each batch in the three-batch timeline with its behavior and training-time versions, then write pause/resume and duplicate-delivery state machines. Next compare synchronous and bounded async tiny-model runs. Keep unperformed convergence, long-task and fault experiments explicitly planned. This makes the conclusion specific: which wait was removed, which older data were used, and whether the same learning target was reached sooner.

Continue: [SAO's algorithmic differences](../05-post-training/async-policy-learning.en.md), [post-training sample contracts](../05-post-training/post-training-infrastructure.en.md), [GPU and cluster scheduling](execution-boundaries.en.md).
