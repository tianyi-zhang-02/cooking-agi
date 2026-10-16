# Why RL remains hard to train: signal, cost, and stability

[中文](why-rl-is-hard.md) · **English**

> Reading time: about 13–17 minutes · Last reviewed: 2026-10 · Systems documentation checked: 2026-10-11

The training script runs, the GPUs stay busy, and reward barely improves. Change the sampling temperature and the curve looks completely different. In RL, all of those observations can coexist.

RL is neither new nor lacking a common framework. **Sample, collect feedback, estimate an update, change the policy** is a well-established pattern. The difficulty is that a configuration may stop working when the model, task, reward, or budget changes. A shared framework is not a portable recipe.

This article focuses on LLM post-training. Start with [what efficiency means](#efficiency), [account for a rollout batch](#rollout-budget), then read the [debugging sequence](#diagnosis). The [methods map](methods-map.en.md) separates the algorithms; the [RL timeline](../../learn/rl-timeline.en.md) traces their history. All budget and timing numbers here are teaching assumptions, not framework benchmarks.

## What makes this different from SFT? {#moving-data}

SFT on fixed data can be difficult too, but an update generally does not change the contents of the next batch. Online RL does: after updating the policy, the model reaches different states and generates different responses, then trains on that new data.

Suppose a coding model often fails to compile. Working on compilation failures is a sensible start. Once those mostly disappear, the remaining problems may involve boundary cases and hidden tests. Continuing to sample the same easy tasks costs money while yielding less feedback that distinguishes responses. Difficulty should track capability, but removing hard tasks just to make training reward look better is not progress.

<figure class="worked-update" aria-label="How an update changes the next round of data">
<ol>
<li><small>Current policy</small><strong>Generate code and run tests</strong><span>Record responses, sampling probabilities, failure types, and costs—not just a total score.</span></li>
<li><small>Current update</small><strong>Which attempts should become more likely?</strong><span>Labels can be noisy, advantages inaccurate, and updates too large.</span></li>
<li><small>Next rollout</small><strong>The errors change again</strong><span>Even with the same prompts, the response distribution changes. Recheck difficulty, coverage, and independent evaluation.</span></li>
</ol>
<figcaption>This loop allows learning beyond demonstrations. It also couples the training distribution, estimation errors, and systems costs.</figcaption>
</figure>

Data construction and evaluation matter enormously, but getting them right does not automatically stabilize RL. Data determines what gets practiced; feedback determines what gets rewarded. Exploration, credit assignment, and optimization determine whether the model can use that feedback. Training and inference systems determine whether the intended algorithm is actually being executed.

## Token efficiency needs a denominator {#efficiency}

“More token-efficient” can mean at least four different things. They are not interchangeable.

| What you want to save | What to compare | A common mistake |
| --- | --- | --- |
| Interaction samples (sample efficiency) | Interactions, attempts, or annotations needed to reach the same fixed evaluation target | Fewer optimizer steps need not mean fewer interactions |
| Training tokens | Tokens generated, scored, and trained on, separately, for the same quality target | Counting only loss-bearing tokens and omitting discarded rollouts |
| Tokens at use time | Total generated tokens, attempts, and tool calls under the same test set and quality requirement | Shorter individual answers may require more attempts to succeed |
| Compute and time | GPU-hours, cost, wall-clock time, and time to target quality | Higher tokens/s can accompany worse quality or greater total cost |

A token does not cost the same in prefill, sequential decode, reward-model inference, and backpropagation; how decode cost changes with batch size and the KV cache is in [performance: roofline and MFU](../../12-infra/performance.en.md#decode-batching). Token counts are not a direct FLOP comparison across models and lengths either.

A useful report fixes the evaluation tasks, sampling budget, and grader, then plots **quality versus cumulative rollout tokens** and **quality versus GPU-hours**. Report output length and success rate separately. This answers more than reward or GPU utilization alone.

## Account for a rollout batch {#rollout-budget}

Suppose a batch contains 64 prompts, with 8 responses each and exactly 2,048 generated tokens per response:

$$
N_{\mathrm{generated}}=64\times8\times2048=1{,}048{,}576.
$$

If filtering retains one quarter of the groups, training keeps 262,144 output tokens and excludes 786,432. The excluded tokens **have already been generated**, and may already have passed through a verifier. This count still excludes prompt prefill, scoring, reference forwards, backpropagation, and retries.

<figure class="rl-budget" aria-label="Teaching budget: one quarter of generated output is retained">
<div class="rl-budget__bar"><span class="rl-budget__kept">25%</span><span class="rl-budget__discarded">75%</span></div>
<figcaption><strong>Retained 262,144 · Excluded 786,432</strong><br>Color shows where tokens go, not gradient quality or model performance. Filtering poor data may still be worthwhile, but its generation cost belongs in the budget.</figcaption>
</figure>

With the same filtering rule, a fixed retention rate, and equal response lengths, retaining the original 64 groups requires sampling 256 groups on average. Real retention rates change with the model, and lengths vary; this factor is not a constant throughput prediction.

Why might so many groups be excluded? Tasks may be too hard, with every response failing, or too easy, with every response succeeding. Neither necessarily provides a within-group relative-reward signal. Keeping only successful responses is not the solution. [DAPO (2025), §3.2](https://arxiv.org/html/2503.14476v2) uses dynamic sampling to address such groups, at an additional sampling cost. See the [sparse-group example](llm-bridge.en.md#sparse-groups).

## Why more GPUs or more parallelism may not save money {#system-cost}

Start by accounting for a synchronous round stage by stage:

$$
T_{\mathrm{round}}=T_{\mathrm{rollout}}+T_{\mathrm{verify}}+T_{\mathrm{update}}+T_{\mathrm{sync}}.
$$

Suppose the stages take 60, 20, 30, and 10 seconds: 120 seconds total. With sufficiently independent resources and full overlap, the ideal steady-state pipeline lower bound is the slowest stage: 60 seconds per batch here. **The first batch does not finish in 60 seconds, and a twofold speedup is not guaranteed.** Queues, pipeline fill and drain, network contention, weight transfers, and additional GPUs must all be counted.

<figure class="worked-update" aria-label="Responsibilities of components in an LLM RL system">
<ol>
<li><small>Rollout · Inference</small><strong>Produce trajectories and behavior log-probs</strong><span>Engines such as vLLM / SGLang handle generation and caching. A worker must also track versions, sampling settings, and termination reasons.</span></li>
<li><small>Reward · Environment</small><strong>Return trustworthy feedback</strong><span>Test sandboxes, tools, or grading models. Distinguish timeouts from task failures and match feedback to the correct trajectory.</span></li>
<li><small>Update · Training</small><strong>Turn trajectories into an update</strong><span>A training backend handles forward/backward computation and communication. A controller coordinates batches, weight synchronization, checkpoints, and evaluation.</span></li>
</ol>
<figcaption>FSDP handles responsibilities such as sharding training state, not defining rewards. An inference engine does not choose an advantage estimator. A framework connects components; task semantics still need checking.</figcaption>
</figure>

Which layer FSDP, NCCL, and the other training-side pieces handle is mapped in [the framework layer's multi-GPU map](../../12-infra/frameworks.en.md#nccl-map).

Synchronization makes it easier to track which policy generated and trained on each batch, but it waits for slow requests. Asynchrony reduces waiting while allowing samples to lag behind the current policy. [AReaL (2025), §4–5](https://arxiv.org/html/2505.24298v2) decouples generation and training while controlling staleness and changing the update objective—not merely adding workers. Its [current documentation](https://areal-ai.io/docs/en/algorithms/async.html) likewise treats staleness and throughput as a tradeoff.

Another issue is subtler: identical weights need not produce identical probabilities in training and inference backends. Precision, kernels, and sampling transformations can matter. Distinguish actual behavior log-probabilities, training-side recomputed old log-probabilities, and current-policy log-probabilities. [verl’s rollout correction documentation](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) addresses such distribution differences. Importance weighting, clipping, and filtering have assumptions and variance/bias costs; none guarantees a repair for every mismatch.

## Stability takes more than lowering the learning rate {#stability}

| Observation | Possible bottleneck | What to check next |
| --- | --- | --- |
| Reward stays flat | Tasks too hard, sparse feedback, or constant grader scores | Success rates and within-group variance by task; read failures manually |
| Reward rises but independent evaluation does not | Grader exploitation, narrowed coverage, or a changed evaluation budget | Rerun at fixed budget and inspect held-out cases |
| Responses become nearly identical | Weak exploration, falling entropy, or rewards favoring one form | Compare repeated samples, entropy, and independent correctness |
| Responses keep growing | Genuine benefit, an accidental length reward, or broken termination | Evaluate by length bucket; inspect EOS, truncation, and tool loops |
| Loss or ratios suddenly spike | Version, log-probability, mask, denominator, or numerical mismatch | Hold a small batch fixed and compare quantities token by token |
| Asynchrony is faster but quality falls | Stale data or filtering that changes the task distribution | Compare by policy lag, response length, and task type |

These are hypotheses to test, not diagnoses from a curve alone. Longer responses, for example, may be a real gain from long chains of thought or just an accidental length reward; why length grows on its own in R1-Zero is worked through in [how long chains of thought are trained](../../11-frontier/test-time-compute/long-chains.en.md). A smaller learning rate may delay a failure without fixing rewards assigned to the wrong trajectory, mixed-up samples, or an incorrect denominator.

Recent methods target different parts of this picture. [DAPO §3](https://arxiv.org/html/2503.14476v2) discusses exploration, sampling, token-level reduction, and overlong responses. [GSPO (2025), §4–5](https://arxiv.org/html/2507.18071v1) changes sequence/token weighting and clipping, evaluating stability in its Qwen3 MoE setup. These are testable designs, not evidence of one best configuration for every model, reward, and budget. Continue with [DAPO](../dapo.en.md) and [policy ratios](../policy-ratios.en.md).

## Data configuration is part of the learning problem {#data-contract}

Even with the same code and algorithm name, changing any of these can change the experiment:

| What needs an explicit definition | Why it is not just plumbing |
| --- | --- |
| Prompt provenance, deduplication, task mix, and difficulty | Determine where attempts are spent; do not use the independent validation set to tune curriculum selection |
| Prompt / response / tool-observation masks | Identify policy actions and loss-bearing tokens |
| Verifier version, timeouts, retries, and partial credit | Define failure; infrastructure faults should not silently become model failures |
| Group IDs, sampling settings, policy versions, and behavior log-probs | Make comparisons and ratios traceable; mixed-version trajectories may require segment-level records |
| EOS, truncation, and context budgets | Define when attempts stop; a timeout need not be true environment termination |
| Token / sequence averaging and distributed global denominators | Determine the weights of long/short responses and samples on different ranks |

A checkpoint is more than weights. Restore the optimizer, scheduler, RNG, and data cursor, and specify how unfinished rollouts and queues are resumed or discarded. Restarting successfully does not mean continuing the same training trajectory. See the [post-training data pipeline](../../practice/post-training/data-pipeline.en.md#supervision-smoke-test) and [training frameworks](../../06-systems/training-frameworks.en.md).

## Some difficulties cannot be fixed by switching frameworks {#limits}

**Credit assignment in long tasks.** If the final tests fail, was the mistake in reading the problem, the third tool call, or the final output format? A terminal reward arrives at the end and gives coarse credit-assignment feedback; whether producing it is cheap still depends on the tests or grading model. Process rewards add detail while increasing annotation and verification difficulty. A Critic provides estimates and introduces its own errors. Longer reasoning text alone does not give more accurate credit assignment. For multi-turn agents, approaches such as GiGPO's grouping by state and turn-level critics are in [credit assignment for agent RL](../../11-frontier/agentic-rl/credit-and-training.en.md).

**The cost of exploration.** If the initial policy never produces a useful attempt, even a perfect verifier keeps reporting failure. Demonstrations, curricula, or stronger exploration can help, but they change the data, budget, and prior information and belong in the comparison. RL is neither necessarily limited to rearranging existing answers nor guaranteed to create new capabilities in any base model given a reward.

**The gap between feedback and real requirements.** Passing a test suite does not cover every use case. Open-ended answers are harder still to score completely. Models encounter situations the evaluator did not anticipate. Using the same judge to generate training data, assign reward, and perform final evaluation can turn shared blind spots into apparent progress. Separate training feedback from independent evaluation; see [data as supervision](../../01-data-and-feedback/data-as-supervision.en.md).

The practical response is to isolate failures and run controlled comparisons, not declare either a universal RL recipe or the uselessness of RL. This is a review of public mechanisms and engineering limits, not a complete survey of every lab’s latest training system.

## A practical debugging order {#diagnosis}

1. **Define improvement first.** Hold out independent data and fix generation budgets and grading. Keep the original model and simple SFT / DPO baselines; record differences in their data and compute costs.
2. **Check correctness on a few fixed trajectories.** Calculate rewards, masks, advantages, ratios, and denominators by hand. Verify the direction of one update and that recovery does not duplicate rewards or misassign trajectories.
3. **Run a small synchronous baseline.** Read actual attempts and error slices, not only aggregate reward. Use multiple seeds to check whether a result survives randomness.
4. **Change one class of issue at a time.** Compare group size, difficulty, rewards, and update methods separately. Track quality at the same evaluation budget alongside cumulative costs.
5. **Scale parallelism after establishing correctness.** Adding asynchrony, lower precision, or a new backend requires rechecking log-probs, sample distributions, and final quality—not just speed.

When quality stalls, a more complex loss is not automatically the next step. Better tests or a different difficulty mix may matter more. When the signal is clear but training still collapses, investigate estimation, updates, and systems behavior. **The difficulty is their interaction, not a permanent ranking of which component matters most.**

Run `python3 05-post-training/deep-rl/code/rl_tradeoffs.py` to check the budget and objective examples. These local calculations do not simulate full training or establish faster convergence for any method.
