# The LLM bridge: which RL intuitions transfer?

[中文](llm-bridge.md) · **English**

> Reading time: ~14 min · Last reviewed: 2026-10

What changes when a model goes from answering one question to repeatedly editing code and running tests? Both can be trained with RL, but an action, an episode ending, and a useful improvement may mean different things.

We will start with an update over 3 tokens, then look at group comparisons, Critics, and context compaction in longer tasks. For the main tradeoffs, read the [group-reward example](#sparse-groups) and [long-task comparison](#long-horizon-choice). To work through the numbers, continue to [3-step GAE](#credit-example).

## Start with a mapping

| Classical RL | One text-generation counterpart | Important caveat |
| --- | --- | --- |
| State / observation | Prompt and current context | Hidden environmental state may be absent from text |
| Action | A token or a tool call | These granularities have different durations and costs |
| Policy | Autoregressive model | Log probabilities under the actual sampling distribution |
| Reward | Preference, tests, task outcomes | Optimizable does not mean aligned with the user goal |
| Episode | One answer or a multi-turn task | EOS, timeout, and tool failure are different boundaries |

Appending a token is nearly deterministic in ordinary generation, but sampling and reward need not be. Tool agents receive observations from an external world with delays, permissions, and failures.

## What PPO and GRPO estimate

PPO estimates advantages from rollout-policy trajectories, then uses probability ratios and clipping to stop rewarding excessive changes in certain directions; it does not hard-bound those ratios. A reference-model KL penalty is different: it penalizes drift from a fixed anchor, while PPO clipping modifies the local optimization objective for the collected batch.

GRPO commonly uses relative rewards among several responses to one prompt to construct group-based advantages, reducing reliance on a separate value Critic. Identical group rewards can yield zero relative signal after normalization. Grouping and normalization also affect prompt weighting. They do not automatically fix sparse rewards, faulty verifiers, or cross-task comparability.

If advantages and probability ratios are still unfamiliar, start with [PPO clipping](../rlhf/ppo-clipping.en.md). For a comparison of the data PPO, GRPO, and DPO use, see [the method overview](../rlhf/after-rlhf.en.md).

## Follow three tokens through an update

Suppose a prompt produces three tokens, ending in EOS, and the completed answer earns reward 1. The process is readable without playing an animation:

<div class="drl-flow drl-sequence" aria-label="Text generation and reward attribution">
<span>Prompt<small>Not this rollout’s action</small></span><b>→</b><span>Tokens 1, 2<small>Save per-step log-prob</small></span><b>→</b><span>EOS<small>Score the full answer</small></span><b>→</b><span>Update<small>Credit valid actions</small></span>
</div>

For the sampled tokens, suppose old conditional probabilities are $[0.5,0.25,0.5]$ and new probabilities are $[0.6,0.2,0.6]$. Token ratios are $[1.2,0.8,1.2]$, while the sequence ratio is their product, 1.152. Token-level PPO must not interchange them; long-sequence products have different scale and variance.

A terminal score can provide learning signals to earlier actions. That does not mean each reasoning step was independently verified. Process rewards add intermediate feedback but can introduce their own bias; denser feedback is not automatically more accurate.

## When does GRPO’s within-group signal disappear?

Take four responses to the same prompt with rewards $[0,0,1,1]$. Under the population-standard-deviation convention, mean and standard deviation are both 0.5, giving normalized advantages $[-1,-1,1,1]$. Responses above the group mean are encouraged; those below are discouraged.

If all four rewards are 0, centered rewards are zero. An $\epsilon$ denominator prevents division by zero but creates no task-reward gradient. All-one groups have the same issue. Other terms, such as KL regularization, may still produce gradients, so this does not mean the entire update is zero. Variance conventions, whether standard deviation is used, and token-versus-sequence averaging also change updates.

### What does sampling 8 times actually buy? {#sparse-groups}

Make a small, explicit assumption: for one task, independent attempts each succeed with probability $p$. Success earns 1 and failure earns 0. A group of $G$ attempts has nonzero reward variation only if it contains both a success and a failure.

$$
\begin{gathered}
P_{\mathrm{mix}}=1-p^G-(1-p)^G.\\
G=8,\quad p=0.02\quad\Rightarrow\quad 14.9\%.
\end{gathered}
$$

Here $P_{\mathrm{mix}}$ is the probability that a group contains both successes and failures. Subtract the two uniform outcomes—all successes and all failures—from 1. Holding the group size at 8 gives the following comparison.

<figure class="worked-update" lang="en" id="group-signal-comparison">
<figcaption>Eight attempts per group, different task difficulties. These numbers assume independent sampling and binary rewards.</figcaption>
<ol>
<li><small>2% success probability</small><strong>14.9% of groups vary</strong><span>Most groups contain 8 failures. More sampling costs resources without necessarily providing relative reward information.</span></li>
<li><small>20% success probability</small><strong>83.2% of groups vary</strong><span>Successes and failures are more likely to appear together, giving the group comparison something to work with.</span></li>
<li><small>98% success probability</small><strong>14.9% of groups vary</strong><span>Now most groups are all successful. Little relative signal need not mean the model cannot solve the task.</span></li>
</ol>
</figure>

**This is not a prediction of GRPO performance.** Tasks differ, samples can be correlated, and rewards need not be binary. We calculated the probability of reward variation—not gradient quality or accuracy after training. Before increasing group size, inspect all-zero and all-one groups separately. Depending on the result, changing task difficulty or adding trustworthy intermediate feedback may be more useful.

<details markdown="1">
<summary>Check the probabilities in Python; no model or GPU needed</summary>

```python
def mixed_group_probability(success_probability, group_size):
    if not 0 <= success_probability <= 1:
        raise ValueError("success_probability must be in [0, 1]")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    return 1 - success_probability ** group_size - (1 - success_probability) ** group_size

for success_probability in [0.02, 0.2, 0.98]:
    probability = mixed_group_probability(success_probability, 8)
    print(f"{success_probability:.0%}: {probability:.1%}")
```

</details>

| Choice | Benefit | Cost |
| --- | --- | --- |
| Learned Value Critic | Cross-state return estimates and temporal attribution | Another training path with estimation error |
| Within-group relative rewards | Avoids training the same separate Value Critic | Multiple generations per prompt; weak signal without reward variation |
| Executable reward checks | Direct feedback on verifiable constraints | Limited to what the verifier checks |

These are not mutually exclusive choices on one axis. Verifiable rewards work with different policy optimizers. GRPO cannot repair a verifier that rewards incorrect answers.

## Old policy, reference model, and reward model are different roles

In the simplest synchronous PPO setup, the old policy generated this batch of actions, and its saved log-probabilities stay fixed during the batch's updates. That assumption needs checking in asynchronous training: the actual behavior policy may be older, while temperature and inference-backend differences can change sampling probabilities. Record the behavior probabilities separately from the old-policy anchor used for the update; see [where rollout data comes from](../post-training-infrastructure.en.md).

A reference model has a different job. It is usually a steadier anchor for constraints such as KL—not a reference answer, and not necessarily the old policy. A reward model or verifier scores the result. These roles need separate accounting even when they use similar model architectures.

Preserve prompt/response boundaries, EOS, padding, sampling temperature, and policy versions. Otherwise action positions and probability ratios become ambiguous. Tool outputs are observations, not tokens sampled by the Actor. Concatenating them into one sequence does not turn them into policy actions.

These examples explain shared mechanisms, not a universal PPO/GRPO loss. Compare a specific implementation with [DeepSeekMath’s GRPO design](https://arxiv.org/abs/2402.03300) and the [RLHF model roles](../rlhf/three-stages.en.md).

## Long tasks: what memory changes

Two kinds of length are worth separating. **Long context** concerns how much input the model can read at once. **Long horizon** concerns how many decisions it must make and how long their consequences take to appear. Reading a large report and answering once can require substantial context without repeated interaction. Debugging for dozens of turns, seeing only a few logs each time, can involve a long decision chain with a modest context. If a step means one token, the two are related again. State whether your experiment counts tokens, tool calls, or interaction turns.

As history grows, something has to be retained or discarded. A summary that says “tests failed” but drops “already tried approach A; the error is at the input boundary” may send the agent back down the same dead end. The missing ingredient is decision-relevant information, not another reward. Memory can help, but a summary need not contain the full environment state or restore the Markov property.

Task failure, service timeout, cancellation, and budget exhaustion can all stop a rollout. Decide how to record, resume, and score each one before increasing concurrency. Otherwise a faster collector may just produce incorrectly labeled data faster.

## Does a long trajectory require replacing GRPO with PPO?

<span id="long-horizon-choice"></span>

Not necessarily. One attempt fixes a bug in 4 steps; another still fails after 10. Their completed outcomes remain comparable. **GRPO does not require matching fifth actions or equal trajectory lengths.** The harder questions are whether repeated complete attempts are affordable and whether final success or failure provides enough learning signal.

[The official GLM-5.2 announcement](https://z.ai/blog/glm-5.2) describes one concrete choice: critic-based PPO for training on variable numbers and lengths of compacted sub-traces, with a token-level loss. Published on June 16, 2026, this is a team's implementation decision—not an industry-wide finding that GRPO is obsolete, or a controlled experiment establishing that PPO always wins.

Whole attempts and training fragments are also different units. Splitting one attempt into 3 fragments and another into 6 does not make the 9 fragments independent responses to the same prompt: they have different starting contexts and future returns. But when the complete task identity and outcome are retained, fragmentation alone does not make group-relative methods mathematically invalid.

### Should every step in a successful attempt be encouraged? {#credit-example}

Consider a repair task with 3 decisions and rewards $[0,0,1]$: tests pass at the end. Outcome-only GRPO assigns the trajectory's actions the same group-relative advantage. Their token gradients need not be identical; they share a signal about how this attempt compares with the group average.

A Critic estimates something more local: **given where we are now, how much return should we expect next?** The values below are made up for calculation, not model measurements. Set $\gamma=1$ and $\lambda=0.5$, with zero value beyond the terminal state.

<figure class="worked-update" lang="en" id="credit-assignment-example">
<figcaption>Start at step 3, then carry its result back to steps 2 and 1. One successful attempt can have different local estimates.</figcaption>
<ol>
<li><small>Step 1 · Try an initial edit</small><strong>δ₁ = 0 + 0.3 − 0.9 = −0.6</strong><span>Â₁ = −0.6 + 0.5 × 0.6 = −0.3. The next state's estimated value is lower.</span></li>
<li><small>Step 2 · Recheck the failing test</small><strong>δ₂ = 0 + 0.8 − 0.3 = 0.5</strong><span>Â₂ = 0.5 + 0.5 × 0.2 = 0.6. Some of the later signal is carried back.</span></li>
<li><small>Step 3 · Fix it and pass</small><strong>δ₃ = 1 + 0 − 0.8 = 0.2</strong><span>Â₃ = 0.2. The task really ends, so there is no future value to add.</span></li>
</ol>
</figure>

The negative value does not establish that the first edit was harmful. Perhaps the initial estimate of 0.9 was simply too optimistic. A Critic provides a statistical estimate, not a proof of each action's causal contribution. Change $\lambda$ to 1 in this same example and complete returns minus state values give $[0.1,0.7,0.2]$: the first advantage becomes positive. The estimator matters too.

<details markdown="1">
<summary>Equations and code: why calculate backwards?</summary>

Let $d_t$ indicate true task termination and $V_t$ estimate the current state's value. Write $B_t$ for the future value to bootstrap from; it is zero at a true terminal. Then calculate the TD residual:

$$
\begin{gathered}
B_t=(1-d_t)V_{t+1},\\
\delta_t=r_t+\gamma B_t-V_t.
\end{gathered}
$$

Within a continuing trajectory, the GAE recurrence is:

$$
\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}.
$$

Calculate the recurrence backwards from the fragment's end, stopping it at true termination, resets, or fragment boundaries that cannot be continued. Smaller $\lambda$ relies more on local value estimates. On a complete terminated trajectory, $\lambda=1$ gives the observed discounted return minus the baseline. See [the GAE paper, §3](https://arxiv.org/html/1506.02438v6#S3) and [the implementation walkthrough](actor-critic-gae.en.md).

This snippet uses `gae` from [rl_checks.py](code/rl_checks.py). Run it from the repository's `05-post-training/deep-rl/code/` directory. Both language versions use the same code.

```python
from rl_checks import gae

rewards = [0, 0, 1]
values = [0.9, 0.3, 0.8]
next_values = [0.3, 0.8, 0]
terminated = [False, False, True]
truncated = [False, False, False]

for trace_decay in [0.5, 1.0]:
    advantages = gae(rewards, values, next_values, terminated, truncated,
                     gamma=1.0, trace_decay=trace_decay)
    print(trace_decay, [round(advantage, 3) for advantage in advantages])
```

Here each step represents a decision. A language-model trainer also needs token-level bookkeeping, action masks, KL terms, and loss reduction. This is not a complete trainer.

</details>

That comparison concerns outcome-only rewards. [DeepSeekMath §4.1.3](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS3) also discusses process-supervised GRPO. No Critic does not mean no intermediate feedback; the quality of that feedback still needs testing.

### Compaction is not task termination {#segment-boundary}

Suppose the last step of a fragment has reward 0, current value 0.6, and continuation value 0.8, with $\gamma=0.9$. If the task continues, the one-step target is $0+0.9\times0.8=0.72$, giving residual 0.12. Mark it terminal by mistake and the target becomes 0, giving residual −0.6. One incorrect boundary flag changes the signal's sign.

Keep the context, environment state, and next-state value actually used to continue the task. Do not bootstrap from a fresh task's reset state. A summary changes the observation; it does not automatically end the task. Continuing GAE across fragments also requires genuinely contiguous, aligned data—not just concatenating files. [Gymnasium's time-limit guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) explains termination versus external truncation.

Segmentation can also change weights. Suppose two tasks contribute 2 and 6 valid action positions, with per-position losses of 1 and 3 respectively. Averaging within each task and then across tasks gives 2. Averaging all 8 positions gives 2.5. The latter weights the longer task more; it does not automatically eliminate length bias. Whichever objective you choose, preserve its intended denominator through segmentation, padding, and distributed aggregation.

### How would you compare the designs? {#compare-designs}

| Situation | Design to consider | New risk |
| --- | --- | --- |
| Several complete outcomes per task are affordable | Group-relative outcome advantage | Equal rewards, expensive rollouts, coarse attribution |
| Long trajectories need intermediate estimates | Critic with TD / GAE | Value bias, extra training and memory |
| A training segment ends but the task continues | Bootstrap from the next state, or process completed returns later | Don't label a segment boundary as a failed terminal |
| History is compacted | Preserve the resulting observation and policy version | Summaries may discard relevant state |

I would first fix the task set, initial model, tool permissions, verifier, and evaluation-time sampling budget. Matching update counts is not enough: sampling 8 attempts per task for one method and 1 for the other changes generation cost, while adding a Critic changes training cost. Report complete rollouts, generated tokens, tool time, and training resources alongside the loss.

Then inspect two kinds of evidence. During training, how often are group rewards uniform, how well does the Critic predict completed returns it was not fitted on, and does segmentation change weights? At evaluation, under the same test budget, does task success improve, can the model recover from tool failures, and does the improvement extend beyond familiar repositories or templates? A different estimator cannot rescue a broken task or verifier.

Audit cheating separately. Editing the tests or reading protected answers should not count as solving a coding task. GLM-5.2's public description also treats anti-hacking as part of long-task training. Our examples check calculations only; they do not train a model or establish a winning algorithm.

## Multiple agents make the environment nonstationary

If other agents learn too, your transition and reward distributions change with them. Centralized training with decentralized execution can give a Critic joint information during training while each Actor uses only permitted local information at execution.

This is an entry point, not a complete MARL course. Communication, game objectives, nonstationarity, and cooperative credit assignment deserve separate treatment. Reference: [MADDPG](https://arxiv.org/abs/1706.02275).

## What evidence is missing when reward rises?

Check independent task success, length and cost, unseen scenarios, recovery from tool failure, and reward hacking. Using the same judge for training rewards and final scores does not provide independent validation.

Next, use the [evaluation stack](../../07-evaluation/evaluation-stack.en.md) to record task outcomes, failures, and costs separately. If model-based scoring is needed, continue to [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md). A higher reward is encouraging, but we still need to know what the model has actually become better at.
