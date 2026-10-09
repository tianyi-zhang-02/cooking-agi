# The LLM bridge: which RL intuitions transfer?

[中文](llm-bridge.md) · **English**

> Reading time: ~6 min · Last reviewed: 2026-10

Replacing actions with tokens preserves many RL concepts, but not an entire continuous-control architecture. Transfer the questions: **what is the objective, who generated the data, what estimates the update, what constrains it, and how is it validated?**

## Start with a mapping

| Classical RL | One text-generation counterpart | Important caveat |
| --- | --- | --- |
| State / observation | Prompt and current context | Hidden environmental state may be absent from text |
| Action | A token or a tool call | These granularities have different durations and costs |
| Policy | Autoregressive model | Sampling temperature and logged probabilities must agree |
| Reward | Preference, tests, task outcomes | Optimizable does not mean aligned with the user goal |
| Episode | One answer or a multi-turn task | EOS, timeout, and tool failure are different boundaries |

Appending a token is nearly deterministic in ordinary generation, but sampling and reward need not be. Tool agents receive observations from an external world with delays, permissions, and failures.

## What PPO and GRPO estimate

PPO estimates advantages from rollout-policy trajectories, then uses probability ratios and clipping to stop rewarding excessive changes in certain directions; it does not hard-bound those ratios. A reference-model KL penalty is different: it penalizes drift from a fixed anchor, while PPO clipping modifies the local optimization objective for the collected batch.

GRPO commonly uses relative rewards among several responses to one prompt to construct group-based advantages, reducing reliance on a separate value Critic. Identical group rewards can yield zero relative signal after normalization. Grouping and normalization also affect prompt weighting. They do not automatically fix sparse rewards, faulty verifiers, or cross-task comparability.

Equations and existing interactives are in [PPO clipping](../rlhf/ppo-clipping.en.md) and [GRPO / DPO / RLVR](../rlhf/after-rlhf.en.md). This chapter connects the classical RL questions rather than introducing duplicate notation.

## Follow three tokens through an update

Suppose a prompt produces three tokens, ending in EOS, and the completed answer earns reward 1. The process is readable without playing an animation:

<div class="drl-flow drl-sequence" aria-label="Text generation and reward attribution">
<span>Prompt<small>Not this rollout’s action</small></span><b>→</b><span>Tokens 1, 2<small>Save per-step log-prob</small></span><b>→</b><span>EOS<small>Score the full answer</small></span><b>→</b><span>Update<small>Credit valid actions</small></span>
</div>

For the sampled tokens, suppose old conditional probabilities are $[0.5,0.25,0.5]$ and new probabilities are $[0.6,0.2,0.6]$. Token ratios are $[1.2,0.8,1.2]$, while the sequence ratio is their product, 1.152. Token-level PPO must not interchange them; long-sequence products have different scale and variance.

A terminal score can provide learning signals to earlier actions. That does not mean each reasoning step was independently verified. Process rewards add intermediate feedback but can introduce their own bias; denser feedback is not automatically more accurate.

## When does GRPO’s within-group signal disappear?

Take four responses to the same prompt with rewards $[0,0,1,1]$. Under the population-standard-deviation convention, mean and standard deviation are both 0.5, giving normalized advantages $[-1,-1,1,1]$. Responses above the group mean are encouraged; those below are discouraged.

If all four rewards are 0, centered rewards are zero. An $\epsilon$ denominator prevents division by zero but creates no task-reward gradient. All-one groups can have the same issue. Relative comparison needs informative differences, not simply larger groups. Variance conventions, whether standard deviation is used, and token-versus-sequence averaging also change updates.

| Choice | Benefit | Cost |
| --- | --- | --- |
| Learned Value Critic | Cross-state return estimates and temporal attribution | Another training path with estimation error |
| Within-group relative rewards | Avoids training the same separate Value Critic | Multiple generations per prompt; weak signal without reward variation |
| Executable reward checks | Direct feedback on verifiable constraints | Limited to what the verifier checks |

These are not mutually exclusive choices on one axis. Verifiable rewards work with different policy optimizers. GRPO cannot repair a verifier that rewards incorrect answers.

## Old policy, reference model, and reward model are different roles

The old policy generated this batch’s actions; its log-probabilities stay fixed during the batch’s updates. A reference model is usually a steadier behavioral anchor for a KL penalty and need not equal the current old policy. A reward model or verifier scores the output and has another role entirely.

Preserve prompt/response boundaries, EOS, padding, sampling temperature, and policy versions. Otherwise action positions and probability ratios become ambiguous. Tool outputs are observations, not tokens sampled by the Actor. Concatenating them into one sequence does not turn them into policy actions.

These examples explain shared mechanisms, not a universal PPO/GRPO loss. Compare a specific implementation with [DeepSeekMath’s GRPO design](https://arxiv.org/abs/2402.03300) and the [RLHF model roles](../rlhf/three-stages.en.md).

## Long tasks: what memory changes

When the environment is partially observed, maintain history or learn a memory state. Memory is not a reward function and does not automatically restore the Markov property. The question is whether it retains information needed for future decisions.

Tool trajectories also distinguish task failure, service timeout, cancellation, and budget exhaustion. Each can stop a rollout without implying identical training masks, credit assignment, or retry behavior. Define failure semantics before increasing concurrency.

## Does a long trajectory require replacing GRPO with PPO?

Not necessarily. GRPO can compare whole-trajectory rewards for the same task without requiring the fifth action in each trajectory to mean the same thing or trajectories to have equal length. Ask whether terminal rewards are informative enough, how expensive repeated rollouts are, and how rewards and token weights survive training on segments.

Suppose a state has estimated value 0.4 and the next state 0.7. With zero immediate reward and $\gamma=1$ for this example, the TD residual is $0+0.7-0.4=0.3$. A Critic supplies state-dependent information, but this does not establish a causal contribution of 0.3: both estimates may be wrong, and the Critic still needs training from observed returns.

| Situation | Design to consider | New risk |
| --- | --- | --- |
| Several complete outcomes per task are affordable | Group-relative outcome advantage | Equal rewards, expensive rollouts, coarse attribution |
| Long trajectories need intermediate estimates | Critic with TD / GAE | Value bias, extra training and memory |
| A training segment ends but the task continues | Bootstrap from the next state, or process completed returns later | Don't label a segment boundary as a failed terminal |
| History is compacted | Preserve the resulting observation and policy version | Summaries may discard relevant state |

Context compaction does not itself terminate a task. Setting an intermediate bootstrap value to zero changes the learning target; bootstrapping beyond a true terminal invents a future that doesn't exist. See [TD, termination, and truncation](returns-and-td.en.md) and [Actor–Critic / GAE](actor-critic-gae.en.md). Compare estimator quality, complete-rollout cost, and final task performance rather than selecting an algorithm from the phrase “long horizon.”

## Multiple agents make the environment nonstationary

If other agents learn too, your transition and reward distributions change with them. Centralized training with decentralized execution can give a Critic joint information during training while each Actor uses only permitted local information at execution.

This is an entry point, not a complete MARL course. Communication, game objectives, nonstationarity, and cooperative credit assignment deserve separate treatment. Reference: [MADDPG](https://arxiv.org/abs/1706.02275).

## What evidence is missing when reward rises?

Check independent task success, length and cost, unseen scenarios, recovery from tool failure, and reward hacking. Using the same judge for training rewards and final scores does not provide independent validation.

Use the [evaluation stack](../../07-evaluation/evaluation-stack.en.md) to structure validation and [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md) for scoring rules and human anchors. The recurring question remains: did the model learn the task, or learn how to make the measuring stick report a better number?
