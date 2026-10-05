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

## Long tasks: what memory changes

When the environment is partially observed, maintain history or learn a memory state. Memory is not a reward function and does not automatically restore the Markov property. The question is whether it retains information needed for future decisions.

Tool trajectories also distinguish task failure, service timeout, cancellation, and budget exhaustion. Each can stop a rollout without implying identical training masks, credit assignment, or retry behavior. Define failure semantics before increasing concurrency.

## Multiple agents make the environment nonstationary

If other agents learn too, your transition and reward distributions change with them. Centralized training with decentralized execution can give a Critic joint information during training while each Actor uses only permitted local information at execution.

This is an entry point, not a complete MARL course. Communication, game objectives, nonstationarity, and cooperative credit assignment deserve separate treatment. Reference: [MADDPG](https://arxiv.org/abs/1706.02275).

## What evidence is missing when reward rises?

Check independent task success, length and cost, unseen scenarios, recovery from tool failure, and reward hacking. Using the same judge for training rewards and final scores does not provide independent validation.

Use the [evaluation stack](../../07-evaluation/evaluation-stack.en.md) to structure validation and [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md) for scoring rules and human anchors. The recurring question remains: did the model learn the task, or learn how to make the measuring stick report a better number?
