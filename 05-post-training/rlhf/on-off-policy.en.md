# RLHF: on-policy, off-policy, and why a Reward Model is not a Critic

[中文](on-off-policy.md) · **English**

> Reading time: ~3 min · Level: core · Last reviewed: 2026-09

## On-policy, off-policy, and offline RL

The distinction is not whether the code is literally running online. Ask instead: **is the behaviour policy that generated the data the same as, or sufficiently close to, the target policy being learned?**

| Setting | How data is produced | Typical property |
| --- | --- | --- |
| **On-policy** | the current or recent policy samples new trajectories | matched distribution, but rollouts are expensive and data expires quickly |
| **Off-policy** | another behaviour policy or an older policy generated the data | can reuse replay buffers and logs, but must handle distribution mismatch |
| **Offline RL** | training receives one fixed dataset and cannot collect more environment interaction | usually a special off-policy setting, strongly limited by dataset coverage |

PPO samples with $\pi_{\text{old}}$ and performs several constrained updates to $\pi_\theta$ on that batch. Although both old and new policies appear, they remain close and the batch is soon replaced by fresh rollouts, so PPO is still on-policy or near-on-policy. Off-policy is not synonymous with offline: SAC may keep collecting data while repeatedly learning from replay-buffer experience produced by earlier policies.

Standard DPO uses fixed chosen/rejected pairs and therefore has offline-data characteristics, but it has no Bellman backup, Critic, or environment rollout. **Offline preference optimization** is more precise than calling it classical off-policy RL.

<details class="interview" markdown="1">
<summary>When does industry actually need RL?</summary>

RL is most useful when an action changes later states and the product cares about long-term outcomes. Recommenders may trade immediate clicks against long-term retention; ad systems balance conversions, budgets, and user experience; logistics and robotics optimise interdependent action sequences; Conversational AI can treat retrieval, tool calls, clarification, answers, and human escalation as one episode.

Real exploration can harm users or incur cost. Production systems therefore combine logged data, simulation, offline evaluation, action constraints, limited exploration, and controlled A/B tests. The algorithm name is not the first decision: define state, action, and reward; determine whether reward is verifiable, who generates the data, and whether new trajectories can be collected safely.

</details>

## A Reward Model is not a Critic

Both emit scalars, which makes them easy to confuse:

| | Reward Model | Critic / Value Model |
| --- | --- | --- |
| Input | prompt + completed answer | current prompt + generated prefix |
| Output | learned proxy preference score | expected return from the current state |
| Question answered | “How good does this completed answer look?” | “If the current policy continues from here, what return should it expect?” |
| During PPO | usually frozen | trained alongside the current Actor |

The Reward Model does not provide “true human satisfaction.” It supplies a **proxy reward** fitted on limited preference data, so it can be wrong, favour superficial styles, and be gamed. The Critic estimates a conditional expectation under the current policy; when the Actor changes, its target changes too.

At whole-response granularity this resembles a contextual bandit: receive a prompt, generate one answer, then receive one overall score. Inside generation it remains a sequential decision process whose state changes with every token. The two descriptions differ only in abstraction level.
