# After PPO: every algorithm deletes one of its parts

[中文](after-ppo.md) · **English**

> Reading time: ~11 min · Type: chapter · Last reviewed: 2026-09

## Start with what each algorithm removes

The long list of algorithms that followed PPO looks like a pile of separate inventions. It isn't: each one **deletes a component of PPO and then deals with whatever surfaces afterwards.** Work out what each removed and what it cost, and you no longer have to memorize the table.

PPO, GRPO, and DPO answer the same question: once an SFT model exists, how can response quality or human preference make good answers more likely and bad answers less likely? They differ in how feedback enters training:

- **PPO** uses response rewards, a Critic's value baseline, and clipped policy updates;
- **GRPO** samples a group of responses to one prompt and replaces the Critic with group-relative rewards;
- **DPO** reads chosen/rejected pairs and updates the policy through a preference-classification loss.

### Build intuition with one problem

Use the same prompt, “calculate $17\times24$,” in all three cases:

| Method | What it observes | Where the update direction comes from |
| --- | --- | --- |
| PPO | the Actor answers 388, reward $R=0$, and the Critic predicts $V=0.6$ | $A=R-V=-0.6$, so reduce the probability of this trajectory |
| GRPO | answers `[408, 388, 408 with work, 428]` get rewards `[1,0,1,0]` | group normalization gives $A=[1,-1,1,-1]`; raise relatively good responses and lower relatively bad ones |
| DPO | a fixed record says `chosen=408, rejected=388` | make the current policy prefer chosen more than the Reference does |

One sentence each: PPO asks “how much better was the result than the Critic expected?” GRPO asks “how much better was this result than the other answers to the same prompt?” DPO does not estimate an advantage; it directly asks whether chosen improved more than rejected relative to the Reference.

## What PPO has available to delete

A classic PPO-RLHF diagram has four conceptual roles, two of them training:

| Model | Training? | Job |
| --- | --- | --- |
| Policy | yes | the model you're trying to produce |
| Critic | **yes** | estimates state value, the baseline for the advantage |
| Reward | frozen | scores responses |
| Reference | frozen | KL anchor, keeps the policy from drifting |

**The Critic is one of the most expensive extra roles**: it trains alongside the current policy and carries optimizer state. It may be a separate value model, or share an Actor backbone and add only a value head. These are conceptual roles, not a promise that four independent full models remain resident in memory. Much of the later story is still about removing the Critic.

## The main line: delete the Critic

The Critic exists for exactly one reason: **to supply a baseline for the advantage**, to cut variance. If all you need is a baseline, it does not have to be a learned network.

| Algorithm | Where the baseline comes from | Cost |
| --- | --- | --- |
| **GRPO** | sample a group of responses per prompt; use the group mean (divided by its std) | multiple samples per prompt, so sampling cost goes up |
| **RLOO** | leave-one-out: one sample's baseline is the mean reward of the others | same sampling cost; and it treats the whole response as **one action**, giving up token-level credit assignment |
| **REINFORCE++** | batch-level statistics for advantage normalization | a coarser baseline, but it keeps PPO's clipping and KL stabilization |

What they share: **the baseline goes from learned to sampled.** That removes a full-size training model — and plants the seed of every problem below.

RLOO's baseline, with $k$ samples per prompt:

$$\hat A_i = r_i - \frac{1}{k-1}\sum_{j \neq i} r_j$$

Its argument is that RLHF starts from a trained SFT model rather than a randomly initialized network, so PPO's machinery for stabilizing unstable training (GAE, per-token value estimation) may not be necessary.

### GRPO's group-relative advantage

Sample $G$ responses for one prompt and score them $R_1,\ldots,R_G$. A common response-level advantage is

$$
\mu_R=\frac{1}{G}\sum_{i=1}^{G}R_i,\qquad
\sigma_R=\sqrt{\frac{1}{G}\sum_i(R_i-\mu_R)^2},\qquad
\hat A_i=\frac{R_i-\mu_R}{\sigma_R+\varepsilon}.
$$

For rewards $[1,0,1,0]$, $\mu_R=0.5$, $\sigma_R=0.5$, and the advantages are $[1,-1,1,-1]$. Rewards may come from a Reward Model, humans, a math verifier, unit tests, code execution, or a format checker.

A common GRPO implementation broadcasts one sequence-level advantage to the tokens in that response and then uses a PPO-style clipped objective:

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})},
$$

$$
L_{\mathrm{GRPO}}=
\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t
\min\!\left(
\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right),
$$

usually with a KL constraint against a frozen Reference. GRPO therefore does not remove rollout, reward, policy ratios, or stabilization; its main deletion is the learned value baseline.

| | PPO baseline | GRPO baseline |
| --- | --- | --- |
| Source | $V_\phi(s_t)$ learned by a Critic | within-prompt reward statistics |
| Granularity | can vary with each prefix/token state | commonly one relative advantage for a whole response |
| Main cost | train a value function | multiple rollouts per prompt |
| Typical failure | Critic is inaccurate or unstable | no reward variation within the group, hence no learning signal |

## What surfaces once the Critic is gone

This section is the point. **A sampled baseline fails differently than a learned one.**

**One: when a group's rewards don't differ, the baseline carries no information.**
If every response is right or every one is wrong—for example, $R=[0,0,0,0]$—subtracting the mean leaves every advantage at zero. An $\varepsilon$ in the denominator prevents division by zero but cannot create a relative signal. The group contributes **no useful policy-gradient signal**. On tasks that are too easy or too hard, that share gets large.

**Two: normalization introduces bias.**
Dividing by the group's standard deviation looks like standardization but weights groups unequally; normalizing by sequence length pushes systematically toward one length. That is Dr. GRPO's argument: those terms bias the model toward **longer answers that aren't necessarily better**. Remove them and token efficiency improves visibly.

**Three: symmetric clipping can't lift low-probability tokens.**
PPO clips the probability ratio to $[1-\epsilon, 1+\epsilon]$. For a token whose current probability is tiny, the ceiling $1+\epsilon$ permits only a minuscule absolute increase — it has almost no path back. The long-run consequence is entropy collapse: the model gets more certain and less diverse.

**Four: sequence-level loss dilutes long answers.**
Average the loss per sample and a 1000-token response carries the same weight as a 50-token one, so each token in the long answer receives a thinner share of gradient. Reasoning tasks are exactly the ones that need long answers.

DAPO's four changes map onto these four problems: dynamic sampling filters out all-right/all-wrong groups, the clip bounds are decoupled (Clip-Higher), the loss becomes token-level, and overlong responses get soft shaping instead of a hard cut.

## Another line: delete the RL loop entirely

DPO goes further: in its standard offline training phase, it needs **no explicit Reward Model, Critic, or online rollout loop**.

Each record is $(x,y_w,y_l)$: a prompt, chosen response, and rejected response. Define how each response's log probability changes relative to a frozen Reference:

$$
\Delta_w=\log\pi_\theta(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x),
$$

$$
\Delta_l=\log\pi_\theta(y_l\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x).
$$

DPO minimizes

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\!\left(\beta[\Delta_w-\Delta_l]\right).
$$

Do not memorize the expansion. It says only:

$$
\boxed{\text{Relative to the Reference, chosen should improve more than rejected.}}
$$

Why is a separate Reward Model unnecessary? A Bradley–Terry preference model says

$$
P(y_w\succ y_l\mid x)=\sigma\!\left(r(x,y_w)-r(x,y_l)\right).
$$

At the optimum of KL-constrained reward maximization, policy and reward satisfy

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+C(x).
$$

Substitute this relation into the preference probability: $C(x)$ cancels in the reward difference for the same prompt, so a policy/reference log-ratio directly represents the implicit reward difference. Reward has not vanished; it has been **absorbed into the policy objective**. See [the three stages of RLHF](rlhf-pipeline.en.md) for the full derivation.

### DPO versus SFT

If A is chosen and B is rejected,

$$
\mathcal L_{\mathrm{SFT}}=-\log\pi_\theta(A\mid x)
$$

only says “A is worth imitating.” DPO observes A and B together and learns “A is preferred to B.” Chosen need not be the one uniquely perfect answer, and an ordinary binary preference usually supplies only the order $A>B$, not how much better A is.

$\beta$ participates both in the theoretical Reference constraint and in the preference-logit scale. Mechanically memorizing “larger means more conservative” is unsafe: derivation conventions, loss implementation, and data scale all affect observed training behavior. In an interview, write the exact formula or library convention first, then reason about increasing or decreasing it.

Standard DPO typically pays for its simplicity with **fixed offline preference pairs**. The policy changes while the data came from an older policy, so distribution mismatch grows and training cannot actively discover the current policy's new failures. Online DPO and iterative data refresh can reduce this problem; “DPO is always offline” is not an absolute boundary for the whole family. The deeper question is whether data follows the current policy and closes the feedback loop.

Two follow-ups:

- **IPO** — DPO overfits preference data fast. IPO adds a regularizer so the model converges without tricks like early stopping.
- **KTO** — no pairs required, just a "good" or "bad" label per sample. An order of magnitude cheaper to collect, at the price of a coarser signal.

## And one more: replace the reward model with a program

Math has answers to check. Code has tests to run. For these, the reward **doesn't need to be learned** — write a checker.

What this deletes is the Reward Model. A deterministic checker removes one major attack surface—the approximation error of a learned reward—but the checker can still encode the wrong specification or leave loopholes. Verifiable reward narrows reward hacking; it does not make the objective automatically correct.

The limitation is obvious: it only applies where you can write the checker.

## The table

| | Typical training data | Explicit reward | Critic | Training-time rollout | Central trade-off |
| --- | --- | --- | --- | --- | --- |
| PPO | prompts + responses from the current/recent policy | RM, rule, or environment | yes | yes | flexible feedback and online exploration; most complex system |
| GRPO | prompts + a group of responses per prompt | RM, verifier, or environment | no | yes, multiple times per prompt | removes the Critic; needs within-group variation and spends more generation compute |
| DPO | fixed chosen/rejected pairs | no explicit call; reward difference is implicit in the loss | no | not in standard offline training | simple and stable; limited by preference coverage and distribution mismatch |

The extensions still fit the “what was removed?” lens: RLOO replaces the Critic with a leave-one-out baseline; REINFORCE++ uses batch statistics; RLVR replaces a learned Reward Model with a verifier; DAPO moves beyond deletion and repairs GRPO's sampling, clipping, token weighting, and overlong-response behavior one by one.

## How to choose

**Ask first whether the reward can be verified.** If you can write a checker, write it — that pays more than any algorithm swap.

**Then ask whether you can sample online.** No (static preference data only) → the DPO family, accepting the mismatch. Yes → the GRPO family.

**Only then ask which component to delete.** Memory-bound → drop the Critic. Long outputs and worried about entropy collapse → ship DAPO's fixes alongside.

Don't invert that order — **picking the algorithm first and retrofitting the data is the most common source of rework in these projects.**

## Down to a checklist

1. Is my reward learned or verified? If learned, am I watching for reward hacking?
2. With a group-relative baseline, what fraction of groups is all-right or all-wrong? That fraction was sampled for nothing.
3. What bias do my normalization terms introduce? Is response length climbing monotonically through training?
4. Is each token in a long answer getting a thinner gradient than one in a short answer?
5. If using DPO: which policy produced my preference data, and how far is it from the current one?
6. If someone says “DPO completely replaces PPO,” have they ignored online exploration, environment interaction, and the current-policy data loop?

## Can you explain it in two minutes?

1. From memory, write PPO's $A\approx G-V$, GRPO's group-normalized advantage, and the DPO loss.
2. Explain why GRPO removes the Critic but not rollout, and why an all-right or all-wrong group teaches nothing.
3. Explain why DPO needs no explicit Reward Model and identify the exact equation behind “reward is absorbed implicitly.”
4. Answer whether DPO can fully replace PPO: fixed preference pairs are cheap and stable, but they do not automatically explore a current policy's new failures. When the task needs online sampling, environment feedback, or verifiable multi-step outcomes, an online method in the PPO/GRPO family is more natural—and also more expensive and harder to stabilize.

## Where to read next

- [The three stages of RLHF, and what came after](rlhf-pipeline.en.md): what each model does, and the DPO derivation
- [Data and feedback](../01-data-and-feedback/): the quality of preference labels themselves
- [Evaluation](../07-evaluation/): telling whether alignment actually helped

## Starting papers

- [PPO](https://arxiv.org/abs/1707.06347) — clipped objective and trust region
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [DAPO](https://arxiv.org/abs/2503.14476) — four fixes for four GRPO failure modes
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) — Dr. GRPO, the length bias from normalization
- [Back to Basics](https://arxiv.org/abs/2402.14740) — RLOO
- [REINFORCE++](https://arxiv.org/abs/2501.03262) — drop the Critic, keep PPO's stabilizers
- [DPO](https://arxiv.org/abs/2305.18290) · [IPO](https://arxiv.org/abs/2310.12036) · [KTO](https://arxiv.org/abs/2402.01306)

## Further reading (Chinese)

- [大模型中的强化学习 — RL in large models](https://zhuanlan.zhihu.com/p/693582342) — by @大家好我是爱因 on Zhihu. This chapter is organized along a single axis, "what got deleted," and covers only the trunk. That piece is the encyclopedic view — MDP elements, the Bellman equation, MC/TD/GAE, then each algorithm in detail. Start there if you want the fuller map.
