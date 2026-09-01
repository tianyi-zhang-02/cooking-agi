# After PPO: every algorithm deletes one of its parts

[中文](after-ppo.md) · **English**

> Reading time: ~14 min · Type: chapter · Last reviewed: 2026-09

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

<details class="interview" markdown="1">
<summary>Step 1: can the reward be verified automatically?</summary>

**When verification is possible, consider a verifier / RLVR first.** Math answers, code tests, and structural constraints can be judged by a stable program without first fitting a Reward Model. Still audit whether the verifier checks only final answers, leaves test loopholes, or fails to represent the actual product objective.

</details>

<details class="interview" markdown="1">
<summary>Step 2: can training keep sampling from the current policy?</summary>

With only static chosen/rejected pairs, the DPO family is usually the direct choice, accepting coverage gaps and distribution mismatch. If training can keep rolling out and obtaining rewards or environment feedback, an online method such as PPO or GRPO can discover the current policy's new failures. It costs more generation compute, system complexity, and optimization variance.

</details>

<details class="interview" markdown="1">
<summary>Step 3: what is the real systems or optimization bottleneck?</summary>

If Critic memory, compute, or stability is the bottleneck, consider critic-free methods such as GRPO or RLOO. If long responses are diluted, low-probability tokens cannot recover, or entropy keeps falling, consider DAPO's token-level loss, Clip-Higher, dynamic sampling, and length handling. Do not choose an algorithm name first and force the data to fit it afterwards.

</details>

The order is: **validate the feedback, decide whether an online loop is possible, then diagnose the system bottleneck.**

## Down to a checklist

<details class="interview" markdown="1">
<summary>1. Is the reward learned or verified, and what should each monitor?</summary>

For a learned Reward Model, monitor reward hacking, length or style shortcuts, and whether agreement with human judgment decays as the policy drifts. For a verifier, audit the specification gap: whether tests cover the real requirement, final-answer checking rewards guessing, or the model can exploit formatting and execution loopholes.

</details>

<details class="interview" markdown="1">
<summary>2. What fraction of GRPO groups are all-right or all-wrong?</summary>

Log the zero-variance group rate directly. When all rewards match, normalized advantages are zero and the rollouts supply no relative gradient. If the rate is high, adjust problem difficulty, group size, or reward resolution, or use dynamic sampling to filter uninformative groups—while monitoring the distribution shift caused by filtering.

</details>

<details class="interview" markdown="1">
<summary>3. What bias do reward and length normalization introduce?</summary>

Plot reward, correctness, response length, entropy, and within-group reward variance together. Group-standard-deviation normalization changes the weight of low-variance groups; response-length averaging can dilute tokens in long sequences. If length rises while correctness stays flat, the model may be exploiting normalization or a reward shortcut.

</details>

<details class="interview" markdown="1">
<summary>4. Are long-response tokens being outweighed by short responses?</summary>

First determine whether the loss uses a per-sequence mean or batch-wide per-token aggregation. If every response has equal total weight, a token in a 1000-token answer usually receives less weight than one in a 50-token answer. Long-reasoning tasks may need token-level loss aggregation and both response-level and token-level metrics.

</details>

<details class="interview" markdown="1">
<summary>5. Which policy produced the DPO preference data, and why does it matter?</summary>

Record the generation checkpoint, sampling temperature, decoding constraints, and annotation date. As the current policy moves away from the producer policy, fixed pairs cover fewer of its present failures. Test coverage on held-out prompts and resample, relabel, or switch to an iterative / online preference loop after meaningful drift.

</details>

<details class="interview" markdown="1">
<summary>6. Can DPO fully replace PPO / GRPO?</summary>

Not in general. Standard DPO is cheap and stable when high-quality offline pairs already cover the desired behavior. PPO / GRPO can roll out the current policy and receive new rewards, which fits exploration, environment interaction, and verifiable multi-step tasks more naturally—but training is more expensive and less stable. The feedback loop matters more than the loss name.

</details>

## Can you explain it in two minutes?

<details class="interview" markdown="1">
<summary>Write the three central equations for PPO, GRPO, and DPO.</summary>

PPO's direction starts with $\hat A_t\approx G_t-V_\phi(s_t)$ and uses a clipped probability-ratio surrogate to bound one update. GRPO replaces the learned value baseline with $\hat A_i=(R_i-\mu_R)/(\sigma_R+\varepsilon)$. DPO uses $-\log\sigma\!\left(\beta[\Delta_w-\Delta_l]\right)$, where each $\Delta$ is the current policy's log-probability change relative to the Reference.

</details>

<details class="interview" markdown="1">
<summary>Why does GRPO remove the Critic but not rollout?</summary>

It still needs the current or recent policy to generate multiple responses to one prompt and score every response; otherwise there is no group-relative baseline. What disappears is the Critic that learns $V_\phi(s)$. If a group is all-right or all-wrong, $R_i-\mu_R=0$, so it supplies no relative policy-gradient signal.

</details>

<details class="interview" markdown="1">
<summary>Why does DPO need no explicit Reward Model?</summary>

The KL-constrained optimum satisfies $r(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+C(x)$. Substitute it into a Bradley–Terry preference probability and $C(x)$ cancels for two responses to the same prompt. The reward difference becomes a chosen/rejected policy-to-reference log-ratio. Reward did not vanish; it is implicit in the DPO loss.

</details>

<details class="interview" markdown="1">
<summary>Two-minute conclusion: can DPO fully replace PPO?</summary>

No. DPO is simple and stable when fixed, high-quality preference pairs cover the target behavior, but it does not automatically explore a current policy's new failures. PPO / GRPO can keep sampling the current policy and receive feedback from an environment, Reward Model, or verifier, which better fits exploration and multi-step outcomes but costs more rollout compute and is harder to stabilize. Online DPO shows that the real boundary is whether data updates with the policy, not the algorithm's name.

</details>

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
