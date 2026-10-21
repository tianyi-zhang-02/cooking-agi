# RLHF: evaluation and review questions

[中文](evaluation-and-review.md) · **English**

> Reading time: ~6 min · Last reviewed: 2026-10

## Reward increased: what explanations should we rule out? {#independent-evaluation}

A higher training reward might reflect better ability, longer answers, more attempts, or better accommodation of a particular judge. Before reducing everything to one score, fix the comparison conditions.

| Check | Procedure | Misinterpretation prevented |
|---|---|---|
| Problems and data splits | Hold out problems from training and tuning; audit near-duplicates and template leakage | Memorization mistaken for generalization |
| Generation budget | Match sample counts, token limits, and decoding; report costs too | Extra attempts mistaken for a better model |
| Independent acceptance | Add boundary tests, human audits, or another standard beyond the training verifier | A faulty reward function grading its own success |
| Subgroups | Break down difficulty, language, length, and task type | Average gains concealing a subgroup regression |
| Variability and uncertainty | Keep paired per-problem results; report sample size, seeds, and suitable intervals | A small sample or lucky generation mistaken for a stable improvement |
| Capability and cost regression | Check previous tasks, refusals, factuality, latency, and output length | Improving one behavior while harming another |

A small example: if each independent sample succeeds with probability $0.5$, four attempts yield at least one success with probability $1-(1-0.5)^4=0.9375$. **The model did not change; the budget did.** This teaching calculation assumes independent attempts with a fixed success probability. Real problems vary in difficulty, so an aggregate average success rate cannot simply be substituted into this formula. Distinguish single-generation and multiple-attempt results; if a selector must identify the correct answer, include its error and cost as well.

For a preference judge, randomize A/B order and check whether it merely favors longer answers. Do not discard every disagreement with human labels: disagreement may reveal an unclear rubric. See [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md) for examples and [metric robustness](../../07-evaluation/metric-robustness.en.md) for paired comparisons and population mixtures.

<span id="evaluation-closes-the-loop"></span>

## How evaluation feeds back into training {#closing-the-loop}

The checks above cover one comparison. Across the whole training process, evaluate capability regression, factual grounding, safety and policy compliance, tool use and task completion, multi-turn consistency, latency, and cost separately. Open-ended responses can combine human review, calibrated LLM judges, and deterministic checks; math, code, and structured tasks should prefer verifiers where possible.

From launch to feeding failures back into training, work in this order:

1. **Before launch**: keep a frozen regression suite and compare every candidate version on the same problems.
2. **Shadow evaluation**: the new model runs alongside real traffic, but its outputs are not returned to users.
3. **Controlled A/B test**: expose only part of the traffic, with the metrics and stopping conditions fixed in advance.
4. **Handle production failures**: do not feed them all straight back. Classify failures by type, deduplicate, audit, and stratify them, then decide where each kind belongs:

| What this kind of failure can supply | Better destination |
|---|---|
| A correct answer can be written | SFT demonstration |
| One of two answers can be judged better | chosen/rejected pair |
| A program can judge right or wrong | verifier case |
| The problem lies in format, permissions, or call flow | system rule; the model may not need to change |

## Interview questions

<details class="interview" markdown="1">
<summary>What roles appear in typical PPO-style RLHF, and which train?</summary>

The four common roles are Actor, Critic, Reward, and Reference; they do not require four full models to be resident simultaneously. **Actor and Critic update weights in this setup.** Reward is frozen after stage 2; Reference is a frozen SFT copy. Actor and Reference start identical, but only the Actor changes as policy training proceeds.

</details>

<details class="interview" markdown="1">
<summary>Why are the reference model and KL penalty needed? What happens if you remove them?</summary>

Preferences learned from limited data may not generalize to later policy outputs. Stronger optimization can find high-scoring answers that people do not actually prefer, such as accommodating a stylistic bias instead of improving the answer.

Reference KL adds a cost for deviating from an anchor. Increasing $\beta$ generally strengthens that cost, but its effect also depends on reward scale. Removing KL does not guarantee collapse, and retaining it does not guarantee freedom from reward hacking. Compare its presence and coefficient with independent evaluation rather than treating a nonzero coefficient as a law. See [Reference and Critic](reference-and-critic.en.md) for the derivation and numerical example.

</details>

<details class="interview" markdown="1">
<summary>What is the Critic for, and how does GRPO avoid it?</summary>

The Critic estimates state value $V_t$ to construct advantages such as $\hat A_t=\hat G_t-V_t$. A suitable baseline can reduce variance; TD or GAE also introduces a bootstrapping bias tradeoff. Training without a Critic is still possible; REINFORCE is one example.

GRPO constructs relative signals from answers to the same prompt instead of training a separate Critic. It saves Critic-related costs but requires group sampling; all-correct or all-wrong groups (see [DAPO's dynamic sampling](../dapo.en.md#dynamic-sampling)), reward scales, and length normalization still matter. Counting fewer models does not establish lower total training cost.

</details>

<details class="interview" markdown="1">
<summary>What is the key difference between DPO and PPO?</summary>

DPO uses a derivation: a KL-regularized reward objective with fixed $\beta>0$ has a closed-form optimum under the policy-support and integrability conditions. This is not a per-update guarantee of a fixed KL radius. The reward can then be rewritten in terms of the policy and the preference loss differentiated directly. Standard offline DPO therefore trains no separate reward model and runs no online RL loop; the reward difference is implicit in the loss. See the [DPO derivation](../after-ppo.en.md#dpo-derivation) for the full objective and conditions.

One limitation of standard offline DPO is its fixed preference dataset. The policy changes while the data does not, so training cannot actively explore the current policy's new failures. PPO and other online RL methods resample from the current policy and are more natural for exploration, environment interaction, or verifiable multi-step outcomes, but rollout is expensive and training less stable.

DPO therefore cannot replace PPO everywhere, but PPO is not universally better either. Use DPO when high-quality static preferences cover the task; use online methods when the current policy must keep producing new evidence. Online DPO variants reinforce that the real distinction is the data-and-feedback loop, not only the loss name.

</details>

<details class="interview" markdown="1">
<summary>Can reward-model scores be compared directly?</summary>

Bradley–Terry directly constrains the **gap** between chosen and rejected answers to the same prompt. Adding a constant to every score leaves the loss unchanged, so the zero point has no identifiable meaning; a score of $2.4$ is not “2.4 units of satisfaction.”

A fixed Reward Model's raw outputs can of course be used numerically, but comparisons across prompts, domains, or model versions require evidence that calibration and scale are stable. Implementations may also whiten or normalize rewards; that is an optimization choice, not a rule implied by Bradley–Terry.

</details>

<details class="interview" markdown="1">
<summary>Why is independent evaluation still needed with a verifier?</summary>

A verifier only checks the conditions implemented in it. Incomplete tests, faulty answer extraction, or missing boundary cases can reward incorrect outputs. A deterministic program need not express the full task correctly.

RLVR fits tasks or subgoals with automatically checkable outcomes. Open-ended dialogue can also use format or tool-argument checks, but these do not cover all aspects of answer quality. Reserve independent problems, different test families, and human review where needed to distinguish task learning from checker exploitation. [Verifiable rewards](../verifiable-rewards.en.md) starts from a worked counterexample: a constant-output “solution” that passes a single sorting test.

</details>

## Self-check

<div class="taste-check">
  <strong>Try explaining these in your own words:</strong>
  <ol>
    <li>What each role does in typical PPO-style RLHF, and why four roles do not imply four resident weight copies.</li>
    <li>What the reference KL and the old-policy ratio each compare the current policy against.</li>
    <li>What the Critic helps estimate, and what tradeoffs GRPO makes by not training one.</li>
    <li>Why DPO needs no reward model, and what it gives up for that.</li>
    <li>For PPO-style clipping, which two combinations of advantage sign and ratio crossing make the local gradient zero?</li>
  </ol>
</div>

## Next

- [Post-training overview](../README.en.md)
- [Data & feedback](../../01-data-and-feedback/README.en.md) — the quality of preference labels themselves
- [Evaluation](../../07-evaluation/README.en.md)

## Further reading (Chinese)

This chapter covers the main RLHF workflow. Separate notes examine the individual algorithms: start with [PPO, GRPO, and DPO](../after-ppo.en.md),
then read [DAPO](../dapo.en.md), [Clipping and gradients](../clipping-variants.en.md), [GSPO and ASPO](../policy-ratios.en.md),
and [asynchronous training](../async-policy-learning.en.md) as needed.

- [Reinforcement learning in large models](https://zhuanlan.zhihu.com/p/693582342) — a Zhihu article, in Chinese, as a supplement.
  It runs from MDP elements, the Bellman equation, and the bias-variance tradeoff across
  MC/TD/GAE, through PPO's four-model setup and DPO with IPO/KTO, to what GRPO, DAPO,
  Dr. GRPO, RLOO, and REINFORCE++ each set out to fix.

## Papers to start with

- [InstructGPT](https://arxiv.org/abs/2203.02155) — where the three stages come from
- [PPO](https://arxiv.org/abs/1707.06347)
- [DPO](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL and reward hacking
