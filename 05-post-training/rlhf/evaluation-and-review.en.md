# RLHF: evaluation closes the loop, and review questions

[中文](evaluation-and-review.md) · **English**

> Reading time: ~5 min · Level: core · Last reviewed: 2026-09

## Evaluation closes the loop

A lower training loss or higher average reward does not imply a more useful model.
Evaluate capability regression, factual grounding, safety and policy compliance, tool
use and task completion, multi-turn consistency, latency, and cost separately.
Open-ended responses can combine human review, calibrated LLM judges, and deterministic
checks; maths, code, and structured tasks should prefer verifiers where possible.

Keep a frozen regression suite before launch, then use shadow evaluation and controlled
A/B tests. Do not feed every production failure straight back into training. Deduplicate,
audit, and stratify it with a failure taxonomy, then decide whether it belongs as an SFT
demonstration, preference pair, verifier case, or system rule.

## Interview questions

<details class="interview" markdown="1">
<summary>How many models are in RLHF stage 3, and which train?</summary>

Four: Actor, Critic, Reward, Reference. **Only Actor and Critic update weights.** Reward is frozen after stage 2; Reference is a frozen copy of the SFT model. Actor and Reference start identical — the Actor drifts as it trains, and the Reference stays put as the KL anchor.

</details>

<details class="interview" markdown="1">
<summary>Why is the reference model and KL penalty needed? What if you remove it?</summary>

The reward model is a fit on limited data with no constraint off-distribution. Optimise freely and the policy finds outputs it scores highly and people reject — reward hacking. Typical symptoms: answers grow longer, pile up ingratiating phrasing, or collapse into a repetitive pattern.

The KL penalty tethers the policy near SFT. Too large a $\beta$ and nothing moves; too small and it drifts off. It is the method's central hyperparameter, not an optional regulariser.

</details>

<details class="interview" markdown="1">
<summary>What is the Critic for, and how does GRPO avoid it?</summary>

It estimates $V_t$ so the advantage $A_t = R_t - V_t$ has lower variance. Plain returns make policy-gradient variance too large to train through.

GRPO's observation: if a baseline is all you need, sampling a group of answers per prompt and normalising by the group's mean and standard deviation supplies one. That removes a full-size network *that was being trained* — real memory and real compute.

</details>

<details class="interview" markdown="1">
<summary>What is the essential difference between DPO and PPO?</summary>

DPO uses a derivation: KL-constrained reward maximisation has a closed-form optimum, so the reward can be rewritten in terms of the policy and the preference loss differentiated directly. The reward model and the RL loop both disappear.

Standard DPO's cost is **offline** preference data. The policy changes while the data does not, so it cannot actively explore its current failures. PPO and other online RL methods resample from the current policy and are more natural for exploration, environment interaction, or verifiable multi-step outcomes, but rollout is expensive and training less stable.

DPO therefore cannot replace PPO everywhere, but PPO is not universally better either. Use DPO when high-quality static preferences cover the task; use online methods when the current policy must keep producing new evidence. Online DPO variants reinforce that the real distinction is the data-and-feedback loop, not only the loss name.

</details>

<details class="interview" markdown="1">
<summary>Can reward-model scores be compared directly?</summary>

Bradley–Terry directly constrains the **gap** between chosen and rejected answers to the same prompt. Adding a constant to every score leaves the loss unchanged, so the zero point has no identifiable meaning; a score of $2.4$ is not “2.4 units of satisfaction.”

A fixed Reward Model's raw outputs can of course be used numerically, but comparisons across prompts, domains, or model versions require evidence that calibration and scale are stable. Implementations may also whiten or normalise rewards; that is an optimisation choice, not a rule implied by Bradley–Terry.

</details>

<details class="interview" markdown="1">
<summary>Why is there less reward hacking under RLVR?</summary>

Because the reward is no longer fitted. Checking a maths answer or running a test suite is a fixed program — there is no "outside the training distribution" to exploit.

The trade is coverage: it only applies where outcomes are automatically verifiable. Writing and open-ended dialogue have no checker and still need a learned reward model. And verifiers can be gamed too — code that passes the tests while being wrong.

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>Where each of the four models comes from, and which two update weights.</li>
    <li>What happens without the KL penalty, and the symptoms of $\beta$ being too large or too small.</li>
    <li>Whether the Critic reduces variance or improves accuracy, and what GRPO replaces it with.</li>
    <li>Why DPO needs no reward model, and what it gives up for that.</li>
    <li>For PPO-style clipping, which two combinations of advantage sign and ratio crossing make the local gradient zero?</li>
  </ol>
</div>

## Next

- [Post-training overview](../README.en.md)
- [Data & feedback](../../01-data-and-feedback/README.en.md) — the quality of preference labels themselves
- [Evaluation](../../07-evaluation/README.en.md)

## Papers to start with

- [InstructGPT](https://arxiv.org/abs/2203.02155) — where the three stages come from
- [PPO](https://arxiv.org/abs/1707.06347)
- [DPO](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL and reward hacking

## Further reading (Chinese)

- [Reinforcement learning in large models](https://zhuanlan.zhihu.com/p/693582342) — a Zhihu article, in Chinese.
  This chapter deliberately covers only the RLHF trunk. The algorithm taxonomy is in
  that piece: MDP elements, the Bellman equation, the bias-variance tradeoff across
  MC/TD/GAE, PPO's four-model setup, DPO with IPO/KTO, and what GRPO, DAPO,
  Dr. GRPO, RLOO, and REINFORCE++ each set out to fix.
