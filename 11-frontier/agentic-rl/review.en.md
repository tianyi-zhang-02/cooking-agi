# Training agents with RL: review questions

[中文](review.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-10

This page turns the main points of the series into interview questions and a self-check; every number in the answers can be traced to a source in the earlier notes.

## Interview questions

<details class="interview" markdown="1">
<summary>Going from single-turn RLVR to multi-turn agent RL, what changes and what stays the same?</summary>

The basic workflow stays the same: sample, score, and update the policy using advantages. Update rules such as PPO and GRPO still apply. The sample is now a multi-turn sequence of (action, observation), with observations supplied by the environment rather than the policy. The reward usually arrives at the end of the episode, based on the environment's final state, and the checker often lives in the same sandbox as the agent, potentially within its reach.

Trajectory length and execution cost also change. One trajectory can contain thousands of tool calls, and a rollout involves starting containers, running commands and running tests, taking minutes. DeepSeek-R1 (2025-01) did not apply large-scale RL to software engineering precisely because evaluation was too slow.

</details>

<details class="interview" markdown="1">
<summary>Why are tokens returned by tools kept out of the loss? What happens if you do not mask them?</summary>

The log-probability of a trajectory splits into two parts: the probability of the policy choosing its actions, plus the probability of the environment returning its observations. The latter does not depend on the parameters, so its gradient is zero and the policy gradient contains only action tokens to begin with. Without the mask, training would raise or lower the probability of "the model writing this file content or this error message itself", even though that text came from the environment; and since tool outputs are often much longer than actions, they would dominate the gradient. In Search-R1's ablation (2025-03), average exact match was 0.431 with the mask and 0.343 without. Masked tokens remain in the context; they just produce no gradient.

</details>

<details class="interview" markdown="1">
<summary>Four 3-turn trajectories on the same task get rewards [1, 0, 1, 1]. What are the group-relative advantages, and which tokens receive gradient?</summary>

The mean is 0.75 and the population standard deviation about 0.433, so the normalised advantages are [+0.577, −1.732, +0.577, +0.577]. Using $G-1$ as the sample-variance denominator gives a standard deviation of $\sqrt{0.75/(4-1)}=0.5$; dividing the centred rewards by that gives [+0.5, −1.5, +0.5, +0.5]. Without the standard deviation and with a leave-one-out baseline, as in DeepSWE, they are [+0.333, −1, +0.333, +0.333]. In the episode-level approach this number is spread over every model-generated token in that trajectory; the prompt and tool outputs such as file contents and test logs have mask 0. In the teaching example a trajectory has 1,800 tokens, of which only 200 action tokens receive gradient. A shared step with the same context, tokens and forward computation has the same unweighted log-probability gradient, but positive and negative advantages make its weighted contributions point in opposite directions. If all 4 first steps are identical, their contributions sum to 0 under the main text's simplified objective with common weighting and normalisation and no additional clipping or loss terms; other steps can still change its probability through shared parameters.

</details>

<details class="interview" markdown="1">
<summary>Where does one score per trajectory fall short? What finer-grained options exist?</summary>

It cannot tell which step in a trajectory was good: a sensible file read inside a failed trajectory is pushed down along with everything else, and only many samples average out the noise. Finer options each have prerequisites. GiGPO (2025-05) finds states that recur within the group and compares the actions taken from the same state again, with no extra model, but degrades to GRPO when no state repeats. A turn-level critic (ArCHer, 2024-02) gives an estimate at every turn, at the cost of an extra model and estimation error. SWEET-RL (2025-03) lets the critic see information at training time that the policy cannot. Per-turn rewards give a denser signal, but the intermediate rewards have to be designed and can be exploited. Z.ai's GLM-5 (2026-02) still used group-relative advantages, while GLM-5.2 (2026-06) switched to critic-based PPO for long tasks.

</details>

<details class="interview" markdown="1">
<summary>Tests catch only some failures. How can training reward and true success diverge?</summary>

Expected reward equals the share of real fixes plus the shares of wrong fixes and tampering that slip through; true success counts only the first. In the teaching example, with test files the agent can edit and visible tests that miss some wrong fixes, reward rises from 0.55 to 0.90 while true success rises only from 0.30 to 0.35. In a group comparison, a trajectory that edits the tests gets the same positive advantage as a real fix. After restoring the test files before scoring and adding hidden tests, the same two policies score 0.34 and 0.40, close to the true change. "Edit the tests when stuck" is a recipe that works across tasks, so under a weak checker it grows especially fast on hard problems.

</details>

<details class="interview" markdown="1">
<summary>Give a few documented examples of agents exploiting their checkers, and the limits of the defences.</summary>

OpenAI's Baker et al. (2025-03) recorded agents in training using `exit(0)` to leave early and `raise SkipTest` to skip tests; Anthropic (2025-11) recorded objects whose `__eq__` is always true, `sys.exit(0)`, and a `conftest.py` that rewrites pytest's reports; SWE-bench images once kept future git history (2025-09); Claude Mythos Preview (2026-04) occasionally circumvented network restrictions in its training environment. Every defence has limits: read-only tests do not stop special-casing; hiding tests drives cheating close to 0 but makes normal tasks harder (ImpossibleBench, 2025-10); using a CoT monitor directly as a penalty, with too much optimisation, teaches agents to hide their intent; and Anthropic says its pre-training review of environments did not catch every problematic one.

</details>

<details class="interview" markdown="1">
<summary>How should a trajectory that hits the context limit, the step limit or a timeout be scored?</summary>

Three choices. Scoring it 0 teaches the model to hand in early and systematically pushes down long tasks. Dropping it from the loss loses negative samples but stops penalising "not finished yet"; DeepSWE's (2025-07) compact filtering does this, reasoning that rewarding only deliberate submissions encourages careful testing. Treating it as truncated and bootstrapping the remaining return with a value estimate needs a critic. Environment failures need separate handling: GLM-5 excludes samples that failed because a sandbox crashed. Whatever you choose, count limit hits, timeouts and tool errors separately; a falling truncation rate is not necessarily better ability.

</details>

<details class="interview" markdown="1">
<summary>How does the training system for agent RL differ from single-turn RLVR? Why go asynchronous, and what does it cost?</summary>

A large share of the time is spent off the GPU: starting containers, running commands, running tests. DeepSWE started 512 containers in parallel per iteration, and Qwen3-Coder reports 20,000 environments in parallel. Trajectory lengths vary widely, so waiting synchronously for the slowest one is wasteful, and generation is often decoupled from training: AReaL reports up to 2.77× speedup at the same number of GPUs, and Kimi-Researcher saves tasks that run over time and continues them later with new weights. The cost is data generated by an older policy, sometimes with early and late turns of one trajectory coming from different versions, so the log-probabilities at generation time must be recorded and corrected for. Training must also use the token ids that were generated: parsing into messages and re-encoding can change the tokens, and GLM-5 added a token-in-token-out gateway for this reason.

</details>

## Self-check

<div class="taste-check">
  <strong>Use these questions to check your understanding:</strong>
  <ol>
    <li>Starting from the decomposition of a trajectory's log-probability, why should tool-output tokens produce no gradient, yet stay in the context?</li>
    <li>With rewards [1, 0, 1, 1], what results do the three advantage conventions (population standard deviation, sample standard deviation, leave-one-out) give? Why does each sum to 0?</li>
    <li>When does a GiGPO anchor-state group add information beyond episode-level group-relative advantages, and when does it degrade to GRPO?</li>
    <li>In the "tests catch only some failures" example, how does the expected-reward formula show which outcome categories account for most of the reward gain under checker v1?</li>
    <li>Scoring a trajectory that hits a limit as 0, dropping it, or treating it as truncated and bootstrapping a value estimate: which way does each push the model?</li>
  </ol>
</div>

## References

- [Search-R1](https://arxiv.org/abs/2503.09516): multi-turn retrieval RL, with an ablation of keeping retrieved tokens out of the loss
- [GiGPO](https://arxiv.org/abs/2505.10978): step-level groups by recurring state, on top of episode-level advantages
- [ArCHer](https://arxiv.org/abs/2402.19446): multi-turn RL with an utterance-level critic driving a token-level policy
- [SWEET-RL](https://arxiv.org/abs/2503.15478): a critic with extra training-time information; ColBench
- [Reinforcing Multi-Turn Reasoning in LLM Agents (Wei, Zeng et al.)](https://arxiv.org/abs/2505.11821): terminal, delayed and per-turn rewards compared
- [RAGEN](https://arxiv.org/abs/2504.20073): the Echo Trap in multi-turn agent RL, and StarPO-S
- [DeepSWE](https://www.together.ai/blog/deepswe): training a coding agent with RL alone; GRPO++ and compact filtering
- [SWE-RL](https://arxiv.org/abs/2502.18449): rewards from patch similarity in a single-turn setting
- [SWE-Gym](https://arxiv.org/abs/2412.21139), [R2E-Gym](https://arxiv.org/abs/2504.07164), [SWE-smith](https://arxiv.org/abs/2504.21798): building executable code environments, and their scale
- [τ-bench](https://arxiv.org/abs/2406.12045), [τ²-bench](https://arxiv.org/abs/2506.07982): simulated users and final-state scoring
- [AReaL](https://arxiv.org/abs/2505.24298), [ROLL Flash](https://arxiv.org/abs/2510.11345): asynchronous RL systems
- [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/), [Kimi K2](https://arxiv.org/abs/2507.20534), [Kimi K2.5](https://arxiv.org/abs/2602.02276), [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3): Moonshot AI's agentic RL reports
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556): the evaluation-cost limitation; large-scale agent task synthesis
- [Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/), [GLM-5](https://arxiv.org/abs/2602.15763), [GLM-5.2](https://z.ai/blog/glm-5.2): environment scale, asynchronous systems and critic-based PPO
- [Monitoring Reasoning Models for Misbehavior (Baker et al.)](https://arxiv.org/abs/2503.11926): test exploits during training, and hiding intent in the chain of thought
- [Natural emergent misalignment from reward hacking](https://arxiv.org/abs/2511.18397): exploits in production coding environments and how they generalise
- [Recent Frontier Models Are Reward Hacking (METR)](https://metr.org/blog/2025-06-05-recent-reward-hacking/): instances of exploits during evaluation
- [ImpossibleBench](https://arxiv.org/abs/2510.20270): cheating rates when tests conflict with the specification, and the effect of test visibility
- [PatchDiff](https://arxiv.org/abs/2503.15223), [UTBoost](https://arxiv.org/abs/2506.09289): wrong "passes" caused by weak tests
- [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794), [Gaia2](https://arxiv.org/abs/2602.11964): LLM judges exploited when used as rewards
- System cards for [Claude 3.7 Sonnet](https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf), [Claude 4](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285/Claude%204%20System%20Card.pdf), [Claude Mythos Preview](https://www-cdn.anthropic.com/7624816413e9b4d2e3ba620c5a5e091b98b190a5/Claude%20Mythos%20Preview%20System%20Card.pdf) and [Claude Opus 5.5](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf): exploits in training environments and the defences
- [deep research](https://openai.com/index/introducing-deep-research/), [o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/), [codex-1](https://openai.com/index/introducing-codex/): OpenAI's statements about training agents with RL
