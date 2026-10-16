# Technology timeline: RL and post-training

[中文](rl-timeline.md) · **English**

[← Back to the technology tree](technology-timeline.en.md#branch-rl)

> Reading time: about 9–12 minutes · Last reviewed: 2026-10

Why does RL still feel difficult after decades of research? Follow the problems it has tackled: estimating values for a few actions became generating long trajectories with large models. Objectives, feedback sources, and systems costs changed along the way.

This page follows **one line from RL to LLM post-training**, not the whole history of AI. Dates refer to the original publication or public release; later revisions do not change that date. Each entry asks what problem motivated the work, what changed, and what tradeoff remains. Detailed equations live in linked chapters. You do not need to memorize the timeline first.

<nav class="timeline-jumps" aria-label="Browse the technology timeline by period">
<a href="#learning">1989–2015 · Learning to act</a>
<a href="#feedback">2017–2024 · Learning from feedback</a>
<a href="#scale">2025–2026 · Cost and stability</a>
</nav>

## Learning values and changing policies {#learning}

<div class="tech-timeline" markdown="1">

### 1989 / 1992 · Q-learning: learn without a transition model {#q-learning}

With known dynamics, dynamic programming can compute values. With experience alone, an agent needs to update estimates from observed transitions. Q-learning connects immediate reward to the maximum estimated value at the next state. Watkins introduced the method in 1989; the Watkins–Dayan convergence paper appeared in 1992. Those are different milestones.

Tabular conditions do not transfer unchanged to large networks. Bootstrapping, off-policy data, and unstable value estimates remain relevant today. [Original paper](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf) · [Calculate a Q update](../05-post-training/deep-rl/methods-map.en.md#q-update)

### 1992 · REINFORCE: change action probabilities directly {#reinforce}

Another route adjusts a stochastic policy rather than first selecting the largest Q. Williams’s REINFORCE work described parameter adjustments along the gradient of expected reward. This idea did not originate with language-model training.

Direct probability updates do not make a single sample a reliable direction estimate. Baselines and variance help explain why Actor–Critic methods matter. [Original paper](https://gwern.net/doc/reinforcement-learning/model-free/1992-williams.pdf) · [Policy gradients](../05-post-training/deep-rl/policy-gradients.en.md)

### 2013 · Deep Q-learning: pixels instead of a small table {#dqn}

Atari screens cannot be handled by filling out a small state table. Early DQN estimated action values with a neural network and sampled stored transitions through experience replay, addressing correlated observations and changing data distributions. This entry covers the 2013 report, not every design added in later versions.

More expressive representations expanded the tasks an agent could tackle while complicating estimation and stability. Replay, targets, and data distributions are more useful lessons than a game score alone. [Original paper, §§2 and 4](https://arxiv.org/html/1312.5602) · [DQN and Double DQN](../05-post-training/deep-rl/dqn.en.md)

### 2015 · TRPO: large changes make old-policy evidence unreliable {#trpo}

Judging a new policy with data from an old one becomes harder as the policies diverge. TRPO develops a practical trust-region algorithm from policy-improvement analysis. Its practical approximations are not an unconditional monotonic-improvement guarantee.

This line leads into PPO. Current discussions of KL, probability ratios, and update size still concern the same underlying problem. [Original paper, §3–5](https://arxiv.org/html/1502.05477) · [Natural gradient and TRPO](../05-post-training/deep-rl/trust-region.en.md)

</div>

## The algorithm also needs someone to judge the outcome {#feedback}

<div class="tech-timeline" markdown="1">

### 2017 · Human preferences: learn rewards from comparisons {#human-preferences}

Some behavior is hard to specify as a reward function but easier to recognize in a comparison. Christiano and colleagues learned rewards from preferences over trajectory segments, then optimized policies with RL. Their experiments included Atari and simulated robotics, not chat models.

Human feedback with RL predates current LLMs. A learned scorer remains an approximation whose omissions a policy may exploit. [Original paper, §2.2](https://arxiv.org/html/1706.03741) · [Imitation and rewards](../05-post-training/deep-rl/imitation-and-rewards.en.md)

### 2017 · PPO: make policy updates easier to implement {#ppo}

PPO studies policy objectives suitable for first-order optimization; its clipped surrogate is a widely used option. It simplifies parts of the update rather than solving all reward-design and sampling problems.

Being more convenient than TRPO does not make clipping a hard bound on actual KL. Advantages and update sizes still need checking. [Original paper, §3](https://arxiv.org/pdf/1707.06347) · [PPO clipping](../05-post-training/rlhf/ppo-clipping.en.md)

### 2022 · InstructGPT: connect feedback to instruction following {#instructgpt}

Continuing text is not the same as answering a user’s request. InstructGPT combined demonstration fine-tuning, a preference reward model, and PPO in an instruction-training pipeline. It built on earlier feedback-learning work rather than originating RLHF.

Annotators, prompt sources, and definitions of reliable behavior matter alongside the optimizer. [Original paper, §3](https://arxiv.org/html/2203.02155v1) · [What the three stages do](../05-post-training/rlhf/three-stages.en.md)

### 2023 · DPO: can preference pairs avoid an online training loop? {#dpo}

DPO relates a particular KL-regularized reward objective to a preference model, yielding a pairwise policy loss. Standard offline training needs no ongoing rollouts inside its optimization loop.

This is another preference-learning route, not a higher version of PPO. Removing stages does not remove dataset bias. [Original paper, §3–4](https://arxiv.org/html/2305.18290v3) · [DPO example](../05-post-training/deep-rl/methods-map.en.md#dpo-pair)

### 2024 · GRPO: a group of responses instead of a learned value baseline {#grpo}

DeepSeekMath estimates relative quality from multiple responses to the same question, reducing the need for a separate value model. GRPO predates DeepSeek-R1.

Saving some Critic overhead shifts attention to group-sampling cost and signal quality. All-correct, all-incorrect, and highly correlated responses still need care. [Original paper, §4.1](https://arxiv.org/html/2402.03300v3) · [Within-group signal](../05-post-training/deep-rl/llm-bridge.en.md#sparse-groups)

</div>

## At scale, cost and stability become harder to separate {#scale}

<div class="tech-timeline" markdown="1">

### 2025 · DeepSeek-R1: verifiable feedback for reasoning training {#r1}

R1-Zero demonstrated RL without an initial SFT stage in its base-model and task setting. The full R1 pipeline also includes cold-start data and multiple training stages. The two should not be merged into a single “RL only” claim. The work first appeared in 2025; the link here is the later v2 report.

Verifiable mathematics and coding feedback is useful, but it does not establish that open-ended tasks can use the same recipe. Comparisons also need fixed inference sampling and length budgets. [Report, §2–3](https://arxiv.org/html/2501.12948v2) · [RL for LLMs and agents](../05-post-training/deep-rl/llm-bridge.en.md) · [How long chains of thought are trained](../11-frontier/test-time-compute/long-chains.en.md)

### 2025 · DAPO, AReaL, GSPO: sampling, pipelines, and updates {#engineering}

These are not three renamings of one algorithm. [DAPO](https://arxiv.org/html/2503.14476v2) addresses sampling and optimization details; [AReaL](https://arxiv.org/html/2505.24298v2) studies asynchronous generation and training; [GSPO](https://arxiv.org/html/2507.18071v1) changes sequence-level policy updates. Sharing a year does not mean solving the same problem.

Ask what gets saved—generation, computation, or waiting—and which objective or data distribution changes. Each reported benefit has its own experimental conditions. [Cost and stability breakdown](../05-post-training/deep-rl/why-rl-is-hard.en.md)

### 2026 · Current checkpoint: engineering limits have not disappeared {#current-checkpoint}

This is the site’s review checkpoint as of 2026-10-11, not a new algorithm called “2026 RL.” [Current verl documentation](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) still describes correcting for the gap between training and inference distributions; [AReaL’s asynchronous guide](https://areal-ai.io/docs/en/algorithms/async.html) still describes how to bound policy staleness.

A usable framework does not imply a reliable recipe for every task. Future updates should identify what new work changes, not attach a new year to an old method. [How to investigate a run](../05-post-training/deep-rl/why-rl-is-hard.en.md#diagnosis)

</div>

## Change the environment, change the hard part {#domains}

Games, robots, and language models can all follow “observe → act → receive feedback.” That does not make their training recipes interchangeable.

| Setting | One action | Feedback and sampling | Particular risks |
| --- | --- | --- | --- |
| Games such as Atari | Choose a button | Simulator scores; episodes can be restarted | Assigning credit to earlier actions; overfitting to training environments |
| Robot control | Joint torques, velocities, or action sequences | Simulation or hardware, sometimes with manual resets | Safety, sample cost, and the simulation-to-reality gap |
| LLM post-training | A token, or a higher-level tool call | Often scored after generation; tools may give intermediate feedback | Long trajectories, reward loopholes, generation cost, and evaluation-to-use gaps |

### Robotics: reusing data matters, and failures have a cost {#robot-control}

Consider a robot learning to place a cup on a tray. A failure costs more than a low score: someone may need to replace the cup and check the equipment before another attempt. That is why **sample efficiency** matters, not just gradient updates per second.

In 2018, [SAC](https://arxiv.org/html/1801.01290) combined off-policy data reuse with a maximum-entropy objective for continuous control. Old trajectories can contribute to training, but not all old data is equally useful. State coverage, value-estimation errors, and changes in the environment still matter. Simulation reduces collection costs without automatically closing the sim-to-real gap.

In 2023, [RT-2](https://arxiv.org/abs/2307.15818) showed a different connection: bringing vision-language knowledge and robot demonstrations into action prediction. This is a [VLA](../13-embodied/robot-learning/vla.en.md) example, not RL merely because the output is an action. A working system may combine demonstrations, planning, RL, and safety control; these components solve different problems.

Continue with [continuous control](../05-post-training/deep-rl/continuous-control.en.md) → [SAC's objective](../05-post-training/deep-rl/soft-actor-critic.en.md) → [planning and control](../05-post-training/deep-rl/planning-and-control.en.md).

### Language models: start with a generator, then choose the feedback {#llm-post-training}

Suppose a model can already write a sorting function. To make it more reliable, you could use correct code for SFT, collect preferred/rejected pairs for DPO, or generate code with the current policy, run tests, and perform online RL. The task is similar; **where data comes from, when it is generated, and how updates use it** are different.

This branch connects 2017 [human preference learning](#human-preferences), 2022 [InstructGPT](#instructgpt), 2023 [DPO](#dpo), and later reasoning work with verifiable feedback. DPO is a related preference-optimization branch, not a mandatory next generation of RL. Verifiable rewards do not require GRPO either.

Tokens are discrete actions, but that does not make them cheap. Eight long answers to one prompt require eight autoregressive generations. When the score arrives only at the end, credit assignment remains difficult. Tests check some behaviors; passing finite tests is not proof of correctness for every input. Tool use adds execution costs and safety constraints; for how multi-step tasks are trained, see [agent RL](../11-frontier/agentic-rl/README.en.md).

Continue with [Q-learning, PPO, and DPO](../05-post-training/deep-rl/methods-map.en.md) → [LLM / agent RL](../05-post-training/deep-rl/llm-bridge.en.md) → [rollout cost and stability](../05-post-training/deep-rl/why-rl-is-hard.en.md).

## Return to the problem you want to understand {#read-next}

| Your question | Continue with |
| --- | --- |
| Why is “which is more advanced?” a poor comparison of Q-learning and DPO? | [Methods map](../05-post-training/deep-rl/methods-map.en.md) |
| Why can so much generation produce so little learning? | [Learning signal and rollout cost](../05-post-training/deep-rl/why-rl-is-hard.en.md) |
| What about continuous control and offline RL, not developed here? | [SAC](../05-post-training/deep-rl/soft-actor-critic.en.md) · [Offline RL](../05-post-training/deep-rl/offline-and-ope.en.md) · [OPE estimators](../05-post-training/deep-rl/ope-estimators.en.md) |
| How should the data and evaluation behind these methods be designed? | [Data as supervision](../01-data-and-feedback/data-as-supervision.en.md) · [Evaluation stack](../07-evaluation/evaluation-stack.en.md) |

This timeline selects milestones that help explain current post-training choices. Dynamic programming, control, exploration, model-based RL ([which model to learn](../05-post-training/deep-rl/model-types.en.md), [planning and learning](../05-post-training/deep-rl/planning-and-learning.en.md)), and multi-agent learning have much longer histories. Omission does not imply irrelevance. When reading a new method, keep asking which old problem it solves and where it moves the cost.
