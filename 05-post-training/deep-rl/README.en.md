# Deep RL: from one decision to a training loop

[中文](README.md) · **English**

> Reading time: ~8–10 min · Last reviewed: 2026-11

You can take 2 points now, or wait one step for a possible 4. Which should you choose? **Reinforcement learning asks not just what a decision earns now, but where it leads.** We start with small questions like this, then work toward value estimates, policy updates, and complete training loops.

Each chapter starts with an example before introducing equations and code. On a first read, follow the examples; when reviewing, use the tables to compare algorithms and check implementations. The examples are original toy environments, not course assignments or reproduced slides.

If the names blur together, start with [Q-learning, PPO, and DPO](methods-map.en.md). For current difficulties, read [signal, cost, and stability](why-rl-is-hard.en.md). For their origins, follow the technology tree into the [RL timeline](../../learn/rl-timeline.en.md).

## A route through the material

| Route | Read | What you should be able to do |
| --- | --- | --- |
| Define the problem | [MDPs and Bellman](mdp-bellman.en.md) → [MC, TD, and boundaries](returns-and-td.en.md) | Compute returns and training targets from a trajectory |
| Change the policy directly | [Policy gradients](policy-gradients.en.md) → [Actor–Critic and GAE](actor-critic-gae.en.md) → [Natural gradient and TRPO](trust-region.en.md) | Explain gradients, baselines, the two masks, and why step size needs control |
| Choose actions through value | [Fitted Q and convergence limits](fitted-q.en.md) → [DQN](dqn.en.md) → [DDPG / TD3](continuous-control.en.md) → [SAC](soft-actor-critic.en.md) | Separate regression loss from policy performance and trace network gradients |
| Think ahead with a model | [Model-based RL](model-based.en.md) → [Which model to learn](model-types.en.md) → [Planning and LQR](planning-and-control.en.md) → [Combining planning and learning](planning-and-learning.en.md) | Separate dynamics learning, planning, and policy learning; say what a model should predict and how search results become training targets |
| Understand data and rewards | [Offline RL and OPE](offline-and-ope.en.md) → [Per-decision weights, doubly robust, and LSTD](ope-estimators.en.md) · [Exploration and hierarchy](exploration-and-hierarchy.en.md) · [Imitation and rewards](imitation-and-rewards.en.md) | State the added assumptions, not just the algorithm names |
| Learn from demonstrations | [Imitation and rewards](imitation-and-rewards.en.md) → [MaxEnt IRL](maxent-irl.en.md) → [GAIL](gail.en.md) · [Reward-model ranking losses](../reward-ranking.en.md) | Fit feature expectations, calculate a discriminator reward, and distinguish pairwise from full-ranking supervision |
| Run a meaningful experiment | [Experiments and debugging](experiments.en.md) → [The LLM bridge](llm-bridge.en.md) | Check the implementation before interpreting reward as progress |

These lessons go beyond algorithm names. We work through Bellman residual bounds, TD targets, gradient directions, PPO clipping, and control trajectories, then examine how CQL, IQL, and FQE handle offline data. The imitation sequence adds reward fitting and distribution matching through worked examples. Hierarchy and multi-agent learning remain introductions rather than comprehensive theoretical treatments.

For language-model training, continue with [RLOO's leave-one-out baseline](../rloo.en.md), then [audit how a batch becomes an RL loss](../../practice/post-training/rl-batch-audit.en.md). The first adds a baseline proof and worked arithmetic; the second checks grouping, token shifting, EOS, and denominators. Neither requires a multi-GPU setup.

## Six questions for reviewing this sequence {#reward-review}

If you know policy gradients and want to prepare for a technical discussion or debug training code, start here. Each row links to a specific calculation or derivation, so you can revisit one point without rereading the whole chapter. If baselines, returns, and advantages are still unfamiliar, follow the foundations route above first.

| Question | What your explanation should include | Derivation or worked example |
| --- | --- | --- |
| Same-prompt rewards are 1, 0, and 0.5. What are the leave-one-out advantages, and why exclude each sample itself? | The advantages are 0.75, −0.75, and 0. Unbiasedness uses independent sampling conditional on the prompt and a zero-mean score function; subtracting any mean is not equivalent | [RLOO calculation and proof](../rloo.en.md#why-leave-one-out) |
| Does assigning higher reward to demonstrated routes necessarily improve an IRL model? | Feasible alternatives matter, so the partition function cannot be dropped. The three-route gradient is $1.6-10/7$; it matches a feature expectation, not every route frequency | [The MaxEnt IRL gradient](maxent-irl.en.md#gradient) |
| Does 50% discriminator accuracy establish successful GAIL imitation? | It can indicate matching distributions or a weak discriminator. Check independent task outcomes too; at ideal matching, our per-step reward is $\log2$, not zero | [Discrimination and distribution matching](gail.en.md#js) |
| Can a ranking of three responses be treated as three independent preference pairs? | Pairwise loss is a valid choice but differs from full-ranking PL loss. State whether prompts or pairs receive equal weight, so extra candidates do not silently dominate training | [Pairwise versus ranking objectives](../reward-ranking.en.md#pairs-versus-ranking) |
| If EOS and PAD share an ID, can every occurrence of that ID be masked out? | No: a generated EOS is an action; padding is not. Build masks from generation boundaries and align logits with the following token | [Token shifting and EOS checks](../../practice/post-training/rl-batch-audit.en.md#alignment) |
| For per-token losses $[2,4]$ and $[1,1,1,1]$, why do three means differ? | Mean sequence sums give 5, mean per-response means give 2, and the valid-token mean gives $5/3$. The denominator changes sample weights; microbatching should preserve the chosen objective | [Worked loss reductions](../../practice/post-training/rl-batch-audit.en.md#reduction) |

After reproducing the numbers, change an assumption: correlated samples, unequal candidate counts, variable trajectory lengths, or a reward-service timeout. Which step still works, and which needs a new definition? Explaining those changes reveals more than naming an algorithm.

## Different reading goals, not separate sets of notes

| What you want today | How to read | What to take away |
| --- | --- | --- |
| Build intuition | Follow the opening task, diagrams, and calculation tables; skip derivations for now | Explain why a component is needed |
| Understand the mechanism | Revisit equations, conditional expectations, symbols, and assumptions | Calculate an example and locate the limits of a guarantee |
| Implement it | Inspect data sources, targets, gradients, masks, and update timing | Debug in a tiny environment rather than relying only on curves |

Key calculations are visible in the text without moving sliders or opening an interactive panel. A few flows briefly highlight reading order; all content remains visible afterward, and reduced-motion settings are respected. Existing interactive experiments remain optional ways to change parameters, not substitutes for the lesson.

## Find these 3 parts in any algorithm

<div class="drl-flow drl-sequence" aria-label="Reinforcement learning loop">
<span>Sample<br><small>Which policy collects data?</small></span><b>→</b><span>Estimate<br><small>What is the target?</small></span><b>→</b><span>Update<br><small>Where do gradients go?</small></span>
</div>

| Method | Main learned object | Action selection | An easily missed cost |
| --- | --- | --- | --- |
| REINFORCE | Policy | Sample from the policy | High return variance; old data cannot be mixed in naively |
| DQN | Q-value | Discrete argmax with exploration during training | Bootstrapping, approximation, and off-policy data can destabilize learning |
| State-value Actor–Critic | Policy + V | Sample from the policy | Critic errors affect the Actor |
| TD3 / SAC | Policy + two Q-functions | A continuous-action policy | Reusing data saves interaction, not necessarily computation |
| Model-based | Dynamics, sometimes policy / value | Planning or policy | Accurate predictions do not ensure reliable planned behavior |

On-policy / off-policy compares **the behavior policy collecting data with the target policy being evaluated or improved**. Online / offline describes **whether training can obtain new interactions**. A method that keeps interacting while learning from older data can be both online and off-policy.

## Prerequisites and notation

Conditional expectation, the chain rule, and gradients are enough to start. For a refresher on probability and gradients, see [ML mathematics](../../00-foundations/ml-math-interview.en.md). The PyTorch examples assume familiarity with autograd and tensor shapes.

Throughout, $s_t$ is the state, $a_t$ the action, $r_t$ the reward received after that action, and $\gamma$ the discount. Finite episodes use $t=0,\ldots,T-1$; terminal value is zero. We use $\pi_\theta$ for a stochastic policy and $\phi$ for value parameters. Reward, return, and value are not interchangeable.

These terms recur throughout the notes:

| Term | Meaning here |
| --- | --- |
| Policy | Chooses an action or assigns action probabilities given a state |
| Reward | Feedback after one step, not the entire trajectory's outcome |
| Return | Future rewards accumulated from a step using the chosen convention |
| Value | Expected future return under a particular policy |
| Training target | The number fitted in this update; it may be an estimate, not ground truth |
| Bootstrapping | Using an existing value estimate for the future not yet observed |
| Trajectory / rollout | A sequence of states, actions, and rewards; a rollout may be only a segment |

## Change one assumption at a time

After a worked example, change just one condition: replace termination with a timeout, move $\lambda$ from 0 to 1, or make two Q-networks rank actions differently. The small interactive experiments are for exactly this. They illustrate computations, not complete training runs or invented performance gains.

Run the [reference calculations](code/rl_checks.py) to check the arithmetic, then the [PyTorch checks](code/torch_updates.py) to inspect stop-gradient behavior. Neither requires a GPU.

## Start with an experiment if you prefer

| Question | Chapter | What to change |
| --- | --- | --- |
| How does reward propagate backward? | [MDPs and Bellman](mdp-bellman.en.md) | Adjust the discount and step through backups |
| Which future steps affect this advantage? | [Actor–Critic and GAE](actor-critic-gae.en.md) | Adjust λ and inspect the contribution table |
| Why can better regression still diverge? | [Fitted Q](fitted-q.en.md) | Advance fits and compare value curves with MSE |
| How much does one policy update change? | [Natural gradient and TRPO](trust-region.en.md) | Adjust the logit step and KL budget |
| How does feedback help a wrong model? | [Model-based RL](model-based.en.md) | Adjust model error and horizon; compare real trajectories |

After changing a parameter, try explaining the result before opening the calculation table. The point is not to memorize which way a curve moves, but to see how a change carries through the calculation.

## Public references

For terminology and derivations, see [Spinning Up: RL foundations](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html); individual chapters link original papers. PPO and RLHF already have a separate sequence. Continue there after these foundations rather than memorizing the same definitions twice. To see these methods applied to multi-step agents and world models, continue to [training agents with RL](../../11-frontier/agentic-rl/README.en.md) and [world models](../../11-frontier/world-models/README.en.md) in the frontier series.
