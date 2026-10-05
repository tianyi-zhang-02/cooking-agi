# Deep RL: from one decision to a training loop

[中文](README.md) · **English**

> Reading time: ~6–9 min · Last reviewed: 2026-10

You can take 2 points now, or wait one step for a possible 4. Which should you choose? **Reinforcement learning asks not just what a decision earns now, but where it leads.** We start with small questions like this, then work toward value estimates, policy updates, and complete training loops.

Each chapter starts with an example before introducing equations and code. On a first read, follow the examples; when reviewing, use the tables to compare algorithms and check implementations. The examples are original toy environments, not course assignments or reproduced slides.

## A route through the material

| Route | Read | What you should be able to do |
| --- | --- | --- |
| Define the problem | [MDPs and Bellman](mdp-bellman.en.md) → [MC, TD, and boundaries](returns-and-td.en.md) | Compute returns and training targets from a trajectory |
| Change the policy directly | [Policy gradients](policy-gradients.en.md) → [Actor–Critic and GAE](actor-critic-gae.en.md) → [Natural gradient and TRPO](trust-region.en.md) | Explain gradients, baselines, the two masks, and why step size needs control |
| Choose actions through value | [Fitted Q and convergence limits](fitted-q.en.md) → [DQN](dqn.en.md) → [DDPG / TD3](continuous-control.en.md) → [SAC](soft-actor-critic.en.md) | Separate regression loss from policy performance and trace network gradients |
| Think ahead with a model | [Model-based RL](model-based.en.md) → [Planning and LQR](planning-and-control.en.md) | Separate dynamics learning, planning, and policy learning |
| Understand data and rewards | [Offline RL and OPE](offline-and-ope.en.md) · [Exploration and hierarchy](exploration-and-hierarchy.en.md) · [Imitation and rewards](imitation-and-rewards.en.md) | State the added assumptions, not just the algorithm names |
| Run a meaningful experiment | [Experiments and debugging](experiments.en.md) → [The LLM bridge](llm-bridge.en.md) | Check the implementation before interpreting reward as progress |

The foundations and core algorithms develop equations and worked examples. Offline RL, hierarchy, IRL, and multi-agent learning are currently **introductions and trade-offs**, not comprehensive surveys. Proofs that are not developed here are explicitly identified.

## Find these 3 parts in any algorithm

<div class="drl-flow" aria-label="Reinforcement learning loop">
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

Conditional expectation, the chain rule, and gradients are enough to start. Revisit the [probability guide](../../quant/probability/study-guide.en.md) when needed, especially conditional and total expectation. The PyTorch examples assume familiarity with autograd and tensor shapes.

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

For terminology and derivations, see [Spinning Up: RL foundations](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html); individual chapters link original papers. PPO and RLHF already have a separate sequence. Continue there after these foundations rather than memorizing the same definitions twice.
