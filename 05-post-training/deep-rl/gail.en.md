# GAIL: learning expert-like behavior without a task reward

[中文](gail.md) · **English**

> Last reviewed: 2026-11 · Prerequisites: [Imitation and rewards](imitation-and-rewards.en.md), [Policy gradients](policy-gradients.en.md)

Behavioral cloning (BC) practices action selection at states visited by an expert. After a few steps, however, the learned policy may reach states the expert never encountered. Action-prediction accuracy on demonstrations says little about how far a rollout might drift.

Generative Adversarial Imitation Learning (GAIL) also examines data produced by the learner: which states and actions does the expert visit, and which does the current policy visit? It trains a discriminator to distinguish the two, then improves the policy through environment interaction. [Ho and Ermon (2016), §5](https://arxiv.org/html/1606.03476v1#S5) introduce this approach.

The environment need not supply a task reward, but it must still produce new trajectories. **A fixed expert dataset with no opportunity to interact is not the standard GAIL setting.**

## Actions on a dataset versus behavior during rollouts {#occupancy}

BC typically optimizes

$$
L_{\mathrm{BC}}=-\mathbb E_{(s,a)\sim\text{expert data}}\log\pi_\theta(a\mid s).
$$

GAIL concerns the state–action pairs actually visited by a policy. To make the distribution sum to one, we use a normalized discounted occupancy distribution:

$$
d_\pi(s,a)=(1-\gamma)\sum_{t=0}^\infty
\gamma^tP_\pi(s_t=s,a_t=a),\qquad 0\le\gamma<1.
$$

It includes both how often the policy reaches $s$ and how often it chooses $a$ there. Its mass is one in a continuing MDP; episodic tasks can include an explicit absorbing state. If samples stop at termination, total mass may depend on the policy. It is then not automatically the same normalized distribution.

The original paper uses an unnormalized measure $\rho$, with $d=(1-\gamma)\rho$. We normalize the entropy term consistently:

$$
H_d(\pi)=\mathbb E_{(s,a)\sim d_\pi}[-\log\pi(a\mid s)].
$$

A fixed-duration experiment can instead specify its own sampling times and weights. Averaging over entire trajectories does not by itself produce samples from the discounted distribution above.

## Start with two behaviors {#two-behaviors}

Temporarily reduce the environment to one state, two actions, and one decision. We compare only that decision here, equivalent to setting $\gamma=0$ above. The expert selects A and B with probabilities 0.75 and 0.25; the current policy does the reverse. Define **$D(s,a)$ as the probability of expert origin**: expert label 1, learner label 0.

| Action | Expert distribution $d_E$ | Learner distribution $d_\pi$ | Optimal discriminator $D^*$ |
| --- | ---: | ---: | ---: |
| A | 0.75 | 0.25 | 0.75 |
| B | 0.25 | 0.75 | 0.25 |

Why those discriminator outputs? Give the two classes equal weight and maximize

$$
J_D=\mathbb E_{d_E}\log D+\mathbb E_{d_\pi}\log(1-D).
$$

At a particular state–action pair, let the two distributions have masses $p,q$:

$$
\frac{\partial}{\partial D}[p\log D+q\log(1-D)]
=\frac pD-\frac q{1-D}=0
\quad\Rightarrow\quad D^*=\frac p{p+q}.
$$

Where $p=q=0$, any discriminator output gives the same objective. Unequal class weighting would also change the optimal probability; the table assumes equal weights.

## Turning discrimination into a policy reward {#reward}

With this label convention, the adversarial objective is

$$
\min_\pi\max_D\left[
\mathbb E_{d_E}\log D+
\mathbb E_{d_\pi}\log(1-D)-\lambda H_d(\pi)
\right].
$$

Hold the discriminator fixed for a policy update. The expert term does not depend on the policy parameters. Minimizing the learner term is equivalent to maximizing

$$
r_D(s,a)=-\log(1-D(s,a)),
$$

plus entropy regularization. A receives $-\log0.25\approx1.386$; B receives $-\log0.75\approx0.288$. The policy has a signal to increase A.

Let one logit $z$ determine A's probability $u=\sigma(z)$. Expected one-step reward is

$$
J(z)=u\,r_A+(1-u)r_B,\qquad
\frac{dJ}{dz}=u(1-u)(r_A-r_B).
$$

At $u=0.25$, the derivative is about 0.206, which is positive. This gives the update direction **with the current discriminator fixed**. Once the policy changes, the discriminator must be trained again.

The original GAIL paper uses the opposite convention: $D$ predicts learner origin, and the policy minimizes the cost $\log D$. Either convention works, but labels, loss, and reward must change together. Some implementations choose other discriminator-derived rewards; that is a separate training choice, with different gradients.

## Why this matches distributions {#js}

Let $m=(d_E+d_\pi)/2$ and substitute the optimal discriminator:

$$
J_D^*=
\sum d_E\log\frac{d_E}{d_E+d_\pi}
+\sum d_\pi\log\frac{d_\pi}{d_E+d_\pi}.
$$

The denominator is $2m$, giving

$$
J_D^*=-\log4+
D_{\mathrm{KL}}(d_E\|m)+D_{\mathrm{KL}}(d_\pi\|m)
=-\log4+2\,\mathrm{JS}(d_E,d_\pi).
$$

We use the standard Jensen–Shannon divergence definition with factors of $1/2$. This classification identity also appears in [Goodfellow et al. (2014), §4.1](https://arxiv.org/html/1406.2661v1#S4.SS1). Our example gives $\mathrm{JS}\approx0.130812$ and $J_D^*\approx-1.124670$.

When the distributions match, the discriminator outputs 0.5 wherever they have mass, and the objective is $-\log4$. The policy reward is nevertheless $-\log0.5=\log2$, **not zero**. Its absolute value is not a distance to the expert.

This argument assumes an optimal discriminator and accurately known distributions. Neural networks, finite samples, and alternating updates do not automatically satisfy those assumptions. With $\lambda>0$, the policy also trades off entropy. Discriminator accuracy near 50% might simply mean that it failed to learn; it is not sufficient evidence of successful imitation.

## What changes during a training iteration? {#training-loop}

<div class="drl-flow" aria-label="One GAIL training iteration">
<span>Run the policy<br><small>Collect states and actions</small></span><b>→</b><span>Compare with demonstrations<br><small>Update the discriminator</small></span><b>→</b><span>Freeze it to score rollouts<br><small>Update the policy with RL; sample again</small></span>
</div>

| Stage | Data and parameters | Check |
| --- | --- | --- |
| Sample | Collect fresh learner trajectories; expert data can remain fixed | Matching environments, observations, and action definitions |
| Train the discriminator | Label expert samples 1 and learner samples 0 | Class weights, sampling times, and preprocessing |
| Construct rewards | Hold this discriminator fixed and compute $r_D$ | Label direction and numerically stable logarithms |
| Update the policy | Apply RL using returns or advantages | A per-step discriminator reward is not a complete advantage |
| Evaluate behavior | Run tasks and inspect failures independently | Success, violations, and state coverage, not just discriminator loss |

The original policy update uses TRPO. Changing the optimizer preserves the basic alternation between discriminator and policy, but changes stability. With a non-differentiable environment, one cannot simply backpropagate through sampled actions and dynamics; policy gradients estimate updates from trajectories.

## Where apparent progress can mislead {#failure-modes}

**The discriminator recognizes the data source.** Expert and learner trajectories may have different padding, camera backgrounds, or termination markers. Accurate classification then supplies little useful behavioral signal. Align preprocessing and test changes that preserve behavior while altering backgrounds or encoding.

**The policy learns to delay termination.** With positive per-step rewards and accumulation that stops at termination, staying alive can earn extra reward. The $\log2$ reward even at distribution matching illustrates why termination matters. A fixed horizon, explicit absorbing states, and termination modeling have different assumptions. Arbitrarily subtracting a constant need not preserve behavior.

**The discriminator separates the classes too easily.** If learner and expert regions barely overlap, current samples may all receive nearly zero reward, giving little guidance about where to go. Theoretical reward at expert-only points can even diverge. Stable logit computations prevent numerical overflow, not missing exploration. BC initialization, regularization, and update ratios require experimental comparison.

**It works only in the original environment.** Different obstacles, dynamics, or goals may require a different visitation distribution. GAIL learns an imitation signal, not a guaranteed transferable “true reward.” Reward transfer is a separate objective to evaluate, not something established by a discriminator score.

## Choosing among related methods {#comparison}

| Method | Available supervision | Additional requirements | What it does not automatically provide |
| --- | --- | --- | --- |
| BC | Expert state–action pairs | Supervised fitting; deployment-shift checks | Recovery at unseen states |
| DAgger | An expert who can label new states | Learner execution and repeated expert queries | Free or risk-free data collection |
| MaxEnt IRL | Expert trajectories and reward assumptions | Computing or estimating model feature expectations | A uniquely identified true reward |
| GAIL | Expert trajectories | Learner interaction, discriminator fitting, and RL updates | Purely offline training, transferable rewards, or outperforming the expert |

“A human prefers response A to B” and “this state–action pair came from an expert” are different labels. Both can support classification losses, but imply different training distributions, rewards, and sampling costs. Having a discriminator does not make GAIL another name for ordinary RLHF.

## Checks and further reading {#checks}

The [companion program](code/reward_learning_checks.py) checks discriminator orientation, the two action rewards, the JS identity, and both matching and disjoint-support cases:

```bash
python3 05-post-training/deep-rl/code/reward_learning_checks.py
python3 -m unittest discover -s site/tests -p test_reward_learning_checks.py
```

These are mechanism checks, not a control-task reproduction or evidence of performance. Continue with [feature matching and planning in MaxEnt IRL](maxent-irl.en.md), or [reward-model ranking losses](../reward-ranking.en.md) for multiple responses to the same prompt.
