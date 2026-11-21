# MaxEnt IRL: inferring rewards from demonstrations

[中文](maxent-irl.md) · **English**

> Last reviewed: 2026-11 · Prerequisites: [Imitation and rewards](imitation-and-rewards.en.md), [MDPs and Bellman equations](mdp-bellman.en.md)

Watching someone choose a route ten times tells you which routes they take, but not necessarily why. They might like walking by the river, or other roads might have been closed. Inverse reinforcement learning (IRL) tries to infer rewards from behavior. It needs a model of both the available choices and how rewards affect those choices.

Maximum entropy IRL provides one such probabilistic model. We will calculate probabilities for three routes, see why the partition function requires planning, and then examine what changes in a stochastic environment. All numbers are illustrative.

## What can three routes tell us? {#routes}

Fix the starting point, destination, and a two-step horizon. Transitions are deterministic. Each route has one measurable feature $f(\tau)$: its number of riverside segments. Ten demonstrations give:

| Route | Feature $f(\tau)$ | Demonstration count | Observed frequency |
| --- | ---: | ---: | ---: |
| A | 0 | 1 | 0.1 |
| B | 1 | 2 | 0.2 |
| C | 2 | 7 | 0.7 |

Write the total route reward as $R_\theta(\tau)=\theta f(\tau)$. There is one parameter: positive $\theta$ favors more riverside segments. Time, safety, and congestion are absent from this model, so it can answer only this limited question.

Start with $\theta=\ln2$ and make route probabilities proportional to exponentiated rewards:

$$
p_\theta(\tau)=\frac{\exp(\theta f(\tau))}{Z(\theta)},\qquad
Z(\theta)=\sum_{\tau'\in\mathcal T}\exp(\theta f(\tau')).
$$

The weights are $1,2,4$, so $Z=7$ and the probabilities are $1/7,2/7,4/7$. The model's average number of riverside segments is

$$
\mathbb E_\theta[f]=0\frac17+1\frac27+2\frac47=\frac{10}{7}\approx1.429.
$$

The demonstrations average $(0+2+14)/10=1.6$. The model does not favor riverside routes as much as the demonstrations do, suggesting an increase in $\theta$.

We are fitting an **expected feature**, not requiring all three route probabilities to become 0.1, 0.2, and 0.7. A single parameter cannot generally fit all those frequencies.

## Where does the exponential form come from? {#entropy}

For this finite set of deterministic routes, maximize entropy subject to normalization and the observed feature mean $\bar f$:

$$
\max_p-\sum_\tau p_\tau\log p_\tau
\quad\text{s.t.}\quad
\sum_\tau p_\tau=1,\quad \sum_\tau p_\tau f(\tau)=\bar f.
$$

This does not claim that the expert deliberately randomizes. It avoids imposing extra restrictions on route probabilities when all we know is a feature mean. The Lagrangian is

$$
\mathcal L=-\sum_\tau p_\tau\log p_\tau
+\theta\left(\sum_\tau p_\tau f(\tau)-\bar f\right)
+\eta\left(\sum_\tau p_\tau-1\right).
$$

Differentiate with respect to $p_\tau$ and set the result to zero:

$$
-(\log p_\tau+1)+\theta f(\tau)+\eta=0.
$$

Thus $p_\tau=\exp(\theta f(\tau)+\eta-1)$. Normalization gives the exponential model above. The constraint multiplier $\theta$ becomes a linear reward weight. At a boundary of the feasible feature means, an optimal distribution may require parameters to diverge; noisy data also motivate regularization or relaxed matching constraints.

Earlier [apprenticeship learning (Abbeel and Ng, 2004)](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf) already asked the learned policy to match the expert's feature expectations. [Ziebart et al. (AAAI 2008)](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf) point out that when demonstrations are not perfectly optimal, many route distributions match those expectations; maximum entropy picks the one with no further preference among routes, which gives the MaxEnt IRL trajectory model. We use our small example to work through the calculation, rather than reproduce their navigation experiments.

## Why is the gradient observed minus predicted? {#gradient}

The average demonstration log-likelihood is

$$
\ell(\theta)=\theta\bar f-\log Z(\theta).
$$

The first term rewards observed features. The second compares them against **all feasible routes**. Differentiating gives

$$
\frac{\partial\log Z}{\partial\theta}
=\frac{\sum_\tau f(\tau)e^{\theta f(\tau)}}{Z}
=\mathbb E_\theta[f],
\qquad
\nabla_\theta\ell=\bar f-\mathbb E_\theta[f].
$$

Without $\log Z$, raising $\theta$ would keep increasing the demonstrations' linear reward without any penalty. That would no longer fit a normalized behavior model.

One gradient-ascent step with learning rate 0.5 gives:

| Quantity | Before | After |
| --- | ---: | ---: |
| $\theta$ | 0.693147 | 0.778861 |
| Model feature mean | 1.428571 | 1.472815 |
| Average log-likelihood | −0.836875 | −0.824095 |

The update is $0.5(1.6-10/7)=0.085714$. Likelihood improves, but one step has not finished the fit.

Differentiating again,

$$
\nabla_\theta^2\ell=-\operatorname{Var}_{p_\theta}[f]\le0.
$$

Before the update this is $-26/49$. With vector-valued linear features, the Hessian is the negative covariance matrix. The log-likelihood is therefore concave for a fixed feasible trajectory set and linear rewards. This does not establish concavity for neural rewards or approximate planning.

## Avoid enumerating every route {#soft-planning}

Represent the three routes as a two-step tree, paying each route's total reward at the second step. Both first actions have zero immediate reward. The left branch leads only to A; the right branch allows B or C. The left branch has one valid second action, keeping all three routes the same length.

<div class="drl-flow" aria-label="Partition function for three routes">
<span>Start<br><small>Total weight 7</small></span><b>→</b><span>Left: A<br><small>Weight 1</small></span><b>/</b><span>Right: B or C<br><small>Weight 2 + 4 = 6</small></span>
</div>

The first action goes right with probability $6/7$. Conditional on going right, B and C have probabilities $1/3$ and $2/3$. Multiplication recovers $2/7$ and $4/7$, exactly as enumeration did.

With deterministic transitions and a fixed finite horizon, a soft Bellman recursion computes these weights:

$$
V_T(s)=0,\qquad Q_t(s,a)=r_t(s,a)+V_{t+1}(s'),
$$
$$
V_t(s)=\log\sum_{a\in\mathcal A(s)}e^{Q_t(s,a)},\qquad
\pi_t(a\mid s)=e^{Q_t(s,a)-V_t(s)}.
$$

The right branch has $V=\log(e^{\ln2}+e^{2\ln2})=\log6$; the root has $V=\log(1+6)=\log7$. An action's probability accounts for its possible continuations, not just its immediate reward.

Once the policy is available, propagate state probabilities forward:

$$
d_0(s)=p_0(s),\qquad
d_{t+1}(s')=\sum_{s,a}d_t(s)\pi_t(a\mid s)P(s'\mid s,a).
$$

The model feature expectation is $\sum_{t,s,a}d_t(s)\pi_t(a\mid s)f_t(s,a)$. Each reward-fitting iteration therefore includes a backward pass to obtain a policy and a forward pass to compute visitation frequencies. At scale, this inner computation can cost much more than updating the reward parameters.

With unknown transitions and continuous states, even this recursion is unavailable. [Guided Cost Learning (Finn et al., 2016), §4.1–4.2](https://arxiv.org/html/1603.00448v3#S4.SS1) estimates the partition function from samples instead, generated by a policy that keeps optimizing against the current reward; the reward itself becomes a neural network.

## A policy cannot change its luck {#causal-entropy}

Now consider one decision with known rewards, to check what happens when outcomes are stochastic:

| Action | Outcome | Expected reward |
| --- | --- | ---: |
| Safe | Always 1 | 1 |
| Risky | Equal chances of 0 and 2 | 1 |

Suppose we weight complete outcome trajectories by $e^R$, using their environment probabilities as base weights. The risky action gets weight $(1+e^2)/2$, larger than the safe action's $e$. Normalization assigns roughly 0.607 probability to taking the risk.

The issue goes beyond favoring the risky action. In that trajectory distribution, **conditional on taking the risk, the chance of receiving 2 becomes $e^2/(1+e^2)\approx0.881$**. In the real environment it remains 0.5. A policy controls the action, not the coin.

Maximum causal entropy models randomization over actions using information available at the time, while holding environment transitions fixed. [Ziebart, Bagnell, and Dey (ICML 2010), §3.1](https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf) address this information structure. For a finite horizon and entropy coefficient 1:

$$
Q_t(s,a)=r_t(s,a)+\mathbb E_{s'\sim P(\cdot\mid s,a)}V_{t+1}(s'),\qquad
V_t(s)=\log\sum_a e^{Q_t(s,a)}.
$$

If rewards depend on the random outcome, include them inside the conditional expectation as well. Both actions in our example have $Q=1$, so the policy splits equally between them. Conditional success on the risky action stays at 0.5. The objective is expected reward plus action entropy, not advance knowledge of future outcomes.

The easily confused expressions are $\log\mathbb E[e^V]$ and $\mathbb E[V]$. They agree for deterministic outcomes but generally differ otherwise. The earlier three-route model cannot be extended to stochastic control without additional care.

[Levine (2018), §2.3](https://arxiv.org/html/1805.00909v3#S2.SS3) describes the same effect in the control-as-inference framework: inferring whole trajectories directly gives a $\log\mathbb E[e^V]$ backup that favors risk. [§3](https://arxiv.org/html/1805.00909v3#S3) explains why: it amounts to letting the agent alter the transitions as well. Holding the dynamics fixed returns the backup to $\mathbb E[V]$.

## What remains unidentified? {#limits}

| Situation | Why it is ambiguous | What to check |
| --- | --- | --- |
| Add the same constant to every complete trajectory reward | Numerator and denominator scale equally; probabilities stay fixed | Do not interpret the reward origin as measured utility |
| Add a constant per step instead | Different lengths receive different total shifts | Include termination and time costs in the model |
| Add a potential difference $\Phi(s')-\Phi(s)$ at every step | A route gains only $\Phi(\text{end})-\Phi(\text{start})$; with shared start and end, probabilities stay fixed | Test under different dynamics; [AIRL (Fu et al., 2017), §5](https://arxiv.org/html/1710.11248v2#S5) shows such shaped rewards may stop working when the environment changes |
| Two features always occur together | Data cannot distinguish their contributions | Seek demonstrations that separate them |
| The expert did not know a route or could not access it | The assumed choice set is wrong | Check dynamics, information, and available actions |
| Both reward and temperature $\alpha$ are free | $e^{R/\alpha}$ identifies only their relative scale | Fix a temperature convention before interpreting magnitudes |

Even a good feature fit does not establish what the expert “really wanted.” It produces a model that explains behavior under stated assumptions.

## Reproduce the calculation and continue {#checks}

Run the [companion Python examples](code/reward_learning_checks.py):

```bash
python3 05-post-training/deep-rl/code/reward_learning_checks.py
python3 -m unittest discover -s site/tests -p test_reward_learning_checks.py
```

They check one parameter update, analytic against numerical gradients, dynamic programming against enumeration, and the stochastic coin counterexample. Change the demonstration counts to `[7, 2, 1]`, predict the gradient's sign, then run the code. This makes feature matching more concrete than memorizing its name.

If the goal is expert-like behavior rather than a separately recovered reward, continue to [GAIL](gail.en.md). If supervision consists of ranked responses instead of action trajectories, read [reward-model ranking losses](../reward-ranking.en.md). The two likelihoods share a form: [Zhu, Jiao and Jordan (2023), §6.1](https://arxiv.org/html/2301.11270v5#S6.SS1) write MaxEnt IRL with deterministic dynamics as the PL probability of choosing the demonstrated trajectory from all feasible ones, the same as the first-place step of a ranking. The data differ: in IRL the candidates are every feasible route and only the chosen one is observed; in ranking annotation they are the few responses that were shown. They are different sources of supervision.
