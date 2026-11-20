# Offline model-based RL: which predictions should train the policy?

[中文](offline-models.md) · **English**

> Reading time: ~15–18 min · Last reviewed: 2026-11

Suppose you have robot-arm logs but cannot collect more interactions. One option is to fit a dynamics model that predicts the next state and reward, then train a policy on predicted trajectories. That gives you more training data without operating the robot.

But simulating another hundred thousand steps does not give the robot another hundred thousand real attempts. The policy may repeatedly exploit the same modeling error: a shortcut looks successful in the model but causes a collision on the robot.

Offline model-based RL therefore needs more than accurate predictions: it also needs to stop the policy from exploiting model errors. MOPO subtracts a risk penalty from predicted rewards, MOReL stops trusting predictions in unreliable regions, and COMBO restricts overestimation during value learning. We will calculate how each changes an update. The examples are original toy problems, not reported experimental results.

## Follow the data through one loop {#data-flow}

<div class="drl-flow" aria-label="Offline model-based reinforcement learning data flow">
<span>Real logs<br><small>Fit dynamics and rewards</small></span><b>→</b><span>Short predictions<br><small>Start from logged states</small></span><b>→</b><span>Conservative values<br><small>Mix real and predicted data</small></span><b>→</b><span>Policy update<br><small>Generate fresh predictions</small></span>
</div>

The real dataset $\mathcal D$ stays fixed. The model buffer $\mathcal D_{\rm model}$ changes with the rollout policy, starting states, and generation time. Both can train a critic, but the latter does not supply new real observations.

Starting from real states and predicting only a few steps is an important design in [MBPO (2019)](https://arxiv.org/abs/1906.08253): model rollouts need not reach the end of the task. MBPO originally continues collecting environmental data, though. Borrowing its short rollouts alone does not solve offline distribution shift.

| Data or component | What it provides | What it does not establish |
| --- | --- | --- |
| Real logs $\mathcal D$ | Observed transitions for fitting and checking the model | An unobserved action must be poor |
| Dynamics and reward models | Predicted next states and rewards | Low average prediction error ensures a useful policy |
| Model buffer $\mathcal D_{\rm model}$ | More predicted transitions for value updates | Simulation count equals independent environmental sample count |
| Conservative treatment | A restriction on unsupported high estimates | Larger penalties are always better, or safety is proven |

On a first read, follow the calculation tables. For review, return to the equations and ask what changed and which assumptions it needs.

## MOPO: charge for model risk at each step {#mopo-penalty}

[MOPO (2020), §4](https://arxiv.org/abs/2005.13239) penalizes rewards in the learned model:

$$
\widetilde r(s,a)=\widehat r(s,a)-\lambda u(s,a).
$$

Here $u$ is a model uncertainty or error proxy, and $\lambda\geq0$ sets its strength. The performance bound requires control of the relevant model error; naming a practical uncertainty estimate $u$ does not make it satisfy that condition. The [previous note](offline-and-ope.en.md#offline-model-pessimism) gives a valid error-bound example and a failure case. Here we examine accumulation across steps.

Consider just two rewards, with $\gamma=0.9$, $\lambda=3$, and no terminal-value contribution:

| Step | Predicted reward $\widehat r_t$ | Uncertainty $u_t$ | Penalty $3u_t$ | Adjusted reward |
| --- | --- | --- | --- | --- |
| 0 | 2 | 0.1 | 0.3 | 1.7 |
| 1 | 4 | 0.4 | 1.2 | 2.8 |

The original return is $2+0.9\times4=5.6$. The adjusted return is:

$$
\widetilde G=1.7+0.9\times2.8=4.22.
$$

Equivalently, subtract the discounted total penalty:

$$
\begin{aligned}
\widetilde G
&=\sum_{t=0}^{H-1}\gamma^t\widehat r_t
-\lambda\sum_{t=0}^{H-1}\gamma^t u_t\\
&=5.6-3(0.1+0.9\times0.4)\\
&=4.22.
\end{aligned}
$$

Subtracting the mean uncertainty once at the end is not interchangeable. It changes temporal weighting and can treat long and short trajectories quite differently. Changing the units of $u$, reward normalization, or the discount also changes how $\lambda$ should be interpreted.

## MOReL: stop trusting predictions in unknown regions {#morel-halt}

[MOReL (2020), Definitions 1 and 2](https://proceedings.neurips.cc/paper_files/paper/2020/file/f7efa4f864ae9b88d43527f4b14f750f-Paper.pdf) takes a different approach: a detected unknown state–action pair transitions to an additional absorbing state, **HALT**. This is a conservative construction in the training problem, not a physical collision state.

To make reward timing explicit, we follow the **equations in Definition 2**: ordinary states still receive $r(s,a)$; once in HALT, every step receives $-\kappa$ and stays there. Assume $0\leq\gamma<1$.

$$
V(\mathrm{HALT})=-\kappa+\gamma V(\mathrm{HALT})
=-\frac{\kappa}{1-\gamma}.
$$

With $\kappa=2$ and $\gamma=0.9$, HALT is worth $-20$, not $-2$. An action flagged as unknown in an ordinary state, earning 1 immediately before transitioning to HALT, has:

$$
Q(s,a)=1+0.9\times(-20)=-17.
$$

| Transition representation | TD target | Preserves the stated definition? |
| --- | --- | --- |
| Reward 1; next state HALT; keep bootstrapping | $1+0.9(-20)=-17$ | Yes |
| Fold the entire absorbing-state tail into the immediate reward, then terminate | $-17$ | Yes, an equivalent representation of this absorbing tail |
| Give $-2$ once and terminate immediately | $-2$ | No: it drops the future cost and changes the immediate reward |

The prose immediately after the paper's equations summarizes this as penalizing unknown actions. An implementation still needs precise timing: when does the penalty start, and is it repeated or paid once? Our $-17$ follows the explicit convention above and should not be mixed with another one.

The detector matters too. The theoretical “known” label requires a guarantee on model error. Agreement among practical predictors is only evidence of a kind; if they agree on a wrong prediction, the HALT mechanism may never trigger.

## COMBO: restrict overestimation inside critic fitting {#combo-critic}

[COMBO (2021), Equations (2) and (4)](https://proceedings.neurips.cc/paper/2021/file/f29a179746902e331572c483c45e5086-Paper.pdf) avoids an explicit uncertainty score and adds a distribution contrast to Q fitting. Holding one Bellman target $y$ fixed, write:

$$
\begin{aligned}
L(Q)&=\beta\left(\mathbb E_{\rho}Q-\mathbb E_d Q\right)\\
&\quad+\frac12\mathbb E_{d_f}(Q-y)^2.
\end{aligned}
$$

- $d$: the state–action distribution in real data.
- $\rho$: the distribution used for the conservative term; the theoretical construction uses policy visitation in the learned model.
- $d_f$: the Bellman-regression sampling distribution, mixing real data with model rollouts. It need not equal $\rho$.

The first term pushes some values down and others up. Its direction depends on the distribution difference, not simply on whether a sample came from a model.

### Fix two targets and solve one update {#combo-example}

Consider two state–action entries, A and B. Set targets $y=(2,5)$, real-data frequencies $d=(0.9,0.1)$, and conservative-term frequencies $\rho=(0.5,0.5)$. For this example only, also let model-regression frequencies equal $\rho$, mixing real and model data equally:

$$
d_f=0.5d+0.5\rho=(0.7,0.3).
$$

The two Q-values are independent scalar parameters, not outputs sharing a neural network. Differentiating each coordinate gives:

$$
\frac{\partial L}{\partial Q_i}
=\beta(\rho_i-d_i)+d_{f,i}(Q_i-y_i).
$$

Set the derivative to zero, with $d_{f,i}>0$:

$$
Q_i^*=y_i-\beta\frac{\rho_i-d_i}{d_{f,i}}.
$$

| Entry | Target $y_i$ | $\rho_i-d_i$ | $d_{f,i}$ | $Q_i^*$ with $\beta=1$ | $Q_i^*$ with $\beta=2$ |
| --- | --- | --- | --- | --- | --- |
| A | 2 | −0.4 | 0.7 | $2+4/7\approx2.5714$ | $2+8/7\approx3.1429$ |
| B | 5 | 0.4 | 0.3 | $5-4/3\approx3.6667$ | $5-8/3\approx2.3333$ |

B is rare in real data but common in the conservative term, so its value is pushed down. A moves the other way. Raising $\beta$ from 1 to 2 even reverses their ranking. The penalty changes decisions; it is not an irrelevant addition to the loss.

This is **one exact minimization with fixed targets and tabular parameters**. Neural networks share parameters, and later targets change. The table is not a convergence proof for training. If $d=\rho$, the linear conservative term vanishes. If some $d_{f,i}=0$ while its linear coefficient is nonzero, that unconstrained scalar objective can be unbounded below; the division is invalid.

The paper's Proposition 4.1 gives a lower bound in expectation under the initial distribution and policy, with error-control assumptions and sufficiently large $\beta$. It **does not say every state–action value lies below its true value**. Avoiding explicit $u$ estimation does not remove model bias or coverage concerns.

## What changes in each method? {#comparison}

| Method | Where conservatism enters | First question to ask |
| --- | --- | --- |
| MOPO | A soft penalty on model rewards | What does $u$ measure, and is the penalty discounted step by step? |
| MOReL | Unknown-region transitions and the HALT cost | What triggers detection, and when do entry and continuing costs apply? |
| COMBO | A distribution contrast in critic fitting | Where do the three sampling distributions come from, and which values does $\beta$ change? |

All three address unsupported high estimates, but with different assumptions, data, and implementations. Their publication dates do not form an upgrade sequence.

## A short rollout is not necessarily a terminal episode {#rollout-cutoff}

Sample 32 starting states from the logs and predict 5 steps each: that produces at most 160 model transitions. They share a learned model, and starting states may repeat. You cannot treat them as 160 independent environmental observations when constructing confidence intervals.

A second issue is easy to miss: did generation stop at step 5 because the rollout budget ended, or because the task ended?

Suppose the last retained transition earns 1, its next state has estimated value 4, and $\gamma=0.9$. If the task continues but model generation reached its length limit:

$$
y=1+0.9\times4=4.6.
$$

Marking it as genuinely terminal changes the target to 1. The estimate 4 may itself be biased; retaining the bootstrap only preserves the continuing-task objective, not target correctness. Genuine termination uses terminal value zero. Keep artificial cutoffs, environmental termination, and an equivalent HALT-tail collapse distinct. See [MC, TD, and boundaries](returns-and-td.en.md).

## Before trusting the training curve {#implementation-checks}

| Check | A concrete way to inspect it |
| --- | --- |
| Real and model data become indistinguishable | Retain sample provenance; report errors on each source separately, not only a mixed loss |
| Old predictions survive model or rollout-policy changes | Record model version, policy version, and generation batch; define buffer refresh, retention, and eviction |
| Training and validation objectives get confused | Track original rewards, penalized rewards, and independent task measures separately; a penalized score is not actual environmental return |
| Train/validation splits leak | Split by trajectory or task rather than adjacent transitions; separately examine regions the new policy visits but the data rarely covers |
| Hyperparameter selection quietly uses new feedback | Distinguish offline training from an entirely offline selection process; report any real rollouts used for tuning |
| OPE shares data with policy training | Keep an evaluation split and inspect coverage and uncertainty; agreement among predictions is not independent validation |

These checks do not depend on a framework. Before scaling up, take one transition and explain its source, reward, next state, boundary flag, and target. That often exposes problems a larger model would only make harder to locate.

## Review and try it {#review}

1. Change $\gamma$ to 1 in the MOPO example. What is the two-step adjusted return, and why is it no longer 4.22?
2. HALT costs 2 per step. If $\gamma$ changes from 0.9 to 0.99, what replaces the value −20? Must a collapsed one-time cost change too?
3. Replace $d$ with $\rho$ in the COMBO example. Which term disappears? Does that establish model correctness?
4. After a one-step model rollout, should the last transition bootstrap? State whether the environment terminates before answering.

The first two answers are $1.7+2.8=4.5$ and $-2/(1-0.99)=-200$. The third removes the linear distribution-contrast term, not model error. The fourth depends on the task boundary, not rollout length alone.

Run the [companion calculations](code/offline_model_checks.py) to reproduce the penalty, absorbing-state value, and COMBO table. They check the toy mathematics, not a trained policy. Continue with [OPE estimators](ope-estimators.en.md) to consider how to evaluate the selected policy without new interactions.
