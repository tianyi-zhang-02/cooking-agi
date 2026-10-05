# Model-based RL: prediction is not planning

[中文](model-based.md) · **English**

> Reading time: ~10–12 min · Last reviewed: 2026-10

When trial and error is expensive, can we learn an environment model and test actions inside it first? Yes. But if the model mistakes a dangerous route for a shortcut, the planner may prefer that route. **Once predictions guide decisions, good average prediction accuracy is not enough.**

## Separate 3 objects

A dynamics model predicts what happens after an action. A planner searches action sequences. A policy maps states directly to actions. A model and planner need not include a separately trained policy. Planning with known physics is not automatically learned RL.

A simple model predicts state differences rather than the entire next state:

$$
\hat s_{t+1}=s_t+f_\psi(s_t,a_t),\qquad
L_{\rm model}=\mathbb E_{\mathcal D}\|f_\psi(s,a)-(s'-s)\|^2.
$$

This is a modeling convenience, not a theorem. Stochastic environments may require distributions; pixel observations may need latent state rather than uniformly weighted pixel MSE.

## MPC plans a sequence and executes one action

Model predictive control searches a length-$H$ action sequence from the current real state:

$$
a_{t:t+H-1}^*=
\arg\max_{a_{t:t+H-1}}
\sum_{k=0}^{H-1}\gamma^k\hat r(\hat s_{t+k},a_{t+k})
+\gamma^H\hat V(\hat s_{t+H}).
$$

This expression uses a deterministic model: $\hat s_t=s_t$, and later states follow from the model and candidate actions. A stochastic model also needs an expectation over predicted returns or an explicit risk criterion. If no reliable terminal-value estimate exists, omit the tail term, recognizing that rewards beyond the horizon are then ignored. Execute only the first action, observe the environment, and replan rather than carrying out the initial sequence unchanged.

```text
real state → sample action sequences → model rollouts → score sequences
           → retain elites and update sampling distribution → first action
           → observe the actual next state → plan again
```

CEM, the cross-entropy method, repeatedly samples sequences, selects elites, and refits their distribution. It is a search optimizer, not a guarantee of global optimality.

## Watch a model think it has arrived

Start a cart at $x=3$ with goal 0. The real system is $x_{t+1}=x_t+a_t$, but the model assumes $\hat x_{t+1}=\hat x_t+\hat b a_t$. Available actions are -1, 0, and 1. Planning minimizes:

$$
\sum_{k=0}^{H-1}\left(\hat x_{t+k+1}^{\,2}+0.1a_{t+k}^2\right).
$$

Adjust the model gain and planning horizon. This toy enumerates all $3^H$ sequences: **it is not CEM or a neural network**. Exact enumeration isolates model mismatch and feedback.

<div class="drl-lab" data-drl-lab="mpc"><p>Static example: H=5 and model gain=1.5. The initial plan predicts arrival in two steps but actually stops at 1; observing and replanning reaches 0 in this toy. Real cumulative costs are 8.2 and 5.3.</p></div>

| Try | Observe | What it does not establish |
| --- | --- | --- |
| Set model gain to 1 | Prediction and execution agree in this toy | Real world models need no calibration |
| Set it to 1.5 | The initial plan stops early; feedback prompts another move | MPC fixes every kind of model error |
| Set it to 0.5 | The initial plan overshoots; replanning stops in time | Every control problem permits safe recovery |
| Increase H | Candidate count grows as $3^H$ | Longer horizons always improve real outcomes |

Both methods execute $H$ real steps, but MPC searches another $H$ steps ahead at each decision. It spends more computation, so this is not an equal-compute benchmark. There are no obstacles, observation noise, or irreversible actions here.

## Is the world random, or does the model not know?

| Uncertainty | Meaning | Intuition | Response |
| --- | --- | --- | --- |
| Aleatoric | Outcomes remain variable given current information | Random gusts affect the same action | Predict distributions and plan for risk, not only the mean |
| Epistemic | Insufficient data or model knowledge | An unfamiliar surface has unknown braking behavior | Gather data, compare models, and limit extrapolation |

Irreducibility is relative to available information: a wind sensor may make some variation predictable. Ensemble disagreement only approximates model uncertainty; it is not a reliability certificate.

Hold the state and action fixed and consider one next-state component $X'$. The law of total variance separates variation within and between ensemble members. With member index $M$:

$$
\operatorname{Var}(X')=
\mathbb E_M[\operatorname{Var}(X'\mid M)]
+\operatorname{Var}_M(\mathbb E[X'\mid M]).
$$

Two equally weighted models with means 0 and 2 and variance 1 each yield mixture variance $1+1=2$. The first term is within-member variation; the second is between-member variation. This is an exact decomposition of the mixture. Interpreting the second term as true epistemic uncertainty requires modeling and calibration assumptions.

## Why rollout errors compound

Fix the same initial state and action sequence for deterministic true dynamics $f$ and model $\hat f$. Assume their one-step difference is uniformly bounded by $\epsilon$ over the region visited, and that $f$ is $L$-Lipschitz in state. With $e_k=\|s_k-\hat s_k\|$:

$$
e_0=0,\qquad e_{k+1}\le Le_k+\epsilon,\qquad
e_H\le\epsilon\sum_{j=0}^{H-1}L^j.
$$

For $L=1$, error is at most $H\epsilon$; for $L>1$, it can grow faster. If the initial states differ, add $L^H e_0$. This bounds state error under the same actions, not the return difference between two feedback policies. Average training MSE is not the uniform error bound required here. Planning may also reach unfamiliar regions where these assumptions fail.

If the model underestimates collision risk near obstacles, its optimal route may hug those obstacles. Average one-step MSE can hide exactly the errors that matter to decisions.

## Low one-step MSE can average two paths into a wall

Suppose the same input leads equally often to positions −1 and +1. The optimal squared-error point prediction is their mean, 0, although the system never actually reaches 0. If that represents an impassable location, planning with the average state can fail.

A distributional model can represent uncertainty beyond one mean. A Gaussian negative log-likelihood includes:

$$
\tfrac12\left[(s'-\mu)^2/\sigma^2+\log\sigma^2\right]+\text{constant}.
$$

Variance explains randomness, but the log-variance term penalizes unlimited inflation. One Gaussian may still miss separated modes; mixtures, latent variables, or other models may be appropriate. An ensemble does not automatically fix an inadequate distribution family shared by all members.

## Why can harder planning exploit more model errors?

Two routes have predicted returns 3 and 10 but actual returns 3 and −5. Choosing the second is not weak search: the planner optimized the incorrect prediction successfully. Additional search may simply find the same flaw more reliably.

Hold the model fixed and increase search budget, comparing predicted and real returns. If predictions improve while real outcomes worsen, inspect exploited regions before tuning search further. Then fix the planner, add real data from those regions, and check whether the gap closes.

Short rollouts, disagreement penalties, and constraints can reduce some risks but have costs. Short horizons miss distant gains; pessimism may reject genuinely good routes; an incorrect constraint model gives no safety guarantee. Compare with model-free baselines and an oracle model, not only the learned model’s own scores.

## Planning-budget trade-offs

| Choice | Benefit | Risk |
| --- | --- | --- |
| Longer horizon | Sees delayed benefits | Accumulated error and harder search |
| More candidate sequences | More thorough search | Higher decision latency |
| Ensembles / uncertainty penalties | Flag some unreliable regions | Models can agree and still be wrong; not a safety guarantee |
| Short model rollouts plus real data | Limits long-horizon drift | Synthetic-data ratio and bias still matter |

PETS combines probabilistic models and ensembles with sampling-based planning. MCTS instead allocates search effort in a discrete action tree. Both need predictive structure, but organize search differently. A complete tree-search implementation is outside this chapter.

## Report more than model loss

| Check | Hold fixed | Inspect |
| --- | --- | --- |
| One-step prediction | Held-out real transitions | Mean error and distribution calibration, not just aggregate MSE |
| Multistep prediction | Initial state and action sequence | Error growth with horizon |
| Decision-relevant error | Actions actually selected by the planner | Whether planning exploits inaccurate regions |
| Closed-loop control | Real environment and budgets | Return, safety failures, latency, and interaction count |

Prefer trajectory-level or temporal splits over merely shuffling adjacent transitions. Otherwise test data may remain immediately beside training trajectories and hide extrapolation failures.

Measure real-environment return, planning latency, errors at different horizons, and failures outside training coverage. Comparisons with model-free methods must state both interaction and computation budgets. Fewer samples do not automatically mean lower cost or latency.

Reference: [PETS](https://arxiv.org/abs/1805.12114). Go deeper with [planning and LQR](planning-and-control.en.md), then [experiments and debugging](experiments.en.md), or [RL from a fixed dataset](offline-and-ope.en.md).
