# Inside planning: CEM, LQR, and learning in a model

[中文](planning-and-control.md) · **English**

> Reading time: ~6–8 min · Last reviewed: 2026-10

Model-based RL is not one fixed algorithm. First ask: **does computation happen at each decision, or during training beforehand?** Then ask whether derivatives exist, actions are continuous, and constraints may be violated. This is more useful than choosing a planner by name.

## CEM searches without derivatives

Suppose a plan contains 5 scalar actions. Initialize a mean and standard deviation for each coordinate, sample 100 sequences, predict and score them with the model, retain the best 10, and refit the distribution. Repeat for a few rounds. These are illustrative numbers, not recommended defaults.

CEM parallelizes easily and does not require differentiability, but dimensionality and horizon enlarge the search space rapidly. Too few elites or premature variance collapse can miss another useful route. After MPC executes the first action, the remaining sequence can be shifted to warm-start the next search.

Reduce it to one step for a hand calculation. A cart starts at 2 and ends at $2+u$, with cost $(2+u)^2+0.1u^2$. Suppose this round sampled the following 4 actions; retain the two lowest-cost candidates:

| Candidate u | Predicted endpoint | Cost, lower is better | Retain? |
| --- | --- | --- | --- |
| -2 | 0 | 0.4 | Yes |
| -1 | 1 | 1.1 | Yes |
| 0 | 2 | 4 | No |
| 1 | 3 | 9.1 | No |

Refitting a Gaussian to -2 and -1 gives mean -1.5 and standard deviation 0.5, using maximum likelihood with denominator 2. Sample around that region next; **do not declare the mean the optimal action**. A variance floor or smoothed parameter updates can prevent premature collapse.

<details markdown="1">
<summary>What is the actual optimum in this example?</summary>

Differentiating gives $2(2+u)+0.2u=0$, hence $u=-2/1.1\approx-1.8182$. Neither the best sampled candidate -2 nor the elite mean -1.5 equals it. CEM moves future sampling toward promising regions; elite selection does not provide an analytic optimum.

</details>

## LQR is a solvable special case

Assume known linear dynamics $x_{t+1}=Ax_t+Bu_t$ and quadratic cost, with positive-semidefinite $Q,Q_f$, positive-definite $R$, and no additional action constraints:

$$
J=\sum_{t=0}^{T-1}(x_t^\top Qx_t+u_t^\top Ru_t)+x_T^\top Q_fx_T.
$$

If the next cost-to-go is $V_{t+1}(x)=x^\top P_{t+1}x$, substitute into Bellman and differentiate with respect to $u$:

$$
u_t^*=-K_tx_t,\qquad
K_t=(R+B^\top P_{t+1}B)^{-1}B^\top P_{t+1}A.
$$

Substitution gives the Riccati recursion:

$$
P_t=Q+A^\top P_{t+1}A-A^\top P_{t+1}BK_t,\qquad P_T=Q_f.
$$

Implement this with linear solves, not explicit matrix inversion. It is multidimensional completing-the-square: linear transitions and quadratic cost preserve the quadratic form through Bellman updates.

For the final single step, take $A=B=Q=R=Q_f=1$ and $x=2$. Minimize $4+u^2+(2+u)^2$. Its derivative is $4u+4$, giving $u=-1$ and cost 6. Under a hard bound $|u|\le0.2$, that unconstrained solution is infeasible; it is no longer valid to call $-1$ the optimal admissible control.

## One more step reveals the feedback gains

Keep $A=B=Q=R=Q_f=1$ but allow two actions. Start from terminal $P_2=1$ and work backward:

| Time | Next $P$ | Current gain $K$ | Current $P$ |
| --- | --- | --- | --- |
| $t=1$ | 1 | $1/(1+1)=0.5$ | $1+1-1\times0.5=1.5$ |
| $t=0$ | 1.5 | $1.5/(1+1.5)=0.6$ | $1+1.5-1.5\times0.6=1.6$ |

At $x_0=2$, action $u_0=-1.2$ gives $x_1=0.8$; then $u_1=-0.4$ gives $x_2=0.4$. Total cost is $4+1.44+0.64+0.16+0.16=6.4$, equal to $P_0x_0^2=1.6\times4$.

This is not worse control than the one-stage cost of 6: the objectives include different numbers of stage costs. Interpret values with their horizon. Feedback also recomputes actions from observed states; after a disturbance, the original numeric action sequence is no longer equivalent to feedback control.

## Constraints are not an afterthought clip

Clipping an unconstrained LQR action makes its value legal but does not generally solve the constrained trajectory problem optimally. An unavailable large action today changes tomorrow’s state and remaining choices. Include the constraint in planning.

Likewise, heavily penalizing a collision in CEM differs from requiring every candidate to satisfy a safety constraint. A finite penalty may still be traded for enough reward. If no feasible candidate is found, specify a failure response rather than labeling the least-bad violating sequence a success.

## iLQR, shooting, and collocation

iLQR locally linearizes dynamics and quadratizes cost around the current trajectory, solves backward for a local control correction, then rolls forward with a line search. It reuses LQR's structure but **solves local approximations**, not a guaranteed global optimum.

Shooting optimizes actions and obtains states by rolling the model forward; long derivative chains can vanish or explode. Direct collocation also optimizes states and imposes dynamics constraints explicitly. It introduces more variables but can exploit sparsity. This section explains the mechanisms, not a complete constrained solver.

| Method | Optimized variables | Requirements | Main limitation |
| --- | --- | --- | --- |
| Random shooting / CEM | Action sequences | Repeated model calls and scoring | Search difficulty grows with dimension and horizon |
| Gradient shooting | Action sequences | Differentiable model and cost | Long gradient chains and local optima |
| iLQR | Local action corrections around a trajectory | Local derivatives and quadratic approximations | Poor initial trajectories can invalidate local approximations |
| Direct collocation | States and actions, with transition constraints | A constrained optimizer | More variables, feasibility, and solver cost |

MPC is the execution scheme of replanning after new observations, not a fifth optimizer in this table. It can use CEM or another solver. Planner choice and feedback frequency are separate decisions.

## Models can train policies rather than search online

| Use | What the model does | Deployment cost |
| --- | --- | --- |
| MPC / CEM | Searches candidate sequences at each decision | Search latency |
| Dyna / MBPO-style | Generates transitions to train policy / Q | Mostly policy inference |
| Latent imagination | Rolls out latent states to train Actor / Critic | Encoding and policy inference, depending on architecture |
| Planner distillation | Supervises a fast policy with plans | Student inference, potentially losing planning capability |

One MBPO principle is to start short rollouts near real data rather than trust a model indefinitely. If latent state discards control-relevant information, attractive reconstructions do not fix it. Training inside a model is not verification of deployment safety.

## Choose a practical starting point

With low-dimensional continuous state and a known linear approximation, begin with an interpretable control baseline. A black-box model with cheap evaluations can support sampling-based planning. Tight decision latency may favor moving computation into training. Always report real interactions, model-training effort, and per-action search budget together.

References: [MIT's LQR derivation](https://underactuated.mit.edu/lqr.html) · [MBPO](https://arxiv.org/abs/1906.08253). Return to the [model-based overview](model-based.en.md).
