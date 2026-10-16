# How planning and learning feed each other: distillation, terminal values, and search as policy improvement

[中文](planning-and-learning.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

With a model in hand, computation can be spent at two moments: before deployment, turning rollouts in the model into training data; or at deployment, searching afresh at every step. The two are not exclusive. Systems like AlphaZero do both and let each improve the other. **Planning can produce better labels for learning, and learning can make planning cheaper; when you connect them, be clear about which side is correcting which side's errors.**

## Spend computation before deployment, or at every step?

Background planning uses the model to generate experience and then applies ordinary RL updates to learn a policy or Q; at deployment it only calls the learned policy. Decision-time planning searches from the current real state before every action. The first is cheap to deploy; the second can compute for the exact state in front of it, even one never seen in training.

Dyna, introduced by [Sutton (1990)](http://incompleteideas.net/papers/sutton-90.pdf), is the prototype of background planning: one learning process alternates between the real environment and a learned model, and Dyna-Q adds this step on top of Q-learning. [Sutton (1991)](http://incompleteideas.net/papers/sutton-91a.pdf) describes the structure as compiling planning into a reactive policy: the policy produces actions with almost no delay while planning keeps improving it in the background.

Start with the smallest example. A corridor $A\to B\to G$ has one action, "forward"; entering $G$ earns 1 point and ends the episode, with $\gamma=0.9$. The Q table starts at 0, the environment is deterministic, and for hand computation the learning rate is 1. The first real episode updates in order: first $Q(A)=0+0.9\,Q(B)=0$, because $Q(B)$ is still 0; then $Q(B)=1$. With real interaction alone, it takes another episode before $Q(A)$ becomes 0.9.

Dyna stores both transitions in its model. After the episode, it pulls $(A\to B)$ from the model and updates once more: $Q(A)=0+0.9\times1=0.9$. **Same real data, one extra computation, and the value reaches the start.** Here the "model" is just the stored transitions, no different from replay (see [which model to learn](model-types.en.md)). A learned parametric model can also generate data from unseen states, and that is where model error enters; [MBPO (Janner et al., 2019)](https://arxiv.org/abs/1906.08253) therefore runs only short rollouts branched from real data.

The per-step deployment cost of each use is in the [comparison table](#combinations) at the end. First, a different question: how the two kinds of planning and network training improve each other.

## MCTS: give simulations to branches that look promising but are under-explored

Each simulation of Monte Carlo tree search (MCTS) walks from the root to a leaf in 4 steps:

1. **Selection**: at each node, pick an edge by the rule below, until reaching an unexpanded position;
2. **Expansion**: use the model to compute the new node's state; AlphaZero uses the game rules, MuZero its learned dynamics function;
3. **Evaluation**: let the network assign the new node a value $v$, instead of playing randomly to the end;
4. **Backup**: send $v$ (and any intermediate rewards) back along the path; each edge's $Q$ is the average over the simulations that passed through it, and its visit count $N$ goes up by 1.

Selection uses a PUCT-style rule:

$$
a=\arg\max_a\left[Q(s,a)+c\,P(s,a)\,\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\right].
$$

$Q$ is the average backed-up value, $P$ the network's prior probability, $N$ the visit count, and $c$ sets how much to explore. Edges with a high prior and few visits get a bonus; the more an edge is visited, the smaller its bonus; as the parent's total visits grow, every bonus grows slowly with the square root. [AlphaZero (Silver et al., 2017)](https://arxiv.org/abs/1712.01815) summarizes the rule as favoring moves with low visit count, high prior probability, and high value. Equation (2) in the [MuZero](https://arxiv.org/abs/1911.08265) appendix also multiplies the bonus by a coefficient $c_1+\log\frac{\sum_bN+c_2+1}{c_2}$ that grows slowly with total visits (the paper uses $c_1=1.25$ and $c_2=19652$); with visit counts far below $c_2$, the coefficient is about $c_1$, which gives the rule above.

## Hand-compute a two-action tree

The root has two actions, L and R. The network prior is $P(L)=0.6$, $P(R)=0.4$; the network values the two children at $v(L)=0.2$ and $v(R)=0.6$. Take $c=1$, $Q=0$ for unvisited edges, and break ties toward the larger prior. For hand computation, assume the tree is one level deep and every selection backs up that fixed value; real MCTS would expand further on a second visit, and $Q$ would average deeper values.

| Simulation | $\sum N$ | Score of L | Score of R | Selected | Then $(N_L,N_R)$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | L (tie, larger prior) | (1, 0) |
| 2 | 1 | $0.2+0.6\times1/2=0.50$ | $0+0.4\times1/1=0.40$ | L | (2, 0) |
| 3 | 2 | $0.2+0.6\times1.414/3=0.48$ | $0+0.4\times1.414/1=0.57$ | R | (2, 1) |
| 4 | 3 | $0.2+0.6\times1.732/3=0.55$ | $0.6+0.4\times1.732/2=0.95$ | R | (2, 2) |
| 5 | 4 | $0.2+0.6\times2/3=0.60$ | $0.6+0.4\times2/3=0.87$ | R | (2, 3) |
| 6 | 5 | $0.2+0.6\times2.236/3=0.65$ | $0.6+0.4\times2.236/4=0.82$ | R | (2, 4) |

The prior favors L, but once R is seen to be worth more, later simulations concentrate on R. L's bonus grows slowly with $\sqrt{\sum N}$, so further simulations will eventually revisit it; a larger $c$ revisits sooner.

After the search, turn the root visit counts into a distribution:

$$
\pi(a)=\frac{N(a)^{1/\tau}}{\sum_bN(b)^{1/\tau}}.
$$

With $\tau=1$, $\pi=(2/6,\,4/6)\approx(0.33,\,0.67)$; with $\tau=0.5$, square the counts, $(4,16)/20=(0.2,\,0.8)$; as $\tau\to0$, all mass goes to R. The network's prior was $(0.6,\,0.4)$; search turned it into $(0.33,\,0.67)$. **Use this $\pi$ as the policy head's training target, and the next search starts from a better prior.** In Atari, MuZero samples actions from this distribution and lowers $\tau$ from 1 to 0.5 to 0.25 over training; AlphaZero picks self-play moves in proportion to visit counts and picks greedily by visit count during evaluation.

<!-- widget:step-mcts-visits -->

The example also exposes an assumption: the search fully trusts $v(R)=0.6$. If the network overestimated R, the search would concentrate there just as diligently. **Search improves the policy relative to the network's own values; errors in those values must be corrected by real outcomes.**

## AlphaZero: search provides better labels, and the network makes search cheaper

AlphaZero's network $(p,v)=f_\theta(s)$ outputs both move probabilities and a position value. MCTS uses $p$ as the prior and $v$ to evaluate leaves; self-play moves follow the search output $\pi$; each game ends with an outcome $z\in\{-1,0,+1\}$. The loss is:

$$
l=(z-v)^2-\pi^\top\log p+c\|\theta\|^2.
$$

The terms do different jobs. The second pulls $p$ toward the search output $\pi$: treat search as a "policy improvement operator" that takes $p$ and returns a better $\pi$; training moves $p$ toward $\pi$, which leaves search less to improve. The first pulls $v$ toward the real game outcome $z$, not toward search's own estimate; that term is what corrects the value errors from the previous section. The last term is weight regularization.

In return, the network makes search cheaper: the prior $p$ concentrates simulations on a few moves, and $v$ replaces random rollouts played to the end. [Expert Iteration (Anthony et al., 2017)](https://arxiv.org/abs/1705.08439) states the same division of labor plainly in its abstract: "Planning new policies is performed by tree search, while a … network generalises those plans"; the network then guides the search.

Note that AlphaZero's "model" is the exact rules of the game, not something learned. Where the rules are known and simulation is perfect, this loop is established practice; the problems in the next two sections become prominent when the model itself must be learned.

## Terminal values let a short-sighted MPC see beyond its horizon

The MPC objective in the [Model-based note](model-based.en.md) ends with a term $\gamma^H\hat V(\hat s_{t+H})$. That term is what learning feeds into planning: the planner looks $H$ steps ahead and hands the rest of the future to a learned value.

A corridor has positions 0, 1, and 2. Start at 0; reaching 2 earns 10 points and ends the episode; each move costs 1 point, staying put costs nothing, $\gamma=0.9$, and the model is exact. The true values are $V^*(1)=-1+10=9$ and $V^*(0)=-1+0.9\times9=7.1$. Compare the first action under several planning settings:

| Planning setting | Best score starting with "stay" | Best score starting with "right" | Choice |
| --- | --- | --- | --- |
| $H=1$, no terminal value | 0 | $-1$ | Stay, and stay at every later step too |
| $H=1$, $\hat V=V^*$ | $0.9\times7.1=6.39$ | $-1+0.9\times9=7.1$ | Right |
| $H=1$, $\hat V(1)$ underestimated as 7 | 6.39 | $-1+0.9\times7=5.3$ | Stay |
| $H=2$, $\hat V(1)$ still 7 | $0.81\times7.1\approx5.75$ | $-1+0.9\times9=7.1$, reaching the goal in two steps | Right |

Without a terminal value, the $H=1$ planner only sees that moving costs 1 point and never sees the 10. With an exact $\hat V$, one-step planning makes the far-sighted choice. But an error in $\hat V$ at the boundary changes the decision directly: underestimate by 2 points and the agent freezes again. With $H=2$, the real reward enters the search, so the error in $\hat V(1)$ no longer matters, at the cost of more search and more steps over which model error accumulates.

[POLO (Lowrey et al., 2018)](https://arxiv.org/abs/1811.01848) studies exactly this pairing: approximate value functions can shorten the planning horizon, and local trajectory optimization can tolerate errors in the value function and speed up its learning. MuZero trains its value on $n$ real rewards followed by the value from search, the same idea applied to training targets.

## What is lost when a planner is distilled into a fast policy

Another combination runs the planner in many states and trains a policy network to imitate its outputs. Deployment then needs one forward pass. [Guided Policy Search (Levine & Koltun, 2013)](https://proceedings.mlr.press/v28/levine13.html) uses trajectory optimization to guide policy search and avoid poor local optima; [Levine et al. (2016)](https://arxiv.org/abs/1504.00702) turn policy search into supervised learning, with a trajectory optimizer that sees the full state as teacher and a student that sees only raw images.

Deployment-time search is saved, and 3 things are lost:

- **Replanning on the spot.** If an obstacle suddenly appears ahead, or the reward changes, the planner can recompute; the student only repeats what it learned.
- **Recovery after drifting.** If the student makes a small error at each step, it reaches states the teacher never labeled, and errors compound. This is the problem [DAgger addresses in imitation learning](imitation-and-rewards.en.md). Levine et al. alternate training so that the teacher's supervision gradually lands on states the student itself visits, and at convergence the two behave the same.
- **Consistency.** To get around a post, going left (action $-1$) and going right ($+1$) are equally good, and the planner picks either depending on the random seed. A student trained with squared error learns the average, 0, and drives straight into the post. Either keep the teacher consistent or give the student an output distribution that can represent two peaks.

## Implicit planning: a model learned only so that planning gets values right

In the combinations above, the model and the planner are trained and checked separately. Another route puts the planning computation inside the network and trains end to end on value or return targets alone; what the model becomes is decided by what planning finds useful.

- [Value Prediction Network (Oh et al., 2017)](https://arxiv.org/abs/1707.03497) learns dynamics over abstract states and trains them to predict future values (discounted sums of rewards) rather than future observations; the paper reports beating DQN on several Atari games even with very short lookahead.
- [TreeQN (Farquhar et al., 2017)](https://arxiv.org/abs/1710.11417) recursively applies a learned transition function inside the network to build a tree of fixed depth, then combines predicted rewards and values into Q-values with a tree backup; end-to-end training optimizes the transition function for how the tree uses it.
- MuZero predicts only rewards, values, and policies at each hypothetical step, then runs MCTS over those hidden states.

<!-- paper-figure:muzero-model -->

[Which model to learn](model-types.en.md) used box pushing to show the cost: such a model is tied to the current task and must be retrained when the reward changes; its hidden states do not correspond to inspectable physical states, so the steps that planning "imagines" cannot be checked against reality one by one, only the predicted rewards and values.

As of October 2026, whether it is better to learn as complete a world model as possible and then plan, or to learn only an implicit model of what matters for value, is still changing quickly, and the answer differs by domain. The site's [world models series](../../11-frontier/world-models/README.en.md) tracks that question; the MCTS, PUCT, and AlphaZero training targets in this note are settled material.

## Models are useful beyond planning

**Exploration.** Prediction error can serve as intrinsic reward: [Pathak et al. (2017)](https://arxiv.org/abs/1705.05363) define curiosity as the error in predicting the consequences of the agent's own actions, and they predict in a feature space learned by an inverse dynamics model, so that changes the agent cannot affect are ignored. RND and the "noisy TV" problem are in [exploration and hierarchy](exploration-and-hierarchy.en.md). One step further is planning to explore: [Plan2Explore (Sekar et al., 2020)](https://arxiv.org/abs/2005.05960) plans inside a learned world model to seek expected future novelty, measured as disagreement among an ensemble of latent transition models; exploration needs no task reward, and the agent then adapts to downstream tasks zero-shot or few-shot.

**Adapting to a new reward, or to new dynamics.** With a state model, a new reward is just a new scoring function: once box pushing charges 0.1 points per push, the planner can immediately recompare $(1,3)$ and $(1,4)$, while a model-free policy or Q must be retrained. When the dynamics change, the model must adapt first. [Nagabandi et al. (2018)](https://arxiv.org/abs/1803.11347) meta-learn a prior for the dynamics model, adapt it online with recent data at deployment, and plan with MPC; the paper reports a real legged millirobot adapting online after losing a leg.

**As an auxiliary loss.** Predicting rewards or future observations can serve only to learn a better representation, with the predictions never used for planning. [UNREAL (Jaderberg et al., 2016)](https://arxiv.org/abs/1611.05397) has the agent optimize several pseudo-reward tasks at once, including reward prediction, all sharing one representation.

<span id="the-combinations-side-by-side"></span>

## The combinations side by side {#combinations}

| Combination | What is learned | What is searched | Per-step deployment cost | Typical failure |
| --- | --- | --- | --- | --- |
| Dyna / MBPO | Dynamics + policy or Q | No online search; the model generates training data | One policy forward pass | Long rollouts bias the generated data |
| MPC + learned model (e.g. PETS) | Dynamics, possibly an ensemble | Action sequences at every step | Horizon × candidates model calls | Planner exploits model errors; high latency |
| MPC + terminal value (e.g. POLO) | Dynamics or known model + $V$ | Short-horizon search at every step | As above, but with a shorter horizon | Errors in $V$ at the boundary change the choice directly |
| AlphaZero | Policy and value network; rules known | MCTS | Several simulations per step, one network evaluation each | Wrong values concentrate search on the wrong branch; needs an exact simulator |
| MuZero | Value-equivalent model + policy and value | MCTS over hidden states | One dynamics and one prediction call per simulation | Retraining when the task changes; hidden states hard to inspect |
| Planner distillation (e.g. GPS) | Student policy | The planner searches during training | One student forward pass | Compounding errors, averaged inconsistent teachers, no replanning on the spot |
| Implicit planning in the network (VPN, TreeQN) | Abstract transitions + values | Fixed-depth unrolling inside the network | A fixed number of forward passes | Works only for the training task; limited unrolling depth |
| Model as auxiliary loss | Shared representation | Nothing | One policy forward pass | No help when the auxiliary task is unrelated to the main task |

The table compares structure, not a performance ranking; for PETS see [Chua et al. (2018)](https://arxiv.org/abs/1805.12114).

## Where this note stops

This note covers the combinations and what each costs; it does not implement full MCTS. Details such as multi-level expansion, value sign flips in two-player games, and Dirichlet noise added to root priors follow the AlphaZero and MuZero papers. For what the model itself should learn, return to [which model to learn](model-types.en.md); for concrete trajectory optimizers, see [planning and LQR](planning-and-control.en.md); for why planning amplifies model errors, see the [Model-based overview](model-based.en.md).
