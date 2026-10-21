# Autonomous driving: motion forecasting and planning

[中文](prediction-planning.md) · **English**

> Reading time: ~11 min · Last reviewed: 2026-10

The car in front of you at the intersection might go straight, turn left or turn right; it has not started turning yet, and you already have to commit to your own next 4 seconds. **Forecasts must give several possibilities with probabilities, not one averaged trajectory; the planner picks one trajectory from a set of candidates by cost, where the weights decide "what counts as good" and the hard constraints decide "what is never allowed".**

## What a forecast must deliver

The inputs are each road user's past trajectory (from [tracking](../perception/tracking.en.md)), the map (lanes, traffic lights) and the ego state; the output, for each road user, is $K$ candidate trajectories over the next few seconds, each with a probability. Why several? The same history can belong to very different intentions: a car slowing toward an intersection may be about to turn left, or just waiting at a red light. Training a single trajectory with squared error averages left and right into a path nobody drives, the same problem as "averaging two paths into a wall" in [Model-based RL](../../05-post-training/deep-rl/model-based.en.md).

## Work it once: minADE and minFDE over 3 forecasts

These are teaching numbers. The lead car at the intersection actually turns left, with positions $(4,0)$, $(8,2)$, $(10,6)$ at $t=1,2,3$ s (in metres). The model gives $K=3$ forecasts:

| Forecast | $t=1$ | $t=2$ | $t=3$ | Error per step (m) | ADE | FDE |
| --- | --- | --- | --- | --- | --- | --- |
| $k_1$ straight | (4, 0) | (8, 0) | (10, 0) | 0, 2, 6 | 2.67 | 6 |
| $k_2$ left | (4, 0) | (8, 3) | (10, 5) | 0, 1, 1 | 0.67 | 1 |
| $k_3$ right | (4, 0) | (8, −2) | (10, −6) | 0, 4, 12 | 5.33 | 12 |

$\hat p^{(k)}_t$ is forecast $k$'s position at time $t$, $p_t$ the true position, and $T$ the number of forecast steps:

$$
\text{ADE}_k=\frac1T\sum_{t=1}^{T}\big\|\hat p^{(k)}_t-p_t\big\|,\qquad
\text{FDE}_k=\big\|\hat p^{(k)}_T-p_T\big\|,\qquad
\text{minADE}_K=\min_k\text{ADE}_k .
$$

minFDE takes the minimum the same way. Here minADE$_3=0.67$ m and minFDE$_3=1$ m, both from $k_2$.

What if the model may hand over only one? Say it assigns probabilities $(0.5,0.3,0.2)$ to the three. Handing over only the most likely, straight: ADE 2.67 m, FDE 6 m. Handing over the probability-weighted mean: at $t=2$, $y=0.3\times3+0.2\times(-2)=0.5$, error 1.5; at $t=3$, $y=0.3\times5+0.2\times(-6)=0.3$, error 5.7; ADE $=(0+1.5+5.7)/3=2.4$ m. This mean trajectory goes straight through the middle of the intersection, drifting slightly left, which none of the three intentions would do.

### What this metric rewards, and what it misses

- **Adding a forecast can only keep minADE the same or lower it.** Add a U-turn forecast $k_4$ and minADE and minFDE stay at 0.67 and 1. So compare only at the same $K$; NVIDIA reports Alpamayo 2 Super's minADE at $K=6$ and a 6.4 s horizon on its own selection of challenging samples (0.911 m, vendor-reported, see the [model page](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/llm-info/)).
- **It ignores probabilities.** Change $k_2$'s probability to 0.01 and minADE does not move. Planning uses the probabilities, so you also need confidence-aware metrics such as mAP or likelihood.
- **How "the best one" is chosen varies by benchmark.** Some take the forecast with the lowest ADE; some first pick one by endpoint error and then compute its ADE. Check the protocol before comparing papers.
- **Each road user is scored on its own.** Whether two cars' forecasts are consistent together is outside minADE.

## Joint or marginal forecasts

Two cars meet at an unprotected intersection; either A goes first or B does. Taken separately (the marginals), each car "goes first" with probability 0.5. Treat them as independent and multiply, and "both go first" gets $0.5\times0.5=0.25$: the planner believes a quarter of the possible futures contain the two cars colliding, and a quarter of the scenes sampled from this distribution are crashes. The real joint distribution looks more like (A goes, B yields) 0.5 and (A yields, B goes) 0.5, with both going close to 0.

Some major forecasting models, each described from its abstract:

- [MTR](https://arxiv.org/abs/2209.13508) (Motion Transformer) avoids dense goal candidates and uses a small set of learnable motion query pairs, each responsible for one motion mode, first locating the intention coarsely and then refining the trajectory.
- [Wayformer](https://arxiv.org/abs/2207.05844) is an attention-based forecaster that compares early, late and hierarchical fusion; early fusion works best.
- [MotionLM](https://arxiv.org/abs/2309.16534) (Waymo, ICCV 2023) treats multi-agent forecasting as language modelling over discrete motion tokens, decoding all road users' joint future in one autoregressive pass; its abstract claims first place on the Waymo Open Motion Dataset interactive leaderboard.

The [Waymo Open Motion Dataset](https://arxiv.org/abs/2104.10133) is a common dataset and benchmark.

<!-- paper-figure:motionlm-joint -->

## Planning: generate candidates, then pick one

The planner takes the route, map, forecasts and ego state and outputs the ego trajectory for the next few seconds. Method families differ mainly in "where candidates come from and how one is picked":

| Method | How trajectories are produced | Property | Limits | Primary source |
| --- | --- | --- | --- | --- |
| A* | Expands nodes on a discrete graph by "cost so far + heuristic" | Finds the shortest path when the heuristic never overestimates | Grid paths ignore the vehicle's turning radius | [Hart, Nilsson, Raphael 1968](https://doi.org/10.1109/TSSC.1968.300136) |
| Hybrid A* | Stores a continuous state $(x,y,\theta)$ inside discrete cells and expands by vehicle kinematics | Drivable paths, followed by smoothing and optimisation | Not guaranteed optimal; common in unstructured spaces such as car parks | [Dolgov et al. 2008](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf) |
| RRT / RRT* | Samples at random and grows a tree toward the samples | RRT is probabilistically complete; RRT* is asymptotically optimal | Jagged paths need post-processing; dynamics constraints must be added | [LaValle 1998](http://lavalle.pl/papers/Lav98c.pdf), [Karaman and Frazzoli 2011](https://arxiv.org/abs/1105.1186) |
| Frenet / lattice sampling | Combines lateral and longitudinal polynomials along a reference line into a batch of candidates, then picks by cost | Simple, parallel, easy to check constraints one by one | Cannot find good trajectories outside the candidate set | [Werling et al. 2010](https://doi.org/10.1109/ROBOT.2010.5509799) |
| Optimisation (including MPC) | Treats the trajectory as variables, minimises cost subject to constraints | Locally optimal; constraints can be written in explicitly | Depends on initialisation when non-convex | [Rawlings, Mayne, Diehl](https://sites.engineering.ucsb.edu/~jbraw/mpc/) |

On structured roads, sample-and-score is common, so let us work one.

## Work it once: choosing among 4 candidate trajectories

This continues the scene from the guide; the numbers are a teaching setup. Ego at 12 m/s; the lead car in this lane is 30 m ahead at a steady 8 m/s; a car in the left lane is 16 m behind at 13 m/s; the curb is on the right. The planning horizon is 4 s.

| Candidate | Distance in 4 s | Minimum gap | Peak acceleration | Progress $c_p$ | Comfort $c_c$ | Safety $c_s$ | Hard constraint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A keep lane, keep 12 m/s | 48 m | 14 m (lead car) | 0 | 0 | 0 | 2 | Pass |
| B slow evenly to 8 m/s in lane | 40 m | 22 m (lead car) | 1 m/s² | 2 | 1 | 0 | Pass |
| C change to left lane, keep 12 m/s | 48 m | 12 m (left-lane car behind) | 1.5 m/s² | 0 | 1.5 | 2.67 | Pass |
| D change to the right | — | — | — | — | — | — | Leaves the drivable area, rejected |

The gaps: A closes on the lead car at 4 m/s, $30-4\times4=14$; B's gap is $30-4t+0.5t^2$, smallest at $t=4$, which is 22; C's follower closes at 1 m/s, $16-4=12$. The three cost terms (also a teaching setup): progress $c_p=(48-\text{distance})/4$; comfort $c_c$ is the absolute peak acceleration; safety $c_s=\max(0,\,20-\text{minimum gap})/3$. The total cost:

$$
J=w_p\,c_p+w_c\,c_c+w_s\,c_s ,
$$

and among candidates that pass the hard constraints, take the one with the lowest $J$.

- **Weights $(1,1,1)$:** $J_A=2$, $J_B=2+1=3$, $J_C=1.5+2.67=4.17$; choose A.
- **Safety weight doubled, $(1,1,2)$:** $J_A=4$, $J_B=3$, $J_C=1.5+5.33=6.83$; choose B.

With $w_p=w_c=1$, A and B tie at $2w_s=3$, that is $w_s=1.5$. "What counts as good" hides in these weights; they come from human trade-offs, or are learned from human driving data.

The table says three more things:

1. **C is dominated by A.** Progress is the same and comfort and safety are worse, so non-negative weights guarantee $J_C\ge J_A$. C has strictly higher cost if either the comfort or safety weight is positive; if both are zero, C ties with A. To guarantee that C is not selected in a tie, prefer A in the tie-breaking rule or remove dominated candidates beforehand.
2. **D is rejected, not penalised.** Leaving the road is not "minus 10 points" but not allowed; a finite penalty can always be outweighed by a large enough gain, see the section on constraints in [planning and LQR](../../05-post-training/deep-rl/planning-and-control.en.md).
3. **A truncated horizon misleads.** At $t=4$ s, A is 14 m behind the lead car and still closing at 4 m/s; a 4 s cost cannot see how close A gets in seconds 5 and 6. Lengthen the horizon, or add a time-to-collision term, and A's cost rises. NAVSIM's score in note 3 includes exactly such a time-to-collision term.

<!-- widget:step-trajectory-cost -->

## Why a rule-based planner is still a strong baseline

[PDM](https://arxiv.org/abs/2306.07962) (Dauner et al., CoRL 2023) provides a concrete implementation. PDM-Closed first searches the route graph for the lane centreline to follow, then uses IDM (a car-following model) to generate the longitudinal speed along it: five target speeds at 20%, 40%, 60%, 80%, and 100% of the speed limit, each with three lateral offsets of −1 m, 0, and +1 m. This gives 15 candidates, each spanning 4 s at 10 Hz. Other vehicles are extrapolated at constant velocity.

Each candidate is then simulated with nuPlan's own LQR controller and kinematic bicycle model. Rules close to nuPlan's evaluation score compliance, progress, and comfort, and the highest-scoring candidate is selected. If that candidate is expected to collide within 2 s, the planner instead commands emergency braking.

<!-- paper-figure:pdm-arch -->

The paper's Val14 results (Table 2). CLS is the closed-loop score, with R meaning other vehicles react and NR that they do not; OLS is the open-loop score:

| Method | CLS-R | CLS-NR | OLS |
| --- | --- | --- | --- |
| IDM | 77 | 76 | 38 |
| PDM-Open (learned MLP, sees only the centreline and ego history) | 54 | 50 | 86 |
| PDM-Closed | 92 | 93 | 42 |
| Log Replay (replaying the human trajectory) | 80 | 94 | 100 |

Comparing only PDM-Open and PDM-Closed, the learned PDM-Open scores higher in open loop but lower on both closed-loop measures; the rule-based PDM-Closed shows the reverse pattern. These are not best-and-worst rankings across the whole table: Log Replay has a higher open-loop score than either. Even replaying the human trajectory does not get a perfect closed-loop score; one of the paper's explanations is that nuPlan's LQR controller sometimes drifts from the given trajectory, which is why PDM-Closed scores candidates by what the controller will actually drive. The authors list this as the first finding in their introduction: "Open- and closed-loop evaluation are misaligned" ([§1](https://arxiv.org/html/2306.07962v2#S1.p2)). The stronger claim in the abstract is about something else: that precise short-term planning and long-horizon ego-forecasting are fundamentally at odds and should be handled separately. A preliminary version of PDM-Hybrid ranked first in the 2023 nuPlan challenge (out of 25 teams).

## Where learning comes in

- **Learning the cost.** Lift-Splat-Shoot learns a BEV cost map and scores 1,000 template trajectories on it, see [note 1](bev-occupancy.en.md).
- **Learning the proposals, then searching.** At AI Day on 2021-08-19, Tesla said neural networks propose state and action distributions for Monte-Carlo tree search, scored by explicit costs such as collisions, comfort and time, and by intervention data; AI Day on 2022-09-30 presented a parallel tree search called "interaction search", with a decision every 50 ms (vendor-reported). Both are talks ([AI Day 2021](https://www.youtube.com/watch?v=j0z4FweCy4M), [AI Day 2022](https://www.youtube.com/watch?v=ODSJsviD_SU)), paraphrased from auto-captions, and describe Tesla's stack before its move to end-to-end; the 2021 version was presented as work in progress.
- **Learning the scorer.** Per its abstract, [Hydra-MDP](https://arxiv.org/abs/2406.06978) distils a student network from human demonstrations and rule-based teachers, with one decoder head per evaluation metric, and ranked first in the NAVSIM end-to-end track of the CVPR 2024 challenge.
- **Learning the whole planner with a large model.** Waymo's 2025-06-13 [blog post](https://waymo.com/blog/2025/06/scaling-laws-in-autonomous-driving/) reports that for models trained on about 500,000 hours of internal driving data, motion-forecasting quality improves as a power law in training compute and closed-loop planning metrics also improve with scale (vendor-reported). The companion [technical report](https://arxiv.org/abs/2506.08228) (arXiv 2025-06-09, autoregressive encoder-decoder Transformers) adds that compute-optimal scaling grows model size about 1.5 times as fast as data, and notes that this bears on whether open-loop metrics are suitable for guiding model development.

## Handing the trajectory to control

The planner replans every few tens to a hundred milliseconds, and the controller tracks the current trajectory at a higher rate: this is the MPC rhythm of "plan a sequence, execute only the first step" from [the model-based note](../../05-post-training/deep-rl/model-based.en.md). Tracking itself often uses LQR or MPC, derived in [planning and LQR](../../05-post-training/deep-rl/planning-and-control.en.md). nuPlan, NAVSIM and PDM all execute trajectories with LQR plus a kinematic bicycle model, so "the planned trajectory" and "the path the car actually drives" differ, and evaluations must say which one they mean.

## What to check when reproducing

| What to try first | What to observe | What you can NOT conclude |
| --- | --- | --- |
| Compare minADE at a fixed $K$, and also report $K=1$ | How much multimodality buys | A low minADE at large $K$ means good forecasts |
| Shuffle the probabilities across forecasts | minADE / minFDE unchanged; mAP or likelihood worse | Probability calibration does not matter |
| Break results down into straight, turning and interactive | Whether turning and interactive segments have clearly larger errors | The average error represents hard scenes |
| Double each cost weight in turn | In which scenes the choice flips | One set of weights is right for every scene |
| Stretch the planning horizon from 4 s to 8 s | Whether candidates like A, which fail only beyond the horizon, are caught | Longer horizons are always better (forecast error also grows with horizon) |
| Report open-loop and closed-loop scores together | Whether the two rankings agree (in the PDM paper they do not) | A planner with a high open-loop score is also good in closed loop |

References: [PDM](https://arxiv.org/abs/2306.07962) · [MTR](https://arxiv.org/abs/2209.13508) · [MotionLM](https://arxiv.org/abs/2309.16534) · LaValle's free textbook [Planning Algorithms](http://lavalle.pl/planning/) · Rawlings, Mayne and Diehl's [Model Predictive Control](https://sites.engineering.ucsb.edu/~jbraw/mpc/). Previous: [from surround cameras to bird's-eye view and occupancy](bev-occupancy.en.md); next: [end-to-end models and closed-loop evaluation](end-to-end.en.md).
