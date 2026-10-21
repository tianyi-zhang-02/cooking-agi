# Autonomous driving: end-to-end models and closed-loop evaluation

[中文](end-to-end.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

Two driving models have exactly the same L2 error in open-loop evaluation. In closed-loop simulation, one crosses the lane line in under two seconds while the other stays in its lane. Why?

Open-loop evaluation returns the car to the logged position at every step, so the model's small deviations do not accumulate. In closed loop, those deviations change what it sees next. As end-to-end models become more capable, closed-loop or near-closed-loop evaluation remains essential. Before road deployment, they also need a safety case.

## What "end-to-end" actually means

What they share: the gradient of the planning or driving objective can flow back into perception, and intermediate outputs are no longer required interfaces. They differ in how much structure stays inside and whether a language model takes part.

| System | Date | Input → output | Internal structure | Nature of the source |
| --- | --- | --- | --- | --- |
| [UniAD](https://arxiv.org/abs/2212.10156) | arXiv 2022-12, CVPR 2023 Best Paper | Multi-camera → planned trajectory | Tracking, mapping, motion forecasting, occupancy and planning chained by queries | Paper and [README](https://github.com/OpenDriveLab/UniAD) |
| [EMMA](https://waymo.com/blog/2024/10/introducing-emma/) (Waymo) | 2024-10 | Raw camera images → trajectories, 3D objects and road graph as text | Gemini-based multimodal model, switched by task prompts | Waymo blog; says it uses no lidar or radar and has high inference cost; not presented as deployed |
| Waymo Foundation Model | 2025-12-09 | Camera, lidar, radar → trajectories, objects, maps, validation signals | Sensor fusion encoder plus a Gemini-trained Driving VLM; large Teachers distilled into onboard Students | [Waymo blog](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving); no parameter counts |
| Tesla FSD v12–v14 | From 2023-12 | Cameras and more → vehicle controls | Tesla has published no architecture paper | Shareholder letters and talks, see below |
| [Alpamayo](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/llm-info/) (NVIDIA) | [R1](https://arxiv.org/abs/2511.00088) 2025-10; 2 Super weights 2026-08-04 | Cameras → reasoning text and trajectory | Cosmos reasoning VLM plus a diffusion trajectory decoder; 2 Super is 32B plus a 2.3B action expert | Vendor-reported; the AlpaSim score has no external definition |

<!-- paper-figure:uniad-pipeline -->

Below, only Tesla's own documents and talks, by date; all numbers are vendor-reported, and none of these primary sources gives a layer-by-layer structure for v12–v14:

- [Q4 2023 shareholder letter](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2023-Update.pdf): the V12 rollout began in late December 2023; it uses AI to influence vehicle controls instead of hard-coding every driving behaviour, and the letter says it uses end-to-end training.
- [Text version](https://x.com/aelluswamy/status/1981644831790379245) of the October 2025 ICCV talk (Elluswamy, 2025-10-24): the inputs are pixels from the cameras, kinematic signals such as vehicle speed, audio, maps and navigation; the output is the control commands that drive the car.
- [Q4 2025 letter](https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q4-2025-Update.pdf): FSD runs on an end-to-end foundation model trained on real-world data from both customer cars and Robotaxi.
- Q1 and Q2 2026 letters (SEC copies: [Q1](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026026551/exhibit991.htm), [Q2](https://www.sec.gov/Archives/edgar/data/1318605/000162828026049213/exhibit991.htm)): v14.3 upgraded the RL stage for long-tail cases and cut inference latency by up to 20%; v14 lite distils the driving behaviour of AI4's v14 into AI3 hardware.
- CVPR Workshop on Autonomous Driving (WAD) keynote, 3 June 2026 ([schedule](https://cvpr2026.wad.vision/), [video](https://www.youtube.com/watch?v=2V0ddkF1Svo)): according to the video's auto-captions, Elluswamy said the network runs as one copy on each of two redundant AI4 chips; there is no transcript to check this against.

## Beyond imitation: RL and self-play

Most end-to-end models start with imitation learning on human driving logs and face the compounding-error problem described in [imitation and rewards](../../05-post-training/deep-rl/imitation-and-rewards.en.md). The logs contain almost exclusively states where the human stays in the lane centre, with no demonstrations of how to recover after drifting. An early approach, [ChauffeurNet](https://arxiv.org/abs/1812.03079), adds synthetic perturbations to the training data so the model encounters off-centre states.

The other route is to let the model drive itself in a simulator and learn from outcomes:

- Apple's [GigaFlow](https://arxiv.org/abs/2502.03349) ([ICML 2025](https://proceedings.mlr.press/v267/cusumano-towner25a.html)) does RL with self-play only in a batched simulator. The authors report 1.6 billion km simulated and 17.5 years on average between incidents in simulation, with state-of-the-art results on CARLA, nuPlan and Waymax; their stated limits are simulation only, with perception abstracted away.
- In the [text version](https://x.com/aelluswamy/status/1981644831790379245) of his October 2025 ICCV talk, Elluswamy wrote that Tesla's neural world simulator can validate new driving models against historical data, synthesise new adversarial scenarios to test corner cases, and run large-scale closed-loop RL. He also wrote that loss on open-loop predictions may not correlate with real-world performance, and that evaluation is the most difficult challenge.

RL gets around "the logs contain no off-centre states", but brings a new problem: wherever the simulator differs from the real world, the policy may exploit it, the same thing as a planner exploiting model errors in [the model-based note](../../05-post-training/deep-rl/model-based.en.md).

## Work it once: the same open-loop score, opposite closed-loop outcomes

Everything below is a teaching setup. A straight road, lane width 3.5 m, car width 1.8 m. To isolate lateral drift, approximate the body as a fixed-width interval that remains parallel to the lane, ignoring vehicle length and the effect of heading on its footprint. The body boundary then touches the lane line when the centre is $(3.5-1.8)/2=0.85$ m off-centre. At 1.75 m the centre reaches the line; it crosses only beyond that offset. Body width is not track width, and tyre width is unspecified, so this does not establish when the wheels touch the line. Speed $v=10$ m/s. In the log, the human drives straight down the lane centre. Both policies have the same small bias $\varepsilon=0.02$ rad (about 1.15°):

- **Policy A: extrapolate along the current ego heading, plus $\varepsilon$.** It looks only at the ego state, not at lane lines.
- **Policy B: look at the lane lines and pull toward the centre.** Planned heading $=\varepsilon-g\,y$, where $y$ is the car centre's offset to the left (m) and $g=0.1$ rad/m.

**Open loop**, nuScenes-style: at every frame, feed the model the logged state ($y=0$, heading 0), let it output a 3 s trajectory, and compare with the log. In this state both policies output a straight line at heading 0.02, with lateral offset $10\times t\times0.02=0.2t$: L2 errors of 0.2, 0.4 and 0.6 m at 1, 2 and 3 s, averaging 0.4 m. The next frame goes back to the logged state, and the error starts from zero again.

**Closed loop**: execute the first 0.5 s (5 m) of each plan, then replan from where the car actually is. With the small-angle approximation $\sin\theta\approx\theta$:

$$
\text{A:}\ \theta_k=\theta_{k-1}+\varepsilon,\qquad
\text{B:}\ \theta_k=\varepsilon-g\,y_{k-1},\qquad
y_k=y_{k-1}+v\,\Delta t\,\theta_k .
$$

| Time (s) | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
| --- | --- | --- | --- | --- | --- | --- |
| A heading (rad) | 0.02 | 0.04 | 0.06 | 0.08 | 0.10 | 0.12 |
| A lateral offset (m) | 0.1 | 0.3 | 0.6 | 1.0 | 1.5 | 2.1 |
| B lateral offset (m) | 0.100 | 0.150 | 0.175 | 0.188 | 0.194 | 0.197 |

A's heading grows by 0.02 per step and its offset grows as $0.05\,k(k+1)$: with the simplified footprint above, the body boundary touches the line between 1.5 s and 2.0 s, and at 3 s the car's centre is already in the next lane. B follows $y_k=0.5\,y_{k-1}+0.1$ each step and converges to $\varepsilon/g=0.2$ m, staying in its lane throughout under this approximation.

| Metric | Policy A | Policy B |
| --- | --- | --- |
| Open-loop L2 (1 / 2 / 3 s) | 0.2 / 0.4 / 0.6 m | 0.2 / 0.4 / 0.6 m |
| Closed-loop lateral offset after 3 s | 2.1 m | 0.197 m |
| Body boundary touches the line (simplified footprint) | Between 1.5 and 2.0 s | Never |

Open loop cannot tell them apart because the log contains no state with $y\neq0$; "what do you do after drifting" is never asked. Change one condition: a policy that ignores the road entirely and just extrapolates the current speed and heading has an open-loop L2 of 0 on this straight road, a perfect score. Policy A is exactly such an "ego-state extrapolator" with a small bias added.

<!-- widget:step-open-closed-loop -->

## Why nuScenes open-loop metrics are no longer trusted

nuScenes open-loop planning evaluation compares the model's predicted ego trajectory with the logged one, reporting L2 error and collision rate at 1, 2 and 3 s. Some primary results:

- [Li et al.](https://arxiv.org/abs/2312.03031) (CVPR 2024) count 73.9% of nuScenes as straight driving; their abstract says "These models tend to rely predominantly on the ego vehicle's status for future path planning", and that chasing this leaderboard may not yield convincing conclusions.
- The abstract of [AD-MLP](https://arxiv.org/abs/2305.10430) says an MLP that takes only past trajectory and velocity, with no images or lidar, actually lowers average L2 by about 20%, though its collision rate is worse than that of methods using perception. Put the same AD-MLP into Bench2Drive's closed-loop test and its success rate is 0 (the README's v0.0.4 table).
- Implementations are not uniform either. UniAD's own [README](https://github.com/OpenDriveLab/UniAD) reports a nuScenes average L2 of 0.90 m, while the comparison table in the [SparseDrive](https://github.com/swc-17/SparseDrive) README lists UniAD at 0.73 m; collision rates are also computed differently across papers.
- The [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) README (2025-02-05) calls this metric unpersuasive, and its maintainers stopped reporting it.

## What closed-loop and near-closed-loop evaluations measure

| Evaluation | Can the car leave the log? | Do other vehicles react? | Main metric | Main limits |
| --- | --- | --- | --- | --- |
| nuScenes open loop | No, back to the log every step | Not applicable | L2, collision rate | Mostly straight driving, ego-state shortcut, inconsistent implementations |
| NAVSIM v1 | Only within one 4 s simulation | No | PDMS | Ignores compounding error and other vehicles' reactions |
| NAVSIM v2 | Stage two starts from perturbed start states | No | EPDMS | Start states are precomputed; images come from reconstruction rendering |
| nuPlan | Yes | Yes in CLS-R | CLS-R / CLS-NR / OLS | Inputs are given object states; perception is not tested |
| CARLA / Bench2Drive | Yes, drives the whole route itself | Yes, scripted | Driving score DS, success rate | Images and traffic come from a game engine |
| Learned world models | Yes | Depends on the model | Each company's own | No checkable fidelity numbers published as of 2026-10 |

The last row means generative simulators such as the Waymo World Model (2026-02), Wayve GAIA-4 (2026-08) and Tesla's neural world simulator (from auto-captions of its talks); why pure reconstruction breaks down once the car leaves the log is covered in [3D reconstruction and neural rendering](../perception/reconstruction.en.md), and how world models serve as evaluators in [world models: robots and driving](../../11-frontier/world-models/robotics-driving.en.md).

### NAVSIM: hand over one plan, simulate 4 seconds

[NAVSIM v1](https://arxiv.org/abs/2406.15349) (NeurIPS 2024) hands the model's plan to an LQR controller, simulates it non-reactively for 4 s in a top-down view, and scores it by rules:

$$
\text{PDMS}=\text{NC}\times\text{DAC}\times\frac{5\,\text{EP}+5\,\text{TTC}+2\,\text{C}}{12}
$$

NC is no at-fault collisions, taking 0, 1/2 or 1; DAC is drivable-area compliance, 0 or 1; EP is ego progress, between 0 and 1; TTC means time-to-collision stays above its bound, and C is comfort, both 0 or 1 ([metrics documentation](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md)). Work it once: NC = DAC = 1, EP = 0.8, TTC = C = 1 gives PDMS $=(4+5+2)/12\approx0.917$, written 91.7 in papers. If, like candidate A in note 2, the car gets too close to the lead car and TTC = 0, PDMS drops to $(4+0+2)/12=0.5$; if it leaves the road and DAC = 0, it is 0 however good the rest is. The structure matches note 2: hard constraints multiply, soft terms are a weighted average.

PDMS on the paper's navtest split: human 94.8, TransFuser 84.0, UniAD 83.4, an MLP using only the ego state 65.6, constant-velocity extrapolation 20.6. Here the ego-state-only method falls clearly behind, which nuScenes L2 does not show. The paper also states its own limits: it does not simulate other vehicles' reactions, ignores compounding error, has no traffic-light or stop-sign rules, and recommends running CARLA alongside it.

Back to our example: from the logged start, A and B give the very same plan, with a lateral offset of $0.2\times4=0.8$ m after 4 s, so their PDMS is naturally the same. v1 cannot tell them apart either, which is exactly what "ignores compounding error" means.

[NAVSIM v2](https://arxiv.org/abs/2506.04218) (CoRL 2025) runs "pseudo-simulation" in two stages: stage one scores the plan as usual; stage two scores again from a set of precomputed, perturbed start states rendered with 3D Gaussian splatting, weighting each by how close its start is to the stage-one endpoint, and the two stages are multiplied. Its EPDMS adds driving direction and traffic lights to the multiplied terms, replaces comfort with history comfort, and adds lane keeping and extended comfort as weighted terms ([metrics documentation](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md)). The abstract reports an $R^2$ of 0.8 against closed-loop results, versus 0.7 for the best open-loop metric. By this design, if stage two's start states include something like "already 0.8 m left with the nose turned", A keeps drifting while B pulls back, and only then do the lane-keeping and drivable-area terms get a chance to separate them.

<!-- paper-figure:navsim2-pseudosim -->

### CARLA and Bench2Drive: let the car drive the whole route

The [CARLA Leaderboard 2.0](https://leaderboard.carla.org/evaluation_v2_0/) driving score is route completion times an infraction penalty, multiplying by the matching coefficient for each infraction: pedestrian collision 0.50, vehicle collision 0.60, static collision 0.65, red light 0.70, stop sign 0.80.

$$
\text{DS}=\text{RC}\times\prod_j p_j^{\,n_j}
$$

Work it once: 90% of the route completed, one vehicle collision and one red light, $\text{DS}=90\times0.60\times0.70=37.8$. Another model stops halfway with no infractions, $\text{DS}=50\times1=50$, which is higher. This is not hypothetical: the CARLA site says participants found that ending a route early scored better than completing it, so version 2.1, from March 2025, changed the infraction penalty from multiplicative to linear, and 2.1 scores are not directly comparable with 2.0; the site also shows no challenge was held in 2025. Read DS together with route completion and infraction counts.

[Bench2Drive](https://arxiv.org/abs/2406.03877) (NeurIPS 2024) tests closed-loop abilities one by one in CARLA 0.9.15, with 220 routes across 44 scenario types; its dataset is 2 million frames from 13,638 clips. Driving scores collected in the [README](https://github.com/Thinklab-SJTU/Bench2Drive) for v0.0.4 (2026-08-06) range from 38.69 for UniAD to 95.59 for LEAD; the maintainers collected this table, and training setups may differ between methods.

## Safety cases: what each standard covers

Evaluation scores answer "how well does it drive"; a safety case answers "why should anyone believe it is safe enough within its operational design domain". ISO standard texts are paid, and the pages also refused our automated fetch, so the ISO and SAE rows are written from the standard titles and their usual positioning, without reading the texts:

| Standard | What it covers | Relation to this note |
| --- | --- | --- |
| ISO 26262 (road vehicles, functional safety; edition 2, 2018) | Hazards caused by malfunctions of electrical and electronic systems | A sensor dies, a chip errs; the redundant chips from Tesla's talk above belong, in our reading, to this category |
| ISO 21448:2022 (safety of the intended functionality, SOTIF) | Hazards with no broken part, where the function itself is insufficient, or from foreseeable misuse | Policy A is not broken, just under-trained; in our reading it is closer to this category |
| ISO/PAS 8800:2024 (road vehicles, safety and artificial intelligence) | Safety of AI components in road vehicles | An end-to-end network is exactly such a component |
| [UL 4600](https://www.ul.com/news/ul-4600-edition-3-updates-incorporate-autonomous-trucking) (edition 3, 2023-03-17) | Built around a safety case, goal-based and not prescribing specific technology; sets no fixed road-test mileage and calls for a mix of analysis, simulation, closed-course and public-road testing | Tells you what evidence a case needs, not which model to use |
| SAE J3016 | Taxonomy of driving-automation levels 0–5 | Makes clear whether a system is driver assistance or driverless |

End-to-end models make "arguing component by component" harder: there is no module called "perception" that can produce its own test report. Waymo's approach is a separate validation layer on board, with the same base model serving as Driver, Simulator and Critic ([blog](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving)); it has also published its safety methodology ([Webb et al. 2020](https://arxiv.org/abs/2011.00054)). Another approach writes "what counts as safe" as formal rules, for example [RSS](https://arxiv.org/abs/1708.06374).

## As of October 2026: deployment and public evidence

Operating on public roads is the ultimate closed-loop test. Each company's ride counts and city counts are in the [guide](README.en.md); here we list only public evidence that bears on "how it performs in closed loop".

> **As of 2026-10-15 (this changes quickly; check dates before citing)**
>
> - **Waymo's [Safety Impact](https://waymo.com/safety/impact/) (data through 2026-06, vendor-reported):** 271.3 million rider-only miles; compared with a human driving benchmark on similar city streets, 95% fewer crashes with serious injury or worse (55 fewer) and 82% fewer injury-causing crashes. This is Waymo's own relative comparison; absolute serious-injury counts are small, and it covers only 5 mature markets.
> - **The long tail exposed by recalls (NHTSA recall filings):** Waymo [26E026](https://static.nhtsa.gov/odi/rcl/2026/RCLRPT-26E026-6527.pdf) (2026-05-06, 3,791 vehicles): on higher-speed roads, may slow but not stop for a flooded lane; Waymo 26E035 (2026-06, 3,871 vehicles): could enter a closed freeway construction zone at speed (read via an [aggregator](https://vehiclesafetyrecalls.com/recall/26E035000)); Zoox [26E044](https://static.nhtsa.gov/odi/rcl/2026/RCLRPT-26E044-2993.pdf) (2026-07-16, 105 driverless vehicles): may not detect and respond to heavy smoke at emergency scenes.
>
> Flooded lanes, construction zones and smoke are all rare in training logs, and open-loop L2 cannot measure them.

## What to check when reading an end-to-end paper

| What to look at first | What you should find | Without it, you can NOT conclude |
| --- | --- | --- |
| Which benchmark and version | NAVSIM v1 or v2, CARLA 2.0 or 2.1 | Scores are comparable across versions |
| An ablation that removes the ego state | How far the score drops without it | The model really uses perception |
| Route completion and infraction counts in closed loop | Reported alongside DS | A high DS means good driving |
| A comparison with rule-based baselines | PDM-Closed, constant-velocity extrapolation | The learned planner is better |
| Whether numbers are self-reported or third-party | An official leaderboard or independent reproduction | The conclusion carries over to other settings |
| Whether training and test cities and routes are separated | An explicit split | It generalises to new cities |

References: [Parting with Misconceptions (PDM)](https://arxiv.org/abs/2306.07962) · [DAgger](https://arxiv.org/abs/1011.0686). Previous: [motion forecasting and planning](prediction-planning.en.md); back to the [guide](README.en.md).
