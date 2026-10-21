# Autonomous driving: from sensors to a trajectory

[中文](README.md) · **English**

> Reading time: ~7 min · Last reviewed: 2026-10

A car carries 6–8 cameras, often lidar and radar as well, producing dozens of frames per second; yet every few tens of milliseconds it only has to hand over one thing: a trajectory for the next few seconds, or the steering, throttle and brake commands directly. What happens in between? **The classic approach splits the path into perception, prediction, planning and control, with interfaces that can each be checked on their own; the end-to-end approach trains them as one network and turns the intermediate outputs into optional auxiliary heads. Either way, the hard part is not driving like a human on a recorded log, but showing that the car stays safe when it drives itself.**

## One scene, four questions

Start with one scene that lays out the problem; note 2's planning example uses it: you are driving at 12 m/s (about 43 km/h) on a two-lane road, a car 30 m ahead in your lane is doing only 8 m/s, a car is closing in from behind in the left lane, and there is a curb on the right. The system has to answer, in order:

1. What is around me, and where? That is perception; note 1 shows how several cameras become one top-down map.
2. How might the lead car and the car behind move over the next few seconds? That is prediction; note 2 explains why it must give several possibilities.
3. Which path do I take? That is planning; note 2 works through 4 candidate trajectories.
4. How do I make the car actually follow that path? That is control, already covered in [planning and LQR](../../05-post-training/deep-rl/planning-and-control.en.md).

## The classic modular stack

| Module | Input | Output | Common methods | Where it is covered |
| --- | --- | --- | --- | --- |
| Perception | Multi-camera images, lidar point clouds, radar | Bird's-eye-view features, 3D boxes, lane lines, occupancy grid | Lift-splat, BEVFormer, BEVFusion, occupancy networks | [Note 1](bev-occupancy.en.md); camera geometry in the [perception series](../perception/camera-geometry.en.md) |
| Tracking and state estimation | Per-frame detections | Object tracks with IDs and velocities | Kalman filter, data association | [State estimation and tracking](../perception/tracking.en.md) |
| Prediction | Object histories, map | Several future trajectories per object, with probabilities | MTR, Wayformer, MotionLM | [Note 2](prediction-planning.en.md) |
| Planning | Predictions, map, route, ego state | Ego trajectory for the next few seconds | Search, sample-and-score, optimisation | [Note 2](prediction-planning.en.md) |
| Control | Target trajectory, ego state | Steering, throttle, brake | PID, LQR, MPC | [Planning and LQR](../../05-post-training/deep-rl/planning-and-control.en.md) |

Splitting it up means every interface can be read, tested and replaced on its own, and after an incident you can find which layer was wrong. The cost is that people decide the interfaces: if perception only hands over boxes, planning cannot see what a box cannot describe, such as an oddly shaped piece of debris on the road; and errors upstream pass down layer by layer, while downstream modules cannot ask upstream ones to "optimise for planning". Occupancy grids were added for the first of these problems, as note 1 explains.

## End-to-end: replacing interfaces with gradients

"End-to-end" means different things in different systems, at least three:

- **Intermediate tasks remain, but are trained together.** [UniAD](https://arxiv.org/abs/2212.10156) (CVPR 2023 Best Paper) chains tracking, mapping, motion forecasting, occupancy and planning through queries in one network, with planning as the final goal; the tasks are linked by queries and can be trained together.
- **One network from pixels to actions.** Tesla's [Q4 2023 shareholder letter](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2023-Update.pdf) says V12 uses AI to influence vehicle controls (steering wheel, pedals, indicators and so on) instead of hard-coding every driving behaviour; in the [text version](https://x.com/aelluswamy/status/1981644831790379245) of his October 2025 ICCV talk (posted on X on 2025-10-24), Tesla's Ashok Elluswamy wrote that one end-to-end network takes pixels from the cameras, kinematic signals such as vehicle speed, audio, maps and navigation, and directly produces the control commands that drive the car; it can also produce interpretable intermediate tokens, such as 3D geometry.
- **A large model with structured outputs.** The Foundation Model described in Waymo's 2025-12-09 [blog post](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving) keeps structured outputs such as objects and the road graph, adds a "Driving VLM" trained with Gemini, distils large Teacher models that cannot run in the car into onboard Students, and adds a separate validation layer on board.

| Comparison | Modular stack | End-to-end model |
| --- | --- | --- |
| Interfaces | Defined by people: boxes, trajectories, maps | Learned features; structured outputs optional |
| Training signal | Each module has its own labels | The planning or driving objective's gradient runs through the whole network |
| Locating a failure | You can find the layer | Needs auxiliary heads, probes, simulated replay |
| Main risk | Interfaces lose information; errors pass downstream | Learned shortcuts: good open-loop scores, closed-loop failure |
| Main validation | Module tests plus system tests | Closed-loop simulation, road testing and a safety case |

Real systems mostly sit in between. Note 3 explains why "good open-loop score, closed-loop failure" matters especially for end-to-end models.

## How the three notes fit

- **Note 1** answers "where are things": how multi-camera images become one metric top-down grid, and what an occupancy grid adds over boxes.
- **Note 2** answers "how will others move, and which path do I take": why forecasts must offer several possibilities, how candidate trajectories are chosen by cost, and why a rule-based planner is still a strong baseline.
- **Note 3** answers "how do you judge a model that does all of this": why open-loop scores mislead, what closed-loop and near-closed-loop evaluations measure, and what safety case is needed before driving on public roads.

## What to read first

- From the perception series: [camera geometry](../perception/camera-geometry.en.md) (note 1 of this series uses it directly), [state estimation and tracking](../perception/tracking.en.md) and [detection and segmentation](../perception/detection.en.md).
- MPC and planners exploiting model errors: [Model-based RL](../../05-post-training/deep-rl/model-based.en.md); LQR and constraints: [planning and LQR](../../05-post-training/deep-rl/planning-and-control.en.md); compounding error in behaviour cloning: [imitation and rewards](../../05-post-training/deep-rl/imitation-and-rewards.en.md), on which note 3's open-loop vs closed-loop comparison builds.
- World models as driving simulators: [world models: robots and driving](../../11-frontier/world-models/robotics-driving.en.md).

If you are heading toward autonomous driving work, one order is: perception note 1 (camera geometry) → perception note 2 (state estimation and tracking) → notes 1, 2 and 3 of this series, then perception note 5, [3D reconstruction and neural rendering](../perception/reconstruction.en.md).

## As of October 2026: state of the field and deployment

> **State as of 2026-10-15 (this changes quickly; check dates before citing)**
>
> - **Learned simulators became the main direction for closed-loop testing.** Tesla presented a neural world simulator in the [text version](https://x.com/aelluswamy/status/1981644831790379245) of its October 2025 ICCV talk; the [Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation) (2026-02-06); Wayve [GAIA-4](https://wayve.ai/thinking/gaia-4/) (2026-08-03). See the [world-models series](../../11-frontier/world-models/robotics-driving.en.md).
> - **nuScenes open-loop L2 and collision rates have largely left the stage.** The [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) maintainers stopped reporting them; 2026 papers mostly report NAVSIM v1/v2, HUGSIM, nuPlan and Bench2Drive. Note 3 explains why.
> - **Driving VLAs with open weights exist.** NVIDIA's [Alpamayo 2 Super](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/llm-info/) (34B) released weights on 2026-08-04 under the OpenMDW-1.1 licence; its performance numbers are all NVIDIA's own.

The operating numbers below are all vendor-reported unless stated otherwise; each refers to a different date, so they cannot be compared directly.

| Company | As of | Public statement | Nature |
| --- | --- | --- | --- |
| Waymo | 2026-04-29 | More than 500,000 fully autonomous rides a week (Pichai in Alphabet's Q1 [earnings release](https://www.sec.gov/Archives/edgar/data/0001652044/000165204426000043/googexhibit991q12026.htm)) | Vendor-reported; as of 2026-10-15 we found no newer weekly figure |
| Waymo | 2026-10-15 | The [updates page](https://waymo.com/updates/) lists 15 U.S. cities serving public riders | Company statement |
| Waymo | 2026-06 | 271.3 million cumulative rider-only miles ([Safety Impact](https://waymo.com/safety/impact/)) | Vendor-reported |
| Tesla | 2026-07-22 | Tesla's Q2 2026 [shareholder letter](https://www.sec.gov/Archives/edgar/data/1318605/000162828026049213/exhibit991.htm) says Robotaxi is live in 7 major metros: the Bay Area with a safety driver; Austin, Dallas, Houston, Miami, Orlando and Tampa marked "Ramping Unsupervised" | Tesla's own statement; the letter does not say any city is fully unsupervised |
| Tesla | 2026-09-17 | On the California DMV [list](https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/autonomous-vehicle-testing-permit-holders/), Tesla Robotaxi LLC holds only a testing permit with a safety driver | Regulator list |
| Zoox | 2026-07-31 | NHTSA granted a Part 555 exemption to deploy commercially, up to 2,500 vehicles per 12 months ([Federal Register](https://public-inspection.federalregister.gov/2026-15485.pdf)) | Regulatory document |
| Baidu Apollo Go | 2026 Q1 / Q2 | 3.2 million fully driverless rides in Q1 ([quarterly release](https://www.sec.gov/Archives/edgar/data/0001329099/000119312526228123/d156481dex991.htm)); about 1 million in Q2; on the 2026-08-18 call Robin Li said ride volume was affected by "operational adjustments" in certain domestic cities due to regulatory considerations, and that operations in those cities began to resume in August ([call transcript](https://www.fool.com/earnings/call-transcripts/2026/08/18/baidu-bidu-q2-2026-earnings-call-transcript/), by The Motley Fool) | Vendor-reported; the Q2 [press release](https://www.prnewswire.com/news-releases/baidu-announces-second-quarter-2026-results-302853860.html) gives no ride count, so the figure comes only from the call |

## Reading order

1. [From surround cameras to bird's-eye view and occupancy](bev-occupancy.en.md): lift-splat by hand, BEV queries, sparse queries, camera–lidar fusion, occupancy grids and their labels
2. [Motion forecasting and planning](prediction-planning.en.md): minADE / minFDE by hand, joint vs marginal forecasts, search and sampling planners, a cost-weighted choice among 4 candidate trajectories, PDM-Closed
3. [End-to-end models and closed-loop evaluation](end-to-end.en.md): what end-to-end means, imitation and self-play, open vs closed loop by hand, NAVSIM / Bench2Drive / CARLA, safety standards, deployment
