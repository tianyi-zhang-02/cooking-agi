# World models for robots and cars: simulator, evaluator, and policy

[中文](robotics-driving.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

You have 5 checkpoints of a cup-pushing policy and want to know which is best. Fifty real trials each makes 250 trials, and someone has to reset the cup and make sure it does not crash off the table. Could a world model "act it out" first? **In robotics and autonomous driving, world models do three different jobs: generate training data, score policies, and act as the policy. Each job needs different validation; as an evaluator, a high correlation with real success rates still does not guarantee you pick the best policy.**

## One model, three jobs

The [Cosmos paper](https://arxiv.org/abs/2501.03575) (NVIDIA, January 2025) writes a world foundation model $\mathcal W$ as predicting the next state from past observations and a current "perturbation" (Fig. 3).

$$
\hat x_{t+1}=\mathcal W(x_{0:t},\,c_t)
$$

$x_{0:t}$ are the observations up to time $t$, usually video frames; $c_t$ is the current perturbation, which can be a robot action, an ego-vehicle trajectory or a piece of text. The same predictor used in different ways does different jobs:

| Job | Input → output | What it replaces | What must be validated | Examples |
| --- | --- | --- | --- | --- |
| Data engine | Scene + instruction → synthetic video, then inferred actions | Collecting more real demonstrations | Whether policies trained on the synthetic data improve on real robots | DreamGen, Cosmos |
| Policy evaluator | Policy actions → rollout inside the model → success or not | Real-robot evaluation | Whether the model's ranking matches the real ranking | WorldEval, WorldGym, Veo evaluator, dWorldEval, Ctrl-World, 1X |
| The policy itself | Current view + task → predicted future → action | A separately trained VLA policy | Closed-loop success, latency, generalisation | DreamZero, Cosmos Policy, 1X NEO, Praxis-1 |

[WorldArena](https://arxiv.org/abs/2602.08971) (Shang et al., February 2026 paper) tested 14 embodied world models in exactly these three uses, data engine, policy evaluator and planner, and concluded in the abstract that "high visual quality does not necessarily translate into strong embodied task capability"; the authors call it a perception–functionality gap. [World-in-World](https://arxiv.org/abs/2510.18135) (October 2025, ICLR 2026 Oral) also reports that closed-loop task success depends more on controllability than on visual quality.

## Job one: generating data

The abstract of [DreamGen](https://arxiv.org/abs/2505.12705) (NVIDIA GEAR et al., May 2025 paper) describes the pipeline: adapt an image-to-video model to the target robot, then generate videos of familiar or new tasks in varied scenes; the videos have no action labels, so pseudo-actions are inferred with a latent action model or an inverse dynamics model (IDM). This is the same idea as Genie's latent actions in the [interactive video note](interactive-video.en.md). The paper calls this synthetic data neural trajectories, and reports that a humanoid learned 22 new behaviours from teleoperation data for a single pick-and-place task in a single environment.

<!-- paper-figure:dreamgen-overview -->

The cost counts too. The [DreamGen paper](https://arxiv.org/abs/2505.12705) reports that its 240k-sample RoboCasa set took 54 hours on 1,500 L40 GPUs, which is $1500\times54=81{,}000$ GPU-hours, or about $81{,}000/240{,}000\approx0.34$ GPU-hours per sample, roughly 20 minutes of GPU time per sample. Whether a real demonstration is expensive has to be judged against that figure.

NVIDIA's Cosmos is the most complete open series on this line. The table lists only what we could confirm in public sources; all dates are NVIDIA's release dates:

| Date | Version | What was released |
| --- | --- | --- |
| 2025-01 | Cosmos WFM platform | Open diffusion and autoregressive world foundation models (4B–14B), tokenizers, a curation pipeline; framed as a "digital twin of the world" for physical AI |
| 2025-03 | [Cosmos-Transfer1](https://arxiv.org/abs/2503.14492) / [Reason1](https://arxiv.org/abs/2503.15558) | Control map → video; a physical-reasoning VLM (7B / 56B) |
| 2025-06-11 | [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) | 2B / 14B Video2World, plus an action-conditioned model |
| 2025-10-28 | [Predict2.5 / Transfer2.5](https://arxiv.org/abs/2511.00062) | A unified flow model (2B / 14B) trained on 200M curated clips |
| 2026-01-06 | [CES update](https://forums.developer.nvidia.com/t/nvidia-cosmos-announcements-at-ces-2026/356629) | Cosmos Reason 2, Predict 2.5, Transfer 2.5-2B |
| 2026-03-16 / 05-31 | [Cosmos 3](https://arxiv.org/abs/2606.02800) | Announced at GTC, launched May 31; a two-tower mixture-of-transformers: an autoregressive VLM reasoner plus a diffusion generator over text, image, video, audio and actions |
| 2026-07-15 | Cosmos 3 Edge | 4B, based on Nemotron |

NVIDIA's own two statements of Cosmos 3's size disagree: the [developer blog](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3) says Nano 16B and Super 64B, while the [Hugging Face collection](https://huggingface.co/collections/nvidia/cosmos3) lists Super as 65B and Edge as 4B. NVIDIA's 2026-05-31 [press release](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai) claims first place among open models on several leaderboards including Physics-IQ and PAI-Bench, which is vendor-reported; on the third-party [Physics-IQ Verified leaderboard](https://github.com/google-deepmind/physics-IQ-benchmark), read on 2026-10-15, Cosmos3-Super (multiframe) scores 50.8 ± 2.2.

Another cost data point: in the [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) (June 2025), Meta placed Cosmos as the world model inside the same CEM planner, and it needed 4 minutes per action with 0% pick-and-place success, against 16 seconds per action for V-JEPA 2-AC. This is Meta's measurement and only shows that the Cosmos used in that experiment did not suit the core of such a planner; it says nothing against Cosmos as a data engine. Details are in the [JEPA note](jepa.en.md).

## Job two: evaluating policies

Estimating a new policy's performance without deploying it is called off-policy evaluation in RL; see [RL from a fixed dataset](../../05-post-training/deep-rl/offline-and-ope.en.md). World models offer another route: let the policy act inside the model and count successes. Below are several works that use a world model as an evaluator; where numbers exist, each was measured by its authors on policies and tasks they chose:

| System (organisation, date) | Reported agreement with reality | Read together with |
| --- | --- | --- |
| [WorldEval](https://arxiv.org/abs/2505.19017) (Midea / ECNU, May 2025) | r = 0.942 with real results | Policies and tasks chosen by the authors |
| [WorldGym](https://arxiv.org/abs/2506.00613) (Stanford, NYU, GDM, June 2025) | Per-task r = 0.78 | Same |
| [1X World Model](https://www.1x.tech/discover/redwood-ai-world-model) (1X, 2025-06-16) | Action-conditioned video plus a value model used to rank humanoid policies | No correlation published |
| [Ctrl-World](https://arxiv.org/abs/2510.10125) (Stanford, Tsinghua, October 2025) | Multi-view robot world model for evaluating and improving policies | We found no specific correlation figure |
| [Veo evaluator](https://arxiv.org/abs/2512.10675) (Google DeepMind, December 2025, v2 January 2026) | Pearson 0.86, MMRV 0.06 in the OOD study | See below |
| [dWorldEval](https://arxiv.org/abs/2604.22152) (Li et al., April 2026) | r ≈ 0.91–0.93 | A discrete-diffusion world model |
| [PiL-World](https://arxiv.org/abs/2606.05773) (June 2026) | Success-rate estimation error from 63.2% down to 12.0% | Only 3 dual-arm tasks |

We read the Veo evaluator paper itself. It fine-tunes the Veo 2 text-to-video model on robotics data and evaluates policies on the ALOHA 2 bimanual platform (4 cameras, §2, §3.1); the abstract reports more than 1,600 real-world evaluations covering 8 checkpoints and 5 tasks. The Pearson 0.86 and MMRV 0.06 refer to the single strongest policy, comparing predicted and real success rates across the OOD generalisation axes of background, small distractor, large distractor and object (§4.1); the text does not state the number of points, so count them in Fig. 8. Limitations the paper states: predicted absolute success rates are lower than real ones (§3.2, §4.1); contact-rich interactions, especially with small objects, are hard to simulate (§7); each rollout is 8 seconds, and minute-scale multi-view consistent generation remains a goal (§7); success is currently judged by humans (§7).

<!-- paper-figure:veo-eval-ood -->

The Veo evaluator paper defines MMRV in §3.2 as follows (the metric comes from Li et al. 2024):

$$
\text{MMRV}=\frac1n\sum_{i=1}^n\max_j\;|R_i-R_j|\cdot\mathbb 1\big[(\hat R_i<\hat R_j)\neq(R_i<R_j)\big]
$$

$R_i$ is policy $i$'s real success rate and $\hat R_i$ the world model's predicted rate. Each policy records only "the opponent it is misordered with that has the largest real gap", then the result is averaged over policies. Zero means the orderings agree completely.

### Compute it once: correlation 0.98, and the top pick is still wrong (a teaching example)

Five cup-pushing policies, with invented numbers:

| Policy | Real success $R$ | Model prediction $\hat R$ | Real rank | Model rank |
| --- | --- | --- | --- | --- |
| A | 0.10 | 0.05 | 5 | 5 |
| B | 0.25 | 0.20 | 4 | 4 |
| C | 0.45 | 0.30 | 3 | 3 |
| D | 0.70 | 0.55 | 2 | 1 |
| E | 0.80 | 0.50 | 1 | 2 |

- **Pearson**: $\bar R=0.46$, $\bar{\hat R}=0.32$; the cross term $\sum(R-\bar R)(\hat R-\bar{\hat R})=0.239$ and the two sums of squares are 0.347 and 0.173, so $r=0.239/\sqrt{0.347\times0.173}\approx0.98$.
- **Spearman**: only D and E swap ranks, the sum of squared rank differences is $\sum d^2=2$, and $\rho=1-6\sum d^2/(n(n^2-1))=1-12/120=0.9$.
- **MMRV**: only the pair (D, E) is misordered, with a real gap of 0.10; D and E each record 0.10 and the rest record 0, so $\text{MMRV}=0.2/5=0.04$.

All three metrics look good, yet the best policy by the model is D, while on the real robot E is 10 percentage points higher. The correlation is held up mainly by the large gap between weak policies like A and B and the good ones. Look only at the three you actually choose among, C, D and E, and Spearman becomes $1-6\times2/(3\times8)=0.5$.

<!-- widget:step-evaluator-rank -->

Two more points:

- **Absolute values.** Pearson is unchanged under $\hat R=aR+b$ with $a>0$: a model that reports every success rate at half its true value has $r=1$. The model in the table also underestimates across the board, just as the Veo evaluator paper says of itself. If you need the evaluator to decide whether a policy clears a 70% launch bar, correlation does not help.
- **The real side is noisy too.** With only 50 trials per policy, the standard error of the difference between D and E is about $\sqrt{0.7\times0.3/50+0.8\times0.2/50}\approx0.086$. A gap of 0.10 is only about 1.2 standard errors, so even the real ranking is shaky. For comparing two policies, see [what a score increase actually tells you](../../07-evaluation/metric-robustness.en.md); for what agreement, correlation and calibration each measure, see the [judge calibration note](../../07-evaluation/llm-as-a-judge/calibration.en.md).

| A high correlation can show | It cannot show |
| --- | --- |
| Across these policies, model scores broadly rise and fall with real success | That the best policy is ranked first |
| Using it to screen out clearly weak checkpoints is fairly low-risk | Which of two close checkpoints is better |
| It is usable on the tasks and scenes it was validated on | That it still holds for new tasks, robots or policy families |
| The linear relation is strong | That absolute success rates are right and can decide whether a bar is met |

### The policy under evaluation will exploit the evaluator

The [model-based RL note](../../05-post-training/deep-rl/model-based.en.md) showed that the harder you plan, the more likely you are to find the model's flaws. Evaluation is the same. Pick the checkpoint with the highest world-model score out of 20, and the winner tends to be the one the model overestimates most; go further and use the world-model score as a training reward, and the policy will actively seek actions that the model judges successful but that fail on the real robot. On 2026-01-12, 1X itself wrote that its generated videos are over-optimistic about task completion. When validating an evaluator, keep at least one set of policies that never took part in selection, and spot-check on real robots regularly.

## Job three: acting as the policy (world action models)

The idea is to first "imagine" inside the model what the next few seconds should look like, then read actions off the imagined frames; or to have the model generate video and actions together.

```text
Route A (1X NEO): current view + task → 14B video model generates future video → inverse dynamics model reads off actions
Route B (DreamZero, Cosmos Policy): current view + task → one model jointly generates future frames and actions
```

- **DreamZero** (NVIDIA et al., 2026-02-17 [paper](https://arxiv.org/abs/2602.15922)): a 14B autoregressive video-diffusion "World Action Model" running closed-loop control at 7 Hz. The abstract reports more than 2× the generalisation of state-of-the-art VLAs to new tasks and environments on real robots; 10–20 minutes of video-only demonstrations from other robots or humans improve unseen-task performance by over 42% (relative); adapting to a new embodiment takes 30 minutes of play data. On 2026-03-16 NVIDIA said GR00T N2 builds on DreamZero research (vendor-reported).
- **Cosmos Policy** (NVIDIA and Stanford, 2026-01-22 [paper](https://arxiv.org/abs/2601.16163)): post-trains Cosmos-Predict2-2B into a policy, encoding actions, future states and values as latent frames. In simulation it averages 98.5% on LIBERO and 67.1% on RoboCasa; on a real ALOHA its average score is 93.6, above π0.5's 88.6; out of distribution, π0.5 is better (92.5 vs 89.3).
- **1X world model as NEO's policy** (1X, 2026-01-12 [blog](https://www.1x.tech/discover/world-model-self-learning)): a 14B video backbone mid-trained on 900 hours of egocentric human video and fine-tuned on 70 hours of NEO data; generating 5 seconds of video takes 11 seconds, plus about 1 second for the IDM. Pull-tissue success is 30% with one generation and 45% with 8 parallel generations. Limitations 1X lists itself: generated videos are over-optimistic about task completion, 3D grounding is weak, and evaluation was not blind.
- **Runway Praxis-1** (2026-09-30 [announcement](https://runway.com/research/introducing-praxis-1)): announced as an open-weight world action model; as of 2026-10-15 we could check only the announcement.

Now the timing: 7 Hz means about $1000/7\approx143$ ms per control step. 1X needs 11 seconds to generate 5 seconds of future, $11/5=2.2$ times slower than real time, and about 12 seconds including the IDM before a stretch of actions is ready. Our inference: this suits slow tasks that can wait, not scenes that need an immediate reaction. The Cosmos Policy comparison is a reminder that "video models as policies generalise better" depends on conditions: it wins in distribution and loses out of distribution. Separately, Physical Intelligence mentioned on 2026-04-16, introducing [π0.7](https://www.pi.website/blog/pi07), that it uses a "lightweight world model" to generate visual subgoals, without details.

## Driving: GAIA and the Waymo World Model

In driving, first separate open loop from closed loop. Open loop replays recorded sensor data and looks at what the policy would do at each frame; once the policy drives differently from the recording, the following frames no longer match. Closed loop lets the policy's actions drive the world model's next sensor frames, and the policy decides again from the new view; only then can you see what happens after a deviation.

| Date | System (organisation) | What was made public |
| --- | --- | --- |
| 2023-09 | [GAIA-1](https://arxiv.org/abs/2309.17080) (Wayve) | Video, text and action tokens; 6.5B world model plus 2.6B decoder; 4,700 hours of London driving |
| 2025-03 | [GAIA-2](https://arxiv.org/abs/2503.20523) (Wayve) | 8.4B latent diffusion, multi-camera, controllable |
| 2025-12-02 | [GAIA-3](https://wayve.ai/thinking/gaia-3/) (Wayve) | 15B latent diffusion, framed for evaluating end-to-end driving systems |
| 2026-02-06 | [Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation) (Waymo) | Genie 3 post-trained to emit camera and lidar; controllable by driving actions, scene layout and language; can turn dashcam or phone video into simulation; no fidelity metrics published |
| 2026-06-10 | [Oasis 3](https://decart.ai/publications/introducing-oasis-3-first-interactive-world-model-for-physical-ai) (Decart) | Driving and robot world model with three synchronised cameras, 512×768 at 22 FPS, latency under 200 ms (vendor-reported) |
| 2026-08-03 | [GAIA-4](https://wayve.ai/thinking/gaia-4/) (Wayve) | Closed loop: the AI Driver's actions drive the next sensor frames; adds radar output (Wayve calls it an industry first); the only number is "2.5× better reconstruction faithfulness" (vendor-reported); no paper or parameter count |

As of 2026-10-15, we found no published correlation with real outcomes for GAIA-3/4, the Waymo World Model or the 1X evaluator. Tesla described its own "neural world simulator" for closed-loop evaluation in talks at ICCV 2025 and CVPR 2026 ([talk video](https://www.youtube.com/watch?v=2V0ddkF1Svo); content taken from the video's auto-generated captions), but has published no paper and no comparison with real outcomes; see the [end-to-end note in the driving series](../../13-embodied/driving/end-to-end.en.md).

## Risk checklist

| Risk | What goes wrong | Example | How to check |
| --- | --- | --- | --- |
| Sim-to-real gap | Realistic visuals, but contact, friction and small objects are wrong | The Veo evaluator says small-object contact is hard to simulate; WorldArena's perception–functionality gap | Compare with real robots grouped by failure type, not just overall correlation |
| Evaluator exploitation | The policy learns actions the model judges successful but that fail for real | 1X says its generated videos are over-optimistic; Ha & Schmidhuber's adversarial VizDoom policy | Hold out policies that never took part in selection; spot-check on real robots regularly |
| Incomplete coverage of failure modes | Failures rare in training data cannot be acted out by the model | An 8-second rollout misses slower failures, such as a cup creeping off the edge | List real failure types and check one by one whether the model reproduces them |
| Wrong absolute values | Ranking right, threshold decisions wrong | The Veo evaluator's predicted success rates are low across the board | Calibrate separately, or use it only for ranking |
| Compute and latency | Generation is slower or costlier than real trials | DreamGen used 1,500 L40 GPUs for 54 hours; 1X needs 11 seconds for 5 seconds of video; Cosmos took 4 minutes per action in planning | Put generation cost and real-trial cost in the same ledger |
| Opacity | Outsiders cannot check independently | Waymo and GAIA-4 published no fidelity metrics or architecture | Treat only publicly validated uses as conclusions |

## What to watch on this line next

1. **Do evaluators hold across robots, tasks and policy families?** The Veo evaluator (Google DeepMind, December 2025) was validated on one platform (ALOHA 2) with 8 checkpoints; WorldEval (May 2025) and dWorldEval (April 2026) each on settings chosen by their authors. A result that would change the picture: correlation, MMRV and "was the top policy picked correctly" reported together across several embodiments and held-out policy families.
2. **Will driving world models publish fidelity and closed-loop validation?** Neither Waymo (February 2026) nor Wayve's GAIA-4 (August 2026) published a correlation with real outcomes. A result that would change the picture: public comparisons of crash or takeover rates in simulation with real-road data.
3. **When do world action models beat VLAs, and when not?** Cosmos Policy (January 2026) wins in distribution and loses to π0.5 out of distribution; DreamZero (February 2026) reports more than 2× generalisation; Runway announced open weights for Praxis-1 on 2026-09-30. A result that would change the picture: third-party comparisons on the same real-robot tasks with the same amount of data.
4. **Can latency and cost drop to closed-loop use?** DreamZero self-reports 7 Hz; 1X self-reports generation 2.2 times slower than real time. A result that would change the picture: real-time operation on stated hardware without a drop in success rate.

Next, read [do they actually understand the world?](do-they-understand.en.md); to revisit how world models are used for planning, go back to [model-based RL](../../05-post-training/deep-rl/model-based.en.md).
