# Robot learning: vision-language-action models and sim-to-real

[中文](vla.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

You have a vision-language model (VLM) that answers questions about images, and you want it to control a robot arm directly: two camera views and "put the cup in the sink" go in, joint commands at 50 Hz come out. The most direct approach is to write actions as text tokens too, but the latency estimate below shows why generating those tokens one at a time cannot keep up with the control rate.

As of October 2026, a common design uses the VLM to interpret the scene and instruction, then a flow-matching "action expert" to generate a whole chunk of continuous actions at once. Humanoid robots also separate slower reasoning from faster control across two or three layers. Randomization and online adaptation help policies trained in simulation transfer to real robots; evaluation is likewise moving from near-saturated simulation leaderboards to real-robot comparisons.

Read first: [Learning actions from demonstrations](imitation.en.md) (multimodal actions, action chunking) and [LLaVA and DeepSeek-VL](../../03-multimodal-learning/vlm-designs.en.md) (how images enter a language model).

## How actions are written: four designs

The term VLA (vision-language-action) comes from [RT-2](https://arxiv.org/abs/2307.15818v1) (Google DeepMind, 2023-07). To tell VLAs apart, first ask where the action comes out of the model and in what form.

**1. Actions as text tokens.** RT-2's action is 8 numbers: the end effector's 6-DoF translation and rotation, gripper extension, and a "terminate" flag; the continuous dimensions are split uniformly into 256 bins, the whole action is written as a string of integer tokens, and the model is co-fine-tuned together with web visual question-answering data (§3.2). The 55B version runs on a multi-TPU cloud service at 1–3 Hz, the 5B version at about 5 Hz (§3.3). [OpenVLA](https://arxiv.org/abs/2406.09246v1) (Stanford-led, 2024-06) follows the same idea: Llama 2 7B with DINOv2 and SigLIP visual features; 7-D actions, each dimension split into 256 bins between the 1st and 99th percentile of the training data, taking over the 256 least-used tokens in the Llama vocabulary; about 6 Hz on one RTX 4090 (without compilation or speculative decoding).

<!-- paper-figure:openvla-arch -->

**2. Compress first, then tokenize.** The observation behind [FAST](https://arxiv.org/abs/2501.09747v1) (Physical Intelligence, UC Berkeley, Stanford, 2025-01): at high control rates consecutive steps are nearly identical, so most per-step, per-dimension tokens can be predicted by "copying the previous one", and the next-token learning signal becomes weak: "this marginal information approaches zero as the control frequency of the training signal increases …" ([§IV](https://arxiv.org/html/2501.09747v1#S4)). FAST applies a discrete cosine transform (DCT) to each dimension, quantizes, and compresses with BPE. Tokens per 1-second action chunk as reported by the authors: table bussing (7-D, 20 Hz) drops from 140 to 28; T-shirt folding (14-D, 50 Hz) from 700 to 53 (Table I). The authors report that π0-FAST matches the diffusion version of π0 with a training speedup of up to 5x.

**3. A continuous action expert.** [π0](https://arxiv.org/abs/2410.24164v1) (2024-10) combines PaliGemma 3B with a flow-matching action expert of about 300M parameters, using 50-step chunks at up to 50 Hz. Inference takes 10 integration steps: it computes the prefix of images, text and robot state once, caches its keys and values, and then recomputes only the action part at each step (§IV). [π0.5](https://arxiv.org/abs/2504.16054v1) (2025-04) first predicts the next subtask as text, then the 300M expert outputs a 50-step chunk; [π0.7](https://arxiv.org/abs/2604.15483v1) (2026-04) moves to Gemma 3 4B with an 860M expert and adds metadata such as quality and speed, plus subgoal images, to the prompt.

NVIDIA's [GR00T N1](https://arxiv.org/abs/2503.14734v1) (2025-03) trains a VLM (System 2) and a diffusion-transformer action module (System 1) jointly end to end. Hugging Face's [SmolVLA](https://arxiv.org/abs/2506.01844v1) (2025-06) has about 450M parameters in total, about 100M of them in the flow-matching expert.

Attaching a randomly initialized continuous action expert to a VLM can, however, disrupt training. PI's [knowledge insulation](https://arxiv.org/abs/2505.23705v1) (2025-05) reports substantially slower training and worse knowledge transfer. The paper (§1, §5.2) attributes this to gradients from the expert damaging the VLM backbone, and prevents those gradients from updating the backbone.

<!-- paper-figure:pi0-overview -->

**4. Neither tokens nor diffusion.** [OpenVLA-OFT](https://arxiv.org/abs/2502.19645v1) (Stanford, 2025-02) changes OpenVLA to parallel decoding (bidirectional attention, a whole chunk in one forward pass), continuous actions and L1 regression. The authors report the LIBERO average rising from 76.5% to 97.1% (the best configuration uses extra inputs) and action-generation throughput rising 26-fold (measured with 8-step chunks).

| Design | Examples | Sequential forward passes for one 50-step × 7-D chunk | How multiple modes are represented |
| --- | --- | --- | --- |
| Per-dimension bin tokens | RT-2, OpenVLA | 350 (OFT paper: chunk length $K$ and dimension $D$ need $KD$) | Softmax over 256 bins per token |
| Compressed tokens | π0-FAST | Typically 30–60 according to the authors | Softmax over the BPE vocabulary per token (1,024 entries in the single-dataset experiments) |
| Flow-matching expert | π0, π0.5, π0.7, GR00T, SmolVLA | 1 prefix pass + 10 expert passes | Noise sampling, see the previous note |
| Parallel decoding + regression | OpenVLA-OFT | 1 | Not represented; outputs a point estimate |

## Worked example: how long one chunk takes (teaching numbers)

The per-step costs below are **teaching assumptions** chosen so the orders of magnitude can be computed by hand; they are not measurements of any model:

- Prefix: images, instruction and robot state through the VLM once, $T_{\rm prefix}=40$ ms, the same for all three designs;
- Autoregressive decoding: each token from a backbone of about 3B parameters, $c_{\rm tok}=10$ ms;
- One flow step: each integration step of an action expert of about 300M parameters (prefix keys and values cached), $c_{\rm flow}=3$ ms;
- Action chunk: 50 steps × 7 dimensions, exactly 1 second of motion at 50 Hz; after FAST compression, assume 40 tokens (the authors say typically 30–60).

$$
T_{\rm AR}=T_{\rm prefix}+N_{\rm tok}\,c_{\rm tok},\qquad
T_{\rm flow}=T_{\rm prefix}+K\,c_{\rm flow}
$$

$N_{\rm tok}$ is the number of tokens generated sequentially and $K$ the number of flow integration steps. Plugging in:

| Design | Sequential steps | Time per chunk | Against 1 second of motion |
| --- | --- | --- | --- |
| Per-dimension bin tokens | $N_{\rm tok}=50\times7=350$ | $40+350\times10=3{,}540$ ms | Generation is 3.5 times slower than execution; cannot keep up |
| FAST (assume 40 tokens) | $N_{\rm tok}=40$ | $40+40\times10=440$ ms | Keeps up on average, but each chunk takes almost half a second |
| Flow-matching expert | $K=10$ | $40+10\times3=70$ ms | Plenty of margin |

Per-dimension tokens are about $3540/70\approx51$ times slower than flow, FAST about $440/70\approx6$ times. If per-dimension tokens skip chunking and output one step at a time, it takes $40+7\times10=110$ ms, at most about 9 Hz, the same single-digit-Hz range that RT-2 (1–3 Hz) and OpenVLA (about 6 Hz) measured.

Now compare with real measurements. The π0 paper ([v1 Appendix D, Table I](https://arxiv.org/abs/2410.24164v1)) measured on an RTX 4090 with 3 cameras: image encoders 14 ms, prefix forward pass 32 ms, 10 flow steps 27 ms in total, 73 ms on board; for a mobile robot running inference off board over WiFi, network latency adds 13 ms. The FAST paper (§VI-E) reports that, also on an RTX 4090, the diffusion π0 produces a 1-second chunk in under 100 ms while π0-FAST needs about 750 ms per chunk, because it decodes 30–60 tokens and each one goes through the full 2B-parameter backbone rather than a 300M expert. The authors also say this latency did not hurt performance on the static manipulation tasks they tested.

<!-- widget:step-vla-latency -->

What this example does **not** tell you:

- Sequential steps and policy quality are separate questions: π0-FAST takes far more sequential steps, and its authors report that it matches π0 in success rate.
- Per-step cost varies widely with GPU, quantization, batch and implementation; comparisons must state hardware and configuration.
- In the flow design the prefix dominates: $40/70\approx57\%$ in the teaching numbers, and image encoders plus prefix are $46/73\approx63\%$ in π0's measurement. To go faster, shrink the prefix first.
- Real systems overlap inference with execution: on 50 Hz robots π0 infers the next chunk after executing 25 steps (0.5 s); SmolVLA's asynchronous inference cuts task completion time from 13.75 s to 9.7 s ([paper §4.6](https://arxiv.org/abs/2506.01844v1), author-reported).

## Slow and fast layers: why humanoids split into two or three

Balancing and grasping need feedback within milliseconds, while the meaning of "put the cup in the sink" changes only every few seconds. A 200 Hz loop needs a control update every 5 ms. Whether a model with billions of parameters can handle that loop on its own depends on onboard hardware, image and state inputs, numerical precision and inference implementation, not parameter count alone. In Helix's published dual-embedded-GPU configuration, shown below, the 7B vision-language layer updates at 7–9 Hz while the fast layer runs at 200 Hz. This split lets the slow layer interpret the scene and instruction while the fast layer combines that information with the latest observation to produce joint commands.

| System (date) | Slow layer | Fast layer(s) | Notes |
| --- | --- | --- | --- |
| Figure [Helix](https://www.figure.ai/news/helix) (2025-02) | S2: built on an open-source, open-weight 7B VLM, at 7–9 Hz | S1: an 80M-parameter transformer at 200 Hz controlling a 35-DoF upper body | Two low-power embedded GPUs on board; about 500 h of teleoperation data (self-reported); Helix itself is not open |
| Figure [Helix 02](https://www.figure.ai/news/helix-02) (2026-01) | S2 | S1 outputs full-body joint targets (200 Hz); new S0: a 10M-parameter whole-body controller at 1 kHz, trained in simulation on 1,000+ h of retargeted human motion | A 4-minute autonomous dishwasher unload and reload; S0 replaces 109,504 lines of C++ (self-reported) |
| NVIDIA [GR00T N1](https://arxiv.org/abs/2503.14734v1) (2025-03) | VLM (System 2) | Diffusion-transformer action module (System 1) | The two parts are trained jointly; demonstrated on Fourier GR-1 |
| Google DeepMind [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) (2026-07-30) | ER 2: plans tasks lasting several minutes; its [model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/) says it is built on Gemini 3.5 Flash | The same VLA checkpoint runs whole-body control on Apptronik Apollo 2 and also controls the two-arm Franka Duo | The 11 success rates in the blog range from 32% to 92%, e.g. unscrewing a bulb 92%, using a dustpan 32% (self-reported); DeepMind itself says speed needs work and multi-finger dexterity is still hard |
| [π0.5](https://arxiv.org/abs/2504.16054v1) (2025-04) | The same model first predicts a subtask as text | Then the action expert outputs a 50-step chunk | Two stages inside one model, not two models |

The rhythm between layers: take S2 at 8 Hz, updating every 125 ms; meanwhile S1 at 200 Hz runs $125/5=25$ times, and S0 at 1 kHz runs 5 times within each S1 step, 125 times in total. The cost: when the slow layer misreads the scene, the error persists for a whole slow cycle, and what passes between layers (a latent vector, a text subtask or joint targets) is rarely described in detail.

## RL on top of VLAs: learning what demonstrations lack

Pure behavior cloning fits demonstrated actions; its objective does not directly reward completing the task faster or more reliably than the demonstrations. The data also shapes the learned timing and state coverage. That does not make the demonstrator's skill or speed an unbreakable mathematical ceiling: action-fitting error and task return are different measures, so the training objective alone establishes no such performance bound. Demonstrations also rarely show how to carry on after a slip or a jam. In 2025–26 Physical Intelligence began running RL on VLAs with the robot's own experience (all self-reported):

- **[π\*0.6 / Recap](https://www.pi.website/blog/pistar06) (2025-11-17).** A value function is trained first and used to estimate the advantage of each piece of experience (how much better than average a step was, see [Actor–Critic and GAE](../../05-post-training/deep-rl/actor-critic-gae.en.md)); π0.6 then learns conditioned on that signal, with an offline-RL pretraining stage (see [Offline RL and OPE](../../05-post-training/deep-rl/offline-and-ope.en.md)). The data includes demonstrations, the robot's own experience and corrections from teleoperators. PI reports more than double the throughput on some of the hardest tasks and failure rates cut by half or more, and shows the robot making espresso from 5:30 am to 11:30 pm.
- **[RL tokens](https://www.pi.website/research/rlt) (2026-03-19).** The VLA is first trained to output an "RL token" and then frozen; a small actor-critic trained online on that token edits the actions the VLA proposes. PI reports the most precise phases running up to 3x faster with as little as 15 minutes of real-robot data; it tested 4 precision tasks (an M3 screw, a zip tie, an Ethernet plug and a power cord), with RL targeting their critical phases.
- **[π0.7](https://www.pi.website/blog/pi07) (2026-04-16).** PI says this generalist model matches the RL-trained π\*0.6 specialists on laundry folding, espresso and box folding; the conclusion of the [paper](https://www.pi.website/download/pi07.pdf) also acknowledges that unseen tasks succeed only 60%–80% of the time.

For legged or whole-body controllers trained in simulation, common RL algorithms are PPO and SAC, see [How much should a policy change?](../../05-post-training/deep-rl/trust-region.en.md) and [SAC](../../05-post-training/deep-rl/soft-actor-critic.en.md); Helix 02's S0 was also trained in simulation.

## Evaluation: after scores reach 97%

- **[LIBERO](https://arxiv.org/abs/2306.03310v1) (UT Austin and others, 2023-06)**: 130 simulated manipulation tasks in total. VLA papers usually evaluate on 4 suites (Spatial, Object, Goal, Long) of 10 tasks each; OpenVLA-OFT's 97.1% is the average over these 40 tasks. With scores this high, strong methods are hard to separate; "LIBERO is near its ceiling" is our judgement, not a conclusion of any paper.
- **[SIMPLER](https://arxiv.org/abs/2405.05941v1) (2024-05)**: recreates real tabletops and robots in simulation to evaluate policies trained on real-robot data; the authors report strong correlation between simulated and real results.
- **[RoboArena](https://arxiv.org/abs/2506.18123v1) (2025-06)**: distributed, double-blind pairwise real-robot comparisons on the DROID platform, with 7 academic institutions, 600+ pairwise episodes and 7 policies.

Why the top of the scale is hard to compare, by hand. OFT's protocol runs 500 trials per suite (10 tasks × 50), 2,000 over the 4 suites. At a success rate of 0.97 the standard error is $\sqrt{0.97\times0.03/2000}\approx0.0038$, so a 95% interval is about ±0.75 percentage points; for two methods 0.5 points apart, the standard error of the difference is about $\sqrt2\times0.0038\approx0.0054$, so the gap is less than one standard error. (This treats the 2,000 trials as independent; if trials within a task are correlated, the interval is wider still.) For how to compare two scores, see [The score improved. What does that actually establish?](../../07-evaluation/metric-robustness.en.md).

Using world models instead of real robots as evaluators is a separate route, covered in [world models as evaluators for robots and cars](../../11-frontier/world-models/robotics-driving.en.md).

## Sim-to-real: where the gap comes from and how to close it

Simulation is cheap, parallel and safe to crash in, but friction, mass, latency and images all differ from the real robot. A hand-computable example shows three ways to close the gap.

Teaching example: the arm pushes a puck, which slides $d=v^2/(2\mu g)$ from initial speed $v$ before stopping; the target is $d^\ast=0.5$ m and $g=9.8\ \text{m/s}^2$. The real table's friction coefficient $\mu$ lies between 0.3 and 0.7 and is not known in advance.

- **Tune in simulation at $\mu=0.5$ only.** $v^2=2\times0.5\times9.8\times0.5=4.9$. On the real table: at $\mu=0.3$ the puck slides $4.9/5.88\approx0.83$ m, 0.33 m too far; at $\mu=0.7$ only $4.9/13.72\approx0.36$ m.
- **Domain randomization.** During training $\mu$ is drawn at random between 0.3 and 0.7; the policy does not know this episode's value and can only pick a $v$ that is acceptable everywhere. If the goal is the smallest worst-case error, make the errors at both ends equal: $\frac{v^2}{2g}\left(\frac1{0.3}+\frac1{0.7}\right)=1$, so $v^2\approx4.12$; the puck slides 0.70 m and 0.30 m at the two ends, an error of 0.20 m either way. More robust, but rarely on target.
- **Online adaptation (the RMA idea).** First nudge the puck and watch how fast it decelerates (the deceleration equals $\mu g$), estimate $\mu$, then push. A measured deceleration of $2.94\ \text{m/s}^2$ gives $\hat\mu=0.3$; take $v^2=2\times0.3\times9.8\times0.5=2.94$ and the puck slides exactly 0.5 m; the remaining error comes only from the estimate.

The corresponding papers:

- [Tobin et al.](https://arxiv.org/abs/1703.06907v1) (OpenAI, UC Berkeley, 2017-03) randomize rendering so that the real world looks like just one more variation; an object detector trained only on simulated images localizes to 1.5 cm in the real world (authors). For dynamics randomization, see [Peng et al. (2017-10)](https://arxiv.org/abs/1710.06537).
- [OpenAI's Rubik's Cube work](https://arxiv.org/abs/1910.07113v1) (2019-10) uses automatic domain randomization (ADR): once the policy does well enough on a range, the randomization range widens; trained only in simulation, the robot hand completed Rubik's Cube solves (the move sequence comes from the Kociemba solver; what was learned is the in-hand manipulation).
- [RMA](https://arxiv.org/abs/2107.04034v1) (UC Berkeley, CMU, Facebook AI, 2021-07): a base policy plus an adaptation module that estimates, online and from recent state and action history, a latent vector encoding the environment's conditions; trained only in simulation, the authors report the A1 quadruped walking on sand, stairs, grass and other terrain without fine-tuning. Another classic sim-to-real paper for legged robots is [Hwangbo et al. (2019-01)](https://arxiv.org/abs/1901.08652).

As of October 2026, simulators are converging on GPU parallelism:

| Date | System | What it is | Notes |
| --- | --- | --- | --- |
| 2025-02 | [MuJoCo Playground](https://arxiv.org/abs/2502.08844v1) (UC Berkeley, Google DeepMind and others) | MJX-based environment suite with batch rendering | Authors report training in minutes on one GPU and zero-shot transfer to real robots from both state and pixel inputs |
| 2025-09-29 | [Newton](https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning) (Disney Research, Google DeepMind, NVIDIA, contributed to the Linux Foundation) | GPU physics engine built on NVIDIA Warp and OpenUSD, with multiple solvers | Version 1.0 was published to [PyPI](https://pypi.org/project/newton/) on 2026-03-10 |
| 2025-11 | [Isaac Lab](https://arxiv.org/abs/2511.04831v1) (NVIDIA) | Successor to Isaac Gym: GPU physics, rendering, actuator and sensor models, randomization tools | Per NVIDIA's 2026-03-16 [press release](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world), Isaac Lab 3.0 (early access) is built on Newton 1.0 plus PhysX |
| ongoing | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | MuJoCo rewritten in Warp by Google DeepMind and NVIDIA, part of the Newton project | Apache-2.0; the [docs](https://mujoco.readthedocs.io/en/latest/mjwarp/) say it is not currently differentiable (MJX is) |

## Which models can you download? (as of 2026-10-15)

| Model | Weights | Licence and conditions | Source |
| --- | --- | --- | --- |
| OpenVLA | Open | Code MIT; the README says the models fall under the Llama 2 Community License, while the Hugging Face model card is tagged MIT | [GitHub](https://github.com/openvla/openvla) |
| Octo | Open | MIT on the GitHub repo and Hugging Face (the project page states none) | [Project page](https://octo-models.github.io/) |
| π0, π0-FAST, π0.5 (openpi) | Open | Code Apache-2.0; a copy of the Gemma Terms of Use (LICENSE_GEMMA.txt) is included, and the repo does not say which files it covers | [GitHub](https://github.com/Physical-Intelligence/openpi) |
| SmolVLA | Open (code, weights, data) | Model card tagged Apache-2.0 | [Blog](https://huggingface.co/blog/smolvla) |
| GR00T N1.5, N1.6 | Open | Non-commercial (NVIDIA OneWay Noncommercial License, per the model cards and the developer page) | [N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/), [developer page](https://developer.nvidia.com/isaac/gr00t/llm-info.md) |
| GR00T N1.7 | Open (early access from 2026-04-17) | Sources conflict: the release blog calls it commercially licensed and the model card names the NVIDIA Open Model License, but the LICENSE file in the repo is identical to N1.6's non-commercial licence | [Blog](https://huggingface.co/blog/nvidia/gr00t-n1-7), [developer page](https://developer.nvidia.com/isaac/gr00t/llm-info.md) |
| π0.6, π\*0.6, π0.7 | Not stated | PI opened a hosted API beta on 2026-10-13: waitlist, no pricing, for an unnamed newest model | [PI blog](https://www.pi.website/blog/api-beta) |
| Gemini Robotics 2 VLA, On-Device 2; Figure Helix, Helix 02 | Not open | The Gemini VLA goes only to early-access partners, while ER 2 is available in Google AI Studio; Helix is not offered externally | [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/), [Figure](https://www.figure.ai/news/helix) |
| GR00T N2 | Not released | Previewed on 2026-03-16 as coming "by the end of the year"; no release found as of 2026-10-15 | [NVIDIA press release](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) |
| LeRobot (library) | Open | Apache-2.0; includes ACT, Diffusion Policy, π0 / π0.5, SmolVLA, GR00T and other policies | [GitHub](https://github.com/huggingface/lerobot) |

GR00T N1.6's backbone is also disputed: NVIDIA's research page names an internal Cosmos-2B variant, while the GitHub repo and the developer page name Eagle.

## Humanoid robots: what the companies have said (as of 2026-10-15)

For Tesla Optimus, only Tesla's own quarterly update letters (the Exhibit 99.1 copies filed with the SEC):

| Letter (date) | What Tesla wrote |
| --- | --- |
| [Q3 2025](https://www.sec.gov/Archives/edgar/data/0001318605/000162828025045861/exhibit991.htm) (2025-10-22) | First-generation Optimus lines are being installed in anticipation of "volume production" |
| [Q4 / FY 2025](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026003837/exhibit991.htm) (2026-01-28) | Plans to unveil Gen 3 in Q1 2026, described as the first design meant for mass production; start of production planned before the end of 2026, eventual capacity 1 million robots a year |
| [Q1 2026](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026026551/exhibit991.htm) (2026-04-22) | The first-generation line in Fremont, designed for 1 million robots a year, replaces the Model S / X lines; Gigafactory Texas is being prepared for a second-generation line designed for a long-term 10 million a year |
| [Q2 2026](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026049213/exhibit991.htm) (2026-07-22) | The S / X lines have been decommissioned and first-generation Optimus lines are being installed, with production expected to start soon; initial builds go to the "Optimus Academy" to collect training data; the outlook wording changes from "volume production" to "production in 2026" |

As of 2026-10-15, no official Gen 3 / V3 unveil appears in Tesla's primary sources, and the Q3 2026 letter is not yet out. Claims about a V3 unveil and weekly production rates appear only in press reports and are not used here.

Other companies (self-reported unless noted):

- **Figure**: BMW's [press release](https://www.press.bmwgroup.com/global/article/detail/T0455864EN/bmw-group-to-deploy-humanoid-robots-in-production-in-germany-for-the-first-time?language=en) of 2026-02-27 confirms that Figure 02 helped build more than 30,000 X3s at the Spartanburg plant (the release does not say how many robots); Figure said on 2026-04-29 that it had delivered 350+ Figure 03 robots ([Figure blog](https://www.figure.ai/news/ramping-figure-03-production)), allocated to internal R&D, data collection and commercial use-case development, without naming customers or giving a field deployment count.
- **Agility**: Digit moved more than 100,000 totes in a GXO warehouse ([2025-11-20](https://www.agilityrobotics.com/content/digit-moves-over-100k-totes)).
- **Boston Dynamics**: unveiled the product version of the electric Atlas on 2026-01-05 and said all 2026 deployments are already committed, going to Hyundai and Google DeepMind ([blog](https://bostondynamics.com/blog/boston-dynamics-unveils-new-atlas-robot-to-revolutionize-industry/)); no later primary source confirming deliveries was found.
- **1X**: the 2026-04-30 [factory announcement](https://www.1x.tech/discover/neo-factory) says NEO customer deliveries had not started and the first units went to internal home testing, with customer shipments promised in 2026; no later official confirmation of deliveries was found.

## What to watch next

- Whether NVIDIA releases GR00T N2 by year end as previewed, and with what weights and licence; whether π0.6 / π0.7 weights are opened or access stays with the API.
- Whether real-robot comparisons appear that are not run by the model's own authors, for example more policies entering double-blind evaluations such as RoboArena.
- How Tesla's Q3 2026 letter describes Optimus production.

Back to the [series guide](README.en.md); action-head details are in the [previous note](imitation.en.md); world models used directly as policies (DreamZero, 1X) are covered in the [Frontier block](../../11-frontier/world-models/robotics-driving.en.md).

References: [RT-2](https://arxiv.org/abs/2307.15818v1) · [OpenVLA](https://arxiv.org/abs/2406.09246v1) · [FAST](https://arxiv.org/abs/2501.09747v1) · [π0](https://arxiv.org/abs/2410.24164v1) · [OpenVLA-OFT](https://arxiv.org/abs/2502.19645v1) · [RMA](https://arxiv.org/abs/2107.04034v1)
