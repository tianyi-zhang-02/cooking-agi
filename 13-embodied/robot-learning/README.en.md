# Robot learning: from demonstrations to general policies

[中文](README.md) · **English**

> Reading time: ~6 min · Last reviewed: 2026-10

Suppose we record 100 demonstrations of a robot arm reaching around a cup to pick up a box. After training, we want to know whether it can complete that action—and what happens with a different table, instruction, or robot.

The first note covers learning from demonstrations: how to represent actions, why averaging two valid paths can lead into an obstacle, and how action chunks and data affect learning. The second introduces vision-language-action models (VLAs) and examines what needs checking when transferring from simulation to a real robot.

## How the two notes divide the work

| Note | What it answers | Hand-worked example | After reading you should be able to explain |
| --- | --- | --- | --- |
| [Learning actions from demonstrations](imitation.en.md) | Which extra problems behavior cloning meets on robots; what generative action heads and action chunking each fix; where the data comes from | Left or right around an obstacle: MSE picks 0 and hits the cup; flow matching with only 1 step also falls back to 0 and needs several steps to split to either side; jitter and reaction delay for chunk lengths from 1 to 50 | Why current policies output a stretch of actions at once, and why with diffusion or flow matching |
| [VLA models and sim-to-real](vla.en.md) | How a VLM becomes a robot policy; why humanoids split into fast and slow layers; how RL is added on top of a VLA; how to evaluate; how to close the gap from simulation to the real robot | One 50-step × 7-D chunk: about 3.5 s with token-by-token decoding, about 70 ms with a flow-matching expert (teaching numbers); pushing a puck when friction is uncertain | Where each VLA's actions come out and why; which models are open; what humanoid companies have officially said |

An action travels from data to motors roughly through these steps; the two notes cover the first and second halves:

```text
Demonstrations: teleoperation, hand-held grippers, first-person human video
→ Policy: images + instruction + robot state → a stretch of actions (action chunk)
→ Action head: regression / discrete tokens / diffusion or flow matching
→ Execution: sent to the controller at 20–50 Hz, re-inferred after part of the chunk
→ Evaluation: simulation benchmarks, double-blind real-robot comparisons
→ Improvement: RL on the robot's own experience; randomize in simulation, then deploy
```

## What to read first

| Read first | Why |
| --- | --- |
| [Imitation and rewards](../../05-post-training/deep-rl/imitation-and-rewards.en.md) | Behavior cloning, compounding error, DAgger, causal confusion; the first note assumes you have read it |
| [Model-based RL](../../05-post-training/deep-rl/model-based.en.md) | "Averaging two paths into a wall" and MPC executing only the first action; both examples in the first note continue from here |
| [Qwen-Image: flow matching](../../03-multimodal-learning/image-generation.en.md) | The flow-matching objective and a 1-D example; the first note keeps its notation |
| [LLaVA and DeepSeek-VL](../../03-multimodal-learning/vlm-designs.en.md) | How images enter a language model; the second note assumes you know the basic VLM structure |
| [Actor–Critic and GAE](../../05-post-training/deep-rl/actor-critic-gae.en.md), [How much should a policy change? (PPO)](../../05-post-training/deep-rl/trust-region.en.md), [Offline RL and OPE](../../05-post-training/deep-rl/offline-and-ope.en.md) | Used in the second note for RL on top of VLAs and for training controllers in simulation |

Optional: [How a 3D point becomes a pixel](../perception/camera-geometry.en.md) (camera geometry, helpful for multi-view inputs); [World models as simulators, evaluators and policies for robots and cars](../../11-frontier/world-models/robotics-driving.en.md) (not repeated here).

## State as of October 2026

Each item below is as of 2026-10-15, with its source in the same sentence. "Self-reported" means the company published the number itself and nobody has checked it independently.

- **The usual VLA recipe**: a pretrained VLM plus a continuous "action expert" (flow matching or diffusion) that outputs about 50 steps at a time, at up to 50 Hz. For example, [π0](https://arxiv.org/abs/2410.24164v1) is PaliGemma 3B plus a 300M-parameter expert, and [π0.7](https://arxiv.org/abs/2604.15483v1) is Gemma 3 4B plus an 860M-parameter expert; GR00T N1.x and SmolVLA follow the same pattern.
- **Humanoids split into fast and slow layers**: Figure's [Helix](https://www.figure.ai/news/helix) pairs a 7B VLM at 7–9 Hz with an 80M-parameter policy at 200 Hz, and [Helix 02](https://www.figure.ai/news/helix-02) adds a 1 kHz whole-body controller (self-reported); GR00T and [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) (2026-07-30) use the same split.
- **RL now runs on VLAs**: Physical Intelligence's [π\*0.6 / Recap](https://www.pi.website/blog/pistar06) (2025-11) and [RL tokens](https://www.pi.website/research/rlt) (2026-03) both report gains in throughput or speed (self-reported).
- **The data bet is first-person human video**: NVIDIA's [EgoScale](https://arxiv.org/abs/2602.16710v1) uses 20,854 hours; Figure's [Index](https://www.figure.ai/news/introducing-index) claims more than 16 million videos; Generalist's [GEN-0](https://generalistai.com/blog/gen-0) claims 270,000 hours of manipulation data, growing by about 10,000 hours a week and collected by "1,000s of data collection devices and robots", without saying how much came from people versus robots (all self-reported).
- **Video world models are entering the policy itself**: for example the 14B [DreamZero](https://arxiv.org/abs/2602.15922v1) (2026-02); NVIDIA's [GR00T N2](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world), previewed in March 2026, is based on it, and no official release was found as of 2026-10-15. Details are in the [Frontier block's world-model note](../../11-frontier/world-models/robotics-driving.en.md).
- **Openness is split**: [openpi](https://github.com/Physical-Intelligence/openpi) (π0, π0-FAST, π0.5), GR00T N1.x, OpenVLA, Octo and SmolVLA have open weights; Gemini Robotics VLAs, Figure Helix and π0.6 / π0.7 do not. PI opened a hosted API beta on 2026-10-13 ([PI blog](https://www.pi.website/blog/api-beta)). Licence details are in the second note's table.
- **Simulation is consolidating on GPU parallelism**: [Newton](https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning) went to the Linux Foundation in September 2025, and version 1.0 was released on 2026-03-10; NVIDIA's [press release](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) of 2026-03-16 says Isaac Lab 3.0 (early access) is built on Newton 1.0 plus PhysX; MuJoCo Warp is also part of the Newton project.
- **Evaluation is moving to real robots**: [RoboArena](https://arxiv.org/abs/2506.18123v1) runs double-blind pairwise real-robot comparisons; OpenVLA-OFT reports an average of 97.1% on LIBERO's 4 commonly used evaluation suites, and our judgement is that this simulation benchmark is near its ceiling.
- **Humanoid deployments are real but small**: BMW confirms that Figure 02 helped build more than 30,000 X3s (without saying how many robots); Tesla's update letter of 2026-07-22 says first-generation Optimus lines are being installed "in anticipation of production in 2026", with initial builds used first to collect training data. A company-by-company list based on official statements is at the [end of the second note](vla.en.md).

### Sidewalk delivery robots

Delivery robots also run in the real world; of the three below, Starship has delivered the most and says it has completed 10 million deliveries. This section relies mainly on the companies' own material (all numbers self-reported) and keeps their wording:

- **Coco Robotics**: its [About page](https://www.cocodelivery.com/about) (read 2026-10-15) lists 500,000 deliveries and 1,000 robots produced (written as 500K and 1K), and says the two founders started the project as students at UCLA; press releases from June 2025 and 2026-07-15 ([a reprint of the PR Newswire release](https://www.webull.com/news/15231685580170240)) both say "over 500,000", unchanged for more than a year. Partners include Uber Eats (Los Angeles, from August 2024, per [The Robot Report](https://www.therobotreport.com/uber-eats-rolls-out-coco-sidewalk-delivery-robots-in-la/)), [Wolt](https://www.cocodelivery.com/blog/wolt-and-coco-launch-robot-deliveries-in-turku) in Finland, and [Deliveroo](https://www.cocodelivery.com/blog/coco-robotics-deliveroo-uk-partnership) in the UK, announced 2026-10-05. **One inconsistency is worth keeping in view**: on the About page, the CEO's bio says the company builds "human-piloted" sidewalk robots, while the rest of that page and [press releases](https://www.prnewswire.com/news-releases/coco-robotics-raises-80m-to-expand-autonomous-delivery-and-ai-platform-302478405.html) since 2025 call them "autonomous"; the [Coco 2 page](https://www.cocodelivery.com/coco2) describes end-to-end neural networks with human oversight and a "Pilot Terminal" that steps in when needed. Coco has not published how much driving is autonomous versus remotely piloted, and we found no reputable outlet that gives the share.
- **Starship**: said on 2026-04-28 that it had completed 10 million autonomous deliveries with more than 3,000 robots in 8 countries ([press release](https://www.starship.xyz/press/autonomous-delivery-moves-into-the-mainstream-as-starship-technologies-passes-10-million-deliveries/)), describing this as Level 4 without active human supervision; on 2026-06-04 it announced it is winding down U.S. university-campus operations and moving more than 1,200 campus robots to grocery retailers ([press release](https://www.starship.xyz/press/starship-technologies-doubles-down-on-grocery/)). No remote-assistance share appears in its 2026 announcements.
- **Serve Robotics**: its annual report for 2025 ([10-K](https://www.sec.gov/Archives/edgar/data/1832483/000183248326000010/patr-20251231.htm)) gives a fleet of more than 2,000 robots at 2025-12-31 and describes them as Level 4 robots that can run without humans in the loop "for periods of time", backed by remote supervisors; daily active robots averaged 792 in Q2 2026 ([8-K Exhibit 99.1](https://www.sec.gov/Archives/edgar/data/1832483/000183248326000034/serv-20260630_ex991earning.htm)).

These are all the companies' own statements. "Autonomous" means different things in different companies' material; before comparing, ask how many robots one remote operator watches, when they take over, and whether takeover counts are published. The same questions for cars are in the [autonomous driving series](../driving/README.en.md).

## What this series does not cover

- Low-level control such as PID, LQR and MPC: see [Inside planning](../../05-post-training/deep-rl/planning-and-control.en.md).
- Camera geometry, tracking, 3D reconstruction: see the [perception and geometry series](../perception/README.en.md).
- Autonomous driving: see the [autonomous driving series](../driving/README.en.md).
- Any company's hiring process or interview questions: not collected here.

## Reading order

1. [Learning actions from demonstrations](imitation.en.md): mean squared error on the left-or-right obstacle, flow-matching sampling steps, the chunk-length trade-off, ALOHA, UMI, DROID, Open X-Embodiment, AgiBot World and first-person human video
2. [VLA models and sim-to-real](vla.en.md): from text action tokens to FAST to flow-matching action experts, a hand-worked latency estimate, slow and fast layers, RL on VLAs, evaluation, sim-to-real, an openness table and humanoid status
