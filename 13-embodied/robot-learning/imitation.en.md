# Robot learning: learning actions from demonstrations

[中文](imitation.md) · **English**

> Reading time: ~11 min · Draft, pending review

A cup sits on the table. You teleoperate a robot arm around it to reach a box behind it and record 100 demonstrations: 50 pass on the left, 50 on the right. Train a policy with mean squared error and it drives straight into the cup. **Imitation learning on robots is hard in three places: one image can have several correct actions, small per-step deviations add up, and demonstrations are expensive and narrow. The common combination today is a generative action head (diffusion or flow matching) to represent "several answers are correct", action chunking to decide a short stretch at once and keep consecutive actions consistent, and cheaper ways to collect more data.**

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

## What changes on a real robot

The behavior cloning (BC) objective, why errors can accumulate as $O(T^2\epsilon)$ in the worst case, how DAgger asks the expert about states the learner itself reaches, and causal confusion are all covered in [Imitation and rewards](../../05-post-training/deep-rl/imitation-and-rewards.en.md). "One-step mean squared error averages two paths into a wall" appears in [Model-based RL](../../05-post-training/deep-rl/model-based.en.md) for state prediction; the same logic holds for action prediction. This note does not repeat them. It looks at three things that change on a real robot:

| What changes on a robot | Concretely | Section in this note |
| --- | --- | --- |
| Actions are continuous, high-dimensional and fast | A single arm commonly has 7 dimensions (end-effector translation, rotation, gripper), a fixed-base two-arm setup 14; commands go out at 20–50 Hz | Generative action heads, action chunking |
| Human demonstrations naturally have several modes | Same image, the operator passes left this time and right the next | Worked example: left or right around an obstacle |
| Demonstrations are expensive | Someone has to operate the device, reset the scene and check quality | Where the data comes from |

The dimension and frequency examples come from [OpenVLA](https://arxiv.org/abs/2406.09246v1) (7-D actions) and [π0](https://arxiv.org/abs/2410.24164v1) (14-D for fixed-base two-arm robots, up to 50 Hz; its mobile two-arm robots use 16–17-D).

## Worked example: why mean squared error drives into the obstacle

Teaching setup: look at a single number. The action $a$ is the arm's sideways offset in front of the cup: $+1$ means pass on the left, $-1$ pass on the right, $0$ means straight at the cup. Half the demonstrations do each, and the image alone does not say which side this episode should take.

If the policy outputs one number $c$, the expected squared error is:

$$
\mathbb E[(c-a)^2]=\tfrac12(c-1)^2+\tfrac12(c+1)^2=c^2+1 .
$$

The loss is smallest at $c=0$, where it equals 1; honestly predicting $+1$ costs 2. **The better the training, the more surely it picks 0, straight into the cup.** L1 does not help either: $\tfrac12|c-1|+\tfrac12|c+1|$ equals 1 everywhere on $[-1,1]$, so 0 is still one of the optima and the model has no reason to prefer a side.

| Output | What it gives in this example | What happens |
| --- | --- | --- |
| Regress one number (MSE) | 0 | Drives into the cup |
| Regress one number (L1) | Any point in $[-1,1]$; 0 is also optimal | No guarantee of avoiding it |
| Learn the whole distribution, then sample | Half the samples are $+1$, half $-1$ | Avoids the cup every time; left or right at random |

"The average of the samples is also 0" is not a problem: the robot executes one sample at a time, never the average. The real risk is different, and it shows up with action chunking: if the policy draws a fresh sample at every step, it may go left on one step and right on the next.

The example is deliberately extreme. Real demonstrations are rarely exactly 50/50, and the region between two modes is not always an obstacle; when demonstrations follow essentially one strategy, regression is often enough (see the table at the end).

## Generative action heads: let the noise pick the side

[Diffusion Policy](https://arxiv.org/abs/2303.04137v1) (Columbia, TRI, MIT, 2023-03) writes the policy as conditional denoising: starting from noise, it denoises a sequence of actions conditioned on the observation. The paper's T-block pushing example shows a state from which the block can be approached on the left or on the right; the policy learned both and commits to one of them in each rollout (§IV-A, Fig. 4). The authors report an average improvement of 46.9% over the previous best methods across 11 tasks from 4 benchmarks.

<!-- paper-figure:dp-multimodal -->

Flow matching is another way to train the same family of generative models. The objective and a 1-D example are in the [Qwen-Image note](../../03-multimodal-learning/image-generation.en.md); this note keeps its notation: time runs from the noise end $t=0$ to the data end $t=1$, $x_t=(1-t)\epsilon+ta$, and the model learns a velocity. Applied to the obstacle example, it shows why the number of sampling steps matters.

Teaching setup: no network is trained; we use the **ideal** velocity field for this two-point distribution, so the only source of error is the number of integration steps. Given $a=\pm1$, $x_t$ is normal with mean $ta$ and standard deviation $1-t$, so:

$$
v^*(x,t)=\frac{\mathbb E[a\mid x_t=x]-x}{1-t},\qquad
\mathbb E[a\mid x_t=x]=\tanh\frac{xt}{(1-t)^2}.
$$

First formula: the target velocity is $u=a-\epsilon$; substituting $\epsilon=(x-ta)/(1-t)$ gives $u=(a-x)/(1-t)$, then take the conditional expectation over $a$. Second formula: the log ratio of the two normal likelihoods is $2xt/(1-t)^2$, which turns into $\tanh$ for the expectation of $\pm1$. Start from noise $\epsilon$ and take $n$ Euler steps of size $1/n$:

- **1 step.** At $t=0$, $\tanh 0=0$ and the velocity is $-x$, so one full step gives $x_1=\epsilon-\epsilon=0$. Whatever the noise, the result is the mean 0, into the cup just like MSE regression.
- **2 steps, from $\epsilon=0.5$.** The first step reaches $0.5-0.5\times0.5=0.25$. The last step size equals the remaining $1-t$, and substituting shows it lands on $\mathbb E[a\mid x_{0.5}=0.25]=\tanh 0.5\approx0.46$, halfway.
- **10 steps, from $\epsilon=0.5$.** The first four steps move toward the middle (0.45, 0.41, 0.37, 0.35) and only separate after the halfway point; step 9 reaches 0.84, and the last step lands on $\tanh(0.84\times90)\approx1.00$. Starting from $-0.5$ ends symmetrically at $-1.00$.

| Euler steps $n$ | End point from $\epsilon=0.5$ | End point from $\epsilon=0.02$ |
| --- | --- | --- |
| 1 | 0 | 0 |
| 2 | 0.46 | 0.02 |
| 5 | 1.00 | 0.57 |
| 10 | 1.00 | 1.00 |

Two conclusions. First, with too few steps a generative action head falls back to the average, and noise close to the boundary between the two modes needs more steps to separate. Second, the noise decides the side: trajectories of a 1-D ODE cannot cross, so noise below 0 ends at $-1$ and noise above 0 at $+1$. π0 integrates 10 steps at inference (step size 0.1, [paper §IV](https://arxiv.org/abs/2410.24164v1)). This example uses the ideal field; a learned field adds its own errors, and one-step generation needs dedicated training methods that this note does not cover.

<!-- widget:step-flow-euler-steps -->

Diffusion-type models are not the only heads that can represent "several answers are correct":

| Action head | Can it represent several modes? | Cost |
| --- | --- | --- |
| Gaussian mixture | Yes, but the number of modes is fixed in advance | In Diffusion Policy's Fig. 4, the LSTM-GMM baseline (and IBC) leans toward one mode |
| Bin each dimension and classify (action tokens) | Yes | Dimensions × steps tokens must be generated one by one; the next note computes the latency |
| CVAE (ACT's approach) | During training, through a latent $z$ | At test time ACT sets $z$ to the prior mean 0 and decodes deterministically |
| Diffusion / flow matching | Yes | Inference needs several integration steps; the [OpenVLA-OFT](https://arxiv.org/abs/2502.19645v1) authors also note slower training |

## Action chunking: decide a stretch at once

[ACT](https://arxiv.org/abs/2304.13705v1) (Action Chunking with Transformers, Stanford, 2023-04) reads an observation every $k$ steps and outputs the next $k$ actions at once. The paper's reason: the number of decisions drops to $1/k$, which shortens the task's effective horizon $k$-fold and directly targets compounding error (§IV-A). Diffusion Policy adds a second reason: if each step is sampled independently from a multimodal distribution, consecutive steps can come from different modes and the motion jitters between two valid routes (§IV-C).

<!-- paper-figure:act-chunking -->

### Worked example: choosing the chunk length (teaching example)

Setup: 50 Hz control, 20 ms per step. During the first 4 steps in front of the cup, the image does not yet show which side the arm has started toward, so each decision the policy makes is a fresh coin flip, half left and half right; after 4 steps the arm is clearly on one side and the policy follows through. Separately, at some moment someone bumps the cup, and the policy only notices at its next observation.

- **Probability of not switching sides during the ambiguous window.** With chunk length $k=1$ there are 4 flips in 4 steps, and the chance that the last 3 all match the first is $0.5^3=0.125$; with $k=2$ there are 2 flips, probability $0.5$; with $k\ge4$ one chunk covers the whole window, probability 1.
- **Reaction delay.** Actions inside a chunk are already fixed, so after a disturbance the robot still executes on average $(k-1)/2$ old actions, at worst $k-1$, at 20 ms each.

| Chunk length $k$ | No side switch in the ambiguous window | Old actions still executed after a disturbance (mean) | In time (mean / worst) |
| --- | --- | --- | --- |
| 1 | 0.125 | 0 | 0 / 0 ms |
| 2 | 0.5 | 0.5 | 10 / 20 ms |
| 4 | 1 | 1.5 | 30 / 60 ms |
| 10 | 1 | 4.5 | 90 / 180 ms |
| 25 | 1 | 12 | 240 / 480 ms |
| 50 | 1 | 24.5 | 490 / 980 ms |

Going from 1 to 4 removes the jitter; going further only buys slower reactions. The example ignores inference time and assumes every action in the chunk is correct; longer chunks also mean longer sequences to predict, which are harder to learn accurately.

How real systems handle this tension:

- **ACT's ablation.** With temporal ensembling off and averaged over 4 simulated settings, success is 1% at $k=1$, 44% at $k=100$, and slightly lower at $k=200$ and $400$; the authors attribute the dip to near open-loop control, too little reactivity, and the difficulty of modeling long sequences (§VI-A).
- **ACT's temporal ensembling.** The policy is queried at every step, so each time step receives predictions from several chunks, which are averaged with weights $w_i=\exp(-m\,i)$, where $w_0$ goes to the oldest prediction. The authors find both chunking and temporal ensembling important for success.
- **Diffusion Policy's receding horizon.** It reads the latest $T_o$ observation steps, predicts $T_p$ steps, executes $T_a$ of them and then predicts again (§II-C). This is the same idea as [MPC executing only the first action](../../05-post-training/deep-rl/model-based.en.md), with several actions executed at a time. The paper's ablation in Fig. 6 describes the choice of $T_a$ as a trade-off between temporal consistency and responsiveness.
- **What π0 does.** It outputs 50 steps at once; 50 Hz robots re-run inference after executing 25 steps (0.5 s), 20 Hz robots after 16 (0.8 s). The appendix says: "We tried temporal ensembling … early on and found that it hurt policy performance …", so chunks are executed open loop ([Appendix A-D](https://arxiv.org/html/2410.24164v1#A1.SS4)).

ACT finds temporal ensembling important and π0 finds it harmful; the two papers differ in model, action head and tasks, so this does not settle which is right. Our inference: if chunks come from a multimodal sampler, averaging one chunk that chose left with one that chose right brings you back to 0, while ACT decodes deterministically at test time, so its chunks are more likely to agree. Neither paper tests this directly.

## Where the data comes from

| Date | Dataset / tool | How it is collected | Scale (as reported by each) | Notes |
| --- | --- | --- | --- | --- |
| 2023-04 | [ALOHA](https://arxiv.org/abs/2304.13705v1) | Low-cost bimanual teleoperation, recorded at 50 Hz | 6 real tasks with about 10 minutes of demonstrations each; success from 20% to 96%, with the abstract's examples (opening a condiment cup, slotting a battery) at 80–90% (authors) | One platform |
| 2023-08 | [BridgeData V2](https://arxiv.org/abs/2308.12952v1) | Low-cost robot arm | 60,096 trajectories, 24 environments | Open |
| 2023-10 | [Open X-Embodiment](https://arxiv.org/abs/2310.08864v1) | Pools 60 existing datasets | 1M+ trajectories, 22 embodiments, 527 skills | Contributor count: 21 institutions in the abstract, 34 labs in the body text |
| 2024-01 | [Mobile ALOHA](https://arxiv.org/abs/2401.02117v1) | Whole-body teleoperation on a mobile base | 50 demonstrations per task; co-training with static ALOHA data raises success by up to 90% (authors) | |
| 2024-02 | [UMI](https://arxiv.org/abs/2402.10329v1) | Hand-held gripper used outside the lab; actions stored as relative trajectories, with latency matching | Dynamic, bimanual and long-horizon tasks (authors) | The abstract does not name the camera |
| 2024-03 | [DROID](https://arxiv.org/abs/2403.12945v1) | 50 collectors on 3 continents with Franka arms over 12 months | 76k trajectories / 350 h, 564 scenes; 84 tasks per the arXiv abstract, 86 in the v1 body text | Open |
| 2025-03 | [AgiBot World](https://arxiv.org/abs/2503.06669v1) | Large in-house real-robot collection | 1M+ trajectories, 217 tasks; says pretraining beats OXE-only by 30% (self-reported) | Robot count not in the abstract |
| 2025-05 | [EgoDex](https://arxiv.org/abs/2505.11709v1) | First-person video on Apple Vision Pro, with 3D hand and finger tracking | 829 h, 194 tabletop tasks | No robot actions |
| 2026-02 | [EgoScale](https://arxiv.org/abs/2602.16710v1) | Action-labelled first-person human video, then alignment on 50 h of human data plus 4 h of robot data | 20,854 h; validation loss falls log-linearly with data and moves with real-robot success; on 5 dexterous tasks with a 22-DoF hand, the abstract reports average success 54% above a no-pretraining baseline (without saying whether the gain is absolute or relative; self-reported) | NVIDIA's [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) readme says GR00T N1.7 is pretrained on it |

A few numbers you can check yourself. A DROID trajectory averages about $350\times3600/76{,}000\approx16.6$ seconds. EgoScale's 20,854 hours are about 60 times DROID, and about twice π0's own robot data (the paper says about 10,000 hours).

Why the 2025–26 bet is first-person human video: it needs no robot and no teleoperation rig, and covers far more scenes. [Index](https://www.figure.ai/news/introducing-index), introduced by Figure on 2026-08-25, is an app through which Figure pays users (Creators) to record real tasks at home and at work; Figure says it has more than 16 million videos (self-reported). Two costs:

- **The video has no robot actions.** Hand positions in 3D have to be estimated from the video and used as actions. Physical Intelligence's [human-to-robot](https://www.pi.website/research/human_to_robot) work (2025-12) records human video with a head camera and wrist cameras, estimates 3D hand keypoints with visual SLAM as the actions, and mixes this video with robot data when fine-tuning π0.5; it reports roughly 2x better results across 4 generalization scenarios (self-reported).
- **A human hand is not a gripper.** EgoScale adds a dedicated human–robot alignment stage. PI also reports that transfer from human video improves with more diverse robot pretraining data (self-reported), so human video complements robot data rather than replacing it.

## Think before you run it

| What to try | What you should observe | What you can NOT conclude |
| --- | --- | --- |
| Train an MSE policy on demonstrations split half left, half right | Predictions cluster near 0 | That regression never works: OpenVLA-OFT, with L1 regression and action chunks, reports 97.1% on LIBERO (authors); with near-unimodal demonstrations regression is often enough |
| Cut flow integration from 10 steps to 1 | Samples collapse toward the mean | That few-step generation can never work: one-step generation needs dedicated training methods not covered here |
| Change the chunk length from 1 to 4 to 50 | First the jitter disappears, then reactions slow down | That longer chunks are always better: ACT dips slightly at $k=200$ and $400$ |
| Add human-video pretraining | Change in success rate in new scenes | That robot data is no longer needed |

The next note puts these pieces inside a vision-language model: [VLA models and sim-to-real](vla.en.md). Go back to the [series guide](README.en.md), or review [DAgger and causal confusion](../../05-post-training/deep-rl/imitation-and-rewards.en.md).

References: [ACT / ALOHA](https://arxiv.org/abs/2304.13705v1) · [Diffusion Policy](https://arxiv.org/abs/2303.04137v1) · [π0](https://arxiv.org/abs/2410.24164v1) · [DAgger](https://arxiv.org/abs/1011.0686) · [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979)
