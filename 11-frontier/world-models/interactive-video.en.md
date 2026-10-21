# World models: video you can act in (Genie and playable worlds)

[中文](interactive-video.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

Give a video model the prompt "a robot arm pushes the cup 5 cm toward the table edge" and you get a clip a few seconds long. But if you want to stop halfway, push only 2 cm instead, and then turn to look at the other side of the table, you need a model that accepts actions, returns the next frame in real time, and remembers what is out of view.

Alongside single-frame quality, an interactive world model must meet three constraints at once: causal generation, a latency budget and memory. As of October 2026, Genie 3, a closely watched system on this line, still has no public paper or architecture.

## From "generate a clip" to "press a key, get a frame"

An offline video generator receives the whole prompt and delivers the whole clip at once; until delivery, any frame can still change. An interactive model must hand frame $t$ to the player before it knows what the player presses next. So it can only condition on "past frames plus actions so far" when generating the next frame. That is causal generation.

Take the series example (a teaching setup): a robot arm is pushing a cup toward the table edge. Offline, "push 2 cm" and "push 5 cm" are two prompts that yield two unrelated clips. Interactively, the first 10 frames are identical, and only at frame 11 does the player decide whether to keep pushing; while generating the first 10 frames, the model cannot peek at that decision.

| | Offline generation (Sora, Veo) | Interactive world model (Genie 3 and similar) |
| --- | --- | --- |
| When conditions arrive | All at once, before generation | A new action may arrive at any frame |
| Can generation use later frames | Yes, the clip is editable until delivery | No, later frames depend on keys not yet pressed |
| Latency requirement | Tens of seconds of waiting is fine | Every frame must arrive within the frame interval |
| Length | Clips of a few to a dozen seconds | Measured in sessions, minutes long |
| Memory | Consistent within the clip | Turn away and back, and things should still be there |

Roughly speaking, systems like Genie 3, GWM and Odyssey add action conditioning, real-time causal generation and persistence, while Sora and Veo are offline prompt-to-clip generators. The two camps also describe themselves differently:

- In its February 2024 [Sora technical report](https://openai.com/index/video-generation-models-as-world-simulators/), OpenAI called scaling video generation a promising path toward general-purpose simulators of the physical world, and also listed failures such as glass shattering and eating that leaves no change on the food. The [Sora 2 launch page](https://openai.com/index/sora-2/) (2025-09-30) argued that a useful simulator must model failure: a missed basketball shot should bounce off the rim, not teleport into the hoop. According to OpenAI's [help page](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation), the Sora app shut down on 2026-04-26 and the API on 2026-09-24.
- Google's [Veo 3 technical report](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf) describes joint latent diffusion over audio and video latents and makes no world-model claim. The page for [Gemini Omni](https://deepmind.google/models/gemini-omni/), Google's 2026 video model, does not use the term either, and lists Genie 3 separately as a specialised model.

## The latency budget: about 42 ms per frame at 24 fps

Start with arithmetic. With target frame rate fps and $K$ sampling steps per frame (diffusion denoising steps or MaskGIT decoding steps):

$$
\text{maximum time per sampling step}\approx\frac{1000\ \text{ms}}{\text{fps}\times K}
$$

This ignores decoding to pixels, network transfer and queueing, so it is an upper bound. At 24 fps a frame has only $1000/24\approx41.7$ ms. With 25 steps per frame ([Genie](https://arxiv.org/abs/2402.15391) uses 25 MaskGIT steps per frame at inference), each step gets about 1.7 ms; distilled to 4 steps, about 10.4 ms; at 3 steps, about 13.9 ms. The Genie paper (February 2024) itself states in §5 that it runs at about 1 FPS.

This explains why 2025–26 open systems converged on one direction: a causal, frame-by-frame diffusion Transformer trained with diffusion forcing or self-forcing, then distilled to 2–4 sampling steps. That is our synthesis across several works, not the conclusion of any single paper. For example, Skywork's [Matrix-Game 2.0](https://arxiv.org/abs/2508.13009) (August 2025) uses 3 steps; Dreamer 4 (September 2025) runs 4 forward passes per frame; Overworld's [Waypoint-1](https://huggingface.co/blog/waypoint-1) uses diffusion forcing, then self-forcing. Diffusion forcing gives each frame its own noise level; see the [Dreamer note](dreamer.en.md) for details.

Frame rate is not response latency. A system can pipeline a steady 24 fps while the frame that reacts to a key press arrives several frames late. Google, introducing [Project Genie](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/) on 2026-01-29, wrote that characters can lag; Decart reported on 2026-06-10 that [Oasis 3](https://decart.ai/publications/introducing-oasis-3-first-interactive-world-model-for-physical-ai) has latency under 200 ms (vendor-reported). The [Omni streaming note](../../03-multimodal-learning/omni-streaming.en.md) makes the same point: hearing the first sound early does not mean the whole stream is smooth.

Hardware belongs next to every number. All the frame rates below are reported by the releasing organisation: Matrix-Game 2.0 reaches 25 FPS on one H100; [Matrix-Game 3.0](https://arxiv.org/abs/2604.08995) reaches 40 FPS at 720p using eight GPUs for the DiT plus one for the VAE (paper §3.4); the [Waypoint-1.5-1B model card](https://huggingface.co/Overworld/Waypoint-1.5-1B) states 720p at 56 FPS on one RTX 5090; [Dreamer 4](https://arxiv.org/abs/2509.24527) Table 1 gives 640×360 at 21 FPS on one H100. Resolutions and GPU counts differ, so these cannot be ranked directly.

## Genie's recipe: learning "key presses" without action labels

The web holds huge amounts of gameplay video but almost no record of which keys players pressed. [Genie](https://arxiv.org/abs/2402.15391) (Google DeepMind, February 2024, ICML 2024 Best Paper) learns its own set of discrete "actions" from the change between consecutive frames. Section 2.1 of the paper splits the model into 3 parts, all built from spatiotemporal (ST) Transformer blocks:

```text
video frames x_1..x_T
 ├─ 1. video tokenizer (VQ-VAE, causal in time) → discrete tokens per frame z_1..z_T
 ├─ 2. latent action model (LAM)
 │     encoder: sees x_1..x_t and x_{t+1} → continuous vector ã_t → quantised to one of 8 codes
 │     decoder: sees only the history and ã_t → must reconstruct x_{t+1}
 └─ 3. dynamics model (MaskGIT Transformer)
       inputs z_1..z_{t-1} and ã_1..ã_{t-1} (stop-gradient) → predicts the next frame's tokens
inference: only the LAM codebook is kept; the player enters an integer 0-7 → codebook lookup → dynamics model makes the next frame
```

The key is the LAM decoder. It cannot see the future frame, so only the action code can tell it what changed between past and future. And the codebook has only $|A|=8$ codes (the paper says this size was chosen so humans can play), too small to hold a whole image; it can only carry the most important change. The paper gives no numbered loss equation for the LAM, only a prose description.

<!-- paper-figure:genie-lam -->

### Walking through 8 codes (a teaching example)

Suppose that after training the codebook holds the 8 vectors below. The numbers are invented so you can compute by hand; real codebooks have more dimensions, and what each code means is only learned by people playing with it afterwards.

| Code | Vector | Label a person gives it after playing |
| --- | --- | --- |
| 0 | (0, 0) | No move |
| 1 | (0.5, 0) | Push right a little |
| 2 | (1.2, 0) | Push right a lot |
| 3 | (−0.5, 0) | Left a little |
| 4 | (−1.2, 0) | Left a lot |
| 5 | (0, 0.6) | Lift |
| 6 | (0, −0.6) | Lower |
| 7 | (0.8, 0.8) | Up and to the right |

Between two frames the cup is pushed 2 cm and the LAM encoder outputs $\tilde a=(0.45,0.05)$. Find the nearest code by squared distance: to code 1 it is $0.05^2+0.05^2=0.005$, to code 0 it is $0.45^2+0.05^2=0.205$, to code 2 it is $0.75^2+0.05^2=0.565$, so it quantises to code 1. For a 5 cm push the encoder outputs $(1.1,0.1)$: to code 2 it is $0.01+0.01=0.02$, to code 1 it is $0.36+0.01=0.37$, so code 2 wins.

Now try a 3.5 cm push with encoder output $(0.85,0)$: the squared distance to both code 1 and code 2 is $0.35^2=0.1225$, exactly on the boundary. The player has only 8 keys, and "push 3.5 cm" is not one of them. A small codebook is easy to play and easy to learn, but the actions it can express are coarse; that is a design trade-off.

One more point is our inference, not the paper's conclusion: the objective only asks the code to describe the main change from past to future, and does not distinguish whether the agent caused it. If the training video often has a shaking camera, one code may end up meaning "camera shake".

How is controllability measured? Section 3 of the paper defines:

$$
\Delta_t\text{PSNR}=\text{PSNR}(x_t,\hat x_t)-\text{PSNR}(x_t,\hat x'_t)
$$

$\hat x_t$ is generated with latent actions inferred from the real video, $\hat x'_t$ with random latent actions, and the paper reports $t=4$. Higher PSNR means the generated frame is closer to the real one. Teaching numbers: 27.0 dB for the first and 24.5 dB for the second, a gap of 2.5 dB, means the codes really steer the picture; if the dynamics model ignored the codes, the two would be about equal and the gap near 0. The metric shows only that actions have an effect, not that the effect is physically right.

On scale, §3 of the paper filters 30k hours (6.8M 16-second clips at 10 FPS, 160×90) from 55M platformer clips. The abstract says 11B; §3.1 itemises 10.1B for dynamics, 200M for the tokenizer and 300M for the LAM. Limitations stated in §5: it can hallucinate unrealistic futures, has only 16 frames of memory, and runs at about 1 FPS. The same idea of inferring actions from unlabelled video was later used to add pseudo-actions to robot video; see DreamGen in the [robots and cars note](robotics-driving.en.md).

## From Genie to Genie 3, and other systems of the same period

| Date | System (organisation) | What the releasing organisation made public |
| --- | --- | --- |
| 2024-08 | [GameNGen](https://arxiv.org/abs/2408.14837) (Google) | Stable Diffusion v1.4 fine-tuned into a real-time DOOM engine, 20 FPS on one TPU |
| 2024-10-31 | [Oasis](https://oasis-model.github.io/) (Decart + Etched) | Real-time Minecraft-like world model, DiT + ViT VAE, Diffusion Forcing; 500M-parameter weights open |
| 2024-12-04 | [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) (Google DeepMind) | Image-prompted, action-controllable 3D worlds, consistent for up to about a minute; moved to autoregressive latent diffusion, with classifier-free guidance for action control |
| 2025-02-19 | [WHAM / Muse](https://www.nature.com/articles/s41586-025-08600-3) (Microsoft Research + Ninja Theory) | 1.6B, trained on more than 7 years of Bleeding Edge gameplay; Nature paper |
| 2025-04-04 | [WHAMM](https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/) (Microsoft Research) | Real-time Quake II demo, 10+ fps at 640×360 |
| 2025-08-05 | [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) (Google DeepMind) | Text-prompted, real-time 720p at about 24 fps; consistent for several minutes, recalls visual details from about a minute earlier; text-triggered world events (change the weather, add objects or characters); at launch a research preview for a small group of academics and creators |
| 2025-11-13 | [SIMA 2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/) (Google DeepMind) | Gemini-based agent, tested in Genie 3 worlds and self-improving inside them |
| 2026-01-29 | Project Genie (Google DeepMind / Google Labs) | Web prototype combining Genie 3, Nano Banana Pro and Gemini; open only to Google AI Ultra subscribers in the US aged 18+; 60-second generations; Google itself states that worlds may not follow prompts or physics, characters can lag, and promptable events are not yet included |
| 2026-02-06 | [Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation) (Waymo) | Genie 3 post-trained for driving, emitting camera and lidar; no fidelity metrics published |
| 2026-05-19 | [Project Genie expands](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie-expands/) (Google) | Adds Street View grounding (US places only) and a gradual worldwide rollout to Ultra subscribers |

Google DeepMind's Genie 3 blog (2025-08-05) says each frame is generated autoregressively, conditioned on the growing session history, and that consistency is "emergent" rather than built from an explicit 3D representation such as NeRF or Gaussian Splatting. The same post lists limitations: a narrow set of actions an agent can take directly, difficulty modelling several independent agents, imperfect geography for real places, frequently garbled text unless it appears in the prompt, and sessions lasting minutes rather than hours.

**As of 2026-10-15, we found no public Genie 3 paper or technical report, no API, and no Genie 4; its architecture, parameter count and training data are unpublished.** So no outsider can tell whether its consistency comes from long context, explicit memory or something else, and Genie 1's structure is no basis for a guess.

Other players of the same period (frame rates and resolutions are self-reported by the releasing organisation):

- **Runway**: [GWM-1](https://runway.com/research/introducing-runway-gwm-1) (2025-12-11) is an autoregressive real-time world-model family built on Gen-4.5 (Worlds / Avatars / Robotics); [GWM Worlds 2](https://runway.com/research/introducing-gwm-worlds-2) (2026-09-03, research preview) reports real-time 720p at 24 fps with 48 kHz audio.
- **Odyssey**: [Odyssey-2](https://odyssey.systems/introducing-odyssey-2) (2025-10-27) is causal, text-steerable interactive video at 20 fps; [Odyssey-2 Pro](https://odyssey.systems/the-gpt-2-moment-for-world-models) (2026-01-23) runs 720p at 22 FPS with an API; [Odyssey-3](https://odyssey.systems/introducing-odyssey-3) (2026-09-15) is a foundation world model with action decoders.
- **Skywork**: Matrix-Game 2.0 (August 2025) is an open 1.8B model; Matrix-Game 3.0 (March/April 2026) is 5B with memory, in an MIT-licensed repo; on 2026-10-15 the [repo](https://github.com/SkyworkAI/Matrix-Game) added Matrix-Game 4.0 (a game engine plus a causal video renderer), which we checked only from the repository description, with no paper to read.
- **Overworld**: Waypoint-1 (2026-01-20) is a 2.3B frame-causal rectified-flow model reporting 30–60 FPS on an RTX 5090; Waypoint-1 / 1.5 are released under Apache-2.0.
- **Tencent**: [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) (paper title WorldPlay, also called HY-World 1.5, 2025-12-16) is an open streaming video-diffusion world model at 720p and 24 FPS; WorldPlay-5B followed on 2026-01-06.
- Also Decart's [Lucy 2](https://decart.ai/publications/lucy-2-introducing-sota-video-generation-in-realtime) (2026-01-26, vendor-reported 1080p at 30 fps) and Robbyant's [LingBot-World](https://arxiv.org/abs/2601.20540) (2026-01-28, open, 16 fps).

## What to ask of an interactive world-model demo

| Ask first | Why | Example from this note |
| --- | --- | --- |
| On what hardware were frame rate and resolution measured? | Eight GPUs plus one and a single consumer card are different things | Matrix-Game 3.0's 40 FPS uses eight GPUs plus one; Waypoint-1.5-1B states an RTX 5090 |
| How long from key press to visible change? | High frame rate does not mean fast response | Project Genie says characters can lag |
| How long does it stay consistent? | A few good seconds are easy; long sessions drift | Genie 1 remembers 16 frames; Genie 3 claims minutes; Project Genie generates 60 seconds |
| What controls exist? | Arrow keys alone differ a lot from adding objects or changing weather | Genie 1 has 8 latent actions; Genie 3 has promptable events, not yet in Project Genie at launch |
| Is out-of-view content still there, and still changing? | Static recall and dynamic evolution are different abilities | WRBench and WorldMemBench findings are in the next section |
| How open is it? | A video-only demo cannot be reproduced or tested by others | Genie 3 has no paper and no API; Matrix-Game, Waypoint and HY-WorldPlay have open weights |

## Limits: memory, physics and drift

**Memory.** Published numbers are growing, but none is long: Genie 1 remembers 16 frames; Genie 2 about a minute; Google DeepMind says Genie 3 recalls visual details from about a minute earlier and stays consistent for a few minutes; Dreamer 4's context is 9.6 seconds. Several 2026 benchmarks ask finer questions:

- [WRBench](https://arxiv.org/abs/2606.20545) (Lu et al., USTC and X-Humanoid, June 2026 paper, 9,600 videos from 23 models) found that when the camera looks away and back, models tend to restore the scene as it was left instead of advancing the event that was in motion. In the cup example: the cup was sliding toward the edge when the camera turned; on return it should be on the floor, but models often leave it at the edge.
- [WorldMemBench](https://arxiv.org/abs/2610.19063) (Zhou et al., October 2026 paper) concludes that good static recall does not mean reliable evolution of state. [MBench](https://arxiv.org/abs/2606.00793) (June 2026 paper) also reports systematic memory failures.

**Control and stability.** [WorldMark](https://arxiv.org/abs/2604.21686) (Xu et al., April 2026 paper) gave 10 models one shared WASD action vocabulary and 500 test cases, and found that the fastest responders are often the least stable and the best-looking model ranks last on direction accuracy. [WBench](https://arxiv.org/abs/2605.25874) (Meituan LongCat and Fudan, May 2026 paper, 289 cases, 20 models) found no model strong on every dimension.

**Physics and drift.** On 2026-01-29 Google wrote that Project Genie's worlds may not follow prompts or physics. In autoregressive generation every frame is conditioned on the model's own past output, so small errors carry forward; this is the same family of problem as rollout error accumulation in [model-based RL](../../05-post-training/deep-rl/model-based.en.md). The Dreamer 4 paper argues that v-prediction's high-frequency outputs accumulate errors over long frame-by-frame rollouts, and so predicts clean latents directly (see the [Dreamer note](dreamer.en.md)).

<!-- paper-figure:gamengen-drift -->

There are several ways to keep out-of-view content. The trade-offs below are our analysis, not the vendors' conclusions:

- Rely on an ever-longer history: the Genie 3 blog describes conditioning on session history with emergent consistency. No extra structure is needed, but a longer history costs more, and anything beyond the window is forgotten.
- Use posed past frames as spatial memory: World Labs' [RTFM](https://www.worldlabs.ai/blog/rtfm) (2025-10-16, one H100) does this. Returning to a viewpoint retrieves the matching view, but objects that keep changing need separate handling.
- Generate explicit 3D: World Labs' [Marble](https://www.worldlabs.ai/blog/marble-world-model) outputs Gaussian splats and meshes. Static geometry is never forgotten, but a cup that is sliding off is not part of that representation.

**Exploitation by agents.** SIMA 2 self-improved inside Genie 3 worlds. [Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122) already found that a VizDoom agent learned, inside the "dream", an adversarial policy that stopped the monsters from firing; only a higher sampling temperature made the dream harder to exploit (§4.5). When an interactive world model is a training ground, progress has to be checked in the real environment.

## What to watch on this line next

1. **Will Genie 3's technical details become public?** As of 2026-10-15 no Genie 3 paper, API or open weights can be found; Project Genie is a prototype for subscribers. A result that would change the picture: a paper or technical report stating what makes it consistent (long context or explicit memory) and on what hardware it runs in real time.
2. **Can minute-scale memory cover content that is out of view and still changing?** WRBench (June 2026) and WorldMemBench (October 2026) both report failures. A result that would change the picture: a model that reliably passes the dynamic-evolution part of such tests, not only static recall, reproduced by a third party.
3. **Can speed and stability coexist?** WorldMark (April 2026) found the fastest responders are often the least stable. A result that would change the picture: one model that is both fast and stable under a shared action vocabulary and stated hardware.
4. **Do skills learned in generated worlds carry back to the real one?** SIMA 2 (November 2025) self-improved in Genie 3 worlds; Waymo's Genie-3-based model (February 2026) published no fidelity metrics. A result that would change the picture: public data placing progress in generated worlds next to real-task performance.

Next, read [robots and cars: simulator, evaluator and policy](robotics-driving.en.md), or skip to [do they actually understand the world?](do-they-understand.en.md).
