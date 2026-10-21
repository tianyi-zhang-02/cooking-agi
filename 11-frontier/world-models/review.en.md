# World models: review questions

[中文](review.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-10

## Interview questions

<details class="interview" markdown="1">
<summary>When someone says "world model", which questions do you ask first to know which kind they mean?</summary>

Ask four things: what it predicts, in what space, conditioned on what (actions or not), and what it is used for. With those four, common usages split into learned simulators in RL (Dreamer, MuZero), representation-space prediction in the JEPA line, interactive video (Genie 3), world foundation models for robots and driving (Cosmos, GAIA), 3D spatial generation (Marble), and the question of whether LLMs or video models carry an implicit world model inside. World Labs' June 2026 taxonomy is more compact, sorting by output: renderers output observations, simulators output state, planners output actions.

Models with the same name can learn completely different things: Dreamer reconstructs its inputs, JEPA predicts embeddings, and MuZero predicts only reward, value and policy, with hidden states that need not correspond to any real state. See the [guide](README.en.md).

</details>

<details class="interview" markdown="1">
<summary>What are the parts of DreamerV3's RSSM, and how is the actor trained "in imagination"?</summary>

A deterministic recurrent state $h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1})$ carries memory, and a stochastic categorical latent $z_t$ expresses uncertainty; the encoder gives the posterior $q(z_t\mid h_t,x_t)$, the dynamics give a prior $p(z_t\mid h_t)$ that sees no observation, and there are reward, continue and decoder heads. The loss sums a prediction term, $L_{\rm dyn}$ (the prior chases the posterior) and $0.1\,L_{\rm rep}$ (the encoder becomes predictable), with both KL terms clipped at 1 nat by free bits.

Imagination starts from replayed real states and rolls the prior forward 16 steps; the critic fits the λ-return, and the actor's policy gradient is weighted by "advantage divided by the 5th–95th percentile range of returns (at least 1)". The actor sees only the numbers the model gives it. Dreamer 4 (September 2025) swapped in a 2B video world model and obtained Minecraft diamonds from offline data only, but in just 0.7% of episodes. See [Dreamer](dreamer.en.md).

</details>

<details class="interview" markdown="1">
<summary>Why does JEPA not predict pixels? What is "collapse", and how is it prevented?</summary>

LeCun's 2022 reasoning: pixel-level futures (leaves, textures) are inherently unpredictable, and predicting in representation space lets the model drop such detail. The cost is a shortcut: if the encoder outputs the same constant for every input, prediction error is zero. That is collapse (as defined in I-JEPA §2).

Three families of fixes: VICReg's variance and covariance constraints; the EMA target encoder plus stop-gradient used by I-JEPA and V-JEPA 2, which works empirically even though no loss term explicitly forbids collapse; and LeJEPA's SIGReg (November 2025), which requires random 1-D projections of the embeddings to look standard normal. V-JEPA 2-AC adds an action-conditioned predictor and plans by using CEM to minimise the L1 distance from the predicted embedding to a goal-image embedding. See [JEPA](jepa.en.md).

</details>

<details class="interview" markdown="1">
<summary>How does Genie learn controllability from video without action labels, and where are the limits?</summary>

The latent action model (LAM) encoder sees the past and the next frame and outputs a continuous vector, which is quantised to a codebook of only 8 codes; the decoder sees only the history and that code yet must reconstruct the next frame, so the code is forced to carry "the main change". The dynamics model predicts the next frame's tokens conditioned on these codes; at inference the encoder is dropped and the player picks 0–7 directly. Controllability is measured with $\Delta_t$PSNR: generate with inferred actions and with random actions, and compare which is closer to the real frame.

Limits: what each code means is learned only by people playing with it afterwards; 8 codes are coarse, so "push 3.5 cm" can sit right on the boundary between two codes; the objective does not distinguish whether the agent caused the change (this point is an inference); the original model has 16 frames of memory and runs at about 1 FPS. DreamGen uses a latent action model or an IDM to add pseudo-actions to robot video, the same idea. See [interactive video](interactive-video.en.md).

</details>

<details class="interview" markdown="1">
<summary>What fundamentally separates interactive world models from video generators like Sora and Veo, and what should you ask of a demo?</summary>

An interactive model must deliver frame $t$ before it knows the next key press, so it can only generate causally. At 24 fps a frame has about 41.7 ms, so with $K$ sampling steps per frame each step gets at most about $41.7/K$ ms, which is why open systems commonly distil to 2–4 steps; it also has to remember out-of-view content across a minute-scale session. An offline generator receives the whole prompt and then delivers the whole clip.

Ask of a demo: on what hardware frame rate and resolution were measured, the latency from key press to response, how long it stays consistent, what controls exist, whether out-of-view content is still there and still changing (WRBench and WorldMemBench report frequent failures here), and how open it is. As of 2026-10-15, Genie 3 has no paper and no API, and its architecture is unpublished.

</details>

<details class="interview" markdown="1">
<summary>A world-model evaluator has Pearson 0.86 with real success rates. Can you use it directly to pick the policy to ship?</summary>

Not directly. Pearson measures linear association across this set of policies and is unchanged under $\hat R=aR+b$ with $a>0$, so absolute success rates can be low across the board (as the Veo evaluator paper says of itself) and cannot decide whether a bar is cleared; it is also often propped up by the large gap between weak and strong policies. In the teaching example, Pearson is 0.98, Spearman 0.9 and MMRV 0.04, yet the model's top pick is still wrong; restricted to the top three, Spearman drops to 0.5. Also, the 0.86 in the Veo paper was computed for one policy across a few OOD axes.

In practice: report ranking metrics together with "was the top policy picked correctly", hold out policies that never took part in selection, and spot-check on real robots regularly so policies cannot specialise in exploiting the evaluator. See [robots and cars](robotics-driving.en.md).

</details>

<details class="interview" markdown="1">
<summary>A probe reads "the board state" out of a model. Does that show the model has a world model?</summary>

It shows only that the information is readable. Also check: whether intervening along the probe direction changes the output as expected (Othello-GPT did this); whether behaviour is consistent, since in Vafa et al. 2024 the taxi model has about 1.00 next-step validity but only 0.10–0.50 sequence-compression precision, and valid routes drop from 0.99 to 0.08 under 10% detours; whether it applies the right law out of distribution, since the orbit model exceeds 0.9999 trajectory $R^2$ yet infers non-Newtonian force laws; and whether outcome distributions are calibrated.

The Richens et al. theorem says a world model can be extracted from the policy of any agent competent at multi-step goals; but it assumes full observability, and it does not show that the model is used. See [do they actually understand the world?](do-they-understand.en.md).

</details>

<details class="interview" markdown="1">
<summary>Tomorrow you read about a new "world-model breakthrough". How do you judge it?</summary>

Ask in order: which kind of world model is it, does it output observations, state or actions, and does it take actions; who measured it and when, whether it is a paper with methods, a blog-only announcement, a vendor-reported number or second-hand coverage; whether the comparison conditions match, including hardware, number of samples and prompts, as with the top Physics-IQ Verified leaderboard entries that use best-of-N and custom prompts; and how long it holds up, including whether anyone tested that out-of-view content keeps changing.

Then ask whether it actually helps in a downstream closed loop rather than just looking good (WorldArena's perception–functionality gap); whether it can be reproduced, with a paper, weights or an API; and what is unpublished, as with Genie 3 and GAIA-4, neither of which has a public architecture. When you repeat the claim, phrase it as "as of a given date, self-reported by a given organisation", not as a permanent conclusion.

</details>

## Self-check

<div class="taste-check">
  <strong>Try explaining these in your own words:</strong>
  <ol>
    <li>why an interactive video model can only generate causally, and how many milliseconds each step gets at most at 24 fps with 4 sampling steps per frame;</li>
    <li>where the gradients of $L_{\rm dyn}$ and $L_{\rm rep}$ go in the RSSM, and what free bits do when the KL is below 1 nat;</li>
    <li>what the cheapest "solution" is once a JEPA drops the EMA target encoder and stop-gradient, and why SIGReg penalises it;</li>
    <li>why, in the five-policy example, a Pearson of 0.98 still picks the wrong top policy, and what Spearman becomes when only the top three are compared;</li>
    <li>with a real fall rate of 30%, what the expected Brier scores are for a model that always says "falls" and one that always says "stays", and why a single video cannot tell them apart.</li>
  </ol>
</div>

## References

- [World Models](https://arxiv.org/abs/1803.10122): Ha & Schmidhuber 2018, training a controller inside the model's "dream"
- [DreamerV3](https://arxiv.org/abs/2301.04104): RSSM and imagination training, one configuration across 150+ tasks
- [Dreamer 4](https://arxiv.org/abs/2509.24527): a video world model plus offline imagination training
- [MuZero](https://arxiv.org/abs/1911.08265): a model that predicts only reward, value and policy (for contrast)
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf): LeCun 2022, JEPA and energy-based planning
- [I-JEPA](https://arxiv.org/abs/2301.08243): image JEPA with an EMA target encoder
- [V-JEPA 2](https://arxiv.org/abs/2506.09985): includes V-JEPA 2-AC planning in embedding space
- [LeJEPA](https://arxiv.org/abs/2511.08544): preventing collapse with SIGReg
- [Genie](https://arxiv.org/abs/2402.15391): learning latent actions from unlabelled video
- [Genie 3 blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/): as of October 2026 only an official blog, no paper
- [GameNGen](https://arxiv.org/abs/2408.14837): a diffusion model as a real-time DOOM engine
- [Cosmos World Foundation Model Platform](https://arxiv.org/abs/2501.03575): NVIDIA's world foundation model platform
- [DreamGen](https://arxiv.org/abs/2505.12705): robot training data generated by a video world model
- [Evaluating Gemini Robotics Policies in a Veo World Simulator](https://arxiv.org/abs/2512.10675): the Veo evaluator, Pearson and MMRV
- [DreamZero](https://arxiv.org/abs/2602.15922): a world action model used directly as the policy
- [Cosmos Policy](https://arxiv.org/abs/2601.16163): a video model post-trained into a policy
- [Emergent World Representations (Othello-GPT)](https://arxiv.org/abs/2210.13382): probes and interventions
- [Evaluating the World Model Implicit in a Generative Model](https://arxiv.org/abs/2406.03689): Myhill–Nerode-style consistency metrics
- [What Has a Foundation Model Found?](https://arxiv.org/abs/2507.06952): inductive-bias probes and the orbit model
- [General agents contain world models](https://arxiv.org/abs/2506.01622): competence implies an extractable world model
- [Physics-IQ](https://arxiv.org/abs/2501.09038) and [Physics-IQ Verified](https://arxiv.org/abs/2606.18943): a real-video physics benchmark and its audited version
- [WorldArena](https://arxiv.org/abs/2602.08971): testing the three uses of embodied world models separately
- [WRBench](https://arxiv.org/abs/2606.20545) and [WorldMemBench](https://arxiv.org/abs/2610.19063): memory and advancing state
- [A Definition and Roadmap for World Models](https://arxiv.org/abs/2607.06401): a 2026 definition and roadmap
- [A Functional Taxonomy of World Models](https://www.worldlabs.ai/blog/taxonomy-of-world-models): renderers / simulators / planners
