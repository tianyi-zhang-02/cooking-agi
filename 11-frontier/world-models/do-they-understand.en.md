# World models: do they actually understand the world?

[中文](do-they-understand.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

A video model renders a cup sliding off a table convincingly; a language model answers that "a 5 cm push will knock the cup off". Do they "have a world model"? The question has been argued for years, largely because people mean different things by "have". **Being able to read state out of the model, changing that state and seeing the output change, behaving consistently across rephrasings and new situations, and predicting outcome distributions that match the real world are evidence of different strength. So far, most positive evidence stops at "readable", while tests of behavioural consistency and distributional calibration often fail.**

## First, what "having a world model" means

[Chen et al.](https://arxiv.org/abs/2607.06401), in a July 2026 perspective paper, say plainly in the abstract that "there is no consensus on what a world model fundamentally is". What several primary sources say:

| Source | What it says a world model is |
| --- | --- |
| [Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122) | A generative model of an RL environment that learns a compressed spatial and temporal representation; an agent can be trained entirely inside its "dream" and transferred back |
| [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf) | One of six modules of an autonomous agent: estimates missing state information, predicts plausible futures (possibly conditioned on actions), and predicts in an abstract representation space |
| [Genie 3 blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) (August 2025) | Uses an understanding of the world to simulate aspects of it, so agents can predict how an environment evolves and how their actions affect it |
| Chen et al. (July 2026) | An internal simulator that learns an environment's structure and dynamics; proposes a scientific definition and a staged roadmap |
| [World Labs functional taxonomy](https://www.worldlabs.ai/blog/taxonomy-of-world-models) (2026-06-03) | Sorted by output: renderers output observations (Genie 3, RTFM), simulators output state such as geometry and physics, planners output actions (VLAs, world action models); framed as a POMDP, and traces the term to Craik (1943) |

Some are sceptical of the term itself. Melanie Mitchell argues it is ill-defined and that evidence for abstract world models in LLMs is weak ([blog](https://aiguide.substack.com/p/llms-and-world-models-part-1)); Gary Marcus argues LLMs lack robust explicit world models ([blog](https://garymarcus.substack.com/p/generative-ais-crippling-and-widespread)); Xing et al. (MBZUAI / CMU, July 2025, [arXiv](https://arxiv.org/abs/2507.05169)) argue that video generators alone fall short of supporting reasoning and acting, and propose a different architecture.

To discuss the evidence separately, this note uses a working framework. It is our way of organising things, not a field standard:

1. **Readable**: environment state can be decoded from inside the model, such as how far the cup is from the edge.
2. **Used**: changing that internal representation changes the output as expected.
3. **Behaviourally consistent**: when two histories lead to the same real state, the model predicts the same future; when they lead to different states, it tells them apart.
4. **Distributionally calibrated**: from the same start and the same action, the model's outcome distribution matches real frequencies.

## A theorem: competence at multi-step goals implies a world model inside the policy

[Richens et al.](https://arxiv.org/abs/2506.01622) (Google DeepMind, ICML 2025) titled v1 "...need world models"; it is now "General agents contain world models". The abstract's central claim is that "any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model …" of its environment. We read the paper; the conditions are:

- **Environment**: a controlled Markov process, i.e. an MDP without rewards; finite, communicating, stationary, and fully observed by the agent (§2.2, §2.4).
- **Agent**: a goal-conditioned policy whose success probability on every composite goal of depth at most $n$ is at least $(1-\delta)$ times optimal (§2.4). Only competence is assumed, not rationality.
- **Conclusion**: an approximate transition model can be extracted from the policy alone; the smaller $\delta$ and the larger $n$, the tighter the error bound (§3, Theorem 1). Optimal agents that pursue only one-step goals need not learn transition probabilities (Theorem 2).
- **Boundaries the authors state** (§4): the theorem proves that a world model is encoded in the policy, not how it is used (for example, for planning); it covers only fully observed environments.

In the cup example: from the choices of an agent that reliably achieves arbitrary combinations of goals like "push the cup to 1 cm from the edge, then pull it back to the centre", you could back out how far each push moves the cup and how likely the cup is to fall. But the theorem does not say the agent "thought with" that model, and it does not cover camera-only settings where state is not fully visible, which is exactly where video models live.

## Probes say "yes"

A probe is a small classifier or regressor trained on a frozen model's activations to see whether some environment variable can be read out. Representative results:

- [Othello-GPT](https://arxiv.org/abs/2210.13382) (Li et al., October 2022, ICLR 2023): a GPT trained only on move sequences; nonlinear probes read out the board state with a best error of 1.7%, and intervening along the probe direction changes the model's next-move predictions. [Nanda et al.](https://arxiv.org/abs/2309.00941) further found that in "mine / theirs" coordinates the board is linearly readable (99.6%).
- [Gurnee & Tegmark](https://arxiv.org/abs/2310.02207) (October 2023): Llama-2-70B linearly encodes geographic coordinates ($R^2$ 0.911) and historical time (0.835).
- [Joseph et al.](https://arxiv.org/abs/2602.07050) (February 2026, ICML 2026): in V-JEPA 2 and VideoMAE-v2, physics variables become decodable abruptly at about one-third depth, which the authors call a Physics Emergence Zone.
- [Garrido et al.](https://arxiv.org/abs/2502.11831) (February 2025, with LeCun among the authors): on violation-of-expectation tests, representation-space predictors (V-JEPA) show intuitive physics, while pixel-space generators and multimodal LLMs sit near chance.

But [Punzo et al.](https://arxiv.org/abs/2606.09646) (June 2026), comparing V-JEPA, VideoMAE and a diffusion generator, conclude that what can be decoded depends more on the benchmark and the probe than on the pretraining objective.

### Worked example: readable is not the same as used (a teaching example)

First, write down a linear probe and an intervention along the probe direction:

$$
\hat y=w^\top h+b,\qquad h'=h+\alpha\,\frac{w}{\|w\|}
$$

$h$ is the activation at some layer, $w,b$ are the probe parameters, and $\hat y$ is the variable read out (say, the cup's distance to the edge). The second expression pushes the activation by $\alpha$ along the probe direction to see whether the model's output changes.

Imagine a model that watches cup-pushing videos and predicts "falls or not" (all numbers invented). Inside it are two features: $h_1$ encodes the cup's distance to the edge, because rendering the next frame requires knowing where the cup is; $h_2$ encodes how far this push goes. In the training videos the cup always starts 4 cm from the edge (a teaching assumption), so the rule "falls if the push is at least 4 cm", which looks only at $h_2$, is 100% correct on the training set, and the model's "falls or not" head uses only that.

- **Probe**: reading distance from $h_1$ on held-out data gives $R^2=0.97$. It looks as if "the model knows where the cup is".
- **Intervention**: setting $h_1$ to "1 cm from the edge" in 200 test videos changes the "falls or not" output 0 times; changing $h_2$ changes it 180 times (90%).
- **Behaviour**: start the cup 1 cm from the edge and push 2 cm. In the real world it falls; the model says it does not, because 2 < 4.

Readable (level 1) holds, used (level 2) does not, and behavioural consistency (level 3) fails at a new starting point. Othello-GPT did run interventions, which goes further than probing alone; but some interpretability work argues it may be running "a bag of heuristics" rather than one coherent algorithm, and that [Alignment Forum post](https://www.alignmentforum.org/posts/gcpNuEZnxAPayaKBY) is not peer reviewed.

<!-- paper-figure:othello-intervention -->

## Behavioural tests say "often inconsistent"

[Vafa et al. 2024](https://arxiv.org/abs/2406.03689) (NeurIPS 2024) borrow the idea of the Myhill–Nerode theorem: in a deterministic finite automaton, two histories that reach the same state must allow exactly the same continuations; histories that reach different states must be distinguishable by some continuation. That gives two tests: does the model treat different histories of the same state the same way (compression), and can it tell different states apart (distinction)?

They trained Transformers on Manhattan taxi routes: next-step validity is about 1.00, yet sequence-compression precision is only 0.10–0.50; a shortest-path model facing 10% random detours drops from 0.99 to 0.08 valid routes.

<!-- paper-figure:vafa-manhattan-maps -->

Now with the cup (a teaching setup that assumes no momentum and uniform friction): history (a) pushes 3 cm and then 2 cm, history (b) pushes 5 cm once. Both leave the cup in the same place, so reasonable predictions afterwards should match. If the model says the cup is still on the table after (a) but falls after (b), it is tracking "how far the last push went", not the cup.

More results of the same kind:

- [Vafa et al. 2025](https://arxiv.org/abs/2507.06952) (ICML 2025): a Transformer trained on planetary orbits predicts trajectories with $R^2$ above 0.9999; but when an inductive-bias probe makes it infer forces, each sample yields a different, non-Newtonian force law.
- [Kang et al.](https://arxiv.org/abs/2411.02385) (November 2024): in 2D physics, out-of-distribution failure does not improve with scale, and models seem to generalise "case by case" from the nearest training example.
- [Song & Cai](https://arxiv.org/abs/2606.31399) (June 2026): LLM agents' state fidelity collapses abruptly once state load and horizon pass a threshold.
- [AutumnBench / WorldTest](https://arxiv.org/abs/2510.19788) (October 2025): learning world models through interaction in 43 grid worlds, humans are far ahead of frontier models.

## Video models sit a physics exam

**Physics-IQ.** [Motamed et al.](https://arxiv.org/abs/2501.09038) (Google DeepMind and INSAIT, January 2025) built a physics benchmark from real video; the best score in the original paper was 29.5% (VideoPoet, multiframe), and they found physical understanding unrelated to visual realism. [Physics-IQ Verified](https://arxiv.org/abs/2606.18943) (Rädsch et al., 2026-06-17) audited the benchmark, refined 57.6% of samples, moved to sample-level scoring, and saw model rankings change moderately (Kendall τ = 0.46).

The [Physics-IQ Verified leaderboard](https://github.com/google-deepmind/physics-IQ-benchmark) sits in Google DeepMind's repository, but its README says it is an independent third-party benchmark not endorsed by Google DeepMind. Read on 2026-10-15: the top entry is Odyssey-3 Pro with best-of-8 at 66.10 (video-to-video, self-reported by Odyssey); among frontier image-to-video models, Sora 2 scores 26.5, Veo 3.1 Fast 29.96 and Seedance 2.5 42.43; Cosmos3-Super (multiframe) scores 50.8 ± 2.2. Entries arrive by pull request with a report or reproducible code; some use best-of-N and custom prompts, and input modes differ, so they are not like-for-like.

How much can best-of-N inflate a score? A teaching calculation: if a single generation is "physically right" with probability 0.3, generations are independent, and a judge always picks the right one when present, then best-of-8 succeeds with probability $1-0.7^8\approx0.94$. Real Physics-IQ scores are not binary and selection is not perfect, but the direction is the same: comparisons are fair only at the same N and the same prompting policy.

**Other angles.** Google DeepMind's [Wiedemer et al.](https://arxiv.org/abs/2509.20328) (September 2025) had Veo 3 attempt 62 qualitative and 7 quantitative vision tasks zero-shot, solving 5×5 mazes 78% of the time at pass@10 (Veo 2: 14%); the authors warn that the API's LLM prompt rewriter may solve some tasks for the model and that results are highly prompt-sensitive. In 2026, benchmarks moved from "does it look real" to "is it useful": [WorldArena](https://arxiv.org/abs/2602.08971) (February 2026) found a gap between visual quality and embodied usefulness; the memory problems tested by WRBench (June 2026) and WorldMemBench (October 2026) are covered in the [interactive video note](interactive-video.en.md).

### Worked example: the cup falls 30% of the time, but the model says 0% or 100% (a teaching example)

Push 5 cm; friction varies slightly, and in real trials the cup falls 30% of the time. Three models each generate with many random seeds: A always makes the cup fall, B never does, C has it fall about 30% of the time. Every single video from A or B looks reasonable on its own, and a reviewer who sees one video at a time cannot tell them apart.

Compare them with the expected Brier score. $Y=1$ means the cup fell in a real trial, $q=0.3$ is the real fall rate, and $p$ is the fraction of the model's generations in which the cup falls:

$$
\mathbb E\big[(p-Y)^2\big]=q\,(1-p)^2+(1-q)\,p^2
$$

A ($p=1$): $0.3\times0+0.7\times1=0.70$. B ($p=0$): $0.3\times1+0=0.30$. C ($p=0.3$): $0.3\times0.49+0.7\times0.09=0.21$, the best. B beats A only because falling is the rarer outcome. Use B to evaluate a cup-pushing robot policy and it will never report "dangerous".

[CaliBench](https://odyssey.systems/introducing-calibench), released by Odyssey on 2026-08-10, tests exactly this: it reports that six frontier image-to-video models mostly mode-collapse, and that Veo 3.1 rolled the same die face on every seed. This is a vendor-built benchmark from a company that also builds world models, so the finding needs third-party replication. OpenAI's Sora 2 launch page (2025-09-30) likewise argued that a useful simulator must model failure.

## What each kind of evidence shows

| Evidence type | Shows | Cannot show | Examples |
| --- | --- | --- | --- |
| Linear or nonlinear probe | The information can be read out of the model | That the output uses it; whether the readout comes from the model or the probe itself | Othello-GPT, Llama-2 space and time, V-JEPA 2 physics variables |
| Intervention along the probe direction | This representation affects the output | That the whole is one coherent algorithm rather than a bag of heuristics | Othello-GPT interventions |
| Next-token or next-frame accuracy | Accurate prediction on the training distribution | Coherent internal state; validity out of distribution | Taxi models with about 1.00 next-step validity |
| Myhill–Nerode-style consistency tests | Whether different histories of the same state are treated alike | How exactly the model represents state inside | Vafa et al. 2024 |
| Inductive-bias probes, out-of-distribution tests | Which law the model uses when transferring to a new task | Whether more scale would change it | Orbit model, Kang et al. |
| Single-sample physics benchmark | How physically consistent outputs are with real video | Whether outcome distributions are calibrated; whether a best-of-N score equals single-shot ability | Physics-IQ / Verified |
| Multi-seed distribution test | Whether the outcome distribution matches real frequencies | Whether each individual outcome is right in detail | CaliBench (vendor-built) |
| Closed-loop use | Whether it actually helps when used for planning or evaluation | What it "understands" | WorldArena, World-in-World |
| Theorem | Under stated conditions, competence implies an extractable model | That the model is used; partially observed settings | Richens et al. |

## What to watch on this line next

1. **Will definitions converge?** Chen et al. (July 2026) proposed a definition and roadmap; World Labs (June 2026) proposed the renderer / simulator / planner taxonomy. A result that would change the picture: benchmarks that adopt one definition and report "readable, used, behaviourally consistent, calibrated" level by level.
2. **Will any video model score high on Physics-IQ Verified without best-of-N and custom prompts?** As of 2026-10-15, the top leaderboard entries rely on both. A result that would change the picture: a high single-generation score under uniform prompts, checked by the benchmark maintainers.
3. **Do models actually use the physics variables that probes find?** Joseph et al. (February 2026) found decodable physics variables in V-JEPA 2; Punzo et al. (June 2026) argue decodability depends more on the probe and benchmark. A result that would change the picture: causal interventions on video world models in which changing those variables shifts predictions in physically correct ways.
4. **Can distributional calibration be replicated independently?** CaliBench (Odyssey, August 2026) is a vendor-built benchmark. A result that would change the picture: third-party multi-seed distribution tests covering more models and physical scenarios.

Finally, use the [review questions](review.en.md) to tie the whole series together.
