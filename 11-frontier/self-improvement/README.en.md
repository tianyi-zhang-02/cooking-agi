# Recursive self-improvement: first ask what changes

[中文](README.md) · **English**

> Reading time: ~7 min · Last reviewed: 2026-10

A model retrains on its correct answers, an agent edits its own code, or researchers use AI to run experiments faster. All three may be called “self-improvement,” though they automate different parts of the work and require different evidence of success.

To understand a system, follow one proposed change. What changes? Who proposes and implements it? Who decides to keep it, and based on what result? We use these questions to compare the approaches without treating them all as systems that build their own successors.

## Where the idea comes from

In a 1965 paper (the volume is often dated 1966), I. J. Good imagined that a machine smarter than people could design even better machines, setting off an "intelligence explosion" ([DOI](https://doi.org/10.1016/S0065-2458%2808%2960418-0)). Schmidhuber's 2003 [Gödel Machine](https://arxiv.org/abs/cs/0309048) put a strict gate on self-rewriting: the system changes its own code only after it finds a formal proof that the rewrite helps under a given utility function. In [LOGI (2007)](https://intelligence.org/files/LOGI.pdf), Yudkowsky described a seed AI as a system designed to understand and modify itself, and stressed that recursive self-improvement is a consequence of seed AI, not a cheap way to build AI.

In 2026, Favaro and Clark of the Anthropic Institute gave a more concrete definition in [When AI builds itself](https://www.anthropic.com/institute/recursive-self-improvement): "an AI system capable of fully autonomously designing and developing its own successor". What these accounts share is that the improvement acts on the thing doing the improving, so the next round might go faster. Where they differ is how far the loop has actually closed today.

## Four things people call "self-improvement"

The split below is an operational definition for teaching (each example is cited). It is not an industry standard.

| Type | Object that changes | Who proposes / applies the change | Signal that accepts the change | Verified examples |
| --- | --- | --- | --- | --- |
| (a) Model improves its own weights | Model parameters, via a training procedure the model itself designs or runs | The model writes data pipelines, losses, training code, and launches runs | Evals defined by the model or its operators | **No public demonstration of the full form found.** [DGM](https://arxiv.org/abs/2505.22954) lists rewriting its own training script to update the foundation model as future work. Partial precursors: [PostTrainBench](https://arxiv.org/abs/2603.08640) has agents post-train a different base model; [autoresearch](https://github.com/karpathy/autoresearch) has an agent edit a small model's `train.py` |
| (b) System improves its scaffold, code, or prompts | Agent code, tools, prompts, memory, and workflows; weights frozen | The agent edits its own repository | Benchmark score, often on held-out tasks | DGM, SICA, Huxley-Gödel Machine, AIDE² |
| (c) AI accelerates AI research | The human-led research process and its artifacts: kernels, schedulers, architectures, experiments, papers | Agents do parts of the work; humans set direction and gate deployment | Experiment metrics, human review, production telemetry | AlphaEvolve, ASI-Arch, AI Scientist, Anthropic's automated alignment researchers, OpenAI's "research intern" metrics |
| (d) Self-generated training data and self-play | Model weights, but through a fixed update algorithm designed by people | The model supplies data, rationales, or rewards | Ground-truth answers, code execution, or the model's own judgement | STaR, ReST-EM, Self-Rewarding LMs, Absolute Zero, R-Zero |

Both (a) and (d) change weights. The difference is who designs the update: in (d) it is a human-written "filter, then fine-tune" or DPO/RL loop; in (a) it would have to be the model. Everyday use of "RSI" usually means (a), or (c) pushed until humans are out of the loop, while most published systems labelled "self-improving" are (b) or (d). A July 2026 [survey](https://arxiv.org/abs/2607.07663) by Chen, Wang and Qu counted 1,250 papers and reports that nearly all of them study bounded self-refinement, not closed loops.

<!-- paper-figure:rsi-survey-taxonomy -->

This series uses one teaching example throughout: **a model that writes Python functions, plus a test suite that accepts or rejects them.** "Generate solutions, keep the ones the tests accept, fine-tune on them" is type (d). An agent rewriting its own code for calling tests and splitting tasks is type (b). Agents helping researchers tune training kernels are type (c). Only when the model itself decides how to build data, change the loss and when to retrain itself is it type (a). Once the tests start accepting wrong solutions, every one of these loops is affected; the later notes keep coming back to that point.

## Where things stand as of October 2026

- **Self-reported acceleration in labs.** On 2026-09-06 OpenAI [announced](https://openai.com/index/research-acceleration-view-inside-openai/) that it had met its self-set "automated research intern" goal, self-reporting 3.1 agent-workdays per human workday in its research organisation in mid-August 2026; the same post states that over half of successful 4-8 hour tasks involved at least one human intervention, and that available compute also grew significantly. Anthropic's August 2026 [risk report](https://www.anthropic.com/aug-2026-risk-report) self-reports that Claude authors most of the code merged into its production codebases and that internal R&D is clearly faster, but not yet by a factor of 2. Google DeepMind's 2026-10-13 [Gemini 4 Argon safety report](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_safety_report.pdf) finds, by its own measurement, no significant end-to-end acceleration of AI R&D.
- **Determinations against formal thresholds.** As of 2026-10-15, no lab has declared a model over its formal AI-R&D or self-improvement threshold. The wording has grown more cautious: Anthropic said of Claude Opus 4.5 (2025-11) that ruling out the threshold was becoming increasingly difficult, and of Claude Mythos Preview (2026-04-07) that it held the verdict with less confidence than for any prior model. Details are in [note 3](measuring.en.md).
- **Checkable results in published research.** Agents that rewrite their own scaffolds score much higher with frozen weights (DGM, 2025-05: SWE-bench 20.0% → 50.0%). Evolutionary code search improved the infrastructure that trains the next model (AlphaEvolve, 2025-05: 1% less Gemini training time). Training on self-generated data improves weights but saturates within a few rounds without an external verifier.
- **What does not exist yet.** No public source documents a system that autonomously designs and trains its own successor. The Anthropic Institute essay says we are not there yet; OpenAI's 2026-09-06 post says it does not yet know how to get safely all the way to aligned, full RSI.

What follows are statements and predictions, not measurements. Dario Amodei wrote in January 2026 that we may be only 1-2 years from the current generation of AI autonomously building the next ([essay](https://darioamodei.com/essay/the-adolescence-of-technology)). Jack Clark put the chance of no-human-involved AI R&D by the end of 2028 at 60% or more on 2026-05-04 ([Import AI 455](https://importai.substack.com/p/import-ai-455-automating-ai-research)). OpenAI's Chris Lehane said on 2026-09-09 that fully autonomous recursive self-improvement is not happening today and should not be pursued until it can be done safely ([post](https://openai.com/index/ai-policy-window/)).

## Timeline

| Date | Event | Org / author | One line |
| --- | --- | --- | --- |
| 1965/66 | [Speculations Concerning the First Ultraintelligent Machine](https://doi.org/10.1016/S0065-2458%2808%2960418-0) | I. J. Good | Coins "intelligence explosion" |
| 2003-09 | [Gödel Machine](https://arxiv.org/abs/cs/0309048) | Schmidhuber | Rewrites itself only after proving the rewrite helps |
| 2022-03 | [STaR](https://arxiv.org/abs/2203.14465) | Stanford / Google Research | Fine-tunes on its own correct rationales |
| 2024-07 | [Model collapse](https://www.nature.com/articles/s41586-024-07566-y) (*Nature*) | Shumailov et al. | Repeated training on generated data makes distribution tails vanish |
| 2024-12 | [Mind the Gap](https://arxiv.org/abs/2412.02674) | Song et al. | The generation-verification gap governs self-improvement; saturates in 2-3 rounds |
| 2025-03 | [Time-horizon paper](https://arxiv.org/abs/2503.14499) | METR | 50% time horizon doubling about every 7 months since 2019 |
| 2025-05-14 | [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | Google DeepMind | Evolves code; 1% less Gemini training time |
| 2025-05-29 | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | UBC / Vector / Sakana AI | Agents edit their own code; SWE-bench 20.0% → 50.0% |
| 2026-01-29 | [Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) | METR | 228 tasks; post-2023 doubling 131 days |
| 2026-05-19 | [Frontier Risk Report](https://metr.org/blog/2026-05-19-frontier-risk-report/) | METR | TH1.1 suite essentially saturated |
| 2026-08-18 | [Pacing model development](https://openai.com/index/pacing-model-development-cyber-capabilities/) | OpenAI | After agents in July internal cyber evaluations escaped isolation, two-week pause in RL training on deployment-track models |
| 2026-09-06 | [Research acceleration](https://openai.com/index/research-acceleration-view-inside-openai/) | OpenAI | Declares the "research intern" goal met (self-reported) |
| 2026-09-22 | [AIDE²](https://arxiv.org/abs/2609.26457) | Weco AI | Research agent rewrites its own code; 7 accepted improvements in 8 days |
| 2026-10-13 | [Gemini 4 Argon safety report](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_safety_report.pdf) | Google DeepMind | ML R&D thresholds not reached; no significant end-to-end acceleration |

## What has settled and what is still moving

Each cell points to a dated public record. The "signals" column says what kind of result would change the picture; it is not a prediction that it will happen.

| Settled | Still moving | Signals to watch next |
| --- | --- | --- |
| Without an external verifier, training on your own data saturates within a few rounds: Song et al. (2024-12) measured the gap falling to near zero in two or three rounds; ReST-EM (2023-12) saw diminishing returns after a couple of iterations | How far self-proposed, self-checked tasks with a code executor can go: Absolute Zero (2025-05) and R-Zero (2025-08) report gains relative to the base model only | Self-training results that keep improving on independent benchmarks well beyond three rounds |
| Editing only the scaffold, with frozen weights, can raise benchmark scores a lot: DGM (2025-05) SWE-bench 20.0% → 50.0% | Whether such gains compound: HGM (2025-10) found benchmark score predicts descendants' performance poorly; in AIDE² (2026-09-22), a discovered agent used as the outer-loop optimiser could not be decisively distinguished from the baseline | A discovered agent that, across several seeds, clearly improves agents better than the agent that created it |
| METR's task-level time horizon has risen for years: about a 7-month doubling in the 2025-03 paper, 131 days post-2023 under TH1.1 (2026-01-29) | The measuring tools themselves: on 2026-05-19 METR called TH1.1 essentially saturated, readings above 16 h are unreliable, and its page has not been actively updated since 2026-09-08; OpenAI replaced its self-improvement eval suite on 2026-07-09 because the old evals saturated | METR or another group publishing a suite of longer tasks with cheating review |
| As of 2026-10-15, no lab has declared its formal AI-R&D threshold crossed: Anthropic Claude Opus 5.5 (2026-09-22), OpenAI GPT-6 October update (2026-10-07), Google DeepMind Gemini 4 Argon (2026-10-13) | How labs measure their own R&D acceleration: OpenAI 3.1 agent-workdays (2026-09-06, self-reported), Anthropic "not yet by a factor of 2" (2026-08, self-reported), a non-public ~1.5x estimate quoted by METR (2026-09-22) | A lab declaring a formal threshold crossed; public data from the RSI reporting to CAISI that OpenAI's 2026-06 policy blueprint proposes |
| No public demonstration of a type (a) loop: as of 2026-10-15, no public record of a system autonomously designing and training its successor | Pacing and governance: OpenAI paused some RL training for two weeks on 2026-08-18; Amodei proposed a speed limit on RSI in 2026-09 | A public, checkable demonstration of a system that designs, trains and evaluates its own successor |

## How to read this series

For background: basic RL notation is in [Deep RL](../../05-post-training/deep-rl/README.en.md), how to judge that something "got better" is in [Evaluation](../../07-evaluation/README.en.md), and the structure of an agent is in [Agents](../../10-agents/README.en.md).

1. [Generate, verify, retrain: why the verifier sets the ceiling](generate-verify-train.en.md): STaR, ReST-EM, Self-Rewarding, Absolute Zero, and how much a loop learns when the tests accept wrong solutions
2. [Changing code, not weights](code-not-weights.en.md): how AlphaEvolve and the Darwin Gödel Machine improve programs with frozen weights, and what is missing for "a model changes its own weights"
3. [How to measure "AI speeding up AI"](measuring.en.md): METR time horizons, AI R&D benchmarks, lab thresholds and determinations, and how to read self-reported numbers
4. [Review questions](review.en.md): interview questions and a self-check
