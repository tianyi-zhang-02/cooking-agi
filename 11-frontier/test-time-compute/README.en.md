# Reasoning models and test-time compute: when thinking longer helps

[中文](README.md) · **English**

> Reading time: ~7 min · Last reviewed: 2026-10

For the same question, a model can generate one answer, produce 64 candidates and vote, or use a longer reasoning process to check and revise its answer. Each option uses more computation, but accuracy does not necessarily improve.

With multiple candidates, the model needs to produce a correct answer and the selection method needs to recognize it. With a longer reasoning chain, later steps may correct an error, repeat earlier work, or overturn a correct answer. Reasoning models since 2024 have made these capabilities a training target and exposed inference budgets as settings. We examine where the extra computation goes and when it helps.

## What test-time compute means

Once training is done the weights stay fixed; any extra computation spent answering a question is test-time compute (also called inference-time compute). Generating one token costs roughly $2N$ floating-point operations ($N$ is the parameter count, attention not included), so extra compute is mostly extra generated tokens; [Snell et al. (2024-08)](https://arxiv.org/abs/2408.03314) count it as $2N$ times the number of inference tokens. There are four ways to spend it, and real systems often combine them:

| Way to spend | Where the extra compute goes | What it needs | Examples | Typical failure |
| --- | --- | --- | --- | --- |
| Sample many, then pick | N independent answers to the same question | Answers that can be compared, or a verifier | Self-consistency voting, best-of-N | The right answer is not the most common one; the verifier accepts wrong answers |
| Think longer | Checking, backtracking and switching methods inside one chain | A model trained to use long chains | o1, DeepSeek-R1 | Overthinking easy problems, hopping between ideas on hard ones |
| Search over intermediate steps | Score each step, keep only promising prefixes | A process reward model or value estimate | Beam search, Tree of Thoughts | The search finds the scorer's blind spots |
| Generate, check, revise | A program or model checks, then the answer is revised | A checker, or a model that checks itself | Unit tests, Lean, proof verifiers | The specification being checked differs from the question you care about |

<!-- paper-figure:snell-search-methods -->

## Two axes: parallel and sequential

Parallel means generating N independent answers at once and picking one by vote or verifier. As long as the server has room, latency barely grows with N while cost grows linearly. Sequential means thinking longer within one chain, or revising each version after looking at the previous one: later steps can use earlier findings, at the price of latency that grows with length.

On PaLM 2-S* and the MATH dataset, Snell et al. found that the effectiveness of each approach "critically varies depending on the difficulty of the prompt" ([abstract](https://arxiv.org/abs/2408.03314)), and that the two axes suit different problems: easy problems gain more from sequential revision, while on hard problems the best budget is a mix of both. Most systems today use both. DeepSeekMath-V2 (2025-11) starts each problem with 64 proofs, verifies each one 64 times, then revises the high scorers for up to 16 rounds; OpenAI's GPT-5 system card (2025-08) says gpt-5-thinking-pro in ChatGPT uses a "parallel test time compute" setting; Google DeepMind says the Gemini Deep Think that reached a gold-medal score at IMO 2025 explores and combines several candidate solutions at once. The last two did not publish the details.

<!-- paper-figure:snell-parallel-sequential -->

This series uses one teaching example throughout: **a math problem whose answer can be checked, where the model answers correctly with probability $p$ on a single try.** Note 1 asks how many samples to draw and how to pick; note 2 asks how one chain gets trained to be longer and how its length is controlled; note 3 asks how to compare the numbers in release posts.

## Timeline

| Date | Event | Organisation | In one line |
| --- | --- | --- | --- |
| 2021-10 | [Training Verifiers](https://arxiv.org/abs/2110.14168) | OpenAI | GSM8K; generate many answers, pick one with a trained verifier |
| 2022-03 | [Self-consistency](https://arxiv.org/abs/2203.11171) | Google | Sample several reasoning paths, vote on the final answer |
| 2023-05 | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | OpenAI | A process reward model picks answers better than an outcome reward model or voting; PRM800K released |
| 2024-07 | [Large Language Monkeys](https://arxiv.org/abs/2407.21787) | Stanford and others | Coverage grows roughly log-linearly with the number of samples |
| 2024-08 | [Compute-optimal test-time scaling](https://arxiv.org/abs/2408.03314) | UC Berkeley / Google DeepMind | Allocate test-time compute by difficulty; it does not trade 1:1 with more parameters |
| 2024-09-12 | [o1-preview](https://openai.com/index/learning-to-reason-with-llms/) | OpenAI | Long chains trained with RL; self-reported smooth gains with both training compute and thinking time |
| 2024-12-20 | [o3 on ARC-AGI-1](https://arcprize.org/blog/oai-o3-pub-breakthrough) | ARC Prize / OpenAI | 6 samples 75.7%; 1024 samples (about 172x the compute) 87.5% |
| 2025-01-22 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) ([*Nature* version](https://doi.org/10.1038/s41586-025-09422-z) 2025-09-17) | DeepSeek | Open weights and report: RL with rule-based rewards alone makes answers longer and more accurate |
| 2025-02-24 | [Claude 3.7 Sonnet extended thinking](https://www.anthropic.com/news/visible-extended-thinking) | Anthropic | One model can answer fast or think at length; the caller sets a token budget |
| 2025-04-16 | [o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/) | OpenAI | The model decides when to call tools during its reasoning |
| 2025-07-21 | [Gemini Deep Think at the IMO](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) | Google DeepMind | 35/42, graded by IMO coordinators against the student standard |
| 2025-08 | [GPT-5 system card](https://arxiv.org/abs/2601.03267) | OpenAI | A real-time router chooses between a fast model and a thinking model; the pro setting uses parallel test-time compute |
| 2025-11-27 | [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) | DeepSeek | Train a proof verifier, use it as the reward, and scale verification at inference |
| 2026-02-05 | [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) | Anthropic | Adaptive thinking: the model decides whether and how long to think; the caller only sets effort |
| 2026-02-23 | [Stops reporting SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | OpenAI | Flawed tests, and frontier models have seen the problems and fixes |
| 2026-03-25 | [ARC-AGI-3](https://arcprize.org/blog/arc-agi-3-launch) | ARC Prize | Interactive environments; at launch humans 100%, frontier AI 0.51% |
| 2026-04-24 | [DeepSeek V4 preview](https://api-docs.deepseek.com/news/news260424/) | DeepSeek | One model with thinking and non-thinking modes; low / high / max levels added on 2026-08-13 |
| September 2026 | [Navier–Stokes proof](https://openai.com/index/navier-stokes-solution/) | OpenAI | Self-reported: about 10,000 concurrent agents for about 88 hours, with a Lean formalisation; what it settles is disputed ([Scientific American, 2026-09-21](https://www.scientificamerican.com/article/did-openai-solve-the-wrong-navier-stokes-problem/)) |

## What has settled and what is still moving

Each cell points to a dated public record. The "signals" column says what kind of result would change the picture; it is not a prediction that it will happen.

| Settled | Still moving | Signals to watch next |
| --- | --- | --- |
| Sampling more raises coverage, but without an automatic verifier you cannot pick the right one: Brown et al. (2024-07) went from 15.9% with 1 sample to 56% with 250 on SWE-bench Lite (where tests decide correctness directly); voting and reward models stop improving after a few hundred samples | Better ways to pick: DeepConf (2025-08) filters reasoning traces by the model's own confidence and self-reports up to 84.7% fewer generated tokens than full parallel sampling on AIME 2025; DeepSeekMath-V2 (2025-11) trains a dedicated proof verifier | On tasks without reference answers, selection accuracy approaching coverage under independent grading |
| RL with verifiable rewards alone can train long chains, and length grows with training: DeepSeek-R1-Zero (2025-01 report, 2025-09 *Nature* version) | Whether RL widens the capability boundary or only makes existing solutions easier to sample: Yue et al. (2025-04, NeurIPS 2025) versus ProRL (2025-05) and Wen et al. (2025-06); the question-by-question analysis by Wu et al., updated 2026-10-14, still finds more shrinkage than expansion | Results across several model families, with reasoning checked, where pass@k at large $k$ also stays above the base model |
| The benefit of test-time compute depends on problem difficulty and does not trade 1:1 with pretraining: Snell et al. (2024-08) | Who decides how long to think: from 2026-02-05 Anthropic lets the model decide; as of October 2026, the OpenAI, Anthropic and Google APIs all use discrete effort levels as the main control; Fan et al. (2026-08) found that models do not allocate a budget shared across problems by difficulty or point value | Scores on shared-budget evaluations that reward allocating compute by difficulty |
| Written reasoning is not necessarily the computation the model relied on: Turpin et al. (2023-05); Chen et al. (2025-05) measured that Claude 3.7 Sonnet and DeepSeek R1 mention a hint they used only 25% and 39% of the time on average | How long chain-of-thought monitoring stays useful: Baker et al. (2025-03) found that putting a monitor into the reward, under strong optimisation, teaches the agent to hide its intent in the chain; researchers from many organisations jointly called for preserving monitorability in 2025-07 | Labs publishing monitorability measurements in system cards and stating whether training pressure is applied to the chain of thought |
| Old benchmarks are contaminated or saturated: MathArena (2025-05) found most models score 10–20 points higher on AIME 2024 than their AIME 2025 results predict; OpenAI stopped reporting SWE-bench Verified on 2026-02-23 | New rulers: ARC-AGI-3 (2026-03-25) launched with frontier AI at 0.51%; MathArena evaluates on newly released competitions as they happen | New benchmark results reported at equal compute and with confidence intervals |

## How to read this series

It helps to read these first: [decoding strategies](../../00-foundations/core/decoding.en.md) (sampling parameters), [rejection sampling and best-of-N](../../05-post-training/rejection-sampling.en.md), [verifiable rewards](../../05-post-training/verifiable-rewards.en.md), [GRPO's group-relative signal](../../05-post-training/rlhf/after-rlhf.en.md#group-example) and [DAPO](../../05-post-training/dapo.en.md). Neighbouring notes: [generate, verify, retrain](../self-improvement/generate-verify-train.en.md) shows the same verifier ceiling inside a training loop; [choosing a model for an agent](../../10-agents/model-choice.en.md#thinking-control) covers "who decides whether to think"; [Looped Transformer](../../00-foundations/looped/README.en.md) covers computing more rounds in the hidden state without writing tokens; how reliable a judge is, see [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md).

1. [Sampling, voting, and verifiers](parallel-and-verify.en.md): why voting only helps when the right answer is the most common one, how a verifier's error rate sets the ceiling, process reward models and search, and what Snell et al.'s compute-optimal result actually says
2. [How long chains of thought are trained](long-chains.en.md): what R1-Zero did and why length grows, distillation, controlling how long to think, overthinking and underthinking, and whether the chain is faithful
3. [How to compare and read the numbers](measuring.en.md): comparing at equal compute, pass@1 versus pass@k versus maj@k, the "RL only sharpens" debate, contamination and saturation
4. [Review questions](review.en.md): interview questions and a self-check
