# Test-time compute: how to compare and read the numbers

[中文](measuring.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

When OpenAI released o4-mini on 2025-04-16, it wrote that the model scored 99.5% pass@1 and 100% with an 8-sample vote on AIME 2025; the same paragraph says this was with access to a Python interpreter and should not be compared with results without tools. Behind one number, the compute, the metric and the problems can all differ. **When you read a reasoning model's score, ask four things first: how much compute went into each problem (tokens, samples, tools), whether the metric is pass@1, pass@k or maj@k, whether the model could have seen the problems, and whether there are enough problems to tell the difference.**

## One score, several definitions

| Metric | How it is computed | Usable in deployment? | What it measures |
| --- | --- | --- | --- |
| pass@1 (greedy) | One generation at temperature 0 | Yes | A single output; long-chain models tend to repeat under greedy decoding |
| pass@1 (sampled average) | Sample $k$ times per problem, average the accuracy | Yes, in expectation | Expected accuracy of a single sample |
| pass@k | At least one of $k$ samples is right | Only when correctness can be checked automatically | Coverage: whether the answer is among the samples |
| maj@k (cons@k) | Vote over $k$ samples | Yes, if answers can be compared | Whether the mode is right |
| best-of-k (reward model) | A reward model picks one | Yes | The generator and verifier together |
| With tools | Interpreter, search and so on allowed | Depends on the product | Effectively a different problem |

The pass@1 in the DeepSeek-R1 report is the second row: temperature 0.6, top-p 0.95, 4 to 64 samples per problem, averaged. The authors say greedy decoding makes long outputs repetitive and varies a lot across checkpoints. So two numbers both labelled pass@1, one greedy and one a sampled average, can differ noticeably; read the footnote. Why voting sometimes backfires is in [note 1](parallel-and-verify.en.md).

### Computing the unbiased pass@k estimate by hand

Sample $n$ answers per problem, of which $c$ are correct. [Chen et al. (2021-07)](https://arxiv.org/abs/2107.03374) give the unbiased estimator:

$$
\text{pass@}k=\mathbb{E}_{\text{problems}}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

What is subtracted inside the brackets is the probability that $k$ answers drawn without replacement from the $n$ are all wrong. Take $n=10$ and $c=3$:

| $k$ | Unbiased estimate | Plugging in $1-(1-c/n)^k$ |
| --- | --- | --- |
| 1 | 0.300 | 0.300 |
| 5 | $1-\binom{7}{5}/\binom{10}{5}=1-21/252\approx0.917$ | 0.832 |
| 10 | 1 (one of the 10 is right) | 0.972 |

The plug-in version is low on average, because $1-(1-x)^k$ is concave. Yue et al. use this estimator with $n$ set to the largest $k$ on the curve, 1024 for AIME and AMC.

## Comparing at equal compute

"Model A thinks for 20,000 tokens and scores 78%; model B thinks for 2,000 and scores 70%" does not yet say A is stronger. Compute has at least four units:

- **Output tokens**: the easiest to obtain, and what APIs bill (thinking tokens count as output).
- **Number of samples**: common for parallel methods, though samples can differ in length.
- **FLOPs**: about $2N$ per generated token, where $N$ is the number of parameters doing the work (only active parameters for MoE); with very long contexts, attention and KV-cache costs come on top, see [KV cache and inference](../../00-foundations/deep-dives/kv-cache-and-inference.en.md).
- **Latency and dollars**: these depend on parallelism, hardware and prices, and change over time.

Work it once with teaching numbers. Model A has 8 billion parameters and thinks for 20,000 tokens; model B has 70 billion and generates only 2,000. Using $2N$, A spends $2\times8\times10^{9}\times20{,}000=3.2\times10^{14}$ FLOPs and B spends $2\times70\times10^{9}\times2{,}000=2.8\times10^{14}$ FLOPs, almost the same. "A thought 10 times longer" holds in tokens but not in FLOPs, and API prices give yet another answer. So compare curves: one unit of compute on the horizontal axis, accuracy on the vertical, and see whose curve is higher, rather than each side reporting one point.

An example from the public record: on 2024-12-20 ARC Prize tested o3 on the ARC-AGI-1 semi-private set, scoring 75.7% in a high-efficiency configuration (6 samples) and 87.5% in a low-efficiency one (1024 samples, about 172 times the compute); the same model differed by 12 points, which is why ARC Prize publishes cost per task alongside scores. The same post states that the tested version had been trained on 75% of the public training set; its 2025-04 update adds that OpenAI confirmed this version differs from the o3 released later. [FEval-TTC (2025-11)](https://arxiv.org/abs/2511.01203) proposes an evaluation protocol with fixed few-shot prompts, answer extraction and a cost model, comparing test-time methods in both tokens and dollars. For converting FLOPs and inference load into "can test-time compute stand in for a larger model", see Snell et al.'s table in note 1.

## Does RL only "sharpen"? The pass@k debate

[Yue et al. (2025-04, NeurIPS 2025 oral)](https://arxiv.org/abs/2504.13837) compared base models with the same models after RL with verifiable rewards on math, code and visual reasoning: at small $k$ (for example 1) the RL models are better; at large $k$ the base models overtake them, and the longer the training, the fewer problems remain solvable. The abstract concludes that "the current training setup does not elicit fundamentally new reasoning patterns". They use perplexity to argue that the RL models' reasoning paths were already within the base model's sampling distribution; six RLVR algorithms performed similarly and remained far from using the base model's potential fully; distillation, by contrast, brings in the teacher's new reasoning patterns and stays above the base model at every $k$. To rule out lucky guesses they checked chains of thought on hard problems by hand: among AIME 2024 problems with average accuracy below 5%, the base model solved 7, and of the 6 that were not ambiguous, 5 had at least one correct chain; the RL model solved 6, of which 4 did.

<!-- paper-figure:rlvr-search-trees -->

### Work it once: why $k=1$ and $k=100$ can point in opposite directions

A teaching example. Two problems; the base model's single-try accuracy is (0.5, 0.02) and after RL it is (0.95, 0): RL makes the easy problem reliable but trains away the occasional success on the hard one. Per problem pass@k $=1-(1-p)^k$, averaged over the two:

| $k$ | Base | After RL |
| --- | --- | --- |
| 1 | 0.260 | 0.475 |
| 3 | 0.467 | 0.500 |
| 4 | 0.508 | 0.500 |
| 100 | 0.934 | 0.500 |

At $k=1$ RL almost doubles the score; by $k=4$ it has already been overtaken; at $k=100$ the base model covers even the hard problem with probability $1-0.98^{100}\approx0.867$. Average pass@1 rising while average pass@k falls can both be true at once; [alignment tax](../../05-post-training/alignment-tax.en.md#how-to-measure-it-no-single-score-will-show-you) discusses the same situation from the angle of diversity.

### The other side, and additions

| Work | Claim | Evidence and conditions |
| --- | --- | --- |
| [ProRL](https://arxiv.org/abs/2505.24864) (NVIDIA, 2025-05) | Long enough RL widens the boundary | Starting from DeepSeek-R1-Distill-Qwen-1.5B, more than 2,000 steps on 136,000 problems across five domains, with KL control and reference-policy resets; reports gains even on tasks the base model fails at any number of samples, with gains correlated with base-model competence and training length |
| [Wen et al.](https://arxiv.org/abs/2506.14245) (Microsoft Research Asia and others, 2025-06) | pass@k counts lucky guesses as successes | Proposes CoT-Pass@k, which counts a success only when both reasoning and answer are right; by this measure RLVR widens the reasoning boundary on math and code |
| [Yuan et al.](https://arxiv.org/abs/2509.25123) (2025-09) | RL can compose existing skills into new ones | Synthetic string-function tasks: when the model already knows $f$ and $g$, RL teaches unseen compositions $g(f(x))$, which transfer to another task; next-token training on the same data does not |
| [Wu et al., "Invisible Leash"](https://arxiv.org/abs/2507.14843) (2025-07, v5 2026-10-14) | Problem by problem, shrinkage outweighs expansion | Matching verified successes per question: across the evaluated models and benchmarks, more questions lose a verified success than gain one, while pass@1 rises |
| [Shao et al., Spurious Rewards](https://arxiv.org/abs/2506.10947) (2025-06, v2 2026-02) | Part of the RL gain comes from activating existing behaviour | Qwen2.5-Math-7B trained with random rewards still gains 21.4 points on MATH-500 (29.1 with true rewards); Llama3 and OLMo2 show no such effect; the authors attribute it to a bias from GRPO's clipping |

As of October 2026 the debate has no consensus, and every side's conclusion depends on conditions: what the base model can already do, how long training runs, which domains the problems cover, how large $k$ is, and whether reasoning is checked. When reading such a paper, find these first.

## Has the model seen the problems? Contamination and saturation

### The number of problems limits what you can distinguish

AIME has two exams a year with 15 problems each, 30 in total, so one problem is 3.3 points. Averaging 64 samples per problem reduces the noise of "right this time, wrong next time" on a given problem, but cannot change "only these 30 problems were drawn". Treating the 30 as a sample from similar problems, at 80% accuracy the half-width of a 95% confidence interval is about $1.96\sqrt{0.8\times0.2/30}\approx0.14$, that is 14 points; GPQA Diamond has 198 four-option questions, giving about 5.6 points at the same 80%. Two models 5 points apart on one year's AIME tell you little. More systematic paired comparisons are in [the score improved: what does that actually establish?](../../07-evaluation/metric-robustness.en.md).

### Problems the model has seen

- **AIME 2024.** [MathArena (2025-05, v3 2026-01)](https://arxiv.org/abs/2505.23281) evaluates models right after each new competition's problems are released. Comparing AIME 2024 with 2025, and correcting for year-to-year difficulty with human percentiles, most models score 10–20 points higher on the 2024 problems, and QwQ-32B-Preview nearly 60 points higher; on HMMT the gap is much smaller. They also found 8 AIME 2025 problems with similar versions already online, mostly easy ones.
- **DeepSeek's own check.** The R1 v2 report decontaminates with 10-gram matching, removing about six million pretraining texts in math alone, but acknowledges this cannot catch paraphrased problems, so benchmarks released before 2024 may be contaminated. They also tested AIME 2025, released after training: 75% pass@1, against 79.8% on AIME 2024.
- **SWE-bench Verified.** [OpenAI announced on 2026-02-23 that it no longer reports it](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/): of 138 problems that o3 could not solve consistently over 64 runs, 59.4% had material issues in tests or problem descriptions (35.5% with tests that enforce implementation details, 18.8% testing functionality the problem never asked for); every frontier model they tested could reproduce the original fix or problem details for some tasks; the best score had moved only from 74.9% to 80.9% over the previous 6 months. They now recommend SWE-bench Pro.
- **GPQA.** The [original paper](https://arxiv.org/abs/2311.12022) embeds a canary string in the data and asks people not to post the questions in plain text, to reduce the chance they end up in training corpora. The 198 questions of the Diamond subset require both experts to answer correctly (or the second expert's mistake to be clearly documented) and most non-experts to answer incorrectly.

<!-- paper-figure:matharena-contamination -->

### Saturation

- Competitions scored only by final answers are close to perfect scores: o4-mini with Python reached 99.5% pass@1 on AIME 2025 (OpenAI, 2025-04-16, self-reported); [DeepConf](https://arxiv.org/abs/2508.15260), sampling GPT-OSS-120B 512 times and filtering by confidence, reached 99.9% on AIME 2025 (Meta AI and others, 2025-08, self-reported). The DeepSeekMath-V2 authors (2025-11) also write that RL with final-answer rewards saturated competitions like AIME and HMMT within a year, which is why they moved to proofs.
- ARC-AGI is on its third generation: on ARC-AGI-1, o3 reached 87.5% with 1024 samples (2024-12-20); in the [ARC-AGI-2 paper](https://arxiv.org/abs/2505.11831) (2025-05), o3 (medium) scored only 3.0%, and the authors say scores below 5% are mostly noise; [ARC-AGI-3](https://arcprize.org/blog/arc-agi-3-launch) (2026-03-25) switched to interactive game environments with no instructions at all, launching with humans at 100% and frontier AI at 0.51%. Later leaderboard figures contradict each other in secondary sources, so none are listed here.

## When you read a claim, ask these first

| The claim you read | Ask first | Example from this series |
| --- | --- | --- |
| "9x% on AIME" | Which year's problems? Is pass@1 greedy or a sampled average? Any voting, any tools? How wide is the interval over 30 problems? | o4-mini with Python at 99.5%; suspected 10–20 points of contamination on AIME 2024 |
| "The longer it thinks, the more accurate" | Is the horizontal axis tokens, effort levels or samples? The same model? Where does it flatten? | s1 stops improving by the sixth forced extension; the inverse-scaling tasks of Gema et al. |
| "A small model plus test-time compute beats a big model" | Equal tokens or equal FLOPs? How large is the inference load $R$? What is the difficulty mix? | Snell et al.: holds on easy problems at low inference load, not on hard ones |
| "RL taught the model new reasoning abilities" | Is pass@k at large $k$ also above the base model? Was the reasoning checked? Does it hold in another model family? | Yue et al. versus ProRL; random rewards also raise Qwen2.5-Math scores |
| "Verifier-picked answers reach 9x%" | Is the generator fixed? How large is N? How many wrong answers does the verifier accept on held-out data? | Lightman et al. pick 1 of 1860; the ceiling at $f=0.1$ is 87% |
| "The chain of thought shows the model doing X" | Any intervention experiments? How often does it mention a hint it used? | Chen et al.: 25% and 39% |
| "New record on a new benchmark" | Is the benchmark saturated or contaminated? Are its tests themselves correct? | SWE-bench Verified dropped on 2026-02-23 |
| "xx% on ARC-AGI" | Which generation? How much compute per task? | o3's 75.7% and 87.5% are about 172 times the compute apart |

## What to watch on this line next

These are open questions and what kind of result would change the current picture; they are not predictions.

1. **Will equal-compute reporting become the norm?** Since 2024-12 ARC Prize has published cost per task alongside scores, while most model releases still give single-point scores. The more reports publish accuracy-versus-tokens or accuracy-versus-dollars curves, the more trustworthy cross-model comparisons become.
2. **How the pass@k debate resolves.** Wu et al.'s question-by-question analysis was still being updated on 2026-10-14. Replications across several model families, with reasoning checked and training compute matched, will decide how far "RL only sharpens" can be pushed.
3. **How long the new rulers last.** MathArena keeps evaluating on new competitions; ARC-AGI-3 launched with frontier AI at 0.51%; OpenAI recommends SWE-bench Pro. Watch how fast these benchmarks are saturated and whether new evidence of contamination appears.
4. **Budget allocation becoming a measured ability.** [Fan et al. (2026-08)](https://arxiv.org/abs/2608.07968) turned "several problems sharing one budget" into an evaluation. If labs start reporting such scores, "how long to think" moves from a product setting to part of the model's measured ability.

Finally: [review questions](review.en.md).
