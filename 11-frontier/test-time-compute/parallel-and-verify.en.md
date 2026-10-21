# Test-time compute: sampling, voting, and verifiers

[中文](parallel-and-verify.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

A competition problem has an integer answer. The model is sampled 64 times: 26 times it says A, 30 times B, and the other 8 answers are all different. Voting returns B; if the right answer is A, more samples only make B win more firmly. **Voting converges to the answer the model gives most often, so it helps only when the right answer is already the most common one; a verifier can get around this, but the rate at which it accepts wrong answers sets the ceiling you can reach.**

## Three things to separate first: coverage, selection, search

- **Coverage**: at least one of the N samples is correct. If one try on a problem is right with probability $p$ and tries are independent, coverage is $1-(1-p)^N$.
- **Selection**: hand back one of the N. Voting, verifiers and reward models all belong to this step.
- **Search**: spend compute on intermediate steps before the whole answer is written, keeping only promising prefixes.

Coverage only says the right answer is somewhere in the samples, not that you can hand it over. [Brown et al. (2024-07)](https://arxiv.org/abs/2407.21787) saw coverage grow roughly log-linearly in the number of samples over four orders of magnitude on several tasks; on SWE-bench Lite, DeepSeek-Coder-V2-Instruct solved 15.9% with one sample and 56% with 250, because every patch could be run against tests. On tasks without an automatic verifier, they found that voting and reward models stop improving with more samples after a few hundred. The gap between coverage and selection is what this note is about. The general relation between pass@1 and pass@k is already covered in [alignment tax](../../05-post-training/alignment-tax.en.md#how-to-measure-it-no-single-score-will-show-you).

<!-- paper-figure:monkeys-coverage-selection -->

## Majority voting: work it once

[Self-consistency](https://arxiv.org/abs/2203.11171) (Wang et al., 2022-03, ICLR 2023) samples several reasoning paths, looks only at the final answers, and returns the most frequent one; the paper reports 17.9 points over greedy chain-of-thought decoding on GSM8K. Below, majority-vote accuracy maj@N is computed for three made-up answer distributions. The rule: return the answer with the most votes, and break ties at random.

**Case 1: the wrong answers are all different.** $p=0.4$, and the wrong 60% are a different number every time, say because the arithmetic goes wrong in many ways. With N=3, let the right answer get $a$ votes:

$$
\text{maj@3}=P(a\ge2)+\tfrac13P(a=1)=(0.064+0.288)+\tfrac13\times0.432=0.496
$$

When $a=1$, each of the three answers has one vote, a three-way tie, and a random pick lands on the right answer with probability 1/3. With N=5 the same reasoning gives $0.663+\tfrac15\times0.259\approx0.715$: two votes are already a unique maximum.

**Case 2: the wrong answers all coincide.** Still $p=0.4$, but the wrong 60% are always the same answer B, say because the problem has a common trap. Now the right answer needs a majority to win: maj@3 $=P(a\ge2)=0.352$ and maj@5 $=P(a\ge3)\approx0.317$, below a single sample.

**Case 3:** $p=0.6$, with all wrong answers being B (0.4). maj@3 $=0.648$ and maj@5 $\approx0.683$.

| Answer distribution on one problem | N=1 | N=3 | N=5 | N→∞ |
| --- | --- | --- | --- | --- |
| Right 0.4, wrong answers all different | 0.400 | 0.496 | 0.715 | 1 |
| Right 0.4, wrong answers all B (0.6) | 0.400 | 0.352 | 0.317 | 0 |
| Right 0.6, wrong answers all B (0.4) | 0.600 | 0.648 | 0.683 | 1 |

As N grows, voting gets closer to "return the model's most common answer on this problem". There is only one rule: **if the right answer is the mode for this problem, voting helps; if not, voting makes the wrong answer win more firmly.** Cases 1 and 2 have the same $p$ and opposite outcomes; the only difference is whether the errors are spread out or concentrated. Accuracy averaged over a dataset mixes both kinds of problem, so the average $p$ cannot predict the average score after voting. You have to look problem by problem.

Public results show both. OpenAI self-reported on 2024-09-12 that o1 scored 74.4% pass@1 and 83.3% with a 64-sample vote on AIME 2024; in the same table, on the biology subset of GPQA Diamond, the 64-sample vote gave 68.4% against 69.2% pass@1, so voting did not help (the gap is small and does not show that voting hurts). In [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), majority voting stalled at 69.6% with 1860 samples per problem, below both reward models. Voting also needs answers that can be compared: numbers and options can be checked directly, but open-ended writing needs a judge to decide whether two answers mean the same thing; the judge's own problems are covered in [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md).

## Best-of-N with a verifier: the error rate sets the ceiling

Now pick differently. Each sample goes to a verifier that only says accept or reject. Let the verifier accept a correct answer with probability $r$ and wrongly accept a wrong answer with probability $f$ (the false positive rate), judging each sample independently. The strategy: draw N samples and return any accepted one; if none is accepted, return the first sample.

A single sample is rejected with probability $q=1-pr-(1-p)f$; among accepted samples, the fraction that is correct (the precision) is $pr/\big(pr+(1-p)f\big)$. So

$$
\text{acc}(N)=\big(1-q^N\big)\cdot\frac{pr}{pr+(1-p)f}+q^N\cdot\frac{p(1-r)}{q}
$$

The first term is the chance of returning a correct answer when at least one sample is accepted; the second is the chance that, with every sample rejected, the first sample happens to be correct. Take $p=0.4$ and $r=1$ (correct answers are always accepted); the second term is then 0:

| Fraction of wrong answers accepted, $f$ | N=1 | N=4 | N=16 | Ceiling (N→∞) |
| --- | --- | --- | --- | --- |
| 0 (perfect verifier) | 0.400 | 0.870 | 1.000 | 1 |
| 0.1 | 0.400 | 0.796 | 0.870 | 0.870 |
| 0.3 | 0.400 | 0.668 | 0.690 | 0.690 |

With a perfect verifier, best-of-N equals coverage, $1-0.6^N$. As soon as $f>0$, more samples can only approach the precision $0.4/(0.4+0.6f)$: 87.0% at $f=0.1$ and 69.0% at $f=0.3$. An $r<1$ also lowers the ceiling; with $r=0.8$ and $f=0.1$ the precision is $0.32/0.38\approx84.2\%$. These two numbers match the "tests that accept wrong solutions" example in [generate, verify, retrain](../self-improvement/generate-verify-train.en.md) exactly: picking answers with a verifier at inference time and filtering training data with it at training time hit the same ceiling.

Compare with voting. In case 2 (all wrong answers are B), voting tends to 0, while a verifier with $f=0.1$ still reaches 87%, because the verifier brings in information from outside the model's own distribution. But the calculation assumed the verifier's mistakes are independent across samples. If it falls for the same trap and accepts B every time ($f=1$ for B), the correct fraction among accepted samples drops back to 0.4, no better than one sample. **Whether a verifier helps depends on whether its errors are correlated with the generator's errors.**

When the verifier returns a score instead of 0/1, the usual move is to take the highest score. The larger N is, the easier it becomes to pick a sample that scores high but is wrong; [rejection sampling](../../05-post-training/rejection-sampling.en.md#best-of-n) has a counterexample you can see with just two samples. Snell et al. report that a process reward model trained on the public PRM800K data could be exploited by plain best-of-N when picking PaLM 2's answers, which they attribute to the different generator distributions.

## Outcome reward models and process reward models

A verifier can be a program (unit tests, answer matching, Lean) or a trained model. Trained ones come in two kinds: an outcome reward model (ORM) gives one score to the whole answer; a process reward model (PRM) scores every step and combines the step scores into a score for the answer.

<!-- paper-figure:lightman-prm-grading -->

[Lightman et al. (2023-05)](https://arxiv.org/abs/2305.20050) used a generator fine-tuned from GPT-4 on 500 MATH test problems, picking one of 1860 samples per problem: the PRM picked correctly 78.2% of the time, the ORM 72.4% and majority voting 69.6%, and the PRM's lead grew with N. The PRM's training data, PRM800K, holds 800,000 step-level human labels on 75,000 solutions to 12,000 problems; during labelling, they deliberately showed people "convincing wrong-answer solutions" that the current PRM rated highly. An answer's score is the product of the per-step probabilities of being correct. The study fixes the generator and compares how well each method picks answers; it does not train the generator with RL, a distinction that [verifiable rewards](../../05-post-training/verifiable-rewards.en.md#outcome-rewards-or-process-rewards) discusses directly.

PRMs are hard to label and hard to define "one step" for. Snell et al. instead trained the PRM on Monte Carlo rollouts estimating "the chance of still getting it right after this step", with no human labels; they scored an answer by its last step; and they summed the scores of all samples that share a final answer before comparing (they call this best-of-N weighted), which combines voting and verification. The DeepSeek-R1 v1 report (§4.2) lists PRMs among its unsuccessful attempts: fine-grained steps are hard to define in general reasoning, automatic labelling by models does not work well while human labelling does not scale, and a neural PRM inside large-scale RL gets reward-hacked. Their conclusion is that PRMs suit reranking the top N or guiding search, but in large-scale RL the benefit did not justify the overhead.

Proofs have no final answer to match. [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) (2025-11-27) first trains an LLM verifier that finds problems in a proof and scores it, then uses it as the reward for a proof generator, and spends extra verification compute labelling new hard-to-verify proofs so the verifier keeps up with the improving generator. At inference it keeps a candidate pool per problem: 64 proofs are generated and each is verified 64 times; each round, the 64 proofs with the highest average score are each paired with 8 verification analyses and revised, for up to 16 rounds or until a proof passes all 64 verifications. The paper self-reports 5 of 6 problems solved at IMO 2025 and 118/120 on Putnam 2024, graded by mathematical experts the authors engaged.

## Searching over intermediate steps

Best-of-N waits until whole answers are written before picking; search prunes with the scorer while writing. Snell et al. compared three methods (§5.2):

- **Beam search**: at each step, sample several candidates and keep $M$ prefixes by PRM score. The structure is the same as [beam search in decoding](../../00-foundations/core/decoding.en.md#beam-search-why-generative-llms-rarely-use-it), except the score is the PRM instead of likelihood.
- **Lookahead search**: roll each candidate step forward $k$ more steps and score the step by what the rollout reaches; the cost counts as $N\times(k+1)$ samples. With $k=0$ it is beam search; it can also be seen as MCTS without the random exploration.
- **Best-of-N weighted**: the whole-answer selection from the previous section, as the baseline.

The results: at small budgets beam search clearly beats best-of-N; as the budget grows its gains shrink and it ends below best-of-N; at equal budgets lookahead is generally worse. Split by difficulty, beam search gets worse with more budget on easy problems, a sign of over-optimising the PRM; on medium problems it is consistently better; on the hardest problems no method does well. This is the same thing as "the harder the planner tries, the more it exploits the model's errors" in [model-based RL](../../05-post-training/deep-rl/model-based.en.md): when the scorer has errors, a stronger search finds them.

The earlier [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (Yao et al., 2023-05) treats intermediate "thoughts" as tree nodes and searches them with BFS or DFS plus the model's self-evaluation; with GPT-4 it raised the Game of 24 success rate from 4% with chain-of-thought to 74% (breadth 5). The DeepSeek-R1 v1 report also tried MCTS: the token space is so large that the expansions per node had to be capped, which tends to get stuck in local optima; a fine-grained value model is hard to train, so iterating self-search to improve the model did not work. Their conclusion: with a pretrained value model, MCTS can raise scores at inference, but using it to repeatedly train a stronger model remains hard.

## Compute-optimal: what Snell et al. actually said

The paper is often shortened to "a small model that thinks longer beats a 14 times larger model". The accurate version needs its conditions: the [abstract](https://arxiv.org/abs/2408.03314) says that in a FLOPs-matched comparison, on problems where the smaller model already has a non-trivial success rate, "test-time compute can be used to outperform a 14x larger model".

**Setting.** The model is PaLM 2-S* and the problems are MATH (using Lightman et al.'s split of 12,000 training and 500 test problems). Two mechanisms for spending test-time compute are compared: search with a PRM, and a model fine-tuned to revise its answers after seeing earlier attempts, used for sequential revision at inference.

**Definition (Eq. 1).** For a problem $q$ and budget $N$, the hyperparameters $\theta$ of the test-time strategy (for example beam width, or the ratio of sequential to parallel) determine an output distribution $\text{Target}(\theta,N,q)$:

$$
\theta^{*}_{q}(N)=\arg\max_{\theta}\;\mathbb{E}_{y\sim \text{Target}(\theta,N,q)}\big[\mathbf{1}\{y=y^{*}(q)\}\big]
$$

$y^*(q)$ is the reference answer. In words: for each problem and each budget, choose the way of spending that maximises the chance of being right.

**Approximation.** The best $\theta$ per problem is unknown in advance, so they use "difficulty" as a sufficient statistic: the base model's pass@1 on each problem is estimated from 2048 samples and split into 5 bins. The oracle version judges correctness with the reference answers; the predicted version uses only the PRM's average final score, without answers. They then choose the best strategy per bin; to avoid choosing and testing on the same problems, they run two-fold cross-validation within each difficulty bin of the test set.

**Findings.** First, choosing a strategy by difficulty reaches nearly the accuracy of best-of-N with up to 4 times less test-time compute, for both PRM search and sequential revision. Second, the ratio of sequential to parallel has an optimum: easy problems gain more from sequential revision, while hard problems need a balance. Third, against scaling parameters (§7): pretraining FLOPs are counted as $6ND_{\text{pretrain}}$ and inference FLOPs as $2ND_{\text{inference}}$. Scaling parameters by $M$ multiplies both terms by $M$; for the small model to spend the same total FLOPs, its inference compute can be multiplied by

$$
K=M+3\,\frac{D_{\text{pretrain}}}{D_{\text{inference}}}\,(M-1)
$$

Write $R=D_{\text{inference}}/D_{\text{pretrain}}$. The paper uses $M\approx14$ and three values of $R$; plugging them in:

| $R$ | Situation | Extra inference compute $K$ for the small model |
| --- | --- | --- |
| 0.16 | Far less inference than pretraining, e.g. a self-improvement loop | $14+3\times6.25\times13\approx258$ times |
| 0.79 | About equal | About 63 times |
| 22 | Far more inference than pretraining, e.g. serving at scale | About 16 times |

The heavier the inference load, the thinner the larger model's extra pretraining cost is spread, and the less extra compute the small model gets. The paper's conclusion (Figure 9): on easy and medium problems, or with low inference load ($R\ll1$), test-time compute is the better buy; on the hardest problems, or with high inference load ($R\gg1$), scaling pretraining is more effective. The authors state that the two are not exchangeable one for one.

**What you cannot conclude from it.** The results come from one model family and one math dataset and predate long-chain RL models such as R1; estimating difficulty itself costs compute; "14 times" is the comparison they chose, not a universal exchange rate.

## Numbers from real systems

All of the following are self-reported by companies or authors under different conditions, so they cannot be compared with each other:

- **o1** (OpenAI, 2024-09-12): on AIME 2024, 74% with one sample, 83% with a 64-sample vote, and 93% when re-ranking 1000 samples with a learned scoring function. At IOI 2024, for each problem 50 submissions were chosen from many candidates using public tests, model-generated tests and the scoring function, scoring 213 points; random submissions would average only 156; with 10,000 submissions allowed per problem, it scored 362.14.
- **Claude 3.7 Sonnet** (Anthropic, 2025-02-24): in a research setting, with compute equivalent to 256 independent samples, a learned scoring model and up to a 64k-token thinking budget, GPQA 84.8%; the post says this parallel setting was not available in the deployed model.
- **Navier–Stokes** (OpenAI, September 2026): a multi-agent system driven by an internal model; the group that produced the result had about 10,000 agents running at once, produced the proof after about 88 hours, then took about 17 more hours for a Lean formalisation, using about 130 billion output tokens on this problem. Here Lean is the verifier: it guarantees the proof holds for the formal statement. But according to [Scientific American (2026-09-21)](https://www.scientificamerican.com/article/did-openai-solve-the-wrong-navier-stokes-problem/), mathematicians point out that it proves the version that allows an external force, and the unforced problem they actually care about remains open. A verifier checks the specification you wrote down; whether that specification is the question you care about is a separate matter.

## What to check when you try it

| Try first | What to observe | What you can NOT conclude |
| --- | --- | --- |
| Sample N = 1 to 64 on one set of problems; plot coverage and maj@N together | The gap between the two curves is what selection loses | That accuracy can be as high as coverage |
| Compare before and after voting, problem by problem | Which problems voting gets wrong: those where the right answer is not the mode | The per-problem voting outcome from the overall average $p$ |
| Estimate the verifier's $r$ and $f$ on a hand-checked batch of samples | The ceiling is about the precision $pr/\big(pr+(1-p)f\big)$ | That $f$ stays the same when the problem distribution or generator changes |
| Increase N step by step, tracking both verifier score and true accuracy | Score still rising while accuracy flattens or falls means the verifier is being exploited | That a larger N is always better |
| Fix the total number of generated tokens when comparing two methods | Which is more accurate at the same cost | Comparing only each method's best point |

## What to watch on this line next

These are open questions and what kind of result would change the current picture; they are not predictions.

1. **How to pick without reference answers.** Brown et al. (2024-07) saw voting and reward models stop improving after a few hundred samples. [DeepConf](https://arxiv.org/abs/2508.15260) (Meta AI and others, 2025-08) filters reasoning traces by the model's own confidence and self-reports 99.9% on AIME 2025 for GPT-OSS-120B with 512 samples, generating up to 84.7% fewer tokens than full parallel sampling; but AIME has reference answers. If selection accuracy on open-ended tasks approaches coverage under independent grading, the ceiling in this note would need rewriting.
2. **Whether verifiers can stay ahead of generators.** DeepSeekMath-V2 (2025-11) spends verification compute specifically to keep a gap between generation and verification; the [self-improvement series](../self-improvement/generate-verify-train.en.md) shows that once the gap closes, the loop stops. Watch whether verifier error rates on newer generators are published and whether they hold steady.
3. **The specification behind formal verifiers.** The Navier–Stokes dispute (September 2026) is not about whether the Lean proof is correct but whether the formal statement is the question people meant. A community-agreed library of formal statements would make such results easier to judge.
4. **The bill for large-scale parallelism.** GPT-5's pro setting (2025-08), Gemini Deep Think (2025-07) and ten thousand agents (September 2026) all came without equal-compute comparisons. Published accuracy-versus-compute curves, in tokens or FLOPs, are what would show how much of these gains comes simply from more compute.

Next: [how long chains of thought are trained](long-chains.en.md).
