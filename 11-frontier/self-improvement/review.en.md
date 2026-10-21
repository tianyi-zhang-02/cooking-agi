# Recursive self-improvement: review questions

[中文](review.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-10

## Interview questions

<details class="interview" markdown="1">
<summary>What does "recursive self-improvement" mean in different settings, and how do you tell them apart quickly?</summary>

Ask three things: what object changes, who proposes and applies the change, and what signal accepts it. Those questions give four types: (a) the model designs and runs its own training and improves its own weights; (b) a system edits its own scaffold, code or prompts with frozen weights; (c) AI accelerates human-led research; (d) training on self-generated data or self-play, with a human-written update rule. Both (a) and (d) change weights; the difference is who designs the update. Most public "self-improving" systems are (b) or (d); as of 2026-10-15, no full public demonstration of (a) was found.

</details>

<details class="interview" markdown="1">
<summary>How does "filter, then fine-tune" in the style of STaR relate to RL?</summary>

STaR treats it as an approximation to policy gradient on a 0/1 reward: wrong samples contribute zero gradient, so "keep the correct ones and run SFT" approximates one policy-gradient step, and the filter is the reward. ReST-EM writes it as EM: the M-step is reward-weighted maximum likelihood on samples from the previous model. Unlike online RL such as PPO, the sampling policy is fixed during this step, so data collection and optimisation are decoupled. The consequence: the verifier defines what gets optimised, and the model learns toward whatever the tests reward.

</details>

<details class="interview" markdown="1">
<summary>Why does training on your own data saturate within a few rounds?</summary>

Song et al. define the generation-verification gap: how much the true score rises when samples are reweighted by the verifier. If the update learns the reweighted distribution to within $\varepsilon$, one round's gain is about the gap plus or minus $\varepsilon$. They measured the gap falling to near zero within two or three rounds, non-positive gaps for small models under most verification methods, and falling pass@k at large $k$. ReST-EM sees diminishing returns after a couple of iterations; Self-Rewarding ran only 3 iterations and its authors expect saturation. These results limit sustained gains under the corresponding verification signals and updates; they do not establish that learning without new external information is impossible.

</details>

<details class="interview" markdown="1">
<summary>A model is truly correct 40% of the time, and the tests accept 10% of wrong solutions. After filtering by the tests and fine-tuning, how high can true correctness go?</summary>

This is a teaching assumption. Accepted samples make up $0.40+0.60\times0.10=0.46$, of which $0.40/0.46\approx87\%$ are correct. If fine-tuning imitates them perfectly, true correctness reaches 87% while the tests show 100%. In the next round the tests treat "correct" and "wrong but passing" alike, the gap is 0, and the loop stops at 87%. With a 30% false-accept rate the ceiling is 69%; at 2% it rises to about 97%. Raising the ceiling means strengthening the verifier, for example with hidden tests and edge cases.

</details>

<details class="interview" markdown="1">
<summary>Why does Absolute Zero's learnability reward give 0 both to tasks that are always solved and to tasks that are never solved?</summary>

Neither provides a success-versus-failure contrast under this within-group binary-reward comparison; other training objectives may still use these samples. The proposer's reward is 0 when the average success $\bar r$ is 0, and $1-\bar r$ otherwise. $\bar r=0.5$ earns 0.5, $\bar r=1$ earns 0, and $\bar r=0.25$ earns 0.75, so the highest reward goes to tasks solved only occasionally; with $G$ attempts the maximum is $1-1/G$. "Zero" means no human-curated tasks or answers; a pretrained model and a Python executor are still required.

</details>

<details class="interview" markdown="1">
<summary>Do AlphaEvolve and the Darwin Gödel Machine count as "improving themselves"? How do they differ from "a model changes its own weights"?</summary>

Both change code; the LLM weights stay fixed. AlphaEvolve evolves programs against a human-written evaluation function; the kernel tiling it found cut Gemini training time by 1%, including training the models behind AlphaEvolve, and the authors say that feedback cycle runs on the order of months. DGM has a coding agent edit its own repository and keeps every working agent in an archive; after 80 iterations SWE-bench goes from 20.0% to 50.0%, but the authors state it is limited by the underlying FM and leave rewriting the training script to update the FM as future work. To be type (a), the object would have to be weights, the update designed by the model, and the acceptance signal out of the edited system's reach.

</details>

<details class="interview" markdown="1">
<summary>How is METR's 50% time horizon computed, and why are readings above 16 hours unreliable?</summary>

For each model, fit a logistic curve of success against the (log) human time of each task; the human time at which success is 50% is the horizon. If every task group succeeds more than half the time, the 50% point can only be extrapolated beyond the longest task, and one success more or less can change the estimate several-fold; METR's suite has no task over 30 hours. METR also said on 2026-05-19 that TH1.1 is essentially saturated and that at least 16% of successes on tasks of 8 hours or more were illegitimate; correcting for noise in task lengths could lower horizons by about another 30%.

</details>

<details class="interview" markdown="1">
<summary>Tomorrow you read a new "AI is accelerating AI" claim. How would you evaluate it?</summary>

First classify it: a task-level measurement, a lab determination against a threshold, a self-reported lab number, or a statement or prediction. Then ask five things: who measured it and can outsiders check it; whether the unit is code share, agent hours, researcher uplift or the overall pace of progress; how much humans intervened; how much compute grew over the same period; and whether the instrument is saturated, whether cheating was reviewed, and where any trend break was placed. Finally compare with existing anchors: as of October 2026, every estimate that gives an overall progress multiplier is below 2x (Anthropic self-reported, about 1.5x quoted by METR), Google DeepMind found no significant end-to-end acceleration on 2026-10-13, and no lab has declared a formal threshold crossed. Also remember that in METR's controlled trial, developers' perception and the measurement differed even in sign.

</details>

## Self-check

<div class="taste-check">
  <strong>Use these questions to check your understanding:</strong>
  <ol>
    <li>Given a "self-improving" system, how do you state the object it changes, who proposes changes, and what signal accepts them, and decide which of types (a)–(d) it is?</li>
    <li>Why is STaR's filter equivalent to a 0/1 reward? When "correct" is replaced by "passes the tests", what is the loop actually optimising?</li>
    <li>If the false-accept rate falls from 10% to 2%, how does the ceiling of filter-then-fine-tune change? Why, in the ideal case, does the second round gain nothing?</li>
    <li>In DGM's parent selection, what roles do score and number of children play? Why does HGM argue that score alone is not enough?</li>
    <li>When a model succeeds more than half the time on every task group, how is its 50% time horizon obtained, and how far can that number be trusted?</li>
  </ol>
</div>

## References

- [Speculations Concerning the First Ultraintelligent Machine](https://doi.org/10.1016/S0065-2458%2808%2960418-0): I. J. Good, origin of the intelligence explosion
- [Gödel Machines](https://arxiv.org/abs/cs/0309048): self-rewriting gated by formal proof
- [Levels of Organization in General Intelligence](https://intelligence.org/files/LOGI.pdf): definition of seed AI
- [When AI builds itself](https://www.anthropic.com/institute/recursive-self-improvement): the Anthropic Institute's 2026 definition and self-reported numbers
- [Recursive Self-Improvement in AI (survey)](https://arxiv.org/abs/2607.07663): 1,250 papers and the verification hierarchy
- [STaR](https://arxiv.org/abs/2203.14465): fine-tuning on one's own correct rationales
- [ReST-EM](https://arxiv.org/abs/2312.06585): self-training written as EM
- [Expert Iteration](https://arxiv.org/abs/1705.08439): search as expert, network as apprentice
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020): the model grades itself, then DPO
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798): self-correction without external feedback
- [Absolute Zero](https://arxiv.org/abs/2505.03335): self-proposed tasks, executor checks, learnability reward
- [R-Zero](https://arxiv.org/abs/2508.05004): co-evolving Challenger and Solver
- [Mind the Gap](https://arxiv.org/abs/2412.02674): the generation-verification gap and saturation in two or three rounds
- [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y): model collapse
- [AlphaEvolve white paper](https://arxiv.org/abs/2506.13131) and [one-year impact post](https://deepmind.google/blog/alphaevolve-impact/): evolving code, and its limits
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954): archive, parent selection, objective hacking
- [SICA](https://arxiv.org/abs/2504.15228): the best agent edits its own code
- [Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614): the metaproductivity–performance mismatch
- [AIDE²](https://arxiv.org/abs/2609.26457): a research agent rewrites its own code; the ignition test
- [PostTrainBench](https://arxiv.org/abs/2603.08640): agents post-train a base LLM
- [Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499), [Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/), [Frontier Risk Report](https://metr.org/blog/2026-05-19-frontier-risk-report/): METR's time horizon and its limits
- [RE-Bench](https://arxiv.org/abs/2411.15114), [MLE-bench](https://arxiv.org/abs/2410.07095), [PaperBench](https://arxiv.org/abs/2504.01848): AI R&D benchmarks
- [OpenAI Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf), [Anthropic RSP v3.4](https://www-cdn.anthropic.com/files/4zrzovbb/website/0bacdc8440ea96e62a8766d99ebe1d4eea6d5f3a.pdf), [Google DeepMind FSF v3.0](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/strengthening-our-frontier-safety-framework/frontier-safety-framework_3.pdf): the labs' thresholds
- [Research acceleration: The view inside OpenAI](https://openai.com/index/research-acceleration-view-inside-openai/), [Anthropic's August 2026 risk report](https://www.anthropic.com/aug-2026-risk-report), [Gemini 4 Argon safety report](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_safety_report.pdf): three labs' reports on their own R&D acceleration
- [The Economics of Recursive Self-Improvement](https://arxiv.org/abs/2609.15802): calibrating feedback-loop elasticities
- [METR's developer randomised controlled trial](https://arxiv.org/abs/2507.09089): perceived versus measured speedup
