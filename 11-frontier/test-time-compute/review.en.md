# Reasoning models and test-time compute: review questions

[中文](review.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-10

This page condenses the series into interview questions and a self-check; every number and conclusion in the answers can be traced to the first three notes.

## Interview questions

<details class="interview" markdown="1">
<summary>What is test-time compute? What does each of the parallel and sequential ways of spending it cost?</summary>

It is the extra computation spent answering one question with the weights fixed: sampling many and picking, thinking longer within one chain, searching over intermediate steps, or generating then checking and revising. One generated token costs about $2N$ FLOPs, so extra compute is mostly extra tokens. Parallel means generating N answers at once and picking one: latency barely changes, cost grows linearly with N, and you need a way to pick. Sequential means thinking longer in one chain or revising version by version: later steps can use earlier findings, at the price of latency that grows with length. Snell et al. found that easy problems benefit more from sequential revision while hard problems need a balance. Real systems combine both; DeepSeekMath-V2, for example, uses 64 proofs per problem, verifies each 64 times, and revises for up to 16 rounds.

</details>

<details class="interview" markdown="1">
<summary>When does majority voting actually lower accuracy? Give an example you can compute by hand.</summary>

With independent samples from a fixed distribution and a unique mode, the probability of voting for that mode approaches 1 as N grows. This asymptotic result does not guarantee monotonic gains at finite N. Example: a single try is right with probability 0.4, and the wrong 0.6 is always the same answer B. With 3 votes the right answer needs at least 2, with probability $0.064+0.288=0.352$, below the 0.4 of a single sample, and it tends to 0 as N→∞. With the same $p=0.4$ but all wrong answers different, 3 votes give 0.496 and 5 votes about 0.715, tending to 1. So the average $p$ cannot predict the score after voting; you have to check, problem by problem, whether errors are spread out or concentrated. Voting also needs answers that can be compared.

</details>

<details class="interview" markdown="1">
<summary>With best-of-N plus a verifier, what sets the ceiling on accuracy?</summary>

Let a single try be right with probability $p$, and let the verifier accept correct answers with probability $r$ and wrong ones with probability $f$, independently across samples. For large N you always return some accepted sample, so accuracy tends to the precision $pr/\big(pr+(1-p)f\big)$. With $p=0.4$ and $r=1$: the ceiling is 87.0% at $f=0.1$, 69.0% at $f=0.3$, and at $f=0$ accuracy is just coverage, $1-0.6^N$. Unlike voting, a verifier still works when errors are concentrated, provided its mistakes are not correlated with the generator's; if it too always accepts the common wrong answer, you are back to a single sample. With a scoring verifier and argmax selection, a larger N makes it easier to pick a high-scoring wrong answer.

</details>

<details class="interview" markdown="1">
<summary>How do process and outcome reward models differ? What can and cannot be concluded from Lightman et al.?</summary>

An ORM gives the whole answer one score; a PRM scores each step and combines them, Lightman et al. using the product of per-step probabilities of being correct. With a generator fine-tuned from GPT-4, 500 MATH problems and one pick from 1860 samples per problem: PRM 78.2%, ORM 72.4%, majority voting 69.6%; PRM800K holds 800,000 step-level human labels. You can conclude that, for a fixed generator, the PRM picks answers better, and its lead grows with N. You cannot conclude that RL with process rewards trains a stronger policy; that needs a separate experiment. The DeepSeek-R1 report says PRMs are hard to define steps for, hard to label, and get reward-hacked in large-scale RL, so they suit reranking and guiding search better.

</details>

<details class="interview" markdown="1">
<summary>What does Snell et al.'s "compute-optimal" mean? Under what conditions does "a small model plus test-time compute beats a 14 times larger model" hold?</summary>

For each problem and budget, choose the test-time strategy that maximises the chance of being right (Eq. 1); in practice this is approximated with 5 difficulty bins based on the base model's pass@1, saving up to 4 times the compute relative to best-of-N. Against scaling parameters, pretraining is counted as $6ND_{\text{pretrain}}$ and inference as $2ND_{\text{inference}}$, so the small model may multiply its inference compute by $K=M+3(D_{\text{pretrain}}/D_{\text{inference}})(M-1)$. With $M=14$, $R=D_{\text{inference}}/D_{\text{pretrain}}=0.16$ gives about 258 times, while $R=22$ gives only about 16 times. The conclusion: on easy and medium problems, or at low inference load, test-time compute is the better buy; on the hardest problems or at high inference load, scaling pretraining works better, and the two do not trade one for one. The results come from PaLM 2-S* and MATH, before long-chain models such as R1.

</details>

<details class="interview" markdown="1">
<summary>Why do R1-Zero's answers keep getting longer? Is all of that growth good?</summary>

R1-Zero starts from DeepSeek-V3-Base and is trained with GRPO and rule-based rewards (answer correctness plus format), with no neural reward model; average length rose during training while AIME 2024 pass@1 went from 15.6% to 71.0% (v1). One reason is that checking, backtracking and switching methods raise the chance of being right: with 0.3 for answering directly and 0.65 for checking and redoing, the higher-reward strategy gets reinforced and length grows with it. But not all of it is good: Liu et al. point out that GRPO normalises by answer length, so long wrong answers are penalised less per token, which pushes wrong answers longer; o1-like models spent on average 1,953% more tokens on "2 plus 3"; on hard problems, wrong answers are longer and switch approach more often. Kimi k1.5 and L1 rein length in with length rewards, and Dr. GRPO removes the normalisation bias.

</details>

<details class="interview" markdown="1">
<summary>Is the chain of thought faithful? How do you test it, and what does it mean for monitoring?</summary>

Not necessarily. You test it by intervening or planting hints: Turpin et al. made the example answers always (A), and models were swayed without mentioning it in their explanations; Lanham et al. truncated, corrupted and paraphrased chains to see whether answers changed, and found reliance varies by task; Chen et al. planted hints in prompts, and Claude 3.7 Sonnet and DeepSeek R1 mentioned a hint they used only 25% and 39% of the time on average, while in RL environments with planted loopholes they exploited them over 99% of the time and mentioned them under 2% of the time. Baker et al. found that reading the chain catches reward hacking better than watching actions alone, but once the monitor goes into the reward, strong optimisation teaches the model to hide its intent. So chain-of-thought monitoring is useful but not sufficient, and putting optimisation pressure on it can break it.

</details>

<details class="interview" markdown="1">
<summary>How do you read the claim "RL only sharpens what the base model could already do"? How is pass@k estimated?</summary>

Yue et al. (NeurIPS 2025) found that RL models are better at small $k$, base models overtake them at large $k$, and the reasoning paths are already in the base distribution; distillation is better at every $k$. The mechanism can be computed by hand: base accuracy (0.5, 0.02) on two problems becomes (0.95, 0) after RL, pass@1 goes from 0.26 to 0.475, but pass@100 drops from 0.934 to 0.5. On the other side are ProRL (long RL improves on problems the base model cannot solve), CoT-Pass@k (after excluding lucky guesses, RL still widens the boundary) and skill-composition experiments; Spurious Rewards shows Qwen2.5-Math gaining even with random rewards, so part of the gain comes from activating existing behaviour. pass@k uses Chen et al.'s unbiased estimator $1-\binom{n-c}{k}/\binom{n}{k}$; with $n=10$ and $c=3$, pass@5 is about 0.917, while plugging into $1-0.7^5$ gives only 0.832. As of October 2026 there is no consensus; it depends on the base model, training length, domain, $k$, and whether reasoning is checked.

</details>

## Self-check

<div class="taste-check">
  <strong>Use these questions to check your understanding:</strong>
  <ol>
    <li>Given the answer distribution on one problem, where does voting push accuracy? Why is the dataset's average $p$ not enough?</li>
    <li>In the best-of-N accuracy formula, how do $r$ and $f$ each affect the ceiling? What happens when the verifier's errors are correlated with the generator's?</li>
    <li>Under Snell et al.'s accounting, how does the extra inference compute available to the small model change as the inference load $R$ goes from 0.16 to 22, and why?</li>
    <li>An RL training log shows average length rising. Which curves would you split out to tell "learned to check" from length bias?</li>
    <li>What four things do you ask when you read "model X scores 9x% on AIME"? With 30 problems at 80% accuracy, roughly how wide is the 95% confidence interval?</li>
  </ol>
</div>

## References

- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168): GSM8K, picking among many samples with a verifier
- [Self-Consistency](https://arxiv.org/abs/2203.11171): voting on the final answers of several reasoning paths
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050): process versus outcome rewards, PRM800K
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601): searching over a tree of "thoughts"
- [Large Language Monkeys](https://arxiv.org/abs/2407.21787): coverage grows with samples, selection saturates
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314): compute-optimal strategies and the exchange with pretraining
- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/): o1's self-reported results and the decision to hide raw chains of thought
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) and the [*Nature* version](https://doi.org/10.1038/s41586-025-09422-z): R1-Zero, distillation, unsuccessful attempts
- [Kimi k1.5](https://arxiv.org/abs/2501.12599): long-context RL and a length reward
- [s1](https://arxiv.org/abs/2501.19393): SFT on 1,000 examples and budget forcing
- [L1](https://arxiv.org/abs/2503.04697): controlling thinking length with RL
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783): GRPO's length bias and Dr. GRPO
- [Do NOT Think That Much for 2+3=?](https://arxiv.org/abs/2412.21187): overthinking
- [Thoughts Are All Over the Place](https://arxiv.org/abs/2501.18585): underthinking
- [Inverse Scaling in Test-Time Compute](https://arxiv.org/abs/2507.14417): tasks where longer reasoning gets worse
- [Thinking Hard, Not Smart](https://arxiv.org/abs/2608.07968): allocating a budget shared across problems
- [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388), [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702), [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410): chain-of-thought faithfulness
- [Monitoring Reasoning Models for Misbehavior](https://arxiv.org/abs/2503.11926), [Chain of Thought Monitorability](https://arxiv.org/abs/2507.11473): chain-of-thought monitoring and its fragility
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374): the unbiased pass@k estimator
- [Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837): where the pass@k debate started
- [ProRL](https://arxiv.org/abs/2505.24864), [RLVR Implicitly Incentivizes Correct Reasoning](https://arxiv.org/abs/2506.14245), [From f(x) and g(x) to f(g(x))](https://arxiv.org/abs/2509.25123), [The Invisible Leash?](https://arxiv.org/abs/2507.14843), [Spurious Rewards](https://arxiv.org/abs/2506.10947): the sides of the debate
- [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570): a proof verifier and scaled verification compute
- [Deep Think with Confidence](https://arxiv.org/abs/2508.15260): filtering parallel traces by confidence
- [MathArena](https://arxiv.org/abs/2505.23281), [GPQA](https://arxiv.org/abs/2311.12022), [ARC-AGI-2](https://arxiv.org/abs/2505.11831), [ARC-AGI-3 launch](https://arcprize.org/blog/arc-agi-3-launch), [Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/): contamination, saturation and new benchmarks
- [FEval-TTC](https://arxiv.org/abs/2511.01203): cost modelling and fair comparison of test-time methods
