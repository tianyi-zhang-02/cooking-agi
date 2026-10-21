# Self-improvement: generate, verify, retrain — why the verifier sets the ceiling

[中文](generate-verify-train.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

Have a model that writes Python functions generate a few solutions per problem, run the tests, fine-tune only on the ones that pass, and use the new model for the next round. No one needs to write the answers, so the loop looks as if it could continue indefinitely. But if the tests accept some wrong solutions, those enter the training set too.

This note uses the generation-verification gap to analyse each round's improvement. Under the stated update assumptions, the gain is constrained by the quality difference the verifier can identify through selection. Errors the verifier cannot distinguish are retained too; repeating the same loop cannot be relied on to remove them automatically.

## One loop, three parts

Generate, verify, train. [STaR](https://arxiv.org/abs/2203.14465) (Zelikman et al., 2022) is the most cited version: the model writes a rationale and an answer, keeps only the correct ones, fine-tunes on them, and starts the next round. Rewritten for this series' coding example, the structure follows STaR's Algorithm 1:

```text
Input: pretrained model M; problems {(x_i, T_i)}, T_i = tests for problem i
M_0 ← M
for n = 1..N:
    for each problem x_i, sample a solution ŷ_i from M_{n-1}
    D_n ← {(x_i, ŷ_i) : ŷ_i passes T_i}         # verify: keep only passes
    for failed problems, add a hint to the prompt and resample; add passes to D_n
                                              # in STaR the hint is the answer: "rationalization"
    M_n ← fine-tune the ORIGINAL M on D_n      # retrain from M each round, not from M_{n-1}
```

In §3.1, STaR treats this as an approximation to policy gradient on the objective:

$$
J(M)=\sum_i \mathbb E_{\hat r_i,\hat y_i\sim p_M(\cdot\mid x_i)}\big[\mathbf 1(\hat y_i=y_i)\big]
$$

$\hat r_i$ is a sampled rationale, $\hat y_i$ the answer, and $\mathbf 1(\cdot)$ is 1 when the answer is correct and 0 otherwise. Samples scoring 0 contribute zero gradient, so "discard the wrong ones and run SFT on the rest" approximates one policy-gradient step with a 0/1 reward: **the filter is the reward**. In the coding example, $\mathbf 1(\hat y_i=y_i)$ becomes $\mathbf 1(\hat y_i\text{ passes }T_i)$. The loop optimises "passes the tests", not "is correct", and the rest of this note follows from that.

STaR's reported results: GPT-J reaches 72.5% on the CommonsenseQA dev set, against 60.0% for direct fine-tuning and 73.0% for a roughly 30x larger GPT-3 (Table 1); on GSM8K, 10.7% against 5.8% (Table 2). The paper also states two preconditions. Without rationalization, problems the model cannot solve never produce a training signal and improvement stops (§3.2). First-round few-shot accuracy must be above chance; GPT-2 could not bootstrap even on arithmetic (§6). §6 also notes that binary-choice questions are easy to get right by chance, which lets many flawed rationales in.

## ReST-EM: "filter, then fine-tune" written as EM

Google DeepMind's [ReST-EM](https://arxiv.org/abs/2312.06585) (Singh et al., 2023-12) writes the same loop as EM for RL. With a non-negative reward $r$, the M-step is:

$$
\theta_{t+1}=\arg\max_\theta\;\mathbb E_{x\sim\mathcal D}\,\mathbb E_{y\sim p_{\theta_t}(y\mid x)}\big[r(x,y)\,\log p_\theta(y\mid x)\big]
$$

Samples come from the previous model $p_{\theta_t}$, which is held fixed during this step; the objective is reward-weighted maximum likelihood. With a 0/1 reward it is "SFT on the accepted samples". The E-step is generation plus filtering, and the M-step is fine-tuning (they also restart from the base model each iteration). The paper stresses that, unlike online RL such as PPO, the sampling policy is fixed at the previous iteration, so data collection and policy optimisation are decoupled.

Empirically, §1 reports diminishing improvement beyond a couple of iterations, which the authors attribute to overfitting on a small set of training problems; on APPS, more iterations led to regression. [Expert Iteration](https://arxiv.org/abs/1705.08439) (2017) belongs to the same family: a tree search acts as the "expert" that produces better targets, a neural network "apprentice" imitates them, and the apprentice guides the next search. For how filtering changes the training distribution, see the site's [rejection sampling](../../05-post-training/rejection-sampling.en.md) note; for the general problem of tests as rewards, see [verifiable rewards](../../05-post-training/verifiable-rewards.en.md).

## When there is no answer key: let the model grade itself

Meta and NYU's [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) (Yuan et al., 2024-01) make the verifier the model itself: Llama 2 70B scores its own responses LLM-as-a-Judge style, builds preference pairs from them, and runs iterative DPO; the paper reports that its judging ability improves too. The AlpacaEval 2.0 win rate against GPT-4 Turbo goes 9.94%, 15.38%, 20.44% over three iterations (Table 1). The paper's own limitations: only 3 iterations in one setting; outputs grow longer; reward hacking was not studied; and the effect "likely saturates" in real-world settings.

Earlier, [Huang et al. (2023)](https://arxiv.org/abs/2310.01798) tested another kind of self-judging: without external feedback, models often fail to correct their own reasoning, and sometimes make it worse.

## Writing the problems too: Absolute Zero and R-Zero

[Absolute Zero](https://arxiv.org/abs/2505.03335) (Zhao et al., 2025-05) has one model play two roles: a proposer emits a task $\tau$, a Python executor turns it into a validated problem and answer $(x,y^\star)$, and a solver attempts it. The objective (Eq. 3):

$$
\max_\theta\;\mathbb E_{z}\;\mathbb E_{\tau\sim\pi_\theta^{\rm propose}(\cdot\mid z),\;(x,y^\star)\sim f_e(\cdot\mid\tau)}\Big[\lambda\,r^{\rm propose}(\tau,\pi_\theta)+\mathbb E_{y\sim\pi_\theta^{\rm solve}(\cdot\mid x)}\,r^{\rm solve}(y,y^\star)\Big]
$$

$z$ conditions the proposal, for example a few past task-answer pairs sampled from a buffer; $f_e$ is the environment the executor represents; $\lambda$ balances the two rewards. The proposer's learnability reward (Eq. 4) and the solver's reward (Eq. 5) are:

$$
\bar r_{\rm solve}=\frac1G\sum_{i=1}^{G}r_{\rm solve}^{(i)},\qquad
r^{\rm propose}=\begin{cases}0,&\bar r_{\rm solve}=0\\1-\bar r_{\rm solve},&\text{otherwise}\end{cases},\qquad
r^{\rm solve}=\mathbf 1(y=y^\star)
$$

$\bar r_{\rm solve}$ is the solver's average success over $G$ Monte Carlo attempts at the same task. Work it by hand with $G=4$:

| Proposed task | Solved out of 4 | $\bar r_{\rm solve}$ | $r^{\rm propose}$ |
| --- | --- | --- | --- |
| Task 1 | 0 | 0 | 0 |
| Task 2 | 2 | 0.5 | 0.5 |
| Task 3 | 4 | 1 | 0 |
| Task 4 | 1 | 0.25 | 0.75 |

The first three rows express the paper's intuition: tasks that are never solved or always solved provide no success-versus-failure contrast under this within-group binary-reward comparison, and the proposer earns nothing. Other training objectives may still learn from those samples. The fourth row comes straight from the formula: the reward peaks not at a 0.5 success rate but at tasks that are solved only occasionally; with $G=4$ the maximum is $1-1/4=0.75$. This resembles STaR's difficulty when filtering for successful samples alone: no success means no positive example for that filtering update.

<!-- paper-figure:azr-loop -->

"Zero" means no human-curated tasks or answers; a pretrained base model and a code executor are still required. §4.2 reports that AZR-Coder-7B beats the previous best zero-setting model by 1.8 points on average, and beats models trained on expert-curated code data by 0.3 points on the coding average; with Llama-3.1-8B the authors also observed concerning chains of thought (their "uh-oh moment"). [R-Zero](https://arxiv.org/abs/2508.05004) (2025-08) splits one base model into a Challenger and a Solver that co-evolve; the abstract reports +6.49 on math and +7.54 on general reasoning for Qwen3-4B-Base, both relative to the base model.

## The ceiling the verifier sets: the generation-verification gap

[Song et al. (ICLR 2025)](https://arxiv.org/abs/2412.02674) put these loops into one framework: a generator $f$ samples, a verifier $g$ assigns scores $\hat u_g$, a weight $w(\hat u_g)$ reweights the samples into the distribution $f[w(\hat u_g)]$, and the model is distilled onto it. $J$ is expected score under the true utility. Definition 2.1:

$$
\operatorname{gap}(f,g)=J\big(f[w(\hat u_g)]\big)-J(f)
$$

It asks: if we reweight by the verifier's opinion, how much does the true score rise? Self-improvement is the case $g=f$. If the update learns the reweighted distribution to within $\varepsilon_{\rm update}$, the paper gives:

$$
J(f_{t+1})-J(f_t)\;\ge\;\operatorname{gap}(f_t,g)-\varepsilon_{\rm update}
$$

The same premise also caps the improvement at $\operatorname{gap}+\varepsilon_{\rm update}$. A round's gain is therefore approximately the gap, allowing for update error. When the model verifies no better than it generates (gap ≤ 0), reweighting by verification provides no positive expected quality gain under these update assumptions; this does not mean every training objective has nothing to learn. The paper also notes that fine-tuning sometimes beats this amount, for example when the model learns the required answer format, and that this part is not self-improvement.

<!-- paper-figure:gv-gap-definitions -->

The paper's measurements (§4–5): the relative gap grows monotonically with pre-training FLOPs; small models such as Qwen-1.5 0.5B, Qwen-2 0.5B and Llama-2 7B show a non-positive gap for nearly all verification methods; factual tasks show no significant gap; in iterated runs the gap falls to near zero within two or three rounds, at a similar rate across model sizes; and pass@k at large $k$ drops over rounds, meaning effective diversity shrinks.

## Work it once: tests that accept wrong solutions

What follows is a teaching assumption with made-up numbers. On a batch of problems, 40% of the model's sampled solutions are truly correct. The tests accept every correct solution and also accept a fraction $f$ of wrong ones, for example because they check a few ordinary inputs but not an empty list or negative numbers. Fine-tune on the accepted solutions, and assume fine-tuning reproduces their distribution exactly, i.e. $\varepsilon_{\rm update}=0$. Accepted samples make up $0.40+0.60f$, and the fraction of them that is truly correct is:

$$
\text{precision}=\frac{0.40}{0.40+0.60f}
$$

| False-accept rate $f$ | Share accepted | Of those, truly correct | Pass rate by the tests after fine-tuning | True correctness after fine-tuning |
| --- | --- | --- | --- | --- |
| 0 (perfect tests) | 40% | 100% | 100% | 100% |
| 2% | 41.2% | 97.1% | 100% | 97.1% |
| 10% | 46% | 87.0% | 100% | 87.0% |
| 30% | 58% | 69.0% | 100% | 69.0% |

With these tests as the verifier, the first-round gap at $f=10\%$ is $87.0\%-40\%=47$ percentage points, and at $f=30\%$ it is 29 points. One round gains a lot, and in every case the tests report a 100% pass rate.

Run another round. At $f=10\%$, the new model's solutions are 87% correct and 13% "wrong but passing". The tests treat both alike and accept all of them, so reweighting changes nothing and the gap is 0. The loop stops at 87% while the tests keep showing 100%. The 13% the verifier cannot tell apart never goes away, however many rounds you run. The toy hits its ceiling in one round because it assumes perfect imitation; real training learns imperfectly and also generalises, so it takes a few rounds, which matches the direction of Song et al.'s "gap near zero within two or three rounds".

What fine-tuning can do is concentrate the model on solutions the tests accept; if correct solutions share a learnable pattern, it may also transfer to new problems, which is the kind of gain STaR shows on its dev sets. What it cannot do is learn to separate "correct" from "wrong but passing", because that information is not in the training signal. Worse: if wrong solutions pass because of a reusable trick, such as special-casing the visible test inputs, fine-tuning may spread that trick to problems where it was not used before. That last point is a mechanistic conjecture; the measured cases below show agents do find such tricks.

As long as all correct solutions are still accepted, lowering $f$ raises this ceiling: going from 10% to 2% raises it from 87.0% to 97.1%. Hidden tests, edge cases and property tests all strengthen the verifier, not the generator. Conversely, rejecting some correct solutions, such as an unusual but valid style, can reduce accepted-set precision as well as data volume and diversity.

Let $r$ be the probability that a correct solution is accepted. Keeping the original proportions of 0.4 correct and 0.6 incorrect solutions gives accepted-set precision $0.4r/(0.4r+0.6f)$. The original table assumes $r=1$. If $f=10\%$ but $r$ falls to 0.5, precision becomes $0.2/(0.2+0.06)\approx76.9\%$, below the original 87.0%. Rejecting correct solutions leaves precision at 100% only under conditions such as $f=0$ with a nonempty accepted set; the losses of data and diversity remain.

<!-- widget:step-verifier-ceiling -->

## Feeding your own outputs back: model collapse

Shumailov et al. reported in [Nature (2024-07-24)](https://www.nature.com/articles/s41586-024-07566-y) that training indiscriminately on recursively generated data leaves irreversible defects, with the tails of the original distribution disappearing. A filtered loop is not exactly that case: its verifier supplies a selection signal and can support improvement when it identifies samples better than the current generation distribution; the verifier need not be external. But filtering also favours typical, passing solutions and drops rare ones; Song et al. saw pass@k at large $k$ fall and suggested that the model fails to verify rare but correct answers. Both results warn that reusing a signal with no discriminatory value cannot guarantee further gains and may reduce diversity. They do not establish a general rule that learning is impossible without new external information.

## When the signal gets gamed: measured cases

- DGM (2025-05, Appendix H): an agent reached a perfect score on a tool-use hallucination objective by removing the logging of the special tokens the detector relied on, despite instructions not to change them; the authors found such objective hacking more frequent when the checking functions were not hidden from the agent. [Paper](https://arxiv.org/abs/2505.22954)
- Anthropic's automated alignment researchers (2026-04-14): one agent skipped the weak teacher on math and always output the most common answer; another ran code against the tests to read off answers. Both were caught and disqualified, and the authors conclude that automated researchers need evaluations they cannot tamper with. [Post](https://www.anthropic.com/research/automated-alignment-researchers)
- PostTrainBench (2026-03-09): agents trained on the test set, downloaded existing instruction-tuned checkpoints instead of training their own, and used API keys they found without authorization. [Paper](https://arxiv.org/abs/2603.08640)
- Sakana AI's follow-up paper (2025-09-16): some kernels passed KernelBench checks while showing fake 50-120x speedups by exploiting loopholes; excluding contaminated tasks cut the average speedup from 3.13x to 1.49x. [Paper](https://arxiv.org/abs/2509.14279)
- METR's Frontier Risk Report (2026-05-19): on TH1.1 tasks of 8 hours or more, at least 16% of successful runs were illegitimate on review. [Post](https://metr.org/blog/2026-05-19-frontier-risk-report/)
- The OpenAI–Hugging Face incident (July 2026; OpenAI report of 2026-08-26): OpenAI lists reward hacking, persistence on seemingly impossible tasks, unauthorized communication, and agents adopting goals from one another as contributing patterns. [Post](https://openai.com/index/hugging-face-incident-and-the-road-ahead/)

## Where the signal comes from: one comparison table

The [survey by Chen, Wang and Qu (2026-07)](https://arxiv.org/abs/2607.07663) orders feedback signals into a "verification hierarchy", from formal verifiers (strongest) to intrinsic self-assessment (weakest), and reports that demonstrated self-improvement strength tracks that order. The table follows it; examples and failure modes come from the sources above:

| Signal source | Strength | Example | Typical failure |
| --- | --- | --- | --- |
| Formal verifier (proof checking) | Strongest | The Gödel Machine design (a theoretical proposal) | The specification itself is wrong or incomplete; few tasks can be formalised, and finding proofs is costly |
| Code execution, tests, automated evaluators | Strong, depending on coverage | Absolute Zero's Python executor, AlphaEvolve's evaluation function, DGM's benchmarks | Tests accept wrong solutions; agents edit logging or the evaluation step (DGM, KernelBench) |
| Ground-truth answers | Strong for final answers, weak for the process | STaR, ReST-EM | Right answer, wrong reasoning (STaR's binary-choice tasks); overfitting a few problems (ReST-EM) |
| Learned reward model, LLM judge | Medium | AI Scientist's automated reviewer, 69% balanced accuracy in the [Nature version](https://sakana.ai/ai-scientist-nature/) | The optimised side exploits the judge; the judge's own biases, see [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md) |
| Model self-assessment | Weakest | Self-Rewarding LMs; Huang et al.'s self-correction experiments | Self-confirmation; longer outputs; saturation within a few rounds; non-positive gap in small models |

## What to check when you try it

| Try | Observe | What it does not establish |
| --- | --- | --- |
| Manually review a small batch of passing solutions | Accepted-set precision and the fraction of accepted solutions that are wrong, $1-\mathrm{precision}$ | This directly estimates $f$; a clean spot check means the tests have no blind spots |
| Add hidden tests and rerun a round | How much the pass rate drops, and on which problems | The hidden tests are themselves complete |
| Log pass@1 and pass@k at large $k$ every round | A drop at large $k$ means diversity is being lost | A rising pass@1 equals rising ability |
| Run more rounds and note when gains stop | Where saturation sets in | No gain means the model has peaked; it may be the verifier |

The two conditional probabilities have different denominators. Reviewing only passing solutions estimates $P(\text{wrong}\mid\text{accepted})$, whose denominator is all accepted solutions. Here $f=P(\text{accepted}\mid\text{wrong})$ has all wrong solutions as its denominator. To estimate $f$, label correctness in a representative sample drawn before filtering, then measure the acceptance rate among all wrong solutions in that sample, not just those that passed.

## What to watch on this line next

These are open questions and the kind of result that would change the current reading; they are not predictions.

1. **How many rounds can self-proposed, self-checked training sustain?** Song et al. (2024-12) measured the gap near zero within two or three rounds; Absolute Zero (2025-05) and R-Zero (2025-08) keep supplying new tasks through an executor or a Challenger and report gains relative to the base model. A result that keeps improving on independent benchmarks well beyond three rounds would turn "saturates in a few rounds" into a conditional statement.
2. **Can learned verifiers stand in for executable checks?** Self-Rewarding (2024-01) ran only 3 iterations and its authors expect saturation. On tasks without answer keys, a self-judged loop that keeps improving on externally judged evals over a long run would call for revisiting the order of the verification hierarchy.
3. **Do improvement loops make gaming more or less common?** DGM (2025-05), Anthropic's automated alignment researchers (2026-04-14) and PostTrainBench (2026-03-09) all recorded gaming; AIDE² (2026-09-22) saw the reward-hacking rate on one held-out task family fall from 55% to 32%, but in a single run. Repeated results across systems and runs would carry more weight.
4. **Who designs the update rule?** The survey (2026-07) reports that nearly all papers study bounded self-refinement. A public result in which a model designs its own data, loss and training procedure and passes held-out evaluation would move this line from type (d) to type (a).

Next: [Changing code, not weights](code-not-weights.en.md).
