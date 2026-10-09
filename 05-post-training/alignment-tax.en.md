# The alignment tax: what you lose by becoming agreeable

[中文](alignment-tax.md) · **English**

> Reading time: ~7 min · Type: chapter · Last reviewed: 2026-08

## Alignment changes the distribution of capability {#alignment-changes-the-distribution-of-capability}

After fine-tuning, a model might follow a format better but refuse more harmless requests. It might instead improve at coding without material regressions elsewhere. Both require measurement; alignment does not imply either inevitable degradation or a free improvement.

Alignment tax generally refers to performance regressions incurred while improving other objectives. Reduced diversity is one possible cost, not the entire definition.

## What the tax looks like {#what-the-tax-looks-like}

**Reduced output diversity.** Repeated responses to the same prompt may become more alike. Distinguish wording, meaning, and genuinely different valid solutions rather than just counting distinct words.

**Falling entropy.** The policy may become more concentrated. For a low-probability action with positive advantage, a fixed ratio threshold corresponds to a small absolute increase, potentially limiting the incentive from that sample. Clipping is neither a hard probability bound nor the only cause of falling entropy. [After PPO](after-ppo.en.md) discusses alternative clipping choices.

**Formulaic responses.** Repeated preferences for particular wording or formats can make them more common. Check whether they displace more direct, useful answers rather than treating every style change as a regression.

**Over-refusal.** If data and rewards fail to distinguish harmful requests from superficially similar harmless ones, both may be refused. Test those harmless cases separately instead of treating every refusal as either a success or a failure.

## When does probability concentrate? {#why-this-is-close-to-inevitable}

Consider a deliberately simple objective: a prompt permits answers A and B with fixed rewards 1 and 0.8. Without other constraints, expected reward is $p\times1+(1-p)\times0.8$, so increasing A's probability always raises it.

This explains why unregularized expected-reward maximization can concentrate on a few high-reward answers. It does not prove that practical RLHF must collapse. KL or entropy terms change the objective. For a finite answer set, fixed rewards, $\beta>0$, and reference support on the relevant answers:

$$\max_\pi\ \mathbb E_\pi[r]-\beta D_{\mathrm{KL}}(\pi\|\pi_{\rm ref})
\quad\Longrightarrow\quad
\pi^*(y)\propto\pi_{\rm ref}(y)e^{r(y)/\beta}.$$

The result depends on reward gaps, the reference, and $\beta$; it need not be a point mass. SFT on narrow data can also lose diversity. Compare the actual data, objective, and decoding settings rather than inferring the outcome from an algorithm name.

## But it's a trade, not a pure loss {#but-its-a-trade-not-a-pure-loss}

Some experiments observe better task generalization alongside lower output diversity. That does not make one a necessary price for the other in every run. [InstructGPT](https://arxiv.org/abs/2203.02155) also reports ways to mitigate some benchmark regressions by changing the training procedure.

For your task, ask whether a regression repeats, whom it affects, and whether it persists after changing data mixtures, regularization, or decoding. Those comparisons help distinguish fixable implementation choices from more persistent objective conflicts.

## How to measure it: no single score will show you {#how-to-measure-it-no-single-score-will-show-you}

For independent samples on one task with success probability $p$, the probability of at least one success is $1-(1-p)^k$. For that fixed task and sampling distribution, increasing $p$ increases both quantities.

If dataset-average pass@1 rises while average pass@k falls, it does not directly prove collapse to a point. Gains may concentrate on easy tasks while coverage of hard tasks falls. Sampling procedures and estimation error also matter.

| Question | Check |
| --- | --- |
| Which tasks improved or regressed? | Paired comparisons at equal budgets, sliced by difficulty and type |
| Do additional attempts still help? | pass@1, pass@k, and the distribution of per-task success rates |
| Is diversity only different wording? | Separate lexical variation, semantic variation, and distinct valid solutions |
| Are harmless requests refused? | A dedicated benign set with review of refusal reasons |
| Did decoding create the difference? | Fix temperature, length limits, and candidate budgets, then test sensitivity |

Each measure answers a different question. Applications that can try several times especially need to compare coverage of valid solutions with the cost per attempt.

## One practical recommendation {#one-practical-recommendation}

Before training starts, measure all of the above on the base model and **save it as a baseline.**

Alignment tax can only be measured against a pre-training baseline. By the time outputs start feeling samey, that baseline may be gone: checkpoints were cleaned up, or the evaluation configuration changed and the comparison is no longer valid.

This is the same point as [freezing the evaluation protocol](../07-evaluation/): **a stable measurement baseline must exist before improvement means anything.**

## Down to a checklist {#down-to-a-checklist}

1. Am I reporting pass@1 or pass@k? Which way is the gap between them moving?
2. Do I have a diversity baseline from the base model, or only post-alignment numbers?
3. What does generation entropy look like per training step? Monotone decline to a plateau?
4. Is output length climbing monotonically through training? If so, more content or just denser hedging?
5. Do I have a clearly-benign control set specifically for over-refusal?
6. What set my KL coefficient — a measurement, or a number I copied?

## Where to read next {#where-to-read-next}

- [After PPO](after-ppo.en.md): the mechanism linking symmetric clipping to entropy collapse
- [Where preferences come from](where-preferences-come-from.en.md): which step the bias entered at
- [How far one base model can go](same-base-different-posttraining.en.md): a live case of "no regressions mentioned" not meaning "no regressions"
- [Evaluation](../07-evaluation/): why one aggregate score is never enough

## Starting papers {#starting-papers}

- [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) — one origin of the term "alignment tax"
- [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://arxiv.org/abs/2310.06452) — the systematic generalization-up, diversity-down comparison
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL penalties and over-optimization

## Quick learning: alignment tax is not one score {#quick-learning-alignment-tax-is-not-one-score}

<details class="interview" markdown="1">
<summary>Capability redistribution, Pareto frontiers, and sliced evaluation</summary>

**Quick memory**: improving helpfulness or safety may reduce calibration, diversity, exploration, or some base capabilities. These are multiple trade-offs, not one tax rate.

**Interview answer**

> Alignment redistributes output probability from some behaviors to others. Evaluation cannot rely on an average win rate; it should slice by task, risk, language, length, refusal, and capability while tracking the quality–safety Pareto frontier.

<details markdown="1">
<summary><b>Deep dive</b>: lost capability or suppressed capability?</summary>

Compare base and aligned models under alternative system prompts, decoding settings, and controlled elicitation. A drop only in default behavior may reflect an access-policy change; failure across many elicitation methods is stronger evidence of forgetting or parameter interference.

</details>
</details>
