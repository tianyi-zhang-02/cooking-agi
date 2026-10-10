# Alignment tax: what else changed when the model improved?

[中文](alignment-tax.md) · **English**

> Reading time: ~13 min · Type: chapter · Last reviewed: 2026-10

## Better at one task does not settle the rest {#alignment-changes-the-distribution-of-capability}

After fine-tuning, a model might follow a format better but refuse more harmless requests. It might instead improve at coding without material regressions elsewhere. Both require measurement; alignment does not imply either inevitable degradation or a free improvement.

Alignment tax generally refers to performance regressions incurred while improving other objectives. Reduced diversity is one possible cost, not the entire definition.

This chapter starts with possible changes, then works through the easily misread case of falling entropy. If you are investigating a training run, jump to [controlled comparisons](#entropy-diagnostics).

## Changes worth checking beyond the average score {#what-the-tax-looks-like}

**Reduced output diversity.** Repeated responses to the same prompt may become more alike. Distinguish wording, meaning, and genuinely different valid solutions rather than just counting distinct words.

**Falling entropy.** The policy may become more concentrated. For a low-probability action with positive advantage, a fixed ratio threshold corresponds to a small absolute increase, potentially limiting the incentive from that sample. Clipping is neither a hard probability bound nor the only cause of falling entropy. [After PPO](after-ppo.en.md) discusses alternative clipping choices.

**Formulaic responses.** Repeated preferences for particular wording or formats can make them more common. Check whether they displace more direct, useful answers rather than treating every style change as a regression.

**Over-refusal.** If data and rewards fail to distinguish harmful requests from superficially similar harmless ones, both may be refused. Test those harmless cases separately instead of treating every refusal as either a success or a failure.

## Falling entropy is not automatically collapse {#entropy-measurement}

A task has two possible answers, A and B. Their probabilities move from 0.5 each to 0.9 for A and 0.1 for B. The model is more certain. Whether that is a problem depends on A: a correct answer suggests learning; a wrong answer favored by the scorer suggests a different story.

At a fixed context h, next-token entropy is:

$$
\begin{aligned}
H(\pi\mid h)
&=-\sum_v p_v\log p_v,\\
p_v&=\pi(v\mid h).
\end{aligned}
$$

The sum covers the vocabulary. Natural logarithms give units of nats. Two equally likely choices have entropy $\log2\approx0.693$; probabilities $(0.9,0.1)$ give about 0.325. **Entropy measures concentration, not the quality of what receives the probability.**

Concern about entropy collapse usually goes beyond a falling number: useful exploration may disappear too early, leaving additional attempts unable to find different valid solutions. Entropy, independent quality, and coverage under multiple attempts need to be considered together.

| Logged quantity | What it measures | What it does not tell you |
| --- | --- | --- |
| Full-vocabulary entropy | Next-token uncertainty at a fixed prefix | Semantic diversity of complete reasoning paths |
| Negative log probability of a sampled token | That sample's surprisal | One observation is not entropy; its expectation under the matching distribution is |
| Mean token entropy | Average local uncertainty across a set of prefixes | Its meaning depends on token/response weighting and masks |
| Number of distinct answers or methods | Observed variation under a sampling budget | Different wording need not mean different methods; wrong answers can be diverse |

Under $(0.9,0.1)$, A has surprisal about 0.105 and B about 2.303. Neither is 0.325. If tokens are sampled from a top-p-truncated distribution q but scored with the original model p, their mean negative log probability estimates cross-entropy, not p's own entropy.

Aggregation matters too. A two-token response with mean entropy 1 and an eight-token response with mean entropy 0.1 give a response-weighted mean of 0.55, but a token-weighted mean of 0.28. If response lengths change, those curves can tell different stories. Exclude padding and keep the convention for valid EOS positions fixed.

## Entropy and reward hacking answer different questions {#entropy-versus-hacking}

Consider four deliberately simple policies. Each chooses between two outputs whose actual task success is known independently of the training scorer:

| Output choices | Probabilities | Entropy (nats) | Actual success probability |
| --- | --- | --- | --- |
| Correct A / correct B | 0.9 / 0.1 | 0.325 | 1 |
| Correct A / correct B | 0.5 / 0.5 | 0.693 | 1 |
| Loophole A / loophole B | 0.9 / 0.1 | 0.325 | 0 |
| Loophole A / loophole B | 0.5 / 0.5 | 0.693 | 0 |

A flawed scorer could give every output full reward, making all four rows indistinguishable by training reward. A high-entropy policy can exploit several loopholes; a low-entropy policy can answer correctly and consistently. Raising entropy is not a substitute for repairing the reward. The [verifier example](verifiable-rewards.en.md#reward-entropy-example) makes this concrete.

## When does probability concentrate? {#why-this-is-close-to-inevitable}

Consider a deliberately simple objective: a prompt permits answers A and B with fixed rewards 1 and 0.8. Without other constraints, expected reward is $p\times1+(1-p)\times0.8$, so increasing A's probability always raises it.

This explains why unregularized expected-reward maximization can concentrate on a few high-reward answers. It does not prove that practical RLHF must collapse. KL or entropy terms change the objective. For a finite answer set, fixed rewards, $\beta>0$, and reference support on the relevant answers:

$$\max_\pi\ \mathbb E_\pi[r]-\beta D_{\mathrm{KL}}(\pi\|\pi_{\rm ref})$$

The corresponding optimum is:

$$\pi^*(y)\propto\pi_{\rm ref}(y)e^{r(y)/\beta}.$$

The result depends on reward gaps, the reference, and $\beta$; it need not be a point mass. SFT on narrow data can also lose diversity. Compare the actual data, objective, and decoding settings rather than inferring the outcome from an algorithm name.

<details markdown="1">
<summary>One more calculation: learning a correct answer can raise or lower entropy</summary>

Let the probability of correct answer A be $p=\sigma(\theta)$, with reward 1 for A and 0 for B. Expected reward is J=p, so gradient ascent gives $d\theta/dt=p(1-p)$. Binary entropy satisfies:

$$\frac{dH}{dp}=\log\frac{1-p}{p}.$$

For a small update:

$$
\begin{gathered}
\frac{dH}{dt}
=\frac{dH}{dp}\frac{dp}{d\theta}\frac{d\theta}{dt}\\
=[p(1-p)]^2\log\frac{1-p}{p}.
\end{gathered}
$$

At p=0.2, increasing the probability of the correct answer moves the distribution toward uniformity, raising entropy. At p=0.8, the same direction of improvement concentrates it further, lowering entropy. Accuracy improves in both cases. This is a teaching derivation for one independent binary parameter, not a complete law of shared-parameter LLM training.

</details>

## What can a paper's entropy curve establish? {#entropy-research}

[The Entropy Mechanism of RL (2025, v1), §2–4](https://arxiv.org/html/2505.22617v1) studies early entropy reduction and performance plateaus on math and code tasks, and proposes Clip-Cov / KL-Cov. It also discusses differing behavior under other model and data settings. Its step counts and fitted relationship are not universal training laws.

In that version's Eq. 14, KL-Cov uses the rollout old policy and current policy, not a fixed SFT Reference. Taking the absolute difference of selected-token log probabilities is not equivalent to KL either. An implementation needs explicit distributions, direction, sampling, and masks. See [KL estimation](rlhf/kl-estimators.en.md).

## Hold the comparison still before changing the algorithm {#entropy-diagnostics}

Keep a fixed set of prompts and prefixes. First compare checkpoints' full-vocabulary entropy at **those same prefixes**; then let each checkpoint generate and measure quality and exploration. The former isolates conditional-distribution changes more directly. The latter includes changes in which contexts the model visits. They should not be treated as the same curve.

| Observation | First comparison | Possible next step |
| --- | --- | --- |
| Entropy falls while success improves | Check distinct valid methods at equal budgets | Low entropy alone need not require intervention |
| Entropy and valid-solution coverage both fall | Fix prompts, sampling, and lengths; slice by difficulty | Inspect duplicates, excessive updates, sampling, and truncation |
| Training reward rises while independent quality falls | Re-evaluate using checks not used for selection | Repair rewards or data; entropy bonus alone is insufficient |
| Only rollout-log entropy changes | Recompute at fixed prefixes; check temperature, top-p, and masks | Rule out measurement and sampling changes |
| An intervention raises entropy | Measure equal-budget success, cost, and failure types | Higher entropy alone is not success |

Small ablations can separately change learning rate or update count, data mixtures, KL/entropy coefficients, or clipping bounds. Change one factor at a time and record validation quality and generation cost. Raising temperature changes sampling, not the weights. Stronger KL can preserve old behavior but impede useful changes; an entropy bonus can encourage unhelpful randomness. Task outcomes decide whether the intervention helps.

The [teaching script](code/selection_and_entropy.py) checks the entropy values, weighting schemes, and binary update used here. It does not run LLM training or reproduce DAPO or KL-Cov results.

## Once you see a regression, investigate its source {#but-its-a-trade-not-a-pure-loss}

Some experiments observe better task generalization alongside lower output diversity. That does not make one a necessary price for the other in every run. [InstructGPT](https://arxiv.org/abs/2203.02155) also reports ways to mitigate some benchmark regressions by changing the training procedure.

For your task, ask whether a regression repeats, whom it affects, and whether it persists after changing data mixtures, regularization, or decoding. Those comparisons help distinguish fixable implementation choices from more persistent objective conflicts.

## Succeeding once versus succeeding with several attempts {#how-to-measure-it-no-single-score-will-show-you}

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

A regression needs a meaningful control: often the checkpoint immediately before this training stage, or a controlled ablation. Keeping that checkpoint and its evaluation configuration is easier than trying to reconstruct the comparison after outputs start feeling repetitive.

This is the same point as [freezing the evaluation protocol](../07-evaluation/): **a stable measurement baseline must exist before improvement means anything.**

## Down to a checklist {#down-to-a-checklist}

1. Am I reporting pass@1 or pass@k? Are averages hiding opposite changes across tasks?
2. Do I have a diversity baseline from the base model, or only post-alignment numbers?
3. Which distribution, prefixes, and denominator define entropy? Is falling entropy accompanied by worse quality?
4. Are longer responses adding useful reasoning or repeating themselves?
5. Do I have a clearly-benign control set specifically for over-refusal?
6. What set my KL coefficient — a measurement, or a number I copied?

## Where to read next {#where-to-read-next}

- [After PPO](after-ppo.en.md): how clipping, group rewards, and gradients affect updates
- [Where preferences come from](where-preferences-come-from.en.md): which step the bias entered at
- [How far one base model can go](same-base-different-posttraining.en.md): a live case of "no regressions mentioned" not meaning "no regressions"
- [Evaluation](../07-evaluation/): measure task improvements and regressions separately

## Starting papers {#starting-papers}

- [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) — one origin of the term "alignment tax"
- [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://arxiv.org/abs/2310.06452) — the systematic generalization-up, diversity-down comparison
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL penalties and over-optimization

## Quick learning: alignment tax is not one score {#quick-learning-alignment-tax-is-not-one-score}

<details class="interview" markdown="1">
<summary>For review: separate more certain from more correct</summary>

**Remember**: entropy describes a distribution, not answer quality. Assessing a regression needs a control, independent quality checks, and an account of which tasks changed.

**Interview answer**

> I would compare checkpoints under a fixed budget, tracking success, valid-solution coverage, and benign refusals separately. Falling entropy is a diagnostic clue. It becomes a problem to address when the task-level evidence shows something useful was lost.

<details markdown="1">
<summary>Was a capability lost, or is it no longer the default behavior?</summary>

Compare controlled system prompts and decoding settings. If a different prompt restores performance, calling the change forgetting is premature. Failure across several settings warrants closer investigation of parameter interference and data coverage, although several failed attempts still do not prove that a capability no longer exists.

</details>
</details>
