# Verifiable rewards: did passing the tests complete the task?

[中文](verifiable-rewards.md) · **English**

> Reading time: ~11 min · Type: chapter · Last reviewed: 2026-10

## Replace a learned reward with a checkable rule {#replace-a-learned-reward-with-a-checkable-rule}

A model writes a sorting function. You could ask another model whether it looks good, or execute tests. Tests make part of the judgment reproducible.

If the only test is `[3, 1, 2]`, always returning `[1, 2, 3]` passes. Empty lists, repeated elements, or negative values expose the error. Verifiable rewards offer explicit, repeatable checks, not automatic immunity to loopholes.

## What gets swapped out {#what-gets-swapped-out}

Recall [RLHF's four roles](rlhf/three-stages.en.md): Policy, Critic, Reward, Reference. For objectives that admit explicit checks, a verifier can replace the Reward Model learned from [preference data](where-preferences-come-from.en.md). Other objectives can still use human or model judgment; the entire system need not use one reward source.

$$r(x,y) = \operatorname{verify}(x,y) \in \{0, 1\}$$

Here x includes the task and checking context, and y is the candidate answer. Binary reward is one option; fractions of tests passed or multiple component scores are possible too. Reproducibility depends on fixed test inputs, environment, random seeds, and timeout rules. The reward source **does not determine PPO versus GRPO, or whether a Critic is trained**.

## Why it narrows reward hacking {#why-it-narrows-reward-hacking}

Replacing an RM with tests removes the need to infer code correctness from how the answer looks. The checked properties are more explicit, and failures are easier to reproduce. This avoids some reward-model generalization errors.

A discrete, nondifferentiable reward is not protection against exploitation. Policy gradients adjust sampled-output probabilities using their rewards; they do not need gradients through the test program. An incorrect implementation that passes an incomplete test suite can still be reinforced.

## A rule can miss the actual requirement {#but-reward-hacking-only-moved}

Reward hacking describes behavior that earns more proxy reward without delivering the intended result. The proxy need not be a neural network: handwritten rules can give the wrong incentive too.

Incomplete checks can reinforce unwanted behavior, including:

- hardcode returns for the specific test cases;
- swallow every exception so the program doesn't crash, satisfying "doesn't error" style checks;
- find bugs in the test harness itself.

For math, matching the final answer does not establish that every intermediate step is valid. A guess or an incorrect derivation can happen to reach the right result. Whether the process also needs checking depends on whether the task requires an answer or a dependable argument.

This does not mean every model will discover every loophole. The reward supplies an optimization direction; exploiting it also depends on capability, sampling, and training. Check whether independent measurements support an improvement instead of inferring one from rising training reward.

## What did the sorting check forget? {#reward-entropy-example}

The input is `[3, 1, 1]`. A verifier that only checks ascending order accepts all three outputs:

| Output | Nondecreasing? | Preserves every input element and its count? |
| --- | --- | --- |
| `[1, 1, 3]` | Yes | Yes |
| `[1, 3]` | Yes | No: one 1 is missing |
| `[]` | Yes: no adjacent inversion | No: every element is missing |

Sorting requires both order and preservation of the input multiset. For this integer-list example:

```python
from collections import Counter

def verifies_sort(original, candidate):
    ordered = all(
        left <= right
        for left, right in zip(candidate, candidate[1:])
    )
    return ordered and Counter(original) == Counter(candidate)
```

This teaching function checks **one input/output pair**, not all possible inputs, and does not safely execute unknown programs. A real evaluation also needs contracts for types, input mutation, exceptions, timeouts, and isolation.

Now let a policy choose equally between `[1, 3]` and `[]`. Its outputs vary, with entropy $\log2$, but neither solves the task. Another policy could always return the correct `[1, 1, 3]`, with entropy 0. **Reward loopholes and falling entropy are separate things to check.** The [entropy example](alignment-tax.en.md#entropy-versus-hacking) lays out that distinction.

<details markdown="1">
<summary>How do you test the verifier itself?</summary>

Start with deliberately wrong outputs a person can judge directly: missing elements, duplicated elements, reversed order, and empty results. They should not receive full reward just because they resemble the expected format. Add correct edge cases too: empty input with empty output, repeated values, and negative numbers.

Keep separate training checks and held-out checks that did not participate in training or selection. A holdout is not automatically independent: shared templates or parser bugs can affect both. Add checks targeting different failure mechanisms and review a sample manually. When changing the scoring rules, re-evaluate old checkpoints as well, so a changed ruler is not mistaken for a better model.

</details>

## A new problem: binary rewards are sparse {#a-new-problem-binary-rewards-are-sparse}

Return to sorting: four implementations of the same task receive `[0, 0, 0, 0]`. If training uses only outcome rewards centered by the group mean, all four advantages are zero. This term cannot distinguish which output deserves more probability. `[0, 1, 0, 0]` does provide a relative signal.

This concerns the outcome-reward policy-gradient term. KL, entropy, or other auxiliary objectives can still produce updates. Programmatic rewards need not be binary: a fraction of tests passed is denser, but changes the objective and may reward solving only easy parts.

Task difficulty and sampling budget matter. Measure all-correct and all-incorrect groups before deciding whether you need different tasks, a stronger starting policy, or more attempts.

For one fixed task with independent success probability p per attempt and G answers per group, the probability of a homogeneous group is:

$$P(\text{all equal})=p^G+(1-p)^G.$$

With G=4, p=0.5 gives 12.5%; p=0.1 or 0.9 gives 65.62%. Both very hard and very easy tasks can provide little within-group discrimination. This assumes independent, identically distributed draws. When difficulty varies or samples are correlated, measure groups per task rather than plugging the dataset-wide mean p into the formula.

Filtering homogeneous groups, as in [DAPO](dapo.en.md), increases the share of discriminative groups used for updates. It costs additional sampling and changes which prompts enter the update. It cannot create a successful solution to a hard task from nothing. As with [filtered SFT](rejection-sampling.en.md#prompt-selection), track prompt coverage as well as retention.

## Outcome rewards or process rewards {#outcome-rewards-or-process-rewards}

Final-outcome checks are straightforward to integrate, but do not locate which step went wrong. Process rewards give finer feedback only if the intermediate judgments are reliable. Human annotation costs time, model annotation can misjudge, and formal checks require suitable representations and tools.

The two can be combined: tests check the final result while process signals guide search. Evaluate whether the extra signal improves equal-budget success, rather than merely increasing process scores.

## Verifiability is a spectrum, not a binary {#verifiability-is-a-spectrum-not-a-binary}

It is more useful to ask exactly what each check establishes:

| Check | What it establishes | What it does not establish alone |
| --- | --- | --- |
| Successful compilation | Satisfies the compiler's syntax and type checks | Completes the task |
| Passing unit tests | Matches expectations on those inputs | Correct on all inputs |
| Matching a math answer | Final result matches the answer and parsing rules | Valid intermediate reasoning |
| An existing citation | The source can be found | It supports the claim |
| Human or model judgment | A judgment under specified evidence and criteria | Absence of bias or disagreement |

A task can use several checks. For a report, code might verify numbers and links while a person reviews the argument. Using suitable evidence for each judgment makes failures easier to diagnose than collapsing every requirement into one score.

## Down to a checklist {#down-to-a-checklist}

1. Which deliberately wrong outputs still receive full reward? Can targeted counterexample tests expose them?
2. If the reward is binary, what fraction of my groups is all-right or all-wrong?
3. Am I verifying outcome or process? If outcome, how often does it get there by guessing?
4. Is my task really "unverifiable," or have I just not decomposed it? Which part is checkable?
5. If I add model judgment, can independent checks catch the errors that judge favors?

## Where to read next {#where-to-read-next}

- [Where preferences come from](where-preferences-come-from.en.md): the structural ceilings of a learned reward model
- [After PPO](after-ppo.en.md): how binary sparse rewards and group baselines amplify each other
- [The alignment tax](alignment-tax.en.md): check entropy, solution coverage, and actual quality separately
- [Evaluation](../07-evaluation/): a verifier and an eval set are not the same thing

## Starting papers {#starting-papers}

- [DeepSeek-R1, §2.2.2](https://arxiv.org/html/2501.12948v1#S2.SS2.SSS2) — a concrete design using rule-based accuracy and format rewards
- [DAPO](https://arxiv.org/abs/2503.14476) — dynamic sampling for all-right/all-wrong groups
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process versus outcome supervision
- The sorting and probability examples are checked by the [standard-library teaching script](code/selection_and_entropy.py). It neither executes model-generated programs nor reproduces training results.

## Quick learning: what does a verifiable reward replace? {#quick-learning-what-does-a-verifiable-reward-replace}

<details class="interview" markdown="1">
<summary>For review: explain the reward's limits with the sorting check</summary>

**Remember**: a test establishes specific properties, not necessarily the whole task. High reward, low entropy, and high quality are different measurements.

**Interview answer**

> I would state what the verifier checks, then construct an answer that passes while violating the task. Checking sorted order, for instance, must also preserve element counts. I would then evaluate models with held-out tests and a fixed sampling budget rather than reporting training reward alone.

<details markdown="1">
<summary><b>Deep dive</b>: outcome reward or process reward?</summary>

Outcome-reward reliability depends on the specification and verifier coverage; it is not inherently less biased. Process rewards give finer feedback but add judgments that also need validation. Both need independent quality checks.

</details>
</details>
