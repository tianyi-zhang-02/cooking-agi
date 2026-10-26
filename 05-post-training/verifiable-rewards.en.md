# Verifiable rewards: did passing the tests complete the task?

[中文](verifiable-rewards.md) · **English**

> Reading time: ~11 min · Type: chapter · Last reviewed: 2026-10

## Replace a learned reward with a checkable rule {#replace-a-learned-reward-with-a-checkable-rule}

A model writes a sorting function. You could ask another model to assess it, or run tests. Tests turn part of that assessment into a repeatable check.

If the only test is `[3, 1, 2]`, always returning `[1, 2, 3]` passes. Empty lists, repeated elements, or negative values expose the error. Using such checks as rewards is called Reinforcement Learning with Verifiable Rewards (RLVR). The benefit is explicit, repeatable assessment, not immunity to loopholes.

## What gets swapped out {#what-gets-swapped-out}

Recall the [four roles in RL from Human Feedback (RLHF)](rlhf/three-stages.en.md): the policy generates responses, the critic estimates returns, the reward model scores responses, and the reference model provides a baseline for constraining policy changes. For objectives with explicit checks, a verifier can replace the reward model learned from [preference data](where-preferences-come-from.en.md). Other objectives can still use human or model judgment; a system need not rely on a single reward source.

$$r(x,y) = \operatorname{verify}(x,y) \in \{0, 1\}$$

Here $x$ contains the task and the context needed to check it, and $y$ is the candidate answer. A binary reward is one option; the fraction of tests passed or multiple component scores are alternatives. Reproducibility also depends on fixed test inputs, environment, random seeds, and timeout rules. Verifiable rewards describe the source of feedback. They can be used with Proximal Policy Optimization (PPO) or Group Relative Policy Optimization (GRPO): **the reward source determines neither the optimization algorithm nor whether a critic must be trained**.

## Why tests can reduce some forms of reward hacking {#why-it-narrows-reward-hacking}

Replacing a reward model with tests lets you check aspects of the code's behavior directly, rather than relying only on another model's assessment of the response. The checked properties are explicit, and failures are easier to reproduce. This can avoid some reward-model generalization errors.

A discrete, nondifferentiable reward does not prevent exploitation. Policy gradients use the rewards of sampled outputs to adjust their generation probabilities; they do not need to differentiate through the test program. An incorrect implementation that passes an incomplete test suite can still be reinforced.

## A rule can miss the actual requirement {#but-reward-hacking-only-moved}

Reward hacking occurs when behavior earns a higher proxy reward without better serving the intended goal. The proxy need not be a neural network: handwritten rules can create the wrong incentives too.

Incomplete checks can reinforce behavior such as:

- hardcoding return values for specific test cases;
- swallowing every exception to pass checks that only require the program not to crash;
- exploiting bugs in the test harness itself.

For documented exploits in agent tasks, see the frontier draft [Agent RL: where environments and rewards come from](../11-frontier/agentic-rl/environments-and-rewards.en.md). A separate distinction matters here: improved accuracy after training does not, by itself, establish that the reward supplied useful information about correctness.

In [Spurious Rewards (Shao et al., 2025, §§2–3 and Appendix A)](https://arxiv.org/abs/2506.10947), the authors train Qwen2.5-Math-7B on DeepScaleR with GRPO, replacing correctness rewards with random binary rewards. MATH-500 single-answer accuracy still improves by 21.4 percentage points, versus 29.1 with ground-truth rewards. The tested Llama3 and OLMo2 models generally do not show comparable gains. This is a model- and setup-specific analysis, not evidence that random rewards work in general or direct evidence of reward hacking.

Read those gains alongside the setup: 16 responses per prompt, training temperature 1, no Kullback–Leibler (KL) divergence or entropy loss, and eight A100 GPUs. MATH-500 evaluation uses greedy decoding; the reported gains use the paper's ten-step smoothing at the end of the training curve.

In math, matching the final answer does not establish that every intermediate step is valid. A guess or an incorrect derivation can happen to reach the right result. Whether the reasoning also needs checking depends on whether the task requires a correct answer or a dependable argument.

This does not mean every model will discover every loophole. The reward supplies an optimization direction; exploiting it also depends on capability, sampling, and training. Check whether independent measurements support an improvement instead of inferring one from rising training reward.

## What did the sorting check forget? {#reward-entropy-example}

The input is `[3, 1, 1]`. A verifier that only checks nondecreasing order accepts all three outputs:

| Output | Nondecreasing? | Preserves every input element and its count? |
| --- | --- | --- |
| `[1, 1, 3]` | Yes | Yes |
| `[1, 3]` | Yes | No: one 1 is missing |
| `[]` | Yes: no adjacent inversion | No: every element is missing |

Sorting requires both order and preservation of the input multiset: every element must appear the same number of times. For this integer-list example:

```python
from collections import Counter

def verifies_sort(original, candidate):
    ordered = all(
        left <= right
        for left, right in zip(candidate, candidate[1:])
    )
    return ordered and Counter(original) == Counter(candidate)
```

This teaching function checks **one input/output pair**, not all possible inputs, and does not provide a safe way to execute unknown programs. A real evaluation must also specify allowed types, whether input mutation is permitted, how exceptions and timeouts are handled, and how execution is isolated.

Now let a policy choose equally between `[1, 3]` and `[]`. Its output entropy—a measure of uncertainty over outputs for this fixed input—is $\log2$, but neither output solves the task. Another policy could always return the correct `[1, 1, 3]`, with entropy 0. **Reward loopholes and falling entropy need separate checks.** The [entropy example](alignment-tax.en.md#entropy-versus-hacking) compares the two.

<details markdown="1">
<summary>How do you test the verifier itself?</summary>

Start with deliberately wrong outputs a person can judge directly: missing elements, duplicated elements, reversed order, and empty results. They should not receive full reward just because they resemble the expected format. Add correct edge cases too: empty input with empty output, repeated values, and negative numbers.

Keep training checks separate from held-out checks that were not used for training or selection. Holding tests out does not guarantee independence: shared templates or parser bugs can make both sets fail in the same way. Add checks targeting different failure mechanisms and review a sample manually. After changing the scoring rules, re-evaluate earlier model checkpoints too, so that a change in scoring is not mistaken for a model improvement.

</details>

## When binary rewards provide no within-group signal {#a-new-problem-binary-rewards-are-sparse}

Return to sorting: four implementations of the same task receive `[0, 0, 0, 0]`. Subtracting the group mean from each outcome reward gives four zero advantages. This term cannot distinguish which output's probability should increase. `[0, 1, 0, 0]` does provide a within-group signal. If advantages are also divided by the group standard deviation, zero variance needs explicit handling—for example, a small positive constant in the denominator—rather than computing $0/0$.

This concerns the outcome-reward policy-gradient term. A KL penalty that discourages deviation from a reference policy, an entropy objective that adjusts output diversity, or other auxiliary objectives can still produce updates. Programmatic rewards need not be binary: scoring the fraction of tests passed distinguishes partial solutions, but changes the objective and may encourage solving only the easy parts.

Task difficulty and sampling budget matter. Measure how often groups are entirely correct or entirely incorrect before deciding whether you need different tasks, a stronger starting policy, or more attempts.

We can estimate the frequency of equal-reward groups using [the mixed-group probability in DAPO's dynamic-sampling discussion](dapo.en.md#dynamic-sampling). For a fixed task, assume independent attempts, each with success probability $p$, and $G$ answers per group. All answers succeed with probability $p^G$ and all fail with probability $(1-p)^G$. Their sum, $p^G+(1-p)^G$, is the probability of equal rewards, complementary to the mixed-group probability. With $G=4$, only 12.5% of groups have equal rewards at $p=0.5$, rising to 65.62% at $p=0.1$ or $0.9$. Both very hard and very easy tasks can provide little within-group signal.

This assumes independent, identically distributed draws. When difficulty varies or samples are correlated, record group outcomes for each task rather than plugging the dataset-wide mean success probability $p$ into the formula.

Filtering equal-reward groups, as DAPO does, increases the share of groups with useful reward differences in each update. It requires additional sampling and changes which prompts enter the update. It cannot create a successful solution to a hard task from nothing. As with [supervised fine-tuning after filtering](rejection-sampling.en.md#prompt-selection), track prompt coverage as well as retention.

## Outcome rewards or process rewards {#outcome-rewards-or-process-rewards}

Outcome rewards assess the final result. They are straightforward to integrate into training, but do not locate the step where an error occurred. Process rewards offer finer feedback, provided that intermediate steps can be judged reliably. Human annotation is costly, model judgments can be wrong, and formal proof checking requires suitable representations and tools.

The two can be combined: tests check the final result while process signals guide search. Check whether the extra signal improves success at the same computational budget, rather than merely increasing process scores.

The frontier draft [Test-time compute: sampling, voting, and verifiers](../11-frontier/test-time-compute/parallel-and-verify.en.md) explains how reward models select answers at inference time and guide search over intermediate steps. When reading a paper, distinguish better answer selection from better policy training. [Let's Verify Step by Step, §2.1](https://arxiv.org/html/2305.20050v1#S2.SS1) keeps the generator fixed and compares reward models' ability to select from multiple candidates; it does not update the generator with RL. Better selection with process scores is therefore not, by itself, evidence that training a policy with process rewards would improve it. That requires a separate training and evaluation comparison.

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

## Questions to ask when designing verifiable rewards {#down-to-a-checklist}

1. Which deliberately wrong outputs still receive full reward? Can targeted counterexample tests expose them?
2. If the reward is binary, what fraction of groups in each training batch is entirely correct or entirely incorrect?
3. Am I checking the outcome or the reasoning? If only the outcome, how often can a guess pass?
4. Is the task really unverifiable, or does it need to be broken into smaller checks? Which parts can be verified?
5. If I add model judgment, can independent checks catch the errors that judge favors?

## Where to read next {#where-to-read-next}

- [Where preferences come from](where-preferences-come-from.en.md): the limits of learned reward models
- [PPO, GRPO, and DPO](after-ppo.en.md): how binary sparse rewards and group baselines amplify each other
- [The alignment tax](alignment-tax.en.md): check entropy, solution coverage, and actual quality separately
- [Evaluation](../07-evaluation/): why a verifier is not the same thing as an evaluation set

## Quick learning: what does a verifiable reward replace? {#quick-learning-what-does-a-verifiable-reward-replace}

<details class="interview" markdown="1">
<summary>Use the sorting example to explain what the reward does—and does not—verify</summary>

**Remember**: a test establishes specific properties, not necessarily the whole task. High reward, low entropy, and high quality are different measurements.

**Interview answer**

> I would state what the verifier checks, then construct an answer that passes while violating the task. For sorting, for instance, checking order is not enough; the check must also confirm that every element and its count are preserved. I would then evaluate models with held-out tests and a fixed sampling budget rather than reporting training reward alone.

<details markdown="1">
<summary><b>Deep dive</b>: outcome reward or process reward?</summary>

Outcome-reward reliability depends on the specification and verifier coverage; it is not inherently less biased. Process rewards give finer feedback but add judgments that also need validation. Both need independent quality checks.

</details>
</details>

## Starting papers {#starting-papers}

- [DeepSeek-R1, §2.2.2](https://arxiv.org/html/2501.12948v1#S2.SS2.SSS2) — a concrete design using rule-based accuracy and format rewards
- [DAPO](https://arxiv.org/abs/2503.14476) — dynamic sampling when equal-reward groups provide no outcome-reward gradient
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process versus outcome supervision
- The sorting and probability examples are checked by the [standard-library teaching script](code/selection_and_entropy.py). It neither executes model-generated programs nor reproduces training results.
