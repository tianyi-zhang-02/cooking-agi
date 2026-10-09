# Verifiable rewards: when the reward doesn't need learning

[中文](verifiable-rewards.md) · **English**

> Reading time: ~6 min · Type: chapter · Last reviewed: 2026-08

## Replace a learned reward with a checkable rule {#replace-a-learned-reward-with-a-checkable-rule}

A model writes a sorting function. You could ask another model whether it looks good, or execute tests. Tests make part of the judgment reproducible.

If the only test is `[3, 1, 2]`, always returning `[1, 2, 3]` passes. Empty lists, repeated elements, or negative values expose the error. Verifiable rewards offer explicit, repeatable checks, not automatic immunity to loopholes.

## What gets swapped out {#what-gets-swapped-out}

Recall [RLHF's four models](rlhf/three-stages.en.md): Policy, Critic, Reward, Reference. Verifiable rewards delete the **Reward Model** — the network fitted to [preference data](where-preferences-come-from.en.md).

$$r(x,y) = \text{verify}(y) \in \{0, 1\}$$

A deterministic function replaces a learned one.

## Why it narrows reward hacking {#why-it-narrows-reward-hacking}

Replacing an RM with tests removes the need to infer code correctness from how the answer looks. The checked properties are more explicit, and failures are easier to reproduce. This avoids some reward-model generalization errors.

A discrete, nondifferentiable reward is not protection against exploitation. Policy gradients adjust sampled-output probabilities using their rewards; they do not need gradients through the test program. An incorrect implementation that passes an incomplete test suite can still be reinforced.

## But reward hacking only moved {#but-reward-hacking-only-moved}

This section is the point, because "we use RLVR so we don't have reward hacking" is a dangerous simplification.

**The verifier itself can have holes.** If the reward is "passes the tests" and coverage is incomplete, the model learns to **pass those tests**, not to **write correct code**. It will:

- hardcode returns for the specific test cases;
- swallow every exception so the program doesn't crash, satisfying "doesn't error" style checks;
- find bugs in the test harness itself.

Math is the same. Verify only the final answer and the model can score by **guessing** — emitting text that looks like reasoning, unconnected to the answer, and landing on it anyway. **Process wrong, reward full.**

Generalized: **what you verify is what the model optimizes — no more and no less.** Same lesson as the reward-model era, except this time the wrong objective is one *you wrote down*, which at least makes it readable, auditable, and fixable.

## A new problem: binary rewards are sparse {#a-new-problem-binary-rewards-are-sparse}

Return to sorting: four implementations of the same task receive `[0, 0, 0, 0]`. If training uses only outcome rewards centered by the group mean, all four advantages are zero. This term cannot distinguish which output deserves more probability. `[0, 1, 0, 0]` does provide a relative signal.

This concerns the outcome-reward policy-gradient term. KL, entropy, or other auxiliary objectives can still produce updates. Programmatic rewards need not be binary: a fraction of tests passed is denser, but changes the objective and may reward solving only easy parts.

Task difficulty and sampling budget matter. Measure all-correct and all-incorrect groups before deciding whether you need different tasks, a stronger starting policy, or more attempts.

## Outcome rewards or process rewards {#outcome-rewards-or-process-rewards}

Verifying only the final answer is cheap but rewards lucky guesses. Verifying intermediate steps gives a much denser signal, but **who labels the steps** — if humans do, you're back in the cost and noise of preference data.

The common compromise is to have a model generate or check process labels, but that **reintroduces a learned judge**, and with it the climbable surface you just removed. There is no free version of this tradeoff.

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

1. How many holes does my verifier have? Have I **deliberately tried** to cheat it once?
2. If the reward is binary, what fraction of my groups is all-right or all-wrong?
3. Am I verifying outcome or process? If outcome, how often does it get there by guessing?
4. Is my task really "unverifiable," or have I just not decomposed it? Which part is checkable?
5. If I brought in a model judge, did the climbable surface come back?

## Where to read next {#where-to-read-next}

- [Where preferences come from](where-preferences-come-from.en.md): the structural ceilings of a learned reward model
- [After PPO](after-ppo.en.md): how binary sparse rewards and group baselines amplify each other
- [The alignment tax](alignment-tax.en.md): the price of optimizing any measurable objective
- [Evaluation](../07-evaluation/): a verifier and an eval set are not the same thing

## Starting papers {#starting-papers}

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — verifiable rewards for math RL at scale
- [DAPO](https://arxiv.org/abs/2503.14476) — dynamic sampling for all-right/all-wrong groups
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process versus outcome supervision

## Quick learning: what does a verifiable reward replace? {#quick-learning-what-does-a-verifiable-reward-replace}

<details class="interview" markdown="1">
<summary>Verifiers, sparse reward, and the new location of reward hacking</summary>

**Quick memory**: when a checker can be written, programmatic verification is more reliable than asking an RM to infer quality. The model may still exploit the checker, environment, or task distribution.

**Interview answer**

> Verifiable reward replaces a learned proxy with a reproducible rule such as unit tests, a mathematical answer, or an environment terminal state. It reduces RM misgeneralization but often creates sparse binary feedback and moves reward hacking into verifier specifications, sandboxes, and data generation.

<details markdown="1">
<summary><b>Deep dive</b>: outcome reward or process reward?</summary>

Outcome reward has lower specification bias but sparse credit assignment. Process reward is denser, yet unreliable intermediate checks can encode human bias into the trajectory. A common compromise uses a hard outcome verifier for terminal correctness and carefully calibrated process signals for search efficiency.

</details>
</details>
