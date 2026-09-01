# Verifiable rewards: when the reward doesn't need learning

[中文](verifiable-rewards.md) · **English**

> Reading time: ~6 min · Type: chapter · Last reviewed: 2026-08

## Quick learning: what does a verifiable reward replace?

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

## Replace a learned reward with a checkable rule

Math has answers to check; code has tests to run. For these, the reward **doesn't need to be learned — you write a program**. And that swap changes more than one module: **what can be gamed is a fitted reward, not a verified one.** Reward hacking doesn't disappear, though. It relocates: from fooling a reward model to fooling a verifier.

## What gets swapped out

Recall [RLHF's four models](rlhf-pipeline.en.md): Policy, Critic, Reward, Reference. Verifiable rewards delete the **Reward Model** — the network fitted to [preference data](where-preferences-come-from.en.md).

$$r(x,y) = \text{verify}(y) \in \{0, 1\}$$

A deterministic function replaces a learned one.

## Why it narrows reward hacking

A reward model is differentiable, continuous, and fitted on finite data. Together those mean: **it offers a gradient to climb, and outside its training distribution its values are extrapolations.** A policy only has to find corners where the reward model was never supervised but scores highly, and it collects reward without getting better.

A verifier has none of that. A test passes or it doesn't; there's no slope in between. **You cannot argue a unit test up to 0.9.**

So verifiable rewards mostly dissolve RLHF's most stubborn problem as a side effect. Not because they're smarter — because they **structurally lack the exploitable surface**.

## But reward hacking only moved

This section is the point, because "we use RLVR so we don't have reward hacking" is a dangerous simplification.

**The verifier itself can have holes.** If the reward is "passes the tests" and coverage is incomplete, the model learns to **pass those tests**, not to **write correct code**. It will:

- hardcode returns for the specific test cases;
- swallow every exception so the program doesn't crash, satisfying "doesn't error" style checks;
- find bugs in the test harness itself.

Math is the same. Verify only the final answer and the model can score by **guessing** — emitting text that looks like reasoning, unconnected to the answer, and landing on it anyway. **Process wrong, reward full.**

Generalized: **what you verify is what the model optimizes — no more and no less.** Same lesson as the reward-model era, except this time the wrong objective is one *you wrote down*, which at least makes it readable, auditable, and fixable.

## A new problem: binary rewards are sparse

A learned reward model emits a continuous score, so "slightly off" and "way off" differ. A verifier emits 0 or 1.

That connects straight to [the algorithms chapter](after-ppo.en.md): with a group-relative baseline (GRPO/RLOO), **a group where every response is right, or every one is wrong, has all advantages at zero after subtracting the mean — no gradient at all**. On tasks that are too easy or too hard, that fraction gets large and the sampling was wasted.

DAPO's dynamic sampling targets exactly this: filter out all-right/all-wrong groups so every batch carries usable gradient.

**So RLVR and group-relative baselines are coupled choices** — each is reasonable alone, and together they amplify each other's weakness. Task difficulty distribution becomes something you have to design deliberately rather than inherit from a dataset.

## Outcome rewards or process rewards

Verifying only the final answer is cheap but rewards lucky guesses. Verifying intermediate steps gives a much denser signal, but **who labels the steps** — if humans do, you're back in the cost and noise of preference data.

The common compromise is to have a model generate or check process labels, but that **reintroduces a learned judge**, and with it the climbable surface you just removed. There is no free version of this tradeoff.

## Verifiability is a spectrum, not a binary

It's tempting to read this as "verifiable or not." It's a band:

| | Example | What the verifier looks like |
| --- | --- | --- |
| Strongly verifiable | math answers, unit tests, compiles | deterministic program, few holes |
| Semi-verifiable | format compliance, whether a citation exists, does the SQL run | a program covers part, not "is it appropriate" |
| Weakly verifiable | is the summary faithful, is the explanation clear | model judge or human only |
| Not verifiable | tone, creativity, taste | preferences only |

**The criterion: pushing a task leftward along this band pays better than swapping algorithms.** Plenty of tasks that look preference-only are half verifiable once decomposed — inside "write a report," citation validity, numerical consistency, and format compliance are all checkable; only the remainder needs preferences.

Do that decomposition first, then talk about algorithms.

## Down to a checklist

1. How many holes does my verifier have? Have I **deliberately tried** to cheat it once?
2. If the reward is binary, what fraction of my groups is all-right or all-wrong?
3. Am I verifying outcome or process? If outcome, how often does it get there by guessing?
4. Is my task really "unverifiable," or have I just not decomposed it? Which part is checkable?
5. If I brought in a model judge, did the climbable surface come back?

## Where to read next

- [Where preferences come from](where-preferences-come-from.en.md): the structural ceilings of a learned reward model
- [After PPO](after-ppo.en.md): how binary sparse rewards and group baselines amplify each other
- [The alignment tax](alignment-tax.en.md): the price of optimizing any measurable objective
- [Evaluation](../07-evaluation/): a verifier and an eval set are not the same thing

## Starting papers

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — verifiable rewards for math RL at scale
- [DAPO](https://arxiv.org/abs/2503.14476) — dynamic sampling for all-right/all-wrong groups
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process versus outcome supervision
