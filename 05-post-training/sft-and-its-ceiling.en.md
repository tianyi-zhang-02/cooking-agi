# SFT: how far imitation goes, and where it stops

[中文](sft-and-its-ceiling.md) · **English**

> Reading time: ~6 min · Type: chapter · Last reviewed: 2026-08

## Quick learning: what does SFT teach?

<details class="interview" markdown="1">
<summary>Assistant-only CE, behavior cloning, and the capability ceiling</summary>

**Quick memory**: SFT remains next-token CE, usually masked to assistant tokens. It raises the probability of demonstrated behavior but does not discover strategies missing from the data.

**Interview answer**

> SFT serializes a conversation, treats system and user tokens as context, and supervises the assistant answer and end token. It is effective for format, tone, tool protocols, and known solutions, but remains behavior cloning constrained by demonstration coverage, quality, and a teacher-forced token objective.

<details markdown="1">
<summary><b>Deep dive</b>: why does low token CE not imply a better complete answer?</summary>

CE decomposes sequence loss over tokens. One decision-critical token can receive little weight inside a long answer, while common phrasing and template tokens contribute many positions. Whole-answer correctness or multi-step strategy needs sequence-level preferences or RL signals when token imitation cannot express the objective.

</details>
</details>

## SFT learns a conditional distribution

SFT teaches "do it like this," so its ceiling is the ceiling of the demonstrations. But the dangerous part isn't what it fails to learn — **it's how confidently it learns things you never meant to teach.**

## What it does

Given input → ideal-output demonstrations, maximize the likelihood of that output. Per-token cross-entropy:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

That's all. No reward, no sampling, no environment. **It is imitation learning, not reinforcement learning** — the model never sees what would have happened had it said something else.

Being simple is why it works well for: fixed task formats, basic instruction following, distilling an expert's process, and giving downstream RL a sane starting point.

## Limit one: only what's in the data

The most obvious limit, and the most underrated. For situations the demonstrations don't cover, the model has no basis for knowing what to do.

The trouble is **it won't tell you they weren't covered**. Faced with something unseen, it still produces fluent output — imitation learns what the answer should *look* like, not whether this is something it knows.

## Limit two: cross-entropy barely notices individual tokens

The loss sums over tokens. Flip a negation inside a sentence and the total barely moves — one token's contribution is diluted by hundreds of others.

But in language, **one negation can invert the meaning of the whole sentence.**

So there's a structural mismatch between SFT's objective and what you actually care about: **it is sensitive to "does this look right overall" and insensitive to "is the pivotal part correct."** That's one reason RL has a place here — reward is assigned over a whole output and can be made sensitive to exactly that kind of local flip.

## Limit three: it forces an answer

The subtlest one, and the one with the worst consequences.

Demonstration data **always contains an answer**. Nobody writes "Q: what is X? A: I don't know" into an SFT set. So what the model learns from the shape of the data is: **when asked, produce an answer.**

When the knowledge isn't in the model, that learned regularity still fires — and it states something it doesn't know in exactly the tone it uses when it does.

**Hallucination here isn't a bug; it's the faithful result of the training objective.** You taught it to imitate "looking like you have the answer," so it does that when it doesn't.

RL can address this because a reward function can score "declining to answer" as mediocre and "fabricating" as very bad. There is nowhere to put that structure in SFT — **SFT has one correct answer and no notion of relative cost between options.**

## A corollary: when the knowledge isn't in the base, SFT teaches tone

Put the three together and you get an uncomfortable conclusion.

If the demonstrations go beyond what the model absorbed in pretraining, SFT cannot install the knowledge — gradients are enough to adjust phrasing, not to insert facts. What the model learns is **to discuss, in an expert register, material it does not command.**

The test: if a capability is entirely absent from the base, SFT typically improves **format and confidence**, not accuracy. The fix there is continued pretraining or retrieval, not more demonstrations.

## So when is SFT enough

Don't read the above as "SFT is weak." It is the right answer when:

- **There's a correct answer and a fixed format** — structured extraction, format conversion. RL is overkill.
- **You need a stable behavioral starting point** — exploring with RL from a random policy is prohibitively expensive. SFT pushes the policy into a sensible region and RL refines inside it. That's why SFT comes first in the three stages.
- **The capability is already in the base and simply isn't being invoked** — here demonstrations act as a switch.

## The division of labor

In one line: **SFT provides the starting point, RL provides the direction.**

| | SFT | RL |
| --- | --- | --- |
| Signal granularity | per token | whole output |
| Data needed | demonstrations of ideal output | something that can score (human, model, or program) |
| Can express "I don't know"? | no | yes — different scores for different options |
| Sensitive to local flips? | no | can be |
| Exploration | none | yes, hence more expensive and harder to tune |

They aren't interchangeable buttons; they solve different learning problems.

## Down to a checklist

1. Which situations do my demonstrations cover? For the ones they don't, how does the model behave — have I tested it?
2. Is the correctness I care about "does it look right overall" or "are these few tokens right"? Cross-entropy can't see the second.
3. Does my SFT set contain any refusals? If not, I am teaching the model to always produce an answer.
4. Is the knowledge I'm teaching actually in the base? If not, I may only be improving tone.
5. Does this task really need RL? With a correct answer and a fixed format, SFT plus better data is usually the cheaper win.

## Where to read next

- [The three stages of RLHF](rlhf-pipeline.en.md): what the two stages after SFT do
- [After PPO](after-ppo.en.md): the algorithm taxonomy on the RL line
- [Data and feedback](../01-data-and-feedback/): the quality of demonstration and preference data itself
