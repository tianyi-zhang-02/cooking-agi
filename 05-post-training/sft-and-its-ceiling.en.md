# SFT: how far imitation goes, and where it stops

[中文](sft-and-its-ceiling.md) · **English**

> Reading time: ~12 min · Type: chapter · Last reviewed: 2026-08

## Quick learning: what does SFT teach?

<details class="interview" markdown="1">
<summary>Assistant-only CE, behavior cloning, and the capability ceiling</summary>

**Quick memory**: SFT remains next-token CE, but the loss is usually computed only on assistant tokens. It raises the probability of demonstrated behavior; it does not automatically discover strategies missing from the data.

**Interview answer**

> SFT serializes a structured conversation, conditions on the system and user tokens, and supervises only the assistant answer and the end token. It is effective for format, tone, tool protocols, and known solutions, but it remains behavior cloning, constrained by demonstration coverage, demonstration quality, and a teacher-forced token objective.

<details markdown="1">
<summary><b>Deep dive</b>: why does low token CE not imply a better complete answer?</summary>

CE decomposes sequence loss over tokens. One token that decides final correctness receives very little weight inside a long answer, while length, common phrasing, and template tokens contribute many positions. If the target is a verifiable whole-answer outcome or a multi-step strategy, only a sequence-level preference or RL signal can directly express whether the whole trajectory is good.

</details>
</details>

## SFT learns a conditional distribution

SFT teaches "do it like this," so its ceiling is the ceiling of the demonstrations. But the dangerous part isn't what it fails to learn — **it's how confidently it learns things you never meant to teach.**

## What it does

Given a batch of input → ideal-output demonstrations, maximize the likelihood that the model produces that output. Per-token cross-entropy:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

That's all. No reward, no sampling, no environment. **It is imitation learning, not reinforcement learning** — the model never sees what would have happened had it said something else.

Being simple is exactly why it works well for these things: fixed task formats, basic instruction following, distilling an expert's process into the model, and giving downstream RL a sane starting point.

## How one conversation sample actually enters SFT

### 1. Pre-training and SFT: similar equations, different supervision

Pre-training text supplies its own next-token targets. Given $x_1,\ldots,x_T$:

$$
\mathcal L_{\text{pretrain}}
=-\sum_{t=2}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

No human must label each token, so this is self-supervised learning. An SFT
demonstration instead specifies the desired assistant behavior for a system-and-user
context. Demonstrations may be human-written, model-generated and filtered, or drawn
from mixed sources; “supervised” means the target behavior is externally selected, not
that every character was typed by a person.

Both stages can use next-token cross-entropy, but their training is not “completely the
same.” Data provenance, sequence structure, loss masks, mixtures, and optimization
intent differ: pre-training learns a language distribution, while SFT shapes existing
capability into selected behavior.

### 2. Label shifting remains, but only selected targets contribute loss

The conversation passes through the
[chat template and tokenizer](../00-foundations/core/tokenization.en.md) to produce
$z_1,\ldots,z_T$. Inputs and labels are still shifted by one position: the model uses
$z_{<t}$ to predict $z_t$. A common assistant-only objective adds a mask:

$$
\mathcal L_{\text{SFT}}
=-\sum_{t=2}^{T}m_t\log p_\theta(z_t\mid z_{<t}),
\qquad m_t\in\{0,1\}.
$$

```text
system / user / assistant role marker  → m_t = 0
assistant answer / end-of-message      → m_t = 1
```

System and user tokens remain in the left context and affect every prediction, but
their reconstruction does not contribute loss. Training code commonly assigns ignored
labels the value `-100`, which tells cross-entropy to skip those positions.

This is common, not universal. Multi-turn data may train every assistant turn or only
the final one, and some training setups score the full sequence. The most dangerous engineering
failure is not choosing one policy over another; it is misaligning template boundaries
and masks so user text or padding accidentally becomes a target.

### 3. The end token is behavioral supervision too

If a target contains only “I am fine.” but omits an end-of-message or EOS token, the
model learns how to begin and continue the answer but receives no explicit supervision
that it should stop there. A common SFT target therefore gives the end marker
$m_t=1$ as well:

```text
I → am fine → . → <|im_end|>
```

The inference system places the corresponding token in its stop set and ends the
message when it appears. Three pieces must agree: the ending marker used as a training
target, its special-token ID in the tokenizer, and the inference engine's stop
configuration.

A maximum-length limit remains a necessary safety fallback, but it is not the same as
teaching natural termination. One forcibly truncates the system; the other makes the
model assign high probability to “the answer is complete.”

## Limit one: only what's in the data

The most obvious limit, and the most underrated. For situations the demonstrations don't cover, the model has no basis for knowing what to do.

The trouble is **it won't tell you they weren't covered**. Faced with something unseen, it still produces fluent output — imitation learns what the answer should *look* like, not whether this is something it knows.

## Limit two: cross-entropy barely notices individual tokens

The loss sums over all tokens. Change the negation inside a sentence and the total barely moves — one token's contribution is diluted by hundreds of others.

But in language, **one negation can invert the meaning of the whole sentence.**

So there's a structural mismatch between SFT's objective and what you actually care about: **it is sensitive to "does this look right overall" and insensitive to "is the pivotal part correct."** That's one reason RL has a place here — reward is assigned over a whole output and can be made sensitive to exactly that kind of local flip.

## Limit three: it forces an answer

The subtlest one, and the one with the worst consequences.

Demonstration data **always contains an answer**. Nobody writes "Q: what is X? A: I don't know" into an SFT set. So what the model learns from the overall shape of the data is: **when asked, produce an answer.**

When the knowledge isn't inside the model, that learned regularity still fires — it states something it doesn't know in exactly the tone it uses when it is sure.

**Hallucination here isn't a bug; it's the faithful result of the training objective.** You taught it to imitate "looking like you have the answer," so it learned to look that way when it has none.

RL can address this because a reward function can give "declining to answer" a middling score and "fabricating" a very low one. There is nowhere to put that score structure in SFT — **SFT has one correct answer and no notion of relative cost between options.**

## A corollary: when the knowledge in the demonstrations isn't in the model, SFT teaches tone

Put the three together and you get an uncomfortable conclusion.

If the demonstrations go beyond what the model learned in pretraining, SFT cannot install the knowledge — gradients are enough to adjust phrasing, not to insert facts. What the model learns is **to discuss, in an expert's register, material it does not command.**

The test: if a capability is entirely absent from the base, SFT typically improves **format and confidence**, not accuracy. The thing to do then is continued pretraining or retrieval, not more demonstrations.

## So when is SFT enough

Don't flip the above into "SFT is useless." It is the best choice when:

- **There's a correct answer and a fixed format** — structured extraction, format conversion. RL is overkill here.
- **You need a stable behavioral starting point** — exploring with RL from a random policy is too expensive. SFT first pushes the policy into a sensible region, and RL then refines inside it. That's why SFT comes first in the three stages of RLHF.
- **The capability is already in the base and simply isn't being invoked** — here demonstrations act as a switch.

## The division of labor between SFT and RL

In one line: **SFT provides the starting point, RL provides the direction.**

| | SFT | RL |
| --- | --- | --- |
| Signal granularity | per token | whole output |
| Data needed | demonstrations of ideal output | something that can score (a human, a model, or a program) |
| Can express "I don't know"? | no | yes — different scores for different options |
| Sensitive to local flips? | no | can be |
| Exploration | none | yes, and therefore more expensive and harder to tune |

They aren't two interchangeable buttons; they solve different learning problems.

## What to check when designing SFT data

1. Which situations do my demonstrations cover? For the ones they don't, how does the model behave — have I tested it?
2. Is the correctness I care about "does it look right overall" or "are these few pivotal tokens right"? If it's the latter, cross-entropy can't measure it.
3. Does my SFT set contain any "decline to answer" samples? If not, I am in fact teaching the model to always produce an answer.
4. Is the knowledge I want to teach actually in the base? If not, I may only be improving tone.
5. Does this task really need RL? With a correct answer and a fixed format, SFT plus good data is usually the better deal.

## Where to read next

- [The three stages of RLHF](rlhf-pipeline.en.md): what the two stages after SFT do
- [After PPO](after-ppo.en.md): the family tree of algorithms on the RL line
- [Data and feedback](../01-data-and-feedback/README.en.md): the quality of demonstration and preference data itself
