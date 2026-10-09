# SFT: how far imitation goes, and where it stops

[中文](sft-and-its-ceiling.md) · **English**

> Reading time: ~12 min · Type: chapter · Last reviewed: 2026-08

## SFT learns a conditional distribution {#sft-learns-a-conditional-distribution}

Give a model support conversations that demonstrate citing a return policy and asking for an order number, and it can learn similar behavior. If every example answers immediately and none asks for missing information, it may learn that habit too.

The question is not only how many examples SFT sees, but what they demonstrate. SFT can generalize and learn new knowledge; demonstration quality is not a strict capability ceiling. Those gains still need testing in unseen situations.

## What it does {#what-it-does}

Given a batch of input → ideal-output demonstrations, maximize the likelihood that the model produces that output. Per-token cross-entropy:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

That's all. No reward, no sampling, no environment. **It is imitation learning, not reinforcement learning** — the model never sees what would have happened had it said something else.

Being simple is exactly why it works well for these things: fixed task formats, basic instruction following, distilling an expert's process into the model, and giving downstream RL a sane starting point.

## How one conversation sample actually enters SFT {#how-one-conversation-sample-actually-enters-sft}

### 1. Pre-training and SFT: similar equations, different supervision {#1-pre-training-and-sft-similar-equations-different-supervision}

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

### 2. Label shifting remains, but only selected targets contribute loss {#2-label-shifting-remains-but-only-selected-targets-contribute-loss}

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

### 3. The end token is behavioral supervision too {#3-the-end-token-is-behavioral-supervision-too}

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

## Limit one: test beyond the demonstrated cases {#limit-one-only-whats-in-the-data}

Training covers routine returns, but evaluation includes cross-border orders and opened products. The model may generalize using prior knowledge and related examples, or apply the wrong rule. Missing coverage does not guarantee failure, but a lower training loss does not establish success.

Hold out different phrasings, combinations, and exceptions. These tests help distinguish learning a task rule from relying on familiar answers.

## Limit two: average loss is not task quality {#limit-two-cross-entropy-barely-notices-individual-tokens}

“Refund allowed” and “refund not allowed” differ by one word but have opposite consequences. Cross-entropy does penalize the wrong token, potentially strongly. Ordinary token weighting just does not know which position matters most to the task.

If the target-token probability falls from 0.9 to 0.1, its negative log-likelihood rises from about 0.105 to 2.303. Averaged over 200 tokens with other terms unchanged, the reported loss rises by only about 0.011. Check the refund decision as well as average loss.

More targeted demonstrations, sample or token weighting, and task-level checks can help. RL is one option, not the only remedy.

## Limit three: clarification and abstention need examples too {#limit-three-it-forces-an-answer}

If every demonstration answers confidently, the model gets little practice asking questions, abstaining, or admitting uncertainty. SFT can teach those behaviors: include examples such as “Please provide the order number” or “There is not enough evidence to tell.”

The challenge is matching behavior to available information. Answer when evidence is sufficient; ask when something essential is missing. Adding generic refusal templates can instead make the model avoid answerable questions.

Preference training or RL can compare the costs of different choices, but still depends on suitable rewards. No training label alone guarantees honesty.

## How do you check whether knowledge was learned? {#a-corollary-when-the-knowledge-in-the-demonstrations-isnt-in-the-model-sft-teaches-tone}

SFT can store new facts in model parameters, but does not guarantee reliable learning from one occurrence or successful recall under a different question. Fluent expert wording is not a substitute for fact checking.

Suppose a return window changes from 30 days to 14. Fine-tuning is one option; if policies change often, retrieving the current document may be easier to update and audit. Both approaches need tests for stale rules, correct citations, and fabrication when evidence is missing.

Choosing SFT, continued pretraining, or retrieval depends on knowledge volume, update frequency, and task structure—not a rule that SFT cannot learn facts.

## So when is SFT enough {#so-when-is-sft-enough}

Don't flip the above into "SFT is useless." It is the best choice when:

- **There's a correct answer and a fixed format** — structured extraction, format conversion. RL is overkill here.
- **You need a stable behavioral starting point** — exploring with RL from a random policy is too expensive. SFT first pushes the policy into a sensible region, and RL then refines inside it. That's why SFT comes first in the three stages of RLHF.
- **The capability is already in the base and simply isn't being invoked** — here demonstrations act as a switch.

## The division of labor between SFT and RL {#the-division-of-labor-between-sft-and-rl}

Start with the reliable signal you can obtain. If you can provide useful demonstrations, try SFT. If the desired process is hard to write down but attempted outcomes can be evaluated, RL may be useful. A workflow can also alternate the two.

| Question | SFT | RL |
| --- | --- | --- |
| Direct signal | Selected target outputs | Rewards for sampled actions or outcomes |
| Typical update granularity | Token-level cross-entropy, with masks and weights | Rewards assigned to actions through returns, advantages, and related estimators |
| Can teach clarification or abstention? | Yes, with context-appropriate examples | Yes, if rewards distinguish useful questions from unnecessary avoidance |
| Requires judging current-policy attempts? | Ordinary offline SFT does not | Usually needs rollouts, or recorded trajectories with appropriate estimators |
| Main checks | Coverage, templates, masks, generalization | Reward validity, sampling distribution, update stability |

Compare under the same task and evaluation budget. More elaborate training does not automatically produce a better result.

## What to check when designing SFT data {#what-to-check-when-designing-sft-data}

1. Which situations do my demonstrations cover? For the ones they don't, how does the model behave — have I tested it?
2. Is the correctness I care about "does it look right overall" or "are these few pivotal tokens right"? For the latter, test the decision separately from mean loss.
3. Does my SFT set contain context-appropriate abstention or clarification? Test what happens when essential evidence is missing.
4. Can the model recall the correct facts under different phrasings, rather than just sound confident?
5. Does this task really need RL? With a correct answer and a fixed format, SFT plus good data is usually the better deal.

## Where to read next {#where-to-read-next}

- [The three stages of RLHF](rlhf/three-stages.en.md): what the two stages after SFT do
- [After PPO](after-ppo.en.md): the family tree of algorithms on the RL line
- [Data and feedback](../01-data-and-feedback/README.en.md): the quality of demonstration and preference data itself

## Quick learning: what does SFT teach? {#quick-learning-what-does-sft-teach}

<details class="interview" markdown="1">
<summary>Assistant-only CE, behavior cloning, and the capability ceiling</summary>

**Quick memory**: SFT remains next-token CE, but the loss is usually computed only on assistant tokens. It raises the probability of demonstrated behavior; it does not automatically discover strategies missing from the data.

**Interview answer**

> SFT serializes a structured conversation, conditions on the system and user tokens, and supervises only the assistant answer and the end token. It is effective for format, tone, tool protocols, and known solutions, but it remains behavior cloning, constrained by demonstration coverage, demonstration quality, and a teacher-forced token objective.

<details markdown="1">
<summary><b>Deep dive</b>: why does low token CE not imply a better complete answer?</summary>

CE penalizes errors at individual target tokens, but a mean over a long response can obscure the few decisions that determine success. Inspect those decisions separately. Task checks, targeted examples, weighting, and preference or reward signals offer different ways to emphasize them.

</details>
</details>
