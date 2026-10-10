# SFT: how a conversation becomes a training signal

[中文](sft-and-its-ceiling.md) · **English**

> Reading time: ~18 min · Type: chapter · Last reviewed: 2026-10

## SFT learns a conditional distribution {#sft-learns-a-conditional-distribution}

Give a model support conversations that demonstrate citing a return policy and asking for an order number, and it can learn similar behavior. If every example answers immediately and none asks for missing information, it may learn that habit too.

The question is not only how many examples SFT sees, but what they demonstrate. SFT can generalize and learn new knowledge; demonstration quality is not a strict capability ceiling. Those gains still need testing in unseen situations.

First follow one conversation through training, then consider what the data can teach. If you already run experiments, jump to [checking loss masks](#mask-example) or [deciding whether to try RL](#rl-readiness).

## What it does {#what-it-does}

Given a batch of input → ideal-output demonstrations, maximize the likelihood that the model produces that output. Per-token cross-entropy:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

An ordinary offline SFT update uses selected demonstrations without first generating attempts and scoring them for rewards. It is imitation learning. The demonstrations themselves may come from sampling and filtering; “no rollout during this update” does not mean the data pipeline never sampled anything.

Being simple is exactly why it works well for these things: fixed task formats, basic instruction following, distilling an expert's process into the model, and giving downstream RL a sane starting point.

## How one conversation sample actually enters SFT {#how-one-conversation-sample-actually-enters-sft}

### 1. Pre-training and SFT: similar equations, different supervision {#1-pre-training-and-sft-similar-equations-different-supervision}

Pre-training text supplies its own next-token targets. Given $x_1,\ldots,x_T$:

$$
\begin{aligned}
\mathcal L_{\text{pretrain}}&=\sum_{t=2}^{T}\ell_t,\\
\ell_t&=-\log p_\theta(x_t\mid x_{<t}).
\end{aligned}
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

### 2. Next-token prediction does not require scoring every token {#2-label-shifting-remains-but-only-selected-targets-contribute-loss}

The conversation passes through the
[chat template and tokenizer](../00-foundations/core/tokenization.en.md) to produce
$z_1,\ldots,z_T$. Inputs and labels are still shifted by one position: the model uses
$z_{<t}$ to predict $z_t$. A common assistant-only objective adds a mask:

$$
\begin{aligned}
\mathcal L_{\text{SFT}}&=\sum_{t=2}^{T}m_t\ell_t,\\
\ell_t&=-\log p_\theta(z_t\mid z_{<t}).
\end{aligned}
$$

Here $m_t\in\{0,1\}$ selects supervised targets. This simplified template is an example; role-marker supervision depends on the actual template.

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

If this target contains only “I am fine.” without an end-of-message or EOS token, this example supplies no explicit stopping supervision. The model may already know how to stop from earlier training. A common SFT target gives the end marker
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

## Follow the mask and the shift {#mask-example}

The user asks “What is 2 + 3?” and the demonstration answers “5.” To make alignment easy to see, represent the whole sequence with six symbols:

```text
position    0        1           2          3     4      5
input     <BOS>   question   <assistant>    5   <EOS>  <PAD>
label     -100     -100        -100        5   <EOS>   -100
```

A real tokenizer splits the question into several tokens; “question” is a diagram placeholder. Initially, labels line up with input positions. **The causal loss then shifts the targets by one:**

| Output position | Predicts | Scored here? |
| --- | --- | --- |
| After `<BOS>` | Question | No |
| After the question | `<assistant>` | Not in this example |
| After `<assistant>` | `5` | Yes |
| After `5` | `<EOS>` | Yes |
| After `<EOS>` | `<PAD>` | No |

The label for `5` sits at position 3, but uses logits from position 2. Using position 3 to predict that same `5` would expose the current target in its own input.

No model is needed to check this. Construct logits where the two supervised targets receive probabilities 0.8 and 0.6. Their mean loss should be $(-\log0.8-\log0.6)/2\approx0.3670$.

The code assigns token ID `4` to the answer text `5`, and ID `5` to `<EOS>`. Text and vocabulary IDs are different things. This tiny invented vocabulary is not a real tokenizer's mapping.

```python
import math
import torch
import torch.nn.functional as functional

probabilities = torch.full((1, 6, 6), 1 / 6, dtype=torch.float64)
probabilities[0, 2] = torch.tensor([.04, .04, .04, .04, .8, .04], dtype=torch.float64)
probabilities[0, 3] = torch.tensor([.08, .08, .08, .08, .08, .6], dtype=torch.float64)
logits = probabilities.log().requires_grad_()
labels = torch.tensor([[-100, -100, -100, 4, 5, -100]])
loss = functional.cross_entropy(
    logits[:, :-1].reshape(-1, 6),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
assert math.isclose(loss.item(), -(math.log(.8) + math.log(.6)) / 2, abs_tol=1e-7)
loss.backward()
active_positions = (logits.grad.abs().sum(-1) > 0).nonzero().tolist()
assert active_positions == [[0, 2], [0, 3]]
```

We shift manually because this code calls CE directly. The [Transformers causal LM loss](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/loss/loss_utils.py) already handles shifting; do not pre-shift labels again when passing them to a model that does this internally.

### If user targets are masked, how does the model learn to use the question?

A loss mask selects predictions to score, not text to read. Predicting `5` still depends on the preceding `2 + 3`. Gradients can pass through attention and shared parameters into computations that process context. **Not scoring reconstruction of the question does not remove its influence on learning.**

An attention mask has a different job: controlling visibility, such as hiding future tokens or padding. Hiding the user message there would remove information needed to answer.

### What about multiple turns and tool calls?

| Conversation content | Common assistant-only treatment | What to verify |
| --- | --- | --- |
| System / user | Context, not prediction targets | A training choice, not a rule for every task |
| Assistant tool call | Can supervise the name and arguments | The template must identify the call correctly |
| Tool result | Usually context for the next answer | Do not mistake tool output for an assistant response |
| Assistant answer | Supervise content and ending marker | Every assistant turn, or only the last? |
| Padding | Exclude from loss | Distinguish by position if PAD and EOS share an ID |

The [versioned TRL guide](https://huggingface.co/docs/trl/v0.23.1/sft_trainer#train-on-assistant-messages-only) exposes assistant-only and completion-only options separately. Their names alone do not establish equivalent behavior. Inspect masks returned by the actual template and decode final labels for a few examples to see what training will reward.

Check truncation too: does the example retain an answer, or only its prompt? Was the ending marker cut off? All-`-100` labels provide no supervision and need handling during data checks. Packing additionally needs sample-boundary and attention checks; a loss mask alone does not isolate conversations.

## Test cases the demonstrations did not cover {#limit-one-only-whats-in-the-data}

Training covers routine returns, but evaluation includes cross-border orders and opened products. The model may generalize using prior knowledge and related examples, or apply the wrong rule. Missing coverage does not guarantee failure, but a lower training loss does not establish success.

Hold out different phrasings, combinations, and exceptions. These tests help distinguish learning a task rule from relying on familiar answers.

## Lower loss, better answers? {#limit-two-cross-entropy-barely-notices-individual-tokens}

“Refund allowed” and “refund not allowed” differ by one word but have opposite consequences. Cross-entropy does penalize the wrong token, potentially strongly. Ordinary token weighting just does not know which position matters most to the task.

If the target-token probability falls from 0.9 to 0.1, its negative log-likelihood rises from about 0.105 to 2.303. Averaged over 200 tokens with other terms unchanged, the reported loss rises by only about 0.011. Check the refund decision as well as average loss.

More targeted demonstrations, sample or token weighting, and task-level checks can help. RL is one option, not the only remedy.

## Did the examples show when to ask a question? {#limit-three-it-forces-an-answer}

If every demonstration answers confidently, the model gets little practice asking questions, abstaining, or admitting uncertainty. SFT can teach those behaviors: include examples such as “Please provide the order number” or “There is not enough evidence to tell.”

The challenge is matching behavior to available information. Answer when evidence is sufficient; ask when something essential is missing. Adding generic refusal templates can instead make the model avoid answerable questions.

Preference training or RL can compare the costs of different choices, but still depends on suitable rewards. No training label alone guarantees honesty.

## How do you check whether knowledge was learned? {#a-corollary-when-the-knowledge-in-the-demonstrations-isnt-in-the-model-sft-teaches-tone}

SFT can store new facts in model parameters, but does not guarantee reliable learning from one occurrence or successful recall under a different question. Fluent expert wording is not a substitute for fact checking.

Suppose a return window changes from 30 days to 14. Fine-tuning is one option; if policies change often, retrieving the current document may be easier to update and audit. Both approaches need tests for stale rules, correct citations, and fabrication when evidence is missing.

Choosing SFT, continued pretraining, or retrieval depends on knowledge volume, update frequency, and task structure—not a rule that SFT cannot learn facts.

## So when is SFT enough {#so-when-is-sft-enough}

If these checks already pass, a more elaborate training pipeline may not be necessary:

- **Both format and task quality meet requirements.** For extraction, check field correctness and missing information, not just parsable JSON.
- **You want a reliable behavioral starting point.** Demonstrations can establish tool protocols and response habits. Their value before RL depends on the base model and task.
- **The main problem is eliciting an existing capability.** Try a small set of useful demonstrations, then check whether improvements transfer to unseen inputs before reaching for a heavier method.

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

## When is it worth trying RL? {#rl-readiness}

There is no universal “start below loss 0.5” rule. Tokenizers, response lengths, masks, and reductions all change that number. A more useful check is to generate answers from several checkpoints on the same held-out tasks and inspect their failures.

Consider a support assistant that uses an order tool. These are **illustrative numbers, not measured results or deployment thresholds**:

| Check | Checkpoint A | Checkpoint B | Interpretation |
| --- | --- | --- | --- |
| Parsable arguments in 100 calls | 72 | 96 | A still struggles with the protocol; templates and demonstrations may be the direct fix |
| Completed tasks out of 100 | 42 | 61 | B has better formatting, but task success still lags |
| Useful clarifications on 20 incomplete requests | 3 | 12 | Overall success must not hide this failure category |

B's numbers do not automatically justify RL. Ask three more questions:

1. **Can the reward distinguish good attempts from bad ones?** Rewarding valid JSON alone can accept irrelevant answers. Check the reward or verifier against a small human-reviewed slice.
2. **Do different attempts on the same task provide a learning signal?** With within-group relative advantages, equal scores give no contrast from that term. If every attempt fails, inspect difficulty, sampling, and reward design; more SFT is not the only possible response.
3. **What does RL solve beyond a simpler, equal-budget alternative?** Start from the same checkpoint and compare continued SFT, generation-and-filtering followed by training, and a short RL run. Track quality, length, cost, and regressions—not just training reward.

Choose metrics and stopping conditions before the pilot. If reward rises while success stays flat and answers get longer, inspect the reward before simply adding steps.

SFT followed by RL is common, not mandatory. [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948v1) investigated RL without a preliminary SFT stage; that did not mean starting from random weights. Whether this stage helps depends on the model and task.

<details markdown="1">
<summary>A closer look: DFT changes token weights, not the gate for starting RL</summary>

[DFT (2025)](https://arxiv.org/html/2508.05629v1#S3.SS3) weights CE by the current target-token probability, with the weight detached:

$$
\ell_{\text{DFT}}=-\operatorname{sg}(p_y)\log p_y.
$$

This is one term; a batch still needs a mask and reduction. It adds no environment reward and specifies no probability threshold for switching to RL.

Forgetting `detach()` changes the update. With binary target probability $p=0.01$, the positive logit's CE gradient is −0.99; detached weighting gives −0.0099. Fully differentiating $-p\log p$ instead gives about +0.0357, so gradient descent would lower this already unlikely target's score.

That is a chain-rule difference, not a precision issue. The [tests](../site/tests/test_loss_contracts.py) check both directions. Probability weighting can also weaken learning from difficult but important examples. Compare on your held-out tasks rather than treating a paper's gains as universal.

</details>

## What to check when designing SFT data {#what-to-check-when-designing-sft-data}

1. Which situations do my demonstrations cover? For the ones they don't, how does the model behave — have I tested it?
2. Is the correctness I care about "does it look right overall" or "are these few pivotal tokens right"? For the latter, test the decision separately from mean loss.
3. Does my SFT set contain context-appropriate abstention or clarification? Test what happens when essential evidence is missing.
4. Can the model recall the correct facts under different phrasings, rather than just sound confident?
5. Does this task really need RL? With a correct answer and a fixed format, SFT plus good data is usually the better deal.

## Where to read next {#where-to-read-next}

- [Implementing cross-entropy](../00-foundations/pytorch/cross-entropy.en.md): stability, soft targets, masks, and denominators
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
