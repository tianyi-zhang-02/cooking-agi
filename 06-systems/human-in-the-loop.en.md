# Human-in-the-Loop: When Should a Human Step In?

[中文](human-in-the-loop.md) · **English**

> Last reviewed: 2026-10

## HITL is a risk-control policy {#hitl-is-a-risk-control-policy}

Human-in-the-Loop does not mean “hand everything to a person whenever the model cannot do it.” It means having the system actively request confirmation, correction, or takeover when **risk is high, evidence is insufficient, the task is novel, or the action cannot be undone**.

## First, a scenario that needs escalation to a human {#first-a-scenario-that-needs-escalation-to-a-human}

Asking a model to “write an email for me” and to “send this email for me directly” look very similar, but the risk is completely different.

- Writing a draft: a mistake can still be fixed, so it can usually be done automatically.
- Sending the email: it affects a real relationship, so it is best to have the user confirm first.
- Sending in bulk to customers: the impact is larger and may require additional permissions and expert review.

The key to HITL is not asking a human whenever model confidence is low. It is looking at **the probability of error, the size of the impact, and whether the action is reversible**, all together.

## What roles can a human play in the system {#what-roles-can-a-human-play-in-the-system}

| Role | What they do |
| --- | --- |
| Teacher | Provides demonstrations, preferences, corrections, and explanations |
| Reviewer | Checks high-risk outputs and critical actions |
| Approver | Authorizes before external side effects occur |
| Collaborator | Works with the model to supply context and decompose the problem |
| Auditor | Periodically looks for blind spots in the automated system and its evaluators |
| The user themselves | Expresses goals, sets boundaries, and corrects personalized state |

## When is escalation to a human worth it {#when-is-escalation-to-a-human-worth-it}

To decide whether an additional review is worthwhile, a useful accounting rule is:

> **Net benefit of review ≈ expected loss without review − residual expected loss after review − review cost**

For a teaching example, assign an error a cost of 100. If the automated error rate is 5%, review leaves a 1% error rate, and reviewing costs 1, the net benefit is `0.05 × 100 − 0.01 × 100 − 1 = 3`. Humans make mistakes too, so residual risk after review is not zero. Estimate those probabilities from observed data rather than substituting a model's self-reported confidence.

Change one condition at a time and the answer can flip:

| Case | Error cost | Automated error rate | Error rate after review | Review cost | Net benefit |
| --- | ---: | ---: | ---: | ---: | ---: |
| The example above | 100 | 5% | 1% | 1 | 3 |
| Reviewers mostly approve what they see | 100 | 5% | 4% | 1 | 0 |
| A draft the user can easily fix | 10 | 5% | 1% | 1 | −0.6 |

In the second row, the reduction in expected loss exactly offsets the review cost: the net benefit is zero, but review still corrects some errors. People shown a model's conclusion tend to go along with it (automation bias; see [LLM decision support](decision-support.en.md)), so estimate the after-review error rate from reviewers' actual correction records instead of assuming they catch every mistake. In the third row the error is cheap: review costs more than the loss it can prevent, so it is better to finish automatically and let the user edit.

This is not a formula for bypassing authorization. An action that requires user approval still requires it, even when its estimated cost is low. Irreversibility affects the loss and response options; it does not need an arbitrary extra multiplier. Include waiting time in the cost as well.

Common triggers include:

- several models or evaluators clearly disagree;
- the input falls outside the historical distribution and the system has never handled it before;
- the retrieved evidence is contradictory, or its sources are not reliable enough;
- the action would send a message, spend money, modify data, or affect other people;
- the user's goal is unclear and the cost of continuing to guess is high;
- the agent enters a loop, falls back repeatedly, or shows abnormal cost.

## Keep the evidence behind human feedback {#human-feedback-cannot-be-stored-as-just-a-label}

“Agree / disagree” can be a useful training label, but it usually does not explain where the failure began. Preserve evidence or a reason for the correction where feasible.

A useful human intervention should, as far as possible, preserve:

- why this case was escalated;
- which evidence the reviewer saw;
- what decision was finally made;
- which part of the model output or action was changed;
- why it was changed that way;
- whether this case should update tests, rules, the policy, or the training data.

## Three timings for intervention {#three-timings-for-intervention}

### Real-time confirmation {#real-time-confirmation}

Have the user approve before the action happens. This suits high-risk or irreversible operations.

Confirmation must identify a specific action, not grant a pass for the whole conversation.
Show the target, material arguments, and impact; bind approval to that action version and an
expiry. The execution layer verifies the approver's authority and checks authorization and
current state before acting. Changes to recipients, attachments, or amounts invalidate the
old confirmation. Rejection or an unanswered confirmation that expires is not approval. See
[OWASP's transaction-authorization principles](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html).

For example, in a quote-email workflow requiring confirmation for each send:

```text
Preview: send to the purchasing contact, attachment quote-v3.pdf
User approves this version → proceed to pre-send validation
Attachment changes to quote-v4.pdf → invalidate approval, preview and confirm again
User rejects or confirmation times out → do not send; record why
```

These are state constraints, not a sending implementation. Resuming a paused workflow does
not necessarily mean “run the next line exactly once”: for example,
[LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts#side-effects-called-before-interrupt-must-be-idempotent)
resume by rerunning the node from its beginning. Perform the external write only after
valid approval, and follow the tool's idempotency rules when retrying.

### After-the-fact spot checks {#after-the-fact-spot-checks}

Sample from completed trajectories to find systematic failures, judge bias, and new types of problems.

How many is enough? Suppose one failure type appears in 1% of completed trajectories. Sampling $n$ uniformly, the chance of seeing at least one is $1-0.99^n$: about 63% for 100 samples and about 95% for 300. If it appears in only 0.1%, 300 samples still give only about a 26% chance of seeing it once.

So oversample rare but high-risk slices (for example, trajectories that issue refunds or send bulk messages), and weight each slice back by its sampling rate when estimating overall rates. Otherwise, as with the error-selected traces in [Agent observability](agent-observability.en.md), the recorded rates drift away from real traffic.

### Periodically redesigning the rules {#periodically-redesigning-the-rules}

As models and tasks change, re-examine the rubric, escalation thresholds, permissions, data retention, and user control.

## Common mistakes {#common-mistakes}

- **Treating the human as a fallback API:** no context is provided, yet a quick decision is demanded.
- **Looking only at the final answer:** search, tools, side effects, and state updates are ignored.
- **Collecting labels but not reasons:** it is harder to work out what needs improving.
- **Escalation rules that never change:** old thresholds are still used after model capability and task risk have changed.
- **Feedback going straight into training:** annotator variation, policy bias, and data leakage are not handled.
- **Automation bias:** the more confident the model sounds, the more readily people accept it without thinking.

## What good HITL should ultimately reduce {#what-good-hitl-should-ultimately-reduce}

HITL should make effective use of human judgement where it is needed, without making more and more tasks depend on manual handling. Decisions collected during review can inform:

- new deterministic rules;
- better eval cases;
- a clearer failure taxonomy;
- reusable training data;
- a more sensible boundary for automation.

## Connections to other chapters {#connections-to-other-chapters}

- [Agent Observability](agent-observability.en.md) helps reviewers trace what happened.
- [Evaluation](../07-evaluation/README.en.md) decides which cases need human calibration.
- [Data and feedback](../01-data-and-feedback/README.en.md) decides how human judgment enters later learning.
- [Personal AGI](../09-personal-agi/README.en.md) requires that the user can control and correct long-term state.

## Quick learning: when should control pass to a human? {#quick-learning-when-should-control-pass-to-a-human}

<details class="interview" markdown="1">
<summary>Consider risk, uncertainty, and whether the action can be undone</summary>

**In brief**: model confidence alone does not determine whether to ask for confirmation. Consider the potential loss, reversibility, permissions, and conflicting evidence. Record what the person changed and why, so those decisions can guide later improvements.

**Interview answer**

> HITL is a selective escalation policy. The system combines model uncertainty, action risk, reversibility, permissions, and business rules to decide between automatic execution, requesting confirmation, or handing off to a human. It records what the human changed, on what basis, and the final outcome, and uses them to update rules, data, and evaluation.

<details markdown="1">
<summary><b>Deep dive</b>: why is an uncertainty threshold alone not enough?</summary>

A model can be highly confident in a wrong answer. The same uncertainty also carries different consequences for “recommend a song” and “refund ten thousand dollars.” Consider the impact of an error, whether it can be undone, and how much risk review actually reduces. Account for reviewer capacity too, or every escalation may just sit in a queue.

</details>
</details>
