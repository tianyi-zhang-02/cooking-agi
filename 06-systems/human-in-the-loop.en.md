# Human-in-the-Loop: When Should a Human Step In?

[中文](human-in-the-loop.md) · **English**

> Last reviewed: 2026-10

> Review-capacity and handoff examples checked: 2026-11-02

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

## Who takes the handoff, and how soon? {#review-capacity}

“Escalate when uncertain” leaves out capacity. Suppose there are 600 tasks an hour, 8% require review, and each review takes 2 minutes on average. Two reviewers work exclusively on this queue, ignoring breaks and other duties for now. This teaching calculation uses average rates; it does not predict waiting time.

| Quantity | Calculation | Result |
| --- | --- | --- |
| Reviews arriving each hour | `600 × 8%` | 48 |
| Estimated hourly capacity per reviewer | `60 / 2` | 30 |
| Combined capacity | `2 × 30` | 60 per hour |
| Average load relative to capacity | `48 / 60` | 80% |

If an update raises escalation to 12%, arrivals become 72 an hour. The same two reviewers can process 60, so sustained traffic adds 12 queued cases each hour. Even the original 80% load does not guarantee immediate service: arrivals bunch up, difficult cases take longer, and people need breaks and handovers.

Track queue wait, oldest pending task, actual review time, and expirations, not just the escalation rate. Add capacity, limit new admissions, or offer read-only help where appropriate; a full queue does not waive required authorization. Define priorities without starving lower-priority cases. This calculation checks long-run capacity, not p95 waiting time, which requires arrival and service-time distributions or direct measurement.

## Give the reviewer enough to take over {#handoff-packet}

In the quote-email example, the review screen needs more than approve and reject buttons. A person should be able to see the following without rereading the whole conversation.

| Show | Example |
| --- | --- |
| Proposed action and changes | Exact recipient; attachment changed from v3 to v4, with the price changes shown |
| Supporting evidence | The quote record and accessible source, not only a model summary |
| Unresolved questions | Whether the recipient is verified, the quote has expired, or the email was already sent |
| Available decisions | Approve this version, edit and resubmit, reject, or hand over to an authorized person |

The order or attachment can change after approval, so recheck the action version and current state before execution. Two reviewers opening the same case must not cause two sends. Server-side task claiming, version checks, and an atomic transition into execution can help; disabling a button in the browser is not sufficient.

Do not turn every human edit straight into a training target. Correcting a wrong date, changing one's mind, and granting a one-off policy exception are different feedback. Record the reason and scope first: the first may become a regression case, the second updates this task, and the third should not silently become a global rule.

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

How many is enough? Assume independent draws with a stable 1% failure probability. In $n$ draws, the chance of seeing at least one is $1-0.99^n$: about 63% for 100 samples and about 95% for 300. At 0.1%, 300 samples give only about a 26% chance. Sampling without replacement from a finite dataset calls for the hypergeometric distribution; traces generated by the same incident may not be independent either.

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

HITL should put decisions in the right hands, not simply minimize manual involvement. As a service expands or risk rises, more human review may be appropriate. Reduce repeated investigation, unnecessary confirmations, and recurring mistakes instead. Decisions collected during review can inform:

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
