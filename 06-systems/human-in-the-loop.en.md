# Human in the Loop

[中文](human-in-the-loop.md) · **English**

## Quick learning: when should control pass to a human?

<details class="interview" markdown="1">
<summary>An escalation policy based on risk, uncertainty, and reversibility</summary>

**Quick memory**: do not escalate merely because the model is uncertain. Escalate when potential impact is high, the action is irreversible or permission-sensitive, or evidence conflicts; return human decisions as structured learning signals.

**Interview answer**

> HITL is a selective escalation policy. The system combines calibrated uncertainty, action risk, reversibility, permissions, and business rules to choose automatic execution, confirmation, or human review. It records what the human changed, the evidence used, and the outcome for future rules, data, and evaluation.

<details markdown="1">
<summary><b>Deep dive</b>: why is an uncertainty threshold alone insufficient?</summary>

Models can be confidently wrong, and identical uncertainty has different consequences for a song recommendation and a large refund. Escalation should combine calibrated uncertainty with impact and irreversibility while respecting reviewer capacity, or risk is merely converted into an unbounded queue.

</details>
</details>

## Core view

Human in the Loop (HITL) does not mean sending every model failure to a person. It is a **control policy** that chooses between automatic execution, user confirmation, and expert escalation based on risk, uncertainty, novelty, and reversibility.

## Human roles

| Role | Function |
| --- | --- |
| Teacher | Provides demonstrations, preferences, corrections, and rationale |
| Reviewer | Inspects high-risk outputs or consequential actions |
| Approver | Authorizes irreversible external side effects |
| Collaborator | Decomposes problems and supplies missing context with the model |
| Auditor | Finds blind spots in the automated system and its evaluators |
| User | Expresses goals, controls boundaries, and corrects personalized state |

## When to escalate

Routing is a decision problem rather than one fixed threshold:

> **Value of escalation ≈ failure probability × impact × uncertainty × irreversibility − human cost**

Common triggers include evaluator disagreement, out-of-distribution tasks, conflicting evidence, consequential external actions, ambiguous goals, loops, repeated fallback, and abnormal cost.

## Capture more than a label

Human intervention should preserve why the case was escalated, what evidence the reviewer saw, the final decision, what changed, the rationale, and which evaluation, policy, or training update should absorb the case. A final label without context provides weak learning signal.

## Three timescales

- **Synchronous intervention:** approval or takeover before a risky action completes.
- **Asynchronous review:** sampled trajectory review for systematic failures and new blind spots.
- **Periodic governance:** revisiting rubrics, escalation thresholds, permissions, retention, and user control.

## Common mistakes

- Treating humans as a fallback API without supplying context
- Reviewing only the final answer rather than tools, evidence, side effects, and state changes
- Collecting agree/disagree without edits or rationale
- Keeping escalation rules static as models and risks change
- Sending feedback directly into training without accounting for annotator variation and policy bias
- Automation bias caused by confident model presentation

## My current view

A good HITL system does not maximize dependence on people. It places the **highest-value human judgment** at the right point and turns that judgment into reusable evaluations, rules, data, and policy improvements.
