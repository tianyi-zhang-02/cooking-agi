# Human-in-the-Loop: When Should a Human Step In?

[中文](human-in-the-loop.md) · **English**

## Quick learning: when should control pass to a human?

<details class="interview" markdown="1">
<summary>An escalation rule built on risk × uncertainty × reversibility</summary>

**Quick memory**: it is not “ask a human whenever the model is uncertain.” Escalate when the potential loss is high, the action is irreversible or permission-sensitive, or the evidence conflicts; and the human's feedback has to flow back as structure the system can learn from.

**Interview answer**

> HITL is a selective escalation policy. The system combines model uncertainty, action risk, reversibility, permissions, and business rules to decide between automatic execution, requesting confirmation, or handing off to a human. It records what the human changed, on what basis, and the final outcome, and uses them to update rules, data, and evaluation.

<details markdown="1">
<summary><b>Deep dive</b>: why is an uncertainty threshold alone not enough?</summary>

A model can be highly confident in a wrong answer, and the same level of uncertainty carries completely different risk for “recommend a song” and “refund ten thousand dollars.” The escalation policy has to calibrate uncertainty and multiply it by impact and irreversibility. It also has to account for the capacity of the human queue, or the system simply converts risk into unbounded waiting.

</details>
</details>

## HITL is a risk-control policy

Human-in-the-Loop does not mean “hand everything to a person whenever the model cannot do it.” It means having the system actively request confirmation, correction, or takeover when **risk is high, evidence is insufficient, the task is novel, or the action cannot be undone**.

## First, a scenario that needs escalation to a human

Asking a model to “write an email for me” and to “send this email for me directly” look very similar, but the risk is completely different.

- Writing a draft: a mistake can still be fixed, so it can usually be done automatically.
- Sending the email: it affects a real relationship, so it is best to have the user confirm first.
- Sending in bulk to customers: the impact is larger and may require additional permissions and expert review.

The key to HITL is not asking a human whenever model confidence is low. It is looking at **the probability of error, the size of the impact, and whether the action is reversible**, all together.

## What roles can a human play in the system

| Role | What they do |
| --- | --- |
| Teacher | Provides demonstrations, preferences, corrections, and explanations |
| Reviewer | Checks high-risk outputs and critical actions |
| Approver | Authorizes before external side effects occur |
| Collaborator | Works with the model to supply context and decompose the problem |
| Auditor | Periodically looks for blind spots in the automated system and its evaluators |
| The user themselves | Expresses goals, sets boundaries, and corrects personalized state |

## When is escalation to a human worth it

One simple intuition:

> **Value of intervention ≈ probability of error × size of impact × degree of irreversibility − human cost**

Common triggers include:

- several models or evaluators clearly disagree;
- the input falls outside the historical distribution and the system has never handled it before;
- the evidence search found is contradictory or not reliable enough;
- the action would send a message, spend money, modify data, or affect other people;
- the user's goal is unclear and the cost of continuing to guess is high;
- the agent enters a loop, falls back repeatedly, or shows abnormal cost.

## Human feedback cannot be stored as just a label

If the reviewer only clicks “agree / disagree,” the model still will not know later what went wrong.

A useful human intervention should, as far as possible, preserve:

- why this case was escalated;
- which evidence the reviewer saw;
- what decision was finally made;
- which part of the model output or action was changed;
- why it was changed that way;
- whether this case should update tests, rules, the policy, or the training data.

## Three timings for intervention

### Real-time confirmation

Have the user approve before the action happens. This suits high-risk or irreversible operations.

### After-the-fact spot checks

Sample from completed trajectories to find systematic failures, judge bias, and new types of problems.

### Periodically redesigning the rules

As models and tasks change, re-examine the rubric, escalation thresholds, permissions, data retention, and user control.

## Common mistakes

- **Treating the human as a fallback API:** no context is provided, yet a quick decision is demanded.
- **Looking only at the final answer:** search, tools, side effects, and state updates are ignored.
- **Collecting labels but not reasons:** no clear learning signal can be formed afterward.
- **Escalation rules that never change:** old thresholds are still used after model capability and task risk have changed.
- **Feedback going straight into training:** annotator variation, policy bias, and data leakage are not handled.
- **Automation bias:** the more confident the model sounds, the more readily people accept it without thinking.

## What good HITL should ultimately reduce

The goal is not to make more and more tasks depend on people. It is to put the most valuable human judgment in the right place, and then turn that judgment into:

- new deterministic rules;
- better eval cases;
- a clearer failure taxonomy;
- reusable training data;
- a more sensible boundary for automation.

## Connections to other chapters

- [Agent Observability](agent-observability.en.md) gives the reviewer complete evidence of the process.
- [Evaluation](../07-evaluation/README.en.md) decides which cases need human calibration.
- [Data and feedback](../01-data-and-feedback/README.en.md) decides how human judgment enters later learning.
- [Personal AGI](../09-personal-agi/README.en.md) requires that the user can control and correct long-term state.
