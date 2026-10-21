# Model experience: does the assistant get more useful over time?

[中文](README.md) · **English**

<span id="model-experience-what-is-the-model-the-user-actually-experiences"></span>

## Look beyond a single answer {#user-experience-is-a-long-term-trajectory}

Model experience is not just about how polished one answer is. It is about continued use: does the assistant remember relevant context, respond to corrections, leave key decisions under your control, and actually reduce your workload?

## What changes across sessions? {#what-the-user-experiences-is-a-process-of-continued-use}

Two assistants can both correctly answer “plan a week of workouts for me.”

Assistant A asks about your goals, equipment, and injuries all over again every time.

With your permission to retain that information, Assistant B remembers that you once injured your knee, but checks whether it still applies before using it. When the plan does not fit, it retains your correction within the scope you allow instead of repeating the mistake.

This is a hypothetical example, not a measured result. The two may score similarly on a single-turn benchmark while offering substantially different long-term experiences.

## The dimensions a user experiences

Staying with the workout plan, turn each dimension into a question you can check:

| Dimension | What to ask | How to check it in the workout plan |
| --- | --- | --- |
| Useful | Did it actually complete the task? | The plan states what to do each day, for how long, with which equipment, and the user can follow it; "work your core more" reads fluently but cannot be used |
| Relevant | Does it understand what the user wants this time? | If the user only wants to rebuild fitness this week, last year's long-term "build muscle" goal should not take over; long-term preference must not override the current task |
| Broad | Does it keep recommending the same kind of thing? | Saying once that you like running does not mean every plan should be only running; the closer recommendations track known preferences, the fewer new options may appear |
| Controllable | Can the user correct and intervene? | The user can view, edit, or delete the "injured knee" memory and see that this plan used it; actions such as booking a paid class for the user need confirmation first |
| Calibrated | Does it say so when information is missing? | If it does not know whether the user owns dumbbells, it asks instead of assuming; when evidence is insufficient, asking or admitting uncertainty is usually better than generating confidently |
| Consistent | Can a new conversation pick up where the last one left off? | A correction from last week, "no training on Wednesdays", is still remembered in a new conversation this week; once the user withdraws it or it expires, it is no longer used |

Consistency does not mean never forgetting, or turning one incidental behavior into a permanent preference. Information should carry across sessions only while its retention is authorized, it remains valid, and it is relevant to the task.

## Why single-turn evaluation is not enough

Many important failures only appear in the interaction trajectory:

- a wrong memory keeps being reused;
- the user has already corrected it, but the system did not update;
- every turn's answer is fine, but the overall goal gradually drifts;
- personalization grows narrower and narrower;
- after a tool fails, the system pretends the task is complete.

So we need session-level evaluation of task completion across turns, and longitudinal evaluation of whether problems are resolved across repeated use—not only single prompt–response pairs.

Start with a timeline: store “I usually prefer quiet accommodation”, add “for this trip only, stay near the station”, let the trip end, then delete the accommodation preference. Checking which record the system uses at each step is more informative than simply asking, “Do you remember me?”

The [memory lifecycle](../02-memory/memory-lifecycle.en.md#timeline-test) provides steps for this test, including replaying only information available at the time and comparing versions with and without memory. For experience evaluation, also hold the model, tools, and budget fixed. Ask again in a new session to confirm that neither the deleted records nor anything derived from them is still used.

These are proposed tests, not demonstrated improvements. Synthetic replay can expose mechanism failures, but cannot establish satisfaction weeks into real use. A real-use comparison can randomly assign users to groups, keep assignments stable for the observation period, and report dropouts and missing feedback, rather than only satisfied users who remain. [LongMemEval (arXiv v2, Section 3)](https://arxiv.org/html/2410.10813v2#S3) offers tests of updates, temporal reasoning, and abstention; its QA scores do not replace this longitudinal experience evaluation.

## How offline metrics connect to experience

Recall, NDCG, judge score, and task success are all useful, but each metric sees only part of the picture.

A more complete evaluation can combine:

- task completion rate and executable verification;
- relevance, coverage, and diversity;
- the number of user corrections and the cost of repeated explanation;
- whether memory updates and retractions succeed;
- the actual outcomes of tools and actions;
- satisfaction and degree of reliance after many sessions.

These are not simply metrics to maximize or minimize. Fewer corrections may mean the user gave up; more frequent use may mean repeated task failure. Read them alongside task outcomes, dropout, and whether users still trust incorrect answers. Aggregate repeated sessions by user, report memory-dependent, history-free, and correction/deletion cases separately, and track latency and cost so frequent users or easy tasks do not hide regressions.

### Worked example: fewer corrections, worse experience (teaching example)

This is an expected-value calculation with assumed numbers, not a user experiment. Twenty users are randomly split into two groups of 10: group A gets a version without memory, group B a version with memory. Assume that 4 and 9 users respectively remain in week 4. Each attempts one task, with assumed success probabilities of 90% and 80%. The outcome here is completion through the assistant, so users who have stopped using it count as not completing through it:

| Version | Users assigned | Still using in week 4 | Corrections per session among those left | Assumed success probability among those left | Expected completion by original assignment |
| --- | --- | --- | --- | --- | --- |
| A (no memory) | 10 | 4 | 0.5 | 90% | $4\times0.9/10=36\%$ |
| B (memory) | 10 | 9 | 0.8 | 80% | $9\times0.8/10=72\%$ |

Looking only at the users who stayed, A has fewer corrections and a higher success probability, so it looks better. Include everyone originally assigned in the denominator, and B's expected completion rate is twice A's. The 36% and 72% are expectations calculated from assumed probabilities, not observed proportions from 4 and 9 people each attempting one task.

A may have fewer corrections because the users who needed to correct it most have left, but this table cannot establish why they left. Stopping use is also different from a missing outcome: this example counts the former as no completion through the assistant, not every unobserved result as a failure.

So report three numbers together: how many users dropped out of each group; the result by original assignment (ITT, intention-to-treat), which keeps dropouts in the denominator; and the result for remaining users only. Report only the last one, and the satisfied users who stayed will hide the regression.

## How it connects to other topics

- [Personal AGI](../09-personal-agi/README.en.md) defines the goal of the long-term relationship.
- [Representation and memory](../02-memory/README.en.md) decides how the system maintains continuity.
- [Search](../04-search/README.en.md) explains how the model finds the information it needs.
- [Evaluation](../07-evaluation/README.en.md) breaks subjective experience into evidence that can be verified.
- [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md) provides the mechanisms for control, escalation, and correction.

## Reference papers

These works study long-context use and preference training; they are not evidence that long-term personalization has improved.

- [Lost in the Middle](https://arxiv.org/abs/2307.03172v1)
- [InstructGPT](https://arxiv.org/abs/2203.02155v1)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290v1)
