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

### Useful: did it actually complete the task

Not just fluent language, but whether the result can be executed and whether it reduces the user's work.

### Relevant: does it understand the current intent

The same user may have completely different goals at different times. Long-term preference must not override the current task.

### Broad: does it trap the user in a narrow region

Heavy personalization may raise short-term relevance while reducing new content and new choices.

### Controllable: can the user correct and intervene

The user should be able to view, edit, or delete memory, confirm high-risk actions, and see which information and permissions the system used.

### Calibrated: does the model know what it does not know

When evidence is insufficient, asking a follow-up question or admitting uncertainty is usually better than generating confidently.

### Consistent: does it stay coherent across sessions

Information should remain usable across sessions only while its retention is authorized, it remains valid, and it is relevant to the task. Consistency does not mean never forgetting, or turning one incidental behavior into a permanent preference.

## Why single-turn evaluation is not enough

Many important failures only appear in the interaction trajectory:

- a wrong memory keeps being reused;
- the user has already corrected it, but the system did not update;
- every turn's answer is fine, but the overall goal gradually drifts;
- personalization grows narrower and narrower;
- after a tool fails, the system pretends the task is complete.

So we need session-level evaluation of task completion across turns, and longitudinal evaluation of whether problems are resolved across repeated use—not only single prompt–response pairs.

Start with a timeline rather than simply asking, “Do you remember me?” Store “I usually prefer quiet accommodation”, add “for this trip only, stay near the station”, let the trip end, then delete the accommodation preference, and check which record the system uses at each step. The [memory lifecycle](../02-memory/memory-lifecycle.en.md#timeline-test) already turns this timeline, the rule of replaying only what was known at the time, and the with- and without-memory comparison into checkable steps. In an experience evaluation, also hold the model, tools, and budget fixed, and ask again in a new session to confirm that neither the deleted records nor anything derived from them is still used.

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
