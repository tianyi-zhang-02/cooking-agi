# Model Experience: What Is the Model the User Actually Experiences?

[中文](README.md) · **English**

## User experience is a long-term trajectory

Model Experience is not how polished one particular answer is. It is what the user perceives across continued use: whether the model understands context, whether it deserves trust, whether it gives a sense of control, and whether it really becomes more and more helpful.

## What the user experiences is a process of continued use

Two assistants can both correctly answer “plan a week of workouts for me.”

Assistant A asks about your goals, equipment, and injuries all over again every time.

Assistant B remembers that you once injured your knee, but before using that information it confirms whether it still holds; when the plan does not fit, it absorbs the correction instead of making the same mistake next time.

On a single-turn benchmark the two may score about the same. The long-term experience is completely different.

## The dimensions a user experiences

### Useful: did it actually complete the task

Not just fluent language, but whether the result can be executed and whether it reduces the user's work.

### Relevant: does it understand the current intent

The same user may have completely different goals at different times. Long-term preference must not override the current task.

### Broad: does it trap the user in a narrow region

Heavy personalization may raise short-term relevance while reducing new content and new choices.

### Controllable: can the user correct and intervene

The user should be able to view and edit memory, confirm high-risk actions, and know why the system did what it did.

### Calibrated: does the model know what it does not know

When evidence is insufficient, asking a follow-up question or admitting uncertainty is usually better than generating confidently.

### Consistent: does it stay coherent across sessions

The model should not learn something today and completely forget it tomorrow; nor should it make one incidental behavior permanent.

## Why single-turn evaluation is not enough

Many important failures only appear in the interaction trajectory:

- a wrong memory keeps being reused;
- the user has already corrected it, but the system did not update;
- every turn's answer is fine, but the overall goal gradually drifts;
- personalization grows narrower and narrower;
- after a tool fails, the system pretends the task is complete.

So we need session-level and longitudinal evaluation, not only single prompt–response pairs.

## How offline metrics connect to experience

Recall, NDCG, judge score, and task success are all useful, but each metric sees only part of the picture.

A more complete evaluation can combine:

- task completion rate and executable verification;
- relevance, coverage, and diversity;
- the number of user corrections and the cost of repeated explanation;
- whether memory updates and retractions succeed;
- the actual outcomes of tools and actions;
- satisfaction and degree of reliance after many sessions.

## How it connects to other topics

- [Personal AGI](../09-personal-agi/README.en.md) defines the goal of the long-term relationship.
- [Representation and memory](../02-memory/README.en.md) decides how the system maintains continuity.
- [Search](../04-search/README.en.md) decides what information space the model can offer.
- [Evaluation](../07-evaluation/README.en.md) breaks subjective experience into evidence that can be verified.
- [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md) provides the mechanisms for control, escalation, and correction.

## Reference papers

- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
