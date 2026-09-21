# Representation and Memory: How Should a Model Remember a Person?

[中文](README.md) · **English**

## Start here: memory is not storing all of history

Memory is not storing every historical event. It is letting the model find, when it needs it, **information that is still relevant, has clear provenance, and can be revised by new evidence**.

## Why one user needs several kinds of memory

Six months ago a user said they do not drink coffee. Today they start researching pour-over.

A crude system might:

- remember “does not drink coffee” forever and keep filtering out all coffee content;
- or look only at today's behavior and immediately decide the user has become a coffee enthusiast.

A better system keeps time and uncertainty:

```text
Long-term state: has usually not drunk coffee in the past
Current intent: has recently been actively learning about pour-over
Confidence: the interest may be shifting; do not draw a permanent conclusion yet
Evidence: three searches + one explicit question
```

This is why Personal AGI needs a **revisable user state** rather than a static personal profile.

## How representation and memory differ

- **Representation**: how information is encoded, right now, into a form the model can use.
- **Memory**: which information persists across time, and how it is later read, updated, and forgotten.

Representation answers “how do we describe it now.” Memory answers “should we still believe it later.”

## Memory can be split into layers

### Working memory

The context the current task is using: the conversation so far, the open files, and this turn's tool results. Its capacity is small, but it needs to be very precise.

### Episodic memory

What specifically happened in the past, for example that the user planned a Tokyo trip last week. It keeps events and time and is in no hurry to generalize them into permanent preferences.

### Semantic memory

Relatively stable facts distilled from many events, for example that the user usually works in Python or has long lived in a particular time zone.

### Preference and goal state

What the user likes, what they are pursuing, and the evidence and confidence behind those judgments. It tends to change, and its parts may conflict.

### Procedural memory

The system learns how to work with this user, for example whether they prefer the conclusion first or the full derivation.

## The four operations a memory system really has to solve

### Write

Not every sentence deserves to become long-term memory. The system has to judge whether the information is stable, whether it bears on future tasks, whether the user has confirmed it, and whether storing it is appropriate.

### Read

Once there is a lot of memory, the key question becomes “which part should the current task pull out.” At its core this is also a search problem.

### Update

New evidence does not necessarily overwrite old information. The system needs to keep the history of change, the source, and the confidence, rather than storing only the last value.

### Forget

Stale, wrong, sensitive, or no-longer-useful information should be deleted, downweighted, or no longer retrieved. A system that never forgets is not necessarily smarter.

## Why one vector may not be enough

One person may like machine learning, travel, rock music, and animals all at once. Compress the whole history into one vector and the mainstream interests may drown out niche but real needs.

So it is worth considering:

- several interest vectors instead of one average point;
- encoding the profile, the long-term history, and the current session separately;
- using routing / gating to decide which part of the state this task calls on;
- storing evidence and time so the representation can be explained and corrected.

The point is not “the more vectors the better.” It is whether the representation preserves the distinctions the task actually needs.

## How to tell whether memory is working

Testing “did it recall past information” is not enough. Also check:

- whether the retrieved memory really helps the current task;
- whether a wrong memory keeps contaminating later answers;
- whether the system really updates after the user corrects it;
- whether, after many sessions, it becomes more helpful rather than narrower and narrower;
- whether the system can explain which evidence a judgment came from.

## How it connects to other topics

- [Data and feedback](../01-data-and-feedback/README.en.md) decides where memory first comes from.
- [Search](../04-search/README.en.md) decides how the current task reads memory and external evidence.
- [Model Experience](../08-model-experience/README.en.md) decides whether the user feels continuity and control.
- [Evaluation](../07-evaluation/README.en.md) checks memory's helpfulness, contamination, and ability to update.
