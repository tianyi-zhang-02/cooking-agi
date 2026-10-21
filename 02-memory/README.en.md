# Representation and Memory: How Should a Model Remember a Person?

[中文](README.md) · **English**

Start with what an assistant might need to remember. Then follow [writing, updating, and forgetting](memory-lifecycle.en.md) to see what happens when the user changes their mind, information expires, or a record needs to be deleted.

For a concrete example, start with [scheduling preferences during a trip](memory-lifecycle.en.md#two-clocks). For experiments, see [memory evaluation and public benchmarks](memory-lifecycle.en.md#memory-evaluation).

## Storing more is not always remembering better

Storing a conversation is easy. The harder part is finding **applicable information with clear provenance**, and updating it when a user changes their mind rather than treating old statements as permanent facts.

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

For an assistant used over time, **user state needs to be revisable**. An old judgment should not become a permanent fact.

## How representation and memory differ

- **Representation**: how information is encoded, right now, into a form the model can use.
- **Memory**: which information persists across time, and how it is later read, updated, and forgotten.

Representation concerns how information is encoded. Memory also concerns what to keep, when to retrieve it, and how to update it later.

## Memory can be split into layers

These categories separate uses, not necessarily storage systems: they can overlap, and do not require five databases. [CoALA §4.1](https://arxiv.org/html/2309.02427v3#S4.SS1) discusses working, episodic, semantic, and procedural memory. We also list preferences and goals separately to examine personal assistants.

### Working memory

The context the current task is using: the conversation so far, open files, and this turn's tool results. Context-window and inference-cost limits make it important to prioritize information that this task can actually use.

### Episodic memory

Specific past events, such as the user planning a Tokyo trip last week. This records the event and when it happened, without necessarily turning it into a lasting preference.

### Semantic memory

Relatively stable facts, drawn from explicit statements or summarized across events: for example, that the user mainly works in Python. Inferred facts still need provenance; storing an inference does not make it certain.

### Preference and goal state

What the user likes, what they are pursuing, and the evidence and confidence behind those judgments. It tends to change, and its parts may conflict.

### Procedural memory

Reusable procedures or skills: for expense reports, extract dates and amounts, check for duplicates, then ask the user to confirm. This differs from “the user likes conclusions first”: one is how to perform a task; the other is a preference. Procedural knowledge may also reside in model parameters or code, not just stored text.

## The four operations a memory system has to get right

### Write

Not every sentence deserves to become long-term memory. The system has to judge whether the information is stable, whether it bears on future tasks, whether the user has confirmed it, and whether storing it is appropriate.

### Read

Once there is a lot of memory, the key question becomes “which part should the current task pull out.” At its core this is also a search problem.

### Update

New evidence does not necessarily overwrite old information. Where retention is permitted, keep change history, provenance, and confidence. But keeping history is not a reason to retain information the user asked to delete.

### Forget

Stale information can be downweighted or excluded from retrieval; incorrect information needs correction or removal. **Downweighting is not deletion.** A deletion request also has to reach copies in summaries, vector indexes, caches, and backups; see [why a summary can remember something that was deleted](memory-lifecycle.en.md#deletion-propagation).

## Why one vector may not be enough

One person may like machine learning, travel, rock music, and animals all at once. Compress the whole history into one vector and the mainstream interests may drown out niche but real needs.

Work through the smallest case. Of a user's last 10 interactions, 8 are about machine learning (ML) and 2 about jazz. Reduce the two kinds of content to the directions $(1,0)$ and $(0,1)$, score by dot product, and keep only the top 2 each time:

| Candidate | One average vector $(0.8,0.2)$ | Two interest vectors $(1,0)$ and $(0,1)$, higher score |
| --- | --- | --- |
| ML tutorial $(1,0)$ | 0.8 | 1 |
| Popular item touching both $(0.6,0.6)$ | 0.6 | 0.6 |
| Jazz album $(0,1)$ | 0.2 | 1 |

With one vector, the top 2 are the ML tutorial and the popular item; the jazz album comes third and is left out. With two interest vectors it returns to the top 2. [MIND (Li et al., 2019)](https://arxiv.org/abs/1904.08030) learns several interest vectors per user for exactly this retrieval stage. The table also shows the cost: the two vectors count equally, so if those 2 jazz interactions were a gift for a friend, they are inflated into an interest as strong as ML, and retrieval runs once more. Interest vectors should therefore carry counts, time, and provenance, and be weighted by the task at read time.

So it is worth considering:

- several interest vectors instead of one average point;
- encoding the profile, the long-term history, and the current session separately;
- using routing / gating to decide which part of the state this task calls on;
- storing evidence and time so the representation can be explained and corrected.

Compare whether these designs retrieve less frequent interests more reliably, mistake temporary behavior for lasting preferences, or add too much retrieval cost. Multiple vectors can preserve more distinctions, but those distinctions still need to help with the task at hand.

## How to tell whether memory is working

Testing “did it recall past information” is not enough. Also check:

- whether the retrieved memory really helps the current task;
- whether a wrong memory keeps contaminating later answers;
- whether the system really updates after the user corrects it;
- whether it becomes more helpful across sessions rather than merely repeating known preferences;
- whether the system can explain which evidence a judgment came from.

## How it connects to other topics

- [Data and feedback](../01-data-and-feedback/README.en.md) decides where memory first comes from.
- [Search](../04-search/README.en.md) decides how the current task reads memory and external evidence.
- [Model Experience](../08-model-experience/README.en.md) decides whether the user feels continuity and control.
- [Evaluation](../07-evaluation/README.en.md) checks memory's helpfulness, contamination, and ability to update.
