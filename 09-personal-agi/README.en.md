# Personal AGI: What Does a Model That Truly “Knows You” Need?

[中文](README.md) · **English**

## Personal AGI needs a revisable user state

Here, Personal AGI names an aspiration for a personal assistant, not a claim that general intelligence has been achieved. Through long-term interaction, it would learn how to help you, ask when uncertain, and accept corrections—not just remember your name.

## Start with planning a trip

You tell the model: “Help me plan next month's trip.”

An ordinary assistant may handle only that one sentence. Personal AGI should also retrieve within its permissions, or ask you to confirm:

- whether you prefer natural scenery or ticking off city landmarks;
- that last time you ended up exhausted because the itinerary was too packed;
- whether this trip is with friends or with family;
- which of these are only past preferences and which still hold;
- which information is uncertain and should be checked with you first.

Remembering more only helps if the information fits the task. **Which details matter for this trip, and which earlier assumptions still hold?**

## How it differs from ordinary personalization

Click prediction is a common personalization task, but personalization can also consider satisfaction, task completion, and long-term goals. Here, the focus is how an assistant could use its understanding of the user across different tasks.

Such an assistant would also need to handle:

- what task the user is trying to accomplish right now;
- how long-term goals and short-term intent coexist;
- how the model transfers its understanding of the user across tasks;
- how the user views, edits, or deletes the state the model has formed;
- how the system knows that its judgment about the user may be wrong.

Recommendation or memory alone is not enough: the system needs to retrieve relevant information, carry out the task, and update its understanding from feedback.

## Six parts to consider {#six-necessary-parts}

This is one way to organize the design, not a sufficient recipe for AGI or a requirement to use six separate models. Start with the tasks the assistant should help with, then choose which capabilities to build.

### 1. User state

Retain only the goals, preferences, knowledge, constraints, and context needed for the task and permitted for storage. Track source, scope, when information was learned, and when it applies. Separate explicit statements from model inferences; a model-written “high confidence” label is not a calibrated probability of correctness.

In the trip example, three records could look like this (teaching example):

| Record | Source | Scope | Valid until | How this trip uses it |
| --- | --- | --- | --- | --- |
| Last itinerary was too packed and tiring | The user said so after the last trip | Travel pace | Long-term, but confirmed before each plan | Schedule one fewer sight per day and tell the user why |
| Traveling with parents this time; keep it relaxed | The user said so for this trip | This trip only | The end of this trip | Affects this plan only |
| Likes the seaside | Model inference: last time the user picked one of the seaside options the assistant offered | Destination suggestions | Never confirmed | Not used as a fact; ask first |

The first two came from the user; the third is the system's guess. Why its evidence is weak is worked out under “Behavior is not the goal” below.

### 2. Long-term memory

Know what is worth keeping, when to retrieve it, how to update it, and when it should be forgotten.

For example, “I'm traveling with my parents this time; keep the itinerary relaxed” should affect this trip, not permanently label the user as disliking busy itineraries. How scope, validity, authorization, and deletion become read rules is covered in the [external memory lifecycle](../02-memory/memory-lifecycle.en.md); it concerns external storage and does not claim that deleting a database record erases information from model parameters.

### 3. Search and tools

Be able to find up-to-date information, personal files, and external evidence instead of relying only on model parameters.

### 4. Reasoning and action

Combine the user state, the current task, and external evidence to decide whether to answer, ask a follow-up question, or execute a tool.

### 5. Feedback and learning

Learn from corrections, preferences, and real outcomes, without treating every observed behavior as true intent. Updating user state is not the same as updating model weights; the latter is a separate process requiring authorization, independent evaluation, and a rollback plan. To tell whether a change touches memory and prompts or the weights, start from the breakdown in the [self-improvement series](../11-frontier/self-improvement/README.en.md).

### 6. User control

Let users inspect what was remembered and the evidence behind it, then correct or retract it. Access to a personal file does not authorize retaining all of it or sending it to an external tool; memory also grants no authority to act on the user's behalf.

Define purposes, access, and retention periods; offer a mode that does not write long-term memory, and enforce user isolation and minimum necessary disclosure at storage and tool boundaries. Explain which copies deletion covers, when it completes, and how logs or backups are handled. Disappearing from the interface does not mean every copy has been erased. These are design checks, informed by [NIST Privacy Framework 1.0, CT.PO-P and CT.DM-P](https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.01162020.pdf#page=27), not an achieved privacy guarantee.

## The hardest problems

### A user is not one fixed vector

A person can hold several interests and roles at once, and they change over time. Compressing all history into one representation may make niche intents harder to retrieve. Multiple vectors are not automatically better either: test whether they preserve useful distinctions under the same budget.

### Behavior is not the goal

What the user clicked is shaped by exposure, interface, and timing. The system needs to distinguish preference from a choice restricted by what was shown.

Suppose the assistant only recommends seaside destinations and the user selects one. Treating that as evidence that they only like the seaside creates a loop: recommendations determine exposure, exposure produces feedback, and feedback narrows recommendations. Log what was shown and what was explicitly corrected; offer different options within the user's constraints rather than treating unseen options as rejected. [Chaney et al., v1](https://arxiv.org/abs/1710.11214v1) study this confounding risk in simulations, not as an inevitable outcome of all personalization.

Walk through it with numbers (teaching example). Over three planning sessions, the assistant offers 3 destinations each time. The first time it offers 2 seaside places and 1 mountain place, and the user picks the seaside; the assistant then follows “recommend more of what was picked”, so sessions 2 and 3 offer only seaside places, and the user can only pick the seaside.

| Session | Options shown | User picked | Chance of picking the seaside for a user with no preference who picks at random |
| --- | --- | --- | --- |
| 1 | 2 seaside, 1 mountain | Seaside | $2/3$ |
| 2 | 3 seaside | Seaside | 1 |
| 3 | 3 seaside | Seaside | 1 |

Counting clicks alone, all 3 picks were the seaside, which looks like “loves the seaside 100%”. Yet a user with no preference leaves exactly the same record with probability $2/3\times1\times1\approx0.67$. Compare that with an idealized user who always picks the seaside whenever it is offered: that user leaves the record with probability 1. The probabilities under the two assumptions differ by only $1\big/\tfrac23=1.5$ times, so the record gives limited evidence for distinguishing them.

Change one condition: every session keeps 2 seaside and 1 mountain option. The chance that a user with no preference picks the seaside 3 times in a row drops to $(2/3)^3\approx0.30$, so the same record now differs by $1\big/(2/3)^3=27/8\approx3.4$ times between the two kinds of user and starts to count as evidence of a preference. A few different options buy data that can tell preferences apart; simply asking the user is often cheaper still.

This also assumes the indifferent user chooses independently and uniformly among the three options each time. The multiplier is a likelihood ratio under two behavioral hypotheses, not the probability that the user likes the seaside. Inferring that probability requires further assumptions, including a prior.

### More personalized is not necessarily better

If the model always caters to known preferences, the experience may grow narrower and narrower. A good Personal AGI should know when to exploit familiar information and when to offer new options.

### A long-term relationship needs a different evaluation

One answer can hardly prove that the model really understands the user. What matters more is whether it reduces repeated communication, applies corrections, and remains helpful weeks later. Use the [model-experience timeline](../08-model-experience/README.en.md) to test current intent, temporary exceptions, withdrawal, and a no-memory control. More clicks or greater reliance alone do not establish greater usefulness.

## Continue reading

- [Representation and memory](../02-memory/README.en.md)
- [Data and feedback](../01-data-and-feedback/README.en.md)
- [Search](../04-search/README.en.md)
- [Model Experience](../08-model-experience/README.en.md)
- [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md)

## Reference papers

Generative Agents studies simulated characters; MemGPT studies context and external-memory management. They offer mechanisms to study, not proof of Personal AGI or long-term privacy, safety, and user benefit.

- [Generative Agents](https://arxiv.org/abs/2304.03442v1)
- [MemGPT](https://arxiv.org/abs/2310.08560v1)
