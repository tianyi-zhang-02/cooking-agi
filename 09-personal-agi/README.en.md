# Personal AGI: What Does a Model That Truly “Knows You” Need?

[中文](README.md) · **English**

## Personal AGI needs a revisable user state

Personal AGI is not a chatbot that knows your name. It is a system that, over long-term interaction, forms an understanding, admits uncertainty, accepts correction, and gradually learns how to help you.

## Start with a recommendation that misunderstood you

You tell the model: “Help me plan next month's trip.”

An ordinary assistant may handle only that one sentence. Personal AGI should also know:

- whether you prefer natural scenery or ticking off city landmarks;
- that last time you ended up exhausted because the itinerary was too packed;
- whether this trip is with friends or with family;
- which of these are only past preferences and which still hold;
- which information is uncertain and should be checked with you first.

The real value is not “remembering more.” It is **using the right understanding at the right time**.

## How it differs from ordinary personalization

Traditional personalization often does this: given historical behavior, predict what the user is most likely to click next.

Personal AGI wants to go further and handle:

- what task the user is trying to accomplish right now;
- how long-term goals and short-term intent coexist;
- how the model transfers its understanding of the user across tasks;
- how the user views, edits, or deletes the state the model has formed;
- how the system knows that its judgment about the user may be wrong.

So it is not just a recommender, and not just memory. It needs a complete closed loop.

## Six necessary parts

### 1. User state

Store goals, preferences, knowledge, constraints, and the current situation, but attach time, evidence, and confidence.

### 2. Long-term memory

Know what is worth keeping, when to retrieve it, how to update it, and when it should be forgotten.

### 3. Search and tools

Be able to find up-to-date information, personal files, and external evidence instead of relying only on model parameters.

### 4. Reasoning and action

Combine the user state, the current task, and external evidence to decide whether to answer, ask a follow-up question, or execute a tool.

### 5. Feedback and learning

Learn from corrections, preferences, and real outcomes, without treating every observed behavior as true intent.

### 6. User control

Let the user know what the system has remembered and why it judged the way it did, and let them correct or retract it.

## The hardest problems

### A user is not one fixed vector

A person can hold several interests and roles at once, and they change over time. Compressing the whole history into one average representation easily makes niche but important intents disappear.

### Behavior is not the goal

What the user clicked is shaped by exposure, interface, and timing. The system needs to distinguish “the user chose it” from “the user was only given this choice.”

### More personalized is not necessarily better

If the model always caters to known preferences, the experience may grow narrower and narrower. A good Personal AGI should know when to exploit familiar information and when to offer new options.

### A long-term relationship needs a different evaluation

One answer can hardly prove that the model really understands the user. What matters more is whether it reduces repeated communication, whether it can absorb corrections, and whether it is still helpful weeks later.

## Continue reading

- [Representation and memory](../02-memory/README.en.md)
- [Data and feedback](../01-data-and-feedback/README.en.md)
- [Search](../04-search/README.en.md)
- [Model Experience](../08-model-experience/README.en.md)
- [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md)

## Reference papers

- [Generative Agents](https://arxiv.org/abs/2304.03442)
- [MemGPT](https://arxiv.org/abs/2310.08560)
