# Agents: common building blocks

[中文](patterns.md) · **English**

> Reading time: ~2 min · Level: advanced · Last reviewed: 2026-09

## The smallest unit: an augmented model

Every structure starts from a model wired up with retrieval, tools, and memory (Anthropic calls it the augmented LLM). The workflows and the agent below are different ways of combining that unit.

## Five workflows and one agent

| Structure | How it works | When to use it |
| --- | --- | --- |
| Prompt chaining | Split the task into fixed steps, each step's output is the next one's input, with checks in between | The steps are clear, and a little latency is worth the accuracy |
| Routing | Classify first, then hand off to a specialised path | Inputs differ a lot and are better handled separately |
| Parallelization | Run independent subtasks at once (sectioning), or do the same thing several times and vote (voting) | Subtasks do not depend on each other, or you want a steadier result |
| Orchestrator-workers | One model breaks the task down on the fly, hands pieces to workers, and combines the results | The subtasks cannot be listed in advance, such as which files a change must touch |
| Evaluator-optimizer | One model generates, another evaluates and gives feedback, in a loop | There are clear evaluation criteria, and feedback really improves the result |
| Agent | The model decides every step in a loop, judges progress from the environment's feedback, and stops when the stopping condition holds | Open-ended tasks where the number of steps and the path are unknown in advance |

Going down the table, flexibility rises, and so do unpredictability, cost, and difficulty of debugging.

## How to choose

- Try a single call with a good prompt and retrieval first; many problems stop here;
- if it can be a workflow, make it a workflow: each step can be tested and changed on its own;
- switch to an agent only when the path truly cannot be fixed in advance, and give it a clear stopping condition and a budget.

## A framework, or the API directly

Many frameworks let you build an agent in a few lines, but they can also hide the prompts and the model's real inputs and outputs, which makes problems hard to trace. Anthropic's advice is to start with the model API directly, since many of these patterns take only a few lines; if you use a framework, make sure you understand what it does underneath.
