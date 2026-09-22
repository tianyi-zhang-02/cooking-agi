# Agents: what an agent actually is

[中文](README.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>when to let the model choose its next step, and when not to</strong></div>
  <div><span>Prerequisites</span><strong>LLM calls · tool calling · evaluation</strong></div>
  <div><span>Core mechanism</span><strong>model + tools + loop + stopping condition</strong></div>
  <div><span>Common mistakes</span><strong>making everything an agent; no stopping condition and no way to verify</strong></div>
</div>

## An agent is four things

- **a model** that looks at the current situation and decides what to do next;
- **tools**, the interfaces that actually act on the world: search, reading and writing files, calling APIs, running code;
- **a loop**: a tool's result goes back to the model, which decides again;
- **a stopping condition**: when the job counts as done, or when the budget runs out and it must stop.

Add somewhere to keep intermediate state (the context, memory, external storage) and you have most agents in full.

## Workflows versus agents

Anthropic draws the line clearly in [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): a **workflow** chains models and tools along code paths written in advance; an **agent** lets the model decide the process and which tools to use. Here is the same task done both ways:

<!-- widget:tx-agent-loop -->

## First ask: do you need an agent?

An agent trades higher latency and cost for the ability to adapt to open-ended tasks. The same article advises finding the simplest solution first and adding complexity only when it is needed. A few questions decide it:

1. **Can the steps be listed in advance?** If so, write a workflow.
2. **Is there a verification signal?** Can the tests run, can the answer be checked, can the order be looked up? An agent without one can hardly tell when it got something right.
3. **What does a wrong step cost?** Irreversible actions such as sending email, paying, or deleting data should either stay out of an agent's hands or require a person to confirm.

## ReAct: think a step, take a step

ReAct (Yao et al., 2022) interleaves reasoning and acting: the model writes down its thinking for this step, calls a tool, and thinks about the next step once it sees the result. Most agent loops today are variations on this pattern.

## How this series reads

1. [Common building blocks](patterns.en.md): from a single call to orchestrator-workers, and what each suits
2. [How agents are used in different settings](scenarios.en.md): coding, search and research, customer support, data analysis, computer use, personal assistants
3. [Frontier API or self-hosted](model-choice.en.md): choosing a model, with a cost model you can fill in yourself
4. [Review questions](review.en.md): interview questions and a self-check

Related: [Agent Observability](../06-systems/agent-observability.en.md) on seeing what happened in a run, and [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md) on when a person should step in.
