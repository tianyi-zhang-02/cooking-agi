# Agents: what an agent actually is

[中文](README.md) · **English**

> Reading time: ~5 min · Level: advanced · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>when to let the model choose its next step, and when not to</strong></div>
  <div><span>Prerequisites</span><strong>LLM calls · tool calling · evaluation</strong></div>
  <div><span>Core mechanism</span><strong>model + tools + loop + stopping condition</strong></div>
  <div><span>Common mistakes</span><strong>making everything an agent; no stopping condition and no way to verify</strong></div>
</div>

## First: the word keeps moving

The first thing I think of when I hear "agent" is Snake and Pac-Man :) For a long time an
agent was the thing that moves in a game: it looks at its environment, picks an action, gets
a little reward, and goes round again. Reinforcement learning is built on that loop, and
today's LLM agents are still that loop — what changed is who picks the action: written
rules, then a learned policy, now a model you ask.

<!-- widget:tx-agent-maze -->

A few things from the RL side are worth keeping, because they come back later:

- **It optimises later, not now.** The value function $V(s)$ is "how much is still to be had
  from this square if I keep playing well", and the discount $\gamma$ says how much later
  counts. Small $\gamma$ is short-sighted; large $\gamma$ will take the long way round. That
  is the value map in the figure above.
- **Exploration against exploitation.** Always take the best known route and you never find a
  better one; always try new things and you never collect.
- **It optimises whatever you actually pay for.** The fun and frightening one. OpenAI's boat
  race is the classic ([Faulty reward functions in the wild](https://openai.com/index/faulty-reward-functions/),
  2016): the reward was the pickups along the course, so the agent found that spinning in a
  circle farming pickups scored better than finishing the race — and spun, catching fire, ramming
  other boats and driving the wrong way, with a higher score than finishing normally. Reward hacking in RLHF is the same thing wearing a suit.
- **LLM agents inherit it.** Reward "the tests pass" and you may get edited tests.

So when you see the word "agent", it is worth asking which one is meant. The field moves fast
and the definition is genuinely contested — some people reserve it for systems that choose
their own process, others call a single tool-using call an agent. What follows is the common
sense of the word today.

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
