# Agents: review questions

[中文](review.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-09

## Interview questions

<details class="interview" markdown="1">
<summary>What is the difference between a workflow and an agent? When should you not use an agent?</summary>

A workflow chains models and tools along code paths written in advance; an agent lets the model decide the process and the tools. When the steps can be listed in advance, when there is no verification signal, or when a wrong step is costly (sending, paying, deleting), do not reach for an agent directly: write a workflow first, or add a person's confirmation at the critical steps.

</details>

<details class="interview" markdown="1">
<summary>What parts make up an agent?</summary>

A model (decides the next step), tools (act on the world), a loop (results go back to the model), and a stopping condition (done, or out of budget), plus a context or memory holding intermediate state. ReAct's "think a step, take a step" is the most common form of the loop.

</details>

<details class="interview" markdown="1">
<summary>Why is coding one of the most mature settings for agents?</summary>

Because its verification signal is the strongest: code runs, tests pass, every step gets objective feedback, and the agent can safely try again. Settings such as customer support or personal assistants are much harder to judge automatically and lean more on rules, permissions, and people. Even in coding, passing tests does not mean the change is right.

</details>

<details class="interview" markdown="1">
<summary>How do you evaluate the reliability of a customer-support agent?</summary>

Not by single-run success alone: check whether the same task is done right every time it is repeated, as with τ-bench's pass^k. Also check policy compliance, whether tool calls are correct, whether hard cases are handed to a person, and read complete run traces rather than only the final answer.

</details>

<details class="interview" markdown="1">
<summary>When do you use a frontier API, and when do you self-host an open model?</summary>

Capability first: if an open model misses the bar on your own evaluation set, use a frontier model. If data cannot leave your environment, you must deploy it yourself. With high, steady traffic, or a need to fine-tune on your own data, self-hosting is cheaper and more controllable. At low or uncertain volume, start on an API and measure quality and cost. In practice they are often mixed: routing, cascades, distillation, fallback.

</details>

<details class="interview" markdown="1">
<summary>Why does a small gap in per-step accuracy grow large in an agent?</summary>

Errors compound across steps. At 95% per step, all 20 steps are right only about 36% of the time; at 99% per step, about 82%. In a long task, a small gap in per-step capability decides whether the whole thing is usable.

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>where the line between a workflow and an agent sits, and what each costs;</li>
    <li>why having a verification signal is the key test of whether a setting suits an agent;</li>
    <li>why an agent that reads web pages must treat their content as data, not instructions;</li>
    <li>why self-hosting has a fixed cost floor, and when it becomes cheaper than an API;</li>
    <li>what besides cost decides between a frontier API and an open model.</li>
  </ol>
</div>

## Further reading

- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): the workflow and agent distinction, and common structures
- [ReAct](https://arxiv.org/abs/2210.03629): interleaving reasoning and acting
- [SWE-bench](https://arxiv.org/abs/2310.06770): evaluating coding agents on real issues
- [τ-bench](https://arxiv.org/abs/2406.12045): tools and dialogue under policy constraints, and pass^k
- [WebArena](https://arxiv.org/abs/2307.13854)
- [OSWorld](https://arxiv.org/abs/2404.07972)
- [Not what you've signed up for](https://arxiv.org/abs/2302.12173): indirect prompt injection
- [FrugalGPT](https://arxiv.org/abs/2305.05176): model cascades
- [RouteLLM](https://arxiv.org/abs/2406.18665): routing between a strong and a weak model
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang](https://arxiv.org/abs/2312.07104)
