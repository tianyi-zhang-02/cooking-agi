# Agents: frontier API or self-hosted

[中文](model-choice.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-09

## First check the capability

An agent's errors compound across steps. If each step is independently right 95% of the time, the chance that all 20 steps are right is only

$$0.95^{20} \approx 0.36$$

so a small gap in per-step capability grows large over a long task. Evaluate on **your own task** first: does an open model meet the bar? If not, nothing else matters.

## Several dimensions at once

| Dimension | Frontier API | Self-hosted (open weights) |
| --- | --- | --- |
| Capability ceiling | usually higher, especially for long tasks and complex planning | depends on the model chosen; may be enough for a narrow task |
| Cost structure | pay per token, cheap at low volume | fixed GPU and staff costs, better value at high, steady volume |
| Latency | subject to the network and the provider's load | batch size and placement can be tuned to your needs |
| Data and compliance | data is sent to the provider | data stays in your environment |
| Customisation | mostly through prompts, some fine-tuning | fine-tuning and RL on your own data |
| Version control | the provider decides when a model changes or retires | you decide when to switch |
| Operations | almost none | inference stack, scaling, and monitoring are all yours |

## How to count the cost

Agents multiply token usage: one task often takes several to dozens of model calls, and every step resends an ever-longer context. The model below takes your own numbers: API cost grows linearly with tokens; self-hosting has a fixed floor (GPUs running around the clock plus the people who run them) and steps up only when one replica's capacity is exceeded.

<!-- widget:tx-agent-cost -->

## Not a choice of one: mixing them

Most systems end up serving the same traffic with both models rather than picking a side. Four ways to spend a request:

<!-- widget:tx-model-router -->

- **Cascade**: the small model answers first, and anything doubtful is escalated. This is what FrugalGPT (2023) studies. The escalated requests are paid for twice.
- **Router**: decide up front who gets the request, so nothing is paid for twice. RouteLLM (2024) trains exactly that. It commits before seeing the answer, so a wrong guess is never caught.
- **Distillation**: use a frontier model to produce traces on a narrow, high-volume task, then fine-tune a small model you can serve yourself.
- **Fallback**: your own model handles the common cases, and unusual ones go to a frontier model.

What decides whether either middle plan is worth it is the accuracy of the difficulty judgement in that figure. Judge well and both are a bargain; judge barely better than chance and neither beats picking one model and keeping the system simple — the extra layer is then only more surface to fail on.

## Three things cost does not cover

**Latency adds up in series.** An agent makes several model calls per task, and the user feels their sum. Time to first token and time per token pull in different directions: short replies are dominated by the first, long ones by the second. Self-hosting lets you tune batching, hardware and location; with an API you wait.

**Context is where the money goes.** Every step resends a longer history, so cost grows roughly with the square of the number of steps. Prompt caching — keeping the unchanged prefix warm — saves the most here, and what a provider caches, and for how long, is worth asking before you commit.

**Models move.** The model behind an API is updated and retired, and the same prompt can answer differently months later. Self-hosted weights do not drift on their own — the price is that security updates are yours too. Either way you need regression tests that catch "this worked last week".

## What self-hosting actually costs

Not "how many cards" — utilisation:

- **Concurrency and batching**: the more requests a card serves at once, the less each one costs. Traffic has peaks and troughs, and buying for the peak while running at the mean is the gap you pay for.
- **KV cache eats memory**: with long-context agents, memory runs out before compute does, which caps how many requests fit at once. vLLM's PagedAttention and SGLang's RadixAttention both exist to fight this.
- **People**: upgrades, autoscaling, monitoring, being on call. In a small team this line is often more expensive than the GPUs.

## An order of questions

1. Can the data leave your environment? If not, you must deploy it yourself.
2. Does an open model meet the bar on your own evaluation set? If not, use a frontier model.
3. Is traffic both high and steady? Only then does self-hosting's cost advantage appear.
4. Do you need to fine-tune or run RL on your own data? If so, lean towards open weights.
5. If none of this is clear, launch on an API, measure quality and cost, and revisit.
