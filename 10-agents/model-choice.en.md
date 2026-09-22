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

## The usual compromise: mix them

- **Routing and cascades**: easy steps go to a small model, hard ones to a frontier model. FrugalGPT (2023) studies cascades that call models in order and stop once the answer is good enough; RouteLLM (2024) trains a router that splits requests between a strong and a weak model.
- **Distillation**: use a frontier model to produce traces on a narrow, high-volume task, then fine-tune a small model you can serve yourself.
- **Fallback**: your own model handles the common cases, and uncertain ones go to a frontier model.

For self-hosting, vLLM (PagedAttention) and SGLang are common inference stacks.

## An order of questions

1. Can the data leave your environment? If not, you must deploy it yourself.
2. Does an open model meet the bar on your own evaluation set? If not, use a frontier model.
3. Is traffic both high and steady? Only then does self-hosting's cost advantage appear.
4. Do you need to fine-tune or run RL on your own data? If so, lean towards open weights.
5. If none of this is clear, launch on an API, measure quality and cost, and revisit.
