# Choosing an agent model: APIs, self-hosting, and routing {#agents-frontier-api-or-self-hosted}

[中文](model-choice.md) · **English**

> Reading time: ~12 min · Level: foundations to engineering · Last reviewed: 2026-10

Extracting a date from a paragraph and debugging across several files do not necessarily need the same compute budget. One may need a short answer; the other may require reading code, running tests, and investigating the next failure.

Separate three decisions: **which model to use, where it runs, and how much computation this request gets.** Open-weight models can run behind hosted APIs. One model may support different reasoning budgets. An external router can choose between models. These are different controls.

For a practical starting point, read [the comparison](#several-dimensions-at-once) and [the checklist](#an-order-of-questions). For automatic thinking, start with [who decides](#thinking-control). The [four-request example](#routing-example) is small enough to check by hand.

## 1. Start with whether the task gets done {#first-check-the-capability}

Consider a notes-review assistant. One model may polish prose well but miss a sign error in an equation. Another calculates correctly but keeps rewriting passages the user wanted left alone. An aggregate benchmark score can hide both differences.

Collect a small, representative task set: definite mistakes, notes without errors, conflicting sources, and tool failures. Define completion before running candidates under the same conditions. Check the answer and whether the assistant read the right file, used the right version, and respected a review-only request.

Evaluate the whole execution for long tasks. Reading the wrong version early can send later reasoning in the wrong direction. Conversely, a failed step need not doom the task: the system might retry, use another tool, or ask for missing information.

<details markdown="1">
<summary>Why “95% per step becomes 36% over 20 steps” is only a warning</summary>

If a task requires exactly 20 independent steps, each succeeds with probability 0.95, and any failed step fails the task, then:

$$P_{\mathrm{success}}=0.95^{20}\approx0.358.$$

Real difficulty changes, errors correlate, and recovery may be possible. This motivates end-to-end evaluation rather than replacing it. Record task success, failure locations, recovery attempts, total time, and total cost.

</details>

## 2. What work comes with each deployment choice? {#several-dimensions-at-once}

| Dimension | Hosted API | Self-hosted open weights |
| --- | --- | --- |
| Task quality | Evaluate the specific model and version; “API” is not a capability tier | Evaluate here too; fine-tuning and tools may change outcomes |
| Cost | Tokens, caching, retries, and service arrangements | Compute, idle capacity, storage, operations, and maintenance time |
| Latency | Network and provider load; context, concurrency, and output length remain adjustable | Control placement, batching, and parallelism; own queues and maintenance |
| Data boundaries | Check region, retention, contracts, and private connections | Check logs, telemetry, tools, and external dependencies too |
| Customization | Depends on supported fine-tuning, tools, and configuration | More control, with training, evaluation, and release responsibilities |
| Versions | Some services offer snapshots, with retirement schedules to track | Pin weights, tokenizer, templates, and serving environment |

A small team might establish a baseline with an API that meets its data requirements rather than build serving first. A team with spare compute, deployment constraints, or steady traffic may start elsewhere. There is no universal request count at which self-hosting becomes the right choice.

## 3. Who decides whether to reason longer? {#thinking-control}

When a model offers a thinking mode, ask whether the caller selects it or the model chooses during generation. Both can be useful, but they are different capabilities.

<figure class="worked-update" lang="en" id="reasoning-decisions">
<figcaption>Date extraction and debugging can receive budgets at different points</figcaption>
<ol>
<li><strong>Before generation: the application specifies</strong><span>Assign a short budget to extraction and a larger one to debugging. A rule or configuration decides.</span></li>
<li><strong>During generation: the model chooses</strong><span>A suitably trained model can end its reasoning segment early or continue, within the maximum budget.</span></li>
<li><strong>Outside the model: a router selects</strong><span>Inspect the request and choose A or B. This is separate from either model's learned stopping behavior.</span></li>
</ol>
</figure>

**Caller-controlled modes.** Original hybrid Qwen3 models support thinking/non-thinking controls. This lets the caller choose; it does not establish autonomous selection. The later `Qwen3-4B-Instruct-2507` is non-thinking, so an earlier model's controls cannot be assumed for every checkpoint. [Qwen3 announcement](https://qwenlm.github.io/blog/qwen3/) · [Specific model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

**Learned selection.** AutoThink (2025) trains R1-style models to choose when to retain explicit reasoning. Its stages balance modes, improve performance, and introduce length-aware rewards. A prompt that produces both modes does not itself establish difficulty-aware allocation. [AutoThink §2–3](https://arxiv.org/html/2505.10832v2#S3)

A detail worth checking: stage 2 removes batch balancing but retains the mode-and-correctness-dependent base reward, rather than switching to binary accuracy reward. The main experiments concern mathematics. We have not reproduced them or established benefits for arbitrary agents.

**External routing.** The application estimates which model suits a request; neither candidate needs a thinking switch. [Adaptive depth](../00-foundations/looped/adaptive-depth.en.md) instead concerns internal layers or iterations. Hidden-state computation is not the same as generating more text.

## 4. Routing and cascading decide at different times {#not-a-choice-of-one-mixing-them}

| Approach | Decision point | Costs involved | Typical failure |
| --- | --- | --- | --- |
| Pre-generation routing | After reading the request, before generating | Router plus selected model | Select the wrong model without seeing its answer |
| Cascade | Generate, then check whether to escalate | First answer, check, and possible second answer | Miss an error or escalate unnecessarily |
| Fallback | After timeout, rate limiting, or failed validation | Retry, switch, and state recovery | Confuse restored service with correctness |
| Distillation | During training, not within this request | Teacher data, training, and maintenance | Transfer teacher bias or over-specialize |

[FrugalGPT (2023)](https://arxiv.org/abs/2305.05176) studies cascades; [RouteLLM (2024, revised 2025)](https://arxiv.org/html/2406.18665v4#S4) learns pre-generation routing from preferences. Their mechanisms are useful, but historical prices and savings do not establish today's deployment economics.

A wrong routing decision can be caught by output checks or fallback. Include those costs and delays in evaluation. Distillation can complement the serving strategies; it is not a fourth per-request allocation rule.

## 5. Four requests: spending on the wrong one buys nothing {#routing-example}

Everything below is **constructed**, not a model benchmark. Each A call costs 1 unit, each B call costs 5, and routing adds 0.1 per request. B intentionally fails the fourth task to illustrate complementary strengths.

<div class="worked-table" markdown="1">

| Request | A succeeds | B succeeds |
| --- | --- | --- |
| Extract a date | Yes | Yes |
| Rewrite a short passage | Yes | Yes |
| Find a cross-file bug | No | Yes |
| Follow an unusual local format | Yes | No |

</div>

<div class="worked-table" markdown="1">

| Plan | Completed | Total cost |
| --- | --- | --- |
| Always A | 3 / 4 | 4 |
| Always B | 3 / 4 | 20 |
| Useful: A → A → B → A | 4 / 4 | 8.4 |
| Mistaken: A → B → A → A | 3 / 4 | 8.4 |

</div>

Both allocations call B once and cost the same. The mistaken one completes no more tasks than always using A. The challenge is not merely reducing expensive calls; it is spending them where they help.

The useful allocation was chosen after inspecting both outcomes. It is an idealized comparison, not information available at serving time. A real router must use information available before the call; outcome labels or future feedback would create an undeployable result.

<details markdown="1">
<summary>Check the four plans in Python</summary>

```python
import math

outcomes = [(1, 1), (1, 1), (0, 1), (1, 0)]
costs = {"A": 1.0, "B": 5.0}

def score_plan(choices, overhead=0.0):
    if len(choices) != len(outcomes) or any(name not in costs for name in choices):
        raise ValueError("Choose A or B for each request")
    if not math.isfinite(overhead) or overhead < 0:
        raise ValueError("Overhead must be finite and nonnegative")
    completed = sum(result[0 if name == "A" else 1]
                    for result, name in zip(outcomes, choices))
    cost = sum(costs[name] + overhead for name in choices)
    return completed, round(cost, 6)

assert score_plan("AAAA") == (3, 4.0)
assert score_plan("BBBB") == (3, 20.0)
assert score_plan("AABA", 0.1) == (4, 8.4)
assert score_plan("ABAA", 0.1) == (3, 8.4)
```

This replays supplied outcomes; it neither trains nor calls a router. Real experiments need held-out tasks, uncertainty estimates, per-task results, refusals, timeouts, and cost. These four examples cannot estimate production gains.

</details>

<details markdown="1">
<summary>Explore cost and quality with sliders</summary>

This figure uses different assumptions, not measurements extending the four requests. It fixes success rates for two request types and makes the cascade's check more accurate than the pre-generation classifier by construction. Checking and routing overhead are omitted. Explore assumptions here, not vendor selection.

<!-- widget:tx-model-router -->

</details>

## 6. Count failed requests too {#how-to-count-the-cost}

Log every input, output, tool call, and retry. Include money spent on failed tasks: averaging only successful requests hides expensive attempts that never finished.

For a simple estimate, assume fixed prices: $C_A$ per A call, $C_B$ per B call, $r$ for routing, and $v$ for checking an answer. Let $u$ be the fraction sent or escalated to B:

$$
\begin{aligned}
C_{\text{route}}&=r+(1-u)C_A+uC_B,\\
C_{\text{cascade}}&=C_A+v+uC_B.
\end{aligned}
$$

The cascade has already paid for A. Its $u$ need not equal the router's. Real requests differ in length; accumulate actual input, output, and caching charges instead of assigning every call a fixed price.

<details markdown="1">
<summary>Estimate monthly API and self-hosting costs</summary>

Defaults are replaceable assumptions. This model combines prefill and decode throughput and assumes replicas run continuously. It does not simulate queues, peak provisioning, cache hits, or service-level requirements. It illustrates a cost curve, not a quote or capacity plan.

<!-- widget:tx-agent-cost -->

</details>

## 7. Waiting, context, and versions also matter {#three-things-cost-does-not-cover}

**Users wait for the whole task.** Independent concurrent calls should not simply be summed; dependent calls sit on the critical path. Record queues, time to first token, generation, tools, and end-to-end median and tail latency. API clients can still adjust context, concurrency, timeouts, and output length.

**Check what gets sent repeatedly.** If every call resends the full history, starting with $b$ tokens and adding $a$ per step, then $k$ calls send:

$$kb+\frac{ak(k-1)}{2}.$$

For $b=1000,a=200,k=4$, the inputs contain 1000, 1200, 1400, and 1600 tokens: 5200 total. Summarization, windows, retrieval, and caching change storage, computation, and billing. Not every agent has quadratic cost. Outputs and tool returns need their own accounting.

**Pin more than a model name.** Weights, tokenizer, chat template, tool schemas, sampling, and serving versions can all affect results. After an upgrade, rerun representative tasks rather than only checking that the endpoint returns text.

## 8. What self-hosting estimates often miss {#what-self-hosting-actually-costs}

- **Idle and peak capacity.** Machines provisioned for peaks may keep billing through quiet periods. High average utilization can also mean requests are waiting.
- **Prefill, decode, and KV cache.** Long inputs, long outputs, and many concurrent requests stress different resources. Test the actual length distribution.
- **Maintenance time.** Someone must own upgrades, rollback, monitoring, on-call work, and recovery.

PagedAttention primarily improves KV memory management; RadixAttention focuses on reusable prefixes. They are not interchangeable and neither removes every limit. See [one inference request](../06-systems/llm-serving.en.md) and [caching and attention kernels](../00-foundations/deep-dives/attention-kernels.en.md) for mechanisms and sources.

For deeper systems implementation, also see [Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial). This chapter establishes the questions; a cost figure alone is not a deployment design.

## 9. A practical order of checks {#an-order-of-questions}

1. **Define the task and boundaries.** Which data is allowed, which actions need approval, and what failures are acceptable?
2. **Establish a single-model baseline.** Keep independent evaluation tasks before adding routing.
3. **Find meaningful differences.** Which requests benefit from more computation, and which simply lack information?
4. **Compare complete systems.** Include the router, checks, retries, and tail latency alongside quality.
5. **Start small and reevaluate.** Traffic, versions, and prices can change which allocation makes sense.

One final trap: low token entropy or a large top-1/top-2 probability margin indicates a concentrated next-token distribution, not a correct completed task. Before using either to trigger escalation, check its relationship to failures on held-out data. Confidently misunderstanding the request is not necessarily fixed by reasoning longer.
