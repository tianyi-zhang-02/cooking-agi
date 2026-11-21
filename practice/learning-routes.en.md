# Systems learning routes: from models to infrastructure

[中文](learning-routes.md) · **English**

This route looks beyond the model: where data comes from, why serving slows down, how evaluation controls a release, and how resource limits change algorithm choices. Connecting these pieces makes the model's role in the system clearer.

These questions organize five areas. **Production ML and AI Infra are developed first**. The other three currently provide topic outlines and related reading; full routes will follow. Choose one route; the numbers are not a requirement to complete everything in order.

## Start with the question you have now {#start}

| Your situation | Start here | A reasonable stopping point for this pass |
| --- | --- | --- |
| You know training, but not what surrounds deployment | [Production ML](../14-production-ml-and-recs/README.en.md) | Explain one request, one data update, and one experiment |
| A service works until traffic arrives | [AI Infra](../15-ai-infrastructure-and-distributed-systems/README.en.md) | Trace queues, computation, cache, scaling, and failure handling |
| You have system experience and want to check assumptions | [Time consistency](../14-production-ml-and-recs/data-and-experiments.en.md#time) · [capacity and retries](../15-ai-infrastructure-and-distributed-systems/queues-and-failures.en.md) | Find a counterexample to test your design |
| You are preparing for a system-design discussion | [Recommendation request](../14-production-ml-and-recs/ranking-system.en.md) · [model service](../15-ai-infrastructure-and-distributed-systems/serving-system.en.md) | Change the latency budget or input length and explain the tradeoff |

These are reading goals, not daily obligations. Skip familiar derivations, follow prerequisite links where needed, and run the small exercises without a GPU or paid API.

## How the five areas connect {#map}

| Area | Main question | Available entry and intended scope |
| --- | --- | --- |
| **14 · Production ML and recommendation** | How do data, retrieval, ranking, and experiments improve outcomes? | [Route](../14-production-ml-and-recs/README.en.md), complete requests, time and sampling, experiment slices |
| **15 · AI infrastructure and distributed systems** | How do requests and training jobs run reliably with limited resources? | [Route](../15-ai-infrastructure-and-distributed-systems/README.en.md), serving stack, KV budgets, queues and failures |
| **16 · Optimization, control, and time series** | How should scheduling and repeated decisions handle uncertainty and constraints? | Full route pending; existing [planning and control](../05-post-training/deep-rl/planning-and-control.en.md), with forecasting uncertainty, linear/quadratic programming (LP/QP), and model predictive control (MPC) to follow |
| **17 · Data-center power and cooling** | How do electricity and heat constrain capacity and reliability? | Full route pending; the [technology timeline](../learn/technology-timeline.en.md) provides context, followed later by grid-to-rack power delivery, redundancy, liquid cooling, and power usage effectiveness (PUE) |
| **18 · Product, B2B, and technical collaboration** | How do technical changes connect to needs, delivery, and decisions across teams? | Full route pending; existing [system responsibilities](ml-system/README.en.md), with metric trees, order-to-cash, design docs, and roadmap cases to follow |

One example connects them: a heavier ranker improves recommendation relevance but makes requests queue. Serving needs more capacity or less computation per request. Resource and power limits may then require workload scheduling. Finally, someone must establish whether the quality gain justifies the cost and who will validate and deliver it.

This is not a one-way ladder. Data issues may require new platform interfaces, and resource constraints can change model design. Optimization and control are not prerequisites for every recommendation project; power and cooling need not wait until every model has been studied.

## Read without duplicating the existing material {#reading}

The new routes explain connections. Existing algorithms, frameworks, and projects stay at their current addresses.

1. **Start with the whole system.** Choose the [recommendation request](../14-production-ml-and-recs/ranking-system.en.md#system) or [model service](../15-ai-infrastructure-and-distributed-systems/serving-system.en.md#stack), then identify inputs, outputs, and feedback.
2. **Work through one calculation.** Where does candidate coverage disappear? What probability does negative sampling change? How much more KV does a long input need? Assumptions sit beside the arithmetic.
3. **Follow one component deeper.** Two-tower, FSDP, and vLLM details link to their existing chapters instead of maintaining several competing explanations.
4. **Change a condition.** Make data late, inputs longer, or a replica fail. Explain which conclusions survive and which need recalculation.

Papers and reproductions belong with the problems they address rather than in a detached paper list. References identify sources, teaching numbers state assumptions, and framework behavior needs checking against the deployed version.

## Contribute a concrete question or example {#contribute}

Examples, derivations, and small runnable experiments are welcome. A useful contribution need not be long: explain when something holds, where it fails, and how to check it.

The remaining three areas especially need review from readers with relevant experience. Start with a specific question and public references before deciding whether it needs a chapter; you do not need to write an entire route. See the [contribution guide](../community/README.en.md). Keep personal profiles, career plans, internal architectures, and nonpublic interview material out of contributions.
