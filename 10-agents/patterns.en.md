# Agents: common building blocks

[中文](patterns.md) · **English**

> Reading time: ~6 min · Level: advanced · Last reviewed: 2026-10

## The smallest unit: an augmented model

Start with a concrete task: “Find the latest cancellation policy and tell me whether I can cancel tomorrow morning's booking.” Is one search sufficient, or must the system inspect versions, check exceptions, and ask for the exact time? The distinction is who chooses the next step, not how many models are involved.

A model can use retrieval, tools, and memory without every agent requiring all three. Workflows generally follow paths specified in code; agents let the model select subsequent steps using execution feedback. This is a useful design distinction, not the only definition of every product. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

## Five workflows and one agent

| Structure | How it works | When to use it |
| --- | --- | --- |
| Prompt chaining | Split the task into fixed steps, each step's output is the next one's input, with checks in between | The steps are clear, and a little latency is worth the accuracy |
| Routing | Classify first, then hand off to a specialised path | Inputs differ a lot and are better handled separately |
| Parallelization | Run independent subtasks at once (sectioning), or do the same thing several times and vote (voting) | Subtasks do not depend on each other, or you want a steadier result |
| Orchestrator-workers | One model breaks the task down on the fly, hands pieces to workers, and combines the results | The subtasks cannot be listed in advance, such as which files a change must touch |
| Evaluator-optimizer | One model generates, another evaluates and gives feedback, in a loop | There are clear evaluation criteria, and feedback really improves the result |
| Agent | The model decides every step in a loop, judges progress from the environment's feedback, and stops when the stopping condition holds | Open-ended tasks where the number of steps and the path are unknown in advance |

These are not levels in a sophistication ranking, and they can be combined. A fixed workflow calling many models concurrently may cost more than an agent that searches once. Inspect the actual execution.

## Follow one execution before memorizing pattern names

Continue with the booking policy. This is an action record for developers, not a request for a lengthy disclosure of internal reasoning:

| Step | Proposed action | Application responsibility | Observation |
| --- | --- | --- | --- |
| 1 | Search cancellation rules | Validate search arguments and access | Two versions of a notice |
| 2 | Open the notice with the later effective date | Read authorized text and retain source positions | A 24-hour condition and a venue-closure exception |
| 3 | Ask for the exact booking time | Pause rather than invent missing information | Await the user's reply |
| 4 | Answer using the applicable rule | Check the source and time calculation | A conclusion or a remaining uncertainty |

[ReAct](https://arxiv.org/abs/2210.03629) interleaves reasoning, actions, and environmental observations. It does not require a separate “reasoning model” and “generation model.” A tool timeout is not evidence that no policy exists. Retries need limits, or the loop simply spends money repeatedly.

## What do function calling and MCP each do?

| Layer | Responsibility | Not a guarantee |
| --- | --- | --- |
| Model tool-call output | Propose a tool name and arguments | Producing arguments does not grant execution permission |
| Application executor | Validate arguments and access, enforce timeouts and retries, execute calls | Arbitrary tool-returned text is not a system instruction |
| MCP | Connect applications to external tools and resources through a shared protocol | It does not choose the plan, define authorization policy, or certify the answer |

An MCP host manages client connections to servers. This is an integration layer, not another reasoning algorithm. [MCP architecture](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)

Read-only search and booking cancellation also carry different risks. One retrieves information; the other changes external state. A cancellation tool should require confirmation and use an idempotency key so a retry after a timeout does not duplicate the operation. Remembering an action mentioned in history does not authorize executing it.

For a complete example, see [tools, MCP, and skills](tools-and-skills.en.md): follow a review-only request and separate task instructions, tool protocols, and execution permissions.

## Deep research adds more than report length

Change the question to “How have these policies changed, and which exceptions still apply?” One retrieval may not suffice. Track unanswered questions, search for missing evidence, reconcile conflicts, then write conclusions. A small evidence table helps:

| Question | Evidence available | Next step |
| --- | --- | --- |
| When did the 24-hour rule take effect? | A dated official notice | Check for subsequent revisions |
| Is venue closure an exception? | An explicit clause in the current policy | Retain its source location, not just a summary |
| Which version applies to this booking? | The booking date is missing | Ask the user instead of guessing |

Independent questions can run in parallel. Do not force dependent steps into multiple agents merely to increase their number. Five pages repeating one notice are not five independent sources.

Stop when key questions have support, or when the budget, source access, or missing user information prevents further progress. In the latter cases, state what remains unresolved instead of filling gaps with confident prose. [The research-workflow guide](deep-research.en.md) develops the evidence queue, citation checks, and stopping logic with runnable local examples. These are not results from a live multi-agent experiment.

## How to choose

- Try a single call with a good prompt and retrieval first; many problems stop here;
- if it can be a workflow, make it a workflow: each step can be tested and changed on its own;
- switch to an agent only when the path truly cannot be fixed in advance, and give it a clear stopping condition and a budget.

## A framework, or the API directly

Start with a small direct implementation to understand state and failure paths, then decide whether a framework's persistence, recovery, and tracing are useful. Frameworks can save substantial work; you should still be able to inspect inputs, tool arguments, and execution results.

Test more than the happy path: simulate a timeout, denied access, conflicting sources, exhausted step budget, and a write that succeeds but loses its response. Stopping or recovering correctly matters more than adding agent boxes to a diagram.
