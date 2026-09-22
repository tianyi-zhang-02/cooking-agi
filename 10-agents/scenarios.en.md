# Agents: how they are used in different settings

[中文](scenarios.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-09

## The key difference: can the result be checked automatically?

Writing code and booking a flight are both called agents, yet they are entirely different jobs. The biggest difference is **whether you can tell automatically, once it is done, whether it was right**: where results can be verified, an agent can safely try several times; where they cannot, every step leans more on a person.

| Setting | Environment and tools | Automatically verifiable? | Most common failure | Guardrails |
| --- | --- | --- | --- | --- |
| Coding | repository, shell, tests | yes: run the tests | tests pass but the logic is wrong; editing tests to make them pass | sandboxed execution; a person reviews the diff |
| Search and research | search, reading web pages | partly: sources can be checked | invented citations; not knowing when to stop | require sources; set a budget |
| Customer support | order and user systems, policy documents | partly: rules can be checked | breaking policy; answering the same question differently each time | limit tool permissions; hand hard cases to a person |
| Data analysis | SQL, notebooks | yes: queries run | a wrong join or filter whose result still looks plausible | show the query; check row counts and definitions |
| Using a computer or browser | web pages, graphical interfaces | weakly | breaking when the interface changes; instructions injected by page content | confirm irreversible actions; isolated environment |
| Personal assistant | email, calendar, memory | weakly | acting on stale memory; leaking private data | confirm before sending; memory can be corrected |

## Coding: the strongest verification signal

Code that runs and tests that pass make the ideal agent environment: every step gets objective feedback. SWE-bench evaluates such agents on real GitHub issues with their tests. Note that "tests pass" does not mean "fixed correctly": what the tests do not cover still needs a person's eyes.

## Customer support: reliability beats one-off success

An agent that answers a customer's question correctly this time and wrongly next time is not acceptable in support. τ-bench simulates users and a tool environment with policy constraints, and measures pass^k, the share of tasks done right k times in a row, which is closer to the real requirement than a single success.

## Agents that read outside content: beware of injection

As soon as an agent reads uncontrolled content such as web pages, email, or documents, instructions hidden inside can hijack it (indirect prompt injection, Greshake et al., 2023). The basic rules: treat outside content as data, never as instructions; actions with side effects need permission boundaries and a person's confirmation.

## How to evaluate

- code: SWE-bench;
- web tasks: WebArena;
- using a computer: OSWorld;
- dialogue and tools under policy constraints: τ-bench.

Public benchmarks are only a start: before launch, build an evaluation set on your own tasks and look at complete run traces; see [Agent Observability](../06-systems/agent-observability.en.md) and [Evaluation](../07-evaluation/README.en.md).
