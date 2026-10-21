# Agents: how they are used in different settings

[中文](scenarios.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-10

## The key difference: can the result be checked automatically?

Agents can write code or book flights, but these tasks differ in an important way: **can the result be checked automatically?** Where it can, an agent can make several attempts in a sandbox, with a person still reviewing the diff after the tests pass. Where it cannot, each step needs more human oversight.

| Setting | Environment and tools | Automatically verifiable? | Most common failure | Guardrails |
| --- | --- | --- | --- | --- |
| Coding | repository, shell, tests | yes: run the tests | tests pass but the logic is wrong; editing tests to make them pass | sandboxed execution; a person reviews the diff |
| Search and research | search, reading web pages | partly: sources can be checked | invented citations; not knowing when to stop | require sources; set a budget |
| Customer support | order and user systems, policy documents | partly: rules can be checked | breaking policy; answering the same question differently each time | limit tool permissions; hand hard cases to a person |
| Data analysis | SQL, notebooks | partly: queries run, but results still need checking | a wrong join or filter whose result still looks plausible | show the query; check row counts and definitions |
| Using a computer or browser | web pages, graphical interfaces | weakly | breaking when the interface changes; instructions injected by page content | confirm irreversible actions; isolated environment |
| Personal assistant | email, calendar, memory | weakly | acting on stale memory; leaking private data | confirm before sending; memory can be corrected |

### Work it out: retries help only when you can verify {#retry-math}

Suppose each attempt is independent and succeeds with probability $p=0.9$, and the agent runs $k$ times.

- **Automatically verifiable** (code with tests): run $k$ attempts and keep the first one that passes. One success is enough, so the success rate is $1-(1-p)^k$, usually written pass@k.
- **Not verifiable** (a support agent replying to a customer): the user sees every run, so all $k$ must succeed, with probability $p^k$. This is the pass^k used by [τ-bench](https://arxiv.org/abs/2406.12045).

| Runs $k$ | Verifiable: $1-(1-p)^k$ | Not verifiable: $p^k$ |
| --- | --- | --- |
| 1 | 0.900 | 0.900 |
| 3 | 0.999 | 0.729 |
| 8 | ≈1.000 | 0.430 |

The same agent with the same 90% single-run success almost always yields one correct result on the left after a few tries; on the right, getting every run right becomes harder and harder. That is why an agent can be allowed to retry freely where results can be checked, while unverifiable settings first need better single-run reliability, then permission limits and human confirmation.

The left column rests on two assumptions that often fail in practice:

- **The verifier must be reliable.** Wrong code that the tests miss is kept as "passing", so the attempt you keep may still be wrong.
- **Attempts must be independent.** If failures cluster on one kind of hard task, eight more runs with new random seeds can fail the same way eight times.

## Coding: the strongest verification signal

Code that runs and tests that pass make the ideal agent environment: every step gets objective feedback. SWE-bench evaluates such agents on real GitHub issues with their tests. Its most-cited Verified subset is close to saturated, and models may have seen its problems: in a 2026-02-23 post OpenAI said it no longer reports SWE-bench Verified and recommends the harder, broader [SWE-bench Pro](https://arxiv.org/abs/2509.16941) instead ([OpenAI's post](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)). Why benchmarks saturate and get contaminated is covered in [Test-time compute: reading the numbers](../11-frontier/test-time-compute/measuring.en.md). Note that "tests pass" does not mean "fixed correctly": what the tests do not cover still needs a person's eyes.

## Customer support: reliability beats one-off success

An agent that answers a customer's question correctly this time and wrongly next time is not acceptable in support. τ-bench simulates users and a tool environment with policy constraints. It measures reliability with pass^k: the probability that k independent runs of the same task all succeed, averaged over tasks. This is closer to the practical requirement than a single success: a task solved 90% of the time per run is solved in all 8 runs only $0.9^8\approx0.43$ of the time (see [the table above](#retry-math)).

Its successor [τ²-bench](https://arxiv.org/abs/2506.07982) adds a telecom support domain where the user must also operate the device (dual control), so the agent has to guide the user through the fix. As of 2026-10-21, the [repository](https://github.com/sierra-research/tau2-bench)'s [latest release](https://github.com/sierra-research/tau2-bench/releases) is τ-bench 2.0 from October 2026, adding versions of the same customer-service tasks in six languages. Before it, τ³-bench (v1.0) added full-duplex voice evaluation and the banking_knowledge domain. The July 2026 v1.0.1 fixed grading in that domain; the repository says its results from before and after that fix cannot be compared directly.

## Agents that read outside content: beware of injection

As soon as an agent reads uncontrolled content such as web pages, email, or documents, instructions hidden inside can hijack it (indirect prompt injection, [Greshake et al., 2023](https://arxiv.org/abs/2302.12173)). The basic rules: treat outside content as data, never as instructions; actions with side effects need permission boundaries and a person's confirmation.

## How to evaluate

- code: SWE-bench; consider SWE-bench Pro, which OpenAI recommended in its February 2026 post linked above;
- web tasks: WebArena;
- using a computer: OSWorld;
- dialogue and tools under policy constraints: τ-bench and its successors τ²-bench, τ³-bench and τ-bench 2.0.

Public benchmarks are only a start: before launch, build an evaluation set on your own tasks and look at complete run traces; see [Agent Observability](../06-systems/agent-observability.en.md) and [Evaluation](../07-evaluation/README.en.md).
