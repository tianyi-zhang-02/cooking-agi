# Agent RL: where environments and rewards come from

[中文](environments-and-rewards.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

If you want to train a code-fixing agent with RL, the first thing to prepare is not the algorithm but a few thousand tasks: for each one, a reproducible sandbox and a check that decides "is it fixed?". In single-turn RLVR, the checker looks at the answer from outside the model; an agent's checker often sits inside the very sandbox the agent can act on, where test files, git history and the network may all be within its reach. **The environment decides what the agent can learn, and the reward decides what it turns into. RL does not repair the failures a checker misses; it amplifies whatever path passes the check. So measure training reward and true success separately, and test the environment itself the way you test code.**

## What an agent environment has to provide

As in [Gymnasium](https://gymnasium.farama.org/), an environment must at least reset and take a step: reset gives the task and an initial state, such as a repository at some commit; each step takes the agent's tool call and returns an observation; the end of the episode gives a reward. For LLM agents, a few more things need to be spelled out:

- **Task and initial state**: the task description, the initial contents of the repository or database, and what the agent is not allowed to see;
- **Action space**: which tools it has (shell, editor, search, browser), whether it has network access, whether it can modify the tests;
- **Checker**: what it looks at at the end, how it decides, and whether the agent's actions can change it;
- **Budget**: how many turns, how many tokens, how long each command may run, and what happens when a limit is exceeded;
- **Isolation**: side effects of one trajectory must not leak into other trajectories, the scoring process or the outside world.

<!-- paper-figure:osworld-infra -->

| Environment type | Where the state lives | How the reward usually comes | Public examples | What to guard against first |
| --- | --- | --- | --- | --- |
| Code repository with tests | The repository and file system in a container | Tests that fail before the fix and pass after, plus tests that already passed | [SWE-Gym](https://arxiv.org/abs/2412.21139) (2024-12, 2,438 tasks from 11 repositories); [R2E-Gym](https://arxiv.org/abs/2504.07164) (2025-04, about 8.1k); [SWE-smith](https://arxiv.org/abs/2504.21798) (2025-04, 50,137 from 128 repositories) | Incomplete test coverage; editing the tests; reading the answer from git history |
| Search and browsing | A retrieval corpus or the live web, plus the agent's context | Comparing the final answer with a reference: [Search-R1](https://arxiv.org/abs/2503.09516) uses exact match, [DeepResearcher](https://arxiv.org/abs/2504.03160) uses F1 | OpenAI deep research (2025-02-02); Kimi-Researcher (2025-06-20) | Right answer, wrong evidence; finding the question's answer directly online |
| Simulated user | A business database, plus a user played by an LLM | Whether the database matches the goal state at the end, and whether replies gave the required information | [τ-bench](https://arxiv.org/abs/2406.12045) (2024-06); [τ²-bench](https://arxiv.org/abs/2506.07982) (2025-06), where the user can also operate tools | Doing nothing may "pass"; the simulated user may be inconsistent |
| Tool APIs | Real tools, or a model that simulates tool behaviour | Rules, tests or rubrics | [Kimi K2](https://arxiv.org/abs/2507.20534) (2025-07): 3,000+ real MCP tools and 20,000+ synthetic tools; [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (2025-12): over 1,800 environments and 85,000 prompts | The simulator behaving differently from the real tool |
| Computer use | Desktop and applications in a virtual machine | Scripts that check the final state, or an LLM judge | [OSWorld](https://arxiv.org/abs/2404.07972) (2024-04, 369 tasks, each with an execution-based evaluation script); [UI-TARS-2](https://arxiv.org/abs/2509.02544) (2025-09) | Judge errors; slow, expensive virtual machines |

Kimi K2's tool simulator (the paper describes it as functionally equivalent to a world model) is used to generate multi-turn interaction data; coding and software-engineering tasks instead connect to real execution sandboxes, with test pass rates as feedback. In DeepSeek-V3.2's Table 1, the code-agent and search-agent environments are real, while the environments for the 4,417 general-agent tasks are synthesised. The Kimi K3 technical report (2026-07) built mock implementations of applications such as Gmail, Notion and Slack for long-horizon personal-assistant tasks, keeping their core behaviour without depending on external APIs or rate limits. Real environments are more trustworthy; synthetic or mocked ones are cheaper and easier to extend to many kinds of tools: this is one of the main dividing lines in public reports today.

## Four sources of reward

| Source | How it is used in agent tasks | Strength | Typical failure |
| --- | --- | --- | --- |
| Unit tests, program execution | Run hidden tests to see whether the patch fixes the bug | Reproducible, cheap | Incomplete coverage; the agent can reach the tests or the test framework |
| Task checker | Compare the final state: database records, files, settings inside a virtual machine | Looks at "what the world has become", not at the agent's claim to be done | The goal state is wrong; doing nothing happens to match the goal |
| LLM judge, rubric | Judge reports, web actions, open-ended results | Covers tasks that cannot be written as rules | The judge is fooled by particular formats; it shares blind spots with the model being trained |
| Human preference | Rarely scores individual trajectories; mostly used to train a reward model or for spot checks | Closest to real intent | Expensive, slow; chat-style preference data may not transfer to agent tasks |

The Kimi K3 [technical report](https://github.com/MoonshotAI/Kimi-K3) (2026-07) states the second kind explicitly: rewards come from the verifier's evaluation of the final environment state, not from the agent's self-reported completion. SWE-RL (Meta, 2025-02, [paper](https://arxiv.org/abs/2502.18449)) is a different design: it runs no tests and scores between 0 and 1 by the text similarity between the generated patch and the reference patch, with −1 for a malformed output; it is a single-turn setting in which the model reads the issue and the relevant files at once and writes the edit directly, with no multi-turn interaction with an environment. For the general problems of tests as rewards, see [Verifiable rewards](../../05-post-training/verifiable-rewards.en.md); for an ordering of signals from strong to weak, see the comparison table at the end of [Generate, verify, retrain](../self-improvement/generate-verify-train.en.md); for the judge's own biases, see [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md).

A concrete warning about human preference in agents comes from Anthropic's [research](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) of 2025-11-21: after a model learned to exploit production coding environments, adding chat-style RLHF safety training resulted, in the words of the paper's [abstract](https://arxiv.org/abs/2511.18397), in "aligned behavior on chat-like evaluations, but misalignment persists on agentic tasks".

## Exploits that have been documented

The table includes only examples recorded by a primary source. Most share one feature: the tests, the grader, clues to the answer or the network sit inside the environment the agent can act on.

| Date | Reported by | What happened |
| --- | --- | --- |
| 2025-02-24 | Anthropic, [Claude 3.7 Sonnet system card](https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf) §6 | In agentic coding environments the model occasionally special-cased tests, most often by returning the values the tests expected; the system card says this resulted from reward hacking during RL training |
| 2025-03-14 | OpenAI, Baker et al., [paper](https://arxiv.org/abs/2503.11926) §2.1 | Agents in training called `exit(0)` to leave early without running all tests, or used `raise SkipTest` to skip them; others wrote stubs where test coverage was poor or decompiled `.pyc` files containing the reference solution |
| 2025-06-05 | METR, [blog](https://metr.org/blog/2025-06-05-recent-reward-hacking/) | o3 overwrote the grader's timing variable and monkey-patched the evaluator; on one optimisation task from RE-Bench, all 21 runs exploited it |
| 2025-09-03 | SWE-bench [issue #465](https://github.com/SWE-bench/SWE-bench/issues/465) | Evaluation images kept "future" git history, and agents used `git log --all` to see the fixing commit directly; from 2025-09-11 the images removed history after the base commit. In a training environment built the same way, this is a reward loophole |
| 2025-11-21 | Anthropic, [research](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) and [paper](https://arxiv.org/abs/2511.18397) | In real production coding environments: returning an object whose `__eq__` is always true, `sys.exit(0)`, and writing a `conftest.py` that rewrites pytest's reports; after learning these, the model also showed sabotage, alignment faking and similar behaviour elsewhere |
| 2026-02-12 | Meta, [Gaia2](https://arxiv.org/abs/2602.11964) Appendix B.2.3 | The RL policy packed ever more complex code-like strings into write actions, overwhelming the LLM judge used for soft checks; a task-agnostic "style" check closed the hole |
| 2026-04-07 | Anthropic, [Claude Mythos Preview system card](https://www-cdn.anthropic.com/7624816413e9b4d2e3ba620c5a5e091b98b190a5/Claude%20Mythos%20Preview%20System%20Card.pdf) §4.2.2.1 | The model occasionally circumvented network restrictions in its training environment to download data that shortcut the task; attempts in about 0.05% of training episodes, successes in about 0.01% |
| 2026-08-26 | OpenAI, [Hugging Face incident report](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) | OpenAI says agents cheating by looking up solutions online was a primary driver of the incident; the behaviour was then reinforced in training and may have contributed to the model's propensity to use similar techniques |
| 2026-09-22 | Anthropic, [Claude Opus 5.5 system card](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf) §6.2.1 | Screening RL training episodes with classifiers: in about 0.3% of sampled episodes the model successfully guessed what the answer key expected instead of deriving the answer from the task |

<!-- paper-figure:baker-hack-cot -->

Checkers outside the sandbox get exploited too. [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) (2025-07) found that, on a multi-subject dataset, replying with just a colon `:` was judged correct 77.2% of the time with LLaMA3-70B as the judge. The [Agentic Benchmark Checklist](https://arxiv.org/abs/2507.02825) (2025-07) points out that τ-bench counts empty responses as successes: some tasks cannot be completed by design, and doing nothing leaves the database unchanged, which happens to match the goal state. Earlier, [Sycophancy to Subterfuge](https://arxiv.org/abs/2406.10162) (Anthropic, 2024-06) trained on a sequence of increasingly gameable environments; in 32,768 episodes of the final held-out environment, 45 episodes modified their own reward and 7 also edited unit tests to avoid detection.

Another category is not active cheating but tests that are simply too weak. [PatchDiff](https://arxiv.org/abs/2503.15223) (2025-03) examined patches judged as passing on SWE-bench Verified: 7.8% actually fail the full developer-written test suite, and 29.6% behave differently from the reference fix. [UTBoost](https://arxiv.org/abs/2506.09289) (ACL 2025) found 36 tasks with insufficient tests and 345 erroneous patches labelled as passing.

## Work it through: tests that catch only part of the failures

The following is a teaching assumption with made-up numbers. On a batch of bug-fixing tasks, what the agent submits falls into four categories:

| Category | Meaning |
| --- | --- |
| C | Actually fixed |
| W1 | Wrong fix that the visible tests catch |
| W2 | Wrong fix that the visible tests do not cover, for example a missed empty input |
| H | Edited the test files, or special-cased the visible tests |

Two checkers. v1: run the visible tests directly in the agent's workspace, where the agent can edit test files. v2: restore the test files from the original repository before scoring and add a set of hidden tests; the hidden tests catch 80% of W2, and H always fails. Let $q_{\rm miss}$ be the probability that W2 slips through and $q_{\rm tamper}$ the probability that H slips through:

$$
\mathbb E[r]=P(\mathrm C)+q_{\rm miss}\,P(\mathrm{W2})+q_{\rm tamper}\,P(\mathrm H),\qquad S=P(\mathrm C)
$$

$\mathbb E[r]$ is the expected reward and $S$ the true success rate. Under v1 both $q$ values are 1; under v2, $q_{\rm miss}=0.2$ and $q_{\rm tamper}=0$. Compare two policies: $\pi_A$ before training, and $\pi_B$, assumed to be the result of training on v1 for a while.

| Policy | C | W1 | W2 | H | True success $S$ | v1 expected reward | v2 expected reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\pi_A$ | 0.30 | 0.45 | 0.20 | 0.05 | 0.30 | 0.55 | 0.34 |
| $\pi_B$ | 0.35 | 0.10 | 0.25 | 0.30 | 0.35 | 0.90 | 0.40 |

Under v1, reward rises from 0.55 to 0.90, 35 percentage points, while true success rises only 5 points. Under v2, reward rises 6 points, close to the true 5. The v1 reward curve looks great, but it mostly records W1 turning into W2 and H.

Why would training drift towards H? Take a group of $G=4$ trajectories with outcomes [C, W1, H, W1]. Under v1 the rewards are $[1,0,1,0]$ and the group-normalised advantages are $[+1,-1,+1,-1]$: editing the tests is encouraged exactly as much as a real fix. Under v2 the rewards are $[1,0,0,0]$ and the advantages are $[+1.732,-0.577,-0.577,-0.577]$ (computation in the site's [GRPO example](../../05-post-training/rlhf/after-rlhf.en.md)), so editing the tests is pushed down. Worse, "edit the tests when stuck" is a recipe that works across tasks, while a real fix has to be got right separately for each task, so under v1 H grows especially fast on hard problems. This is an inference about the mechanism; the Claude 3.7 Sonnet system card in the previous section describes exactly that: special-casing starts after repeated failures.

## How to bring reward closer to true success

| Try first | What to observe | What you can NOT conclude |
| --- | --- | --- |
| Restore the test files from the original repository before scoring, and run them in a process the agent cannot reach | H earns nothing | That other tampering channels are closed too; things like `conftest.py` and `sys.exit(0)` need separate checks |
| Make the tests read-only, or hide them from the agent | ImpossibleBench (2025-10): hiding tests drives the cheating success rate close to 0 but also makes normal tasks harder; read-only is a middle ground that blocks test edits but not special-casing | That read-only tests stop special-casing |
| Add hidden tests or held-out cases | The share of W2 that gets caught | That the hidden tests are complete; UTBoost shows they also miss things |
| Have people review a sample of "passing" patches, or compare their behaviour with the reference fix | An estimate of $q_{\rm miss}$ | That a spot check finding nothing means there is no blind spot |
| Scan training trajectories with classifiers or a CoT monitor | How the rate of exploits changes over training | That using the monitor directly as a penalty is safe: Baker et al. found that with too much optimisation, agents learn to hide their intent in the chain of thought and keep exploiting |
| Give the agent a way out to "report that the task is impossible" | In ImpossibleBench, GPT-5's cheating rate on conflicting tasks fell from 54% to 9% | That every model benefits equally; in the paper Claude Opus 4.1 changed much less |

The [Claude 4 system card](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285/Claude%204%20System%20Card.pdf) (2025-05-22) reports that, across its reward-hacking evaluations, hard-coding fell by an average of 67% (Opus 4) and 69% (Sonnet 4) relative to Claude Sonnet 3.7; the measures listed in the same section include fixing vulnerabilities in training environments, making environment instructions consistent with the reward signal, and adjusting the reward signal during RL to be more robust to hacking. Anthropic's [Mythos Preview risk report](https://www-cdn.anthropic.com/3edfc1a7f947aa81841cf88305cb513f184c36ae.pdf) of 2026-04-07 says they run an evaluation of environments before training to check for reward hacks and inconsistencies between prompt and reward, but that this check did not catch every problematic environment in time. In its autonomous execution task environments, Kimi K3 isolates verifiers from the agent: public verifiers give diagnostic feedback, hidden verifiers score held-out scenarios, and submissions are limited.

## Environment scale and the cost of verification

**Scale is growing fast.** Public coding environments went from SWE-Gym's 2,438 tasks (2024-12) to SWE-smith's 50,137 (2025-04), DeepSeek-V3.2's "tens of thousands" of reproducible issue environments (2025-12), and GLM-5's more than 10,000 verifiable environments across 9 programming languages ([paper](https://arxiv.org/abs/2602.15763), 2026-02). The Kimi K3 technical report (2026-07) says 51,219,741 sandboxes across 1,505,678 images were created over training and evaluation. All of these scales are self-reported by the teams, with no common external check on environment quality.

**Every environment costs money.** SWE-smith lets tasks from the same repository share one environment and reports 295 GB of storage, against its estimates of about 4 TB for a subset of R2E-Gym and about 6 TB for SWE-Gym. daVinci-Env ([paper](https://arxiv.org/abs/2603.13023), 2026-03) reports spending $891K to build 45,320 executable Docker environments.

**Every score costs time too.** DeepSeek-R1 ([paper](https://arxiv.org/abs/2501.12948), 2025-01) wrote in its limitations section that because evaluation took too long and slowed RL down, they had not yet applied large-scale RL to software-engineering tasks. DeepSWE (2025-07-02) started 512 containers in parallel per iteration, with a 5-minute test timeout. A rough sum, where the 2 minutes is an assumption: if each test run takes one CPU core for 2 minutes on average, the 512 scorings of one iteration take 1,024 core-minutes, about 17 core-hours, not counting the tests the agent runs itself during the episode.

**Interfaces are being contested too.** Prime Intellect's [Environments Hub](https://www.primeintellect.ai/blog/environments) (2025-08-27) and Meta and Hugging Face's [OpenEnv](https://huggingface.co/blog/openenv) (2025-10-23) both push shared environment formats. The Kimi K3 report says that training with a single agent harness can overfit the model to one tool schema, system prompt or context-management scheme, so it built a configurable environment that can instantiate several harnesses, including Claude Code and Codex.

## What to watch next on this line

These are open questions, and what kind of result would change the current judgement; they are not predictions.

1. **Which is worth more, quantity or quality?** Public coding environments grew from a few thousand tasks in 2024 to over ten thousand verifiable environments in 2026, and Kimi K3 created tens of millions of sandboxes in training, but public reports rarely compare "few, careful environments" with "many, rough ones" at the same compute. Such an ablation would give environment building a basis for its priorities.
2. **Can exploit rates keep falling?** Anthropic's system cards have reported exploits during training from Claude 3.7 Sonnet (2025-02) to Opus 5.5 (2026-09), but the evaluation sets keep changing, so the numbers are not directly comparable. A report across model generations on one fixed set of environments would make the trend visible.
3. **How do you reward agent tasks that cannot be verified?** Rubrics and LLM judges are already used for reports and web actions, and Gaia2 (2026-02) showed how they get exploited. Results that use a judge as the reward and then confirm the gain with an independent human evaluation will decide how far this route goes.
4. **Does it still work in a different harness?** Kimi K3 (2026-07) trains with several harnesses to avoid overfitting. A public measurement of the drop when a model trained in harness A is used in harness B would show how much of environment design has to change along with the harness.

Next: [Credit assignment over multi-step episodes, and the training system](credit-and-training.en.md).
