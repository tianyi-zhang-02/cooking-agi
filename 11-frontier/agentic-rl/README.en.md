# Training agents with RL: from single answers to multi-step tasks

[中文](README.md) · **English**

> Reading time: ~7 min · Last reviewed: 2026-10

When you train on maths problems with RLVR, one sample is a prompt, one answer and one score. Now you want to train an agent that fixes bugs: it reads files, edits code, runs tests, reads the error and edits again, and after dozens of turns it submits; hidden tests decide whether it gets 1 or 0. **The skeleton of the algorithm does not change: sample, score, update the policy by the advantage. What changes is what a sample looks like: it is an interaction run inside a sandbox, the reward usually arrives once at the end, the text returned by tools sits inside the sequence without being the model's actions, and one rollout can take minutes to tens of minutes. So the hard part moves away from "which loss" to three places: whether the environment and reward can be trusted, how one trajectory's success or failure is split across its dozens of actions, and how to afford the rollouts.**

## From single answers to multi-step tasks: what changes

| | Single-turn RLVR | Multi-turn agent RL | Where it is covered |
| --- | --- | --- | --- |
| One sample | A prompt plus one answer | A prompt plus several turns of (action, observation) | Note 2 |
| State | The prompt and the tokens generated so far | The context, plus the world outside it: files, databases, web pages | Note 1 |
| Observations | None | Tool outputs, test logs, web pages, user replies; none of them is sampled by the model | Note 2 |
| Reward | A checker scores the answer | Usually at the end of the episode, based on what the environment has become | Note 1 |
| Where the checker lives | Out of the model's reach | Often in the same sandbox as the agent | Note 1 |
| Length | One turn | Dozens to thousands of tool calls; the Kimi K3 technical report (2026-07) says a single rollout of its long-horizon personal-assistant tasks may involve thousands of tool calls and millions of context tokens | Note 2 |
| Cost of one rollout | One generation plus a cheap check | Starting containers, running commands, running tests; DeepSWE (2025-07) used a 20-minute generation timeout and a 5-minute test timeout | Notes 1 and 2 |
| How it ends | An end-of-sequence token | Submission, failure, budget exhausted, tool crash, timeout, each different | Note 2 |

<!-- paper-figure:taubench-overview -->

Why tool outputs carry no gradient follows directly from the probability of a trajectory. Let $h_k$ be the whole history before the $k$-th action, $a_k$ the $k$-th action (a span of tokens), and $o_k$ the observation the environment returns next:

$$
\log p_\theta(\tau)=\sum_{k=1}^{T}\log\pi_\theta(a_k\mid h_k)+\sum_{k=1}^{T}\log P_{\rm env}(o_k\mid h_k,a_k)
$$

The second sum is the environment's transition. It does not depend on the parameters $\theta$, so it vanishes under the gradient, and the policy gradient keeps only the first term:

$$
\nabla_\theta J(\theta)=\mathbb E_{\tau}\Big[R(\tau)\sum_{k=1}^{T}\nabla_\theta\log\pi_\theta(a_k\mid h_k)\Big]
$$

A number to make it concrete: a two-turn trajectory where the policy picks $a_1$ with probability 0.5, the tool returns $o_1$ with probability 0.9, the policy picks $a_2$ with probability 0.2, and $o_2$ is deterministic; the whole trajectory has probability $0.5\times0.9\times0.2\times1=0.09$. Changing $\theta$ can only change the 0.5 and the 0.2; the 0.9 belongs to the environment. This assumes the environment does not use the same weights: if subagents run on the model being trained, their outputs do depend on $\theta$; Kimi K2.5 (2026-02) freezes its subagents and treats their outputs as observations. Masking tool-output tokens out of the loss in an implementation does exactly this. Real training replaces $R(\tau)$ with a baseline-subtracted advantage and adds probability ratios and clipping; see [The LLM bridge](../../05-post-training/deep-rl/llm-bridge.en.md).

## Timeline

Only events that can be checked against a primary source. Every score a team reports is self-reported, under its own evaluation settings.

| Date | Event | Organisation | In one line |
| --- | --- | --- | --- |
| 2024-06 | [τ-bench](https://arxiv.org/abs/2406.12045) | Sierra | An LLM plays the user; scoring compares the database state at the end |
| 2025-01-22 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek | Its limitations section says evaluation took too long, so large-scale RL was not yet applied to software engineering |
| 2025-01-23 | [Computer-Using Agent](https://openai.com/index/computer-using-agent/) | OpenAI | Says CUA combines GPT-4o's vision with reasoning obtained through RL |
| 2025-02-02 | [deep research](https://openai.com/index/introducing-deep-research/) | OpenAI | Says it was trained with end-to-end RL on hard browsing and reasoning tasks |
| 2025-02-24 | [Claude 3.7 Sonnet system card](https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf) | Anthropic | Reward hacking during RL training led the model to special-case tests in agentic coding environments |
| 2025-02-25 | [SWE-RL](https://arxiv.org/abs/2502.18449) | Meta | Single-turn setting: reward is the similarity between the patch and the reference patch |
| 2025-03-12 | [Search-R1](https://arxiv.org/abs/2503.09516) | Jin et al. | Multi-turn retrieval RL; retrieved tokens are kept out of the loss |
| 2025-04-16 | [o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/) | OpenAI | Says RL trained the models to use tools and to judge when to use them |
| 2025-05-16 | [codex-1](https://openai.com/index/introducing-codex/) | OpenAI | Says it was trained with RL on real-world coding tasks in a variety of environments |
| 2025-06-20 | [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) | Moonshot AI | End-to-end agentic RL; HLE Pass@1 from 8.6% to 26.9% (self-reported) |
| 2025-07-02 | [DeepSWE](https://www.together.ai/blog/deepswe) | Agentica, Together AI | Qwen3-32B trained with RL only; 42.2% Pass@1 on SWE-bench Verified (self-reported) |
| 2025-07-22 | [Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/) | Qwen | Long-horizon Agent RL with 20,000 independent environments in parallel |
| 2025-11-21 | [Natural emergent misalignment](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) | Anthropic | Hacks learned in production coding environments generalise to other misaligned behaviour |
| 2025-12-02 | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | DeepSeek | Over 1,800 environments and 85,000 prompts; post-training compute above 10% of pre-training cost |
| 2026-02-17 | [GLM-5](https://arxiv.org/abs/2602.15763) | Z.ai | Over 10k verifiable code environments; asynchronous agent RL with generation decoupled from training |
| 2026-06 | [GLM-5.2](https://z.ai/blog/glm-5.2) | Z.ai | Moves long tasks from group-relative optimisation to critic-based PPO |
| 2026-07 | [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3) | Moonshot AI | Million-token agentic RL; 51.2 million sandboxes created over training and evaluation |
| 2026-08-18 | [Pacing model development](https://openai.com/index/pacing-model-development-cyber-capabilities/) | OpenAI | After the Hugging Face incident and evidence that a new model may meet its Critical cybersecurity threshold, paused RL training of models intended for deployment for two weeks |

What the labs publish differs a lot. OpenAI's product announcements state that agents are trained with RL but give no algorithmic or environment detail. Anthropic's system cards do not describe training algorithms, but they keep reporting hacking in training environments and the defences against it: from Claude 3.7 Sonnet's special-casing to Claude Opus 5.5 (2026-09-22) screening training episodes with classifiers. Google DeepMind's [Gemini 2.5 technical report](https://arxiv.org/abs/2507.06261) (2025-07) says its RL environments added tasks that need multi-step actions and tool use. Algorithms, environment counts and infrastructure details come mainly from teams releasing open-weight models (Moonshot AI, DeepSeek, Qwen, Z.ai) and from open projects such as Agentica.

## What is settled, what is still moving

Each cell rests on a dated public record; the "what to watch" column says which kind of result would change the judgement, not a prediction that it will happen.

| Fairly settled | Still moving | Signals to watch |
| --- | --- | --- |
| Outcome rewards with executable checks are enough to train an agent with RL: DeepSWE (2025-07) took Qwen3-32B to 42.2% Pass@1 on SWE-bench Verified with RL alone (self-reported); OpenAI says deep research (2025-02) and codex-1 (2025-05) were trained with RL | How to score tasks without a verifiable reward: Kimi K2 (2025-07) uses a rubric reward from the model's own critique; Gaia2 (2026-02) reports an RL policy fooling the LLM judge behind its soft checks | Agent RL results that use a judge as the reward and confirm the gain with an independent human evaluation |
| Tool outputs stay in the context but out of the loss: in Search-R1's ablation (2025-03), average exact match was 0.431 with the mask and 0.343 without | Granularity of credit assignment: episode-level group-relative (DeepSWE 2025-07, GLM-5 2026-02), grouping by state (GiGPO 2025-05), critic-based PPO (GLM-5.2 2026-06) | A public comparison on long coding tasks at the same rollout budget |
| Checkers get exploited, and learned exploits generalise: Claude 3.7 Sonnet system card (2025-02), OpenAI's Baker et al. (2025-03), Anthropic's production-environment experiments (2025-11) | How to defend: reviewing environments before training (Anthropic said in 2026-04 that this did not catch them all), screening with classifiers during training (Claude Opus 5.5 system card, 2026-09), hidden verifiers (Kimi K3, 2026-07) | Hacking rates reported across model generations on a fixed set of environments |
| Environments and scoring are the main cost: DeepSeek-R1 (2025-01) did not apply large-scale RL to software engineering because evaluation was too slow; DeepSWE ran 512 containers in parallel per iteration | Environment scale: coding tasks went from thousands (SWE-Gym, 2024-12) to over ten thousand verifiable environments (GLM-5, 2026-02), and Kimi K3 (2026-07) created tens of millions of sandboxes over training and evaluation; how much is real and how much synthetic | Ablations of "few and careful" against "many and rough"; the drop when a model trained in one agent harness is used in another |

## How to read this series

For background: how policy gradients, GRPO and critics apply to LLMs is in [The LLM bridge](../../05-post-training/deep-rl/llm-bridge.en.md); what tests miss when used as rewards is in [Verifiable rewards](../../05-post-training/verifiable-rewards.en.md); termination versus truncation is in [MC and TD](../../05-post-training/deep-rl/returns-and-td.en.md); correcting for stale data is in [Asynchronous RL](../../05-post-training/async-policy-learning.en.md); the basic roles in RLHF are in [RLHF](../../05-post-training/rlhf/README.en.md). The structure of an agent is in [Agents](../../10-agents/README.en.md), and when splitting into several agents pays off is in [One agent or several](../../10-agents/single-vs-multi-agent.en.md); training on self-generated data and why it stalls at the verifier is in [Generate, verify, retrain](../self-improvement/generate-verify-train.en.md).

1. [Where environments and rewards come from](environments-and-rewards.en.md): code sandboxes, browsing, simulated users, tool APIs and computer-use environments; four sources of reward; documented exploits; how far expected reward and true success drift apart when tests catch only some failures
2. [Credit assignment over multi-step episodes, and the training system](credit-and-training.en.md): group-relative advantages for four 3-turn trajectories; which tokens receive gradient; GiGPO, turn-level critics and per-turn rewards; long contexts, truncation and asynchronous rollouts
3. [Review questions](review.en.md): interview questions and a self-check
