# Agent RL: credit assignment over multi-step episodes, and the training system

[中文](credit-and-training.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

A bug-fixing agent runs for 3 turns: it reads a file, edits the code and runs the tests, then submits. The hidden tests all pass, so the reward is 1. Training immediately raises two questions: which tokens should that 1 be credited to? And should the thousand-odd tokens returned by tools also go into the loss? **The most common answer today treats the whole trajectory as one sample: run a group of trajectories on the same task, compute one episode-level advantage from the group's outcomes, and spread it over every token the model itself generated in that trajectory; tool outputs stay in the context but are masked out of the loss. Splitting success and failure more finely (comparing by shared states, training a turn-level critic, rewarding each turn) requires environment states that recur, or an extra model, or trustworthy intermediate signals.**

## What a trajectory looks like once it reaches the trainer

Concatenated, one multi-turn agent trajectory is a long token sequence: the task description, the model's first action $a_1$, the observation $o_1$ returned by a tool, the model's second action $a_2$, and so on until it submits or runs out of budget. Actions are sampled by the policy; observations come from the environment, and the model did not "choose" them. [Post-training infrastructure](../../05-post-training/post-training-infrastructure.en.md) already walks through masks and denominators with a 20-position example, so here we only write the multi-turn objective. Leaving out PPO's probability ratio, clipping and KL (see [PPO clipping](../../05-post-training/rlhf/ppo-clipping.en.md)), the loss for a group of $G$ trajectories is:

$$
\mathcal L(\theta)=-\frac{1}{\sum_{i,t} m_{i,t}}\sum_{i=1}^{G}\sum_{t} m_{i,t}\,\hat A_{i,t}\,\log\pi_\theta\big(y_{i,t}\mid y_{i,<t}\big)
$$

$y_{i,t}$ is the $t$-th token of trajectory $i$, and $y_{i,<t}$ is everything before it, tool outputs included; $m_{i,t}$ is 1 on tokens the model generated and 0 on the prompt, tool outputs and padding; $\hat A_{i,t}$ is the advantage assigned to that token. Here the denominator counts all valid action tokens; you can instead average within each trajectory and then across trajectories, and the two choices weight long trajectories differently, see [DAPO's two averages](../../05-post-training/dapo.en.md).

Search-R1 (Jin et al., 2025-03, [paper](https://arxiv.org/abs/2503.09516) §3.1) writes this mask as an indicator $I(y_t)$ so that the policy gradient is computed only over tokens the model generated. In verl's [Agent Loop documentation](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), `response_mask` is 1 for model-generated tokens and 0 for tool-response tokens. Masked tokens stay in the context, and later actions are generated while looking at them; the mask only decides where the gradient lands. Multi-agent systems draw the same line: when Kimi K2.5 ([paper](https://arxiv.org/abs/2602.02276), 2026-02) trains its parallel agents, it freezes the subagents, treats their outputs as environment observations and updates only the orchestrator with RL, to avoid ambiguous credit assignment and unstable training.

<!-- paper-figure:searchr1-rollout -->

## Work it through: 3 turns, 4 trajectories

Everything below is a teaching assumption. The task is to fix a bug in `parse_date` in `utils.py`. Each trajectory has at most 3 turns; after the submission in turn 3 the environment runs hidden tests: all pass gives 1, otherwise 0. We sample $G=4$ trajectories on the same task:

| Trajectory | Turn 1 | Turn 2 | Turn 3 | Reward $R_i$ |
| --- | --- | --- | --- | --- |
| τ1 | Read `utils.py` | Correct edit, run visible tests | Submit | 1 |
| τ2 | Read `utils.py` | Wrong edit that the visible tests miss | Submit | 0 |
| τ3 | Search for callers of `parse_date` | Correct edit, run visible tests | Submit | 1 |
| τ4 | Run the tests first | Correct edit, run visible tests | Submit | 1 |

Normalising by the group's population standard deviation (the same convention as the site's [GRPO example](../../05-post-training/rlhf/after-rlhf.en.md)): the mean is $\bar R=0.75$ and the standard deviation is $\sqrt{0.1875}\approx0.433$, so

$$
\hat A_i=\frac{R_i-\bar R}{\operatorname{std}(R)}\quad\Rightarrow\quad \hat A=[+0.577,\;-1.732,\;+0.577,\;+0.577]
$$

PyTorch's `torch.std` divides by $G-1$ by default ([docs](https://docs.pytorch.org/docs/stable/generated/torch.std.html)), which gives $[+0.5,-1.5,+0.5,+0.5]$: same proportions, different scale. DeepSWE (Agentica and Together AI, 2025-07-02, [blog](https://www.together.ai/blog/deepswe)) dropped the division by the standard deviation in its GRPO variant and used a leave-one-out baseline instead: each trajectory subtracts the mean of the other 3, giving $[+0.333,-1,+0.333,+0.333]$. All three conventions have the same signs and relative sizes, and each sums to 0. In the episode-level approach $\hat A_{i,t}=\hat A_i$: every action token in a trajectory gets the same number.

Expanding τ1 into tokens, with made-up token counts:

| Segment | Content | Tokens | Produced by | $m$ | Advantage per token |
| --- | --- | --- | --- | --- | --- |
| prompt | Task description, tool instructions | 400 | Environment | 0 | Not in the loss |
| $a_1$ | Call `read_file("utils.py")` | 30 | Model | 1 | +0.577 |
| $o_1$ | File contents | 900 | Tool | 0 | Not in the loss |
| $a_2$ | Some reasoning, one edit, a test call | 150 | Model | 1 | +0.577 |
| $o_2$ | Visible test log | 300 | Tool | 0 | Not in the loss |
| $a_3$ | Submit | 20 | Model | 1 | +0.577 |

Of 1,800 tokens, only 200 receive gradient, one ninth. If all 4 trajectories have 200 action tokens, the denominator in the formula above is 800; τ2's 200 action tokens are each multiplied by −1.732 and pushed down, and the other three are pushed up.

### What happens if tool outputs are not masked

τ1's 900 file-content tokens and 300 test-log tokens would also get +0.577: training would raise the probability of "the model writing this file content and this log itself", even though that text came from the environment and was not the model's decision. τ2's log would be multiplied by −1.732, which amounts to teaching the model "do not predict this real error message". In this example, tool outputs outnumber action tokens 6 to 1 (1,200 against 200), so most of the gradient would land on things unrelated to decisions. Search-R1's ablation (§5.4, Table 4, Qwen2.5-7B base, PPO): average exact match 0.431 with the mask and 0.343 without.

## Where one score per trajectory falls short

An episode-level advantage cannot tell which step in a trajectory was good. τ2 went wrong with its edit in turn 2, yet its file read in turn 1 also gets −1.732.

τ1 and τ2 both made that same file read in the same context, so their two gradients point the same way and add up: $0.577-1.732\approx-1.155$, a net push down. Statistically this is not wrong: in this group the two trajectories that read the file first succeeded 1 time in 2, below the group mean of 0.75. But with 4 samples the judgement is mostly noise; episode-level methods rely on many tasks and samples to average it out. Conversely, if all 4 trajectories had an identical first step, their mean-subtracted advantages would sum to exactly 0: a step that every trajectory takes is neither rewarded nor penalised, and only what differs between trajectories gets pulled apart.

### Grouping again by state: GiGPO

[GiGPO](https://arxiv.org/abs/2505.10978) (Feng et al., 2025-05, NeurIPS 2025) keeps the episode-level group-relative advantage $A^E$, then finds environment states that recur within the group (anchor states), puts the actions taken from the same state into a small group, and compares them by their subsequent discounted returns (§4.2–4.3, Eq. 5, 7, 8):

$$
R_t^{(i)}=\sum_{k=t}^{T}\gamma^{\,k-t}r_k^{(i)},\qquad
A(a_t^{(i)})=A^E(\tau_i)+\omega\,A^S(a_t^{(i)})
$$

$r_k^{(i)}$ is the reward of trajectory $i$ at step $k$, nonzero here only at the last step; $A^S$ is the within-group normalisation of $R_t$ inside one anchor-state group. The paper uses $\omega=1$ and $\gamma=0.95$ (Appendix E.1). In our example τ1 and τ2 are in the same state after reading the same file, so their two turn-2 edits form a two-element group: returns $[0.95\times1,\;0]$, normalised to $A^S=[+1,-1]$. The tokens of τ1's edit get $0.577+1=1.577$ in total, and τ2's get $-1.732-1=-2.732$. This comparison happens only between "two choices made from the same place", unaffected by τ3 and τ4 having taken other routes.

<!-- paper-figure:gigpo-grouping -->

The limits need working out too. All 4 trajectories start from the same initial state, so the turn-1 group is the whole group; the returns differ only by a common factor $\gamma^2$, so after normalising by the standard deviation $A^S$ equals $A^E$ exactly, which adds $A^E$ twice. The paper's limitations section says the method depends on state matching and that identical states can be hard to recognise in complex environments; when no state repeats, it degrades to GRPO. In the training details they merge two states whose longest-common-subsequence similarity exceeds 0.9 (§5.1). In a code repository, two trajectories may reach the same state less often than in a text environment like ALFWorld: this is a guess, and the paper did not test coding tasks.

### Other options: turn-level critics and per-turn rewards

| Approach | Where the signal comes from | What it buys | Cost and prerequisites | Examples |
| --- | --- | --- | --- | --- |
| Episode-level return | The final outcome of the whole trajectory; with group-relative baselines, compared with other trajectories on the same task | No critic to train, simple to implement | Group-relative needs several full trajectories per task; every action in a trajectory gets the same score | DeepSWE (2025-07-02) and GLM-5 (2026-02) use group-relative; Kimi-Researcher (2025-06-20) uses REINFORCE and multiplies correct trajectories by a γ-decay factor to encourage shorter paths |
| Grouping by state | States that recur within the group | No extra model, same rollouts | States must be matchable; degrades to GRPO when none repeat | GiGPO (2025-05) |
| Turn-level critic | A learned value estimated per turn | An estimate at every turn, and old data can be reused | One more model, with estimation error | [ArCHer](https://arxiv.org/abs/2402.19446) (2024-02): an utterance-level off-policy critic drives a token-level policy |
| Giving the critic more information at training time | The critic sees things the policy cannot, such as a reference solution | Better scores for intermediate steps | Only works where such information exists at training time | [SWEET-RL](https://arxiv.org/abs/2503.15478) (2025-03), 6 points above other multi-turn RL methods on ColBench |
| Per-turn rewards | Intermediate rewards from the environment or rules | Denser signal | The intermediate reward must be designed and can be exploited | [Wei, Zeng et al.](https://arxiv.org/abs/2505.11821) (2025-05): on search and game tasks, per-turn rewards beat terminal-only rewards under both GRPO and PPO |
| Critic-based PPO for long tasks | A value model estimates the return after a segment | Handles training segments of different lengths and counts | Cost and error of training the critic | GLM-5.2 ([official blog](https://z.ai/blog/glm-5.2), 2026-06) moved from group-relative to critic-based PPO that learns from individual rollouts, see [The LLM bridge](../../05-post-training/deep-rl/llm-bridge.en.md); SAO (2026-07), see [Asynchronous RL](../../05-post-training/async-policy-learning.en.md) |

How a multi-step critic propagates a terminal reward backwards with TD and GAE is worked through in a 3-step example in [The LLM bridge](../../05-post-training/deep-rl/llm-bridge.en.md#credit-example), and [MC and TD](../../05-post-training/deep-rl/returns-and-td.en.md) covers termination versus truncation, so neither is repeated here. RAGEN (Wang et al., 2025-04, [paper](https://arxiv.org/abs/2504.20073)) observed a recurring collapse in multi-turn RL that the authors call the Echo Trap: reward variance falls off a cliff and gradients spike; their stabilised variant StarPO-S adds three kinds of change: trajectory filtering, incorporating a critic, and gradient stabilisation.

## When episodes get long: context, truncation and token alignment

**The context keeps growing, and most of it is tool output.** In the example, 3 turns already make 1,800 tokens, 1,200 of them from tools. Those tokens are not in the loss, yet they decide how expensive each turn's prefill is and when the context limit is hit.

**How do you score a trajectory that hits a limit?** There are three common choices. Counting it as a failure (0) teaches the model to "hand in early" and systematically pushes down long tasks; dropping the whole trajectory from the loss loses some negative samples but stops penalising "not finished yet"; treating it as truncated and bootstrapping the remaining return with a value estimate needs a critic. DeepSWE ([blog](https://www.together.ai/blog/deepswe), 2025-07-02) took the second option and calls it compact filtering: trajectories that reach the maximum context, the maximum number of steps or the generation timeout are masked out of the loss entirely. The authors' reasoning is that giving reward only when the agent deliberately submits encourages it to test properly before submitting, and also delays reward collapse. GLM-5 ([paper](https://arxiv.org/abs/2602.15763) §4.1.2) excludes samples that failed because a sandbox crashed, so the model is not penalised for the environment's problems. The single-turn version of this issue is in [DAPO's handling of overlong samples](../../05-post-training/dapo.en.md), and termination versus truncation is in [MC and TD](../../05-post-training/deep-rl/returns-and-td.en.md).

**Compressing history changes the observation.** When the context no longer fits, the system summarises or drops old content, and later actions are made while looking at the compressed history. Who writes the summary matters: if the policy model writes it, the summary is its action and can be in the loss; if another module writes it, it is an observation. When a long task is cut into segments for training, a segment boundary is not the end of the task, see [the segmenting section of The LLM bridge](../../05-post-training/deep-rl/llm-bridge.en.md#segment-boundary).

**Training must use the tokens that were actually generated.** Many agent frameworks parse model output into messages and tool calls, then re-encode them with the chat template on the next turn. verl's Agent Loop documentation points out that re-encoding the parsed message does not necessarily reproduce the originally generated token ids; this hardly matters for serving but is critical for RL training, so it has the inference engine take and return token ids directly. GLM-5 (§4.1.2) added a token-in-token-out gateway for the same reason, recording the token ids each trajectory actually sampled: re-tokenisation can shift token boundaries slightly, and then actions no longer line up with rewards and advantages.

## The training system: when the environment is slower than the model

In single-turn RLVR, a rollout is mostly the GPU decoding; in agent RL, a large share of the time is spent off the GPU: starting containers, running commands, running tests, waiting for web pages.

| Stage | What it waits for | What public reports do | New risks |
| --- | --- | --- | --- |
| Generation | Decoding; the context grows every turn, so prefill gets more expensive | verl uses a sticky session to send a trajectory's later requests to the same inference server (the docs give no reason; reusing that server's prefix cache is the usual motivation) | Load imbalance |
| Environment | Starting containers, running commands, network | In its final run DeepSWE started 512 Docker containers in parallel per iteration (64 tasks, 8 each) and added Kubernetes scheduling to R2E-Gym for it; Qwen3-Coder reports running 20,000 independent environments in parallel ([blog](https://qwenlm.github.io/blog/qwen3-coder/), 2025-07-22) | Container crashes, timeouts, rate limits on external services |
| Scoring | Running tests or calling a judge | DeepSWE limits test runs to 5 minutes (the official SWE-bench evaluation allows 30), and a timeout scores 0 | Decide in advance how a timeout is scored |
| Training | Filling a batch | Fully decoupling generation from training: AReaL reports up to 2.77× faster than synchronous systems with the same number of GPUs ([paper](https://arxiv.org/abs/2505.24298), 2025-05, maths and code reasoning tasks) | Data generated by an older policy needs correcting |

ROLL Flash ([paper](https://arxiv.org/abs/2510.11345), 2025-10) reports that, at the same GPU budget, asynchronous training is up to 2.24× faster on RLVR tasks and up to 2.72× faster on agentic tasks. Kimi-Researcher ([blog](https://moonshotai.github.io/Kimi-Researcher/), 2025-06-20) uses fully asynchronous rollouts plus turn-level partial rollouts: tasks that exceed a time budget are saved to a replay buffer, and later iterations run their remaining turns with updated weights. It self-reports at least a 1.5× rollout speedup; this also means the early and late turns of one trajectory come from different versions of the policy. The Kimi K3 [technical report](https://github.com/MoonshotAI/Kimi-K3) (2026-07) says 51,219,741 sandboxes across 1,505,678 images were created over training and evaluation; its sandboxes support pause, fork and snapshot, and a forked copy can be used for scoring without disturbing the original environment. Asynchrony makes data stale; how to correct for that, and why the denominator must be the probability at generation time, is in [Asynchronous RL](../../05-post-training/async-policy-learning.en.md); which fields each trajectory should keep is in [Post-training infrastructure](../../05-post-training/post-training-infrastructure.en.md).

## How to check it when you build it

| Try first | What to observe | What you can NOT conclude |
| --- | --- | --- |
| Replay one trajectory and decode the tokens whose mask is 1 | Only model-generated content remains: no file contents, logs or template markers | That a correct mask means the tokens match what was generated; compare the token ids too |
| Count the groups that are all correct or all wrong | How many tasks carry no within-group signal at all | That many all-wrong groups mean a weak model; the environment may be broken or the tests may be timing out |
| Count limit hits, timeouts and tool errors separately | How large each share is and how it moves during training | That a falling truncation rate means better ability; the model may just have learned to submit early |
| Compare episode-level and finer approaches at the same number of rollouts | Success rate and variance on an independent evaluation | That whichever raises training reward faster is better |

## What to watch next on this line

These are open questions, and what kind of result would change the current judgement; they are not predictions.

1. **Is finer credit assignment worth it at scale?** GiGPO (2025-05) on ALFWorld, WebShop and search QA, and the per-turn reward study (2025-05) on search and game tasks, both report gains, while DeepSWE (2025-07) and GLM-5 (2026-02) use group-relative advantages on coding tasks. A public comparison on long coding tasks at the same rollout budget would let this section state the trade-off more concretely.
2. **Will critics come back into the mainstream?** Within the same team, GLM-5 (2026-02) still used group-relative agent RL, and GLM-5.2 (2026-06) switched to critic-based PPO for long tasks; SAO (2026-07) uses a value model for asynchronous single rollouts. If more teams publish similar choices for long tasks, with comparisons against group-relative methods, that would suggest that once tasks are long enough, "dropping the critic" stops paying off.
3. **How much staleness can be tolerated?** AReaL (2025-05) and ROLL Flash (2025-10) report speedups, but the longer the task, the more policy versions a trajectory spans from start to finish. Results that report the effect of corrections sliced by trajectory length would be more useful than a single speedup factor.
4. **How should compressed history and credit assignment fit together?** Who writes the summary, whether it counts as an action, and how segments are tied back to the outcome of the full task currently differ across public implementations; a shared practice with ablations is needed to tell which is more stable.

Next: [Review questions](review.en.md).
