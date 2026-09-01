# The three stages of RLHF, and what happened next

[中文](rlhf-pipeline.md) · **English**

> Reading time: ~18 min · Level: core · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>turning "people prefer this answer" into something optimisable</strong></div>
  <div><span>Prerequisites</span><strong>SFT · preference data · next-token prediction</strong></div>
  <div><span>Core mechanism</span><strong>learn a reward model, then optimise it with RL</strong></div>
  <div><span>Common mistakes</span><strong>thinking all four models train; thinking the KL term is optional</strong></div>
</div>

## Quick learning: Actor, RM, Critic, and Reference

<details class="interview" markdown="1">
<summary>Explain Base → SFT → Preference → RL → Evaluation in two minutes</summary>

**Quick memory**: the Actor generates, the RM scores complete answers, the Critic estimates prefix return, and the Reference limits policy drift. Evaluation checks whether reward still matches the product objective.

**Interview answer**

> SFT first creates a usable policy; preference data trains a Reward Model; PPO updates the Actor with reward-derived advantages while clipping and reference KL constrain change. The Critic is a variance-reduction baseline, not the Reward Model. Current-policy sampling and independent evaluation close the loop.

<details markdown="1">
<summary><b>Deep dive</b>: why are Reference KL and PPO clipping different constraints?</summary>

Clipping limits one optimizer update relative to the rollout policy. Reference KL limits long-term drift from a fixed SFT policy. The former is a local trust region and the latter a behavioral prior; deleting either is not automatically compensated by the other.

</details>
</details>

## Why the detour is necessary

The direct approach to "answer better" would be a loss function. But *better* has no closed form and no reference answer to compare against.

People can **compare**, though: shown two answers, they can say which is better. All of RLHF is turning that into an objective:

```text
people can compare → learn a model that predicts which they'd prefer
                   → use that model as a reward function for RL
```

The cost is a layer of indirection. You are no longer optimising human preference; you are optimising *a model's fit to* human preference. Every difficulty below comes from that.

## Mapping ordinary RL onto a language model

Supervised learning usually supplies a target answer. Reinforcement learning (**RL**) instead evaluates the outcome of a sequence of behaviour. The model can learn the score of a rollout without being told which individual step caused it or what the correct replacement action was.

Language generation can be written directly as sequential decision-making:

| RL concept | In a language model |
| --- | --- |
| **Agent** | the language model being trained |
| **State** $s_t$ | the prompt plus generated prefix $(x, y_{<t})$ |
| **Action** $a_t$ | the next token $y_t$ |
| **Policy** $\pi_\theta(a_t\mid s_t)$ | the model's softmax next-token distribution |
| **Trajectory** $\tau$ | the token sequence from the start to the end of a response |
| **Reward** $r_t$ | scalar feedback from a Reward Model, verifier, or real environment |

The Actor is not a separate controller wrapped around the language model: **the language model itself is the policy**. At state $s_t=(x,y_{<t})$, its next-token probabilities define

$$a_t=y_t\sim\pi_\theta(\cdot\mid x,y_{<t}).$$

Appending that token creates the next state. In plain text generation this transition is almost deterministic concatenation; tool-using or interactive agents also receive search results, execution outputs, or other observations from the environment.

### Reward, return, and credit assignment

A **reward** $r_t$ is immediate feedback at one step. A **return** $G_t$ accumulates future rewards from that step onward:

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.$$

In classic preference-based RLHF, the main reward often arrives only after a complete answer is scored. Every earlier token must then share responsibility for the terminal score. Was a poor result caused by the opening direction or a factual mistake halfway through? That is the **credit-assignment problem**. Implementations often add per-token KL penalties as denser shaping rewards, but those penalties are not human preference themselves.

### Value, Q, and advantage

The Critic does not judge whether a completed answer is good. It predicts the cumulative return expected from the current prefix:

$$V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].$$

Conditioning additionally on the current action gives the action value:

$$Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].$$

Their difference is the **advantage**:

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).$$

It asks not whether the total score is high, but how much better this action was than the normal expectation at that state. The same return of $0.6$ is disappointing if the Critic predicted $0.8$ and encouraging if it predicted $0.2$. Subtracting this baseline leaves the expected policy gradient unchanged while greatly reducing its variance.

The central policy-gradient expression is therefore

$$\nabla_\theta J(\theta)\approx\mathbb E\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right].$$

When $\hat A_t>0$, increase the probability of the sampled action; when it is negative, decrease it. The expression directly updates sampled tokens, with shared parameters carrying the effect to other states. The Critic regresses $V_\phi(s_t)$ toward returns or bootstrapped targets. Its main job is **variance reduction**, not choosing the Actor's next token.

### A Reward Model is not a Critic

Both emit scalars, which makes them easy to confuse:

| | Reward Model | Critic / Value Model |
| --- | --- | --- |
| Input | prompt + completed answer | current prompt + generated prefix |
| Output | learned proxy preference score | expected return from the current state |
| Question answered | “How good does this completed answer look?” | “If the current policy continues from here, what return should it expect?” |
| During PPO | usually frozen | trained alongside the current Actor |

The Reward Model does not provide “true human satisfaction.” It supplies a **proxy reward** fitted on limited preference data, so it can be wrong, favour superficial styles, and be gamed. The Critic estimates a conditional expectation under the current policy; when the Actor changes, its target changes too.

At whole-response granularity this resembles a contextual bandit: receive a prompt, generate one answer, then receive one overall score. Inside generation it remains a sequential decision process whose state changes with every token. The two descriptions differ only in abstraction level.

## The three stages

**Stage 1, SFT.** Finetune the pretrained model on human demonstrations to get something that at least follows the instruction format. It becomes RL's initial policy, and RL cannot rescue a bad starting point.

**Stage 2, the reward model.** Collect **pairwise** rankings — two answers to the same prompt, labelled which is better — and train $r_\phi$ to score answers with the Bradley–Terry loss:

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

It learns **relative** order. Its zero point is unidentifiable: adding the same constant to every score changes no loss. Raw scores therefore are not literal units of “human satisfaction,” and comparisons across prompts or data distributions require calibration checks. The objective directly constrains the chosen–rejected gap for the same prompt.

**Stage 3, optimise with RL.** Four models are present at once, in quite different roles.

## The four models, and which two actually train

This is the part most explanations blur:

| Model | Comes from | Trains? | Role |
| --- | --- | --- | --- |
| **Actor** (policy) | copy of the SFT model | **yes** | the thing being optimised, and what ships |
| **Critic** (value) | often initialised from the reward model | **yes** | estimates $V_t$ to reduce gradient variance |
| **Reward** | stage 2's output | frozen | scores complete answers |
| **Reference** | copy of the SFT model | frozen | the KL anchor that keeps the Actor from drifting |

**Only the first two update weights.** Reward and Reference run forward passes and nothing else.

Actor and Reference start as two copies of the same weights. The Actor trains and drifts; the Reference stays put as the measuring stick.

These are four **conceptual roles**, not necessarily four independent full models that remain GPU-resident at all times. A Critic may be a value head on a shared backbone, while frozen models can be sharded or offloaded. Their training relationships stay the same.

## Why the Reference is mandatory

Because the reward model can be gamed. It is a function fitted on limited preference data, with no constraint whatsoever outside that distribution. Let the policy optimise freely and it finds answers the **reward model scores highly and people reject** — reward hacking.

So the objective actually being optimised carries a KL penalty:

$$r_{\text{total}}(x, y) = r_\phi(x, y) - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)$$

Go toward higher reward, but do not go far from where you started. $\beta$ is how tight that leash is.

Too large and the model cannot move; the output is indistinguishable from SFT. Too small and after a few hundred steps it emits things no human recognises but the reward model loves. **This is not an optional regulariser — it is the precondition for the method working at all.**

## Why a Critic as well

Policy gradients need to know how much better an action was than average — the advantage $A_t$. Using the raw return $R_t$ has enormous variance and training shakes itself apart. The Critic learns a baseline $V_t$ so that

$$A_t = R_t - V_t$$

PPO then smooths this over multiple steps with GAE and clips the update into a trust region:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\big)\Big], \qquad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

The clip exists to stop a single step going too far: once the policy leaves the old policy's support, the importance ratio $\rho_t$ explodes.

Two different rulers are involved. **PPO clipping** compares the current Actor with the old Actor that produced the rollout and limits one optimisation update. **Reference KL** compares the Actor with the frozen SFT Reference and limits cumulative drift across training. The former does not replace the latter.

## What happened next

Four models, two of them training, one of which is a full-size Critic. That cost is what everything since has been cutting.

![how many models each method keeps resident](assets/rlhf-model-count.svg)

**GRPO drops the Critic.** Sample a group of answers for the same prompt and use the group's own spread as the baseline:

$$\hat A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

If all you needed was a baseline for variance reduction, the group mean supplies one — no separate network required. That removes a full-size model *that was being trained*.

### The four GRPO/PPO clipping cases

GRPO changes where advantage comes from while commonly retaining a PPO-style clipped
surrogate. Define the token-level probability ratio between the current and rollout
policies:

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\text{old}}(a_t\mid s_t)}.
$$

The objective maximizes

$$
\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right).
$$

| Advantage | Ratio | Interpretation | Local gradient |
| --- | --- | --- | --- |
| $\hat A>0$ | $\rho\leq1+\epsilon$ | a good action is not over-promoted | keep increasing its probability |
| $\hat A>0$ | $\rho>1+\epsilon$ | a good action was promoted too far | clipped branch; gradient is zero |
| $\hat A<0$ | $\rho\geq1-\epsilon$ | a bad action is not over-suppressed | keep decreasing its probability |
| $\hat A<0$ | $\rho<1-\epsilon$ | a bad action was suppressed too far | clipped branch; gradient is zero |

Clipping does **not** hard-constrain $\rho$ to the interval at all times. It only stops
excessive movement in the direction recommended by the advantage. Positive advantage
clips only the upper side; if a good action became less likely, a ratio below the lower
bound still receives a corrective upward gradient. Negative advantage clips only the
lower side; if a bad action became more likely, a ratio above the upper bound still
receives a downward gradient.

Standard GRPO often applies one group-normalized sequence-level advantage to the
generated tokens of that response. Implementations differ in token aggregation, KL
placement, and clipping details; the table describes the classic local clipped-objective
behavior.

**DPO drops the RL loop entirely.** The key derivation: KL-constrained reward maximisation has a closed-form optimum, and inverting it expresses the reward in terms of the policy itself, so the preference loss can be taken **directly against the policy**:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

No reward model, no sampling, no Critic. Standard DPO usually uses **offline** preference pairs: the policy changes while the data distribution does not, so it cannot actively discover the current policy's new failures. Online DPO can resample and reduce this mismatch. The deeper distinction is whether data follows the current policy and whether feedback comes from preferences, a Reward Model, a verifier, or an environment.

**RLVR replaces the reward with a program.** Maths problems have answers to check; code has tests to run. Rewards like that need not be learned at all. The learned reward model disappears, and so does most of the room for reward hacking — **what gets gamed is a fitted reward, not a verified one.**

## Evaluation closes the loop

A lower training loss or higher average reward does not imply a more useful model.
Evaluate capability regression, factual grounding, safety and policy compliance, tool
use and task completion, multi-turn consistency, latency, and cost separately.
Open-ended responses can combine human review, calibrated LLM judges, and deterministic
checks; maths, code, and structured tasks should prefer verifiers where possible.

Keep a frozen regression suite before launch, then use shadow evaluation and controlled
A/B tests. Do not feed every production failure straight back into training. Deduplicate,
audit, and stratify it with a failure taxonomy, then decide whether it belongs as an SFT
demonstration, preference pair, verifier case, or system rule.

## Interview questions

<details class="interview" markdown="1">
<summary>How many models are in RLHF stage 3, and which train?</summary>

Four: Actor, Critic, Reward, Reference. **Only Actor and Critic update weights.** Reward is frozen after stage 2; Reference is a frozen copy of the SFT model. Actor and Reference start identical — the Actor drifts as it trains, and the Reference stays put as the KL anchor.

</details>

<details class="interview" markdown="1">
<summary>Why is the reference model and KL penalty needed? What if you remove it?</summary>

The reward model is a fit on limited data with no constraint off-distribution. Optimise freely and the policy finds outputs it scores highly and people reject — reward hacking. Typical symptoms: answers grow longer, pile up ingratiating phrasing, or collapse into a repetitive pattern.

The KL penalty tethers the policy near SFT. Too large a $\beta$ and nothing moves; too small and it drifts off. It is the method's central hyperparameter, not an optional regulariser.

</details>

<details class="interview" markdown="1">
<summary>What is the Critic for, and how does GRPO avoid it?</summary>

It estimates $V_t$ so the advantage $A_t = R_t - V_t$ has lower variance. Plain returns make policy-gradient variance too large to train through.

GRPO's observation: if a baseline is all you need, sampling a group of answers per prompt and normalising by the group's mean and standard deviation supplies one. That removes a full-size network *that was being trained* — real memory and real compute.

</details>

<details class="interview" markdown="1">
<summary>What is the essential difference between DPO and PPO?</summary>

DPO uses a derivation: KL-constrained reward maximisation has a closed-form optimum, so the reward can be rewritten in terms of the policy and the preference loss differentiated directly. The reward model and the RL loop both disappear.

Standard DPO's cost is **offline** preference data. The policy changes while the data does not, so it cannot actively explore its current failures. PPO and other online RL methods resample from the current policy and are more natural for exploration, environment interaction, or verifiable multi-step outcomes, but rollout is expensive and training less stable.

DPO therefore cannot replace PPO everywhere, but PPO is not universally better either. Use DPO when high-quality static preferences cover the task; use online methods when the current policy must keep producing new evidence. Online DPO variants reinforce that the real distinction is the data-and-feedback loop, not only the loss name.

</details>

<details class="interview" markdown="1">
<summary>Can reward-model scores be compared directly?</summary>

Bradley–Terry directly constrains the **gap** between chosen and rejected answers to the same prompt. Adding a constant to every score leaves the loss unchanged, so the zero point has no identifiable meaning; a score of $2.4$ is not “2.4 units of satisfaction.”

A fixed Reward Model's raw outputs can of course be used numerically, but comparisons across prompts, domains, or model versions require evidence that calibration and scale are stable. Implementations may also whiten or normalise rewards; that is an optimisation choice, not a rule implied by Bradley–Terry.

</details>

<details class="interview" markdown="1">
<summary>Why is there less reward hacking under RLVR?</summary>

Because the reward is no longer fitted. Checking a maths answer or running a test suite is a fixed program — there is no "outside the training distribution" to exploit.

The trade is coverage: it only applies where outcomes are automatically verifiable. Writing and open-ended dialogue have no checker and still need a learned reward model. And verifiers can be gamed too — code that passes the tests while being wrong.

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>Where each of the four models comes from, and which two update weights.</li>
    <li>What happens without the KL penalty, and the symptoms of $\beta$ being too large or too small.</li>
    <li>Whether the Critic reduces variance or improves accuracy, and what GRPO replaces it with.</li>
    <li>Why DPO needs no reward model, and what it gives up for that.</li>
    <li>For PPO-style clipping, which two combinations of advantage sign and ratio crossing make the local gradient zero?</li>
  </ol>
</div>

## Next

- [Post-training overview](README.en.md)
- [Data & feedback](../01-data-and-feedback/README.en.md) — the quality of preference labels themselves
- [Evaluation](../07-evaluation/README.en.md)

## Papers to start with

- [InstructGPT](https://arxiv.org/abs/2203.02155) — where the three stages come from
- [PPO](https://arxiv.org/abs/1707.06347)
- [DPO](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL and reward hacking

## Further reading (Chinese)

- [大模型中的强化学习 — RL in large models](https://zhuanlan.zhihu.com/p/693582342) — by @大家好我是爱因 on Zhihu (in Chinese).
  This chapter deliberately covers only the RLHF trunk. The algorithm taxonomy is in
  that piece: MDP elements, the Bellman equation, the bias-variance tradeoff across
  MC/TD/GAE, PPO's four-model setup, DPO with IPO/KTO, and what GRPO, DAPO,
  Dr. GRPO, RLOO, and REINFORCE++ each set out to fix.
