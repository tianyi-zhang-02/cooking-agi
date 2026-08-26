# The three stages of RLHF, and what happened next

[中文](rlhf-pipeline.md) · **English**

> Reading time: ~9 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>The problem</span><strong>turning "people prefer this answer" into something optimisable</strong></div>
  <div><span>Prerequisites</span><strong>SFT · preference data · policy gradients</strong></div>
  <div><span>Core mechanism</span><strong>learn a reward model, then optimise it with RL</strong></div>
  <div><span>Common mistakes</span><strong>thinking all four models train; thinking the KL term is optional</strong></div>
</div>

## Why the detour is necessary

The direct approach to "answer better" would be a loss function. But *better* has no closed form and no reference answer to compare against.

People can **compare**, though: shown two answers, they can say which is better. All of RLHF is turning that into an objective:

```text
people can compare → learn a model that predicts which they'd prefer
                   → use that model as a reward function for RL
```

The cost is a layer of indirection. You are no longer optimising human preference; you are optimising *a model's fit to* human preference. Every difficulty below comes from that.

## The three stages

**Stage 1, SFT.** Finetune the pretrained model on human demonstrations to get something that at least follows the instruction format. It becomes RL's initial policy, and RL cannot rescue a bad starting point.

**Stage 2, the reward model.** Collect **pairwise** rankings — two answers to the same prompt, labelled which is better — and train $r_\phi$ to score answers with the Bradley–Terry loss:

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

It only learns **relative** order. Adding a constant to every score changes nothing, so the absolute values are meaningless — which is why reward scores are not comparable across batches.

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

## What happened next

Four models, two of them training, one of which is a full-size Critic. That cost is what everything since has been cutting.

![how many models each method keeps resident](assets/rlhf-model-count.svg)

**GRPO drops the Critic.** Sample a group of answers for the same prompt and use the group's own spread as the baseline:

$$\hat A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

If all you needed was a baseline for variance reduction, the group mean supplies one — no separate network required. That removes a full-size model *that was being trained*.

**DPO drops the RL loop entirely.** The key derivation: KL-constrained reward maximisation has a closed-form optimum, and inverting it expresses the reward in terms of the policy itself, so the preference loss can be taken **directly against the policy**:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

No reward model, no sampling, no Critic. The cost is that it uses **offline** preference pairs: the policy changes during training while the data still comes from the old one, and that mismatch is DPO's main limitation.

**RLVR replaces the reward with a program.** Maths problems have answers to check; code has tests to run. Rewards like that need not be learned at all. The learned reward model disappears, and so does most of the room for reward hacking — **what gets gamed is a fitted reward, not a verified one.**

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

The cost is that DPO's preference pairs are **offline**. The policy moves during training while the data came from an older one — a distribution mismatch. PPO resamples with the current policy every round and is on-policy. So DPO is far cheaper and usually loses on tasks that need exploration.

</details>

<details class="interview" markdown="1">
<summary>Can reward-model scores be compared directly?</summary>

Within a batch yes, across batches no. Bradley–Terry constrains only the *difference*, so adding a constant to every score leaves the loss unchanged and absolute values carry no meaning.

In practice rewards get normalised within the batch; otherwise the effective strength of $\beta$ drifts from batch to batch.

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
