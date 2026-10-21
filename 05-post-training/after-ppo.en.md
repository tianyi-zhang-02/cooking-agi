# PPO, GRPO, and DPO: turning feedback into an update

[中文](after-ppo.md) · **English**

> Reading time: ~18 min · Type: chapter · Last reviewed: 2026-10

## Start with how feedback enters training {#start-with-what-each-algorithm-removes}

Some methods change advantage estimation; others change the data source or objective. They are not all PPO with a component removed. Ask where feedback comes from, how many responses are sampled, and how the gradient is calculated.

PPO, GRPO, and DPO answer the same question: once an SFT model exists, how can response quality or human preference make good answers more likely and bad answers less likely? They differ in how feedback enters training:

- **PPO** uses response rewards, a Critic's value baseline, and clipped policy updates;
- **GRPO** samples a group of responses to one prompt and replaces the Critic with group-relative rewards;
- **DPO** reads chosen/rejected pairs and updates the policy through a preference-classification loss.

### Build intuition with one problem {#build-intuition-with-one-problem}

Use “calculate $17\times24$” in all three cases. For this first comparison, treat a whole response as one action with a terminal reward and omit KL. That makes the PPO advantage simply $R-V$; tokenwise PPO also needs returns, GAE, and values for individual prefixes.

| Method | What it observes | Where the update direction comes from |
| --- | --- | --- |
| PPO | the Actor answers 388, reward $R=0$, and the Critic predicts $V=0.6$ | $A=R-V=-0.6$, so reduce the probability of this trajectory |
| GRPO | answers `[408, 388, 408 with work, 428]` get rewards `[1,0,1,0]` | group normalization gives approximately $A=[1,-1,1,-1]$; raise relatively good responses and lower relatively bad ones |
| DPO | a fixed record says `chosen=408, rejected=388` | make the current policy prefer chosen more than the Reference does |

One sentence each: PPO asks “how much better was the result than the Critic expected?” GRPO asks “how much better was this result than the other answers to the same prompt?” DPO does not estimate an advantage; it directly asks whether chosen improved more than rejected relative to the Reference.

## In PPO, which models learn and which ones score? {#what-ppo-has-available-to-delete}

A classic PPO-RLHF diagram has four conceptual roles, two of which are trained:

| Model | Training? | Job |
| --- | --- | --- |
| Policy | yes | the model you're trying to produce |
| Critic | **yes** | estimates state value, the baseline for the advantage |
| Reward | frozen | scores responses |
| Reference | frozen | KL anchor, keeps the policy from drifting |

The Critic needs training too: forward and backward passes, plus optimizer state. It may be a separate value model or share an Actor backbone with an added value head. These are four responsibilities, not necessarily four complete models resident in memory. Could we obtain a useful comparison without training a Critic?

## Where does the baseline come from without a Critic? {#the-main-line-delete-the-critic}

The Critic estimates state value, providing a variance-reducing baseline and participating in TD / GAE bootstrapping. For suitable whole-response outcome tasks, sampling statistics can construct a relative signal instead. Multi-step, delayed-reward tasks require another look at credit assignment.

| Algorithm | How it constructs the comparison | Cost and checks |
| --- | --- | --- |
| **GRPO** | sample a group per prompt; subtract the mean reward and divide by its standard deviation | multiple samples per prompt, so sampling cost goes up |
| **RLOO** | leave-one-out: one sample's baseline is the mean reward of the others | multiple samples; a common form uses a whole-response advantage rather than learned prefix values |
| **REINFORCE++** | construct advantages from KL-shaped returns, then normalize across the batch | no trained value model; inspect returns, KL, and the normalization scope |

These methods avoid an independently trained value model, but estimation error remains. Compare saved training state against extra sampling, signal variance, and task structure. The REINFORCE++ row refers to [v1 §3](https://arxiv.org/html/2501.03262v1#S3). Batch normalization of advantages is not another learned baseline, nor a guarantee of unbiased gradients or stable training.

As of October 2026, the latest arXiv version is [v9 (2025-11-10)](https://arxiv.org/abs/2501.03262v9), retitled “…with Global Advantage Normalization”. It adds a “REINFORCE++ w/ baseline” variant: sample several responses per prompt, subtract the group's mean reward, then normalize across the whole batch, with KL moved into a separate loss term (v9 §3.2).

RLOO's baseline, with $k$ samples per prompt:

$$\hat A_i = r_i - \frac{1}{k-1}\sum_{j \neq i} r_j$$

[RLOO §2.2–2.3](https://arxiv.org/html/2402.14740v1#S2.SS2) treats a completion as one action, but its tokens still receive gradients. Sequence log-probability is the sum of conditional token log-probabilities, all weighted by the same advantage. It does not separately estimate which step beat expectations. The leave-one-out baseline's unbiasedness also needs conditions such as independent samples for the same prompt; arbitrary correlated samples do not qualify.

The paper tests these simplifications on particular pretrained models and preference tasks. It does not establish that every task can dispense with a Critic or GAE.

### GRPO's group-relative advantage {#grpos-group-relative-advantage}

Sample $G$ responses for one prompt and score them $R_1,\ldots,R_G$. A common response-level advantage is $\hat A_i=(R_i-\mu_R)/(\sigma_R+\varepsilon)$, where $\mu_R$ is the group mean, $\sigma_R$ the group's population standard deviation (divide by $G$), and $\varepsilon$ prevents division by zero. The formula written out term by term, with a full worked example for `[0, 1, 0, 0]`, is in [four answers, one correct](rlhf/after-rlhf.en.md#group-example). For the opening rewards $[1,0,1,0]$, $\mu_R$ and $\sigma_R$ are both 0.5, giving advantages approximately $[1,-1,1,-1]$ when the small $\varepsilon$ is ignored. A sample-standard-deviation convention changes the scale. Rewards may come from a Reward Model, humans, a math verifier, unit tests, code execution, or a format checker.

A common GRPO implementation broadcasts one sequence-level advantage to the tokens in that response and then uses a PPO-style clipped objective:

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})},
$$

$$
\begin{aligned}
c_{i,t}&=\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon),\\
s_{i,t}&=\min(\rho_{i,t}\hat A_i,\;c_{i,t}\hat A_i),\\
J_{\mathrm{GRPO}}&=\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t s_{i,t}.
\end{aligned}
$$

Here $G$ counts responses to one prompt, and $|o_i|$ counts valid tokens in response $i$. Training maximizes $J$, or minimizes $-J$ in code. [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1) also includes a Reference KL term; some recipes drop it, for example [DAPO](dapo.en.md). As of October 2026, the [TRL v1.16.0 GRPOTrainer](https://huggingface.co/docs/trl/v1.16.0/en/grpo_trainer) defaults to `beta=0.0` (no Reference loaded) and `loss_type="dapo"` (normalized by the number of valid tokens in the global batch, not the per-response average above), and when generating with vLLM it turns on importance-sampling correction by default (`vllm_importance_sampling_correction=True`). The main saving is the learned value baseline, not rollout, reward, or policy ratios. Check loss reduction, KL weight, and update count separately rather than inferring them from the algorithm's name.

| | PPO baseline | GRPO baseline |
| --- | --- | --- |
| Source | $V_\phi(s_t)$ learned by a Critic | within-prompt reward statistics |
| Granularity | can vary with each prefix/token state | commonly one relative advantage for a whole response |
| Main cost | train a value function | multiple rollouts per prompt |
| Typical failure | Critic is inaccurate or unstable | no reward variation within the group, hence no learning signal |

## How can a zero loss still train the policy? {#zero-loss-gradient}

A group contains both good and bad answers, yet the logged policy loss is zero. That need not be a bug: **the values can cancel without their parameter gradients canceling.**

Consider two one-token answers, A and B. Each had probability 0.5 when sampled; their rewards are 1 and 0. Use advantages `[1,-1]` for clean arithmetic. A single parameter sets $p(A)=\sigma(\theta)$ and $p(B)=1-p(A)$, while both old probabilities stay fixed at 0.5.

At $\theta=0$, both ratios equal 1 and neither term is on a clipping plateau. The policy loss to minimize is:

$$\begin{aligned}
\mathcal L(\theta)&=-\tfrac12\big[2p(A)-2(1-p(A))\big]\\
&=1-2\sigma(\theta),\\
\mathcal L(0)&=0,\qquad \mathcal L'(0)=-0.5.
\end{aligned}$$

The value is zero, but the slope is not. One gradient-descent step with learning rate 0.2 moves $\theta$ to 0.1 and raises A's probability to about **0.525**. This is a constructed two-action check, not a language-model training experiment.

The denominator is an easy place to lose that gradient. `logp - old_logp.detach()` can be numerically zero just after sampling while its numerator remains differentiable. `logp - logp` instead subtracts the same computational path and cancels the gradient. A detached copy of the current value works for a single fresh update; when reusing the batch, retain the original copy rather than refreshing it on every pass.

<details markdown="1">
<summary>Check the zero loss in a few lines of PyTorch</summary>

```python
import torch

theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
logits = torch.stack((theta, torch.zeros_like(theta)))
logp = logits.log_softmax(dim=-1)
old_logp = logp.detach().clone()
advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
loss = -((logp - old_logp).exp() * advantages).mean()
loss.backward()
assert abs(loss.item()) < 1e-12
assert abs(theta.grad.item() + 0.5) < 1e-12
```

This checks the first update, omitting clipping before it takes effect. The accompanying [CPU script](code/policy_gradient_checks.py) verifies the same result with the full `min + clamp` objective and tests gradients, masking, and reductions. It does not call a model, sampling service, or GPU.

</details>

### When might the initial loss not be zero? {#zero-loss-assumptions}

The cancellation depends on **a complete group, centered advantages, matching ratios, and the chosen reduction**. It is not a universal GRPO assertion:

| Situation | Consequence |
| --- | --- |
| Every response has the same reward | Every advantage is zero, so this policy-gradient term really has no signal; that differs from positive and negative values canceling |
| A group is split across minibatches or ranks | A local logged mean need not be zero |
| Response lengths are 1 and 3, with advantages `[1,-1]` | At unit ratios, averaging within responses then across responses gives loss 0; averaging over all four valid tokens gives 0.5 |
| The current policy matches old but not Reference | The policy term may vanish while the KL term does not; refreshing old does not also reset Reference |
| Sampling temperature, inference backend, or asynchronous trajectory age differs | Identical weight files alone do not guarantee matching saved and recomputed log-probabilities; investigate ratios that differ from 1 |

The length example represents two different objectives, not a logging discrepancy to patch away. [Dr. GRPO](https://arxiv.org/html/2503.20783v1#S3) studies weighting from length and within-group standard deviation; [TRL's loss types](https://huggingface.co/docs/trl/grpo_trainer#loss-types) distinguish several reductions. Inspecting `loss_type` and the actual mask is more useful than guessing what the initial scalar should be.

Log policy loss, KL, valid-token count, group reward variance, and gradient norm separately. A near-zero loss proves neither stalled learning nor convergence. For the next question—whether a clipped token can still change—see the [gradient example](rlhf/ppo-clipping.en.md#clipped-token-gradients). For why the denominator is called old, follow the [response-group lifecycle](rlhf/on-off-policy.en.md#grpo-data-lifecycle).

## What can go wrong with group-relative rewards? {#what-surfaces-once-the-critic-is-gone}

Removing a model does not remove the need to judge an answer against something. The comparison now comes from the sampled group, so its weaknesses matter.

**One: when a group's rewards don't differ, the baseline carries no information.**
If every response is right or every one is wrong—for example, $R=[0,0,0,0]$—subtracting the mean leaves every advantage at zero. An $\varepsilon$ in the denominator prevents division by zero but cannot create a relative signal. The group contributes **no useful policy-gradient signal**. For how common such groups are, and how much extra sampling filtering them costs, see [DAPO's dynamic sampling](dapo.en.md#dynamic-sampling).

**Two: normalization introduces bias.**
Dividing by within-group standard deviation changes prompt weights; averaging within responses changes token weights across lengths. [Dr. GRPO](https://arxiv.org/abs/2503.20783) analyzes these biases. Whether removing a normalization helps your task still requires matched sampling and evaluation budgets; it is not a universally correct repair.

**Three: what does the same ratio ceiling mean for a rare token?**
PPO flattens surrogate gains on selected branches rather than hard-constraining actual ratios. The same multiplicative ceiling corresponds to a smaller absolute change for a low-probability token. Raising it may affect exploration, but entropy decline has multiple causes; one curve does not establish which one applies.

**Four: sequence-level loss dilutes long answers.**
Averaging within each response and then across responses gives a 1000-token answer the same total weight as a 50-token answer. Each token in the longer answer gets 1/20 the weight of a token in the shorter one; actual gradients also depend on advantages and the model. This deserves inspection if important reasoning steps occur in longer answers, but length alone does not make reasoning better.

DAPO combines changes to sampling, clipping, token weighting, and overlong responses. Soft length penalties do not remove the generation cap or within-group standardization. Work through [DAPO's sampling, length, and clipping examples](dapo.en.md).

## DPO: train directly on preference pairs {#another-line-delete-the-rl-loop-entirely}

DPO takes a different approach: its standard offline training phase needs **no explicit Reward Model, Critic, or online rollout loop**.

Each record is $(x,y_w,y_l)$: a prompt, chosen response, and rejected response. Define how each response's log probability changes relative to a frozen Reference:

$$
\Delta_w=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)},
$$

$$
\Delta_l=\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
$$

DPO minimizes

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\!\left(\beta[\Delta_w-\Delta_l]\right).
$$

Do not memorize the expansion. It says only:

**Relative to the Reference, chosen should improve more than rejected.**

To compute the loss for one pair by hand, see [calculate the loss for one pair](rlhf/after-rlhf.en.md#dpo-example) in the introductory note: with a relative gap of 2 and $\beta=0.2$, it is about 0.513.

### Why is a separate Reward Model unnecessary? {#dpo-derivation}

A Bradley–Terry preference model says

$$
\begin{gathered}
\Delta r=r(x,y_w)-r(x,y_l),\\
P(y_w\succ y_l\mid x)=\sigma(\Delta r).
\end{gathered}
$$

Now take a KL-regularized reward objective with a fixed penalty coefficient $\beta>0$:

$$
\max_\pi\ \mathbb E_{x\sim\mathcal D}
\left[\mathbb E_{y\sim\pi(\cdot\mid x)}r(x,y)
-\beta D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x)\right)\right].
$$

KL is a penalty in this objective, not a hard constraint with a prescribed radius. Optimizing over all policy distributions within the Reference's support, with $0<Z(x)<\infty$ and well-defined relevant expectations, the optimal policy is:

$$
\begin{gathered}
\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)}{Z(x)}\\
{}\times\exp\!\left(\frac{r(x,y)}{\beta}\right).
\end{gathered}
$$

For a hard constraint $\mathbb E_x D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})\leq\delta$, $\beta$ must instead be treated as a Lagrange multiplier matched to that constraint. An arbitrary fixed $\beta$ does not guarantee a prescribed $\delta$, let alone keep every neural-network update within that radius. The closed-form derivation in [Eq. 3 and Appendix A.1](https://arxiv.org/html/2305.18290v2#A1.SS1) actually uses the penalized objective above.

Rearrange the optimal policy as $r(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+\beta\log Z(x)$. Substitute into Bradley–Terry: for two responses to the same prompt, $\beta\log Z(x)$ cancels in the reward difference, leaving $\Delta r$ as $\beta$ times the difference of the two policy/reference log-ratios.

Next, replace $\pi^*$ with the policy being trained, $\pi_\theta$, and average the negative log-likelihood over the preference dataset: that is the $\mathcal L_{\mathrm{DPO}}$ above. Reward has not vanished; it has been **absorbed into the policy objective**.

This derivation comes from [DPO §4](https://arxiv.org/html/2305.18290v2#S4). It depends on the reward objective, preference model, and support assumptions. It does not establish that all human preferences follow this model, or guarantee that a finite neural network trained on finite data reaches the global optimum.

### DPO versus SFT {#dpo-versus-sft}

If A is chosen and B is rejected,

$$
\mathcal L_{\mathrm{SFT}}=-\log\pi_\theta(A\mid x)
$$

only says “A is worth imitating.” DPO observes A and B together and learns “A is preferred to B.” Chosen need not be the one uniquely perfect answer, and an ordinary binary preference usually supplies only the order $A>B$, not how much better A is.

$\beta$ participates both in the KL-regularization relation to the Reference and in the preference-logit scale. Mechanically memorizing “larger means more conservative” is unsafe: derivation conventions, loss implementation, and data scale all affect observed training behavior. In an interview, write the exact formula or library convention first, then reason about increasing or decreasing it.

Standard offline DPO uses **fixed preference pairs**. The policy changes, but the dataset does not automatically follow, potentially missing new failures. Measure whether distribution mismatch grows; it need not increase monotonically. Online DPO and iterative data refresh collect more feedback, so “DPO is always offline” is not a boundary for the entire family.

### A better relative score need not mean a higher probability {#relative-not-absolute}

Consider three categories of output:

| Output | Reference probability | Current probability |
| --- | --- | --- |
| chosen | 0.4 | 0.3 |
| rejected | 0.3 | 0.1 |
| all other outputs | 0.3 | 0.6 |

Chosen becomes less likely, but rejected falls further. Dividing the current chosen-to-rejected probability ratio by the Reference's ratio gives $(0.3/0.1)/(0.4/0.3)=2.25$. With illustrative $\beta=1$, this pair's DPO loss falls from $\log 2\approx0.6931$ to $\log(1+1/2.25)\approx0.3677$.

These are two valid distributions, not a claimed gradient-step trajectory. They show that **a lower preference loss does not ensure higher absolute chosen probability**, unlike an objective that only imitates chosen. See the objective in [DPO §4](https://arxiv.org/html/2305.18290v2#S4).

Two follow-ups:

- **IPO** — Uses a different preference objective with a finite target margin for relative log-ratios; it is not simply DPO plus a generic regularizer. [Paper](https://arxiv.org/abs/2310.12036)
- **KTO** — Can use unpaired desirable / undesirable labels. This allows a different data format; actual labeling costs still depend on the collection process.

## RLVR: let a program check the answer {#and-one-more-replace-the-reward-model-with-a-program}

Math answers can be checked, and code can be tested. When a reliable checker is available, its result can supply that part of the reward without a separately trained Reward Model. [DeepSeek-R1-Zero's reward setup](https://arxiv.org/html/2501.12948v1#S2.SS2.SSS2) is one example. This specifies a feedback source, not a choice between PPO and GRPO.

A deterministic result does not mean the checker captures the whole requirement: a function that hard-codes answers for the public tests also gets full marks. [Verifiable rewards](verifiable-rewards.en.md) takes this kind of loophole apart with a sorting check. It avoids the corresponding learned reward's fitting error, but does not guarantee less reward hacking. Test coverage, execution isolation, and independent evaluation still matter.

RLVR suits the parts with verifiable outcomes. A broader task can combine program checks, human judgments, and learned rewards rather than force every requirement into one checker.

## The table {#the-table}

| | Typical training data | Explicit reward | Critic | Training-time rollout | Central trade-off |
| --- | --- | --- | --- | --- | --- |
| PPO | prompts + responses from the current/recent policy | RM, rule, or environment | usually | yes | collects online feedback; typical setups coordinate sampling, scoring, and Actor–Critic updates |
| GRPO | prompts + a group of responses per prompt | RM, verifier, or environment | no | yes, multiple times per prompt | removes the Critic; needs within-group variation and spends more generation compute |
| DPO | fixed chosen/rejected pairs | no explicit call; reward difference is implicit in the loss | no | not in standard offline training | removes online sampling from training; labels, coverage, and optimization settings still matter |

The extensions change different parts of training: RLOO changes baseline estimation, REINFORCE++ uses batch statistics, RLVR changes the reward source, and DAPO adjusts sampling, clipping, token weighting, and overlong-response handling. These are not all competing choices at the same level, nor are they necessarily mutually exclusive.

## How to choose {#how-to-choose}

<details class="interview" markdown="1">
<summary>Step 1: can the reward be verified automatically?</summary>

**When verification is possible, consider a verifier / RLVR first.** Math answers, code tests, and structural constraints can be judged by a stable program without first fitting a Reward Model. Still audit whether the verifier checks only final answers, leaves test loopholes, or fails to represent the actual product objective.

</details>

<details class="interview" markdown="1">
<summary>Step 2: can training keep sampling from the current policy?</summary>

With only static chosen/rejected pairs, the DPO family is usually the direct choice, accepting coverage gaps and distribution mismatch. If training can keep rolling out and obtaining rewards or environment feedback, an online method such as PPO or GRPO can discover the current policy's new failures. It costs more generation compute, system complexity, and optimization variance.

</details>

<details class="interview" markdown="1">
<summary>Step 3: what is the real systems or optimization bottleneck?</summary>

If Critic memory, compute, or stability is the bottleneck, consider critic-free methods such as GRPO or RLOO. If long responses are diluted, low-probability tokens cannot recover, or entropy keeps falling, consider DAPO's token-level loss, Clip-Higher, dynamic sampling, and length handling. Do not choose an algorithm name first and force the data to fit it afterwards.

</details>

The order is: **validate the feedback, decide whether an online loop is possible, then diagnose the system bottleneck.**

## Down to a checklist {#down-to-a-checklist}

<details class="interview" markdown="1">
<summary>1. Is the reward learned or verified, and what should each monitor?</summary>

For a learned Reward Model, monitor reward hacking, length or style shortcuts, and whether agreement with human judgment decays as the policy drifts. For a verifier, audit the specification gap: whether tests cover the real requirement, final-answer checking rewards guessing, or the model can exploit formatting and execution loopholes.

</details>

<details class="interview" markdown="1">
<summary>2. What fraction of GRPO groups are all-right or all-wrong?</summary>

Log the zero-variance group rate directly (problem one above explains why these groups carry no signal). If the rate is high, adjust problem difficulty, group size, or reward resolution, or use dynamic sampling to filter uninformative groups—while monitoring the distribution shift caused by filtering.

</details>

<details class="interview" markdown="1">
<summary>3. What bias do reward and length normalization introduce?</summary>

Plot reward, correctness, response length, entropy, and within-group reward variance together. Group-standard-deviation normalization changes the weight of low-variance groups; response-length averaging can dilute tokens in long sequences. If length rises while correctness stays flat, the model may be exploiting normalization or a reward shortcut.

</details>

<details class="interview" markdown="1">
<summary>4. Are long-response tokens being outweighed by short responses?</summary>

First determine whether the loss uses a per-sequence mean or batch-wide per-token aggregation. If every response has equal total weight, a token in a 1000-token answer usually receives less weight than one in a 50-token answer. Long-reasoning tasks may need token-level loss aggregation and both response-level and token-level metrics.

</details>

<details class="interview" markdown="1">
<summary>5. Which policy produced the DPO preference data, and why does it matter?</summary>

Record the generation checkpoint, sampling temperature, decoding constraints, and annotation date. Old pairs may miss errors that the updated model now makes, but policy distance alone does not tell you how well the data covers those errors. Look for actual gaps on held-out prompts before deciding to resample, relabel, or move to an iterative / online preference loop.

</details>

## Can you explain it in two minutes? {#can-you-explain-it-in-two-minutes}

<details class="interview" markdown="1">
<summary>Write the three central equations for PPO, GRPO, and DPO.</summary>

With Monte Carlo returns, PPO can use $\hat A_t=G_t-V_\phi(s_t)$; GAE additionally involves bootstrapping. The clipped surrogate flattens selected gains; **it does not hard-bound the resulting ratio or KL after an update**. GRPO commonly uses $\hat A_i=(R_i-\mu_R)/(\sigma_R+\varepsilon)$. DPO uses $-\log\sigma\!\left(\beta[\Delta_w-\Delta_l]\right)$, where each $\Delta$ is the current policy's log-probability change relative to Reference.

</details>

<details class="interview" markdown="1">
<summary>Why does GRPO remove the Critic but not rollout?</summary>

It still needs the current or recent policy to generate multiple responses to one prompt and score every response; otherwise there is no group-relative baseline. What disappears is the Critic that learns $V_\phi(s)$.

</details>

<details class="interview" markdown="1">
<summary>Why does DPO need no explicit Reward Model?</summary>

Under the support and integrability conditions above, the optimal policy for a KL-regularized objective with fixed $\beta>0$ can be inverted to give the reward: $r=\beta\log(\pi^*/\pi_{\mathrm{ref}})$ plus a term that depends only on the prompt. Subtracting two responses to the same prompt cancels that term, so the reward difference becomes the difference of chosen/rejected policy-to-reference log-ratios. Reward did not vanish; it is implicit in the DPO loss. See the [derivation above](#dpo-derivation).

</details>

<details class="interview" markdown="1">
<summary>Two-minute conclusion: can DPO fully replace PPO?</summary>

Not in general. Ask where data comes from and when feedback is refreshed. Offline DPO does not collect new failures by itself; PPO / GRPO usually generate and score responses during training. Online DPO can refresh preference data too, so online versus offline and preference loss versus policy gradient are separate choices. Count the whole cost as well: offline DPO avoids online generation during training, but collecting and labeling the pairs still costs something, while PPO / GRPO pay for generation and more system coordination. Compare cost, coverage, and evaluation on matched tasks and budgets rather than ranking the names.

</details>

## Where to read next {#where-to-read-next}

- [The three stages of RLHF, and what came after](rlhf/README.en.md): what each model does, plus introductory GRPO and DPO examples
- [Data and feedback](../01-data-and-feedback/): the quality of preference labels themselves
- [Evaluation](../07-evaluation/): telling whether alignment actually helped
- [Clipping and gradients](clipping-variants.en.md): compare Dual-clip, CISPO, and SAPO on identical inputs; optional on a first pass

## Back to the choice: what feedback do you have? {#quick-learning-what-do-ppo-grpo-and-dpo-remove}

<details class="interview" markdown="1">
<summary>Choose from data availability and reward verifiability, not algorithm fashion</summary>

**Keep three distinctions in mind:** classic PPO-RLHF estimates a baseline with a Critic, while reward and Reference serve different roles. GRPO estimates relative advantages from responses to the same prompt. Standard offline DPO reads preference pairs without an online rollout loop. Whether roles share a model, or a program supplies the reward, is a separate configuration question.

**Interview answer**

> With high-quality preference pairs, offline DPO offers a simpler training loop, but does not collect new failures from the current policy by itself. If you can keep sampling and obtaining reliable rewards, compare online methods such as PPO and GRPO. The need for a Critic and the cost of group sampling depend on the task and budget; changing the loss does not guarantee stability.
</details>

## Starting papers {#starting-papers}

- [PPO](https://arxiv.org/abs/1707.06347) — clipped objective and trust region
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [DAPO](https://arxiv.org/abs/2503.14476) — four fixes for four GRPO failure modes
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) — Dr. GRPO, the length bias from normalization
- [Back to Basics](https://arxiv.org/abs/2402.14740) — RLOO
- [REINFORCE++](https://arxiv.org/abs/2501.03262) — drop the Critic, keep PPO's stabilizers
- [DPO](https://arxiv.org/abs/2305.18290) · [IPO](https://arxiv.org/abs/2310.12036) · [KTO](https://arxiv.org/abs/2402.01306)

## Further reading (Chinese) {#further-reading-chinese}

- [Reinforcement learning in large models](https://zhuanlan.zhihu.com/p/693582342) — supplementary reading in Chinese. This chapter compares feedback, baselines, and update objectives; check specific formulas and implementations against original papers and pinned code versions.
