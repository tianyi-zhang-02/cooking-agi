# World models: learning inside imagination (Dreamer)

[中文](dreamer.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

Every time the robot arm pushes the cup, someone has to put the cup back; if it falls, it may break. Could we learn a model from logged interaction, let the policy practise thousands of times on trajectories the model "imagines", and only then go to the real robot? That is what the Dreamer line does.

It learns a compact latent dynamics model and trains an actor-critic only on imagined latent trajectories. This reduces real interaction, but the policy also inherits, and can even exploit, the model's errors.

## Dreamer spends its computation at training time

MPC in the [model-based RL note](../../05-post-training/deep-rl/model-based.en.md) searches an action sequence with the model at every decision. Dreamer takes the other route: the model "dreams" only during training, and at deployment actions are sampled straight from the actor. The DreamerV3 paper states that it acts without lookahead planning ([Hafner et al., DreamerV3](https://arxiv.org/abs/2301.04104), section "Critic learning"). It is the "Latent imagination" row in the table of the [planning note](../../05-post-training/deep-rl/planning-and-control.en.md).

Origins: [PlaNet (2018)](https://arxiv.org/abs/1811.04551) introduced latent dynamics with both deterministic and stochastic parts and planned online in latent space; [Dreamer (2019)](https://arxiv.org/abs/1912.01603) dropped online planning and trained an actor-critic by backpropagating learned value estimates through imagined trajectories.

## RSSM: a deterministic memory plus a stochastic guess

Dreamer's world model is the RSSM (recurrent state-space model). Following Eq. (1) of the DreamerV3 paper:

$$
h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1}),\qquad
z_t\sim q_\phi(z_t\mid h_t,x_t),\qquad
\hat z_t\sim p_\phi(\hat z_t\mid h_t)
$$

Reward, continuation flag and reconstructed image are all predicted from the model state $(h_t,z_t)$: $\hat r_t,\hat c_t,\hat x_t\sim p_\phi(\cdot\mid h_t,z_t)$.

| Symbol | What it is | In the cup example |
| --- | --- | --- |
| $x_t$ | Observation at step $t$ | One camera frame |
| $a_{t-1}$ | Previous action | "Push forward 1 cm" |
| $h_t$ | Deterministic recurrent state, updated by $f_\phi$ (an RNN) | Remembers how many pushes so far and roughly where the cup is |
| $z_t$ | Stochastic latent sampled from the posterior $q_\phi$, which sees the current observation | After seeing this frame, a judgement of whether the cup is wobbling |
| $\hat z_t$ | Sampled from the prior $p_\phi$, without the observation | A guess with eyes closed: what the cup will probably do after the push |
| $\hat r_t,\hat c_t$ | Predicted reward, and whether the episode continues ($c=0$ means it ends) | Reward when the cup reaches the target zone; the episode ends if it falls off |

The posterior/prior split is what makes "imagination" possible. During training there are real frames, so the posterior can be used; during imagination there is no observation, only the prior guessing step by step. So training must make the prior match the posterior as well as it can. Why have both $h$ and $z$? Intuitively, the deterministic part carries memory reliably, and the stochastic part expresses "the same action can lead to different outcomes".

## Training the world model: reconstruct, predict, pull together

The DreamerV3 world-model loss (Eq. 2–3), with $\phi$ denoting all world-model parameters:

$$
\mathcal L(\phi)=\mathbb E_{q_\phi}\Big[\sum_t\big(\beta_{\rm pred}\mathcal L_{\rm pred}+\beta_{\rm dyn}\mathcal L_{\rm dyn}+\beta_{\rm rep}\mathcal L_{\rm rep}\big)\Big],\qquad
\beta_{\rm pred}=1,\ \beta_{\rm dyn}=1,\ \beta_{\rm rep}=0.1
$$

$$
\begin{aligned}
\mathcal L_{\rm pred}&=-\ln p_\phi(x_t\mid z_t,h_t)-\ln p_\phi(r_t\mid z_t,h_t)-\ln p_\phi(c_t\mid z_t,h_t)\\
\mathcal L_{\rm dyn}&=\max\big(1,\ \mathrm{KL}[\operatorname{sg}(q_\phi(z_t\mid h_t,x_t))\,\|\,p_\phi(z_t\mid h_t)]\big)\\
\mathcal L_{\rm rep}&=\max\big(1,\ \mathrm{KL}[q_\phi(z_t\mid h_t,x_t)\,\|\,\operatorname{sg}(p_\phi(z_t\mid h_t))]\big)
\end{aligned}
$$

- $\mathcal L_{\rm pred}$: reconstruct the observation, predict reward and continuation. Reconstruction keeps the input's information in $z_t$.
- $\mathcal L_{\rm dyn}$: $\operatorname{sg}$ is stop-gradient. The posterior is a fixed target and only the prior is trained to match it.
- $\mathcal L_{\rm rep}$: the other way round; the prior is fixed and the encoder is pushed toward representations that are easier to predict.
- $\max(1,\cdot)$ is called free bits: once the KL is below 1 nat, the term becomes a constant with no gradient, and the model spends its effort on the prediction loss.

Teaching numbers show the division of labour. If the KL at some step is 0.6 nat, both terms are clipped to the constant 1 and produce no gradient. At 2.3 nat the two KLs have the same value and differ only in where gradients flow: the dynamics term contributes $1\times2.3=2.3$, the representation term only $0.1\times2.3=0.23$, so "the prior chases the posterior" carries 10 times the weight of the reverse direction. The asymmetry comes from [DreamerV2](https://arxiv.org/abs/2010.02193)'s KL balancing: on the same KL, weight 0.8 for the prior direction and 0.2 for the posterior (§2.1), so that representations are not dragged toward a poorly trained prior.

## Training the actor and critic in imagination

The actor and critic see only model states $s_t=\{h_t,z_t\}$. Real sequences from the replay buffer are encoded into start states; then the actor picks actions, the prior predicts the next state, the heads predict reward and continuation, and the rollout runs for $T$ steps. The DreamerV3 text gives $T=16$ (the hyperparameter table lists 15), discount $\gamma=0.997$ and $\lambda=0.95$.

<!-- paper-figure:dreamerv3-training -->

Imagination lasts only a dozen or so steps; the critic covers what comes after. The λ-return mixes the two:

$$
R^\lambda_t=r_{t+1}+\gamma\,c_{t+1}\big[(1-\lambda)\,v_{t+1}+\lambda R^\lambda_{t+1}\big],\qquad R^\lambda_T=v_T
$$

- $r_{t+1},c_{t+1}$: the reward and continuation flag predicted on arriving at the next imagined state $s_{t+1}$; $c=0$ cuts off every later return.
- $v_{t+1}$: the critic's value estimate for $s_{t+1}$.
- $\lambda$: the closer to 1, the more the imagined reward sequence is trusted; the closer to 0, the sooner the critic takes over.

Eq. (5) of the paper writes the subscripts of $r,c,v$ as $t$; here they are written $t+1$ to stress that they all belong to "the next step". Check with $\lambda=0$: the target becomes $r_{t+1}+\gamma c_{t+1}v_{t+1}$, the familiar one-step TD target.

The critic fits this target by maximum likelihood, $\mathcal L(\psi)=-\sum_t\ln p_\psi(R^\lambda_t\mid s_t)$, with a categorical output over exponentially spaced bins. The actor loss (Eq. 6–7):

$$
\mathcal L(\theta)=-\sum_t\Big[\operatorname{sg}\Big(\frac{R^\lambda_t-v_\psi(s_t)}{\max(1,S)}\Big)\log\pi_\theta(a_t\mid s_t)+\eta\,\mathrm H[\pi_\theta(\cdot\mid s_t)]\Big],\qquad
S=\operatorname{EMA}\big(\operatorname{Per}(R^\lambda,95)-\operatorname{Per}(R^\lambda,5),\,0.99\big)
$$

The first term is a REINFORCE-style policy gradient: a positive advantage raises the probability of that action. DreamerV3 uses this estimator for both discrete and continuous actions; Dreamer v1 instead backpropagated value gradients through the imagined trajectory. $S$ is the range between the 5th and 95th return percentiles; it only scales large returns down, never returns below 1 up. $\eta=3\times10^{-4}$ is a fixed entropy coefficient.

## By hand: the λ-return of a 3-step imagined rollout

To keep the arithmetic easy, use $\gamma=0.9$ and $\lambda=0.5$; these are teaching values, not the paper's. The start state $s_0$ is a replayed real state: the cup is 6 cm from the edge and the arm is touching it. The actor pushes 1 cm per step and the model imagines 3 steps:

| Imagined state | Predicted reward $r$ | Continue $c$ | Critic value $v$ | $R^\lambda$ |
| --- | --- | --- | --- | --- |
| $s_0$ (start) | — | — | 0.6 | 0.695 |
| $s_1$ | 0 | 1 | 0.5 | 1.044 |
| $s_2$ | 0 | 1 | 0.6 | 1.72 |
| $s_3$ (reaches target zone) | 1 | 1 | 0.8 | 0.8 |

Work backwards:

- $R^\lambda_3=v_3=0.8$;
- $R^\lambda_2=r_3+\gamma c_3[(1-\lambda)v_3+\lambda R^\lambda_3]=1+0.9\times0.8=1.72$;
- $R^\lambda_1=0+0.9\times(0.5\times0.6+0.5\times1.72)=0.9\times1.16=1.044$;
- $R^\lambda_0=0+0.9\times(0.5\times0.5+0.5\times1.044)=0.9\times0.772\approx0.695$.

The advantage at the start is $0.695-0.6=0.095$, which is positive, so the actor slightly raises the probability of "push 1 cm". Here $S<1$, so the denominator is 1.

<!-- widget:step-lambda-return -->

Change one condition to see $\lambda$: with $\lambda=0$, $R^\lambda_0=r_1+\gamma v_1=0.45$, trusting the critic entirely; with $\lambda=1$ it becomes $0.9^2\times1+0.9^3\times0.8\approx1.393$, trusting the imagined rewards entirely and using $v_3$ only once at the end. $\lambda=0.5$ lies in between.

Change another to see a model error. Suppose the model wrongly predicts that the cup falls at step 2, giving $r_2=-1$ and $c_2=0$. Then $R^\lambda_1=-1$ and $R^\lambda_0=0.9\times(0.5\times0.5+0.5\times(-1))=-0.225$, so the advantage becomes $-0.825$: the actor learns to avoid an action that is actually safe. Conversely, if the model misses a real fall, the actor learns to push. **The actor sees only the numbers the model gives it, never the real world.**

## What DreamerV2 and V3 changed

| Version | Main changes | Results reported by the authors |
| --- | --- | --- |
| [DreamerV2](https://arxiv.org/abs/2010.02193) (2020) | Stochastic latents switch from Gaussians to 32 categorical variables with 32 classes each; KL balancing | The first agent to reach human-level performance on 55 Atari tasks by learning behaviours entirely inside a separately trained world model; 200M environment steps in under 10 days on a single V100 |
| [DreamerV3](https://arxiv.org/abs/2301.04104) (2023; [Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)) | One fixed set of hyperparameters; symlog transform, KL balance plus free bits, 1% uniform mixed into every categorical, percentile return normalisation, symexp twohot output heads | One configuration across 150+ tasks; in the abstract's words, "the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula", with every Minecraft run finding diamonds within 100M environment steps |

Most of these changes serve one goal, running without tuning: reward scales can differ by orders of magnitude, so targets are compressed with $\operatorname{symlog}(x)=\operatorname{sign}(x)\ln(|x|+1)$; the 1% uniform mixture keeps categoricals from ever becoming deterministic, so the KL does not spike. The paper's compute comparison is VPT using 720 V100 GPUs for 9 days to get diamonds versus DreamerV3 using a single GPU for 9 days; but DreamerV3 uses the MineRL competition action space, which includes abstract crafting actions.

## Dreamer 4: a video world model and offline data only

Google DeepMind's [Dreamer 4](https://arxiv.org/abs/2509.24527) (Hafner, Yan, Lillicrap, 2025-09) keeps training in imagination but replaces the RSSM with a scalable video world model; Appendix F says the RSSM is light at inference but hard to scale to diverse data.

<!-- paper-figure:dreamer4-world-model -->

- **World model**: a causal tokenizer (400M) plus an interactive dynamics transformer (1.6B), about 2B in total (§4). The shortcut-forcing objective gives each frame its own noise level, also conditions the network on the step size, and predicts clean latents directly; inference uses 4 forward passes per frame (§3.2). The note on [video you can act in](interactive-video.en.md) covers this family of frame-by-frame diffusion recipes.
- **Data**: only VPT's contractor dataset, 2,541 hours of 360p, 20 FPS gameplay with keyboard and mouse actions.
- **Agent**: agent tokens carrying a task embedding are inserted, and other modalities cannot attend to them, so the model's predictions of the future are driven by actions rather than the task. Policy and reward heads are first learned by behaviour cloning, then RL runs on rollouts generated by the world model; at that stage the transformer is frozen and only the policy and value heads are updated (§3.3).
- **RL objective**: the value head still fits the λ-return above ($\gamma=0.997$); the policy head switches to PMPO, which uses only the sign of the advantage and not its size, so returns need no normalising, plus a KL toward the behaviour-cloning policy (§3.3).

The paper also reports (Table 1): 640×360, 9.6 s of context and 21 FPS on one H100; a human player completed 14 of 16 interaction tasks inside it (Oasis-large 5). In Figure 7, with 2,500 hours of video of which only 100 hours carry action labels, action-conditioning accuracy reaches over 80% of the fully labelled level.

## What 0.7% does and does not mean

The main result (§4.1, Figure 3) is evaluated in the real Minecraft: 60-minute episodes, random worlds, empty inventory, 1000 episodes. Milestones up to the stone pickaxe succeed over 90% of the time, the iron pickaxe 29%, diamonds 0.7%, which is about 7 episodes out of 1000. For reference, the paper says experienced human players need 20 minutes and about 24,000 keyboard and mouse actions on average to get a diamond; among the other offline baselines, VPT (finetuned) gets as far as sticks (53%), and a VLA built on Gemma 3 as far as the iron pickaxe (11%).

| What the number shows | What the number does not show |
| --- | --- |
| A policy can reach diamonds with no environment interaction during training; the paper calls this a first | That it gets diamonds reliably: more than 99% of episodes did not |
| Imagination training helps more the harder the milestone (relative to the same model's behaviour-cloning version) | That training used no human data: the data are human players' recordings and actions, and task rewards come from event annotations in the dataset |
| About 100× less data than VPT's offline agent (2.5K hours; VPT also used 270K hours of web video) | That it is directly comparable to DreamerV3's diamonds: V3 used about 1.4K hours of online interaction, 64×64 images plus inventory state, and abstract crafting actions |
| During evaluation a fixed sequence of task prompts guides the agent (Table 6) | That the agent planned the whole tech tree itself: the prompt order was provided by the authors |

The paper states that training used 256 to 1024 TPU-v5p chips but gives no training time or total compute, so "less data" cannot be converted directly into "cheaper".

## Side by side with MuZero and MPC

[MuZero](https://arxiv.org/abs/1911.08265) has 3 functions: representation $s^0=h_\theta(o_1,\dots,o_t)$, dynamics $r^k,s^k=g_\theta(s^{k-1},a^k)$, prediction $p^k,v^k=f_\theta(s^k)$. Its loss asks only for reward, value and the policy produced by MCTS, and the paper stresses that hidden states need not correspond to any environment state. In the cup example, MuZero's model need not know where the cup is, as long as it guesses well how much "push 5 cm" is worth.

| | Dreamer | MuZero | MPC (model-based note) |
| --- | --- | --- | --- |
| What the model predicts | Next latent state, reward, continuation; also reconstructs observations | Only reward, value and policy | Next state (often low-dimensional) and reward |
| Where computation goes | Learning the actor in imagination at training time; sampling directly at deployment | MCTS at every decision | Searching action sequences at every decision |
| Does the model state mean anything? | It can be decoded back into images for inspection | No environment semantics required | Usually it is the physical state itself |
| Representative result | DreamerV3: one hyperparameter set across 150+ tasks | MuZero Reanalyze: 731% median human-normalised score on 57 Atari games (200M frames) | That note's cart example: with a wrong model gain, replanning recovers part of the loss |

These approaches emphasise different training targets: Dreamer reconstructs its inputs, MuZero predicts only decision-relevant quantities, and JEPA in the [next note](jepa.en.md) predicts embeddings.

## What to check when you read a Dreamer-style result

| Check | Why it matters | Example from this note |
| --- | --- | --- |
| Real interaction or data volume | "Saves data" needs an explicit baseline | DreamerV3: 100M environment steps in Minecraft; Dreamer 4: 2,541 hours of offline recordings |
| Online or offline | Online training can correct the model with real feedback; offline cannot | V3 is online; Dreamer 4 trains fully offline and is evaluated in the real game |
| Imagination horizon | Longer means more reliance on the model and more accumulated error | V3 about 16 steps, then the critic takes over |
| Observation and action spaces | A different action space is a different task difficulty | V3: 64×64, inventory state, abstract crafting actions; Dreamer 4: 360×640, raw keyboard and mouse |
| What exactly is reported | Success rates need episode length, episode count and any guidance | Dreamer 4: 60-minute episodes, 1000 episodes, a task prompt sequence |
| Compute | Less interaction is not the same as cheaper | V3: a single GPU, 9 days; Dreamer 4: chip count given, duration not |

## Limits

- **Errors snowball.** The longer the imagination, the more it relies on the model; the compounding-error bound in the [model-based note](../../05-post-training/deep-rl/model-based.en.md) applies here too. Dreamer keeps imagination short and lets the critic cover the rest, at the price that distant returns depend entirely on the critic being accurate.
- **The policy exploits the model's flaws.** [Ha and Schmidhuber (2018)](https://arxiv.org/abs/1803.10122) found that their VizDoom agent learned an adversarial policy that stopped the monsters in the "dream" from firing; raising the sampling temperature $\tau$ to make the dream harder to predict made it harder to exploit (§4.5, Table 2): at $\tau=0.10$ the dream score was 2086 but the real score only 193; transfer was best at $\tau=1.15$, with a real score of 1092 ± 556.
- **Reconstruction spends capacity on irrelevant detail.** The DreamerV3 paper itself notes that complex 3D environments contain many details unnecessary for control and call for a stronger regulariser; its answer is free bits plus a small representation loss. For skipping pixel reconstruction altogether, see [the next note on JEPA](jepa.en.md).
- **Short memory.** Dreamer 4's context is only 9.6 seconds, and the paper says it is far from a full clone of the game: short memory and imprecise inventory predictions (§6).

## What to watch on this line next

Only results that would change the picture, not predictions:

1. **Can offline imagination training become reliable?** Google DeepMind reported a 0.7% diamond rate in September 2025 (1000 episodes of 60 minutes). A clearly higher rate from offline data alone, or the recipe working outside Minecraft (the paper's motivation mentions robots), would make "learning only inside the model" more than a first demonstration.
2. **Can it rely mainly on video without action labels?** Dreamer 4's Figure 7 shows that 100 hours of action labels plus 2,500 hours of video suffice to learn action conditioning, and the paper lists pretraining on general internet video as future work. An agent built mainly on unlabelled web video and then trained successfully in imagination would change the view of the data bottleneck.
3. **Can the memory gap be closed?** The paper's v1 has 9.6 seconds of context, and the authors themselves name short memory and imprecise inventory predictions. Only a result that remembers inventory and map over long spans, and raises late-milestone success because of it, would show this weakness is fixed.
4. **Can the compute accounts be lined up?** DreamerV3 reports a single GPU for 9 days, VPT 720 V100 GPUs for 9 days; Dreamer 4's v1 gives only 256–1024 TPU-v5p chips and no duration. Full compute figures are needed before "100× less data" can be read as "cheaper".

Read on: [Predict representations, not pixels (JEPA)](jepa.en.md).
