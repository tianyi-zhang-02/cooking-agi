# Policy gradients: how does reward become a gradient?

[中文](policy-gradients.md) · **English**

> Reading time: ~9–12 min · Last reviewed: 2026-11

Q-learning estimates which action is valuable and then takes an argmax. Another route changes action probabilities directly: if an outcome is better than expected, make the sampled behavior more likely. But the environment may not be differentiable. Where does the gradient come from?

## Differentiate trajectory probability, not the environment

Start with a finite episode of length $T$ and an undiscounted objective $J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)]$. Assume environment transitions and the initial distribution do not depend on $\theta$:

$$
p_\theta(\tau)=\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

Under regularity conditions allowing differentiation under the integral, use $\nabla p=p\nabla\log p$:

$$
\nabla J
=\mathbb E_\tau[R(\tau)\nabla\log p_\theta(\tau)]
=\mathbb E_\tau\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)R(\tau)\right].
$$

No environment derivative is required. We change the probability of seeing such trajectories again, rather than backpropagating through the reward that already occurred. SAC later uses a different path when actions can be reparameterized and Q is differentiable.

## Why reward-to-go and a baseline are allowed

An action cannot change rewards already received, so replace the whole-episode return with reward-to-go $G_t=\sum_{k=t}^{T-1}r_k$. Then subtract a state-only baseline $b(s_t)$:

$$
\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)b(s)]
=b(s)\nabla\sum_a\pi(a\mid s)=0.
$$

The expected gradient is unchanged, while variance can decrease. $V^\pi(s)$ is a common baseline, not a guarantee of minimum variance under every gradient-weighted criterion. An arbitrary baseline depending on the sampled action invalidates this proof.

Here $b(s_t)$ must be fixed when averaging over the current action. A state-only network fitted to this sample's return can still depend on its action through the fitted parameters; detach removes graph dependence, not statistical dependence. In the bandit below, fitting a baseline to a single sampled reward makes its advantage zero, despite a nonzero true gradient. One safe ordering is to form advantages with the pre-rollout Critic, then train it; see [GAE §6.1](https://arxiv.org/pdf/1506.02438).

Keep discount conventions consistent. If the objective is $\mathbb E[\sum_t\gamma^t r_t]$ and $G_t$ restarts discounting at the current step, the finite-trajectory derivation needs an outer $\gamma^t$. Some literature absorbs this into the state-visitation distribution. Do not mix conventions.

## A two-action calculation

Rewards are 1 and 3, with initial action probabilities 0.5 each. The expected reward is 2. For the two softmax logits:

$$
\frac{\partial J}{\partial z_i}=p_i(r_i-\bar r),\qquad
\nabla_z J=[-0.5,\ 0.5].
$$

Gradient ascent lowers the first logit and raises the second. Reward 1 is positive, but it is below the mean of 2. Positive raw reward does not mean an action's probability should always increase.

## Why the mean return need not minimize variance {#variance-optimal-baseline}

Consider one gradient term at a fixed state $s$. Write its score vector as $z=\nabla_\theta\log\pi_\theta(a\mid s)$, its sampled return as $G$, and its estimator as $g_b=z(G-b)$. Fix the baseline before sampling the current action. Since $\mathbb E[z\mid s]=0$, changing $b$ leaves the mean unchanged. Minimizing the sum of coordinate variances therefore reduces to minimizing a second moment:

$$
\operatorname{tr}\operatorname{Cov}(g_b\mid s)
=\mathbb E[\|z\|^2(G-b)^2\mid s]
-\|\mathbb E[zG\mid s]\|^2.
$$

The second term does not depend on $b$. Differentiate the first term and set the derivative to zero. With finite second moments and $\mathbb E[\|z\|^2\mid s]>0$,

$$
\mathbb E[\|z\|^2(b-G)\mid s]=0
\quad\Longrightarrow\quad
b^*(s)=\frac{\mathbb E[\|z\|^2G\mid s]}{\mathbb E[\|z\|^2\mid s]}.
$$

Returns are weighted by **squared score magnitude**, not just action probability. This follows the variance-minimization approach discussed by [Peters and Schaal (2008), §3.2](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Publications/Neural-Netw-2008-21-682_4867%5B0%5D.pdf). Our criterion is specifically a shared scalar baseline for one gradient term at a fixed state. It does not automatically minimize whole-trajectory variance with cross-time covariances, or remain optimal under a different parameterization or optimizer.

**Change the bandit's action probabilities.** Let action $a=1$ pay 3 and action $a=0$ pay 1, with $p=\pi(a=1)=\sigma(\theta)=0.8$. For this scalar logit, $z=a-p$: the two scores are $-0.8$ and $0.2$. Mean return is 2.6, whereas

$$
b^*=\frac{0.2\times0.8^2\times1+0.8\times0.2^2\times3}
{0.2\times0.8^2+0.8\times0.2^2}=1.4.
$$

| Fixed baseline | $g_b$ when action 0 is sampled | $g_b$ when action 1 is sampled | $\mathbb E[g_b]$ | $\operatorname{Var}(g_b)$ |
| --- | --- | --- | --- | --- |
| None, $b=0$ | −0.8 | 0.6 | 0.32 | 0.3136 |
| Mean return, $b=2.6$ | 1.28 | 0.08 | 0.32 | 0.2304 |
| This example's $b^*=1.4$ | 0.32 | 0.32 | 0.32 | 0 |

Every mean equals the true gradient $p(1-p)(3-1)=0.32$. Zero variance relies on this one-step, deterministic-reward, single-parameter setting; it is not a general RL guarantee. The example separates ordinary value-prediction MSE from Actor-gradient variance. Estimating $b^*$ also costs data and introduces estimation noise. Using the current action's return to manufacture an “optimal” baseline would still violate the statistical-independence requirement discussed earlier.

## A minimal PyTorch update

This is the loss for one on-policy update, not a complete trainer. `log_prob` contains the current policy's log-probabilities of sampled actions; `advantage` is a fixed learning signal. Both are matching one-dimensional tensors. This example uses the undiscounted objective above and no padding. A finite-trajectory discounted objective also needs the outer $\gamma^t$ weights discussed earlier; variable-length trajectories require an explicit per-step or per-trajectory reduction convention.

```python
def actor_loss(log_prob, advantage):
    if log_prob.ndim != 1 or log_prob.shape != advantage.shape or log_prob.numel() == 0:
        raise ValueError("log_prob and advantage must be matching nonempty vectors")
    return -(log_prob * advantage.detach()).mean()
```

The minus sign turns gradient ascent into optimizer descent. Detaching means this update changes action probabilities without letting the Actor change its own scoring signal through the advantage graph. The Critic has a separate loss.

## Why old trajectories cannot simply be replayed

Samples from behavior policy $\mu$ do not follow the current distribution $\pi$. In principle, importance sampling can correct this:

$$
w(\tau)=\prod_t\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}.
$$

Target trajectories must have support under the behavior distribution. Long products can have enormous variance. A ratio for only the current action does not automatically correct shifted state visitation. PPO's short reuse window and clipping are pragmatic compromises, not exact correction for arbitrarily old data.

## Why can past rewards be removed?

Condition on the history $h_t$ before the action. Past reward $R_{<t}$ is now constant. Averaging over the next sampled action gives:

$$
\mathbb E[\nabla\log\pi(a_t\mid s_t)R_{<t}\mid h_t]
=R_{<t}\sum_a\pi(a\mid s_t)\nabla\log\pi(a\mid s_t)=0.
$$

Removing past rewards preserves the expected gradient while dropping randomness the current action cannot explain. This is credit assignment in concrete form: the final action should not receive credit for reward already collected at the first step.

Now add 100 to both rewards in the $[1,3]$ bandit. The best action stays unchanged, but raw returns can produce large positive sample updates for either action. Subtract the equally shifted expected return and advantages remain $[-1,1]$. Relative quality is preserved while removing an unhelpful scale contribution. This does not mean every reward transformation preserves the optimal policy.

## One rollout, several updates: when does the assumption change?

Data comes from $\pi_{\rm old}$. After the first update, parameters have changed. Treating the same samples as fresh current-policy data is no longer the original on-policy estimator.

| Operation | Required convention | Common mistake |
| --- | --- | --- |
| Collect trajectories | Save behavior-policy version, log-probabilities, and boundaries | Recompute old log-probabilities after updates, making ratios 1 |
| Build returns and advantages | Align rewards, states, and sampled actions | Include rewards from the next episode |
| Update Actor | Fix the scoring signal and use the right optimizer sign | Backpropagate through advantage to change the score |
| Reuse data | Specify ratios, approximation scope, and stopping rules | Treat extra epochs as ordinary supervised learning |

Begin debugging with a two-action bandit whose rewards are known. Fix seeds and data, compare the expected gradient direction with the hand calculation, then add a baseline, Critic, and multistep environment. A large environment’s return curve is a poor way to identify a reversed sign.

## Where this leads

The estimator above, which weights log-probabilities by sampled returns, is REINFORCE. It avoids learning an environment model and has a direct mechanism, but return variance and sample cost can be large. A Critic helps estimate the update direction. Continue with [Actor–Critic and GAE](actor-critic-gae.en.md).

Why does step size need more care than choosing a learning rate? Continue with [natural gradient and TRPO](trust-region.en.md), from a coin policy to a KL-constrained update.

References: [Policy-gradient derivation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) · [Original PPO paper](https://arxiv.org/abs/1707.06347).
