# RLHF: why both a Reference and a Critic

[中文](reference-and-critic.md) · **English**

> Reading time: ~2 min · Level: core · Last reviewed: 2026-09

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
