# RLHF: the four cases of PPO clipping

[中文](ppo-clipping.md) · **English**

> Reading time: ~2 min · Level: core · Last reviewed: 2026-09

<details class="interview" markdown="1">
<summary>Quick memory: what are $t$, $\rho_t$, and $A_t$?</summary>

- $t$ is the **time step**: for a language model, usually the position of the $t$-th generated token, not the $t$-th optimizer update.
- $s_t=(x,y_{<t})$ is the prompt plus generated prefix; $a_t=y_t$ is the token actually sampled at that position.
- $\rho_t$ (written as $r_t$ in some notes) is the probability ratio assigned by the new and old policies to that **same sampled token**. It is not the reward.
- $A_t$ says whether that token's probability should rise or fall; $\rho_t$ says how far it has already moved.

> **Advantage chooses the direction, the ratio reports the step size, and clipping only stops an excessive move in the correct direction.**

</details>

GRPO changes where advantage comes from while commonly retaining a PPO-style clipped
surrogate. For the sampled token at rollout step $t$, define the token-level probability
ratio between the current and rollout policies:

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\text{old}}(a_t\mid s_t)}.
$$

$\rho_t>1$ means the current Actor has made the token more likely, while $\rho_t<1$
means it has made the token less likely. Since the ratio is always positive, do not
compare the signs of “ratio and advantage.” Compare $\rho_t-1$ with $A_t$ instead.

PPO maximizes the following clipped surrogate (implementations usually minimize its
negative):

$$
\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right).
$$

The parentheses matter: clip $\rho_t$ into $[1-\epsilon,1+\epsilon]$, multiply by $A_t$,
then take the `min` against the unclipped $\rho_tA_t$. With $\epsilon=0.2$, for example,
the clipping interval is $[0.8,1.2]$. This does **not** hard-constrain the actual ratio
to that interval.

<!-- widget:tx-ppo-clip -->

| Advantage | Ratio relative to 1 | What the current policy did | Direction and clipping |
| --- | --- | --- | --- |
| $A_t>0$ | $\rho_t>1$ | raised the probability of a good token | correct; stop rewarding it beyond $1+\epsilon$ |
| $A_t>0$ | $\rho_t<1$ | lowered the probability of a good token | wrong; no lower-side clip, so the gradient can pull it up |
| $A_t<0$ | $\rho_t<1$ | lowered the probability of a bad token | correct; stop rewarding it below $1-\epsilon$ |
| $A_t<0$ | $\rho_t>1$ | raised the probability of a bad token | wrong; no upper-side clip, so the gradient can push it down |

One sign test captures the direction:

$$
(\rho_t-1)A_t
\begin{cases}
>0, & \text{correct direction; clip only after crossing its boundary},\\
<0, & \text{wrong direction; do not clip, so the gradient can correct it.}
\end{cases}
$$

The `min` creates this one-sided behavior because the sign of $A_t$ reverses the ordering:

$$
\ell_t=
\begin{cases}
\min(\rho_t,1+\epsilon)A_t, & A_t>0,\\
\max(\rho_t,1-\epsilon)A_t, & A_t<0.
\end{cases}
$$

Positive advantage clips only the upper side; negative advantage clips only the lower
side. Once the term enters the clipped plateau, its local gradient is zero. Clipping
stops rewarding further movement, but **does not actively pull an already out-of-range
ratio back inside**. Nor is this gradient clipping: the clipped quantity is the
probability ratio inside the policy objective.

<details markdown="1">
<summary><b>Deep dive</b>: why are the old policy, Reference, and Reward Model different?</summary>

- The **old / rollout policy** supplies $\rho_t$'s denominator. After a rollout batch is collected, its log-probabilities stay fixed during that batch's PPO updates and constrain local movement.
- The **Reference Model** is a long-term frozen SFT copy. Its KL term limits cumulative Actor drift over the whole training run.
- The **Reward Model** is normally frozen during PPO and scores completed responses. Its earlier chosen/rejected training uses a Bradley–Terry loss, not this four-case clipped objective.

PPO clipping and Reference KL are therefore two different rulers, while Reward Model training belongs to a separate stage with a separate objective.

</details>

Standard GRPO often applies one group-normalized sequence-level advantage to the
generated tokens of that response. Implementations differ in token aggregation, KL
placement, and clipping details; the table describes the classic local clipped-objective
behavior.
