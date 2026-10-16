# Q-learning, PPO, and DPO: what is each learning?

[中文](methods-map.md) · **English**

> Reading time: about 10–13 minutes · Last reviewed: 2026-10

These names often appear together, but they are not successive replacements. Q-learning estimates action values, PPO updates a policy directly, and DPO learns from preference pairs. Start with the data each method receives and the quantity it updates, rather than memorizing the dates.

For the overview, read the diagram and final table. For the mechanics, work through the three examples. The [RL timeline](../../learn/rl-timeline.en.md) covers their origins; [why RL remains hard to train](why-rl-is-hard.en.md) covers cost and failure modes.

## Three routes, not an upgrade ladder {#three-routes}

<figure class="worked-update" aria-label="What each of three learning approaches learns">
<ol>
<li><small>Q-learning</small><strong>Estimate value, then choose</strong><span>A transition supplies a state, action, reward, and next state. Update Q, then use it to choose actions.</span></li>
<li><small>Policy gradient / PPO</small><strong>Try actions, then change the policy</strong><span>The policy produces trajectories. Estimate which actions did better than expected and adjust their probabilities.</span></li>
<li><small>Offline DPO</small><strong>Compare a pair of responses</strong><span>Preference pairs share a prompt. Update the responses’ relative probability ratios against a reference.</span></li>
</ol>
<figcaption>These are learning mechanisms, not a performance ranking. Value and policy learning can combine in Actor–Critic; offline DPO can also sit inside an outer loop that collects new preferences.</figcaption>
</figure>

Suppose the task is to write a sorting function. Providing a correct implementation to imitate gives an SFT setup. Providing two implementations and a preference label supports DPO. Having the current model generate code, run tests, and update from the results gives one online RL setup. The task is the same; the supervision and training process are not.

## Q-learning: connect this step to what comes next {#q-update}

Take a two-step task: choose a route, then reach a destination. The current estimate for action $a$ is 2. Taking it earns 1, and the two action estimates at the next state are 3 and 4. Let the discount be $\gamma=0.9$ and the learning rate $\alpha=0.5$.

$$
y=r+\gamma(1-d)\max_{a'}Q(s',a'),\qquad
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl(y-Q(s,a)\bigr).
$$

$d$ indicates true termination. Without termination, the target is $1+0.9\times4=4.6$, and the updated estimate is $2+0.5(4.6-2)=3.3$. If the step ends the task, the target is just 1 and the estimate becomes 1.5. **The same immediate reward produces a different update depending on whether a future remains.**

The 4 is an estimate, not a reward already received: this is bootstrapping. The target uses the maximum next-action value without requiring the data-collection policy to actually choose that action. This is why Q-learning is off-policy. Reusing transitions does not make unlimited reuse safe or training automatically stable.

The convergence guarantee in [Watkins and Dayan (1992)](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf) has conditions: a finite table, sufficient state-action visits, suitable decreasing learning rates, and others. Neural approximation, finite data, and changing sampling distributions do not inherit that guarantee. DQN’s replay and target network address some training problems, not all of them. Continue with [Fitted Q](fitted-q.en.md) and [DQN](dqn.en.md).

## PPO and GRPO: estimate quality before changing probabilities {#policy-update}

Policy gradients need not start by learning a Q table. They can directly raise the probabilities of better-performing actions. “Better” is usually measured against a baseline rather than by raw reward alone; advantage provides that comparison.

A common Actor–Critic implementation of PPO estimates advantages with a value model and GAE. Suppose a sampled token had probability 0.20 under the old policy and now has probability 0.26, giving a ratio $\rho=1.3$. With advantage 2 and clipping interval $[0.8,1.2]$:

$$
L_{\mathrm{clip}}=\min\bigl(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\bigr)
=\min(2.6,2.4)=2.4.
$$

This surrogate is **maximized**; implementations usually negate it to obtain a loss. Raising this sample’s probability further gives no additional benefit on the clipped branch, but other samples can still change shared parameters. Clipping is not a hard bound on the actual ratio or a guarantee on KL. See [PPO (2017), §3](https://arxiv.org/pdf/1707.06347); [PPO clipping](../rlhf/ppo-clipping.en.md) covers the negative-advantage case.

One key change in GRPO is the advantage estimate: sample several responses to the same prompt and use group-standardized rewards in place of a value model's estimate, at the cost of sampling each prompt several times; an all-equal group must also avoid $0/0$ (add a small denominator constant or skip the group). How $[0,0,1,1]$ becomes $[-1,-1,1,1]$, and why an all-equal group has no relative signal yet need not stop all updates, is worked through in the [LLM bridge](llm-bridge.en.md#group-signal); group size, sparse rewards, and long-task costs are in [the same note](llm-bridge.en.md#sparse-groups). [DeepSeekMath (2024), §4.1](https://arxiv.org/html/2402.03300v3) introduced this approach.

## DPO: learn preferences without an online rollout loop {#dpo-pair}

Standard offline DPO uses fixed $(x,y_w,y_l)$ triples: a shared input, a preferred response, and a dispreferred response. A relationship between KL-regularized reward maximization and preference modeling yields a direct policy objective. This training loop need not repeatedly generate responses or fit a separate reward model. **That does not make the data free, and it is not Q-learning.** See [DPO (2023), §3–4](https://arxiv.org/html/2305.18290v3).

Summarize a pair’s changes relative to the reference in one margin:

$$
m=\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right],\qquad
\ell=-\log\sigma(m).
$$

In an original two-response toy distribution, set $\beta=1$. The reference assigns 0.5 to each response; the current policy assigns 0.6 and 0.4. Then $m=\log1.5$ and the loss is $-\log0.6\approx0.5108$, below the initial $-\log0.5\approx0.6931$. A language model uses conditional log-probabilities of complete responses, not a softmax over just these two candidates.

One subtlety: **a lower loss does not guarantee a higher absolute probability for the chosen response.** In a separate three-response example, keep $\beta=1$ and fix the reference at $(0.2,0.2,0.6)$. The current policy changes from that same distribution to $(0.15,0.05,0.80)$, with chosen / rejected in the first two positions. Chosen probability falls, but rejected probability falls further. The ratio of reference-relative ratios is 3, so DPO loss falls from $\log2$ to $\log(4/3)\approx0.2877$. This counterexample describes what the objective constrains; it does not simulate a gradient step.

If preference labels reward length or a pleasing tone, the model may learn those shortcuts. DPO removes some online RL machinery, not the need for clear labeling criteria, coverage, or independent evaluation. See [DPO, GRPO, and RLVR](../rlhf/after-rlhf.en.md) for the fuller derivation.

## Three kinds of “old model” {#model-roles}

| Role | Why it exists | How it usually changes |
| --- | --- | --- |
| Q-learning / DQN target network | A more stable estimate for bootstrap targets | Periodic copies or slow updates; tabular Q-learning need not have a separate network |
| PPO old / behavior policy | Identify the sampling distribution and construct probability ratios | Changes across rollout batches; asynchronous methods may separate actual behavior from a proximal policy used for updates |
| DPO / RLHF reference policy | Anchor preference optimization or KL regularization | Usually fixed within a standard training stage; not simply the previous rollout model |

All three may involve “a copy of the model,” but they serve different purposes. Cached log-probabilities can save some forward passes. Parameter sharing, offloading, and parallelism change costs again. Model count alone is not a memory estimate.

## Choose a starting point from the data you have {#choose}

| What you have | A reasonable first experiment | Costs that remain |
| --- | --- | --- |
| Reliable demonstrations | An SFT baseline | Coverage, consistent inputs and targets, independent evaluation |
| Reliable preference pairs for shared inputs | An offline DPO baseline | Preference collection, reference computation, distribution shift |
| Repeatable interactions with reasonably trustworthy scores | Online methods such as PPO / GRPO | Rollouts, verification, exploration, stable updates |
| Discrete-environment transitions for learning action values | Q-learning / DQN and extensions | Exploration, value errors, coverage, bootstrap stability |

These are starting points, not mandatory rules. PPO is not exclusive to human preferences; GRPO is not exclusive to mathematics. RLVR describes verifiable rewards rather than prescribing an optimizer. Next, [training bottlenecks](why-rl-is-hard.en.md) asks where the time and budget go after choosing an algorithm.

Run the three numerical examples with `python 05-post-training/deep-rl/code/rl_tradeoffs.py`. The program checks teaching examples only; it does not train models or benchmark GPUs.
