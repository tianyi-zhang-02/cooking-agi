# RL experiments: correctness before speed

[中文](experiments.md) · **English**

> Reading time: ~6 min · Last reviewed: 2026-10

RL losses can decrease normally while behavior deteriorates. Sampling, targets, and policies all change. Begin with an environment small enough to predict the answer rather than immediately running a large benchmark.

## A complete learning loop

<div class="drl-flow" aria-label="Separate training from evaluation">
<span>Collect<br><small>Policy version and boundaries</small></span><b>→</b><span>Build targets<br><small>Return / TD / GAE</small></span><b>→</b><span>Update<br><small>Fixed targets and old policy</small></span>
</div>

Check shapes, masks, and reward scale before optimization. Evaluate in a separate environment with fixed settings; do not quietly recycle evaluation trajectories into training. A resumable checkpoint may require optimizer state, target networks, RNG state, counters, and replay—not just network weights.

If you are unsure what a rollout should retain, start with [one training record](../post-training-infrastructure.en.md#rollout-record): only 9 of its 20 input positions are policy actions; tool results and padding should not enter policy loss. That section also separates task termination from collection truncation before you return to checking targets.

## Run 2 small CPU programs first

From the repository root, run the standard-library checks and then the PyTorch checks:

```bash
python3 05-post-training/deep-rl/code/rl_checks.py
python3 05-post-training/deep-rl/code/torch_updates.py
```

[Arithmetic and boundary checks](code/rl_checks.py) cover Bellman updates, returns, GAE, Double DQN, and ESS. [PyTorch checks](code/torch_updates.py) verify Actor-gradient signs, detached targets, and continuous-action gradient flow, then train a two-action bandit.

Bandit rewards are $[1,3]$, starting from equal probabilities. A fixed seed generates sampled actions, and REINFORCE updates the logits. Expect the second action's probability to increase, not reward to improve monotonically every step. **This checks one-stage mechanisms; it is not a full DQN / SAC reproduction or benchmark.**

## Keep a checkable quantity at each stage

| Stage | Record at least | What it can reveal |
| --- | --- | --- |
| Data | Episode length, termination / truncation, reward distribution | Early endings, empty rewards, incorrect resets |
| Value | Targets, predictions, TD errors, magnitude | Divergence or wrong units |
| Policy | Entropy, action frequencies, KL / ratios where relevant | Collapse, excessive changes, no changes |
| Optimization | Gradient norms, learning rates, update counts | Detached gradients or wrong update schedules |
| Evaluation | Multi-seed returns, success, failure categories | Instability hidden by a mean |

Label different axes for loss and reward. High Q with low real return can indicate overestimation. Low entropy can mean learning or getting stuck. Interpret numbers through behavior, not one scalar alone.

## State the budget before comparing algorithms

Environment interactions, gradient updates, wall-clock time, and hardware are different budgets. Several evaluation episodes from one trained seed are not several independent training seeds. Report between-seed variation and appropriate intervals, not just the best curve.

Tune on validation tasks or splits and limit final-test reuse. Retain raw data when smoothing and disclose the window. Count failed runs. Across tasks, explain normalization rather than letting large reward scales dominate the aggregate.

## Same mean, same conclusion?

Suppose five independent training runs each produce a mean return over separate evaluation episodes:

| Method | Five training-run evaluation means | Overall mean | Median |
| --- | --- | --- | --- |
| A | 10, 10, 10, 10, 50 | 18 | 10 |
| B | 16, 17, 18, 19, 20 | 18 | 18 |

Means tie, the best run favors A, and repeatability favors B. There is no context-free correct aggregation: state whether you care about typical performance, failure risk, or the best result after tuning, and report corresponding uncertainty. Five seeds here illustrate arithmetic, not sufficient statistical power.

One hundred evaluation episodes per seed help estimate that trained model’s performance; they do not turn five models into 500 independent training runs. Respect the hierarchy of tasks, training seeds, and evaluation episodes when estimating intervals or resampling.

## Design an ablation someone else can inspect

To study GAE’s $\lambda$, fix environment, reward, networks, collected data budget, optimizer settings, and evaluation rules. Compare Critic capacity or rollout length separately afterward.

| Record | Concrete convention |
| --- | --- |
| Question | Does estimator length improve stability at a fixed interaction budget? |
| Comparison | $\lambda=0$, an intermediate value, and $\lambda=1$, without also changing three networks |
| Primary result | Independent evaluation at a prespecified budget and variation across training seeds |
| Diagnostics | Advantage distribution, Critic error, KL, gradient scale |
| Exceptions | Keep failed runs and specify stopping or exclusion rules |

If loss falls while task performance worsens, inspect saved rollouts. Numbers locate problems but do not replace observing behavior: a robot may stand still for a small reliable reward, or a language model may lengthen answers to please a judge.

## Why is resume testing part of algorithm correctness?

In a small environment, compare continuous training with a run saved and restored at the same step. Inspect subsequent samples, targets, parameters, and counters. Deterministic configurations can match exactly; nondeterministic devices or parallel environments need defined tolerances, not a promise of universal bitwise equality.

Saving only Actor weights can omit optimizer momentum, Critic, targets, replay, random-number state, and exploration schedules. Producing actions after loading does not establish continuation from the same training state. Test loading separately from restoring the intended training semantics.

## Debug in a useful order

Start with random behavior and a simple baseline, calculate one transition by hand, overfit a small fixed batch, inspect randomness and boundaries, then scale to full training. Change one factor at a time.

If one batch cannot fit, inspect the code. If training return is good but evaluation is poor, inspect distributions and evaluation configuration. If reward rises without task improvement, inspect reward specification. More training steps are not the same remedy for all three.

## Extend the small experiments

Add bandit reward noise and compare multi-seed gradient variance with and without a baseline. Change a GAE boundary from termination to truncation and verify separate bootstrap and trace behavior. Swap the two Double DQN rankings and verify selection/evaluation separation.

Reference: [Statistical evaluation in deep RL](https://arxiv.org/abs/2108.13264). Continue to [language models and tool interaction](llm-bridge.en.md).
