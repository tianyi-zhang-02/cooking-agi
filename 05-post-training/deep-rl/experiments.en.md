# RL experiments: correctness before speed

[中文](experiments.md) · **English**

> Reading time: ~6 min · Last reviewed: 2026-10

RL losses can decrease normally while behavior deteriorates. Sampling, targets, and policies all change. Begin with an environment small enough to predict the answer rather than immediately running a large benchmark.

## A complete learning loop

<div class="drl-flow" aria-label="Separate training from evaluation">
<span>Collect<br><small>Policy version and boundaries</small></span><b>→</b><span>Build targets<br><small>Return / TD / GAE</small></span><b>→</b><span>Update<br><small>Fixed targets and old policy</small></span>
</div>

Check shapes, masks, and reward scale before optimization. Evaluate in a separate environment with fixed settings; do not quietly recycle evaluation trajectories into training. A resumable checkpoint may require optimizer state, target networks, RNG state, counters, and replay—not just network weights.

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

## Debug in a useful order

Start with random behavior and a simple baseline, calculate one transition by hand, overfit a small fixed batch, inspect randomness and boundaries, then scale to full training. Change one factor at a time.

If one batch cannot fit, inspect the code. If training return is good but evaluation is poor, inspect distributions and evaluation configuration. If reward rises without task improvement, inspect reward specification. More training steps are not the same remedy for all three.

## Extend the small experiments

Add bandit reward noise and compare multi-seed gradient variance with and without a baseline. Change a GAE boundary from termination to truncation and verify separate bootstrap and trace behavior. Swap the two Double DQN rankings and verify selection/evaluation separation.

Reference: [Statistical evaluation in deep RL](https://arxiv.org/abs/2108.13264). Continue to [language models and tool interaction](llm-bridge.en.md).
