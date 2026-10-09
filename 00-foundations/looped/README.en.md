# Looped Transformers: reusing one block

[中文](README.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-10-09

Run a four-layer block three times: that is twelve layer evaluations with only four layers of stored weights. Each pass receives the state produced by the previous one, so the result usually changes. A looped model reuses parameters to spend more computation; it does not acquire new facts merely by looping.

## Depth and parameters normally come as a pair

In an ordinary $L$-layer Transformer without parameter sharing, each layer has its own weights. Adding layers in that design increases both depth and parameters. Recurrence separates these budgets. Generating intermediate tokens or using tools can also add problem-solving computation, but those are different mechanisms.

## Loop the block

A looped Transformer stores a single $k$-layer block and applies it $L$ times:

$$h^{(t+1)} = f_\theta\big(h^{(t)},\, e\big), \qquad t = 0, 1, \ldots, L-1$$

$\theta$ is shared across loops; $e$ is an input representation that some designs re-inject at each step. The recurrent block stores $k$ layers of weights and performs $kL$ layer evaluations. Embeddings, output heads, and extra adapters count separately. This is not equivalent to having $kL$ independently parameterized layers.

Shared parameters need not produce identical outputs. Starting at 0, repeatedly applying $h\leftarrow 0.5h+1$ gives 1, 1.5, and 1.75: the input state changes. In contrast, $h\leftarrow 2h$ can grow without bound. Reuse alone guarantees neither convergence nor improving answers.

<!-- widget:tx-loop-unroll -->

## An old idea, used in a new way

- [Universal Transformer](https://arxiv.org/abs/1807.03819) (2018) shares transitions across positions and depth and studies ACT for allocating computation.
- [ALBERT](https://arxiv.org/abs/1909.11942) (2019) combines layer sharing with factorized embeddings to reduce parameters. Sharing does not automatically reduce executed layers; the 108M → 12M comparison is not due to sharing alone, nor does it establish unchanged runtime.
- **Huginn and Ouro, introduced in 2025**, apply recurrence to language-model pretraining. Depth is an experimental variable, not a universal reasoning switch: extrapolation beyond training depends on the checkpoint and task.

## How this series reads

1. [Why loops help reasoning](why-loops-reason.en.md): serial steps, the theory, and how it relates to CoT (interactive)
2. [How many loops per token](adaptive-depth.en.md): ACT, PonderNet, Ouro's exit gate, Mixture-of-Recursions (interactive)
3. [What today's looped LMs look like](models.en.md): Huginn, Ouro, Relaxed Recursive Transformers
4. [Costs and limits](costs.en.md): latency, KV cache, training stability, and the scope of knowledge-capacity experiments
5. [Review questions](review.en.md): interview questions and a self-check
