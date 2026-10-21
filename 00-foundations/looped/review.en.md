# Looped Transformers: review questions

[中文](review.md) · **English**

> Reading time: ~3 min · Last reviewed: 2026-10

## Interview questions

<details class="interview" markdown="1">
<summary>How does a looped Transformer differ from ALBERT's parameter sharing?</summary>

Both share parameters. ALBERT also factorizes embeddings, so its total reduction cannot be attributed to sharing alone. Looped models keep weights shared and let execution depth vary. Fixed parameters do not mean fixed compute, KV, or latency, or guaranteed depth extrapolation.

</details>

<details class="interview" markdown="1">
<summary>What does looping add, and what does it not add?</summary>

It adds depth of computation over shared weights, not external facts or independent weights per loop. Whether that computation helps is empirical. Roughly 2 bits per parameter is a small-model synthetic-task measurement, not a measurement of the released Ouro checkpoints or a capacity theorem; more loops need not improve accuracy.

</details>

<details class="interview" markdown="1">
<summary>What is input injection, and why is it needed?</summary>

Each loop receives input representation $e$ again, as when Huginn combines its prelude output with the current state through an adapter. This avoids relying entirely on the recurrent state to retain the input. It is a design choice, not a necessary or sufficient condition for stable training of every looped model.

</details>

<details class="interview" markdown="1">
<summary>What are the approaches to adaptive depth? Does Ouro's early exit save compute in the released code?</summary>

ACT, PonderNet, exit gates, and depth routers use different objectives. Separate conditional halting probabilities, actual exit probabilities, output selection, and skipped work. The three-step example in [adaptive depth](adaptive-depth.en.md) averages 2.2 steps. Ouro snapshot `7ea635b` executes all configured loops before selecting an output, so selected depth is not executed compute.

</details>

<details class="interview" markdown="1">
<summary>Why does a looped model's KV cache grow? Can the loops share it?</summary>

The weights are shared but hidden-state inputs differ, so full caches usually distinguish (loop, layer). Sharing changes computation and needs error measurements. Checkpoints, prefill policies, and generation lengths differ across papers; protocol differences are not strict reproduction failures. See [costs and limits](costs.en.md) for memory accounting and sources.

</details>

<details class="interview" markdown="1">
<summary>How does looping relate to chain-of-thought?</summary>

CoT adds generated positions; looping adds hidden-state updates. They can coexist. Cached per-token forwards do not recompute the whole prefix. Theoretical CoT simulation requires extra structure and input-length conditions; it does not make arbitrary pretrained models interchangeable. Visible text is not necessarily a faithful explanation either.

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>why an ordinary Transformer's depth and parameters are tied, and how looping unties them;</li>
    <li>which budgets differ between shared and independent weights when both execute twelve layer evaluations;</li>
    <li>how ACT's accumulation, PonderNet's stepwise sampling, and Ouro's cumulative-probability threshold differ, and whether all require per-token decisions;</li>
    <li>when full KV storage scales directly with loop count, and why Huginn's 2+4+2 layout needs only 2.5 times the KV at four loops versus one;</li>
    <li>which tasks suit a looped model and which do not.</li>
  </ol>
</div>

## Papers

- [Universal Transformers](https://arxiv.org/abs/1807.03819): sharing across steps, and ACT
- [ALBERT](https://arxiv.org/abs/1909.11942): cross-layer parameter sharing (for contrast)
- [PonderNet](https://arxiv.org/abs/2107.05407): probabilistic halting
- [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196)
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424)
- [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416)
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171): Huginn
- [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672)
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
- [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741): Ouro
- [Continuous Depth Batching](https://arxiv.org/html/2608.09444v2): KV-sharing evaluation in v2, Appendix E.1
- [MELT](https://arxiv.org/html/2605.07721v3): training-free KV-sharing evaluation in v3, Appendix B
