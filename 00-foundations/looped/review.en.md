# Looped Transformers: review questions

[中文](review.md) · **English**

> Reading time: ~4 min · Level: advanced · Last reviewed: 2026-09

## Interview questions

<details class="interview" markdown="1">
<summary>How does a looped Transformer differ from ALBERT's parameter sharing?</summary>

Both share weights across layers. ALBERT's aim is fewer parameters; inference compute is the same as before and the depth is fixed. A looped Transformer's aim is to make depth and compute adjustable: run the same block for more loops and you get deeper computation with the same parameters.

</details>

<details class="interview" markdown="1">
<summary>Why does looping improve reasoning but not knowledge capacity?</summary>

Reasoning needs serial steps, and a token's serial steps are set by depth, which looping adds. Knowledge lives in the parameters, which looping does not add: Ouro measures about 2 bits per parameter with or without loops, and Saunshi et al. find worse perplexity and memorisation for looped models at the same compute, while their reasoning nearly matches deeper models.

</details>

<details class="interview" markdown="1">
<summary>What is input injection, and why is it needed?</summary>

The embedding $e$ of the original input is fed back into the recurrent block every loop; Huginn, for example, uses an adapter that concatenates the current state with $e$. Without it the input appears only once at the start and has to survive in the hidden state loop after loop; with many loops it is easily lost, and training is less stable.

</details>

<details class="interview" markdown="1">
<summary>What are the approaches to adaptive depth? Does Ouro's early exit save compute in the released code?</summary>

ACT (stop once the accumulated halting probability passes a threshold), PonderNet (a halting distribution plus a KL to a geometric prior, with unbiased gradients), Ouro's exit gate (entropy-regularised in stage one, trained on loss-improvement labels in stage two), and Mixture-of-Recursions (a router assigns each token a depth). In Ouro's released Hugging Face code all 4 loops always run; early exit only chooses which loop's output is read, and saves no compute.

</details>

<details class="interview" markdown="1">
<summary>Why does a looped model's KV cache grow? Can the loops share it?</summary>

Every loop runs attention again, so by default each (loop, layer) pair keeps its own K/V: Ouro 1.4B keeps 96 slots per token, 4 times as many as without looping. Whether they can share depends on training: Ouro reports near-lossless reuse of the last loop during decoding, but two independent 2026 papers reproducing it on Ouro saw large drops, while Huginn is insensitive to sharing. Do not assume it is free.

</details>

<details class="interview" markdown="1">
<summary>How does looping relate to chain-of-thought?</summary>

Both add serial computation steps. CoT does it by generating tokens, one forward pass per token, with visible intermediate work; looping takes extra steps inside the hidden state, with no tokens and no visible work. Saunshi et al. prove that $m$ steps of CoT can be simulated by looping a slightly larger block $m$ times, which is why looped models are also called latent reasoning.

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>why an ordinary Transformer's depth and parameters are tied, and how looping unties them;</li>
    <li>what a 1-layer model solving addition when looped 12 times says about what reasoning ability depends on;</li>
    <li>how ACT, PonderNet, and Ouro's exit gate each decide when a token stops;</li>
    <li>why a model looped 4 times keeps, by default, 4 times the KV cache of an unlooped one;</li>
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
- [Continuous Depth Batching](https://arxiv.org/abs/2608.09444): reproducing per-loop KV sharing
- [MELT](https://arxiv.org/abs/2605.07721): evaluating training-free KV sharing
