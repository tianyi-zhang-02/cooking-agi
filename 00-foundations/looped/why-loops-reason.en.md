# Looped Transformers: why loops help reasoning

[中文](why-loops-reason.md) · **English**

> Reading time: ~2 min · Level: advanced · Last reviewed: 2026-09

## Some problems need serial steps

"In which city is the university of A's manager?" First find A's manager, then the manager's university, then the city it is in: every step depends on the one before. Multi-hop questions like this, and iterative algorithms (a carry propagating through a long addition), need a number of **serial steps** that grows with the problem.

For a fixed-depth model, the serial steps available per token are its layers. Looping makes that number adjustable: one more loop, one more step.

<!-- widget:tx-loop-reach -->

## What theory and experiments say

- **It can execute programs**: Giannou et al. (2023) built a 13-layer looped Transformer that runs a program written into its input (like a punch card), emulating a calculator, a linear-algebra library, and even backpropagation in context.
- **It can learn algorithms**: Yang et al. (2023, ICLR 2024) found that on in-context data-fitting tasks a looped Transformer matches an ordinary one with less than 10% of the parameters.
- **Reasoning follows depth, not parameters**: Saunshi et al. (2025) compared two models with the same compute: $k$ layers looped $L$ times, and $kL$ layers without looping. On reasoning tasks the first nearly matches the second with $L$ times fewer parameters, and a 1-layer model looped 12 times solves addition. Accuracy grows roughly with the logarithm of effective depth:

$$\text{acc} \approx \alpha \cdot \log(\text{effective depth}) + \beta$$

At the same compute, though, looped models have worse perplexity and memorisation.

## How it relates to CoT

Chain-of-thought also adds serial steps, but by **generating tokens**: every token written is one more full forward pass. Looping takes the extra steps **inside the hidden state**: no extra tokens, and no visible intermediate work either.

Saunshi et al. show the two can be converted into each other: $m$ steps of CoT from an $L$-layer model can be simulated by looping an $L + O(1)$-layer block $m$ times (with a little extra width, a few extra heads, and $m$ placeholder tokens). That is why looped models are often called "latent reasoning": the thinking moves from text into the hidden state.

## Where it stops

Looping helps with **reasoning over knowledge**, not with **storing knowledge**. In Ouro's experiments the model stores about 2 bits of knowledge per parameter with or without looping; the gains from looping show up in knowledge manipulation and multi-hop reasoning.
