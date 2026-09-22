# Looped Transformers: what today's looped LMs look like

[中文](models.md) · **English**

> Reading time: ~2 min · Level: advanced · Last reviewed: 2026-09

## Huginn: prelude, recurrent block, coda

Huginn (Geiping et al., 2025) is a 3.5B model in three parts:

- **prelude** (2 layers): turns the input into an embedding $e$;
- **recurrent block** (4 layers): runs over and over; every loop an adapter concatenates the current state $s$ with $e$ and projects it back to the original width, so the input is re-injected each loop; the starting state $s_0$ is random;
- **coda** (2 layers): turns the final state back into a distribution over the next token.

The number of loops is sampled at random for every training step, 32 on average, and backpropagation runs only through the last 8 loops to save memory. It was trained on about 800B tokens, and the paper reports reasoning improving with the number of loops up to a compute load equivalent to a 50B-parameter model.

## Ouro: looping built into pretraining

Ouro (2025) loops the whole model:

| | Ouro 1.4B | Ouro 2.6B |
| --- | --- | --- |
| Layers × width | 24 × 2048 | 48 layers (made by duplicating layers) |
| Loops | 4 | 4 |
| Training tokens | 7.7T | 7.7T |

The paper notes that an early run with 8 loops hit loss spikes, and they settled on 4. The exit gate is in the previous note.

## Relaxed Recursive Transformers: converting an existing model

Instead of training from scratch, turn a pretrained LLM into a looped one: tie a few layers into a block that is reused, then give each loop a small LoRA so the loops can differ slightly (Bae et al., 2024, ICLR 2025). The resulting Recursive Gemma 1B outperforms TinyLlama 1.1B and Pythia 1B. The paper also proposes continuous depth-wise batching, estimated to raise throughput 2 to 3 times.

## Side by side

| | What loops | Loops in training | Input re-injected | How depth is set |
| --- | --- | --- | --- | --- |
| Huginn | the middle 4 layers | random, mean 32 | yes | chosen at inference |
| Ouro | the whole model | 4 | no | exit gate |
| Relaxed Recursive | pretrained layers tied into a block, one LoRA per loop | fixed | no | fixed |
| Mixture-of-Recursions | a shared block | assigned by a router | no | one depth per token |
