# Looped Transformers: what today's looped LMs look like

[中文](models.md) · **English**

> Reading time: ~4 min · Level: advanced · Last reviewed: 2026-10-09

## Huginn: prelude, recurrent block, coda

[Huginn (Geiping et al., 2025)](https://arxiv.org/abs/2502.05171) is a 3.5B model in three parts:

- **prelude** (2 layers): turns the input into an embedding $e$;
- **recurrent block** (4 layers): runs over and over; every loop an adapter concatenates the current state $s$ with $e$ and projects it back to the original width, so the input is re-injected each loop; the starting state $s_0$ is random;
- **coda** (2 layers): turns the final state back into a distribution over the next token.

Training samples recurrent depth with a configured expectation around 32 and backpropagates through at most the last eight loops. Earlier states still affect the forward computation, but the full gradient chain is not retained; this is not an unbiased gradient of full unrolling. The paper reports test-time compute benefits after about 800B training tokens. A “50B-equivalent” compute budget is not 50B parameters of capacity or monotonic improvement on every question.

## Ouro: looping built into pretraining

[Ouro (2025)](https://arxiv.org/abs/2510.25741) repeats the decoder stack, not the token-embedding lookup:

| | Ouro 1.4B | Ouro 2.6B |
| --- | --- | --- |
| Layers × width | 24 × 2048 | 48 × 2048 (upcycled by duplicating layers) |
| Loops | 4 | 4 |
| Training tokens | 7.7T | 7.7T |

The 7.7T-token training lineage includes the shared phase before the sizes branch; it is not 7.7T independent tokens at 2.6B from initialization. The paper reports stability issues with deeper recurrence, and depth extrapolation is not guaranteed to help. See [adaptive depth](adaptive-depth.en.md) for the distinction between an exit gate and actually skipping computation.

## Relaxed Recursive Transformers: converting an existing model

[Bae et al.](https://arxiv.org/abs/2410.20672) initialize shared layers from a pretrained model and continue training. Depth-specific LoRAs allow variation and add parameters. Reported comparisons use specific tasks and training budgets, not lossless compression by tying weights alone. The 2–3× throughput potential comes from theoretical / simulated analysis with oracle early exits, not a guaranteed deployment speedup.

## Side by side

| | What loops | Loops in training | Input re-injected | How depth is set |
| --- | --- | --- | --- | --- |
| Huginn | the middle 4 layers | random, mean 32 | yes | chosen at inference |
| Ouro | decoder stack | 4 | no | gate selects output; skipped work depends on engine |
| Relaxed Recursive | shared base with depth-specific LoRAs | configured maximum | no | early exit and depth batching can be studied |
| Mixture-of-Recursions | a shared block | assigned by a router | no | one depth per token |

See [adaptive depth](adaptive-depth.en.md) for Mixture-of-Recursions routing and cache constraints. In comparisons, fix checkpoint, maximum depth, input / output lengths, dtype, KV policy, and batch size. Fewer parameters and faster serving are different claims.
