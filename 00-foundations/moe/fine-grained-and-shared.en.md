# MoE: fine-grained and shared experts

[中文](fine-grained-and-shared.md) · **English**

> Reading time: ~2 min · Level: advanced · Last reviewed: 2026-09

## Cut the experts finer

DeepSeekMoE's (2024) first change: split each of the $N$ experts into $m$ smaller ones, $mN$ in total, and activate $mK$ per token. Compute per token stays the same, but the number of possible combinations jumps.

The paper's example: choosing 2 of 16 experts gives only 120 combinations; cut into 64 and choose 8, and the count is

$$\binom{64}{8} = 4{,}426{,}165{,}368$$

The more combinations there are, the more each expert can learn one narrow thing instead of being forced to learn a bit of everything.

## Shared experts

The second change: set aside $K_s$ shared experts that **every token passes through**, to hold common knowledge. The routed experts then no longer each have to relearn grammar and frequent words, so there is less redundancy.

The paper reports DeepSeekMoE 16B matching LLaMA2 7B with about 40% of the compute. DeepSeek-V3 keeps both changes: of its 61 layers the first 3 are dense, and every other layer has 1 shared expert plus 256 routed experts, 8 chosen per token, each expert with a hidden size of only 2048.

## Not everyone does this

| Model | Routed experts per layer | Experts per token | Shared expert |
| --- | --- | --- | --- |
| DeepSeek-V3 | 256 | 8 | 1 |
| Qwen3-235B-A22B / 30B-A3B | 128 | 8 | none |
| Llama 4 Maverick | 128 | 1 | 1 (alternating with dense layers) |
| Llama 4 Scout | 16 | 1 | 1 |
| gpt-oss-120b / 20b | 128 / 32 | 4 | none |

Qwen3's MoE explicitly drops the shared expert (its predecessor Qwen2.5-MoE had one); Llama 4 sends each token to 1 shared expert plus 1 of 128 routed ones.

## How to choose

Fine-grained experts separate expertise more cleanly, at the cost of more routing decisions and more fragmented cross-device traffic. A shared expert gives common knowledge a fixed home, at the cost of computing those parameters for every token. Both designs ship in current large models and there is no single answer; when you meet a new MoE, ask these two questions first: how finely are the experts cut, and is there a shared expert?
