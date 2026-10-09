# DFlash: drafting, verification, and speed

[中文](dflash.md) · **English**

> Checked: 2026-10-09 · Prerequisite: [generation and KV cache](kv-cache-and-inference.en.md) · Start with the flow and example; read sampling, caches, and the experiment table when implementing.

Generating several tokens can all sound like “parallel prediction.” First ask: **who proposes, who decides, and what happens to the work behind a wrong guess?** MTP concerns training objectives and prediction modules; DFlash uses a small drafter to accelerate a target model. The names describe different levels of a system.

## Separate 3 decisions

| What changes? | Question | What does not follow? |
| --- | --- | --- |
| Training objective | Should representations predict further-ahead tokens? | MTP loss guarantees faster inference |
| Draft generation | How does a small model or auxiliary module propose candidates? | Every proposal can be emitted |
| Verification and scheduling | Which proposals survive, and how are caches managed? | Verifying a block always costs the same as 1 token |

[Early MTP work](https://arxiv.org/abs/2404.19737) uses multiple heads on a shared trunk; [DeepSeek-V3](https://arxiv.org/abs/2412.19437) uses sequential MTP modules. Do not draw every MTP design as the same architecture. These modules can supply speculative drafts, but verification is a separate part of producing reliable output.

## One DFlash generation cycle

The [original DFlash paper](https://arxiv.org/abs/2602.06036) freezes the autoregressive target and trains a lightweight block-diffusion drafter. Target context features enter each draft layer's K/V; a confirmed anchor token and masked positions produce parallel proposals in one forward pass. The target itself is not replaced by a diffusion LLM.

```text
confirmed prefix → target features + confirmed anchor token
                                 ↓
                      [anchor | MASK | MASK | MASK]
                                 ↓  one draft pass
                      [anchor | guess1 | guess2 | guess3]
                                 ↓  causal target verification
                      accepted prefix + target-supplied token
                                 ↓
                         repair caches; start next cycle
```

Training must respect the same information boundary: full answers may be available, but a predicted block must not read target features containing its future answers. The paper uses random anchors and specialized attention masking, with greater loss weight on early positions. The [project explanation](https://z-lab.ai/projects/dflash/) describes feature injection and shared embedding/LM-head components.

Why emphasize early positions? The following example needs no model run.

## Why can't later correct guesses simply stay?

Let `A B X D` denote 4 draft tokens without assuming a tokenizer. Start with greedy decoding: the target chooses its highest-probability token.

| Position | Draft | Target choice under the draft prefix | Action |
| --- | --- | --- | --- |
| 1 | A | A | Accept |
| 2 | B | B | Accept |
| 3 | X | C | First mismatch; emit C |
| 4 | D | D | Cannot accept; computed under a prefix containing X |

The cycle can emit `A B C`, not declare `A B C D` correct. Replacing X with C may change the next conditional distribution. What survives is a **continuous accepted prefix**, not every matching position scattered across the block.

The following implements only greedy verification control flow. `target_choices[position]` must correspond to that draft prefix; the final element is the bonus token when all proposals match. Real inference also needs logits, masks, and caches. This is not a complete DFlash implementation.

```python
def verify_greedy_block(draft, target_choices, eos_token=None):
    if len(target_choices) != len(draft) + 1:
        raise ValueError("expected one target choice per draft token plus a bonus")
    output = []
    accepted = 0
    for position, candidate in enumerate(draft):
        expected = target_choices[position]
        if candidate != expected:
            output.append(expected)
            return output, accepted
        output.append(candidate)
        accepted += 1
        if eos_token is not None and candidate == eos_token:
            return output, accepted
    output.append(target_choices[-1])
    return output, accepted


emitted, accepted = verify_greedy_block([10, 20, 99, 40], [10, 20, 30, 40, 50])
assert emitted == [10, 20, 30] and accepted == 2
```

This also explains cache rollback. Verification may compute K/V for X and its suffix; those states cannot remain in the committed prefix. Retain only valid prefix state. If the replacement C has not gone through a forward pass, its K/V still needs to be computed later. Emitted-token count is not automatically the current cache length. Draft and target caches must each stay aligned.

## Sampling with temperature is not argmax matching

Greedy consistency is straightforward. Sampling must preserve the target's **distribution**, not identical text under the same seed. [Speculative sampling](https://arxiv.org/abs/2302.01318) uses acceptance/rejection correction rather than accepting plausible-looking tokens.

For target distribution $p$, actual proposal distribution $q$, and proposal $y\sim q$, accept with

$$\min\left(1,\frac{p(y)}{q(y)}\right).$$

On rejection, sample from normalized $[p-q]_+$. See the [MTP sampling example](multi-token-prediction.en.md) for an exact 2-token calculation.

“Actual proposal” matters. After top-k, temperature, or a selector changes the proposal, the verifier needs the corresponding $q$, not the old softmax. Parallel guesses are not automatically samples drawn sequentially from target conditionals. The proof concerns distributions; it cannot repair invalid caches, masks, or probability alignment.

## Account for a complete cycle

Let the baseline take $T_{\mathrm{AR}}$ per output token. Drafting, verification, and other cycle costs are $T_d,T_v,T_o$. Let $E[N]$ count actual new emitted tokens, including accepted proposals and the target-supplied token, but not the already-known anchor. Roughly,

$$\text{speedup}\approx
\frac{T_{\mathrm{AR}}E[N]}{T_d+T_v+T_o}.$$

This is an amortized fixed-load estimate, not a queueing model. The following numbers are **teaching assumptions, not DFlash benchmarks**: 10 ms/token baseline, 3 ms draft, 12 ms verify, and 1 ms other cost.

| Tokens emitted per cycle | Average time per token | Relative speed |
| --- | --- | --- |
| 1 | 16 ms | 0.625×; slower |
| 2 | 8 ms | 1.25× |
| 4 | 4 ms | 2.5× |

Change the costs in this small calculation:

```python
import math


def cycle_speedup(baseline_ms, draft_ms, verify_ms, overhead_ms, emitted_tokens):
    values = (baseline_ms, draft_ms, verify_ms, overhead_ms, emitted_tokens)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("inputs must be finite")
    if baseline_ms <= 0 or emitted_tokens <= 0:
        raise ValueError("baseline and emitted tokens must be positive")
    if min(draft_ms, verify_ms, overhead_ms) < 0:
        raise ValueError("costs must be nonnegative")
    cycle_ms = draft_ms + verify_ms + overhead_ms
    if cycle_ms <= 0:
        raise ValueError("cycle cost must be positive")
    return baseline_ms * emitted_tokens / cycle_ms


assert cycle_speedup(10, 3, 12, 1, 4) == 2.5
assert cycle_speedup(10, 3, 12, 1, 1) == 0.625
```

A longer block may face more suffix rejection and higher verification/cache cost. Parallelism is not free. Under concurrency, drafter compute and memory can also affect other requests. A batch=1 latency improvement does not establish a whole-server throughput gain.

## Developments in 2026: making parallel guesses coherent

Original DFlash proposes positions in parallel; a later position has not observed the token just selected at its predecessor. That leaves room to improve local coherence.

The [official DFlash 2 introduction](https://inco.ai/blog/dflash2/), dated 2026-08-18, keeps the parallel backbone, adds lightweight adjacent-candidate path selection, and uses short convolutions for local block mixing. Path traversal still has sequential work; parallel drafting does not mean the system has no dependencies.

An original miniature example: candidate columns contain `{New, Los}` and `{York, Angeles}`. Independent top-1 choices might produce `New Angeles`; adjacent-pair scoring can favor `New York`. This explains coherence, not access to the correct answer. The target still verifies the result.

Another route is [D-Loop](https://arxiv.org/abs/2610.06011), submitted 2026-10-05: reuse the drafter for another pass, letting suffix prediction condition on a selected prefix. It spends an extra forward pass on proposal quality. Only its abstract and mechanism overview have been checked here; no reproduction establishes a hardware-independent speed ranking.

| Approach | Treatment of future positions | Main cost to examine |
| --- | --- | --- |
| MTP | Additional training targets/modules, with varying architectures | Training/module cost; acceleration still needs verification |
| Original DFlash | One-pass parallel proposals conditioned on target features | Proposal quality, verification, and extra state |
| DFlash 2 | Parallel proposals plus local mixing and path selection | Local modules and selection |
| D-Loop | Shared-backbone suffix refinement after initial drafting | Second draft forward pass |

Versions also affect usability. The [current official repository](https://github.com/z-lab/dflash) lists DFlash 2 and backend-specific support; the old project page still describes vLLM integration as in progress. Check the commit, target/draft checkpoints, and backend before running, rather than copying an old installation command.

## What evidence should precede deployment?

| Check | Minimal comparison | Failure to avoid |
| --- | --- | --- |
| Output correctness | Greedy sequence agreement; distribution tests for sampling | Using same-seed equality as the sampling test |
| Early stopping | First rejection, full acceptance, EOS, length limits | Extra/missing tokens or invalid retained cache |
| Stable acceleration | Fixed inputs, lengths, temperature, concurrency, and hardware | Reporting the longest accepted block instead of mean cycle output |
| User experience | TTFT, streaming intervals, tail latency, throughput | Average tokens/s hiding bursty output |
| Integration cost | Both models/caches, feature extraction, engine versions | Counting draft weights alone |

Establish correctness in a controlled setting before choosing where to enable it. The code here checks local logic and arithmetic, not full drafter training or GPU serving performance. For parallel target scoring, revisit [causal masks and decoding](../core/decoder-only.en.md); for training-target alignment, read [MTP](multi-token-prediction.en.md).
