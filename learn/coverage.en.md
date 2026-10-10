# Content coverage: what is here, and what is missing?

[中文](coverage.md) · **English**

This list prevents two mistakes: losing existing material during reorganization, and treating a mentioned term as a completed explanation. It is not a completion leaderboard.

**As of 2026-10-10, some material still needs writing or review.** The main list has 82 topics; identified appendix entries have grown from the 36 visible in screenshots to 67. These lists overlap and must not be added into an article count. Some explanations already exist but are hard to find; some topics are only mentioned; others are missing. The audit distinguishes these cases.

For studying rather than auditing, start with the [reading routes](README.en.md). This page is a gap list, not a question bank you must finish in order.

Published notes are still being revised, and new material is checked locally before release. Unfinished explanations and reviews stay on this list. A passing build or numerical test does not mean every article has been reviewed.

## What this update focuses on {#current-update}

Before expanding the notes further, we're bringing the existing changes together in a PR for review. This snapshot is not a completed site-wide audit; it only reaches the live site after a separate merge and deployment.

- **Language entry:** a first visit to the homepage starts in English, with a small dismissible tip. The top link switches the same note in one click, and Chinese notes include English terms. Your browser remembers your choice; direct links to a particular language still take precedence.
- **Repository README:** the [main README](../README.md) introduces the project in English, with a separate [Chinese version](../README.zh.md). Reading goals come first, then topic routes, contributing, and local setup.
- **Explanations and figures:** BERT, RL / post-training, Python, and retrieval now have more examples and derivations. Figure text is written for each language, with optional detail to expand. Article-by-article review is still ongoing.

**Checked in this batch:** language switching and mobile layouts for the homepage, coverage page, and three pairs of retrieval notes, along with the first-visit tip, saved language choice, and existing links. The README now uses short entry lists instead of a table you need to scroll sideways on a phone.

**Next:** review the remaining references, then continue checking each article's explanations, figures, and wording in both languages. Having an article, reading its reference, testing an example, and reviewing the whole article are different milestones. The records below keep them separate. Teaching examples in engineering notes are not evidence of full GPU training runs.

## How to read the table

The new [LatentMoE chapter](../00-foundations/moe/latent-moe.en.md) separates backbone, internal FFN, and expert input widths before working through projection, compute, and communication costs. [MLA / RoPE](../00-foundations/deep-dives/latent-and-sparse-attention.en.md#rope-absorption) now includes a rotation-order example: the difficulty is reordering computation, not using the two together. Both follow the original reports, with language-specific figure explanations; example checks do not reproduce model quality.

- **Article available**: a dedicated article or substantive explanation is linked; this does not establish equivalence to every detail in the reference.
- **Partial explanation**: related material exists, but derivations, examples, or scope still need expansion.
- **To add**: the topic stays on the list without an empty placeholder article pretending it is finished.

The scope comes from the [language-model study map](https://tcn6r3nlptym.feishu.cn/wiki/N2Yjwoez0iUA0AkGQBRcSps1nPf). The prose of all 71 pages in the nine main chapters has been read through the browser, including all 16 multimodal pages. Sections 7.1–7.7 share one page and count as one article. All seven ML-foundations and 20 PyTorch articles in Appendix 3 have also been read to the end. These counts describe prose reading, not complete validation of code, images, and external links.

The appendix review led to new chapters on [generalization and training diagnosis](../00-foundations/deep-dives/generalization.en.md), [activations and initialization](../00-foundations/deep-dives/activation-and-initialization.en.md), and [optimizers](../00-foundations/deep-dives/optimizers.en.md). Small examples distinguish optimization failure from poor generalization, ReLU second moments from variance, and Adam bias correction from AdamW decay. Related RNN / GRU, normalization, and fundamentals explanations have also been corrected. Examples are written independently rather than copied from the reference.

A new [PyTorch practice path](../00-foundations/pytorch/README.en.md) covers tensor storage, operation shapes, autograd, and training loops in four chapters, with complete code in both languages. Rather than listing APIs, it explains mistakes that can run silently: cross-batch broadcasting, incorrect loss denominators, unretained intermediate gradients, and restoring weights without complete training state. Numerical tests use PyTorch 2.8.0 / CPU, with versions and teaching limitations stated separately.

New bilingual multimodal notes cover [LLaVA / DeepSeek-VL](../03-multimodal-learning/vlm-designs.en.md), [Qwen-VL](../03-multimodal-learning/qwen-vl.en.md), [Omni](../03-multimodal-learning/omni-streaming.en.md), [OCR compression](../03-multimodal-learning/ocr-compression.en.md), [image generation](../03-multimodal-learning/image-generation.en.md), and [fine-tuning checks](../03-multimodal-learning/vlm-finetuning.en.md). Earlier [ViT](../03-multimodal-learning/vit.en.md), [CLIP](../03-multimodal-learning/clip.en.md), and [BLIP / Q-Former](../03-multimodal-learning/blip-and-q-former.en.md) material remains. OCR 2, Qwen3.5 / 3.8-Omni, and later Qwen-Image releases are checked separately. Teaching calculations are not model-training measurements; multimodal GPU fine-tuning has not been run.

Evaluation now includes [BLEU, ROUGE, and edit distance](../07-evaluation/text-metrics.en.md), with sentence calculations, dynamic-programming code, PPL comparison conditions, and counterexamples. [Benchmarks and long-context tests](../07-evaluation/benchmark-protocols.en.md) adds fixed protocols, evidence positions, unanswerable controls, paired comparisons, and contamination checks. The code generates test fixtures; no model-evaluation results are invented.

Post-training adds [DAPO](../05-post-training/dapo.en.md), [GSPO / ASPO](../05-post-training/policy-ratios.en.md), and [SAO / asynchronous training](../05-post-training/async-policy-learning.en.md). Examples cover sampling costs, loss reduction, stop-gradients, and stale trajectories. Earlier claims that clipping hard-constrains probabilities or that a Critic only supplies a baseline have been corrected. The small calculations are tested; the papers' full training results have not been reproduced.

The previous bilingual additions on [RAG evidence and evaluation](../04-search/rag-evidence.en.md), [IVF / PQ indexes](../04-search/vector-indexes.en.md), and [agent execution](../10-agents/patterns.en.md) remain. New [prompting](../10-agents/prompting.en.md) notes cover task contracts and controlled comparisons; [deep research](../10-agents/deep-research.en.md) covers evidence queues, citation checks, and stopping logic. Runnable local examples are not completed model-comparison experiments.

Earlier updates to [DeepSeek](../00-foundations/model-families/deepseek.en.md), [Qwen](../00-foundations/model-families/qwen.en.md), and [distributed training](../06-systems/distributed-training.en.md) remain. This batch adds routing weights, reward boundaries, ring traffic, weight versions, and recovery examples. BLIP filtering audits and an InstructBLIP ablation plan are also added, without claiming completed training experiments.

Earlier [GPT](../00-foundations/model-families/gpt.en.md) and [Llama](../00-foundations/model-families/llama.en.md) additions remain. New notes cover [Qwen1–2.5](../00-foundations/model-families/qwen-early.en.md) and the [Gated DeltaNet derivation](../00-foundations/deep-dives/gated-deltanet.en.md); Qwen's main article now distinguishes public 3.8 text weights, vision models, and hosted services. [DeepSeek-V4](../00-foundations/model-families/deepseek-v4.en.md) develops hybrid attention, causal boundaries, mHC, and post-training examples. gpt-oss now covers disclosed training stages, effort/tool comparisons, and paired cost accounting without inventing an undisclosed training recipe.

DCA chunk-boundary positions and DSA indexer training and missed-selection examples now extend the context and sparse-attention notes. New bilingual articles cover [Gated Attention](../00-foundations/deep-dives/gated-attention.en.md), [Engram](../00-foundations/deep-dives/engram.en.md), and [Block AttnRes](../00-foundations/deep-dives/attention-residuals.en.md). Local examples test gate initialization, hash collisions, depth softmax, and resource arithmetic, not full-model training. Earlier MoE, MLA, NSA, YaRN, and speculative-decoding examples remain.

Reference code has not been executed, and image-based equations, embedded PDFs, and linked videos have not all been checked. Appendix 2's index has been read; six bilingual notes and local tests now cover string matching, monotonic queues, BSTs, knapsack, and sequence / state-machine DP. Forty-three individual Appendix 1 articles have now been read, as recorded below; the remainder is not complete. Explanations, examples, and code are written independently, with claims checked against papers or official implementations. Site-wide currency and article-by-article technical reviews are still incomplete. A successful build does not establish content completeness.

Inference now includes [the request lifecycle](../06-systems/llm-serving.en.md), [a complete memory budget](../00-foundations/deep-dives/kv-cache-and-inference.en.md#inference-budget), and [cache lifetime](../00-foundations/deep-dives/attention-kernels.en.md#cache-lifecycle). Two more reference articles were read: request handling and inference memory. Implementation links are version-pinned; local arithmetic is not presented as GPU performance measurement.

<span id="appendix-questions"></span>

## Appendix questions: checking individual gaps {#appendix-question-map}

Reopening the reference directory revealed **31 more entries**, bringing the inventory to **67**, including the algorithm and PyTorch appendix entries. This is neither the size of the whole library nor a claim to have read 67 articles. Source reading and site coverage are recorded separately; image equations and external links are not automatically counted as verified.

The useful gaps are not limited to new model names:

| Finding | Examples | What follows |
| --- | --- | --- |
| Substantive explanations already exist | Online softmax, MHA implementation, RoPE, BF16/FP16, normalization axes, entropy/CE/KL | Link the right sections and compare additional source questions |
| Related material needs a more direct explanation | Rejection sampling, entropy collapse, and source questions still awaiting comparison | Add the missing derivation, example, or diagnostic rather than another definition |
| Previously only named; now covered in dedicated notes | [BM25 versus TF-IDF](../04-search/tfidf-and-bm25.en.md), [primality and sieves](../interview/algorithms/primes-and-sieves.en.md) | Calculations, proofs, and boundaries are present; BM25 prose has been compared, while the sieve reference still needs reading |
| Exact model confirmed and dedicated coverage added | [Kimi K3 / NoPE](../00-foundations/deep-dives/nope-and-order.en.md) | From swapped inputs to matrix recurrence; distinguish no RoPE from no long-context training |

For example, the distillation article already computes entropy, cross-entropy, and KL on the same distributions, so it should not be reported as missing. BM25 previously appeared only in the hybrid-retrieval article; this batch adds a dedicated scoring explanation. That is the distinction an item-level audit should make.

The 2026-10-09 additions cover [Muon](../00-foundations/deep-dives/muon.en.md), [DFlash versus MTP](../00-foundations/deep-dives/dflash.en.md), BM25 / TF-IDF, primality / sieves, [Qwen3 Embedding versus BGE-M3](../04-search/embedding-models.en.md), [InfoNCE versus CE](../04-search/dual-encoder.en.md#infonce-and-ce), and NoPE with sequence order.

Also added: [BERT from inputs to fine-tuning](../00-foundations/core/bert.en.md). One short sequence separates input corruption, attention masking, and loss masking before connecting MLM/NSP to task heads.

The inventory now has **65 dedicated explanations and 2 entries still needing expansion or comparison**. No listed topic entirely lacks relevant dedicated material; that does not complete the release audit. Forty-three Appendix 1 articles have been read. The latest comparisons cover Qwen3 Embedding / BGE, InfoNCE / CE, and BM25 / TF-IDF: training signals, candidate comparisons, normalization, and tied scores. Broad claims in the reference are not adopted automatically. The earlier [Python names, copies, and function calls](../interview/python-objects.en.md) note remains in the coding route.

In engineering practice, a new [request-tracing example](../practice/post-training/experiments-and-release.en.md#trace-a-failure) distinguishes input, training, and evaluation fixes. Earlier additions cover [GRPO/PPO for long tasks](../05-post-training/deep-rl/llm-bridge.en.md#long-horizon-choice), [student sampling in distillation](../05-post-training/distillation.en.md#opd-choices), and [rollouts as training data](../05-post-training/post-training-infrastructure.en.md#rollout-record). Start with small examples, then open the gradient and code details as needed. CPU checks are not full training or GPU performance replications.

The earlier [two-GPU layer example](../06-systems/tensor-parallel.en.md) and [eight-GPU deployment comparison](../06-systems/distributed-training.en.md#eight-gpus) remain available, including input gradients, communication, and resource arithmetic.

Training engineering now covers [checkpoint recovery](../practice/post-training/checkpoint-and-resume.en.md), [the CPU-to-GPU path](../00-foundations/pytorch/training-loop.en.md), and [memory accounting](../00-foundations/deep-dives/precision-and-memory.en.md). Momentum, three batches, and a billion-parameter estimate introduce the ideas before code and engineering limits. These three reference articles remain unread and do not increase the reading count.

Post-training now explains three common debugging questions: [zero loss with a nonzero gradient](../05-post-training/after-ppo.en.md#zero-loss-gradient), [why an out-of-range ratio can still have a gradient](../05-post-training/rlhf/ppo-clipping.en.md#clipped-token-gradients), and [when a batch is generated, reused, and refreshed](../05-post-training/rlhf/on-off-policy.en.md#grpo-data-lifecycle). Small calculations have matching CPU checks; logged values, gradients, and actual training outcomes remain distinct.

Our [MoE router note](../00-foundations/moe/router.en.md#dispatch-example) now follows dispatch, expert computation, weighted accumulation, and gradient checks. [Load balancing](../00-foundations/moe/load-balancing.en.md#sequence-balance) uses two sequences to show why batch balance differs from sequence balance. Discrete counts, normalization denominators, and padding are checked separately. Teaching examples are not model measurements, and reading a reference does not mean adopting its inaccurate claims.

### Where should readers find these topics?

New pages should not follow the reference question order, nor should everything be filed under “advanced.” Keep the site's four existing reading areas and organize by prerequisites:

| Reading goal | Start here | What can wait |
| --- | --- | --- |
| Understand how models learn | Attention, normalization, cross-entropy, and one parameter update | Muon, distributed communication, speculative decoding |
| Focus on post-training | SFT → PPO/GRPO → sampling, KL, and loss debugging | Every model generation and serving-engine internals |
| Focus on retrieval or deployment | Dual encoders and hybrid retrieval for search; generation and KV cache for deployment | You do not have to finish the other route |
| Implement something | Small modules → shape/gradient checks → a complete small experiment | Getting one module right matters more than collecting algorithm names |

The four groups below organize the audit, not difficulty levels. “Expansion or comparison needed” can mean missing local detail or unread source prose; each entry explains which.

<!-- widget:appendix-coverage -->

## Topic-by-topic mapping

<!-- widget:reference-coverage -->

## Additional material retained here

The reference is not a limit on this site. RNNs / BPTT, the Deep RL route, looped Transformers, evaluation calibration, tools and memory, system design, public recommender-system walkthroughs, Python, and algorithm practice remain available.

This reorganization changes classification and entry points, not public article addresses. Existing examples, code, and diagrams are not replaced with summaries. Previously retired Quant and discussion sections are not republished in this update.

## How the comparison will continue

Continue through the reference in the browser, recording prose, image-based equations, and code separately. Compare definitions, derivations, examples, and implementation details. Fill gaps in existing chapters where possible; create a new article when it answers a distinct question, rather than duplicating another site's table of contents.

To report a gap, include the topic, a public source, and the question you want explained in an issue. Maintenance notes live in `CONTENT_COVERAGE.md`. The main mapping comes from `site/reference-coverage.toml`; the screenshot and browser-directory appendix inventory comes from `site/appendix-coverage.toml`.

New [tools, MCP, and skills](../10-agents/tools-and-skills.en.md) notes and rewritten [model selection and routing](../10-agents/model-choice.en.md) follow a review request and four hypothetical tasks through execution, permissions, budgets, and tradeoffs. No live-model or serving benchmark is claimed.
