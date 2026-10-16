# Content coverage: what is here, and what is missing?

[中文](coverage.md) · **English**

This page tracks what has been written and what still needs checking. It helps us avoid losing material when reorganizing the notes; it is not a checklist readers have to work through.

**As of 2026-10-16, every topic in the reference list has a reading entry, but the review is not complete.** Of the 82 main topics, 81 have substantive explanations and one is an interview-preparation overview. The appendix lists 65 topics plus algorithm and PyTorch indexes. Several entries link to the same note, so these numbers are not an article count. For the new blocks outside the reference list and their review state, see [what this update focuses on](#current-update).

The listed Appendix 1 articles have now been read; the generated summary below records their status. The next task is to check whether our explanations are accurate and our examples useful, not to increase a reading count. Reading the prose does not verify every source figure, external link, or implementation.

For studying rather than auditing, start with the [reading routes](README.en.md). This page is a gap list, not a question bank you must finish in order.

Published notes are still being revised, and new material is checked locally before release. Unfinished explanations and reviews stay on this list. A passing build or numerical test does not mean every article has been reviewed.

## What this update focuses on {#current-update}

On 2026-10-15 we added a fourth area, Frontier watch, two new blocks, and a few single notes. Anything marked “pending review” is a draft: it opens with a notice and is marked “(draft)” in the sidebar until the site owner has read it. Check the primary sources a draft links before relying on its facts.

| Content | Notes | Review state |
| --- | ---: | --- |
| [Frontier watch](../11-frontier/README.en.md) · [World models](../11-frontier/world-models/README.en.md) | 7 | Reviewed |
| Frontier watch · [Recursive self-improvement](../11-frontier/self-improvement/README.en.md) | 5 | Reviewed |
| Frontier watch · [Test-time compute](../11-frontier/test-time-compute/README.en.md) | 5 | Draft, pending review |
| Frontier watch · [Agent RL](../11-frontier/agentic-rl/README.en.md) | 4 | Draft, pending review |
| Frontier watch · [Interpretability](../11-frontier/interpretability/README.en.md) | 5 | Draft, pending review |
| Frontier watch · [Diffusion language models](../11-frontier/diffusion-lm/README.en.md) | 4 | Draft, pending review (2026-10-16) |
| Foundations · [Scaling laws and hyperparameters](../00-foundations/scaling/README.en.md) | 5 | Draft, pending review (2026-10-16) |
| AI infrastructure · [Quantization](../12-infra/quantization/README.en.md) | 6 | Draft, pending review (2026-10-16) |
| [AI infrastructure](../12-infra/README.en.md): wafers and packaging, CPUs, GPUs, NVIDIA generations, CUDA, performance, frameworks, Kubernetes | 10 | Draft, pending review |
| [Perception, robots, and driving](../13-embodied/README.en.md): perception and geometry, autonomous driving, robot learning | 14 | Draft, pending review |
| Deep RL: [which model to learn](../05-post-training/deep-rl/model-types.en.md), [planning and learning](../05-post-training/deep-rl/planning-and-learning.en.md), [off-policy evaluation estimators](../05-post-training/deep-rl/ope-estimators.en.md) | 3 | Reviewed |
| Other chapters: [one agent or several](../10-agents/single-vs-multi-agent.en.md), [LLMs for decision support](../06-systems/decision-support.en.md), [taking evaluation online](../07-evaluation/eval-observability.en.md) | 3 | Draft, pending review |

Counts are per note, including each series overview; every note also has a Chinese version. None of this is in the reference list, so the comparison tables below do not count it. The table records review state only; it is not a completed site-wide audit. The scope and results of each batch's checks are in the repository's `CONTENT_RELEASE_CHECKLIST.md`.

The previous round (2026-10-10) reorganized the language entry (a first visit starts in English; the top link switches the same note), the repository [README](../README.md) and its [Chinese version](../README.zh.md), and the examples and figures for BERT, RL / post-training, Python, and retrieval.

**Still to do:** continue reviewing our explanations, examples, figures, and wording in both languages, and verify any image equations or external material used to support our claims. Unused reference material stays outside that verification scope; “read” is not the same as “validated.” Technical claims are checked against original papers or official implementations. Teaching examples in engineering notes are not evidence of full GPU training runs.

## How to read the table

For example, [LatentMoE](../00-foundations/moe/latent-moe.en.md) explains widths, routing, and costs with worked calculations; [MLA / RoPE](../00-foundations/deep-dives/latent-and-sparse-attention.en.md#rope-absorption) uses rotation order to show why certain computations cannot be rearranged. Both qualify as “article available.” That label does not mean we reproduced model training or verified every experiment in the papers.

- **Article available**: a dedicated article or substantive explanation is linked; this does not establish equivalence to every detail in the reference.
- **Topic index**: an entry connects several notes, such as the algorithm and PyTorch routes; it is not an empty chapter awaiting content.
- **To add**: the topic stays on the list without an empty placeholder article pretending it is finished.

The scope comes from the [language-model study map](https://tcn6r3nlptym.feishu.cn/wiki/N2Yjwoez0iUA0AkGQBRcSps1nPf). The prose of all 71 pages in the nine main chapters has been read through the browser, including all 16 multimodal pages. Sections 7.1–7.7 share one page and count as one article. All seven ML-foundations and 20 PyTorch articles in Appendix 3 have also been read to the end. These counts describe prose reading, not complete validation of code, images, and external links.

<span id="appendix-questions"></span>

## Appendix questions: checking individual gaps {#appendix-question-map}

The inventory contains **67 entries**: 65 topics plus algorithm and PyTorch indexes. These are not 67 separate articles or the size of the whole reference library. Site coverage and source reading are recorded separately; image equations and external links are not automatically counted as verified.

The useful gaps are not limited to new model names:

| Finding | Examples | What follows |
| --- | --- | --- |
| Substantive explanations already exist | Online softmax, MHA implementation, RoPE, BF16/FP16, normalization axes, entropy/CE/KL | Link the relevant sections and check follow-up questions without duplicating chapters |
| Earlier explanations were too indirect | Rejection sampling and entropy collapse | Derivations, examples, and diagnostics have been added; check whether they make the ideas easier to follow |
| Previously only named; now covered in dedicated notes | [BM25 versus TF-IDF](../04-search/tfidf-and-bm25.en.md), [primality and sieves](../interview/algorithms/primes-and-sieves.en.md) | Both reference articles have been read; our calculations, proofs, and boundary cases are independently written |
| Exact model confirmed and dedicated coverage added | [Kimi K3 / NoPE](../00-foundations/deep-dives/nope-and-order.en.md) | From swapped inputs to matrix recurrence; distinguish no RoPE from no long-context training |

For example, the distillation article already computes entropy, cross-entropy, and KL on the same distributions, so it should not be reported as missing. BM25 previously appeared only in the hybrid-retrieval article; this batch adds a dedicated scoring explanation. That is the distinction an item-level audit should make.

The inventory now has **65 dedicated explanations and 2 topic indexes**. No listed topic entirely lacks relevant dedicated material; that does not complete the release audit. For what each addition covers, see [earlier additions](#earlier-updates). Reading a reference does not mean adopting its outdated or blanket claims.

### Where should readers find these topics?

New pages should not follow the reference question order, nor should everything be filed under “advanced.” Keep the site's existing sections and organize by prerequisites:

| Reading goal | Start here | What can wait |
| --- | --- | --- |
| Understand how models learn | Attention, normalization, cross-entropy, and one parameter update | Muon, distributed communication, speculative decoding |
| Focus on post-training | SFT → PPO/GRPO → sampling, KL, and loss debugging | Every model generation and serving-engine internals |
| Focus on retrieval or deployment | Dual encoders and hybrid retrieval for search; generation and KV cache for deployment | You do not have to finish the other route |
| Implement something | Small modules → shape/gradient checks → a complete small experiment | Getting one module right matters more than collecting algorithm names |

The four groups below organize the audit, not difficulty levels. Each entry describes our coverage, how much of the reference was read, and what remains unverified. Algorithm and PyTorch indexes are counted separately from individual topic articles.

<!-- widget:appendix-coverage -->

## Topic-by-topic mapping

<!-- widget:reference-coverage -->

## Earlier additions {#earlier-updates}

This list groups by topic the additions recorded on this page before 2026-10-14; the explanations themselves live in the notes. Small calculations and CPU checks verify mechanisms, not model training or GPU performance, and reading a reference does not mean adopting every claim in it.

- **Training foundations:** [generalization and training diagnosis](../00-foundations/deep-dives/generalization.en.md), [activations and initialization](../00-foundations/deep-dives/activation-and-initialization.en.md), [optimizers](../00-foundations/deep-dives/optimizers.en.md), and [Muon with its sharding counterexample](../00-foundations/deep-dives/muon.en.md). Examples are written independently rather than copied from the reference; related RNN / GRU, normalization, and fundamentals explanations were corrected.
- **PyTorch:** a five-chapter [practice path](../00-foundations/pytorch/README.en.md) on mistakes that run silently: cross-batch broadcasting, wrong loss denominators, unretained intermediate gradients, and restoring weights without full training state. Numerical tests use PyTorch 2.8.0 / CPU.
- **Multimodal:** [LLaVA / DeepSeek-VL](../03-multimodal-learning/vlm-designs.en.md), [Qwen-VL](../03-multimodal-learning/qwen-vl.en.md), [Omni](../03-multimodal-learning/omni-streaming.en.md), [OCR compression](../03-multimodal-learning/ocr-compression.en.md), [image generation](../03-multimodal-learning/image-generation.en.md), [fine-tuning checks](../03-multimodal-learning/vlm-finetuning.en.md), plus BLIP filtering audits and an InstructBLIP ablation plan. OCR 2, Qwen3.5 / 3.8-Omni, and later Qwen-Image releases are checked separately; multimodal GPU fine-tuning has not been run.
- **Evaluation:** [BLEU, ROUGE, and edit distance](../07-evaluation/text-metrics.en.md) and [benchmarks and long-context tests](../07-evaluation/benchmark-protocols.en.md). The code generates test fixtures; no model-evaluation results are invented.
- **Post-training:** [DAPO](../05-post-training/dapo.en.md), [GSPO / ASPO](../05-post-training/policy-ratios.en.md), [SAO / asynchronous training](../05-post-training/async-policy-learning.en.md), [clipping variants](../05-post-training/clipping-variants.en.md), plus [zero loss with a nonzero gradient](../05-post-training/after-ppo.en.md#zero-loss-gradient), [why an out-of-range ratio can still have a gradient](../05-post-training/rlhf/ppo-clipping.en.md#clipped-token-gradients), and [when a batch is generated, reused, and refreshed](../05-post-training/rlhf/on-off-policy.en.md#grpo-data-lifecycle). Earlier claims that clipping hard-constrains probabilities or that a Critic only supplies a baseline were corrected; the papers' full training results have not been reproduced.
- **Retrieval:** [RAG evidence and evaluation](../04-search/rag-evidence.en.md), [IVF / PQ indexes](../04-search/vector-indexes.en.md), [BM25 versus TF-IDF](../04-search/tfidf-and-bm25.en.md) (2026-10-09), [Qwen3 Embedding versus BGE-M3](../04-search/embedding-models.en.md) (2026-10-09), and [InfoNCE versus CE](../04-search/dual-encoder.en.md#infonce-and-ce) (2026-10-09).
- **Agents:** [agent execution](../10-agents/patterns.en.md), [prompting](../10-agents/prompting.en.md), [deep research](../10-agents/deep-research.en.md), [tools, MCP, and skills](../10-agents/tools-and-skills.en.md), and [model selection and routing](../10-agents/model-choice.en.md). Runnable local examples are not model comparisons or serving benchmarks.
- **Model families:** [GPT](../00-foundations/model-families/gpt.en.md) (including gpt-oss's disclosed training stages and cost accounting), [Llama](../00-foundations/model-families/llama.en.md), [Qwen1–2.5](../00-foundations/model-families/qwen-early.en.md), [Qwen](../00-foundations/model-families/qwen.en.md) (updated to the public 3.8 models), [DeepSeek](../00-foundations/model-families/deepseek.en.md), and [DeepSeek-V4](../00-foundations/model-families/deepseek-v4.en.md). Undisclosed training recipes are not filled in.
- **Architecture components and long context:** [BERT from inputs to fine-tuning](../00-foundations/core/bert.en.md), the [Gated DeltaNet derivation](../00-foundations/deep-dives/gated-deltanet.en.md), [Gated Attention](../00-foundations/deep-dives/gated-attention.en.md), [Engram](../00-foundations/deep-dives/engram.en.md), [Block AttnRes](../00-foundations/deep-dives/attention-residuals.en.md), and [Kimi K3 / NoPE](../00-foundations/deep-dives/nope-and-order.en.md) (2026-10-09); DCA and DSA examples extend the context and sparse-attention notes. Full training is not claimed to be reproduced.
- **MoE:** [routing](../00-foundations/moe/router.en.md#dispatch-example) with dispatch, weighted accumulation, and gradient checks; [load balancing](../00-foundations/moe/load-balancing.en.md#sequence-balance) shows why batch balance differs from sequence balance.
- **Training engineering and multiple GPUs:** [distributed training](../06-systems/distributed-training.en.md), the [two-GPU layer example](../06-systems/tensor-parallel.en.md), the [eight-GPU deployment comparison](../06-systems/distributed-training.en.md#eight-gpus), the [communication ledger](../06-systems/distributed-training.en.md#zero-communication), [memory accounting](../00-foundations/deep-dives/precision-and-memory.en.md) with its [recomputation example](../00-foundations/deep-dives/precision-and-memory.en.md#recompute-example), [the CPU-to-GPU path](../00-foundations/pytorch/training-loop.en.md), and [checkpoint recovery](../practice/post-training/checkpoint-and-resume.en.md). The matching reference articles and the ZeRO stages article have been read; CPU checks do not establish GPU memory use or communication performance.
- **Inference:** [the request lifecycle](../06-systems/llm-serving.en.md), [a complete memory budget](../00-foundations/deep-dives/kv-cache-and-inference.en.md#inference-budget), [cache lifetime](../00-foundations/deep-dives/attention-kernels.en.md#cache-lifecycle), and [DFlash versus MTP](../00-foundations/deep-dives/dflash.en.md) with its [features and shapes](../00-foundations/deep-dives/dflash.en.md#feature-shapes) (2026-10-09). The reference articles on request handling and inference memory have been read; implementation links are version-pinned, and capacity arithmetic is not a GPU measurement.
- **Engineering practice:** [tracing a failure through one request](../practice/post-training/experiments-and-release.en.md#trace-a-failure), [GRPO/PPO for long tasks](../05-post-training/deep-rl/llm-bridge.en.md#long-horizon-choice), [student sampling in distillation](../05-post-training/distillation.en.md#opd-choices), and [rollouts as training data](../05-post-training/post-training-infrastructure.en.md#rollout-record).
- **Algorithms and Python:** after reading Appendix 2's algorithm index, six bilingual notes with local tests on string matching, monotonic queues, BSTs, knapsack, and sequence / state-machine DP; [primality and sieves](../interview/algorithms/primes-and-sieves.en.md) (2026-10-09); and [Python names, copies, and function calls](../interview/python-objects.en.md) in the coding route.

Reference code has not been executed, and image-based equations, embedded PDFs, and linked videos have not all been checked. Site-wide currency and article-by-article technical reviews are still incomplete; a successful build does not establish content completeness.

## Additional material retained here

The reference is not a limit on this site. RNNs / BPTT, the Deep RL route, looped Transformers, evaluation calibration, tools and memory, system design, public recommender-system walkthroughs, Python, and algorithm practice remain available.

This reorganization changes classification and entry points, not public article addresses. Existing examples, code, and diagrams are not replaced with summaries. Previously retired Quant and discussion sections are not republished in this update.

## How the comparison will continue

Continue through the reference in the browser, recording prose, image-based equations, and code separately. Compare definitions, derivations, examples, and implementation details. Fill gaps in existing chapters where possible; create a new article when it answers a distinct question, rather than duplicating another site's table of contents.

To report a gap, include the topic, a public source, and the question you want explained in an issue. Maintenance notes live in `CONTENT_COVERAGE.md`. The main mapping comes from `site/reference-coverage.toml`; the screenshot and browser-directory appendix inventory comes from `site/appendix-coverage.toml`.
