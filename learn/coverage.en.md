# Content coverage: what is here, and what is missing?

[中文](coverage.md) · **English**

This list prevents two mistakes: losing existing material during reorganization, and treating a mentioned term as a completed explanation. It is not a completion leaderboard.

**As of 2026-10-09, coverage is still incomplete.** The main list has 82 topics; identified appendix entries have grown from the 36 visible in screenshots to 67. These lists overlap and must not be added into an article count. Some explanations already exist but are hard to find; some topics are only mentioned; others are missing. The audit distinguishes these cases.

For studying rather than auditing, start with the [reading routes](README.en.md). This page is a gap list, not a question bank you must finish in order.

This update organizes and submits the existing material, with further work to follow. Partial explanations and unfinished checks stay visible below; passing builds, links, and numerical tests does not mean every article has completed technical review.

## How to read the table

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

Reference code has not been executed, and image-based equations, embedded PDFs, and linked videos have not all been checked. Appendix 2's index has been read; six bilingual notes and local tests now cover string matching, monotonic queues, BSTs, knapsack, and sequence / state-machine DP. Eight individual Appendix 1 articles have been read, including KV-cache estimation and K3 / NoPE; the remainder is not complete. Explanations, examples, and code are written independently, with claims checked against papers or official implementations. Site-wide currency and article-by-article technical reviews are still incomplete. A successful build does not establish content completeness.

<span id="appendix-questions"></span>

## Appendix questions: checking individual gaps {#appendix-question-map}

Reopening the reference directory revealed **31 more entries**, bringing the inventory to **67**, including the algorithm and PyTorch appendix entries. This is neither the size of the whole library nor a claim to have read 67 articles. Source reading and site coverage are recorded separately; image equations and external links are not automatically counted as verified.

The useful gaps are not limited to new model names:

| Finding | Examples | What follows |
| --- | --- | --- |
| Substantive explanations already exist | Online softmax, MHA implementation, RoPE, BF16/FP16, normalization axes, entropy/CE/KL | Link the right sections and compare additional source questions |
| Related material needs a more direct explanation | Sequence-level MoE balancing, initial zero GRPO loss, clipped-token gradients, rejection sampling, entropy collapse | Add the missing derivation, example, or diagnostic rather than another definition |
| Previously only named; now covered in dedicated notes | [BM25 versus TF-IDF](../04-search/tfidf-and-bm25.en.md), [primality and sieves](../interview/algorithms/primes-and-sieves.en.md) | Calculations, proofs, and boundaries are present; reference prose still needs comparison |
| Exact model confirmed and dedicated coverage added | [Kimi K3 / NoPE](../00-foundations/deep-dives/nope-and-order.en.md) | From swapped inputs to matrix recurrence; distinguish no RoPE from no long-context training |

For example, the distillation article already computes entropy, cross-entropy, and KL on the same distributions, so it should not be reported as missing. BM25 previously appeared only in the hybrid-retrieval article; this batch adds a dedicated scoring explanation. That is the distinction an item-level audit should make.

The 2026-10-09 additions cover [Muon](../00-foundations/deep-dives/muon.en.md), [DFlash versus MTP](../00-foundations/deep-dives/dflash.en.md), BM25 / TF-IDF, primality / sieves, [Qwen3 Embedding versus BGE-M3](../04-search/embedding-models.en.md), [InfoNCE versus CE](../04-search/dual-encoder.en.md#infonce-and-ce), and NoPE with sequence order.

The inventory now has **24 dedicated explanations and 43 entries still needing expansion or comparison**. No listed topic currently lacks dedicated material entirely; that does not mean the release audit is complete. Eight Appendix 1 articles have been read in full, including K3 in this batch. Other reading states remain evidence-based. Teaching examples are not model measurements, and written articles still need source, detail, and presentation checks.

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
