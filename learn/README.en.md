# Foundations

<span id="study-notes"></span>

[中文](README.md) · **English**

This section explains how models work, why training methods take the forms they do, and how to judge their results. Start with examples, then explore derivations and implementations as needed. You do not need every prerequisite or every new model; go straight to the subject you care about.

- **New to the subject?** Follow the examples and computations first. Return to derivations on a second pass.
- **Already have the basics?** Start with your question and fill in missing concepts as needed.
- **Ready to build?** Check the implementation assumptions, run a small experiment, then see how the pieces fit in Engineering practice.

<span id="how-the-sections-fit-together"></span>

Recent additions: [Muon](../00-foundations/deep-dives/muon.en.md) explains matrix updates versus AdamW; [DFlash and MTP](../00-foundations/deep-dives/dflash.en.md) covers drafting, verification, acceleration conditions, and DFlash 2 changes. Beginners can skip these initially; readers working on training or inference can go directly to the relevant question.

## Start with the foundations

If vectors, losses, and gradients are still unfamiliar, begin with [linear models to neural networks](../00-foundations/from-linear-to-neural.en.md). Steps 1–3 below explain computation and learning; step 5 covers checking the result. Post-training and inference optimization can wait until you need them. Run the [small PyTorch experiments](../00-foundations/pytorch/README.en.md) alongside the concepts rather than saving all coding for the end.

<nav class="study-route" aria-label="Suggested reading route">
<a href="../00-foundations/core/tokenization.en.md"><small>01 / INPUT</small><strong>From text to vectors</strong><span>Tokens, vocabularies, and sequences</span></a>
<a href="../00-foundations/core/README.en.md"><small>02 / MODEL</small><strong>What a Transformer computes</strong><span>Attention, residuals, norms, and FFNs</span></a>
<a href="pretraining/README.en.md"><small>03 / TRAINING</small><strong>How data changes parameters</strong><span>Objectives, gradients, and an update</span></a>
<a href="../05-post-training/README.en.md"><small>04 / BEHAVIOR</small><strong>Examples, preferences, and rewards</strong><span>SFT, distillation, and reinforcement learning</span></a>
<a href="../07-evaluation/README.en.md"><small>05 / EVALUATION</small><strong>Did anything actually improve?</strong><span>Comparisons, metrics, and concrete errors</span></a>
<a href="inference/README.en.md"><small>06 / GENERATION</small><strong>Computing an answer</strong><span>Sampling, caching, and inference cost</span></a>
</nav>

<span id="follow-a-question-across-chapters"></span>

## Follow the question you have

You do not need to finish an entire subject to resolve one doubt. Start with the first article in a route. If you can answer the question in the last column, that may be enough for now; the remaining links are ways to go further, not a checklist to finish.

| What you want to understand | Reading route | What to inspect |
| --- | --- | --- |
| The equations make sense, but training code does not | [Tensors and storage](../00-foundations/pytorch/tensors-and-storage.en.md) → [Autograd](../00-foundations/pytorch/autograd.en.md) → [Training loop](../00-foundations/pytorch/training-loop.en.md) | Whether parameters update and validation loss uses the right denominator |
| What to prepare before fine-tuning | [Pretraining data pipeline](../00-foundations/deep-dives/pretraining-pipeline.en.md) → [SFT](../05-post-training/sft-and-its-ceiling.en.md) → [LoRA / QLoRA](../05-post-training/lora-and-qlora.en.md) | Labels, loss masks, data splits, and memory use |
| New architecture names are hard to tell apart | [Component reading map](../00-foundations/deep-dives/README.en.md) → [Model reports](../00-foundations/model-families/README.en.md) | Changes to sequences, caches, experts, or network depth |
| The model accepts an image, but does it use it? | [Image–text matching](../03-multimodal-learning/clip.en.md) → [Visual features in an LLM](../03-multimodal-learning/vision-to-language.en.md) → [VLM training and evaluation](../03-multimodal-learning/vlm-finetuning.en.md) | Whether the answer follows a changed fact in the image |
| RL equations make sense, but experiments are unstable | [Value estimates and GAE](../05-post-training/deep-rl/actor-critic-gae.en.md) → [PPO clipping](../05-post-training/rlhf/ppo-clipping.en.md) → [Debugging experiments](../05-post-training/deep-rl/experiments.en.md) | What changed in rewards, advantages, and update size |
| Retrieval scores improve, but answers do not | [Dual encoders](../04-search/dual-encoder.en.md) → [Hybrid retrieval and reranking](../04-search/hybrid-and-reranking.en.md) → [Metric robustness](../07-evaluation/metric-robustness.en.md) | Which examples improve when the candidate pool is held fixed |
| What changes when I replace an embedding model? | [Qwen Embedding and BGE-M3](../04-search/embedding-models.en.md) → [Vector indexes](../04-search/vector-indexes.en.md) | Input and pooling compatibility, and when to rebuild the index |

<span id="area-coding"></span>
<span id="subject-coding"></span>
<span id="subject-algorithms"></span>
<span id="subject-ml-exercises"></span>
<span id="subject-system-design"></span>

**Want to review or solve problems?** [Interview preparation](../interview/README.en.md) separates ML / LLM questions, ML coding, and Python with LeetCode. **Want to design and build a system?** [Engineering practice](../practice/README.en.md) connects design exercises with public projects and teaching implementations. Small experiments remain alongside the concepts here, without putting every exercise in this directory.

<details markdown="1">
<summary>What should I know before focusing on one subject?</summary>

| Subject | Enough background to get started | What can wait on a first pass |
| --- | --- | --- |
| Multimodal models | Vector similarity, attention, cross-entropy | Distributed training and full RL derivations |
| SFT / preference learning | Next-token loss, gradients, data splits | Pretraining from scratch and MoE deployment |
| RL / post-training | Expectations, log-probabilities, basic differentiation | Robotics; return to continuous control when needed |
| Inference / deployment | Decoder forward pass, tensor shapes, KV cache | Reward models and preference optimization |
| Retrieval / RAG / agents | Embeddings, retrieval metrics, basic API calls | Every model family and training recipe |
| Python / algorithms | Functions, loops, lists, dictionaries | Language models; this route stands on its own |

These are not reader rankings. You can build RAG without first studying SAC, or study training before service scheduling. Follow the links when a concrete question calls for the background.

</details>

<span id="01-foundations-models"></span>
<span id="02-training-applications"></span>
<span id="03-code-exercises"></span>
<span id="04-system-design"></span>

## All chapters

The three areas match the sidebar: **understand models → train and evaluate → inference and applications**. These describe subjects, not difficulty levels. Each article has one main home; links connect related subjects without duplicating content.

Open an overview or expand a chapter for all its articles. Model families follow component fundamentals. Questions and coding exercises live in Interview preparation; design exercises and project walkthroughs live in Engineering practice. Switch whenever you want to try an idea, without finishing the whole section first.

<!-- widget:study-atlas -->

## Reading and updates

Use **EN / 中文** at the top right to switch the current article. Chinese pages retain useful English terminology; English pages contain the full examples, derivations, and code.

Search at the top or press `/` for a concept. “My reading” keeps bookmarks and recent articles in this browser only. The [coverage and remaining-work list](coverage.en.md) tracks gaps. Reference material helps identify missing topics, but the reading order here follows the concepts.
