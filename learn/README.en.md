# Foundations

<span id="study-notes"></span>

[中文](README.md) · **English**

How does a model learn, and what happens when it generates an answer? These notes cover models, training, evaluation, and inference. Start with the small examples, or go straight to the derivations and implementations you need.

<nav class="study-route reading-entry" aria-label="Choose a reading route">
<a href="#start-with-the-foundations"><strong>Start from scratch</strong><span>Follow a computation, then a training update.</span></a>
<a href="#follow-the-question-you-have"><strong>Follow a question</strong><span>Find the chapters that address your problem.</span></a>
<a href="#all-chapters"><strong>Browse chapters</strong><span>Open the directory and choose a subject.</span></a>
</nav>

For hands-on work, try the experiments alongside the notes, then visit [Engineering practice](../practice/README.en.md) for complete systems. The [reading routes](choose-your-path.en.md) can help you choose a starting point. Reviewing for an interview? Go to [Interview prep](../interview/README.en.md).

## Start with the foundations

The concepts are arranged as seven tracks. Each track is a line of stops in reading order, and each stop explains one idea; the top of every stop says which track it is on, what to read first, and where to go next. If vectors, losses, and gradients are still unfamiliar, start at the first stop of the first track, [linear models to neural networks](../00-foundations/from-linear-to-neural.en.md). Pair your reading with the [small PyTorch experiments](../00-foundations/pytorch/README.en.md).

<!-- widget:track-map -->

To connect mathematics with implementation, begin with [matrix gradients and training systems](../00-foundations/deep-dives/matrix-gradients-to-systems.en.md), then follow the [one-question reading route](../practice/runtime-reading/README.en.md#small-steps) into attention, multiple GPUs, inference, and NeMo RL.

<span id="follow-a-question-across-chapters"></span>

## Follow the question you have

Find the question closest to yours in the left column and read the route in the middle column. The right column lists the details worth watching as you read.

| What you want to understand | Reading route | What to inspect |
| --- | --- | --- |
| FSDP, vLLM, and SGLang are familiar names, but their roles are unclear | [Training stacks](../06-systems/training-frameworks.en.md) → [Inference engines](../06-systems/inference-frameworks.en.md) | Who handles training state, request scheduling, and kernels |
| How does a model become a complete product? | [ML system map](../practice/ml-system/README.en.md) → [Data to model](../practice/ml-system/data-to-model.en.md) → [Serving and feedback](../practice/ml-system/serving-and-feedback.en.md) | Inputs, outputs, versions, failures, and team handoffs |
| The equations make sense, but training code does not | [Tensors and storage](../00-foundations/pytorch/tensors-and-storage.en.md) → [Autograd](../00-foundations/pytorch/autograd.en.md) → [Training loop](../00-foundations/pytorch/training-loop.en.md) | Whether parameters update and validation loss uses the right denominator |
| What to prepare before fine-tuning | [Pretraining data pipeline](../00-foundations/deep-dives/pretraining-pipeline.en.md) → [SFT](../05-post-training/sft-and-its-ceiling.en.md) → [LoRA / QLoRA](../05-post-training/lora-and-qlora.en.md) | Labels, loss masks, data splits, and memory use |
| New architecture names are hard to tell apart | [Component reading map](../00-foundations/deep-dives/README.en.md) → [Model reports](../00-foundations/model-families/README.en.md) | Changes to sequences, caches, experts, or network depth |
| The model accepts an image, but does it use it? | [Image–text matching](../03-multimodal-learning/clip.en.md) → [Visual features in an LLM](../03-multimodal-learning/vision-to-language.en.md) → [VLM training and evaluation](../03-multimodal-learning/vlm-finetuning.en.md) | Whether the answer follows a changed fact in the image |
| RL equations make sense, but experiments are unstable | [Value estimates and GAE](../05-post-training/deep-rl/actor-critic-gae.en.md) → [PPO clipping](../05-post-training/rlhf/ppo-clipping.en.md) → [Debugging experiments](../05-post-training/deep-rl/experiments.en.md) | What changed in rewards, advantages, and update size |
| Retrieval scores improve, but answers do not | [Dual encoders](../04-search/dual-encoder.en.md) → [Hybrid retrieval and reranking](../04-search/hybrid-and-reranking.en.md) → [Metric robustness](../07-evaluation/metric-robustness.en.md) | Which examples improve when the candidate pool is held fixed |
| What changes when I replace an embedding model? | [Qwen Embedding and BGE-M3](../04-search/embedding-models.en.md) → [Vector indexes](../04-search/vector-indexes.en.md) | Input and pooling compatibility, and when to rebuild the index |
| `nvidia-smi` shows 100%, yet generation is slow | [AI infrastructure](../12-infra/README.en.md) → [GPU architecture](../12-infra/gpu-architecture.en.md) → [Performance: roofline and MFU](../12-infra/performance.en.md) | Operations per byte moved from memory, and whether bandwidth or compute is the limit |
| How do cars and robots get from pixels to actions? | [Perception and geometry](../13-embodied/perception/README.en.md) → [Autonomous driving](../13-embodied/driving/README.en.md) → [Robot learning](../13-embodied/robot-learning/README.en.md) | What each step hands to the next |
| What do “world model” and “self-improvement” actually mean? | [Frontier](../11-frontier/README.en.md) → [World models](../11-frontier/world-models/README.en.md) → [Recursive self-improvement](../11-frontier/self-improvement/README.en.md) | Which of several things the word refers to, and how far the evidence goes |

<span id="area-coding"></span>
<span id="subject-coding"></span>
<span id="subject-algorithms"></span>
<span id="subject-ml-exercises"></span>
<span id="subject-system-design"></span>

Practice lives in [Interview preparation](../interview/README.en.md): ML / LLM questions, ML coding, Python, and LeetCode. System design and project walkthroughs live in [Engineering practice](../practice/README.en.md). This page keeps the explanations and their small experiments.

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

These are the prerequisites for reading. For example, you can build RAG before studying SAC, and study training before service scheduling.

</details>

<span id="01-foundations-models"></span>
<span id="02-training-applications"></span>
<span id="03-code-exercises"></span>
<span id="04-system-design"></span>

## All chapters

The complete catalog matches the sidebar: the seven tracks above, then **Go deeper** (model reports, components, MoE, looped models and classic deep RL) and **Still moving** (frontier directions). Click a title for its overview, or expand a chapter to find its articles. Every note that is not a stop on a track is here, at the same address as before.

<!-- widget:study-atlas -->

<span id="how-the-sections-fit-together"></span>

<details markdown="1">
<summary>Earlier additions (October 2026)</summary>

- [Frontier watch](../11-frontier/README.en.md), a new fourth area. The [world models](../11-frontier/world-models/README.en.md) and [recursive self-improvement](../11-frontier/self-improvement/README.en.md) series, along with [test-time compute](../11-frontier/test-time-compute/README.en.md), [agent RL](../11-frontier/agentic-rl/README.en.md), and [interpretability](../11-frontier/interpretability/README.en.md), have all been reviewed.
- [AI infrastructure](../12-infra/README.en.md) (from chips, GPUs, and CUDA up to frameworks and clusters) and [Perception, robots, and driving](../13-embodied/README.en.md): two new blocks.
- Three more series, added 2026-10-16: [scaling laws and hyperparameters](../00-foundations/scaling/README.en.md), on choosing model size, data and learning rate before a run; [diffusion language models](../11-frontier/diffusion-lm/README.en.md) in Frontier watch; and [quantization](../12-infra/quantization/README.en.md) in AI infrastructure.
- Three new deep RL notes: [which model to learn](../05-post-training/deep-rl/model-types.en.md), [combining planning and learning](../05-post-training/deep-rl/planning-and-learning.en.md), and [off-policy evaluation estimators](../05-post-training/deep-rl/ope-estimators.en.md).

These series completed their first primary-source check on 2026-10-21, when the draft labels were removed. Further revisions are recorded in the individual notes.

</details>

## Reading and updates

For how these techniques developed, see the [Tech history](technology-timeline.en.md).

Use the language switch at the top right to open the same article in Chinese or English. Chinese pages retain useful English terminology; English pages contain the full examples, derivations, and code.

Search at the top or press `/` to look up a concept. “My reading” keeps your saved and recently read articles in this browser. Progress is tracked in the [coverage and remaining-work list](coverage.en.md).
