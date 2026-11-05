# Core chapters for ML / LLM interviews

[中文](core-chapters.md) · **English**

You may not have time to read every note before an interview. Start with a few core ML / LLM chapters, then choose a path through model training, inference systems, or applied ML. Each link takes you to the full explanation, with derivations, examples, and code where relevant.

First confirm the format of your next round. For an algorithms round, go directly to [Python and LeetCode](leetcode.en.md). For help allocating time, use [quick review](speedrun.en.md). This page answers a different question: which chapter should I read, and what should I be able to do afterward?

## Connect these six ideas first {#core}

This is not a common question bank for every role. The starting set leans toward model-focused Machine Learning Engineer (MLE) and Research Engineer (RE) work, from a parameter update through Transformers, generation, and evaluation. For familiar topics, try the right-hand column before opening the chapter.

| What to understand | Start with this chapter | Try this afterward |
| --- | --- | --- |
| How a matrix gets gradients and an update | [Matrix gradients to training systems](../00-foundations/deep-dives/matrix-gradients-to-systems.en.md) | Calculate a linear layer's output, dW, dX, and one SGD step; explain why backward needs its forward input |
| What happens during one training step | [The PyTorch training loop](../00-foundations/pytorch/training-loop.en.md) | Write the sequence of zeroing, forward, loss, backward, update, and validation; check accumulation and the loss denominator |
| How tokens exchange information | [Multi-head attention](../00-foundations/core/multi-head-attention.en.md) | Label Q/K/V and attention-score shapes; explain scaling, the causal mask, and softmax |
| How a model generates one token at a time | [One generation: prefill, decode, and caching](../practice/runtime-reading/prefill-decode.en.md) | Trace four input tokens and three output tokens; distinguish sampling a token from writing its KV |
| Why better training results can mean worse results on new data | [Generalization and diagnosis](../00-foundations/deep-dives/generalization.en.md) | Distinguish overfitting, leakage, and distribution shift; choose a check for each |
| How to tell whether a change helped | [Evaluation foundations](../07-evaluation/README.en.md) | Explain a project's data split, baseline, metrics, important failure cases, and comparison conditions |

If the gradient example is still difficult, begin with [linear models to neural networks](../00-foundations/from-linear-to-neural.en.md), then return to the matrix calculation. To connect the concepts to code, follow [math → models → systems → NeMo RL](../practice/runtime-reading/README.en.md#small-steps) in sections rather than trying to read a framework all at once.

### Can you use what you just read? {#practice}

Choose a concept and try three things:

- **Change a small condition.** If two samples have different numbers of valid tokens, does your loss average still mean what you intend? Work through [training stability](../practice/runtime-reading/training-stability.en.md), rather than only remembering that denominators matter.
- **Write a test that could catch an error.** For causal attention, change a future token and check earlier outputs. [ML coding](../learn/ml-exercises/README.en.md) has implementation exercises; get a small function working before adding padding and caching.
- **Explain it aloud in the interview language.** For example, start with “The weight gradient sums contributions from every sample that uses the same weights,” then write the shapes and reduction axis. The point is to explain the calculation, not memorize the sentence.

You do not need every model family's configuration or every framework class name at this stage. Next, choose the group that matches the work you want to do.

## Model training, post-training, and research {#training}

| Order | Selected chapters | Questions to follow up |
| --- | --- | --- |
| Align data and objectives | [Data and training objectives](../practice/post-training/data-and-objectives.en.md) | Which positions contribute to loss? Where do samples come from? How do training, validation, and live requests differ? |
| Compare methods | [SFT](../05-post-training/sft-and-its-ceiling.en.md) → [RL and preference methods](../05-post-training/deep-rl/methods-map.en.md) | Why choose SFT, DPO, PPO, or GRPO? How do supervision, sampling cost, and updates differ? |
| Add distributed training when relevant | [Training frameworks](../06-systems/training-frameworks.en.md) → [One update across devices](../06-systems/megatron-basics.en.md) | Where is state stored? What must be communicated? What does saving memory cost elsewhere? |
| Design experiments and debug | [Training stability](../practice/runtime-reading/training-stability.en.md) → [Experiments and release](../practice/post-training/experiments-and-release.en.md) | If loss or reward improves, which capabilities might stay unchanged or regress? What controls would reveal that? |

**Try this discussion:** for your own task, compare the unchanged model with SFT. Specify data, an evaluation set, costs, and release conditions. Explain what result would make you decide against fine-tuning. For RL work, add rollouts, rewards, policy updates, and stale samples, then continue to the [NeMo RL source walkthrough](../practice/runtime-reading/nemo-rl.en.md).

For Research Scientist (RS) roles, return to your own field: choose a concrete hypothesis, explain the evidence and failures, and propose the next experiment. This LLM route does not replace domain knowledge in vision, speech, or robotics.

## Inference, serving, and ML systems {#systems}

Continue from the generation example above, rather than starting by memorizing framework names.

| Order | Selected chapters | Questions to follow up |
| --- | --- | --- |
| Estimate resources | [KV cache and inference costs](../00-foundations/deep-dives/kv-cache-and-inference.en.md) | How much memory goes to weights, KV, and temporary tensors? What grows when batch size or context length changes? |
| Follow request scheduling | [LLM serving](../06-systems/llm-serving.en.md) | How do long inputs coexist with active decoding? What are the tradeoffs among throughput, first-token latency, and tail latency? |
| Choose a framework and its integrations | [vLLM / SGLang](../06-systems/inference-frameworks.en.md#ecosystem) | Do models, hardware, backends, observability, and deployment tools fit? Which differences need measuring on your workload? |
| Locate bottlenecks with measurements | [Performance and profiling](../12-infra/performance.en.md#profiling) | Is time spent queueing, computing, accessing memory, or communicating? What should the comparison hold fixed? |

**Try this design problem:** a document QA service starts accepting long inputs, and users report slower first responses. Draw the request path and identify missing measurements. Compare scheduling changes, more replicas, or prefill/decode disaggregation rather than immediately proposing a different engine.

For training systems, shift attention to gradient reduction, training state, and recovery. For kernel work, continue into GPU memory hierarchy and operator implementation. Similar systems titles can require very different preparation.

## Applied MLE: retrieval, recommendation, and complete systems {#applications}

Not every design discussion should become an LLM-serving discussion. Choose the path closest to your projects or target role:

| Direction | Reading sequence | What to practice |
| --- | --- | --- |
| Retrieval / RAG | [Dual-encoder retrieval](../04-search/dual-encoder.en.md) → [Hybrid retrieval and reranking](../04-search/hybrid-and-reranking.en.md) → [Cited QA design](../learn/system-design/rag.en.md) | Evaluate retrieval, ranking, and answers separately; construct a case that locates the failing stage |
| Recommendation / ranking | [Recommender-system walkthroughs](../practice/recommender-systems/README.en.md) → [Feed design](../learn/system-design/feed.en.md) | How do labels, exposure and sampling, feature freshness, and offline versus online metrics affect model choices? |
| End-to-end ML systems | [A complete ML system](../practice/ml-system/README.en.md) → [Serving and feedback](../practice/ml-system/serving-and-feedback.en.md) | Trace data to predictions and feedback; explain interfaces, versions, failures, and modules owned by other teams |

For multimodal roles, continue with [visual features entering an LLM](../03-multimodal-learning/vision-to-language.en.md) and [VLM fine-tuning and evaluation](../03-multimodal-learning/vlm-finetuning.en.md). Connect inputs, fusion, objectives, and evaluation before going deep into a particular model.

## Leave time for coding and your own projects {#coding-and-projects}

**Practice algorithms and code reading separately.** The [LeetCode pattern map](leetcode.en.md) organizes reusable methods; [code comprehension](code-comprehension.en.md) practices tracing and debugging existing implementations. Explaining attention does not establish that you can handle algorithmic edge cases, and solving algorithms does not establish that you can find a training-loop bug.

**Connect the concepts through a real project.** Use [project preparation](../career/prepare.en.md) to explain the problem, your contribution, alternatives, evidence, and limitations. If asked why you used two-stage retrieval, connect the mechanism to your actual latency budget, data, and measurements. Say when something was not measured rather than inventing a plausible number.

These chapters organize knowledge; they do not predict a company's interview scope. Save individual notes to return to them later. They retain their original links and use the existing “My reading” feature; this index does not create another progress tracker.
