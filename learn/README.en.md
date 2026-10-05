# Study notes: what do you want to work on?

[中文](README.md) · **English**

You don't need to pick a job title first. Start with what you want to do: understand a concept, write code, or work through a design.

<details markdown="1">
<summary>Want to switch languages or compare terms?</summary>

Click **EN / 中文** at the top right to switch directly to the same note in the other language. Chinese pages include English terms by default, such as 注意力（attention）, at their first mention rather than repeating them throughout. Model names, formulas, and code stay unchanged.

Concept cards with a language button let you switch that explanation in place. Chinese pages also offer tappable term definitions and a glossary at the end. The English version is written to stand on its own; it does not require reading the Chinese page first.

</details>

## 01 · Foundations & models

Understand what a component computes before comparing how different models change it.

- [Math & quant review](../quant/README.en.md): probability proofs, counting, linear algebra, statistics, stochastic processes, numerics, and financial mathematics. A coverage map distinguishes full derivations from introductions.
- [Language-model study guide](../00-foundations/study-guide.en.md): vectors, tokens, and Transformers, with room to revisit missing foundations.
- [Model families](../00-foundations/model-families/README.en.md): read Llama, Qwen, DeepSeek, and Gemma using a shared set of questions.
- [Interactive diagrams](../00-foundations/transformer-lab.en.md): change a parameter and see what happens to the computation.

## 02 · Training & applications

A model's computation is only part of the story. What teaches it, how do we evaluate it, and how does it fit into a task?

| What you want to understand | Start here |
| --- | --- |
| How rewards teach sequential decisions | [Deep RL](../05-post-training/deep-rl/README.en.md): foundations → core algorithms → data and experiments |
| How feedback becomes a training objective | [Data & feedback](../01-data-and-feedback/README.en.md) → [Post-training](../05-post-training/README.en.md) |
| What a higher score actually tells us | [Evaluation](../07-evaluation/README.en.md) → [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.en.md) |
| How models use images, memory, and external information | [Multimodal learning](../03-multimodal-learning/README.en.md) · [Memory](../02-memory/README.en.md) · [Retrieval](../04-search/README.en.md) |
| How an AI request gets completed | [Agents](../10-agents/README.en.md) · [Systems & observability](../06-systems/README.en.md) |

## 03 · Code & exercises

Things you can actually practice, whether or not you're interviewing.

- [Python](../interview/python.en.md): containers, functions, language details, and common mistakes.
- [Algorithm patterns](../interview/leetcode.en.md): transferable Easy / Medium techniques, not a race to solve the hardest problems.
- [ML questions & implementations](ml-exercises/README.en.md): explain the idea, calculate a small example, then implement it.

Try first, check the answer, and change a condition before trying again. Understanding why the code works matters more than memorizing it.

## 04 · System design

[Design exercises](system-design/README.en.md) use original public scenarios: a recommendation feed, knowledge-base answers with citations, and editable long-term memory. Each includes constraints, alternatives, and follow-up questions—not a diagram presented as the only correct answer.

## Follow a question across chapters

The same ideas can be approached through different questions. These are neither required sequences nor career tracks; start with whichever question interests you.

| Your question | A connected reading path |
| --- | --- |
| Why is a model learning so little from so many logs? | [Feedback and objectives](../01-data-and-feedback/feedback-to-objectives.en.md) → [How dual encoders learn](../04-search/dual-encoder.en.md) → [What a score supports](../07-evaluation/metric-robustness.en.md) |
| How do I build an assistant that finds evidence and remembers corrections? | [Hybrid retrieval and reranking](../04-search/hybrid-and-reranking.en.md) → [Updating and forgetting memories](../02-memory/memory-lifecycle.en.md) → [Locating evaluation failures](../07-evaluation/evaluation-stack.en.md) |
| Why does a model perform these computations? | [Attention](../00-foundations/core/multi-head-attention.en.md) → [Implementations](../00-foundations/hand-write-kit.en.md) → [Model-family close readings](../00-foundations/model-families/README.en.md) |
| How do I take an algorithm into an experiment? | [Deep RL](../05-post-training/deep-rl/README.en.md) → [Experimental settings](../05-post-training/deep-rl/experiments.en.md) → [Ablations and slices](../07-evaluation/ablation-and-slices.en.md) |

You do not need to read every equation and code block on the first pass. Follow the example, then return to the derivation. If the concept is familiar, go straight to implementation and failure cases. There are no beginner/expert labels to earn, and no requirement to complete every chapter in order.

## How the sections fit together

- **Study notes:** concepts, code, worked questions, and design exercises.
- **[Industry Practice](../practice/README.en.md):** follow public implementations to see where ideas meet constraints. Start with [Twitter / X recommendations](../practice/recommender-systems/README.en.md).
- **[Career](../career/README.en.md):** experiences, mindset, preparation habits, and decisions.
- **[Papers](../papers/README.en.md):** examine what the original research actually supports.

Choose a direction at the top; the sidebar shows its chapters. “My reading” keeps your saved and recently opened notes. Use the search at the top to find a specific concept.
