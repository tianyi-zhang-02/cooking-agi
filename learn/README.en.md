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

- [Probability & statistics](../quant/probability/README.en.md): distributions, conditioning, and uncertainty—useful in both ML and quant work.
- [Language-model study guide](../00-foundations/study-guide.en.md): vectors, tokens, and Transformers, with room to revisit missing foundations.
- [Model families](../00-foundations/model-families/README.en.md): read Llama, Qwen, DeepSeek, and Gemma using a shared set of questions.
- [Interactive diagrams](../00-foundations/transformer-lab.en.md): change a parameter and see what happens to the computation.

## 02 · Training & applications

A model's computation is only part of the story. What teaches it, how do we evaluate it, and how does it fit into a task?

| What you want to understand | Start here |
| --- | --- |
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

## How the sections fit together

- **Study notes:** concepts, code, worked questions, and design exercises.
- **[Industry Practice](../practice/README.en.md):** follow public implementations to see where ideas meet constraints. Start with [Twitter / X recommendations](../practice/recommender-systems/README.en.md).
- **[Career](../career/README.en.md):** experiences, mindset, preparation habits, and decisions.
- **[Papers](../papers/README.en.md):** examine what the original research actually supports.

Choose a direction at the top; the sidebar shows its chapters. “My reading” keeps your saved and recently opened notes. Use the search at the top to find a specific concept.
