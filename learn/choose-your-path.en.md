# Find your starting point

[中文](choose-your-path.md) · **English**

Pick what you'd like to do next: understand a model, practice for an interview, read research, build something, study one topic, or get ready to apply.

<nav class="study-route goal-route" aria-label="Choose a route by goal">
<a href="#getting-started"><small>Build an understanding</small><strong>I'm new to ML / LLMs</strong><span>From a prediction problem to a Transformer</span></a>
<a href="#interviews"><small>Prepare for interviews</small><strong>I need review and coding practice</strong><span>Concepts, ML coding, and algorithms need different practice</span></a>
<a href="#research"><small>Read research</small><strong>I want to understand papers and experiments</strong><span>Mechanisms, comparisons, reproduction, and limits</span></a>
<a href="#projects"><small>Build something</small><strong>I want a small working project</strong><span>Get it running, then examine the design choices</span></a>
<a href="#topics"><small>Follow an interest</small><strong>I only need one topic</strong><span>Post-training, evaluation, RAG, multimodal models, inference…</span></a>
<a href="#applications"><small>Plan the job search</small><strong>I need to decide what to prepare</strong><span>Role choices, project discussions, and interview reflections</span></a>
</nav>

Each route has a starting point and somewhere you can stop for now. Skip what you know and fill in gaps as needed. These notes are closest to ML / LLM learning and MLE / research preparation, not a universal interview syllabus.

## Getting started: follow one computation {#getting-started}

A little Python is enough to begin. If vectors and gradients are unfamiliar, start with [linear models and neural networks](../00-foundations/from-linear-to-neural.en.md). You don't need to finish all the math first.

1. [Tokenization](../00-foundations/core/tokenization.en.md): how a sentence becomes model input.
2. [Transformer walkthrough](../00-foundations/transformer-lab.en.md): follow a short input through attention, FFNs, and residual connections.
3. [Training basics](pretraining/README.en.md): how a prediction error becomes a parameter update.

**A useful stopping point:** explain the input, prediction, loss, and update for a short sentence. You don't need every matrix equation memorized. To check your understanding in code, try the [PyTorch exercises](../00-foundations/pytorch/README.en.md). Model families, MoE, and distributed training can wait.

## Interviews: find the part that needs practice {#interviews}

In [Interview prep](../interview/README.en.md), pick the area you need. With a background, go straight to the [quick review](../interview/speedrun.en.md); to build up step by step, use the [study plan](../interview/study-plan.en.md). The first narrows the scope; the second follows the prerequisites. Both let you start with what you know. Suggested hour budgets are not deadlines.

Start with the job description and any interview information you've received. Two MLE roles may emphasize quite different things. For what to practice first by role, and what to leave when time is short, see the [role-based guide](../interview/ai-roles-prep.en.md#routes). Without a particular interview coming up, try one concept question and one small implementation to see where you get stuck.

1. **Hard to explain the idea?** Use the [ML / LLM questions](../interview/basics/README.en.md). Answer without notes, then explain with an example.
2. **Understand it but can't implement it?** Try [ML coding](ml-exercises/README.en.md). Check shapes, masks, gradients, and boundary cases.
3. **Slow to code or unsure how to approach a problem?** Start with [Python](../interview/python.en.md) and [algorithm patterns](../interview/leetcode.en.md). Practice methods rather than memorizing problem numbers.

**A useful stopping point:** solve a small problem independently, explain your choices, and test a different input. Add [Feed / RAG / Memory design](system-design/README.en.md) when you need system design, and [project preparation](../career/prepare.en.md) for project discussions. One interview isn't a reason to complete every route at once.

## Research: work out what actually changed {#research}

This route is for readers who know roughly how a Transformer works and want to read model reports or discuss a research project. Research Scientist and Research Engineer requirements vary by team. This is practice in reading and discussing work, not a promise of meeting a role's requirements.

1. [Architecture components](../00-foundations/deep-dives/README.en.md): identify the component a paper changes, then fill in unfamiliar mechanisms.
2. [Model families](../00-foundations/model-families/README.en.md): compare designs using the same questions rather than collecting version names.
3. [Evaluation basics](../07-evaluation/README.en.md): examine baselines, data, metrics, and ablations to decide what the results support.

**A useful stopping point:** write one page describing the problem, method, evidence, and something you still aren't sure about. Recalculate an example where possible. If you can't reproduce the full model, state which part you checked. For post-training research, connect the [principles](../05-post-training/README.en.md) to [experiment design](../practice/post-training/README.en.md); you don't need to read every model family first.

## Projects: get one path working first {#projects}

If you haven't built a complete application, start with [this site and a reading-list app](../practice/web-lab/README.en.md). No GPU is needed: learn the build and CI/CD process, then connect a page, an API, and a database.

Already comfortable writing programs? Pick the project you want to explore:

- [Recommendation](../practice/recommender-systems/README.en.md): follow a request through retrieval, ranking, filtering, and evaluation.
- [A RAG knowledge base](../practice/rag/README.en.md): connect document versions and permissions to retrieval, citations, and answer checks.
- [Post-training experiments](../practice/post-training/README.en.md): start with an error to fix, then choose SFT, construct batches, consider multiple GPUs, restore checkpoints, and decide whether to release.

**A useful stopping point:** run a minimal example, change one condition, and explain the effect. Remove a supporting document, for example, or interrupt and resume training. For a design interview, try a [system-design problem](system-design/README.en.md) first, then use the implementations to challenge your proposal.

## Only interested in one topic? Start there {#topics}

You don't need to follow the routes above first. These are topic entrances, each with its own reading order—not a required syllabus.

<div class="worked-table" markdown="1">

| Your interest | Start here |
| --- | --- |
| Model fundamentals and architecture | [Core mechanisms](../00-foundations/core/README.en.md) · [Components and model families](../00-foundations/model-families/README.en.md) |
| Data and training | [Data and feedback](../01-data-and-feedback/README.en.md) · [Pretraining and training engineering](pretraining/README.en.md) |
| Reinforcement learning and post-training | [Deep RL](../05-post-training/deep-rl/README.en.md) · [SFT / RLHF / distillation](../05-post-training/README.en.md) |
| Reliable evaluation | [Metrics and experiments](../07-evaluation/README.en.md) · [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.en.md) |
| Connecting images to language models | [Multimodal learning](../03-multimodal-learning/README.en.md) |
| Retrieval and persistent state | [Search and retrieval](../04-search/README.en.md) · [Memory](../02-memory/README.en.md) |
| Tools and agents | [Agents](../10-agents/README.en.md) · [Personal AGI](../09-personal-agi/README.en.md) |
| Inference speed, cost, and reliability | [Inference](inference/README.en.md) · [AI systems](../06-systems/README.en.md) · [Model experience](../08-model-experience/README.en.md) |
| Chips, GPUs, CUDA, and clusters | [AI infrastructure](../12-infra/README.en.md) (draft, pending review) |
| Perception, robots, and driving | [From pixels to actions](../13-embodied/README.en.md) (draft, pending review) |
| Directions still in motion | [Frontier](../11-frontier/README.en.md): world models and recursive self-improvement are reviewed; test-time compute, agent RL, and interpretability are drafts pending review |

</div>

For a particular term, use the search button at the top or press `/`. Try either the Chinese concept or the English term.

## Job-search preparation: decide what to work on next {#applications}

1. [What to prepare](../career/prepare.en.md): allocate effort around the role and your own gaps.
2. [An interview retrospective](../career/one-interview.en.md): record specific problems after an interview, then adjust your preparation.

**A useful stopping point:** pick a near-term goal, identify the gap that matters most, and plan one concrete exercise. For my experience, see the [25–26 internship search](../career/journey.en.md) and [26–27 new-grad search](../career/new-grad-2026-2027.en.md). They're accounts of a search, not a schedule you need to match.

## See the full collection, or come back later {#all-notes}

Pick an area for what you need: [Foundations](README.en.md) explains mechanisms, [Interview prep](../interview/README.en.md) has review and practice, [Engineering practice](../practice/README.en.md) breaks down systems, [Tech history](technology-timeline.en.md) follows how the field developed, and [Career](../career/README.en.md) covers experiences and decisions. A note lives in one place; routes link to it rather than duplicating it.

Use **中文 / EN** at the top to switch the same note. Save a note with **Save this page** under its title, then find it, along with recently opened pages, under **My reading**. Records stay in this browser; export a backup under Settings before moving devices. The collection is still growing. [Coverage and backlog](coverage.en.md) distinguishes what is written from what still needs checking.
