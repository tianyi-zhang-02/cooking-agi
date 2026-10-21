# Find your starting point

[中文](choose-your-path.md) · **English**

Find material by your current goal: getting started, reviewing for interviews, reading papers, building a project, studying one topic, or preparing to apply. Each route lists a starting point and a suggested exercise.

<nav class="study-route goal-route" aria-label="Choose a route by goal">
<a href="#getting-started"><small>Build an understanding</small><strong>I'm new to ML / LLMs</strong><span>From a prediction problem to a Transformer</span></a>
<a href="#interviews"><small>Prepare for interviews</small><strong>I need review and coding practice</strong><span>Concepts, ML coding, and algorithms need different practice</span></a>
<a href="#research"><small>Read research</small><strong>I want to understand papers and experiments</strong><span>Mechanisms, comparisons, reproduction, and limits</span></a>
<a href="#projects"><small>Build something</small><strong>I want a small working project</strong><span>Get it running, then examine the design choices</span></a>
<a href="#topics"><small>Follow an interest</small><strong>I only need one topic</strong><span>Post-training, evaluation, RAG, multimodal models, inference…</span></a>
<a href="#applications"><small>Plan the job search</small><strong>I need to decide what to prepare</strong><span>Role choices, project discussions, and interview reflections</span></a>
</nav>

Skip what you already know. These notes mainly cover ML / LLM and MLE / research topics; other roles need knowledge from their own fields as well.

## Getting started: follow one computation {#getting-started}

A little Python is enough to begin. If vectors and gradients are unfamiliar, start with [linear models and neural networks](../00-foundations/from-linear-to-neural.en.md). You don't need to finish all the math first.

1. [Tokenization](../00-foundations/core/tokenization.en.md): how a sentence becomes model input.
2. [Transformer walkthrough](../00-foundations/transformer-lab.en.md): follow a short input through attention, FFNs, and residual connections.
3. [Training basics](pretraining/README.en.md): how a prediction error becomes a parameter update.

**Try it:** take a short sentence and explain how the input becomes a prediction and how the loss takes part in the parameter update. Then check the computation against the [PyTorch exercises](../00-foundations/pytorch/README.en.md). Model families, MoE, and distributed training can wait.

## Interviews: find the part that needs practice {#interviews}

In [Interview prep](../interview/README.en.md), pick the area you need. With a background, go straight to the [quick review](../interview/speedrun.en.md); to build up systematically, use the [study plan](../interview/study-plan.en.md). The hours in the routes are a guide; split them to fit your schedule.

Start with the job description and any interview information you've received. Two MLE roles may emphasize quite different things. For what to practice first by role, and what to leave when time is short, see the [role-based guide](../interview/ai-roles-prep.en.md#routes). Without a particular interview coming up, try one concept question and one small implementation to see where you get stuck.

1. **Hard to explain the idea?** Use the [ML / LLM questions](../interview/basics/README.en.md). Answer without notes, then explain with an example.
2. **Understand it but can't implement it?** Try [ML coding](ml-exercises/README.en.md). Check shapes, masks, gradients, and boundary cases.
3. **Slow to code or unsure how to approach a problem?** Start with [Python](../interview/python.en.md) and [algorithm patterns](../interview/leetcode.en.md). Practice methods rather than memorizing problem numbers.

**Practice:** solve a problem on your own, explain your approach, then test a few other inputs. For design questions, continue with [Feed / RAG / Memory design](system-design/README.en.md); for project follow-up questions, see [what to prepare](../career/prepare.en.md).

## Research: work out what actually changed {#research}

This route suits readers who know how a Transformer works and want to read papers or model reports. The focus is what the authors changed and what their experiments support; Research Scientist and Research Engineer requirements still depend on the team.

1. [Architecture components](../00-foundations/deep-dives/README.en.md): identify the component a paper changes, then fill in unfamiliar mechanisms.
2. [Model families](../00-foundations/model-families/README.en.md): compare designs using the same questions rather than collecting version names.
3. [Evaluation basics](../07-evaluation/README.en.md): examine baselines, data, metrics, and ablations to decide what the results support.

**Try it:** summarize the problem, method, results, and open questions on one page. Recompute any example you can; if you only checked part of it, say which part. If post-training interests you, continue with the [principles](../05-post-training/README.en.md) and [experiment practice](../practice/post-training/README.en.md).

## Projects: get one path working first {#projects}

If you haven't built a complete application, start with [this site and a reading-list app](../practice/web-lab/README.en.md). No GPU is needed: learn the build and CI/CD process, then connect a page, an API, and a database.

Already comfortable writing programs? Pick the project you want to explore:

- [Recommendation](../practice/recommender-systems/README.en.md): follow a request through retrieval, ranking, filtering, and evaluation.
- [A RAG knowledge base](../practice/rag/README.en.md): connect document versions and permissions to retrieval, citations, and answer checks.
- [Post-training experiments](../practice/post-training/README.en.md): start with an error to fix, then choose SFT, construct batches, consider multiple GPUs, restore checkpoints, and decide whether to release.

**Try it:** run a minimal example, then change one condition: remove the document that supports an answer, for example, or interrupt training and resume it, and watch how the system handles it. For a design interview, try a [system-design problem](system-design/README.en.md) first, then check your proposal against the project implementations.

## Only interested in one topic? Start there {#topics}

The entries below are grouped by topic; each topic has its own, more detailed reading order.

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
| Chips, GPUs, CUDA, and clusters | [AI infrastructure](../12-infra/README.en.md) |
| Perception, robots, and driving | [From pixels to actions](../13-embodied/README.en.md) |
| Directions still in motion | [Frontier](../11-frontier/README.en.md): world models, recursive self-improvement, test-time compute, agent RL, interpretability, and diffusion language models |

</div>

For a particular term, use the search button at the top or press `/`. Try either the Chinese concept or the English term.

## Job-search preparation: decide what to work on next {#applications}

1. [What to prepare](../career/prepare.en.md): allocate effort around the role and your own gaps.
2. [An interview retrospective](../career/one-interview.en.md): record specific problems after an interview, then adjust your preparation.

Start by listing what you need to fill in before your next interview, then plan specific exercises. My own experience is in the [25–26 internship search](../career/journey.en.md) and [26–27 new-grad search](../career/new-grad-2026-2027.en.md), for reference only.

## See the full collection, or come back later {#all-notes}

Everything lives in [Foundations](README.en.md), [Interview prep](../interview/README.en.md), [Engineering practice](../practice/README.en.md), [Tech history](technology-timeline.en.md), and [Career](../career/README.en.md). The routes are only different reading orders that point to the same articles.

Use **中文 / EN** at the top to switch the same note. Save a note with **Save this page** under its title, then find it, along with recently opened pages, under **My reading**. Records stay in this browser; export a backup under Settings before moving devices. The collection is still growing; [coverage and backlog](coverage.en.md) shows what is written and what still needs checking.
