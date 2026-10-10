# AGI Study Notes

**Understand the idea. Work through an example. Try the code.**

[**Read the notes →**](https://tianyi-zhang-02.github.io/cooking-agi/) · [中文](README.zh.md) · [Behind the notes](contributors.en.md) · [Contribute](CONTRIBUTING.md#english)

[![Build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg?branch=main)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)

ML and LLM notes in English and Chinese, with worked examples, diagrams, and small experiments. The aim is to spend less time hunting for explanations and more time understanding them—whether you're learning a topic, building something, or preparing for interviews.

The site opens in English on your first visit. **中文 / EN** at the top switches the same note in one click; Chinese notes include English technical terms. Your choice is remembered in this browser. Direct links to a particular language stay in that language.

<span id="_1"></span>

## Start here

Pick what you need today. You don't have to read the whole site in order.

- **[Foundations](learn/README.en.md)** — Understand model components, training, evaluation, and inference. New to language models? Start with the [study guide](00-foundations/study-guide.en.md).
- **[Interview prep](interview/README.en.md)** — Review ML / LLM concepts, practice ML coding and Python, and learn transferable algorithm patterns.
- **[Engineering practice](practice/README.en.md)** — Work through recommendation, RAG, and post-training projects: data, evaluation, checkpoints, and design tradeoffs.
- **[Career notes](career/README.en.md)** — Read about internship and new-grad preparation, decisions, and things I'd do differently.

Follow a small example first. If you want the derivation or implementation details, keep reading or open the optional sections. Diagrams explain the main idea without requiring you to click through every step.

<span id="_2"></span>

## Content map

Three routes through the foundations, depending on what you're curious about:

- **How models work:** [Transformer walkthrough](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.en.html) · [BERT](00-foundations/core/bert.en.md) · [CLIP and multimodal learning](03-multimodal-learning/clip.en.md).
- **How models learn:** [Training basics](learn/pretraining/README.en.md) · [Deep RL](05-post-training/deep-rl/README.en.md) · [Post-training](05-post-training/README.en.md) · [LLM-as-a-Judge](07-evaluation/llm-as-a-judge/README.en.md).
- **How models get used:** [Inference](learn/inference/README.en.md) · [Search and retrieval](04-search/README.en.md) · [Agents and tools](10-agents/README.en.md).

For practice, head to [ML coding](learn/ml-exercises/README.en.md), [Python](interview/python.en.md), [algorithm patterns](interview/leetcode.en.md), or [system design](learn/system-design/README.en.md).

**Still being worked on:** some explanations and source comparisons need more review. The [coverage and backlog](learn/coverage.en.md) separates material that exists from material that has been checked. Small tested examples are not claims of full-model training or benchmark reproduction.

<span id="_3"></span>

## Join us!

Found a mistake or a passage that doesn't make sense? [Open an issue](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml) with the page and what tripped you up. A clearer example is just as welcome as a new chapter.

Small fixes can go straight into a PR. For a new article or navigation change, start with a [content proposal](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) so we don't duplicate work. See the [contributing guide](CONTRIBUTING.md#english), [editorial guide](EDITORIAL.en.md), and [area reviewers](community/README.en.md) for the process. Routine merges need a relevant non-author CODEOWNER review and passing checks.

Please use public sources and your own explanations. No internal company material, non-public interview questions, credentials, or private information. AI-assisted contributions are welcome; check their sources and examples, and disclose the assistance in your PR. Contributors are listed in [Behind the notes](contributors.en.md); sharing a country or region for the map is optional.

<span id="_4"></span>

## Run locally

This repository contains the notes, examples, and static-site source. No database is needed. From the repository root, with Python 3.12 recommended:

```bash
python3 -m pip install markdown pygments numpy
python3 site/build.py --serve
```

Open <http://localhost:8000>. To build without starting a server, run `python3 site/build.py`; output goes to `_site/`.

<details markdown="1">
<summary>Checks before submitting</summary>

```bash
python3 site/collaboration.py
python3 -m unittest discover -s site/tests
python3 site/leakcheck.py
python3 site/paritycheck.py --strict
python3 site/build.py
```

Some experiments also need PyTorch, as documented in their chapters; corresponding tests explicitly skip when it is unavailable. Bookmarks, recently opened notes, and reading positions stay in this browser and are not uploaded.

</details>

<span id="_5"></span>

## Why this exists

These notes started while I was moving from academic research toward industry. My preparation materials were scattered across papers, lectures, documentation, and bookmarks. Putting them together helped me see what I understood and what I still needed to work on.

Much of the collection reflects preparation for MLE and research roles, especially language-model work. It isn't a universal recruiting roadmap or a collection of company-specific interview questions. For SDE, frontend/backend, and infrastructure topics outside our experience, we link to people who know them better.

Hopefully this saves you a little searching—and leaves more time for things outside studying, too.
