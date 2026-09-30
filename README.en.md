# AGI Study Notes
**ML fundamentals, language models, and engineering practice**

[**Read online →**](https://tianyi-zhang-02.github.io/cooking-agi/index.en.html) · [中文](README.md) / **English** · [Learning paths](00-foundations/study-guide.en.md) · [Contribute](CONTRIBUTING.md#english)

[![Build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg?branch=main)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)

A place to connect scattered resources: start with a question, work through a diagram or small experiment, and explain why a model is designed that way, where it helps, and where it falls short.

This bilingual collection is growing into a community-maintained project. The aim is to spend less time searching for material, more time understanding it—and leave a little more room for life outside preparation.

## Start here

| What you want to do | Where to go |
| --- | --- |
| Build a foundation | [Study and review guide](00-foundations/study-guide.en.md) → [Language-model learning map](00-foundations/README.en.md) |
| Try something interactive | [Transformer walkthrough](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.en.html) · [CLIP alignment](03-multimodal-learning/clip.en.md) |
| Prepare for an ML internship or new-grad role | [Technical preparation](interview/README.en.md) · [Job-search notes](career/README.en.md) |
| Read a paper or add a note | [Papers](papers/README.en.md) · [Editorial guide](EDITORIAL.en.md) · [Content proposals](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) |

**Read on the website for the full experience.** Language switching, interactive diagrams, and review cards work there. This repository contains the notes, experiments, and site source.

## Content map

Study notes have separate **Quant Researcher** and **AI / ML Engineer** routes. Industry practice, career notes, and papers have their own entry points rather than being forced into one curriculum.

| Area | Topics and links |
| --- | --- |
| Probability and foundations | [Probability, distributions, conditioning](quant/probability/README.en.md) · [Tokens, vectors, Transformers](00-foundations/README.en.md) |
| Language models and multimodal learning | [Model families](00-foundations/model-families/README.en.md) · [MoE](00-foundations/moe/README.en.md) · [CLIP and visual language models](03-multimodal-learning/README.en.md) |
| Training and evaluation | [Post-training](05-post-training/README.en.md) · [Evaluation and LLM-as-a-judge](07-evaluation/README.en.md) |
| Data, memory, and retrieval | [Feedback and objectives](01-data-and-feedback/README.en.md) · [Memory](02-memory/README.en.md) · [Retrieval](04-search/README.en.md) |
| Agents and systems | [Agent patterns](10-agents/README.en.md) · [Observability and human involvement](06-systems/README.en.md) · [Model experience](08-model-experience/README.en.md) · [Personal AGI](09-personal-agi/README.en.md) |
| Putting ideas to work | [Industry practice](practice/README.en.md) (in progress) · [Interview preparation](interview/README.en.md) · [Career](career/README.en.md) · [Papers](papers/README.en.md) |

Notes aim to follow **question → example and diagram → mechanism → check → trade-offs → review**. A formula is not the finish line: explain its assumptions and test it on a small example.

## Join us!

You do not need to write a chapter. A correction, a clearer diagram, or a question about an unclear passage can help the next reader.

- **Questions and corrections:** [open an issue](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml) with the page and passage.
- **New content:** send small fixes as PRs; propose new articles or navigation changes in a [content proposal](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) to avoid duplicate work.
- **Reviews:** see [area contacts and review rules](community/README.en.md). Routine merges require a relevant non-author CODEOWNER review and passing checks.
- **Contributors:** [Behind the notes](contributors.en.md) recognizes participation. Map locations are optional countries or regions, never precise addresses.

[Contributing guide](CONTRIBUTING.md#english) · [Editorial guide](EDITORIAL.en.md)

Share public knowledge, reproducible examples, and public-project notes only. No internal company material, non-public interview questions, credentials, or other people's private information. AI assistance is welcome with source checks, validation, and disclosure in the PR.

## Run locally

The static site builds with Python and needs no database. Run these commands from the repository root; Python 3.12 is recommended:

```bash
python3 -m pip install markdown pygments
python3 site/build.py --serve
```

Open <http://localhost:8000>. For a build without the preview server, run `python3 site/build.py`; output goes to `_site/`.

<details markdown="1">
<summary>Checks before submitting</summary>

```bash
python3 site/collaboration.py
python3 -m unittest discover -s site/tests
python3 site/leakcheck.py
python3 site/paritycheck.py
python3 site/build.py
```

Some teaching experiments also need NumPy or PyTorch, as documented in their chapters. Review progress stays in the current browser and is not uploaded.

</details>

## Why this exists

These notes started during a transition from academic research into industry. There was plenty to learn, but connecting the ideas and explaining them in interviews or projects took preparation and a few wrong turns.

The collection is most relevant to ML, language models, and MLE / Research Scientist paths, and is also for anyone curious about AI. It is not a universal recruiting roadmap or a company-specific question bank. For SDE, frontend/backend, and unfamiliar infrastructure topics, we point to more experienced authors.

Some sections are incomplete, and explanations will keep changing. Questions, corrections, and suggestions are welcome.
