<div align="center" markdown="1">

<img src="site/static/og.png" alt="AGI Study Notes — from a straight line to a system that reads the world" width="760">

<h1>AGI Study Notes</h1>

**Learning AI together, starting with the basics**

Notes from learning AI, reading papers, and preparing for interviews.<br>
There are fundamentals to work through and interactive diagrams to try. Questions and discussions are welcome.

[![read online](https://img.shields.io/badge/read-cooking--agi-E8A672?style=flat-square)](https://tianyi-zhang-02.github.io/cooking-agi/index.en.html)
[![build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)
[![bilingual](https://img.shields.io/badge/English-中文-8a8a8a?style=flat-square)](README.md)

[**Start reading**](https://tianyi-zhang-02.github.io/cooking-agi/index.en.html) ·
[中文](README.md) ·
[Behind the notes](contributors.en.md) ·
[Join us!](CONTRIBUTING.md)

</div>

---

## Why I write these

I often think I understand a concept until I meet it in a paper or in code and struggle to explain what it is doing. I started taking these notes to connect the pieces as I learn.

My interests are mainly in representation learning, LLM post-training, and model / agent evaluation. When reading a paper or running an experiment, I often ask: what signal did the model learn from? If a score went up, did its actual behavior improve too?

I try to explain the problem with examples before getting into formulas and code. Alongside current methods, I also revisit earlier models to understand how these designs came about.

These notes reflect my understanding so far. Some sections are unfinished, and I will get things wrong. I'll keep updating them, and corrections are welcome.

## Where to start

| I want to… | Go to |
| --- | --- |
| Learn it properly, in order | [Foundations](00-foundations/README.en.md), section by section; the figures in [the Transformer lab](00-foundations/transformer-lab.en.md) are draggable |
| See why model families differ | [Model family deep dives](00-foundations/model-families/README.en.md): Llama, Qwen, DeepSeek and Gemma through one set of questions |
| Prepare for interviews | [Interviews](interview/README.en.md) for a quick review; [Job search](career/README.en.md) is my own record of looking for MLE / RE roles |
| Read a paper properly | [Papers](papers/README.en.md): what it claims, whether the evidence holds, what would overturn it |
| See who wrote this | [Behind the notes](contributors.en.md) |

## What is inside

| Section | About |
| --- | --- |
| [Foundations](00-foundations/README.en.md) | Linear models to Transformers: attention, normalisation, residuals, MoE, looped transformers |
| [Post-training](05-post-training/README.en.md) | SFT, RLHF, PPO and its relatives, and what alignment is actually aligning |
| [Evaluation](07-evaluation/README.en.md) | Whether a metric holds, and how to use an LLM judge without fooling yourself |
| [Data & retrieval](01-data-and-feedback/README.en.md) | Where data comes from, how feedback is collected, how retrieval is built |
| [Systems & multimodal](06-systems/README.en.md) | The path a request really takes, observability, and humans in the loop |
| [Agents](10-agents/README.en.md) | Where the word comes from, the usual structures, the scenarios, and choosing a model |
| [AI infra](open-source/README.en.md) | Contributing to NeMo RL: how scattered PRs grew into understanding a system |
| [Job search](career/README.en.md) | Mindset, what to prepare, and my own timeline and reviews |
| [Papers](papers/README.en.md) | One paper at a time |

The interactive figures — Transformer internals, KV cache, MoE routing, PPO clipping, a Pac-Man maze, model routing — all live in [`site/static/tx-lab.js`](site/static/tx-lab.js), hand-written SVG with no charting library.

## Running the site

Static, with one Python file as the build:

```bash
pip install markdown pygments
python3 site/build.py            # builds to _site/
python3 site/build.py --serve    # local preview
```

Two checks before sending anything:

```bash
python3 site/paritycheck.py      # do the Chinese and English versions still match in structure
python3 site/leakcheck.py        # has anything that should stay private slipped in
```

## Join us!

Learning about these topics too? Come join the conversation! If an explanation is unclear, you have a better example, or you want to share your own notes, we'd love to hear from you. You don't need to have everything figured out first—a question or a small correction can help someone else learn.

Open an issue on GitHub or submit an edit. The [contributing guide](CONTRIBUTING.md) explains how to get started.

Meet the people taking part on the [crew page](https://tianyi-zhang-02.github.io/cooking-agi/contributors.en.html). If you'd like a spot on the map, you can [share your country or region](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml)—no precise address needed.

This is a place for public technical knowledge and learning notes. Please don't upload internal company materials, non-public interview content, or other sensitive information.
