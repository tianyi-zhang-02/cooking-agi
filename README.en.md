> Hopefully these notes save you some time searching for resources, and leave you more time for things you enjoy.

<div align="center" markdown="1">

<img src="site/static/og.png" alt="AGI Study Notes" width="760">

<h1>AGI Study Notes</h1>

Fundamentals I've studied, papers I've read, and notes from preparing for jobs.<br>
Available in Chinese and English, with interactive diagrams to try.

[![read online](https://img.shields.io/badge/read-cooking--agi-E8A672?style=flat-square)](https://tianyi-zhang-02.github.io/cooking-agi/index.en.html)
[![build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)
[![bilingual](https://img.shields.io/badge/English-中文-8a8a8a?style=flat-square)](README.md)

[**Read online**](https://tianyi-zhang-02.github.io/cooking-agi/index.en.html) · [中文](README.md) · [Behind the notes](contributors.en.md) · [Contribute](CONTRIBUTING.md#english)

</div>

## About these notes

Last year, I started exploring industry after spending most of college focused on academic research. Preparing for jobs showed me how much I still needed to learn. After a lot of preparation and a few wrong turns, I began organizing these notes so I could revisit what I'd learned—and hopefully help someone on a similar path.

They cover ML and language model fundamentals, post-training, evaluation, paper reading, and my approach to interview preparation and the job search. **This isn't a collection of company-specific interview questions.** Some sections are unfinished, and I won't get everything right. I'll keep adding to them.

## Who they might help

- People looking for **ML internships or new-grad roles** who want to review the fundamentals and plan their preparation;
- People **moving into ML / LLM work** who aren't sure where to start;
- Anyone curious about how models work, whether or not they're looking for a job.

My preparation and interview experience has mainly been with **MLE and Research Scientist** roles, so I'm not the best source for SDE interview advice. For AI infrastructure and other areas I'm still learning, I'll share resources from people who know them better than I do. This isn't a universal roadmap—use whatever is helpful for you.

## What you can read

The [website](https://tianyi-zhang-02.github.io/cooking-agi/index.en.html) has language switching and working interactive diagrams. If you'd like to browse first, try the [Transformer walkthrough](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.en.html).

| Section | Topics |
| --- | --- |
| [Model fundamentals](00-foundations/README.en.md) | Tokenization, RNNs / LSTMs, Transformers, MoE, and model families |
| [Post-training](05-post-training/README.en.md) | How SFT, RLHF, PPO, and related methods work and when to use them |
| [Evaluation](07-evaluation/README.en.md) | Evaluation design, metrics, and LLM-as-a-judge |
| [Data & retrieval](01-data-and-feedback/README.en.md) | Data, feedback signals, and retrieval |
| [Systems & multimodal](06-systems/README.en.md) | Request handling, debugging, and human involvement |
| [Agents](10-agents/README.en.md) | Common architectures, use cases, and model selection |
| [AI infra](open-source/README.en.md) | What I'm learning through open-source contributions to NeMo RL |
| [Interview preparation](interview/README.en.md) | ML review, coding practice, and system design resources |
| [Job search](career/README.en.md) | Preparation, mistakes, and reflections on the process |
| [Papers](papers/README.en.md) | Reading notes, questions, and experiment ideas |

## Join us!

Found a mistake, an unclear explanation, or a topic you'd like to see? [Open an issue](https://github.com/tianyi-zhang-02/cooking-agi/issues). Your own notes and better examples are welcome as PRs too. I'd love to learn from you.

The [contributing guide](CONTRIBUTING.md#english) explains how to get involved, and contributors appear on the [Behind the notes](contributors.en.md) page. If you'd like a spot on the map, you can [share your country or region](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml)—no precise address needed.

Please share only public technical knowledge and learning notes, not internal company materials, non-public interview content, or other sensitive information.

## Run locally

<details markdown="1">
<summary>Show setup, preview, and check commands</summary>

From the repository root:

```bash
pip install markdown pygments
python3 site/build.py          # Build to _site/
python3 site/build.py --serve  # Preview locally
```

Before submitting changes:

```bash
python3 site/paritycheck.py  # Check Chinese / English structure
python3 site/leakcheck.py    # Check for sensitive content
```

</details>
