# Cooking AGI

**Bilingual ML / LLM notes: worked examples, step-by-step figures, coding practice and interview prep.**

<!-- repository-only -->

<span id="si-study-notes"></span>
<span id="agi-study-notes"></span>

<p align="center"><a href="https://tianyi-zhang-02.github.io/cooking-agi/index.en.html"><img src=".github/media/hero.en.svg" width="100%" alt="A tour of Cooking AGI: step-by-step figures, the papers' own figures, frontier watch, two languages, interview prep and search"></a></p>

<p align="center"><a href="https://tianyi-zhang-02.github.io/cooking-agi/index.en.html"><b>Read the notes →</b></a> &nbsp;·&nbsp; <a href="README.zh.md">中文</a> &nbsp;·&nbsp; <a href="#contributors">Contributors</a> &nbsp;·&nbsp; <a href="CONTRIBUTING.md#english">Contribute</a></p>

<!-- /repository-only -->

ML and LLM notes in English and Chinese. Each topic is built around a worked example you can follow, run, or replay, with the papers' own figures and sources alongside. Use it to learn from scratch, review before an interview, or look into a field outside your own. **Free to read, and it will stay that way.**

<span id="_1"></span>

## Start here

| If you want to | Start with |
| --- | --- |
| learn ML from the beginning | [Starting out](interview/speedrun-start.en.md) |
| study on a plan | [Study at your pace](interview/study-plan.en.md) |
| prepare for an interview soon | [Quick review](interview/speedrun.en.md) |
| check what you know | [Coding checks](interview/coding-checks.en.md) · [Role-based preparation](interview/ai-roles-prep.en.md) |
| explore a topic | [Choose a topic](learn/choose-your-path.en.md#topics) · [Tech history](learn/technology-timeline.en.md) |

The site has five sections. You don't need to read them in order.

- **[Foundations](learn/README.en.md)** — how models work, learn, and are evaluated.
- **[Interview prep](interview/README.en.md)** — ML / LLM review, ML coding, Python, and algorithm patterns.
- **[Engineering practice](practice/README.en.md)** — recommendation, RAG, and post-training systems, end to end.
- **[Tech history](learn/technology-timeline.en.md)** — power, chips, networks, software, and AI, with original sources.
- **[Career notes](career/README.en.md)** — what I tried in internship and new-grad searches.

<span id="interview-coach"></span>

### Practice with your AI agent

The interview coach skill runs a mock interview against these notes in Claude Code or Codex: one question at a time, follow-ups on your answer, and a local record of what to revisit.

```bash
npx skills add https://github.com/tianyi-zhang-02/cooking-agi --skill cooking-agi-interview
```

See [skills/README.md](skills/README.md). Contributed by [@Juggernaut0825](https://github.com/Juggernaut0825).

<span id="_2"></span>

## What's inside

<!-- repository-only -->

<a href="https://tianyi-zhang-02.github.io/cooking-agi/learn/coverage.en.html"><picture><source media="(prefers-color-scheme: dark)" srcset="https://tianyi-zhang-02.github.io/cooking-agi/static/readme-overview-en-dark.svg"><img src="https://tianyi-zhang-02.github.io/cooking-agi/static/readme-overview-en-light.svg" width="100%" alt="What the notes are checked and illustrated with, topics by area, and what is being worked on now and next; updated on every deploy"></picture></a>

<!-- /repository-only -->

- **How models work:** [Transformer walkthrough](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.en.html) · [BERT](00-foundations/core/bert.en.md) · [CLIP and multimodal learning](03-multimodal-learning/clip.en.md)
- **How models learn:** [Training basics](learn/pretraining/README.en.md) · [Deep RL](05-post-training/deep-rl/README.en.md) · [Post-training](05-post-training/README.en.md) · [LLM-as-a-Judge](07-evaluation/llm-as-a-judge/README.en.md)
- **How models get used:** [Inference](learn/inference/README.en.md) · [Search and retrieval](04-search/README.en.md) · [Agents and tools](10-agents/README.en.md)
- **What's moving:** [Frontier watch](11-frontier/README.en.md) · [AI infrastructure](12-infra/README.en.md) · [Perception, robots and driving](13-embodied/README.en.md)
- **Practice:** [ML coding](learn/ml-exercises/README.en.md) · [Python](interview/python.en.md) · [Algorithm patterns](interview/leetcode.en.md) · [System design](learn/system-design/README.en.md)

Some material is still being reviewed; [coverage](learn/coverage.en.md) shows what has been checked. What comes next is in the [roadmap](community/roadmap.en.md); how the site is updated is in [updates](community/updates.en.md).

<span id="contributors"></span>

## Contributors and reviewers

<!-- repository-only -->

<picture><source media="(prefers-color-scheme: dark)" srcset="https://tianyi-zhang-02.github.io/cooking-agi/static/readme-people-en-dark.svg"><img src="https://tianyi-zhang-02.github.io/cooking-agi/static/readme-people-en-light.svg" width="100%" alt="Contributors by commits, the reviewer for each area, and where readers went next; rebuilt every day"></picture>

<!-- /repository-only -->

Commit counts tell only part of the story; questions, reviews, and corrections count too. [Add a contribution credit](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml) · [Share your next stop](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=next-stop.yml)

<span id="_3"></span>

## Join us!

If a passage is confusing, something is out of date, or a topic is missing, tell us. You don't need the answer first.

- **Feedback:** [open an issue](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml) and link the passage. Every note also has a feedback link at the end.
- **Changes:** small fixes can go straight into a PR. For a new article, start with a [content proposal](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml).
- **Pass it on:** if the notes helped, share the link with a friend.

These are working notes, not an official curriculum. We share public knowledge and preparation methods, never company-specific interview questions. Notes are CC BY 4.0 and code is MIT; see the [licenses](LICENSE.md).

<details markdown="1">
<summary>Contributing, review, and privacy</summary>

See the [contributing guide](CONTRIBUTING.md#english), [editorial guide](EDITORIAL.en.md), and [area reviewers](community/README.en.md). Community PRs need a relevant non-author CODEOWNER review and passing checks. Use public sources and your own explanations; disclose and check AI-assisted work. Issues are public, so don't attach private progress backups, company material, or unredacted logs.

</details>

<span id="_4"></span>

## Run locally

No database needed. From the repository root, with Python 3.12:

```bash
python3 -m pip install markdown pygments numpy
python3 site/build.py --serve
```

Then open <http://localhost:8000>. `python3 site/build.py` alone writes the site to `_site/`.

<details markdown="1">
<summary>Checks before submitting</summary>

```bash
python3 site/collaboration.py
python3 -m unittest discover -s site/tests
python3 site/leakcheck.py
python3 site/paritycheck.py --strict
python3 site/build.py
```

Some experiments also need PyTorch; their tests skip when it isn't installed.

</details>

<span id="_5"></span>

## Why this exists

When I started preparing for industry roles, my materials were scattered across lectures, papers, documentation, and bookmarks, and something I recognized in a paper was often much harder to explain or implement myself. Writing it down showed me those gaps; sharing it lets others point out the rest.

I hope it makes a field easier to start without a background in it or someone to hand you a reading list. If it saves you some searching and leaves a little more time for the rest of your life, that's a good reason to keep working on it.

<span id="_6"></span>

## Changelog

<details markdown="1">
<summary>Every merge into main, by date</summary>

| Date | Major | Minor |
| --- | --- | --- |
| 2026-10-17 | Interview coach skill: mock interviews in Claude Code or Codex graded against the notes, which now come from a read-only data API on the site ([#103](https://github.com/tianyi-zhang-02/cooking-agi/pull/103), contributed by @Juggernaut0825) · New README: animated preview, and topic, roadmap and contributor cards that update with the site ([#104](https://github.com/tianyi-zhang-02/cooking-agi/pull/104) [#105](https://github.com/tianyi-zhang-02/cooking-agi/pull/105)) | Contributor pages move from the site to the GitHub page ([#104](https://github.com/tianyi-zhang-02/cooking-agi/pull/104)) · Card icons and folded sections ([#106](https://github.com/tianyi-zhang-02/cooking-agi/pull/106)) |
| 2026-10-16 | Every note audited: errors fixed, interview pages kept free of company hiring processes ([#99](https://github.com/tianyi-zhang-02/cooking-agi/pull/99)), facts updated to October 2026 ([#100](https://github.com/tianyi-zhang-02/cooking-agi/pull/100)), repeated explanations folded together and links to the new areas ([#101](https://github.com/tianyi-zhang-02/cooking-agi/pull/101)) · Draft series on scaling laws, diffusion LMs and quantization ([#102](https://github.com/tianyi-zhang-02/cooking-agi/pull/102)) | The unpublished quant notes removed ([#98](https://github.com/tianyi-zhang-02/cooking-agi/pull/98)) |
| 2026-10-15 | Frontier watch, AI infrastructure and embodied AI, with the papers' own figures and 26 step-by-step figures ([#97](https://github.com/tianyi-zhang-02/cooking-agi/pull/97)) | — |
| 2026-10-14 | Step-by-step animations in the LeetCode notes ([#78](https://github.com/tianyi-zhang-02/cooking-agi/pull/78)) · Step-by-step figures and the papers' own figures in four chapters ([#83](https://github.com/tianyi-zhang-02/cooking-agi/pull/83)) · Every note read through once in both languages, checking facts, worked examples and wording ([#84](https://github.com/tianyi-zhang-02/cooking-agi/pull/84) [#85](https://github.com/tianyi-zhang-02/cooking-agi/pull/85) [#87](https://github.com/tianyi-zhang-02/cooking-agi/pull/87) [#88](https://github.com/tianyi-zhang-02/cooking-agi/pull/88) [#89](https://github.com/tianyi-zhang-02/cooking-agi/pull/89) [#90](https://github.com/tianyi-zhang-02/cooking-agi/pull/90)) · Step-by-step figures in 16 training and systems notes ([#94](https://github.com/tianyi-zhang-02/cooking-agi/pull/94)) | Saved notes, reading history and backups fold into “My reading”; the study dashboard goes and its exercises move to the study plan ([#81](https://github.com/tianyi-zhang-02/cooking-agi/pull/81)) · New README promo ([#79](https://github.com/tianyi-zhang-02/cooking-agi/pull/79)) · README roadmap, topic coverage and changelog ([#82](https://github.com/tianyi-zhang-02/cooking-agi/pull/82)) · Line breaks inside Chinese paragraphs no longer render as spaces ([#86](https://github.com/tianyi-zhang-02/cooking-agi/pull/86)) · Review list and changelog ([#91](https://github.com/tianyi-zhang-02/cooking-agi/pull/91)) · The interview-prep home becomes a plain directory ([#93](https://github.com/tianyi-zhang-02/cooking-agi/pull/93)) · With the private repo archived, the leak check is the only gate ([#92](https://github.com/tianyi-zhang-02/cooking-agi/pull/92)) · The gated-attention example uses gates that do not sum to 1 ([#95](https://github.com/tianyi-zhang-02/cooking-agi/pull/95)) |
| 2026-10-13 | Chronicle: a technology world tree with 143 sourced milestones, later a vertical timeline ([#65](https://github.com/tianyi-zhang-02/cooking-agi/pull/65) [#71](https://github.com/tianyi-zhang-02/cooking-agi/pull/71)) · Study dashboard and reading history ([#64](https://github.com/tianyi-zhang-02/cooking-agi/pull/64) [#67](https://github.com/tianyi-zhang-02/cooking-agi/pull/67)) | Reading panel: today, this week, all time ([#66](https://github.com/tianyi-zhang-02/cooking-agi/pull/66) [#70](https://github.com/tianyi-zhang-02/cooking-agi/pull/70)) · One-click page feedback ([#72](https://github.com/tianyi-zhang-02/cooking-agi/pull/72)) · Security hardening ([#73](https://github.com/tianyi-zhang-02/cooking-agi/pull/73)) · Study buddy ([#74](https://github.com/tianyi-zhang-02/cooking-agi/pull/74)) · Anonymous counts ([#75](https://github.com/tianyi-zhang-02/cooking-agi/pull/75) [#76](https://github.com/tianyi-zhang-02/cooking-agi/pull/76)) · vLLM / SGLang comparison ([#68](https://github.com/tianyi-zhang-02/cooking-agi/pull/68)) · README animation ([#69](https://github.com/tianyi-zhang-02/cooking-agi/pull/69)) |
| 2026-10-11 | Bilingual entry, README and learning routes; engineering labs and more worked examples ([#61](https://github.com/tianyi-zhang-02/cooking-agi/pull/61) [#62](https://github.com/tianyi-zhang-02/cooking-agi/pull/62)) | Training, inference, SigLIP and ML systems guides ([#63](https://github.com/tianyi-zhang-02/cooking-agi/pull/63)) |
| 2026-10-09 | Bilingual learning routes reorganized, examples checked one by one ([#60](https://github.com/tianyi-zhang-02/cooking-agi/pull/60)) | — |
| 2026-10-05 | Bilingual Deep RL lessons with interactive labs ([#58](https://github.com/tianyi-zhang-02/cooking-agi/pull/58)) | More natural career notes ([#59](https://github.com/tianyi-zhang-02/cooking-agi/pull/59)) |
| 2026-10-03 | Math and quant review with proofs and worked examples ([#56](https://github.com/tianyi-zhang-02/cooking-agi/pull/56), later removed) | Discussions retired to refocus on the notes ([#57](https://github.com/tianyi-zhang-02/cooking-agi/pull/57)) |
| 2026-10-02 | Bilingual LLM-as-a-Judge lessons with interactive evaluation labs ([#55](https://github.com/tianyi-zhang-02/cooking-agi/pull/55)) | — |
| 2026-10-01 | Navigation, reading tools and community pages ([#45](https://github.com/tianyi-zhang-02/cooking-agi/pull/45)) | Discussions launched and laid out ([#53](https://github.com/tianyi-zhang-02/cooking-agi/pull/53) [#54](https://github.com/tianyi-zhang-02/cooking-agi/pull/54)) |
| 2026-09-30 | Bilingual learning paths and chapter self-checks ([#42](https://github.com/tianyi-zhang-02/cooking-agi/pull/42)) | CLIP / VLM notes and bilingual READMEs ([#44](https://github.com/tianyi-zhang-02/cooking-agi/pull/44)) · The Python you use in coding problems ([#40](https://github.com/tianyi-zhang-02/cooking-agi/pull/40)) |
| 2026-09-25 | — | AI Infra section removed; skipped notes polished, a co-reviewer added ([#37](https://github.com/tianyi-zhang-02/cooking-agi/pull/37) [#38](https://github.com/tianyi-zhang-02/cooking-agi/pull/38)) |
| 2026-09-24 | Quant zone: a probability series ([#36](https://github.com/tianyi-zhang-02/cooking-agi/pull/36), later removed) | — |
| 2026-09-23 | Close readings, agents, model choice, career and AI infra ([#33](https://github.com/tianyi-zhang-02/cooking-agi/pull/33) [#34](https://github.com/tianyi-zhang-02/cooking-agi/pull/34)) | Contributors page and credits ([#30](https://github.com/tianyi-zhang-02/cooking-agi/pull/30) [#31](https://github.com/tianyi-zhang-02/cooking-agi/pull/31) [#32](https://github.com/tianyi-zhang-02/cooking-agi/pull/32)) · ML Systems reading list ([#35](https://github.com/tianyi-zhang-02/cooking-agi/pull/35)) |
| 2026-09-22 | A nine-note RLHF series; MoE, looped Transformers and Agents with seven live figures ([#24](https://github.com/tianyi-zhang-02/cooking-agi/pull/24) [#25](https://github.com/tianyi-zhang-02/cooking-agi/pull/25)) | One language per page ([#18](https://github.com/tianyi-zhang-02/cooking-agi/pull/18)) · Night-sky design and home page ([#19](https://github.com/tianyi-zhang-02/cooking-agi/pull/19) [#20](https://github.com/tianyi-zhang-02/cooking-agi/pull/20) [#21](https://github.com/tianyi-zhang-02/cooking-agi/pull/21) [#22](https://github.com/tianyi-zhang-02/cooking-agi/pull/22)) · Long evaluation notes split ([#23](https://github.com/tianyi-zhang-02/cooking-agi/pull/23)) · Career notes ([#26](https://github.com/tianyi-zhang-02/cooking-agi/pull/26) [#27](https://github.com/tianyi-zhang-02/cooking-agi/pull/27) [#28](https://github.com/tianyi-zhang-02/cooking-agi/pull/28) [#29](https://github.com/tianyi-zhang-02/cooking-agi/pull/29)) |
| 2026-09-21 | Site redesign around career routes, knowledge blocks and quick review ([#17](https://github.com/tianyi-zhang-02/cooking-agi/pull/17)) | — |
| 2026-08-27 | — | Interview fundamentals, a whiteboard checklist and reference implementations ([#15](https://github.com/tianyi-zhang-02/cooking-agi/pull/15) [#16](https://github.com/tianyi-zhang-02/cooking-agi/pull/16)) |
| 2026-08-26 | Post-training series in four steps, from PPO to what came after ([#11](https://github.com/tianyi-zhang-02/cooking-agi/pull/11) [#13](https://github.com/tianyi-zhang-02/cooking-agi/pull/13) [#14](https://github.com/tianyi-zhang-02/cooking-agi/pull/14)) | A Chinese reading guide ([#10](https://github.com/tianyi-zhang-02/cooking-agi/pull/10)) |
| 2026-08-24 | — | Decoding and the RLHF pipeline restored ([#8](https://github.com/tianyi-zhang-02/cooking-agi/pull/8)) |
| 2026-08-14 | — | Multi-head attention, normalization and residual connections ([#7](https://github.com/tianyi-zhang-02/cooking-agi/pull/7)) |
| 2026-08-12 | Foundations chapter, reading order and the first site ([#3](https://github.com/tianyi-zhang-02/cooking-agi/pull/3)) | Smoother Chinese, grouped sidebar ([#4](https://github.com/tianyi-zhang-02/cooking-agi/pull/4)) · Architecture diagrams and intro / advanced layers ([#6](https://github.com/tianyi-zhang-02/cooking-agi/pull/6)) |
| 2026-08-10 | First AI infrastructure study guide ([#1](https://github.com/tianyi-zhang-02/cooking-agi/pull/1)) | Redrawn diagrams ([#2](https://github.com/tianyi-zhang-02/cooking-agi/pull/2)) |

</details>
