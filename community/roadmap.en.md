# What we'd like to build next {#roadmap}

[中文](roadmap.md) · **English**

Notes matter for being accurate, easy to find and quick to open, not for their number. So the site itself comes first, then figures for mechanisms that are hard to explain in words, then ways for AI assistants and a question bank to use the notes directly. As the notes grow, we'd like people from other fields to help explain the parts we know less about.

<nav class="study-route roadmap-route" aria-label="Roadmap navigation">
<a href="#now"><small>NOW</small><strong>Faster search and pages</strong><span>Find the right section, open it fast, skip re-downloads</span></a>
<a href="#next"><small>NEXT</small><strong>Step figures, MCP, a question bank</strong><span>One at a time, in this order</span></a>
<a href="#future"><small>OPEN TO CONTRIBUTORS</small><strong>From vision to robotics</strong><span>Start with one good explanation and one small project</span></a>
</nav>

**See [Updates](updates.en.md) for how the site is updated and where your reading records live, and [Coverage](../learn/coverage.en.md) for individual chapters.** This page describes intentions, not shipped features, and gives no dates.

## Now: faster search and pages {#now}

Most readers come to look up a concept, a derivation or a question, so finding and opening come first:

- **More accurate search:** an English term matches from the start of a word, so "exploration" no longer counts as a hit for "lora"; `kv-cache` and `kv cache` return the same notes; when a note isn't about the term itself, the result opens at the section that matched.
- **Faster search:** one index per language, with the duplicate copy removed. On a phone, the first search drops from about 3.7 s to about 1.8 s, and loading starts as soon as you reach for the search button.
- **Lighter pages:** interactive labs and math rendering load only on pages that use them. Files are versioned by content, so when the site rebuilds on schedule, returning readers don't download anything that hasn't changed.

This batch is in [#128](https://github.com/tianyi-zhang-02/cooking-agi/pull/128). Still to do: misspelled terms (such as "atention") find nothing yet, and some rankings need tuning.

## Next: one at a time, in this order {#next}

1. **Step-by-step figures for more notes.** Some mechanisms are faster to see than to read, such as which positions prefill and decode each compute. We make one figure at a time: a still sketch first, then, once it is right, a figure you can step through. For an example, see prefill and decode in [Decoder-only](../00-foundations/core/decoder-only.en.md).
2. **A remote MCP server.** The site already publishes a read-only data API (`/api/v1/`), which the interview coach skill reads notes and questions from. An MCP server on the same data lets assistants such as Claude, ChatGPT and Cursor look up notes and quiz you from them, without a second copy of the content.
3. **A question bank on the key mechanisms.** Questions now sit inside individual notes, and [Quick review](../interview/questions.en.md) lists them by area. Next, we group them by mechanism, such as attention, the KV cache, normalization, PPO / DPO and decoding, with each question linked to the section that explains it.

**The interview-prep release (v1.0) is dropped.** The interview notes, the study plan, saved notes and reading records stay, but we won't build more interview-prep features. Accounts, cloud sync, streaks and leaderboards aren't planned either.

## Future topics: begin with a small project {#future}

These are areas we'd like to explore, **not complete courses or scheduled releases**. No one has been assigned to maintain them. The existing notes lean toward ML / LLMs; we'd especially welcome authors and reviewers with experience in these other fields.

| Direction | Possible first notes | A manageable first project |
| --- | --- | --- |
| Computer vision (CV) | Image inputs, convolutions and ViTs, classification vs. detection vs. segmentation, data and evaluation | Fix a public dataset subset, build a baseline, and examine where it fails |
| Autonomous driving | Perception, prediction, planning, control, and the information passed between them | Study one component using public data or simulation, without treating offline scores as evidence of road safety |
| Jetson & edge deployment | Model export, device inference, and latency, memory, and power tradeoffs | Deploy a small vision model with device details, software versions, and a measurement procedure; explain the workflow for readers without the hardware |
| Physical simulation | State, actions, time steps, contact, sensor noise, and simplifying assumptions | Build a small control task, change one assumption, and explain the difference |
| Robotics | Coordinate frames, kinematics, control, imitation learning, and RL in a complete system | Start with a simulated arm or mobile robot, identifying what does not transfer directly to hardware |

These shouldn't become five disconnected textbooks. Reuse the notes on data, evaluation, optimization, systems, and deployment where they help. We'd like each project to explain what a component receives, what it produces, and how it fits with the rest.

Add a navigation section and preparation route once there is useful material and someone willing to maintain it, not a row of empty categories first. Contributors don't need to take on an entire field. Work involving vehicles or physical robots needs its own experimental conditions, failure analysis, and safety limits; a successful simulation is not a safety case for hardware.

## Where can I help? {#contribute}

A typo, diagram, or clearer explanation is enough for a small PR. You can also [leave feedback](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml) without knowing the fix. You don't need to be an expert to point out something that is hard to follow.

For a new topic, open a [proposal](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) with four things:

1. **The reader:** someone starting out, someone with a foundation, or someone preparing for a kind of technical interview?
2. **One problem to explain:** a whole series is not required.
3. **The material:** public sources, a small example, or a reproducible experiment.
4. **What you'd like to do:** outline, prose, code, translation, or review, within the time you have.

A maintainer can help narrow the scope, find a home for it, and ask for a willing reviewer. One contribution is not a commitment to ongoing maintenance. Ordinary changes use the [PR review process](README.en.md#review); cross-area changes start with a discussion.

A ready note needs prerequisites, an explanation that builds from an example, a way to practice or check the result, and limits. State costs and versions for hardware work. Provide a CPU or simulated alternative when possible, and say when it isn't. Write each language naturally, credit sources, and distinguish an experiment you ran from one you only read about.

## How is the site updated? {#versions}

There are no version numbers. A change goes live once it is merged into main and deployed; scheduled rebuilds a few times a day refresh data such as the contributor list. The home page's [changelog](../README.md#changelog) lists every merge by date, and each article's source link and repository history show its own edits.

The backup format for saved notes and reading records is versioned separately, so site updates don't invalidate an old backup. Browser storage is not a permanent backup, though: export before clearing site data, changing devices, or moving to another origin.

Proposals can change the directions here; the table is not a queue everyone must follow. The site will stay free. If it helps you, please share it with someone who might find it useful.
