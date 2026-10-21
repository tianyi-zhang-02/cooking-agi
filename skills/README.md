# Agent skills

## cooking-agi-interview

A mock interviewer and review coach for MLE / research engineer / LLM engineer / AI infra
interviews, built from the notes in this repository. It asks one question at a time, grades the
answer against the notes, follows up on gaps, and keeps a local progress log so weak areas come
back. Works with Claude Code, Codex, and other agents that read `SKILL.md` skills. No server of its
own and no API key beyond your agent's: the notes and questions come from the site's read-only
data API and are cached on your machine. The script saves progress locally and does not upload
it to this site. The conversation, including answers you enter, is still handled according to
your agent's and model provider's settings; using this skill does not make it local-only.

Install:

```bash
npx skills add https://github.com/tianyi-zhang-02/cooking-agi --skill cooking-agi-interview
```

Add `-g` to install for your user instead of the current project, or `-a claude-code -a codex`
to choose agents. Then ask your agent things like:

- "Mock interview for an LLM inference engineer role, 5 questions"
- "Drill me on RLHF and DPO until I get three in a row right"
- "Rapid-fire review of Transformer basics"
- "What am I weak at? Quiz me on that"
- "ML system design interview: a RAG assistant over 10M documents"

Progress is stored in `~/.cooking-agi-interview/progress.json`.

### For maintainers

Nothing in this folder is generated. The skill reads the data API that `python site/build.py`
writes into the site on every build (`site/agent_api.py`):

- `/api/v1/topics.json`, `/api/v1/bank.json`, `/api/v1/notes.json`, `/api/v1/meta.json`
- `/llms.txt`, a short index for agents

So editing a note updates the skill after the next deploy; there is nothing to regenerate. Only
published study notes are included (career notes and the project's own pages are left out), and
draft notes are flagged. To try the skill against a local build:

```bash
python site/build.py
COOKING_AGI_API=_site/api/v1 python3 skills/cooking-agi-interview/scripts/interview.py topics
```
