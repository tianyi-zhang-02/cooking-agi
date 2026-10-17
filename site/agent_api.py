"""Publish the notes as a small read-only data API for agents, next to the site.

`python site/build.py` calls build() after rendering, so the API is regenerated on every deploy and
nothing generated is committed. The interview skill (skills/cooking-agi-interview) and any other
agent read it over HTTPS:

    <origin>/api/v1/topics.json   topics in sidebar order, each with its notes
    <origin>/api/v1/bank.json     questions pulled from the notes, with reference answers
    <origin>/api/v1/notes.json    every English note as plain markdown, keyed by repository path
    <origin>/api/v1/meta.json     build time and counts
    <origin>/llms.txt             a short index for agents

Questions come from the same sources interview/questions.md uses:

- interview Q&A: <details class="interview"> / <details class="qa"> anywhere, plus any top-level
  <details> under an "interview" heading (answer included)
- quick-learning cards: <details> under a "Quick learning" heading (answer included)
- self-check items: the <li> of a note's taste-check box (no answer: grade against the note)

Only published notes are included (site/nav.toml); the project's own pages and the career notes,
which are a personal record rather than study material, are left out.

    python site/agent_api.py            # write the API into _site/ (or --out DIR) without a full build

Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
API = "api/v1"
# the project's own pages, and career notes (a personal record, not study material)
SKIP_DIRS = {"community", "templates", "papers", "career"}


def published_sources(nav):
    """(nav section, zh source) in sidebar order; mirrors site/paritycheck.py."""
    claimed = {s for sec in nav["section"] for s in sec.get("include", [])}
    seen = set()
    for sec in nav["section"]:
        d = ROOT / sec["dir"]
        files = [p for p in d.glob("*.md") if not p.name.endswith(".en.md")
                 and p.relative_to(ROOT).as_posix() not in claimed] if d.exists() else []
        files += [ROOT / s for s in sec.get("include", []) if (ROOT / s).exists()]
        if sec["dir"] == ".":
            files = [p for p in files if p.name in sec.get("order", [])]
        rank = {n: i for i, n in enumerate(sec.get("order", []))}
        for src in sorted(files, key=lambda p: (rank.get(p.name, 999), p.name)):
            if src not in seen:
                seen.add(src)
                yield sec, src


def topic_id(directory: str) -> str:
    """'00-foundations/core' -> 'foundations-core'"""
    parts = [re.sub(r"^\d+-", "", p) for p in directory.split("/")]
    return re.sub(r"[^a-z0-9]+", "-", "-".join(parts).lower()).strip("-") or "start-here"


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def top_level_details(body: str):
    """(opening tag, summary text, inner markdown) for each outermost <details>."""
    out, depth, start, tag = [], 0, None, ""
    for m in re.finditer(r"<details[^>]*>|</details>", body):
        if m.group(0).startswith("<details"):
            if depth == 0:
                start, tag = m.end(), m.group(0)
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0 and start is not None:
                block = body[start:m.start()]
                s = re.search(r"<summary>(.*?)</summary>", block, flags=re.S)
                if s:
                    out.append((tag, text_of(s.group(1)), block[s.end():].strip()))
                start = None
    return out


def clean_md(md: str) -> str:
    """Site markup out, readable markdown in. Relative links between notes keep working because
    the bundle keeps the repository's folder layout."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"<svg\b.*?</svg>", "", md, flags=re.S)
    md = re.sub(r"<(script|style)\b.*?</\1>", "", md, flags=re.S)
    md = re.sub(r"(?m)^(#{1,6} .*?)[ \t]*\{#[^}]*\}[ \t]*$", r"\1", md)
    md = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", md, flags=re.S)
    md = re.sub(r"<nav\b.*?</nav>", "", md, flags=re.S)
    md = re.sub(r"!\[([^\]]*)\]\([^)]*\)", lambda m: f"[figure: {m.group(1)}]" if m.group(1) else "", md)
    md = re.sub(r"<summary>(.*?)</summary>", lambda m: f"\n**{text_of(m.group(1))}**\n", md, flags=re.S)
    # card fields written back to back (<small>02 / Engine</small><strong>Schedule work</strong>) keep a separator
    md = re.sub(r"</(small|strong|span|em|b|i|code)>(?=<(small|strong|span|em|b|i|code)\b)", " — ", md)
    md = re.sub(r"</?(details|div|span|figure|figcaption|section|aside|p|ol|ul|strong|em|b|i|small|code|br)\b[^>]*>",
                lambda m: {"<li>": "- "}.get(m.group(0), ""), md)
    md = re.sub(r"<li\b[^>]*>", "- ", md).replace("</li>", "")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def note_title(text: str, fallback: str) -> str:
    m = re.search(r"(?m)^# (.+)$", text)
    return m.group(1).strip() if m else fallback


def note_questions(text: str):
    """(kind, section heading, question, answer markdown) for every question a note contains."""
    out, seen = [], set()
    parts = re.split(r"(?m)^(#{2,3} .*)$", text)
    sections = [("", parts[0])] + [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
    for head, body in sections:
        h = re.sub(r"\s*\{#.*", "", head.lstrip("# ")).strip()
        is_interview = bool(re.search(r"interview", h, re.I))
        is_quick = bool(re.search(r"quick (learning|review)", h, re.I))
        for tag, q, answer in top_level_details(body):
            if not q or re.match(r"(deep dive|deeper|code|show|full|derivation|worked example)", q, re.I):
                continue
            if is_quick or q.lower().startswith("quick learning"):  # summary cards ("Match, normalize, aggregate ..."), not questions
                kind = "concept"
            elif 'class="interview"' in tag or 'class="qa"' in tag or is_interview:
                kind = "interview"
            else:
                continue
            q = re.sub(r"^(Q?\d+[.、:)]|Q:)\s*", "", q)
            if q not in seen:
                seen.add(q)
                out.append((kind, h, q, clean_md(answer)))
    box = re.search(r'<div class="taste-check[^"]*">(.*?)</div>', text, flags=re.S)
    if box:
        for li in re.findall(r"<li>(.*?)</li>", box.group(1), flags=re.S):
            q = text_of(li)
            if q and q not in seen:
                seen.add(q)
                out.append(("self-check", "Self-check", q, ""))
    return out


def site_path(rel: str) -> str:
    """The page a note becomes on the site (mirrors build.Page): x/y.en.md -> x/y.en.html,
    x/README.md -> x/index.html."""
    path = Path(rel)
    en = path.name.endswith(".en.md")
    name = path.name[:-6] if en else path.name[:-3]
    return (path.parent / (("index" if name == "README" else name) + (".en.html" if en else ".html"))).as_posix()


def build(out: Path) -> dict:
    """Write the API under out/ (the built site). Returns the counts."""
    nav = tomllib.loads((SITE / "nav.toml").read_text(encoding="utf-8"))
    site = nav["site"]
    repo_url = f"https://github.com/{site['repo']}/blob/main/"
    origin = site["origin"].rstrip("/")
    topics, bank, notes = {}, [], {}
    for sec, src in published_sources(nav):
        en = src.with_name(src.stem + ".en.md")
        rel = en.relative_to(ROOT).as_posix()
        if not en.exists() or rel.split("/")[0] in SKIP_DIRS or "/" not in rel:
            continue
        if rel in notes:
            continue
        text = en.read_text(encoding="utf-8")
        tid = topic_id(sec["dir"] if sec["dir"] != "." else "")
        topic = topics.setdefault(tid, {"id": tid, "title": sec["en"], "title_zh": sec["zh"],
                                        "group": sec.get("group", ""), "notes": [], "questions": 0})
        title = note_title(text, en.stem)
        qs = note_questions(text)
        draft = "Draft, pending review" in text
        zh_rel = src.relative_to(ROOT).as_posix()
        url = f"{origin}/{site_path(rel)}"
        notes[rel] = {"title": title, "url": url, "zh_url": f"{origin}/{site_path(zh_rel)}",
                      "source": repo_url + rel, "zh_source": repo_url + zh_rel, "draft": draft, "text": clean_md(text)}
        topic["notes"].append({"path": rel, "title": title, "questions": len(qs)})
        topic["questions"] += len(qs)
        for kind, section, q, answer in qs:
            qid = hashlib.sha1(f"{rel}\n{q}".encode()).hexdigest()[:8]
            bank.append({"id": qid, "topic": tid, "kind": kind, "question": q, "answer": answer,
                         "note": rel, "note_title": title, "note_url": url, "section": section, "draft": draft})
    topics = [t for t in topics.values() if t["notes"]]
    counts = {"topics": len(topics), "notes": len(notes), "questions": len(bank),
              "with_answers": sum(1 for b in bank if b["answer"])}
    dest = out / API
    dest.mkdir(parents=True, exist_ok=True)
    dump = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    (dest / "topics.json").write_text(dump(topics), encoding="utf-8")
    (dest / "bank.json").write_text(dump(bank), encoding="utf-8")
    (dest / "notes.json").write_text(dump(notes), encoding="utf-8")
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (dest / "meta.json").write_text(dump({"version": 1, "built": built, "repo": site["repo"], **counts}),
                                    encoding="utf-8")
    (out / "llms.txt").write_text(llms_txt(site, origin, topics, counts), encoding="utf-8")
    return counts


def llms_txt(site: dict, origin: str, topics: list, counts: dict) -> str:
    lines = [f"# {site['title_en']}", "",
             f"> {site['tagline_en']}. Bilingual (Chinese and English) study notes on machine learning and "
             "language models, with worked examples you can recompute.", "",
             f"Source: https://github.com/{site['repo']}", "",
             "## Data for agents (read-only JSON, rebuilt on every deploy)", "",
             f"- [topics]({origin}/{API}/topics.json): {counts['topics']} topics in sidebar order, each with its notes",
             f"- [question bank]({origin}/{API}/bank.json): {counts['questions']} questions, "
             f"{counts['with_answers']} with reference answers",
             f"- [notes]({origin}/{API}/notes.json): {counts['notes']} English notes as plain markdown, keyed by path",
             f"- Interview coach skill: https://github.com/{site['repo']}/tree/main/skills/cooking-agi-interview",
             "", "## Topics", ""]
    lines += [f"- {t['title']} ({len(t['notes'])} notes)" for t in topics]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "_site", help="site directory to write into")
    args = ap.parse_args()
    print(build(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
