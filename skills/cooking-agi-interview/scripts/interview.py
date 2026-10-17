#!/usr/bin/env python3
"""Question picker and progress log for the cooking-agi interview skill. Standard library only.

    interview.py topics                      topics with question counts and your scores
    interview.py notes TOPIC                 the notes behind a topic
    interview.py read PATH [--section WORDS] one note as plain markdown (PATH as listed by notes/ask),
                                             or only the sections whose heading has WORDS
    interview.py ask [-t TOPIC ...] [-n N] [--kind interview|self-check|concept] [--weak]
                                             pick questions (reference answers stay hidden)
    interview.py reveal ID                   reference answer + source note for one question
    interview.py record ID SCORE [--gap TEXT]
    interview.py record --topic TOPIC --question TEXT SCORE [--gap TEXT]
                                             log a 0-4 score (second form: a question you wrote)
    interview.py report                      weak topics, recent gaps, what to drill next
    interview.py search WORDS ...            find notes and lines that mention the words
    interview.py reset                       clear the progress log

The notes and questions are read from the site's data API (rebuilt on every deploy) and cached
for a day in ~/.cooking-agi-interview/cache; with no network the cache is used. --refresh forces a
download. COOKING_AGI_API may point to another API URL or to a local directory (a built _site/api/v1).
Progress lives in ~/.cooking-agi-interview/progress.json (override with COOKING_AGI_PROGRESS).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path.home() / ".cooking-agi-interview"
API = os.environ.get("COOKING_AGI_API", "https://tianyi-zhang-02.github.io/cooking-agi/api/v1")
CACHE = Path(os.environ.get("COOKING_AGI_CACHE", HOME / "cache"))
PROGRESS = Path(os.environ.get("COOKING_AGI_PROGRESS", HOME / "progress.json"))
REPO = "https://github.com/tianyi-zhang-02/cooking-agi/blob/main/"
DAY = 86400
REFRESH = False
_loaded = {}


FILES = ("topics.json", "bank.json", "notes.json")


def fetch_all():
    """Download every API file together, so the cache is one consistent snapshot and every command
    works offline after the first run. Returns the error if any download fails (cache untouched)."""
    got = {}
    try:
        for name in FILES:
            req = urllib.request.Request(f"{API.rstrip('/')}/{name}", headers={"User-Agent": "cooking-agi-interview"})
            with urllib.request.urlopen(req, timeout=30) as r:
                got[name] = r.read().decode("utf-8")
            json.loads(got[name])
    except (urllib.error.URLError, OSError, ValueError) as e:
        return e
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, data in got.items():
        (CACHE / name).write_text(data, encoding="utf-8")
    return None


def load(name):
    """topics.json / bank.json / notes.json from the API, cached for a day; stale cache if offline."""
    if name in _loaded:
        return _loaded[name]
    if Path(API).is_dir():
        data = (Path(API) / name).read_text(encoding="utf-8")
    else:
        cached = CACHE / name
        fresh = all((CACHE / f).exists() and time.time() - (CACHE / f).stat().st_mtime < DAY for f in FILES)
        if (REFRESH or not fresh) and not _loaded.get("_tried"):
            _loaded["_tried"] = True  # one download attempt per run, however many files are read
            error = _loaded["_error"] = fetch_all()
            if error and cached.exists():
                print(f"(offline: using the cached notes and questions; {error})", file=sys.stderr)
        if not cached.exists():
            sys.exit(f"cannot reach {API} ({_loaded.get('_error')}) and there is no cached copy yet")
        data = cached.read_text(encoding="utf-8")
    _loaded[name] = json.loads(data)
    return _loaded[name]


def progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"attempts": []}


def save(p):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def last_attempts(p):
    """{question id: most recent attempt}"""
    out = {}
    for a in p["attempts"]:
        out[a["id"]] = a
    return out


def topic_stats(p):
    stats = {}
    for a in p["attempts"]:
        s = stats.setdefault(a["topic"], [])
        s.append(a["score"])
    return {t: (len(s), sum(s) / len(s)) for t, s in stats.items()}


def resolve_topics(names, topics):
    """Accept exact ids, prefixes ('post-training' -> all its subtopics) or words in the title."""
    if not names:
        return None
    chosen = set()
    for n in names:
        n = n.lower()
        hit = [t["id"] for t in topics if t["id"] == n or t["id"].startswith(n + "-")
               or n in t["title"].lower() or n in t["id"]]
        if not hit:
            sys.exit(f"unknown topic: {n}  (run: interview.py topics)")
        chosen.update(hit)
    return chosen


def cmd_topics(args):
    topics, stats = load("topics.json"), topic_stats(progress())
    print(f"{'topic':44} {'notes':>5} {'qs':>4}  your record")
    for t in topics:
        n, avg = stats.get(t["id"], (0, None))
        rec = f"{n} answered, avg {avg:.1f}/4" if n else "-"
        print(f"{t['id']:44} {len(t['notes']):>5} {t['questions']:>4}  {rec}   # {t['title']}")


def cmd_notes(args):
    topics = load("topics.json")
    for tid in sorted(resolve_topics([args.topic], topics)):
        t = next(t for t in topics if t["id"] == tid)
        print(f"## {t['id']} — {t['title']}")
        for n in t["notes"]:
            print(f"  {n['path']}  ({n['questions']} qs)  {n['title']}")
    print("\nread one with: interview.py read PATH")


def find_note(path):
    """A note by repository path; also accepts links copied from a note (../x.en.md#part),
    Chinese paths (x.md) and site URLs (.../x.en.html)."""
    notes = load("notes.json")
    p = re.sub(r"[#?].*$", "", path.strip()).split("cooking-agi/")[-1]
    p = re.sub(r"^(\./|\.\./)+", "", p)
    p = re.sub(r"\.en\.html$|\.html$", ".en.md", p)
    if p.endswith(".md") and not p.endswith(".en.md"):
        p = p[:-3] + ".en.md"
    if p.endswith("/index.en.md"):
        p = p[:-len("index.en.md")] + "README.en.md"
    if p in notes:
        return p, notes[p]
    ends = [k for k in notes if k.endswith("/" + p) or k == p]
    return (ends[0], notes[ends[0]]) if len(ends) == 1 else (None, None)


def cmd_read(args):
    path, note = find_note(args.path)
    if not note:
        sys.exit(f"no note at {args.path}  (paths are listed by: interview.py notes TOPIC)")
    draft = "  [draft, pending the site owner's review]" if note.get("draft") else ""
    print(f"# {path}{draft}\n# on the site: {note.get('url', note['source'])}\n# source: {note['source']}\n")
    if not args.section:
        print(note["text"])
        return
    words = [w.lower() for w in args.section]
    parts = re.split(r"(?m)^(?=#{2,4} )", note["text"])
    hits, depth = [], 0
    for part in parts:
        level = len(part) - len(part.lstrip("#")) if part.startswith("##") else 0  # not the title (#)
        if depth and level and level <= depth:
            depth = 0  # the matched section has ended
        if not depth and level and all(w in part.splitlines()[0].lower() for w in words):
            depth = level  # a matching heading: keep it with its subsections
        if depth:
            hits.append(part)
    if hits:
        print("\n".join(h.rstrip() + "\n" for h in hits))
    else:
        print("no section heading contains: " + " ".join(args.section) + "\nheadings:")
        print("\n".join("  " + p.splitlines()[0] for p in parts if p.startswith("##")))


def cmd_ask(args):
    bank, topics, p = load("bank.json"), load("topics.json"), progress()
    chosen, last, now = resolve_topics(args.topic, topics), last_attempts(p), time.time()
    rng = random.Random(args.seed)
    pool = []
    for q in bank:
        if chosen and q["topic"] not in chosen:
            continue
        if (args.kind and q["kind"] != args.kind) or (not args.kind and q["kind"] == "concept"):
            continue
        a = last.get(q["id"])
        if args.weak and (not a or a["score"] > 2):
            continue
        if a is None:
            w = 3.0
        else:
            age = (now - a["ts"]) / DAY
            w = (4 - a["score"]) + min(age / 3, 2) + (0.2 if a["score"] >= 3 else 1)
            if age < 0.02:  # asked within the last half hour
                w *= 0.05
        pool.append((w * rng.random(), q))
    pool.sort(key=lambda x: -x[0])
    if not pool:
        print("no matching questions in the bank: write your own from the notes (interview.py notes TOPIC)")
        return
    for _, q in pool[:args.n]:
        tag = "has reference answer" if q["answer"] else "grade against the note"
        prev = last.get(q["id"])
        seen = f" | last score {prev['score']}/4" if prev else ""
        draft = " | draft note" if q.get("draft") else ""
        print(f"[{q['id']}] ({q['topic']}, {q['kind']}, {tag}{seen}{draft})\n  Q: {q['question']}\n  note: {q['note']}\n")


def cmd_reveal(args):
    q = next((q for q in load("bank.json") if q["id"] == args.id), None)
    if not q:
        sys.exit(f"no question with id {args.id}")
    draft = "  [draft, pending the site owner's review]" if q.get("draft") else ""
    print(f"Q: {q['question']}\nnote: {q['note']}  — {q['note_title']}  (section: {q['section']}){draft}")
    print(f"on the site: {q['note_url']}\n" if q.get("note_url") else "")
    print(q["answer"] or "(no written answer: read the note above and grade against it)")


def cmd_record(args):
    if not 0 <= args.score <= 4:
        sys.exit("score must be 0-4")
    p = progress()
    if args.id:
        q = next((q for q in load("bank.json") if q["id"] == args.id), None)
        if not q:
            sys.exit(f"no question with id {args.id}")
        entry = {"id": q["id"], "topic": q["topic"], "question": q["question"]}
    else:
        if not (args.topic and args.question):
            sys.exit("give a question id, or --topic and --question for a question you wrote")
        tid = sorted(resolve_topics([args.topic], load("topics.json")))[0]
        qid = "g" + hashlib.sha1(args.question.encode()).hexdigest()[:7]
        entry = {"id": qid, "topic": tid, "question": args.question}
    entry.update(score=args.score, gap=args.gap or "", ts=int(time.time()))
    p["attempts"].append(entry)
    save(p)
    print(f"recorded {entry['id']} ({entry['topic']}): {args.score}/4")


def cmd_report(args):
    p, topics = progress(), {t["id"]: t for t in load("topics.json")}
    if not p["attempts"]:
        print("no answers recorded yet")
        return
    stats, last = topic_stats(p), last_attempts(p)
    scores = [a["score"] for a in last.values()]
    print(f"{len(p['attempts'])} answers, {len(last)} distinct questions, avg latest score {sum(scores)/len(scores):.2f}/4\n")
    print("by topic (weakest first):")
    for tid, (n, avg) in sorted(stats.items(), key=lambda kv: kv[1][1]):
        print(f"  {avg:.1f}/4  x{n:<3} {tid}  — {topics.get(tid, {}).get('title', '')}")
    weak = [a for a in last.values() if a["score"] <= 2]
    if weak:
        print("\nstill weak (latest score <= 2):")
        for a in sorted(weak, key=lambda a: a["score"])[:15]:
            gap = f"  gap: {a['gap']}" if a.get("gap") else ""
            print(f"  [{a['id']}] {a['score']}/4 {a['question'][:90]}{gap}")
    untouched = [t for t in topics if t not in stats and topics[t]["questions"]]
    if untouched:
        print("\nnot practised yet:", ", ".join(untouched[:12]) + (" ..." if len(untouched) > 12 else ""))


def cmd_search(args):
    words = [w.lower() for w in args.words]
    hits = []
    for f, note in load("notes.json").items():
        lines = note["text"].splitlines()
        low = [l.lower() for l in lines]
        score = sum(l.count(w) for l in low for w in words)
        if score and all(any(w in l for l in low) for w in words):
            heads = [i for i, l in enumerate(low) if l.startswith("#") and any(w in l for w in words)]
            body = [i for i, l in enumerate(low) if any(w in l for w in words)]
            hits.append((score + 20 * len(heads), f, lines, list(dict.fromkeys(heads + body))[:4]))
    hits.sort(key=lambda h: -h[0])
    for score, f, lines, idx in hits[:args.n]:
        print(f"{f}  (score {score})")
        for i in idx:
            print(f"   {i + 1}: {lines[i][:140]}")
    if not hits:
        print("no note mentions all of:", " ".join(args.words))


def cmd_reset(args):
    if PROGRESS.exists():
        PROGRESS.unlink()
    print("progress cleared")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="download the notes and questions again")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("topics").set_defaults(fn=cmd_topics)
    s = sub.add_parser("notes"); s.add_argument("topic"); s.set_defaults(fn=cmd_notes)
    s = sub.add_parser("read"); s.add_argument("path")
    s.add_argument("--section", nargs="+", help="only the sections whose heading contains these words")
    s.set_defaults(fn=cmd_read)
    s = sub.add_parser("ask")
    s.add_argument("-t", "--topic", nargs="*")
    s.add_argument("-n", type=int, default=3)
    s.add_argument("--kind", choices=["interview", "self-check", "concept"],
                   help="concept = summary cards for teach-back and rapid review (left out by default)")
    s.add_argument("--weak", action="store_true", help="only questions you last scored 0-2 on")
    s.add_argument("--seed", type=int)
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("reveal"); s.add_argument("id"); s.set_defaults(fn=cmd_reveal)
    s = sub.add_parser("record")
    s.add_argument("id", nargs="?")
    s.add_argument("score", type=int)
    s.add_argument("--topic"); s.add_argument("--question"); s.add_argument("--gap")
    s.set_defaults(fn=cmd_record)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    s = sub.add_parser("search"); s.add_argument("words", nargs="+"); s.add_argument("-n", type=int, default=6)
    s.set_defaults(fn=cmd_search)
    sub.add_parser("reset").set_defaults(fn=cmd_reset)
    args = ap.parse_args()
    global REFRESH
    REFRESH = args.refresh
    args.fn(args)


if __name__ == "__main__":
    main()
