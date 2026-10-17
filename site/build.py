"""Build the static site from the markdown notes.

Design goals, in order:

  1. The notes stay pure markdown. No front matter, no shortcodes, nothing that
     makes a file render worse on GitHub than it does here. Everything the site
     needs lives in site/nav.toml and site/glossary.tsv.
  2. Adding a note is dropping a .md into a section folder. Nothing to register.
  3. Zero runtime dependencies for the reader; two build-time ones (markdown,
     pygments) that CI installs.

Usage:
    python site/build.py            # -> _site/
    python site/build.py --serve    # build, then serve on :8000
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import math
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import markdown
import agent_api
import collaboration
import curriculum
import prep_progress
import readme_cards
import topic_coverage
import note_metrics
import technology_chronicle
from markdown.extensions.toc import TocExtension, slugify

ROOT = Path(__file__).resolve().parent.parent
SITE = Path(__file__).resolve().parent
OUT = ROOT / "_site"

# tags whose text must never be touched by the glossary annotator
# headings stay clean: an inline gloss there is noise, and it would
# disagree with the (unannotated) text used in the table of contents
PROTECTED = {"code", "pre", "a", "script", "style", "abbr",
             "h1", "h2", "h3", "h4", "h5", "h6", "button", "label",
             "summary", "option", "svg"}

# Structured components whose text is laid out by CSS. An inline gloss chip
# inside one splits a word in half and breaks the grid, so they are skipped
# wholesale -- the annotation belongs in running prose, not in a summary card.
NOGLOSS_CLASSES = {"lesson-recipe", "taste-check", "widget", "mermaid", "topic-coverage",
                   "study-atlas", "study-route", "chapter-context", "worked-update", "engine-map", "technology-tree", "tree-eras", "rl-budget", "prep-hub", "prep-progress", "prep-note", "study-workspace",
                   "home-block", "term", "curriculum-card", "curriculum-hero",
                   "learning-path", "lab-matrix", "bilingual-intro",
                   "drl-flow", "drl-paths", "drl-lab", "interview-roles", "method-map",
                   "paper-fig", "paper-fig-link"}

GLOSS_BLOCKS = {"p", "li", "td", "th", "dd"}
GLOSSARY_COMPOUND_EXCLUSIONS = {"方差": ("平方差", "立方差")}

# elements that never carry an end tag, so they must not push onto the stack
VOID = {"br", "img", "hr", "input", "meta", "link", "source", "col", "wbr"}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_nav():
    with open(SITE / "nav.toml", "rb") as f:
        return tomllib.load(f)


def load_roadmap():
    """site/roadmap.toml: the tracks, stages and modules shown on the home page."""
    f = SITE / "roadmap.toml"
    return tomllib.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def load_glossary(filename="glossary.tsv"):
    """中文 -> (English, gloss). Longest terms first so 多头注意力 wins over 注意力."""
    terms = []
    for line in (SITE / filename).read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            terms.append((parts[0].strip(), parts[1].strip(),
                          parts[2].strip() if len(parts) > 2 else ""))
    return sorted(terms, key=lambda t: -len(t[0]))


def glossary_for_page(page, terms):
    filename = {
        "deep-rl": "glossary.deep-rl.tsv",
        "algorithms": "glossary.algorithms.tsv",
    }.get(page.section.get("group"))
    if filename is None:
        return terms
    scoped = load_glossary(filename)
    names = {term[0] for term in scoped}
    return sorted(scoped + [term for term in terms if term[0] not in names],
                  key=lambda term: -len(term[0]))


# --------------------------------------------------------------------------- #
# git / github metadata
# --------------------------------------------------------------------------- #
def git(*args, default=""):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def last_updated(rel_path: str) -> str:
    """ISO date of the last commit that touched this file (empty if uncommitted)."""
    return git("log", "-1", "--format=%cI", "--", rel_path)


@functools.lru_cache(maxsize=1)
def crew_countries():
    """Where each login says they are, from crew.toml. One place, or two split evenly."""
    path = ROOT / "crew.toml"
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        print(f"  note: crew.toml is not valid TOML ({e}); ignoring it")
        return {}
    out = {}
    for person in data.get("crew", []):
        login = person.get("login")
        said = person.get("countries") or person.get("country") or []
        if isinstance(said, str):
            said = [said]
        codes = [c for c in (str(x).strip().upper() for x in said) if c]
        if login and codes:
            out[str(login).strip().lower()] = tuple(dict.fromkeys(codes))
    return out


def contributor_name(name: str, email: str) -> str:
    name = name.strip()
    if email.strip().casefold() == "noreply@anthropic.com" or name.casefold() == "claude code":
        return "Claude Code"
    if name.casefold() in {"codex", "openai codex"}:
        return "Codex"
    return name


def crew_ai_collaborators():
    path = ROOT / "crew.toml"
    if not path.exists():
        return {}
    credits = {}
    for entry in tomllib.loads(path.read_text(encoding="utf-8")).get("ai", []):
        name = contributor_name(entry.get("name", ""), "")
        if not name or name in credits or not entry.get("zh") or not entry.get("en"):
            raise ValueError("AI credits need unique names and bilingual descriptions")
        credits[name] = {language: entry[language] for language in ("zh", "en")}
    return credits


def contributors(repo: str):
    """Everyone who has committed or is credited as a co-author, most recent first.

    Tries the GitHub API for avatars (CI has a token and it is the accurate
    source), falls back to `git log` so a local build still works offline.
    """
    recency, earliest, counts = {}, {}, {}
    ai_credits = crew_ai_collaborators()
    ai_names = {"Claude Code", "Codex"} | set(ai_credits)
    ai_models = {}
    log = git(
        "log",
        "--format=%an%x09%ae%x09%cI%x09%(trailers:key=Co-authored-by,valueonly,separator=%x1f)",
    )
    for line in log.splitlines():
        parts = line.split("\t", 3)
        if len(parts) < 3:
            continue
        name, email, when = parts[:3]
        identities = [(name, email)]
        credits = parts[3] if len(parts) == 4 else ""
        for credit in credits.split("\x1f"):
            match = re.match(r"\s*(.*?)\s*<([^>]+)>\s*$", credit)
            if match:
                identities.append(match.groups())
        seen = set()
        for name, email in identities:
            key = contributor_name(name, email)
            if key in ai_names:
                ai_models.setdefault(key, set()).add(name.strip())
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            if key not in recency or when > recency[key][0]:
                recency[key] = (when, email.strip())
            if key not in earliest or when < earliest[key]:
                earliest[key] = when

    logins = {}
    for name, (_, email) in recency.items():  # 12345+login@users.noreply.github.com
        m = re.match(r"^(?:\d+\+)?([A-Za-z0-9-]+)@users\.noreply\.github\.com$", email)
        if m:
            logins[name] = m.group(1)

    api = {}
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/contributors?per_page=100",
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "cooking-agi-site-build"})
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=10) as r:
            for c in json.load(r):
                api[c["login"].lower()] = {"login": c["login"], "avatar": c["avatar_url"],
                                           "commits": c["contributions"]}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError) as e:
        print(f"  note: GitHub API unavailable ({type(e).__name__}); using git log only")

    out = []
    for name, (when, _) in sorted(recency.items(), key=lambda kv: kv[1][0], reverse=True):
        login = logins.get(name)
        info = api.get((login or name).lower(), {})
        login = info.get("login") or login
        out.append({
            "name": name,
            "login": login,
            "url": f"https://github.com/{login}" if login else None,
            "avatar": info.get("avatar") or (f"https://github.com/{login}.png?size=80"
                                             if login else None),
            "commits": (counts.get(name, 0) if name in ai_names
                        else info.get("commits", counts.get(name, 0))),
            "last": datetime.fromisoformat(when).astimezone(timezone.utc).date().isoformat(),
            "first": datetime.fromisoformat(earliest.get(name, when)).astimezone(
                timezone.utc).date().isoformat(),
            "countries": crew_countries().get((login or name).lower(), ()),
            "initial": name[:1].upper(),
            "kind": "ai" if name in ai_names else "human",
            "models": sorted(ai_models.get(name, set())),
            "description": ai_credits.get(name, {}),
        })
    for name, description in ai_credits.items():
        if name not in recency:
            out.append({
                "name": name, "login": None, "url": None, "avatar": None,
                "commits": None, "first": "", "last": "", "countries": (),
                "initial": name[:1].upper(), "kind": "ai", "models": [],
                "description": description,
            })
    # a catalogue number, the way a star gets one: by the order it was first seen,
    # so a number stays with the same person as the list grows
    for n, person in enumerate(sorted(out, key=lambda p: (p["first"] or "9999", p["name"])), start=1):
        person["crew_id"] = f"CG {n:03d}"
    return out


# --------------------------------------------------------------------------- #
# glossary annotation
# --------------------------------------------------------------------------- #
class Annotator(HTMLParser):
    """Wrap the FIRST occurrence of each glossary term in readable prose.

    Skips anything inside code, links or headings, so we never corrupt an
    identifier or a URL. Operates on rendered HTML, which is the only place we
    can reliably tell prose from markup.
    """

    def __init__(self, terms):
        super().__init__(convert_charrefs=True)
        self.terms = sorted(terms, key=lambda term: -len(term[0]))
        self.seen = set()
        self.depth = 0
        self.stack = []          # one entry per open element: does it protect?
        self.out = []
        self.used = []
        self.block_counts = []

    @staticmethod
    def _protects(tag, attrs):
        if tag in PROTECTED:
            return True
        classes = set()
        for k, v in attrs:
            if k == "class" and v:
                classes.update(v.split())
        return bool(classes & NOGLOSS_CLASSES)

    def handle_starttag(self, tag, attrs):
        if tag in GLOSS_BLOCKS:
            self.block_counts.append(0)
        if tag not in VOID:
            hit = self._protects(tag, attrs)
            self.stack.append(hit)
            self.depth += hit
        self.out.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.out.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag in GLOSS_BLOCKS and self.block_counts:
            self.block_counts.pop()
        if tag not in VOID and self.stack:
            self.depth -= self.stack.pop()
            self.depth = max(0, self.depth)
        self.out.append(f"</{tag}>")

    def handle_comment(self, data):
        self.out.append(f"<!--{data}-->")

    def handle_decl(self, decl):
        self.out.append(f"<!{decl}>")

    def handle_data(self, data):
        prose, math_stash = protect_math(data)
        text = html.escape(prose, quote=False)
        if self.depth == 0 and self.terms:
            # Collect non-overlapping matches against the ORIGINAL text, then
            # splice once. Replacing term by term would let a later term match
            # inside an earlier term's injected data-tip -- a gloss such as
            # "映到同一向量空间靠内积召回" contains 向量, which would then split
            # the attribute open and leak markup into the page.
            hits, taken = [], []
            matched = set(self.seen)
            for zh, en, gloss in self.terms:
                for occurrence in re.finditer(re.escape(zh), text):
                    start, stop = occurrence.span()
                    if any(text[max(0, stop - len(compound)):stop] == compound
                           for compound in GLOSSARY_COMPOUND_EXCLUSIONS.get(zh, ())):
                        continue
                    if any(start < end and stop > begin for begin, end in taken):
                        continue
                    following = text[stop:]
                    existing = re.match(r"\s*[（(]([^()（）\n]{1,160})[)）]", following)
                    extended = re.match(r"[\u3400-\u9fff]{1,6}\s*[（(]([^()（）\n]{1,160})[)）]", following)
                    if not existing and extended and en.casefold() in extended.group(1).casefold():
                        existing = extended
                    explained = bool(existing and re.search(r"[A-Za-z]", existing.group(1)))
                    consumed = existing.end() if explained else 0
                    leading = re.search(r"([A-Za-z][A-Za-z0-9 /_-]{0,160})\s*[（(][^()（）\n]*$", text[:start])
                    if (leading and en.casefold() in leading.group(1).casefold()
                            and re.match(r"[^()（）\n]*[)）]", following)):
                        explained = True
                    end = stop + consumed
                    taken.append((start, end))
                    if zh not in matched:
                        hits.append((start, zh, en, gloss, explained))
                        matched.add(zh)

            out, prev = [], 0
            remaining = max(0, 2 - self.block_counts[-1]) if self.block_counts else 2
            for i, zh, en, gloss, explained in sorted(hits):
                if not explained and remaining == 0:
                    continue
                self.seen.add(zh)
                self.used.append({"zh": zh, "en": en, "gloss": gloss})
                tip = html.escape(f"{en}" + (f" — {gloss}" if gloss else ""), quote=True)
                out.append(text[prev:i])
                if explained:
                    out.append(zh)
                else:
                    out.append(f'<span class="term" tabindex="0" data-tip="{tip}">{zh}'
                               f'<span class="term-en">（{html.escape(en)}）</span></span>')
                    remaining -= 1
                    if self.block_counts:
                        self.block_counts[-1] += 1
                prev = i + len(zh)
            out.append(text[prev:])
            text = "".join(out)
        self.out.append(restore_math(text, math_stash))

    def result(self):
        return "".join(self.out)


def annotate(html_text, terms):
    a = Annotator(terms)
    a.feed(html_text)
    a.close()
    return a.result(), a.used


# --------------------------------------------------------------------------- #
# link rewriting
# --------------------------------------------------------------------------- #
def rewrite_links(html_text: str, page, repo: str, known: set) -> str:
    """Point markdown-relative links at the built pages.

    Anything with no page on the site -- CONTRIBUTING.md, a .py file, a bare
    directory of figures -- resolves to the source on GitHub rather than 404ing.
    `known` is the set of every page this build produces, so a link can never
    silently point at a file that was not generated.
    """
    suffix = ".en.html" if page.lang == "en" else ".html"
    here = page.out_rel.parent  # directory of this page, repo-relative

    def gh(path, kind):
        target = os.path.normpath(str(here / path)).replace(os.sep, "/").strip("/")
        return f"https://github.com/{repo}/{kind}/main/{target}"

    def fix(m):
        attr, url = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|data:|#|/)", url):
            return m.group(0)
        path, _, frag = url.partition("#")
        frag = f"#{frag}" if frag else ""
        raw = path

        source_path = os.path.normpath(str(here / path)).replace(os.sep, "/")
        if attr == "src" and source_path.startswith("site/static/") and (ROOT / source_path).is_file():
            asset_path = os.path.relpath(source_path[len("site/"):], str(here)).replace(os.sep, "/")
            return f'{attr}="{asset_path}{frag}"'

        if source_path in {"README.md", "README.en.md", "README.zh.md"}:
            home = "index.zh.html" if source_path == "README.zh.md" else "index.html"
            path = os.path.relpath(home, str(here)).replace(os.sep, "/")
        elif path.endswith(".en.md"):
            path = path[:-6] + ".en.html"
        elif path.endswith(".md"):
            path = path[:-3] + suffix
        elif path.endswith("/") or (path and "." not in Path(path).name):
            path = path.rstrip("/") + "/index" + suffix
        elif "assets/" in path:
            asset_url, _, query = path.partition("?")
            asset = (ROOT / here / asset_url).resolve()
            if attr == "src" and asset.suffix == ".svg" and asset.is_relative_to(ROOT) and asset.is_file():
                params = [(key, value) for key, value in urllib.parse.parse_qsl(html.unescape(query)) if key != "v"]
                params.append(("v", hashlib.sha256(asset.read_bytes()).hexdigest()[:12]))
                versioned = asset_url + "?" + urllib.parse.urlencode(params) + frag
                return f'{attr}="{html.escape(versioned, quote=True)}"'
            return f'{attr}="{path}{frag}"'          # copied verbatim into _site
        else:
            return f'{attr}="{gh(raw, "blob")}{frag}"'
        for readme_suffix in (".en.html", ".html"):
            if path.endswith("README" + readme_suffix):
                path = path[: -len("README" + readme_suffix)] + "index" + readme_suffix
                break

        resolved = os.path.normpath(str(here / path)).replace(os.sep, "/")
        if resolved not in known and path.endswith(".en.html"):
            zh_path = path[:-8] + ".html"   # no English version yet: the Chinese page beats a GitHub blob
            if os.path.normpath(str(here / zh_path)).replace(os.sep, "/") in known:
                return f'{attr}="{zh_path}{frag}"'
        if resolved not in known:
            return f'{attr}="{gh(raw, "tree" if raw.endswith("/") else "blob")}{frag}"'
        query = "?lang=en" if resolved == "index.html" else ""
        return f'{attr}="{path}{query}{frag}"'

    return re.sub(r'\b(href|src)="([^"]+)"', fix, html_text)


# --------------------------------------------------------------------------- #
# widgets
# --------------------------------------------------------------------------- #
WIDGETS = {
    "xor": """
<figure class="widget" data-widget="xor">
  <figcaption class="widget-head">
    <span class="widget-kicker" data-zh="图解" data-en="Diagram"></span>
    <span class="widget-title" data-zh="自己训一遍：把激活函数关掉试试"
          data-en="Train it yourself: try switching the activation off"></span>
  </figcaption>
  <div class="widget-body">
    <canvas class="xor-canvas" width="620" height="310"></canvas>
    <div class="widget-controls">
      <label class="switch"><input type="checkbox" class="xor-act" checked>
        <span data-zh="启用 ReLU 激活" data-en="ReLU activation"></span></label>
      <label class="slider"><span data-zh="隐藏单元" data-en="hidden units"></span>
        <input type="range" class="xor-hidden" min="1" max="16" value="8">
        <output class="xor-hidden-out">8</output></label>
      <button class="btn xor-reset" data-zh="重新开始" data-en="Restart"></button>
      <div class="xor-stats">
        <span class="stat"><b class="xor-step">0</b><i data-zh="步" data-en="steps"></i></span>
        <span class="stat"><b class="xor-loss">—</b><i>loss</i></span>
        <span class="stat"><b class="xor-acc">—</b><i data-zh="准确率" data-en="accuracy"></i></span>
      </div>
    </div>
  </div>
</figure>
""",
}


# Transformer lab (static/tx-lab.js). The figures build themselves in the browser,
# so the markup here is only the shell plus a no-JS fallback line.
TX_LAB = [
    ("tx-arch", "结构图：选一个模型家族，看哪里变了", "Architecture map: pick a family and watch what moves"),
    ("tx-attention", "Self-attention 一步一步算（6 个 token 的玩具例子）", "Self-attention step by step (a six-token toy)"),
    ("tx-kv-cache", "解码 10 个 token：每一步到底算了什么", "Decoding 10 tokens: what each step computes"),
    ("tx-kv-heads", "每个 token 要缓存多少个 K/V head", "How many K/V heads each token has to cache"),
    ("tx-windows", "谁能看到谁：attention pattern", "Who can see whom: attention patterns"),
    ("tx-rope", "RoPE：位置就是旋转", "RoPE: position as rotation"),
    ("tx-moe", "MoE：一次一个 token 经过 FFN", "MoE: one token at a time through the FFN"),
    ("tx-flash", "FlashAttention：attention 矩阵放在哪里", "FlashAttention: where the attention matrix lives"),
    ("tx-rlhf", "RLHF 一步一步：三个阶段，四个模型", "RLHF step by step: three stages, four models"),
    ("tx-ppo-clip", "PPO clipping：拖动 ρ，看它什么时候停止鼓励", "PPO clipping: drag ρ and watch when the encouragement stops"),
    ("tx-moe-router", "MoE router：一个 token 怎样选 expert", "MoE router: how one token picks its experts"),
    ("tx-moe-balance", "负载均衡：不做、辅助 loss、只调 bias", "Load balancing: none, auxiliary loss, bias only"),
    ("tx-loop-unroll", "循环：同一个 block 重复用 L 次", "Looping: one block reused L times"),
    ("tx-loop-reach", "多跳：每过一遍多走一跳", "Multi-hop: one more link per pass"),
    ("tx-loop-exit", "自适应深度：每个 token 自己决定转几圈", "Adaptive depth: each token decides when to stop"),
    ("tx-agent-loop", "Agent 循环：修一个失败的测试", "The agent loop: fixing a failing test"),
    ("tx-agent-cost", "用 API 还是自己 serve：粗略的月成本模型", "Frontier API or self-hosted: a rough monthly cost model"),
    ("tx-agent-maze", "同一个迷宫，三种 agent", "One maze, three kinds of agent"),
    ("tx-model-router", "这条请求交给谁：小模型、frontier、级联、路由", "Who gets the request: small, frontier, cascade, router"),
    ("tx-py-sort", "sorted() 加上 key：lambda 到底按什么排", "sorted() with a key: what the lambda actually sorts by"),
    ("tx-prob-events", "把概率画成面积：互斥、独立，还是都不是", "Probability as area: disjoint, independent, or neither"),
    ("tx-prob-dist", "把分布画出来：柱子、曲线和跳跃", "Drawing a distribution: bars, curves, and jumps"),
]
for _name, _zh, _en in TX_LAB:
    WIDGETS[_name] = f"""
<figure class="widget tx-lab" id="{_name}" data-widget="{_name}">
  <figcaption class="widget-head">
    <span class="widget-kicker" data-zh="图解" data-en="Diagram"></span>
    <span class="widget-title" data-zh="{_zh}" data-en="{_en}"></span>
  </figcaption>
  <div class="widget-body"><p class="tx-fallback" data-zh="这张交互图需要启用 JavaScript。" data-en="This interactive figure needs JavaScript."></p></div>
</figure>
"""

# Algorithm lab (static/algo-lab.js): small steppers for the LeetCode method notes.
# Same shell and chrome as the Transformer lab; the frames are built in the browser.
ALGO_LAB = [
    ("algo-sieve", "筛法：遇到质数，就从它的平方开始划", "Sieve: each prime crosses out from its square"),
    ("algo-window-max", "单调队列：谁还有机会接班", "Monotonic queue: who can still take over"),
    ("algo-two-pointers", "双指针：每一步排除一个不可能的端点", "Two pointers: each step rules out an endpoint"),
    ("algo-sliding-window", "滑动窗口：从左边移走，直到不再重复", "Sliding window: shrink from the left until it fits"),
    ("algo-binary-search", "二分：right = middle，left = middle + 1", "Binary search: right = middle, left = middle + 1"),
    ("algo-prefix-count", "前缀和：先查旧前缀，再存当前前缀", "Prefix sums: look up old prefixes, then store"),
    ("algo-rob", "同一个子问题：算一次，存下来", "The same subproblem: compute once, reuse"),
    ("algo-bst-bounds", "BST：只看父子会漏，范围要一路带下去", "BST: parent–child checks miss; carry the bounds"),
    ("algo-reverse-list", "反转链表：先存下一个，再掉头", "Reversing a list: save next, then turn around"),
    ("algo-kmp", "KMP：对上以后不归零，模式往右滑", "KMP: after a match, slide instead of resetting"),
    ("algo-lcs", "LCS：相同就斜着加一，不同就取上方或左方", "LCS: equal steps diagonally, else take above or left"),
    ("algo-cooldown", "三个状态，每天只读昨天", "Three states, each day reads only yesterday"),
    ("algo-jump", "Jump Game：只记最远能到哪", "Jump Game: remember only the farthest reach"),
    ("algo-append", "摊还：偶尔很贵，平均很便宜", "Amortized: occasionally expensive, cheap on average"),
    ("algo-knapsack", "背包：更新顺序决定能拿几次", "Knapsack: update order decides how many copies"),
]
# Chapter figures (also static/algo-lab.js): one per idea where a picture helps.
# kind "step" = a stepper, "still" = a single diagram without controls.
FIGURE_LAB = [
    ("step-contrastive", "一次对比训练：分数 → 概率 → 损失 → 梯度", "One contrastive step: scores → probabilities → loss → gradients", "step"),
    ("step-embedding-swap", "维度一样，坐标不一定对得上", "Same dimension, different coordinates", "step"),
    ("step-rrf", "RRF：只看名次，把两路结果合起来", "RRF: fuse two lists by rank alone", "step"),
    ("step-retrieval-metrics", "同一份结果，四个指标各问什么", "One result list, four questions", "step"),
    ("step-pq", "PQ：查表再相加，得到近似距离", "PQ: look up and add to get an approximate distance", "step"),
    ("step-vit-patches", "ViT：切块、展平、投影", "ViT: patch, flatten, project", "step"),
    ("step-qwen-merge", "14 像素切块，再 2×2 合并", "14-pixel patches, then a 2×2 merge", "step"),
    ("step-qformer-masks", "Q-Former 的三种可见性", "Q-Former's three visibility masks", "still"),
    ("step-flow-matching", "Flow matching：在数轴上走一步", "Flow matching on a number line", "step"),
    ("step-eval-funnel", "逐层嵌套的漏斗：100 → 55", "A nested funnel: 100 → 55", "step"),
    ("step-bleu", "BLEU-2：截断计数和长度惩罚", "BLEU-2: clipped counts and the brevity penalty", "step"),
    ("step-judge-renorm", "先归一到 1–5 分，再算均分", "Renormalize to the 1–5 labels, then average", "step"),
    ("step-win-rate", "胜率的分母怎么定", "Choosing the win-rate denominator", "step"),
    ("step-weighted-slices", "平均分涨了，一组却退步了", "The average rises while one group regresses", "step"),
    ("step-context-staircase", "每次重发全部历史，输入越来越长", "Resending the whole history on every call", "step"),
    ("step-skill-loading", "按需加载，还是一次全读", "On demand, or everything up front", "step"),
    ("step-agent-patterns", "五种 workflow 和一种 agent", "Five workflows and an agent", "still"),
    ("step-audio-frames", "25 ms 窗口、10 ms 步长：2 秒是 198 帧", "25 ms windows, 10 ms hops: 198 frames in 2 seconds", "step"),
    ("step-vl2-layout", "VL2 的序列长度：1415，不是 1372", "VL2 sequence length: 1415, not 1372", "step"),
    ("step-mrope", "M-RoPE：时间、高度、宽度三个序号", "M-RoPE: time, height and width positions", "step"),
    ("step-ppl-windows", "滑动窗口：前文可以重读，目标只计一次", "Sliding windows: reread the context, score each target once", "step"),
    ("step-ivf", "IVF：最近的簇里不一定有最近的点", "IVF: the nearest cluster may not hold the nearest point", "step"),
    ("step-best-of-n", "Best-of-2：谁会被选中", "Best-of-2: who gets picked", "step"),
    ("step-bt-scale", "平移不变，缩放会变", "A shift changes nothing; a scale does", "step"),
    ("step-token-mean", "按回答平均，还是按 token 平均", "Average by response or by token", "step"),
    ("step-kl-penalty", "两种回答的 KL 惩罚", "The KL penalty on two answers", "step"),
    ("step-behavior-ratio", "比率的分母用哪个分布", "Which distribution goes in the denominator", "step"),
    ("step-chunked-prefill", "分块 prefill：每轮的位置怎么分", "Chunked prefill: sharing each round’s positions", "step"),
    ("step-continuous-batching", "等整批结束，还是每轮补位", "Wait for the batch, or refill each round", "step"),
    ("step-l1-l2", "L1 为什么能停在 0", "Why L1 can stop at exactly zero", "step"),
    ("step-bce-vs-mse", "饱和的 sigmoid，梯度消失了吗", "A saturated sigmoid: does the gradient vanish?", "step"),
    ("step-adamw-l2", "AdamW 与 Adam + L2 的第一步", "AdamW vs Adam + L2: the first step", "step"),
    ("step-packing-masks", "Packing：能看谁，哪项算分", "Packing: what can be seen, what is scored", "step"),
    ("step-tp-backward", "反向传播：输入梯度也要相加", "Backward: the input gradient needs a sum too", "step"),
    ("step-bpe-merges", "手算两轮 BPE", "Two rounds of BPE by hand", "step"),
    ("step-swiglu", "跟着两个中间维度算一次 SwiGLU", "SwiGLU through two hidden dimensions", "step"),
    ("step-depth-merge", "分批做深度 softmax，怎样合并", "Merging a depth softmax computed in batches", "step"),
    ("step-expert-budget", "切细与共享，预算怎么算", "Finer and shared experts on one budget", "step"),
    ("step-verifier-ceiling", "测试放过错解：一轮就到顶", "Tests that let wrong answers through: the ceiling in one round", "step"),
    ("step-time-horizon", "拟合 50% 时间跨度，以及超出范围的读数", "Fitting a 50% time horizon, and readings past the task range", "step"),
    ("step-evaluator-rank", "相关系数 0.98，第一名还是选错了", "A correlation of 0.98, and still the wrong winner", "step"),
    ("step-mcts-visits", "六次模拟：访问次数变成策略标签", "Six simulations: visit counts become a policy target", "step"),
    ("step-lambda-return", "从后往前算 λ-return", "Computing a λ-return backwards", "step"),
    ("step-roofline", "同一张 roofline：三个操作，再加 decode batch", "One roofline: three operations, then decode batching", "step"),
    ("step-latency-hiding", "几个 warp 才藏得住 300 个周期的等待", "How many warps hide a 300-cycle wait?", "step"),
    ("step-tiling-reads", "数 global 读取：不分块 128 次，分块 64 次", "Counting global reads: 128 untiled, 64 tiled", "step"),
    ("step-cache-walk", "同一个矩阵：按行走 256 次缺失，按列走 4,096 次", "One matrix: 256 misses by row, 4,096 by column", "step"),
    ("step-reliance-groups", "按组相加：团队什么时候超过 AI", "Adding up the groups: when the team beats the AI", "step"),
    ("step-judge-sampling", "抽样评判：多大的退化看得见", "Sampled judging: how large a regression shows up", "step"),
    ("step-subagent-timeline", "并行子 agent 省下多少时间", "What parallel subagents save in wall-clock time", "step"),
    ("step-lift-splat", "一个像素摊进三格：特征沿视线拖尾", "One pixel into three cells: features smear along the ray", "step"),
    ("step-trajectory-cost", "权重挑轨迹，硬约束先淘汰", "Weights pick a trajectory; hard constraints remove one first", "step"),
    ("step-open-closed-loop", "开环 L2 一样，闭环结局相反", "Same open-loop L2, opposite closed-loop outcomes", "step"),
    ("step-flow-euler-steps", "Euler 步数太少，动作退回均值", "Too few Euler steps fall back to the mean", "step"),
    ("step-vla-latency", "一块 50 × 7 的动作，时间花在哪", "Where the time goes in one 50 × 7 action chunk", "step"),
    ("step-good-dies", "好 die 比面积掉得更快", "Good dies fall faster than die area grows", "step"),
    ("step-fusion-traffic", "relu(x @ w + b)：字节和 CPU 时间花在哪", "relu(x @ w + b): where the bytes and the CPU time go", "step"),
    ("step-decode-floor", "70B 的 decode 下限 = 权重字节 ÷ 总带宽", "70B decode floor = weight bytes ÷ total bandwidth", "step"),
    ("step-gang-scheduling", "同样 24 张卡，三种放法", "The same 24 GPUs, three placements", "step"),
    ("step-pixel-ray", "一个像素是一条射线：地面或第二台相机定出深度", "A pixel is a ray: the ground or a second camera fixes the depth", "step"),
    ("step-kalman-1d", "一维 Kalman：预测变宽，修正变窄", "1D Kalman: prediction widens, correction narrows", "step"),
    ("step-coco-ap", "5 个预测怎样算出 AP@[.5:.95]", "From 5 predictions to AP@[.5:.95]", "step"),
    ("step-depth-scale", "相对深度要一个已知距离，仿射深度要两个", "Relative depth needs one known distance, affine needs two", "step"),
    ("step-alpha-blend", "从前往后叠四个 Gaussian：顺序改颜色，不改透射率", "Blending four Gaussians front to back: order changes the colour, not the transmittance", "step"),
]
for _name, _zh, _en, _kind in [(name, zh, en, "step") for name, zh, en in ALGO_LAB] + FIGURE_LAB:
    _kicker = 'data-zh="动画" data-en="Step by step"' if _kind == "step" else 'data-zh="图解" data-en="Diagram"'
    WIDGETS[_name] = f"""
<figure class="widget tx-lab algo-lab" id="{_name}" data-widget="{_name}">
  <figcaption class="widget-head">
    <span class="widget-kicker" {_kicker}></span>
    <span class="widget-title" data-zh="{_zh}" data-en="{_en}"></span>
  </figcaption>
  <div class="widget-body"><p class="tx-fallback" data-zh="这张图需要启用 JavaScript。" data-en="This figure needs JavaScript."></p></div>
</figure>
"""



# Filled by discover(): "00-foundations/transformer.md" -> {"zh": Page, "en": Page | None}
BY_SRC = {}
NAV = {}


def estimate_minutes(src: Path) -> int:
    """Reading time straight from the markdown, so a widget can show it before the note is rendered."""
    text = re.sub(r"```.*?```", " ", src.read_text(encoding="utf-8"), flags=re.S)
    text = re.sub(r"<[^>]+>|[#>*_`|$-]", " ", text)
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"\b[\w'-]+\b", re.sub(r"[\u3400-\u9fff]", " ", text)))
    return max(1, math.ceil(cjk / 420 + latin / 220))


def note_link(page, note: str):
    """(href, title, minutes, has_own_language) for a note path, written as a source-relative
    .md link so rewrite_links resolves it exactly like a hand-written one."""
    pair = BY_SRC.get(note)
    if not pair:
        return None
    target = pair.get(page.lang) or pair["zh"]
    href = os.path.relpath(ROOT / note, page.src.parent).replace(os.sep, "/")
    return href, target.title, estimate_minutes(target.src), pair.get(page.lang) is not None


def both(d, zh: bool, key=""):
    """(text, "") for a bilingual entry in the page's own language. The second slot used to carry the
    other language as a small caption; pages now keep to one language, so it is always empty."""
    a, b = d.get(f"{key}zh" if key else "zh", ""), d.get(f"{key}en" if key else "en", "")
    return html.escape((a or b) if zh else (b or a)), ""


def block_head(page, anchor, title, desc) -> str:
    """One numbered home-page block: every block opens the same way, numbered in page order."""
    page.block_count = getattr(page, "block_count", 0) + 1
    num = f"{page.block_count:02d}"
    return (f'<header class="block-head" id="{anchor}"><span class="block-num">{num}</span>'
            f'<h2 class="block-title">{title}</h2><p class="block-desc">{desc}</p></header>')


def roadmap_html(page) -> str:
    """Career route -> tech stack -> knowledge breakdown.

    A route is a stack of layers (role-specific on top, shared foundations at the bottom); a layer
    lists the concrete tools and breaks down into knowledge topics; a topic links to a note or is
    honestly marked as planned. Everything is plain <details> + links, so it works without
    JavaScript; app.js only adds the route switcher, expand-all and the read-it checkmarks."""
    zh = page.lang == "zh"
    data = load_roadmap()
    tracks, layers = data.get("track", []), {l["id"]: l for l in data.get("layer", [])}
    if not tracks:
        return ""
    users = {}                     # layer id -> names of the routes that use it
    for track in tracks:
        for ref in track.get("layers", []):
            users.setdefault(ref["id"], []).append(track["zh" if zh else "en"])
    weight_label = {"core": "必备", "plus": "加分"} if zh else {"core": "Core", "plus": "Plus"}
    done_label = "标记为已读" if zh else "Mark as read"

    live = {(v.get("note"), v.get("anchor", "")) for v in data.get("viz", [])}

    def topic_html(topic):
        main, alt = both(topic, zh)
        if (topic.get("note"), topic.get("anchor", "")) in live and topic.get("anchor"):
            main += ' <i class="rm-live">live</i>'
        alt_html = f'<span class="rm-mod-alt">{alt}</span>' if alt else ""
        link = note_link(page, topic["note"]) if "note" in topic else None
        if "note" in topic and not link:
            print(f"  roadmap: no such note {topic['note']!r} -- shown as planned")
        if not link:
            return (f'<li class="rm-mod is-planned"><span class="rm-dot" aria-hidden="true"></span>'
                    f'<span class="rm-planned"><span class="rm-mod-title">{main}</span>{alt_html}'
                    f'<span class="rm-mod-meta">{"待补" if zh else "planned"}</span></span></li>'), None
        href, title, minutes, native = link
        anchor = topic.get("anchor", "")
        key = html.escape(topic["note"][:-3] + (f"#{anchor}" if anchor else ""), quote=True)
        href = html.escape(href + (f"#{anchor}" if anchor else ""), quote=True)
        meta = (f"{minutes} 分钟" if zh else f"{minutes} min") + ("" if native else " · 中文")
        return (f'<li class="rm-mod" data-key="{key}">'
                f'<button type="button" class="rm-check" aria-pressed="false" aria-label="{done_label}"></button>'
                f'<a href="{href}" title="{html.escape(title, quote=True)}"><span class="rm-mod-title">{main}</span>{alt_html}'
                f'<span class="rm-mod-meta">{meta}</span></a></li>'), key

    tabs, panels = [], []
    for ti, track in enumerate(tracks):
        name = html.escape(track["zh" if zh else "en"])
        kind = track.get("kind", "covered")
        side = kind in {"side", "draft"}          # dashed tab: not (yet) a fully written route
        tabs.append(f'<button type="button" class="rm-tab{" is-side" if side else ""}" role="tab" data-track="{track["id"]}" '
                    f'aria-selected="{"true" if ti == 0 else "false"}">{name}</button>')
        slabs, keys = [], set()
        for li, ref in enumerate(track.get("layers", [])):
            layer = layers.get(ref["id"])
            if not layer:
                continue
            main, alt = both(layer, zh)
            tools = layer.get("tools" if zh else "tools_en") or layer.get("tools", [])
            rendered = [topic_html(t) for t in layer.get("topics", [])]
            have = [k for _, k in rendered if k]
            keys.update(have)
            shared = [u for u in users.get(ref["id"], []) if u != track["zh" if zh else "en"]]
            shared_html = (f'<p class="sl-shared">{"这一层也出现在" if zh else "This layer is shared with"}：'
                           f'{html.escape(" · ".join(shared))}</p>' if shared else
                           f'<p class="sl-shared">{"这一层是这条路线专属的。" if zh else "This layer is specific to this route."}</p>')
            weight = ref.get("weight", "core")
            n_q = len(collect_bank(page).get(ref["id"], []))
            bank_link = note_link(page, QUESTION_BANK) if n_q else None
            if bank_link:
                label = f"这一层的 {n_q} 道考题 →" if zh else f"{n_q} interview questions for this layer →"
                shared_html += f'<p class="sl-questions"><a href="{html.escape(bank_link[0], quote=True)}#q-{ref["id"]}">{label}</a></p>'
            slabs.append(
                f'<details class="stack-layer is-{weight}"{" open" if li == 0 else ""}><summary>'
                f'<span class="sl-weight">{weight_label.get(weight, weight)}</span>'
                f'<span class="sl-name"><strong>{main}</strong>{f"<em>{alt}</em>" if alt else ""}</span>'
                f'<span class="sl-tools">{"".join(f"<i>{html.escape(t)}</i>" for t in tools)}</span>'
                f'<span class="sl-count" title="{"已有笔记 / 知识点" if zh else "notes written / topics"}">{len(have)} / {len(rendered)}</span>'
                f'</summary><div class="sl-body">{shared_html}'
                f'<ul class="rm-modules">{"".join(h for h, _ in rendered)}</ul></div></details>')
        # main = the author's own line; covered = written by the author, no badge; draft = an outline the
        # author intends to fill in; side = a route contributed by someone else (credited with `by`).
        badges = {"main": ("我的主线", "My main line"), "draft": ("骨架 · 笔记待补", "Outline · notes to come"),
                  "side": ("社区贡献", "Community route")}
        badge = ""
        if kind in badges:
            a_, b_ = badges[kind] if zh else badges[kind][::-1]
            by = f' · {html.escape(track["by"])}' if kind == "side" and track.get("by") else ""
            badge = f'<span class="rm-kind{"" if kind == "main" else " is-side"}">{a_}{by}</span>'
        caveat = ""
        if kind == "draft":
            caveat = ('<p class="rm-caveat">这条路线我打算聊，但现在只有提纲：stack 的分层是我准备写的大纲，'
                      '大部分知识点还没有笔记，如实标为待补。</p>' if zh else
                      '<p class="rm-caveat">I plan to write about this route, but for now it is an outline: the layers are the '
                      'structure I intend to fill in, and most topics have no note yet. They are marked as planned.</p>')
        elif kind == "side":
            caveat = ('<p class="rm-caveat">这条路线由社区贡献，不是站点作者本人的方向。</p>' if zh else
                      '<p class="rm-caveat">This route was contributed by the community; it is not the site author’s own direction.</p>')
        panels.append(
            f'<div class="rm-track" role="tabpanel" data-track="{track["id"]}" id="track-{track["id"]}">'
            f'<div class="rm-track-head"><strong class="rm-track-name">{name}</strong>'
            f'{badge}'
            f'<p class="rm-blurb">{both(track, zh, "blurb_")[0]}</p>{caveat}'
            f'<div class="rm-progress"><span class="rm-bar"><i></i></span>'
            f'<span class="rm-count">0 / {len(keys)}</span>'
            f'<button type="button" class="rm-expand" data-open="{"全部展开" if zh else "Expand all"}" '
            f'data-close="{"全部收起" if zh else "Collapse all"}"></button></div></div>'
            f'<div class="stack-axis" aria-hidden="true"><span>{"↑ 岗位专属" if zh else "↑ role-specific"}</span></div>'
            f'<div class="stack">{"".join(slabs)}</div>'
            f'<div class="stack-axis" aria-hidden="true"><span>{"↓ 通用基础" if zh else "↓ shared foundations"}</span></div></div>')

    head = block_head(page, "routes", "职业路线与 tech stack" if zh else "Career routes and their tech stack",
                      "先选一条职业路线，看它需要的 tech stack 一层一层是什么；点开任意一层，就是这一层的知识点，每个知识点跳到对应的笔记。"
                      if zh else
                      "Pick a career route and see its tech stack layer by layer. Open any layer for its knowledge breakdown; every topic jumps to the note that covers it.")
    source = f'https://github.com/{NAV.get("site", {}).get("repo", "")}/blob/main/site/roadmap.toml'
    note = ("圆点可以标记已读，进度只存在你自己的浏览器里。“待补”的知识点还没有笔记，不假装有。岗位要求变得很快，这张图有时效性。"
            if zh else
            "Tick a dot to mark a topic as read: progress is stored only in your browser. “Planned” topics have no note yet and do not pretend to. Role requirements change quickly, so this map is time-sensitive.")
    extend = (f'这里只有我自己在做或打算聊的方向。Data Scientist、Data Engineer、AI Infra 等路线不是我的 focus——'
              f'欢迎你来补：在 <a href="{source}">site/roadmap.toml</a> 里加一个 <code>[[track]]</code> 就行，文件开头写了怎么加。'
              if zh else
              f'Only directions I work on, or plan to write about, are here. Data Scientist, Data Engineer, AI Infra and others are not my focus. '
              f'You are welcome to add them: one <code>[[track]]</code> in <a href="{source}">site/roadmap.toml</a>, with instructions at the top of the file.')
    return (f'<section class="roadmap home-block" data-widget="roadmap" aria-labelledby="routes">{head}'
            f'<div class="rm-tabs" role="tablist">{"".join(tabs)}</div>'
            f'{"".join(panels)}<p class="rm-note">{note}</p><p class="rm-note rm-extend">{extend}</p></section>')


QUESTION_BANK = "interview/questions.md"


def top_level_details(body: str):
    """(summary, inner markdown) for each outermost <details> in a section; nested ones stay inside."""
    out, depth, start = [], 0, None
    for m in re.finditer(r"<details[^>]*>|</details>", body):
        if m.group(0).startswith("<details"):
            if depth == 0:
                start = m.end()
            depth += 1
        else:
            depth -= 1
            if depth == 0 and start is not None:
                block = body[start:m.start()]
                s = re.search(r"<summary>(.*?)</summary>", block, flags=re.S)
                if s:
                    out.append((re.sub(r"\s+", " ", strip_tags(s.group(1))).strip(), block[s.end():].strip()))
                start = None
    return out


def note_questions(src_path: Path):
    """(question, anchor, answer markdown) triples a note already contains: the Q&A of its interview
    section (answer included), and the items of its self-check box (no answer: go back and read).
    The note stays the single source; nothing is copied by hand."""
    text = src_path.read_text(encoding="utf-8")
    out = []
    parts = re.split(r"(?m)^(#{2,3} .*)$", text)
    for i in range(1, len(parts), 2):
        if re.search(r"面试|interview", parts[i], re.I):
            for q, answer in top_level_details(parts[i + 1]):
                if q and not re.match(r"(深挖|进阶|deep dive|deeper)", q, re.I):
                    out.append((q, "interview", answer))
    box = re.search(r'<div class="taste-check[^"]*">(.*?)</div>', text, flags=re.S)
    if box:
        for li in re.findall(r"<li>(.*?)</li>", box.group(1), flags=re.S):
            out.append((re.sub(r"\s+", " ", strip_tags(li)).strip(), "self-check", ""))
    return out


def rebase_links(md: str, note: str, page) -> str:
    """An answer lifted out of its note keeps working links: relative URLs are re-pointed from the
    note's folder to the page's folder, and bare #anchors go back to the note."""
    note_dir, here = (ROOT / note).parent, page.src.parent

    def fix(url):
        if re.match(r"^(https?:|mailto:|data:|/)", url):
            return url
        if url.startswith("#"):
            return os.path.relpath(ROOT / note, here).replace(os.sep, "/") + url
        p, _, frag = url.partition("#")
        return os.path.relpath(note_dir / p, here).replace(os.sep, "/") + (f"#{frag}" if frag else "")

    md = re.sub(r"\]\(([^)\s]+)\)", lambda m: "](" + fix(m.group(1)) + ")", md)
    return re.sub(r'\b(href|src)="([^"]+)"', lambda m: f'{m.group(1)}="{fix(m.group(2))}"', md)


def collect_bank(page):
    """{layer id: [(question, note path, anchor, note title, answer md)]}. A note feeds the first block that links it."""
    cache = getattr(collect_bank, "cache", {})
    if page.lang in cache:
        return cache[page.lang]
    data = load_roadmap()
    order = [l for a in data.get("area", []) for l in data.get("layer", []) if l.get("area") == a["id"]]
    bank, seen = {}, set()
    for layer in order:
        rows = []
        notes = [t["note"] for t in layer.get("topics", []) if "note" in t] + ([layer["home"]] if layer.get("home") else [])
        for note in dict.fromkeys(notes):
            pair = BY_SRC.get(note)
            if not pair or note in seen or note == QUESTION_BANK:
                continue
            seen.add(note)
            target = pair.get(page.lang) or pair["zh"]
            rows += [(q, note, anchor, target.title, answer) for q, anchor, answer in note_questions(target.src)]
        bank[layer["id"]] = rows
    cache[page.lang] = bank
    collect_bank.cache = cache
    return bank


def question_bank_md(page) -> str:
    """The quick-review side of the site. Same material as the knowledge blocks, different job:
    there you learn it, here you run through it before an interview. Questions with an answer in the
    notes open in place; self-check questions point back to the note.

    Emitted as *markdown*, so headings land in the page outline, math renders, and links go through
    the same rewriting as hand-written ones."""
    zh = page.lang == "zh"
    data = load_roadmap()
    bank, viz = collect_bank(page), data.get("viz", [])
    rel = lambda note: os.path.relpath(ROOT / note, page.src.parent).replace(os.sep, "/")
    esc = lambda s: s.replace("[", "\\[").replace("]", "\\]")
    total = sum(len(rows) for rows in bank.values())
    answered = sum(1 for rows in bank.values() for r in rows if r[4])
    tools = (f'<div class="qb-tools" data-qbank-tools><span class="qb-count">{total} {"道题" if zh else "questions"} · '
             f'{answered} {"道带答案" if zh else "with answers"}</span>'
             f'<button type="button" class="btn" data-qb="open">{"全部展开" if zh else "Open all"}</button>'
             f'<button type="button" class="btn" data-qb="close">{"全部收起" if zh else "Close all"}</button>'
             f'<button type="button" class="btn" data-qb="random">{"随机抽一题" if zh else "Random question"}</button></div>')
    out = [tools, ""]
    for area in data.get("area", []):
        layers = [l for l in data.get("layer", []) if l.get("area") == area["id"]]
        if not any(bank.get(l["id"]) for l in layers):
            continue
        out.append(f'\n## {area["zh" if zh else "en"]}\n')
        for layer in layers:
            rows = bank.get(layer["id"])
            if not rows:
                continue
            out.append(f'\n<a id="q-{layer["id"]}"></a>\n\n### {layer["zh" if zh else "en"]}\n')
            out.append(f'<p class="q-meta">{len(rows)} {"道题" if zh else "questions"}</p>\n')
            quick = [r for r in rows if r[4]]
            for q, note, anchor, title, answer in quick:
                source = f'<p class="q-src">{"出处" if zh else "From"}：<a href="{rel(note)}#{anchor}">{html.escape(title)}</a></p>'
                out.append(f'<details class="qa" markdown="1">\n<summary>{html.escape(q)}</summary>\n\n'
                           f'{rebase_links(answer, note, page)}\n\n{source}\n\n</details>\n')
            checks = [r for r in rows if not r[4]]
            if checks:
                out.append(f'<p class="q-sub">{"再问自己（答案在笔记里）" if zh else "Ask yourself (the answer is in the note)"}</p>\n')
                for q, note, anchor, title, _ in checks:
                    out.append(f'- [{esc(q)}]({rel(note)}#{anchor}) <span class="q-src">{html.escape(title)}</span>')
                out.append("")
            figs = [v for v in viz if v.get("layer") == layer["id"] and v.get("anchor")]
            if figs:
                links = " · ".join(f'[{esc(v["zh" if zh else "en"])}]({rel(v["note"])}#{v["anchor"]})' for v in figs)
                out.append(f'<p class="q-sub">{"想学透，而不只是过一遍：去知识板块动手玩" if zh else "To learn it rather than skim it: the live figures in the knowledge blocks"}</p>\n\n{links}\n')
    return "\n".join(out) + "\n"


def layer_stats(page, layer, viz):
    """(notes written, topics, live figures) for one knowledge block."""
    topics = layer.get("topics", [])
    have = sum(1 for t in topics if "note" in t and note_link(page, t["note"]))
    figures = sum(1 for v in viz if v.get("layer") == layer["id"])
    return have, len(topics), figures


def blocks_html(page) -> str:
    """Every knowledge block, grouped by area and independent of any route."""
    zh = page.lang == "zh"
    data = load_roadmap()
    layers, viz = data.get("layer", []), data.get("viz", [])
    if not layers:
        return ""
    used = {}
    for track in data.get("track", []):
        for ref in track.get("layers", []):
            used.setdefault(ref["id"], []).append(track["zh" if zh else "en"].split(" · ")[0])
    rows = []
    for area in data.get("area", []):
        cards = []
        for layer in (l for l in layers if l.get("area") == area["id"]):
            main, alt = both(layer, zh)
            have, total, figures = layer_stats(page, layer, viz)
            link = note_link(page, layer["home"]) if layer.get("home") else None
            meta = [(f"{have} / {total} 个知识点已有笔记" if zh else f"{have} / {total} topics written")]
            if figures:
                meta.append(f"{figures} 个交互图" if zh else f"{figures} live figures")
            n_q = len(collect_bank(page).get(layer["id"], []))
            if n_q:
                meta.append(f"{n_q} 道考题" if zh else f"{n_q} questions")
            routes = " · ".join(html.escape(r) for r in used.get(layer["id"], []))
            inner = (f'<strong>{main}</strong>{f"<em>{alt}</em>" if alt else ""}'
                     f'<span class="kb-meta">{" · ".join(meta)}</span>'
                     + (f'<span class="kb-routes">{"路线" if zh else "Routes"}：{routes}</span>' if routes else ""))
            if link and have:
                cards.append(f'<a class="kb-card" href="{html.escape(link[0], quote=True)}">{inner}</a>')
            else:
                cards.append(f'<div class="kb-card is-planned">{inner}</div>')
        a_main, a_alt = both(area, zh)
        rows.append(f'<div class="kb-area"><div class="kb-area-name"><strong>{a_main}</strong>'
                    f'{f"<em>{a_alt}</em>" if a_alt else ""}</div><div class="kb-grid">{"".join(cards)}</div></div>')
    head = block_head(page, "blocks", "知识板块" if zh else "Knowledge blocks",
                      "也可以按主题来读：大模型基础、fine-tuning、post-training、评估……每个板块都会标明已写好的笔记和交互图；虚线表示内容还没补上。"
                      if zh else
                      "By subject instead of by role: LLM foundations, fine-tuning and post-training, evaluation and so on. For each block: how many topics, how many are written, and whether there is something to play with. Dashed blocks have no notes yet.")
    return f'<section class="home-block" data-widget="blocks">{head}{"".join(rows)}</section>'


def gallery_html(page) -> str:
    """Everything on the site you can drag, click or train, in one place."""
    zh = page.lang == "zh"
    data = load_roadmap()
    names = {l["id"]: l for l in data.get("layer", [])}
    cards = []
    for v in data.get("viz", []):
        link = note_link(page, v["note"])
        if not link:
            continue
        href = link[0] + (f'#{v["anchor"]}' if v.get("anchor") else "")
        main, alt = both(v, zh)
        block = both(names[v["layer"]], zh)[0] if v.get("layer") in names else ("求职" if zh else "Career")
        cards.append(f'<a class="viz-card" href="{html.escape(href, quote=True)}"><span class="viz-kicker"><i>live</i>{block}</span>'
                     f'<strong>{main}</strong>{f"<em>{alt}</em>" if alt else ""}'
                     f'<span class="viz-desc">{both(v, zh, "desc_")[0]}</span></a>')
    if not cards:
        return ""
    head = block_head(page, "figures", "交互图解" if zh else "Live figures",
                      "可以拖动参数、点击步骤，或者试着训练一个小模型。结果在浏览器里实时计算，简化的演示例子也会标明。" if zh else
                      "Everything you can drag, click or train live. Numbers are computed in your browser, and toy examples are labelled as such.")
    return f'<section class="home-block" data-widget="gallery">{head}<div class="viz-grid">{"".join(cards)}</div></section>'


def threads_html(page) -> str:
    """Meta pieces: notes that belong to no single block because their job is to connect several."""
    zh = page.lang == "zh"
    data = load_roadmap()
    names = {l["id"]: l for l in data.get("layer", [])}
    cards = []
    for th in data.get("thread", []):
        main, alt = both(th, zh)
        chips = "".join(f'<i>{both(names[c], zh)[0]}</i>' for c in th.get("connects", []) if c in names)
        link = note_link(page, th["note"]) if "note" in th else None
        body = (f'<strong>{main}</strong>{f"<em>{alt}</em>" if alt else ""}'
                + (f'<span class="thread-desc">{both(th, zh, "desc_")[0]}</span>' if th.get("desc_zh") else "")
                + f'<span class="thread-chips">{chips}</span>')
        if link:
            meta = f"{link[2]} 分钟" if zh else f"{link[2]} min"
            cards.append(f'<a class="thread-card" href="{html.escape(link[0], quote=True)}">{body}<span class="thread-meta">{meta}</span></a>')
        else:
            cards.append(f'<div class="thread-card is-planned">{body}<span class="thread-meta">{"待补" if zh else "planned"}</span></div>')
    if not cards:
        return ""
    head = block_head(page, "threads", "串联：把几条线穿起来" if zh else "Threads: tying the lines together",
                      "想把前面学到的内容连起来，可以读这一组文章。每篇下方都标了它涉及的板块。" if zh else
                      "These pieces belong to no single block: their job is to explain how the blocks connect. Each one lists the blocks it ties together.")
    return f'<section class="home-block" data-widget="threads">{head}<div class="thread-grid">{"".join(cards)}</div></section>'


def category_home(cat):
    return BY_SRC.get(cat.get("home", ""))


def categories_html(page) -> str:
    """Category cards with descriptions, note counts, and entry points."""
    zh = page.lang == "zh"
    groups = NAV.get("group", [])
    cards = []
    for i, cat in enumerate(NAV.get("category", []), 1):
        mine = [g for g in groups if g.get("category") == cat["id"]]
        ids = {g["id"] for g in mine}
        count = sum(len(sec["pages"]) for sec in NAV.get("_sections", []) if sec["group"] in ids)
        link = note_link(page, cat.get("home", ""))
        if not link:
            continue
        main, alt = both(cat, zh)
        names = " · ".join(html.escape(g["zh" if zh else "en"]) for g in mine)
        unit = f"{count} 篇" if zh else f"{count} notes"
        cards.append(f'<a class="cat-card" href="{html.escape(link[0], quote=True)}">'
                     f'<span class="cat-num">{chr(64 + i)}</span><strong>{main}{f"<em>{alt}</em>" if alt else ""}</strong>'
                     f'<span class="cat-blurb">{both(cat, zh, "blurb_")[0]}</span>'
                     f'<span class="cat-meta">{unit} · {names}</span></a>')
    if not cards:
        return ""
    head = block_head(page, "library", "按主题浏览" if zh else "Browse by topic",
                      "可以按兴趣选一个入口，侧边栏和顶栏使用相同的分类。" if zh else
                      "Choose a topic to explore. The sidebar and top navigation use the same categories.")
    return f'<section class="home-block" data-widget="categories">{head}<div class="cat-cards">{"".join(cards)}</div></section>'


def about_head_html(page) -> str:
    zh = page.lang == "zh"
    return ('<section class="home-block about-block">' +
            block_head(page, "about", "关于这份笔记" if zh else "About these notes",
                       "这些笔记为什么写、怎么读，以及哪些内容适合放在这里。" if zh else
                       "What I am trying to understand, how the notes are written, and what is and is not public.") +
            '</section>')


DYNAMIC_WIDGETS = {"question-bank": question_bank_md,
                   "study-atlas": lambda page: curriculum.study_atlas(page, NAV, NAV.get('_sections', []), BY_SRC),
                   "reference-coverage": lambda page: curriculum.reference_coverage(page, BY_SRC, SITE / 'reference-coverage.toml'),
                   "appendix-coverage": lambda page: curriculum.appendix_coverage(page, BY_SRC, SITE / 'appendix-coverage.toml'),
                   "roadmap": roadmap_html, "blocks": blocks_html, "gallery": gallery_html,
                   "threads": threads_html, "categories": categories_html,
                   "about-head": about_head_html,
                   "topic-coverage": lambda page: topic_coverage.figure(coverage_rows(), page.lang,
                                                                        page.rel("learn/coverage.html" if page.lang == "zh" else "learn/coverage.en.html"))}


def coverage_rows():
    """Notes per topic and the current review pass (site/expression-review.toml)."""
    _, reviewed = topic_coverage.load_reviewed(SITE / "expression-review.toml", BY_SRC)
    return topic_coverage.rows(NAV, NAV.get("_sections", []), ROOT, reviewed)


def localize_shell(shell: str, page) -> str:
    """Widget shells carry their labels as data-zh / data-en pairs for app.js to fill in. Fill them
    at build time in the page's language instead, so the page reads right without JavaScript, and
    an English page ships no Chinese at all (its data-zh attributes are dropped)."""
    def fill(m):
        zh, en, rest = m.group(1), m.group(2), m.group(3)
        if page.lang == "en":
            return f'data-en="{en}"{rest}>{en}</'
        return f'data-zh="{zh}" data-en="{en}"{rest}>{zh}</'
    return re.sub(r'data-zh="([^"]*)"\s+data-en="([^"]*)"([^>]*)>\s*</', fill, shell)


def load_paper_figures():
    """site/paper_figures.toml: figures from papers, each with its credit and licence."""
    f = SITE / "paper_figures.toml"
    entries = tomllib.loads(f.read_text(encoding="utf-8")).get("figure", []) if f.exists() else []
    return {entry["id"]: entry for entry in entries}


PAPER_FIGURES = load_paper_figures()


def paper_figure_html(name: str, page=None) -> str:
    """A paper's own figure: embedded with its credit and licence when the licence allows
    copying (an `image` is registered), otherwise a card that opens the figure in the paper."""
    fig = PAPER_FIGURES.get(name)
    if fig is None:
        raise SystemExit(f"unknown paper figure {name!r}: register it in site/paper_figures.toml")
    zh = page is None or page.lang == "zh"
    esc = lambda value: html.escape(str(value), quote=True)
    caption = esc(fig["zh"] if zh else fig["en"])
    source = f'{esc(fig["credit"])} · {esc(fig["paper"])} · Figure {fig["figure"]}'
    label = "原文图" if zh else "From the paper"
    if "image" in fig:
        # relative to the note's source, like any markdown image; rewrite_links maps site/static/ to the built asset
        src = f'site/static/paper-figures/{fig["image"]}'
        src = page.rel(src) if page is not None else src
        changes = {"resized": ("缩小尺寸并转为 WebP", "resized and converted to WebP"),
                   "converted": ("转为 WebP", "converted to WebP")}[fig["changes"]]
        tall = " is-tall" if fig["height"] > 0.8 * fig["width"] else ""
        credit = (f'图源：<a href="{esc(fig["url"])}">{source}</a> · <a href="{esc(fig["license_url"])}">{esc(fig["license"])}</a> · {changes[0]}'
                  if zh else
                  f'Source: <a href="{esc(fig["url"])}">{source}</a> · <a href="{esc(fig["license_url"])}">{esc(fig["license"])}</a> · {changes[1]}')
        return (f'<figure class="paper-fig{tall}" id="pf-{esc(name)}">'
                f'<a class="pf-image" href="{esc(src)}"><img src="{esc(src)}" width="{fig["width"]}" height="{fig["height"]}" '
                f'alt="{caption}" loading="lazy" decoding="async"></a>'
                f'<figcaption><span class="pf-label">{label}</span>{caption}<span class="pf-credit">{credit}</span></figcaption></figure>')
    go = "在原论文里看" if zh else "Open in the paper"
    return (f'<aside class="paper-fig-link" id="pf-{esc(name)}"><a href="{esc(fig["url"])}">'
            f'<span class="pf-label">{label}</span><span class="pf-title">{source}</span>'
            f'<span class="pf-desc">{caption}</span><span class="pf-go">{go} ↗</span></a></aside>')


def expand_widgets(md_text: str, page=None) -> str:
    def expand(m):
        name = m.group(1)
        if page is not None and name in DYNAMIC_WIDGETS:
            return DYNAMIC_WIDGETS[name](page)
        shell = WIDGETS.get(name, "")
        return localize_shell(shell, page) if page is not None else shell
    md_text = re.sub(r"<!--\s*paper-figure:([a-z0-9_.-]+)\s*-->", lambda m: paper_figure_html(m.group(1), page), md_text)
    return re.sub(r"<!--\s*widget:([a-z0-9_-]+)\s*-->", expand, md_text)


# --------------------------------------------------------------------------- #
# page model
# --------------------------------------------------------------------------- #
def split_heading_id(heading: str):
    match = re.search(r"\s+\{#([^{}\s]+)\}\s*$", heading)
    if match:
        return heading[:match.start()].strip(), match.group(1)
    return heading.strip(), None


def read_title(src: Path) -> str:
    """First h1 of a file. Needed for every page before any sidebar is built."""
    text = src.read_text(encoding="utf-8")
    heading = re.search(r"^#\s+(.+)$", text, re.M)
    if heading:
        return split_heading_id(heading.group(1))[0]
    heading = re.search(r"<h1\b[^>]*>(.*?)</h1>", text, re.S | re.I)
    return html.unescape(strip_tags(heading.group(1))) if heading else src.stem


class Page:
    def __init__(self, src: Path, section, lang):
        self.src = src
        self.section = section
        self.lang = lang
        rel = src.relative_to(ROOT)
        home = rel.as_posix() in {"README.md", "README.zh.md"}
        name = rel.name[:-6] if lang == "en" else rel.name[:-3]  # strip .en.md / .md
        stem = "index" if name == "README" else name
        self.out_rel = rel.parent / (stem + (".en.html" if lang == "en" else ".html"))
        if home:
            self.out_rel = Path("index.html" if lang == "en" else "index.zh.html")
        self.url = str(self.out_rel).replace(os.sep, "/")
        self.depth = len(self.out_rel.parts) - 1
        self.title = read_title(src)
        self.toc = []
        self.body = ""
        self.text = ""
        self.updated = last_updated(str(rel))
        if home:
            self.kind = "home"
        elif rel.parent.as_posix() == "community" and rel.stem.replace(".en", "") in {"roadmap", "updates"}:
            self.kind = "project"
        elif any(part in {"code", "projects", "modules"} for part in rel.parts):
            self.kind = "workshop"
        elif rel.name in {"README.md", "README.en.md"}:
            self.kind = "index"
        elif any(key in rel.stem for key in ("interview", "hand-write", "EDITORIAL")):
            self.kind = "guide"
        else:
            self.kind = "article"
        self.position = 1
        self.section_count = 1
        self.read_minutes = 1
        self.reviewed = ""
        self.previous = None
        self.next = None

    def rel(self, target: str) -> str:
        return ("../" * self.depth) + target if self.depth else target


def discover(nav):
    pages, sections = [], []
    # A section may `include` notes that live in another folder. Their URLs stay where the
    # file is (no redirects, no broken links); only the sidebar grouping and prev/next move.
    claimed = {x for sec in nav["section"] for x in sec.get("include", [])}
    for sec in nav["section"]:
        d = ROOT if sec["dir"] == "." else ROOT / sec["dir"]
        include = [ROOT / x for x in sec.get("include", []) if (ROOT / x).exists()]
        if not d.exists() and not include:
            continue
        order = sec.get("order", [])
        files = ([p for p in d.glob("*.md") if not p.name.endswith(".en.md")]
                 if d.exists() else [])
        files = [p for p in files if p.relative_to(ROOT).as_posix() not in claimed] + include
        if sec["dir"] == ".":
            files = [p for p in files if p.name in order]
        rank = {n: i for i, n in enumerate(order)}
        files.sort(key=lambda p: (rank.get(p.name, len(order)), p.name))
        entry = {"zh": sec["zh"], "en": sec["en"], "dir": sec["dir"],
                 "group": sec.get("group", "reference"),
                 "home": sec.get("home", ""), "sequence": sec.get("sequence", 0),
                 "intro_zh": sec.get("intro_zh", ""), "intro_en": sec.get("intro_en", ""),
                 "pages": []}
        for f in files:
            home = f == ROOT / "README.md"
            zh = Page(ROOT / "README.zh.md" if home else f, entry, "zh")
            en_src = f if home else f.with_name(f.stem + ".en.md")
            en = Page(en_src, entry, "en") if en_src.exists() else None
            zh.sibling, entry_pages = en, entry["pages"]
            if en:
                en.sibling = zh
            entry_pages.append({"zh": zh, "en": en})
            BY_SRC[f.relative_to(ROOT).as_posix()] = {"zh": zh, "en": en}
            pages.append(zh)
            if en:
                pages.append(en)
        if entry["pages"]:
            count = len(entry["pages"])
            for position, pair in enumerate(entry["pages"], 1):
                for item in pair.values():
                    if item:
                        item.position = position
                        item.section_count = count
            for language in ("zh", "en"):
                ordered = [pair[language] for pair in entry["pages"] if pair[language]]
                for index, item in enumerate(ordered):
                    item.previous = ordered[index - 1] if index else None
                    item.next = ordered[index + 1] if index + 1 < len(ordered) else None
            sections.append(entry)
    group_order = {group['id']: position for position, group in enumerate(nav.get('group', []))}
    sections.sort(key=lambda section: (group_order.get(section['group'], len(group_order)), section['sequence']))
    curriculum.validate_catalog(nav, sections, BY_SRC)
    NAV.update(nav)
    NAV["_sections"] = sections
    return pages, sections


def write_redirects(nav, known):
    """Generate small redirect pages for published URLs moved into topic folders.

    A target may name a section (`index.html#contributors`); the old page's own
    anchors mean nothing there, so only a target without one keeps the reader's hash."""
    redirects = nav.get("redirects", {})
    for old_url, target_url in redirects.items():
        target_page, _, fragment = target_url.partition("#")
        if target_page not in known:
            raise ValueError(f"redirect target is not a generated page: {target_url}")
        old_path = Path(old_url)
        relative_target = os.path.relpath(target_page, old_path.parent).replace(os.sep, "/")
        if old_url.endswith(".en.html") and target_page == "index.html":
            relative_target += "?lang=en"
        if fragment:
            relative_target += f"#{fragment}"
        escaped_target = html.escape(relative_target, quote=True)
        script_target = json.dumps(relative_target) + ("" if fragment else " + location.hash")
        page = f"""<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={escaped_target}">
<link rel="canonical" href="{escaped_target}">
<title>Moved</title>
<script>location.replace({script_target})</script>
<p>This page moved to <a href="{escaped_target}">{escaped_target}</a>.</p>
"""
        dest = OUT / old_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page, encoding="utf-8")
    return len(redirects)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def protect_mermaid(text):
    """```mermaid fences must reach the browser as <pre class="mermaid">, not as
    syntax-highlighted code. GitHub renders these natively, so the source stays
    a plain fence."""
    return re.sub(r"^```mermaid\n(.*?)^```\s*$",
                  lambda m: '<pre class="mermaid">\n' + html.escape(m.group(1)) + "</pre>",
                  text, flags=re.S | re.M)


def protect_math(text):
    r"""Keep TeX delimiters opaque while Python-Markdown parses prose.

    Python-Markdown otherwise treats underscores inside ``$...$`` as emphasis
    and consumes TeX escapes such as ``\!``. Code is stashed first so dollar
    signs in examples are never mistaken for math.
    """
    code_stash = []
    math_stash = []

    def stash_code(match):
        token = f"CODESTASH{len(code_stash):06d}END"
        code_stash.append((token, match.group(0)))
        return token

    fenced = re.compile(
        r"^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^ {0,3}(?P=fence)[ \t]*(?:\n|$)",
        flags=re.S | re.M,
    )
    inline_code = re.compile(r"(`+).*?\1", flags=re.S)
    protected = fenced.sub(stash_code, text)
    protected = inline_code.sub(stash_code, protected)

    def stash_math(match):
        token = f"MATHSTASH{len(math_stash):06d}END"
        math_stash.append((token, match.group(0)))
        return token

    protected = re.sub(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]", stash_math, protected,
                       flags=re.S)
    protected = re.sub(r"(?<!\\)\\\(([^\n]+?)(?<!\\)\\\)", stash_math, protected)
    protected = re.sub(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", stash_math, protected,
                       flags=re.S)
    protected = re.sub(
        r"(?<!\\)(?<!\$)\$(?!\$|\s)([^\n]*?\S)(?<!\\)\$(?!\$)",
        stash_math,
        protected,
    )

    for token, code in code_stash:
        protected = protected.replace(token, code)
    return protected, math_stash


def restore_math(text, math_stash):
    for token, tex in math_stash:
        text = text.replace(token, html.escape(tex, quote=False))
    return text


def restore_math_toc(body, toc, math_stash):
    """Restore math in heading labels and replace placeholder-based anchors."""
    used_ids = set()
    id_replacements = []

    def without_delimiters(tex):
        for opening, closing in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"),
                                 (r"\(", r"\)")):
            if tex.startswith(opening) and tex.endswith(closing):
                return tex[len(opening):-len(closing)]
        return tex

    slug_replacements = [
        (slugify(token, "-"), slugify(without_delimiters(tex), "-"))
        for token, tex in math_stash
    ]

    def visit(tokens):
        for item in tokens:
            item["name"] = restore_math(item["name"], math_stash)
            old_id = item["id"]
            new_id = old_id
            for token_slug, math_slug in slug_replacements:
                new_id = new_id.replace(token_slug, math_slug)
            base = new_id
            suffix = 1
            while new_id in used_ids:
                new_id = f"{base}_{suffix}"
                suffix += 1
            used_ids.add(new_id)
            item["id"] = new_id
            if new_id != old_id:
                id_replacements.append((old_id, new_id))
            visit(item.get("children", []))

    visit(toc)
    for old_id, new_id in id_replacements:
        body = body.replace(f'id="{old_id}"', f'id="{new_id}"')
    return body, toc


def wrap_tables(h):
    """Tables scroll inside their own box; the page body never scrolls sideways."""
    return re.sub(r"<table>", '<div class="table-wrap"><table>', h).replace(
        "</table>", "</table></div>")


CJK = "\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef"
CJK_BREAKS = re.compile(rf"(?<=[{CJK}])[ \t]*\n[ \t]*(?=[{CJK}\u201c\u2018])"
                        rf"|(?<=[{CJK}\u201d\u2019])[ \t]*\n[ \t]*(?=[{CJK}])")
RAW_BLOCKS = re.compile(r"<(pre|code|script|style|textarea)\b.*?</\1>", re.S | re.I)


def join_cjk_breaks(body):
    """Hard-wrapped Chinese paragraphs keep their source line breaks, and Chrome draws each one
    as a space ("是否 退化"). Join breaks that fall between two Chinese characters; a break next
    to Latin text stays a space, and code, preformatted blocks and scripts are left alone."""
    out, last = [], 0
    for block in RAW_BLOCKS.finditer(body):
        out.append(CJK_BREAKS.sub("", body[last:block.start()]))
        out.append(block.group(0))
        last = block.end()
    out.append(CJK_BREAKS.sub("", body[last:]))
    return "".join(out)


def render_markdown(text):
    text, math_stash = protect_math(text)
    md = markdown.Markdown(extensions=[
        "fenced_code", "tables", "footnotes", "attr_list", "sane_lists", "md_in_html",
        TocExtension(anchorlink=False, permalink=False, toc_depth="2-3"),
        "codehilite",
    ], extension_configs={"codehilite": {"guess_lang": False, "css_class": "hl"}})
    body = join_cjk_breaks(md.convert(text))
    toc = getattr(md, "toc_tokens", [])
    body = restore_math(body, math_stash)
    body, toc = restore_math_toc(body, toc, math_stash)
    return body, toc


def strip_tags(h):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h)).strip()


def build_page(page: Page, terms, repo: str, known: set):
    raw = page.src.read_text(encoding="utf-8")
    # A NUL byte means some editing pass left a placeholder behind and ate the
    # text around it. Silent in a diff, invisible on screen, and it destroys
    # links. Fail the build rather than publish it.
    if "\x00" in raw:
        bad = [i for i, l in enumerate(raw.splitlines(), 1) if "\x00" in l]
        raise SystemExit(f"{page.src}: NUL byte on line(s) {bad} -- corrupted source")
    raw = expand_widgets(raw, page)
    raw = protect_mermaid(raw)

    m = re.search(r"^#\s+(.+)$", raw, re.M)
    page.title, heading_id = split_heading_id(m.group(1)) if m else (read_title(page.src), None)
    anchor = f'<span id="{html.escape(heading_id, quote=True)}"></span>' if heading_id else ""
    raw = re.sub(r"^#\s+.+$", lambda match: anchor, raw, count=1, flags=re.M)
    if page.kind == "home":
        raw = re.sub(r"<!-- repository-only -->.*?<!-- /repository-only -->",
                     "", raw, flags=re.S)
        # Useful on GitHub, but redundant (and oddly circular) on the site itself.
        raw = re.sub(
            r"^###\s+[^\n]*(?:阅读请到|Read online|Read it at)[^\n]*\n(?:\n)?",
            "",
            raw,
            count=1,
            flags=re.M,
        )
        home_details = {"怎么用这些笔记": "how-to-use-the-notes",
                        "各个板块放什么": "whats-in-each-section",
                        "现在在做": "now", "主题覆盖": "topic-coverage", "走到现在": "how-we-got-here"}
        for title, identifier in home_details.items():
            raw = re.sub(r"^### " + re.escape(title) + r"$",
                         f"### {title} {{#{identifier}}}", raw, flags=re.M)
        # a section added before the numbered legacy anchors (_1 … _5) gets an English id so they don't shift
        for title, identifier in {"路线图": "roadmap"}.items():
            raw = re.sub(r"^## " + re.escape(title) + r"$",
                         f"## {title} {{#{identifier}}}", raw, flags=re.M)
    meta = re.search(r"^>\s*(?:阅读时间|Reading time)[:：]?\s*(.+)$", raw, re.M)
    if meta:
        parts = [p.strip() for p in meta.group(1).split("·")]
        for part in parts:
            hit = re.search(r"(?:最近审阅|Last reviewed)[:：]?\s*([0-9]{4}-[0-9]{2})", part)
            if hit:
                page.reviewed = hit.group(1)
        raw = raw[:meta.start()] + raw[meta.end():]
    # the lang switcher line right under the title is redundant on the site
    raw = re.sub(r"^\*\*?(中文|English)\*\*?\s*[·|].*$", "", raw, count=1, flags=re.M)
    raw = re.sub(r"^\[中文\]\([^)]*\)\s*[·|].*$", "", raw, count=1, flags=re.M)

    body, toc = render_markdown(raw)
    body = technology_chronicle.expand(body, page.lang)
    used = []
    if page.lang == "zh":
        body, used = annotate(body, glossary_for_page(page, terms))
    body = rewrite_links(body, page, repo, known)
    # stable targets for the question bank, whatever the heading slug turned out to be (notes only:
    # the home page's "interview coach" heading is not a question bank)
    if page.url not in {"index.html", "index.zh.html"}:
        body = re.sub(r'(<h[23] id="[^"]*">)(?=[^<]*(?:面试|[Ii]nterview))', r'<a id="interview"></a>\1', body, count=1)
    body = body.replace('<div class="taste-check', '<div id="self-check" class="taste-check', 1)
    page.body = wrap_tables(body)
    page.toc = [{"id": t["id"], "name": strip_tags(t["name"]), "level": t["level"],
                 "children": [{"id": c["id"], "name": strip_tags(c["name"])}
                              for c in t.get("children", [])]}
                for t in toc]
    # the generated home blocks (routes, category cards) are navigation, not reading
    prose_only = re.sub(r'<section class="[^"]*home-block.*?</section>', " ", body, flags=re.S)
    page.text = strip_tags(prose_only)[:1500]
    page.search_text = html.unescape(strip_tags(prose_only))
    search_prose = re.sub(r'<(pre|svg|script|style)\b[^>]*>.*?</\1>', ' ', prose_only, flags=re.S)
    page.search_preview = html.unescape(strip_tags(search_prose))
    plain = strip_tags(prose_only)
    cjk = len(re.findall(r"[\u3400-\u9fff]", plain))
    latin = len(re.findall(r"\b[\w'-]+\b", re.sub(r"[\u3400-\u9fff]", " ", plain)))
    page.read_minutes = max(1, math.ceil(cjk / 420 + latin / 220))
    page.glossary = used
    return page


# --------------------------------------------------------------------------- #
# html assembly
# --------------------------------------------------------------------------- #
def nav_label(page) -> str:
    """The short name a note goes by in the sidebar: nav.toml [label] if listed, else its
    title up to the first colon ("Tokenization：从文本到 ID" -> "Tokenization")."""
    key = str(page.src.relative_to(ROOT)).replace(os.sep, "/")
    key = key[:-6] + ".md" if key.endswith(".en.md") else key
    if key == "README.zh.md":
        key = "README.md"
    pair = NAV.get("label", {}).get(key)
    if pair:
        return pair[0] if page.lang == "zh" else pair[1]
    head = re.split(r"[：:]", page.title, maxsplit=1)[0].strip()
    return head if 1 < len(head) < len(page.title) else page.title


def section_html(page, sec, show_head=True):
    """Render a chapter with native folding when its parent contains multiple chapters."""
    label = html.escape(sec["zh" if page.lang == "zh" else "en"])
    items, has_active = [], False
    for pair in sec["pages"]:
        target = pair[page.lang] or pair["zh"]
        active = target.url == page.url
        has_active = has_active or active
        cls = ' class="active" aria-current="page"' if active else ""
        note = target.src.relative_to(ROOT).as_posix().replace('.en.md', '.md')
        if note == "README.zh.md":
            note = "README.md"
        items.append(f'<li><a{cls} data-note="{html.escape(note, quote=True)}" data-progress-id="{prep_progress.identifier(note)}" '
                     f'href="{page.rel(target.url)}" title="{html.escape(target.title, quote=True)}">'
                     f'{html.escape(nav_label(target))}</a></li>')
    links = f'<ul class="chapter-pages">{"".join(items)}</ul>'
    if show_head:
        key = html.escape(sec['dir'], quote=True)
        links = (f'<details data-chapter="{key}"{" open" if has_active else ""}>'
                 f'<summary><span class="chapter-chevron" aria-hidden="true">›</span>'
                 f'<span class="grp-name">{label}</span>'
                 f'<span class="grp-count">{len(items)}</span></summary>{links}</details>')
    return f'<li class="sec{"" if show_head else " sec-flat"}">{links}</li>', has_active


def sidebar_html(page, sections, groups):
    """Show the current direction; keep the full catalog so saved notes resolve to their IDs."""
    by_group = {}
    for sec in sections:
        by_group.setdefault(sec.get("group", "reference"), []).append(sec)

    out, seen_zone = [], None
    cats = {c["id"]: c for c in NAV.get("category", [])}
    zones = {z["id"]: z for z in NAV.get("zone", [])}
    scope = page_category(page)
    visible_groups = [group for group in groups if group.get("category") == scope]
    ordered = sorted(groups, key=lambda g: list(cats).index(g["category"]) if g.get("category") in cats else -1)
    for g in ordered:
        if not g.get("category"):          # "start" is the home page itself; the top bar links it
            continue
        secs = [s for s in by_group.get(g["id"], []) if s["pages"]]
        if not secs:
            continue
        visible = g["category"] == scope
        category_attr = (f' data-category="{html.escape(g["category"], quote=True)}"'
                         f'{"" if visible else " hidden"}')
        zone = g.get("zone")                # study notes split by what each role is tested on
        if zone in zones and zone != seen_zone:
            seen_zone = zone
            zone_active = any(group.get("zone") == zone and group["id"] == page.section.get("group") for group in groups)
            out.append(f'<li class="zone{" is-current" if zone_active else ""}"{category_attr}>'
                       f'<span>{both(zones[zone], page.lang == "zh")[0]}</span>'
                       f'{"<small>当前</small>" if zone_active and page.lang == "zh" else "<small>Current</small>" if zone_active else ""}</li>')
        # a heading only where it adds something: several sections in the block, and
        # more than one note in this one (a lone note's label already names it)
        rendered = [section_html(page, s, len(secs) > 1 and len(s["pages"]) > 1) for s in secs]
        body = "".join(h for h, _ in rendered)
        is_active = any(a for _, a in rendered)
        n_pages = sum(len(s["pages"]) for s in secs)
        label = html.escape(g["zh" if page.lang == "zh" else "en"])
        if n_pages == 1:
            out.append(f'<li class="grp grp-single"{category_attr}><div data-grp="{g["id"]}">'
                       f'<span class="grp-name" hidden>{label}</span><ul class="grp-body">{body}</ul></div></li>')
            continue
        expanded = is_active or (visible and len(visible_groups) <= 3)
        out.append(
            f'<li class="grp"{category_attr}><details data-grp="{g["id"]}"'
            f'{" open" if expanded else ""}>'
            f'<summary><svg class="chev" viewBox="0 0 12 12" width="11" height="11" '
            f'aria-hidden="true"><path d="M4 2.5 L7.5 6 L4 9.5" fill="none" '
            f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
            f'stroke-linejoin="round"/></svg>'
            f'<span class="grp-name">{label}</span>'
            f'<span class="grp-count">{n_pages} {"篇" if page.lang == "zh" else "notes"}</span></summary>'
            f'<ul class="grp-body">{body}</ul></details></li>')
    zh = page.lang == 'zh'
    context = cats[scope]["zh" if zh else "en"] if scope in cats else ("本页导航" if zh else "On this page")
    route_url = 'learn/choose-your-path.html' if zh else 'learn/choose-your-path.en.html'
    route_link = (f'<a class="side-route-link" href="{page.rel(route_url)}">'
                  f'{"按目标找路线" if zh else "Find a route for your goal"}'
                  '<span aria-hidden="true"> →</span></a>') if page.url != route_url else ''
    local_toc = ""
    if scope not in cats:
        local_toc = (f'<div class="side-page-toc">{toc_html(page)}</div>' if getattr(page, "toc", [])
                     else f'<p class="side-empty">{"从顶部选择一个板块开始阅读。" if zh else "Choose a direction from the top navigation to get started."}</p>')
    return (
        f'<div class="side-context">{html.escape(context)}</div>'
        f'{route_link}'
        f'<div class="side-scroll"><div id="side-directory">{local_toc}'
        f'<div class="side-directory-actions" hidden>'
        f'<button type="button" data-side-current>{"定位本章" if zh else "Current chapter"}</button>'
        f'<button type="button" data-side-collapse>{"收起其他" if zh else "Collapse others"}</button></div>'
        f'<ul class="nav">{"".join(out)}</ul></div>'
        '</div>')


def page_category(page):
    """Which top-level direction a page belongs to ("home" for the roadmap page)."""
    if page.url in {"index.html", "index.zh.html"}:
        return "home"
    groups = {g["id"]: g for g in NAV.get("group", [])}
    return groups.get(page.section.get("group"), {}).get("category")


def tabs_html(page):
    """Level 1, in the top bar on every page: the roadmap plus the big directions."""
    zh = page.lang == "zh"
    current = page_category(page)
    home = {"id": "home", "zh": "首页", "en": "Home"}
    items = [(home, page.rel("index.zh.html" if zh else "index.html?lang=en"))]
    for cat in NAV.get("category", []):
        pair = category_home(cat)
        target = pair and (pair.get(page.lang) or pair["zh"])
        if target:
            items.append((cat, page.rel(target.url)))
    active = ' class="active" aria-current="true"'
    links = "".join(f'<a href="{href}"{active if cat["id"] == current else ""}>'
                    f'<span>{both(cat, zh)[0]}</span></a>' for cat, href in items)
    site = NAV.get("site", {})
    if site.get("author_url"):
        author = html.escape(site["author_zh" if zh else "author_en"])
        links = (f'<a class="tab-author" href="{site["author_url"]}"><span>{author}</span></a>'
                 f'<i class="tab-sep" aria-hidden="true"></i>') + links
    label_nav = "大方向" if zh else "Directions"
    return f'<nav class="topbar-tabs" aria-label="{label_nav}">{links}</nav>'


def hero_html(page) -> str:
    """The home page opens on the night sky with one line; other pages have no hero."""
    if page.url not in {"index.html", "index.zh.html"}:
        return ""
    zh = page.lang == "zh"
    site = NAV.get("site", {})
    kicker = site["title_zh" if zh else "title_en"]
    title = site["tagline_zh" if zh else "tagline_en"]
    label = "封面" if zh else "Cover"
    hint = "往下看" if zh else "Scroll to the notes"
    return (f'<section class="hero" aria-label="{label}"><p class="hero-kicker">{html.escape(kicker.strip(" ·"))}</p>'
            f'<h1 class="hero-title">{html.escape(title)}</h1>'
            f'<a class="hero-scroll" href="#content" aria-label="{hint}"><span></span></a></section>')


def toc_html(page):
    if not page.toc:
        return ""
    li = []
    for t in page.toc:
        li.append(f'<li><a href="#{t["id"]}">{html.escape(t["name"])}</a></li>')
        for c in t["children"]:
            li.append(f'<li class="l3"><a href="#{c["id"]}">{html.escape(c["name"])}</a></li>')
    return f'<ul>{"".join(li)}</ul>'


def glossary_html(page):
    """A collapsed, mobile-friendly copy of the inline glossary annotations."""
    terms = getattr(page, "glossary", [])
    if page.lang != "zh" or not terms:
        return ""
    rows = []
    for term in terms:
        zh = html.escape(term["zh"])
        en = html.escape(term["en"])
        gloss = html.escape(term["gloss"])
        rows.append(f'<div class="gloss-row"><dt>{zh} <span>{en}</span></dt>'
                    f'<dd>{gloss}</dd></div>')
    return (f'<section class="page-glossary" aria-label="本页术语"><details>'
            f'<summary>本页术语 <span>{len(rows)}</span></summary>'
            f'<dl>{"".join(rows)}</dl></details></section>')


def chapter_home(page):
    source = page.section.get('home')
    if source and source in BY_SRC:
        return BY_SRC[source].get(page.lang) or BY_SRC[source].get('zh')
    for pair in page.section.get("pages", []):
        target = pair.get(page.lang) or pair.get("zh")
        if target and target.src.name in {"README.md", "README.en.md"}:
            return target
    return None


def page_header_html(page):
    zh = page.lang == "zh"
    section = page.section["zh" if zh else "en"]
    labels = {
        "home": "学习笔记" if zh else "Study notes",
        "index": "本章导读" if zh else "Chapter overview",
        "workshop": "实践" if zh else "Practice",
        "guide": "参考资料" if zh else "Reference",
        "article": "概念笔记" if zh else "Concept note",
        "project": "项目进展" if zh else "Project updates",
    }
    if page.section.get("group") == "career":
        labels["article"] = "求职记录" if zh else "Career journal"
    if page.section.get("group") == "study":
        labels["article"] = "阅读导航" if zh else "Reading guide"
    read = f"约 {page.read_minutes} 分钟" if zh else f"{page.read_minutes} min read"
    position = ((f"本章 {page.position} / {page.section_count} 篇" if zh else f"Note {page.position} of {page.section_count}")
                if page.kind not in {"home", "project"} and page.section_count > 1 else "")
    reviewed = (("最近核对 " if zh else "Reviewed ") + page.reviewed
                if page.reviewed else "")
    number_match = re.match(r"^(\d\d)-", page.section["dir"])
    chapter_mark = (f'<em class="chapter-mark" aria-hidden="true">'
                    f'{number_match.group(1)}</em>' if number_match else "")
    bits = [f'<span>{html.escape(read)}</span>']
    if reviewed:
        bits.append(f'<span>{html.escape(reviewed)}</span>')
    if position:
        bits.append(f'<span class="page-position">{position}</span>')
    overview = chapter_home(page)
    section_label = html.escape(section)
    if overview and overview.url != page.url:
        section_label = f'<a href="{html.escape(page.rel(overview.url), quote=True)}">{section_label}</a>'
    tools = ""
    if page.kind not in {"home", "index"}:
        note = html.escape(page.src.relative_to(ROOT).as_posix().replace(".en.md", ".md"), quote=True)
        tools = (f'<div class="note-tools" data-reading-note="{note}" data-progress-id="{prep_progress.identifier(note)}" hidden>'
                 f'<button type="button" data-side-save aria-pressed="false">{"收藏本页" if zh else "Save this page"}</button>'
                 '<a class="note-resume" data-note-resume hidden></a>'
                 f'<span class="note-tool-status" role="status" aria-live="polite"></span></div>')
    return f"""
<header class="article-head">
  {chapter_mark}
  <div class="article-kicker"><span>{html.escape(labels[page.kind])}</span>
    <i>{section_label}</i></div>
  <h1>{html.escape(page.title)}</h1>
  <div class="article-meta">{'<b aria-hidden="true"></b>'.join(bits)}</div>
  {tools}
</header>"""


def mobile_toc_html(page):
    toc = toc_html(page)
    if not toc:
        return ""
    label = "查看本页目录" if page.lang == "zh" else "On this page"
    return (f'<details class="mobile-toc"><summary>{label}'
            f'<span>{len(page.toc)}</span></summary><nav>{toc}</nav></details>')


def page_nav_html(page):
    if page.kind == "home":
        return ""
    if not page.previous and not page.next:
        return ""
    zh = page.lang == "zh"
    items = []
    for direction, target in (("prev", page.previous), ("next", page.next)):
        if not target:
            items.append('<span class="page-turn-empty"></span>')
            continue
        label = (("上一篇" if zh else "Previous") if direction == "prev"
                 else ("下一篇" if zh else "Next"))
        arrow = "←" if direction == "prev" else "→"
        items.append(
            f'<a class="page-turn-{direction}" href="{page.rel(target.url)}">'
            f'<small>{arrow} {label}</small><strong>{html.escape(target.title)}</strong></a>')
    overview = chapter_home(page)
    context = ""
    if overview and overview.url != page.url:
        label = "回到本章目录" if zh else "Back to chapter overview"
        context = (f'<div class="page-turn-context"><a href="{html.escape(page.rel(overview.url), quote=True)}">'
                   f'{label} · {html.escape(page.section["zh" if zh else "en"])}</a></div>')
    return f'<nav class="page-turn" aria-label="{"继续阅读" if zh else "Continue reading"}">' + context + "".join(items) + "</nav>"


def footer_html(page, repo, built):
    when = page.updated[:10] if page.updated else built[:10]
    src = f"https://github.com/{repo}/blob/main/{page.src.relative_to(ROOT)}"
    zh = page.lang == "zh"
    return f"""
<footer class="page-foot">
  {collaboration.page_panel(page, collaboration.load_config(), repo)}
  <div class="foot-meta">
    <span class="updated">{'最后更新' if zh else 'Last updated'}
      <time datetime="{page.updated or built}">{when}</time></span>
    <span class="sep">·</span>
    <a href="{src}">{'查看源文件' if zh else 'View source'}</a>
  </div>
</footer>"""


def share_description(page) -> str:
    """The line a chat app or social card shows under the title: the note's own
    one-sentence framing (the recipe block's "解决什么问题" and "核心机制" cells) when the
    note has one, otherwise its first paragraph cut at a sentence end. The home
    page returns nothing and falls back to the tagline."""
    if page.kind == "home":
        return ""
    body = page.body or ""

    def clean(fragment: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip().rstrip("。．.;；")

    zh = page.lang == "zh"
    i = body.find('class="lesson-recipe')
    if i >= 0:
        cells = [clean(c) for c in re.findall(r"<strong>(.*?)</strong>", body[i:i + 3000], re.S)]
        cells = [c for c in cells if c]
        if len(cells) >= 3:
            return f"{cells[0]}；{cells[2]}。" if zh else f"{cells[0]}. {cells[2]}."
        if cells:
            return cells[0] + ("。" if zh else ".")
    j = body.find('class="article-body"')
    rest = re.sub(r"<blockquote.*?</blockquote>", "", body[j:] if j >= 0 else body, flags=re.S)  # skip the notices
    m = re.search(r"<p[^>]*>(.*?)</p>", rest, re.S)
    text = clean(m.group(1)) if m else ""
    if len(text) > 150:
        stops = [k for k, ch in enumerate(text[:150]) if ch in "。！？.!?" and k > 50]
        text = text[:stops[-1] + 1] if stops else text[:148].rstrip() + "…"
    return text


def reading_controls_html(page):
    chinese = page.lang == "zh"
    sibling = getattr(page, "sibling", None)
    label = "EN" if chinese else "中文"
    if not sibling:
        unavailable = "暂无英文版" if chinese else "Chinese version not available"
        return (f'<span class="lang-btn disabled" aria-disabled="true" '
                f'title="{unavailable}">{label}</span>')
    language = "en" if chinese else "zh-Hans"
    description = "切换到英文 / Read in English" if chinese else "切换到中文 / Read in Chinese"
    target = sibling.url
    if sibling.url == "index.html":
        target += "?lang=en"
    href = html.escape(page.rel(target), quote=True)
    return (f'<a class="lang-btn" href="{href}" lang="{language}" hreflang="{language}" data-language-switch="{"en" if chinese else "zh"}" '
            f'aria-label="{description}" title="{description}">{label}</a>')


# Content-Security-Policy as a <meta> tag: GitHub Pages cannot set response headers.
# Only the template's own inline scripts (theme bootstrap, window.SITE) are hashed.
# An inline script that reached a page any other way, e.g. raw HTML in a note, gets
# no hash, so the browser refuses to run it and site/htmlcheck.py fails the build.
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.9/"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.8/"
CSP = {
    "default-src": "'self'",
    "script-src": f"'self' {KATEX} {MERMAID}",
    # Markdown tables, generated figures and Mermaid all set inline styles.
    "style-src": f"'self' 'unsafe-inline' {KATEX}",
    "font-src": f"'self' {KATEX}",
    "img-src": "'self' data: https://avatars.githubusercontent.com https://github.com",
    "connect-src": "'self'",
    "object-src": "'none'",
    "base-uri": "'self'",
    "form-action": "'self'",
    "frame-src": "'none'",
    "worker-src": "'none'",
}
INLINE_SCRIPT = re.compile(r"<script>(.*?)</script>", re.S)


def metrics_origin():
    """The opt-in note-metrics endpoint, when one is configured, needs connect-src."""
    try:
        endpoint = json.loads((ROOT / "site/static/note-metrics-config.json").read_text())["endpoint"]
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    parts = urllib.parse.urlsplit(endpoint or "")
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme == "https" and parts.netloc else ""


def csp_content(hashes):
    policy = dict(CSP)
    policy["script-src"] += "".join(f" 'sha256-{value}'" for value in sorted(hashes))
    if metrics_origin():
        policy["connect-src"] += " " + metrics_origin()
    return "; ".join(f"{key} {value}" for key, value in policy.items())


def script_hash(text):
    return base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii")


def with_csp(output, template, prefix, lang, built):
    trusted = {script.replace("{{prefix}}", prefix).replace("{{lang}}", lang).replace("{{built}}", built)
               for script in INLINE_SCRIPT.findall(template)}
    hashes = {script_hash(script) for script in INLINE_SCRIPT.findall(output) if script in trusted}
    meta = f'<meta http-equiv="Content-Security-Policy" content="{html.escape(csp_content(hashes), quote=True)}">\n'
    return output.replace('<meta charset="utf-8">\n', '<meta charset="utf-8">\n' + meta, 1)


def assemble(page, sections, nav, built, template):
    site = nav["site"]
    if page.section.get("group") == "career":
        template = template.replace('</head>', f'<link rel="stylesheet" href="{page.rel("static/career.css")}?v={built}">\n</head>')
    zh = page.lang == "zh"
    html_lang = "zh-Hans" if zh else "en"
    sib = getattr(page, "sibling", None)
    lang_href = page.rel(sib.url) if sib else "#"
    lang_cls = "" if sib else " disabled"
    prefix = "../" * page.depth
    origin = site.get("origin", "").rstrip("/")
    canonical = f"{origin}/{page.url}" if origin else page.url
    og_type = "website" if page.kind in {"home", "index"} else "article"
    # Every page gets a card image: the note's first picture when it has one, else the site card.
    card = f"{origin}/static/og.png" if origin else f"{prefix}static/og.png"
    image_url, image_dims = card, '<meta property="og:image:width" content="1200">\n<meta property="og:image:height" content="630">\n'
    if page.kind != "home":
        # a paper's figure is shared with its credit on the page, never as a bare card image
        own_images = re.sub(r'<figure class="paper-fig.*?</figure>', "", page.body, flags=re.S)
        image_match = re.search(r'<img\s+[^>]*src="([^"]+\.(?:png|jpe?g|webp))"', own_images, re.I)
        if image_match:
            raw_image = image_match.group(1)
            if re.match(r"https?://", raw_image):
                image_url = raw_image
            else:
                image_path = os.path.normpath(str(page.out_rel.parent / raw_image)).replace(os.sep, "/")
                image_url = f"{origin}/{image_path}" if origin else image_path
            image_dims = ""
    escaped_image = html.escape(image_url, quote=True)
    social_image = (f'<meta property="og:image" content="{escaped_image}">\n{image_dims}'
                    f'<meta property="og:image:alt" content="{html.escape(page.title, quote=True)}">\n'
                    f'<meta name="twitter:image" content="{escaped_image}">')
    twitter_card = "summary_large_image"
    description = share_description(page) or site["tagline_zh" if zh else "tagline_en"]
    content = curriculum.chapter_context(page, BY_SRC) + page.body
    is_prep_hub = 'data-prep-hub' in page.body
    # the reader's own records (saved notes, reading history, positions), the reading drawer and the study buddy
    template = template.replace('</head>', f'<script defer src="{prefix}static/prep-progress-store.js?v={built}"></script>\n<script defer src="{prefix}static/prep-library-state.js?v={built}"></script>\n<script defer src="{prefix}static/prep-progress.js?v={built}"></script>\n<link rel="stylesheet" href="{prefix}static/reader-hub.css?v={built}">\n<script defer src="{prefix}static/reader-hub.js?v={built}"></script>\n<link rel="stylesheet" href="{prefix}static/study-buddy.css?v={built}">\n<script defer src="{prefix}static/study-buddy.js?v={built}"></script>\n</head>')
    if 'class="topic-coverage"' in page.body:
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/topic-coverage.css?v={built}">\n</head>')
    if is_prep_hub:
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/prep-hub.css?v={built}">\n</head>')
    if 'class="tech-timeline"' in page.body:
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/technology-tree.css?v={built}">\n</head>')
    if 'data-technology-tree' in page.body:
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/chronicle.css?v={built}">\n</head>')
        template = template.replace('</head>', f'<script defer src="{prefix}static/chronicle.js?v={built}"></script>\n</head>')
        template = template.replace('<canvas class="sky" aria-hidden="true"></canvas>\n', "")
        template = re.sub(r'<script defer src="\{\{prefix\}\}static/sky\.js[^>]*></script>\n?', "", template)
    if 'class="paper-fig' in page.body:
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/paper-figures.css?v={built}">\n</head>')
    if 'class="method-map"' in page.body:
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/method-map.css?v={built}">\n</head>')
    if 'data-bert-story' in page.body:
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/bert-story.css?v={built}">\n<script defer src="{prefix}static/bert-story.js?v={built}"></script>\n</head>')
    if 'data-encoder-lab' in page.body:
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/encoder-lab.css?v={built}">\n<script defer src="{prefix}static/encoder-lab.js?v={built}"></script>\n</head>')
    if page.section.get("group") == "deep-rl":
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/deep-rl-lab.css?v={built}">\n<script defer src="{prefix}static/deep-rl-lab.js?v={built}"></script>\n</head>')
    if page.section.get("dir") == "07-evaluation/llm-as-a-judge":
        template = template.replace('</head>', f'<link rel="stylesheet" href="{prefix}static/judge-lab.css?v={built}">\n<script defer src="{prefix}static/judge-lab.js?v={built}"></script>\n</head>')
    config = collaboration.load_config()
    content = content.replace('<div data-collaboration-areas></div>',
                              collaboration.areas_html(config, site['repo'], zh))
    content = content.replace('<div data-community-credits></div>',
                              collaboration.acknowledgements_html(config, zh))
    page_class = f"page-{page.kind}"
    if is_prep_hub:
        page_class += " page-prep-hub"
    if 'data-technology-tree' in page.body:
        page_class += " page-technology-tree"
    output = (template
            .replace("{{lang}}", html_lang)
            .replace("{{dir_class}}", "lang-zh" if zh else "lang-en")
            .replace("{{page_class}}", page_class)
            .replace("{{title}}", html.escape(page.title))
            .replace("{{site_title}}", html.escape(site["title_zh" if zh else "title_en"]))
            .replace("{{tagline}}", html.escape(site["tagline_zh" if zh else "tagline_en"]))
            .replace("{{description}}", html.escape(description, quote=True))
            .replace("{{canonical}}", html.escape(canonical, quote=True))
            .replace("{{og_type}}", og_type)
            .replace("{{social_image}}", social_image)
            .replace("{{twitter_card}}", twitter_card)
            .replace("{{og_locale}}", "zh_CN" if zh else "en_US")
            .replace("{{home}}", page.rel("index.zh.html" if zh else "index.html?lang=en"))
            .replace("{{prefix}}", prefix)
            .replace("{{sidebar}}", sidebar_html(page, sections, nav.get("group", [])))
            .replace("{{sidebar_label}}", "笔记导航" if zh else "Note navigation")
            .replace("{{hero}}", hero_html(page))
            .replace("{{tabs}}", tabs_html(page))
            .replace("{{toc}}", toc_html(page))
            .replace("{{mobile_toc}}", mobile_toc_html(page))
            .replace("{{toc_label}}", "本页目录" if zh else "On this page")
            .replace("{{skip_label}}", "跳到正文" if zh else "Skip to content")
            .replace("{{menu_label}}", "打开目录" if zh else "Open navigation")
            .replace("{{theme_label}}", "切换明暗主题" if zh else "Toggle color theme")
            .replace("{{search_ph}}", "搜索笔记" if zh else "Search notes")
            .replace("{{search_close}}", "关闭搜索" if zh else "Close search")
            .replace("{{search_input_ph}}", "想找什么？概念、模型或一道题…" if zh else "Find a concept, model, or exercise…")
            .replace("{{search_current}}", "中文笔记" if zh else "English notes")
            .replace("{{search_all}}", "中英一起搜" if zh else "Both languages")
            .replace("{{search_hint}}", "↑ ↓ 选择 · Enter 打开 · Esc 关闭" if zh else "↑ ↓ navigate · Enter open · Esc close")
            .replace("{{search_privacy}}", "仅在站内搜索" if zh else "Search stays on this site")
            .replace("{{lang_href}}", lang_href)
            .replace("{{lang_cls}}", lang_cls)
            .replace("{{lang_label}}", "EN" if zh else "中文")
            .replace("{{reading_controls}}", reading_controls_html(page))
            .replace("{{repo}}", site["repo"])
            .replace("{{content}}", content)
            .replace("{{page_header}}", page_header_html(page))
            .replace("{{page_nav}}", "" if is_prep_hub else page_nav_html(page))
            .replace("{{note_metrics}}", note_metrics.panel(page, ROOT))
            .replace("{{glossary}}", glossary_html(page))
            .replace("{{footer}}", footer_html(page, site["repo"], built))
            .replace("{{built}}", built))
    return with_csp(output, template, prefix, html_lang, built)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    nav = load_nav()
    collaboration.validate(collaboration.load_config(), nav)
    terms = load_glossary()
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("discovering pages...")
    pages, sections = discover(nav)
    print(f"  {len(pages)} pages in {len(sections)} sections")

    template = (SITE / "template.html").read_text(encoding="utf-8")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    known = {str(p.out_rel).replace(os.sep, "/") for p in pages}

    print("rendering...")
    index = []
    for page in pages:
        build_page(page, terms, nav['site']['repo'], known)
        dest = OUT / page.out_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(assemble(page, sections, nav, built, template),
                        encoding="utf-8")
        index.append({"u": page.url, "t": page.title, "l": page.lang,
                      "s": page.section["zh" if page.lang == "zh" else "en"],
                      "x": page.search_text, "p": page.search_preview})
    print(f"  {len(pages)} pages")

    redirect_count = write_redirects(nav, known)
    if redirect_count:
        print(f"  {redirect_count} redirects")

    # static assets + every section's figures
    shutil.copytree(SITE / "static", OUT / "static")
    (OUT / 'static/prep-progress-catalog.json').write_text(
        json.dumps(prep_progress.catalog(BY_SRC), ensure_ascii=False), encoding='utf-8')
    # the README embeds these from the deployed site, so GitHub shows current numbers
    coverage = coverage_rows()
    for language in ("zh", "en"):
        for palette in topic_coverage.PALETTES:
            (OUT / f"static/topic-coverage-{language}-{palette}.svg").write_text(
                topic_coverage.svg(coverage, language, palette, built[:10]), encoding="utf-8")
    # README cards: what's inside, roadmap, contributors and reviewers (site/readme_cards.py).
    # Contributors are shown on GitHub, not on the site, so they are collected only for the cards.
    people = contributors(nav["site"]["repo"])
    topics = readme_cards.topic_rows(NAV, NAV.get("_sections", []))
    milestones = readme_cards.load_milestones(SITE / "milestones.toml")
    outcomes = json.loads((ROOT / "community/outcomes.json").read_text(encoding="utf-8"))
    team = readme_cards.people_data(people, collaboration.load_config(), nav["site"]["repo"], outcomes)
    signals = {"reviewed": topic_coverage.totals(coverage)[1],  # the same count as the coverage chart below it
               "figures": len(FIGURE_LAB) + len(ALGO_LAB) + len(TX_LAB), "paper": len(PAPER_FIGURES)}
    for language in ("zh", "en"):
        for palette in readme_cards.PALETTES:
            for name, card in (("topics", readme_cards.topics_svg(topics, language, palette, built[:10], signals)),
                               ("roadmap", readme_cards.roadmap_svg(milestones, language, palette, built[:10])),
                               ("people", readme_cards.people_svg(team, language, palette, built[:10]))):
                (OUT / f"static/readme-{name}-{language}-{palette}.svg").write_text(card, encoding="utf-8")
    for d in ROOT.glob("*/assets"):
        shutil.copytree(d, OUT / d.relative_to(ROOT))
        print(f"  assets: {d.relative_to(ROOT)}")

    (OUT / "search-index.json").write_text(json.dumps(index, ensure_ascii=False),
                                           encoding="utf-8")
    api = agent_api.build(OUT)
    print(f"  agent API: {api['topics']} topics, {api['notes']} notes, {api['questions']} questions")
    (OUT / ".nojekyll").write_text("")
    print(f"\nbuilt -> {OUT}")

    if args.serve:
        import http.server
        import socketserver
        os.chdir(OUT)
        with socketserver.TCPServer(("", args.port),
                                    http.server.SimpleHTTPRequestHandler) as httpd:
            print(f"serving http://localhost:{args.port}  (ctrl-c to stop)")
            httpd.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
