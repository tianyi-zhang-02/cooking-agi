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
import functools
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
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import markdown
from markdown.extensions.toc import TocExtension, slugify

ROOT = Path(__file__).resolve().parent.parent
SITE = Path(__file__).resolve().parent
OUT = ROOT / "_site"

# tags whose text must never be touched by the glossary annotator
# headings stay clean: an inline gloss there is noise, and it would
# disagree with the (unannotated) text used in the table of contents
PROTECTED = {"code", "pre", "a", "script", "style", "abbr",
             "h1", "h2", "h3", "h4", "h5", "h6"}

# Structured components whose text is laid out by CSS. An inline gloss chip
# inside one splits a word in half and breaks the grid, so they are skipped
# wholesale -- the annotation belongs in running prose, not in a summary card.
NOGLOSS_CLASSES = {"lesson-recipe", "taste-check", "widget", "mermaid",
                   "home-block"}

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


def load_glossary():
    """中文 -> (English, gloss). Longest terms first so 多头注意力 wins over 注意力."""
    terms = []
    for line in (SITE / "glossary.tsv").read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            terms.append((parts[0].strip(), parts[1].strip(),
                          parts[2].strip() if len(parts) > 2 else ""))
    return sorted(terms, key=lambda t: -len(t[0]))


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


@functools.lru_cache(maxsize=1)
def world_dots():
    """The baked country grid (site/tools/bake_world.py), or {} when it is missing."""
    path = SITE / "static" / "world-dots.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def contributors(repo: str):
    """Everyone who has committed or is credited as a co-author, most recent first.

    Tries the GitHub API for avatars (CI has a token and it is the accurate
    source), falls back to `git log` so a local build still works offline.
    """
    recency, earliest, counts = {}, {}, {}
    log = git("log", "--format=%an\t%ae\t%cI")
    for line in log.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, email, when = parts
        key = name.strip()
        counts[key] = counts.get(key, 0) + 1
        if key not in recency or when > recency[key][0]:
            recency[key] = (when, email.strip())
        if key not in earliest or when < earliest[key]:
            earliest[key] = when

    # AI-assisted and pair-authored commits keep their credit in trailers even
    # when the primary Git author is the repository owner. Aggregate Claude
    # model names into one honest "Claude Code" crew member instead of
    # pretending each model version is a different person.
    ai_models = set()
    coauthor_log = git(
        "log",
        "--format=%cI%x09%(trailers:key=Co-authored-by,valueonly,separator=%x1f)",
    )
    for line in coauthor_log.splitlines():
        if "\t" not in line:
            continue
        when, credits = line.split("\t", 1)
        for credit in credits.split("\x1f"):
            match = re.match(r"\s*(.*?)\s*<([^>]+)>\s*$", credit)
            if not match:
                continue
            name, email = match.groups()
            key = "Claude Code" if "claude" in name.lower() else name.strip()
            if key == "Claude Code":
                ai_models.add(name.strip())
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
            "commits": info.get("commits", counts.get(name, 0)),
            "last": datetime.fromisoformat(when).astimezone(timezone.utc).date().isoformat(),
            "first": datetime.fromisoformat(earliest.get(name, when)).astimezone(
                timezone.utc).date().isoformat(),
            "countries": crew_countries().get((login or name).lower(), ()),
            "initial": name[:1].upper(),
            "kind": "ai" if name == "Claude Code" else "human",
            "models": sorted(ai_models) if name == "Claude Code" else [],
        })
    # a catalogue number, the way a star gets one: by the order it was first seen,
    # so a number stays with the same person as the list grows
    for n, person in enumerate(sorted(out, key=lambda p: (p["first"], p["name"])), start=1):
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
        self.terms = terms
        self.seen = set()
        self.depth = 0
        self.stack = []          # one entry per open element: does it protect?
        self.out = []
        self.used = []

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
        if tag not in VOID:
            hit = self._protects(tag, attrs)
            self.stack.append(hit)
            self.depth += hit
        self.out.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.out.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag not in VOID and self.stack:
            self.depth -= self.stack.pop()
            self.depth = max(0, self.depth)
        self.out.append(f"</{tag}>")

    def handle_comment(self, data):
        self.out.append(f"<!--{data}-->")

    def handle_decl(self, decl):
        self.out.append(f"<!{decl}>")

    def handle_data(self, data):
        text = html.escape(data, quote=False)
        if self.depth == 0 and self.terms:
            # Collect non-overlapping matches against the ORIGINAL text, then
            # splice once. Replacing term by term would let a later term match
            # inside an earlier term's injected data-tip -- a gloss such as
            # "映到同一向量空间靠内积召回" contains 向量, which would then split
            # the attribute open and leak markup into the page.
            hits, taken = [], []
            for zh, en, gloss in self.terms:
                if zh in self.seen:
                    continue
                i = text.find(zh)
                while i != -1 and any(i < e and i + len(zh) > s for s, e in taken):
                    i = text.find(zh, i + 1)
                if i == -1:
                    continue
                taken.append((i, i + len(zh)))
                hits.append((i, zh, en, gloss))

            out, prev = [], 0
            for i, zh, en, gloss in sorted(hits):
                self.seen.add(zh)
                self.used.append({"zh": zh, "en": en, "gloss": gloss})
                tip = html.escape(f"{en}" + (f" — {gloss}" if gloss else ""), quote=True)
                out.append(text[prev:i])
                out.append(f'<span class="term" tabindex="0" data-tip="{tip}">{zh}'
                           f'<span class="term-en">{html.escape(en)}</span></span>')
                prev = i + len(zh)
            out.append(text[prev:])
            text = "".join(out)
        self.out.append(text)

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

        if source_path == "README.md":
            path = os.path.relpath("index.html", str(here)).replace(os.sep, "/")
        elif path.endswith(".en.md"):
            path = path[:-6] + ".en.html"
        elif path.endswith(".md"):
            path = path[:-3] + suffix
        elif path.endswith("/") or (path and "." not in Path(path).name):
            path = path.rstrip("/") + "/index" + suffix
        elif "assets/" in path:
            return f'{attr}="{path}{frag}"'          # copied verbatim into _site
        else:
            return f'{attr}="{gh(raw, "blob")}"'     # a real file, no page
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
            return f'{attr}="{gh(raw, "tree" if raw.endswith("/") else "blob")}"'
        return f'{attr}="{path}{frag}"'

    return re.sub(r'\b(href|src)="([^"]+)"', fix, html_text)


# --------------------------------------------------------------------------- #
# widgets
# --------------------------------------------------------------------------- #
WIDGETS = {
    "xor": """
<figure class="widget" data-widget="xor">
  <figcaption class="widget-head">
    <span class="widget-kicker">live</span>
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
]
for _name, _zh, _en in TX_LAB:
    WIDGETS[_name] = f"""
<figure class="widget tx-lab" id="{_name}" data-widget="{_name}">
  <figcaption class="widget-head">
    <span class="widget-kicker">live</span>
    <span class="widget-title" data-zh="{_zh}" data-en="{_en}"></span>
  </figcaption>
  <div class="widget-body"><p class="tx-fallback" data-zh="这张交互图需要启用 JavaScript。" data-en="This interactive figure needs JavaScript."></p></div>
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
    extend = (f'这里只有我自己在做或打算聊的方向。Data Scientist、Data Engineer、Quant、AI Infra 等路线不是我的 focus——'
              f'欢迎你来补：在 <a href="{source}">site/roadmap.toml</a> 里加一个 <code>[[track]]</code> 就行，文件开头写了怎么加。'
              if zh else
              f'Only directions I work on, or plan to write about, are here. Data Scientist, Data Engineer, Quant, AI Infra and others are not my focus. '
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
    """The three big categories as cards: what is in each, how much, and where to start."""
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
    head = block_head(page, "library", "三个大类" if zh else "Three categories",
                      "不想按路线走？所有笔记都归在这三类里，侧边栏和顶栏也是同一套分类。" if zh else
                      "Rather browse? Every note lives in one of these three categories; the sidebar and the top bar use the same split.")
    return f'<section class="home-block" data-widget="categories">{head}<div class="cat-cards">{"".join(cards)}</div></section>'


def about_head_html(page) -> str:
    zh = page.lang == "zh"
    return ('<section class="home-block about-block">' +
            block_head(page, "about", "关于这份笔记" if zh else "About these notes",
                       "这些笔记为什么写、怎么读，以及哪些内容适合放在这里。" if zh else
                       "What I am trying to understand, how the notes are written, and what is and is not public.") +
            '</section>')


DYNAMIC_WIDGETS = {"question-bank": question_bank_md,
                   "roadmap": roadmap_html, "blocks": blocks_html, "gallery": gallery_html,
                   "threads": threads_html, "categories": categories_html,
                   "about-head": about_head_html}


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


def expand_widgets(md_text: str, page=None) -> str:
    def expand(m):
        name = m.group(1)
        if page is not None and name in DYNAMIC_WIDGETS:
            return DYNAMIC_WIDGETS[name](page)
        shell = WIDGETS.get(name, "")
        return localize_shell(shell, page) if page is not None else shell
    return re.sub(r"<!--\s*widget:([a-z0-9_-]+)\s*-->", expand, md_text)


# --------------------------------------------------------------------------- #
# page model
# --------------------------------------------------------------------------- #
def read_title(src: Path) -> str:
    """First h1 of a file. Needed for every page before any sidebar is built."""
    text = src.read_text(encoding="utf-8")
    heading = re.search(r"^#\s+(.+)$", text, re.M)
    if heading:
        return heading.group(1).strip()
    heading = re.search(r"<h1\b[^>]*>(.*?)</h1>", text, re.S | re.I)
    return html.unescape(strip_tags(heading.group(1))) if heading else src.stem


class Page:
    def __init__(self, src: Path, section, lang):
        self.src = src
        self.section = section
        self.lang = lang
        rel = src.relative_to(ROOT)
        name = rel.name[:-6] if lang == "en" else rel.name[:-3]  # strip .en.md / .md
        stem = "index" if name == "README" else name
        self.out_rel = rel.parent / (stem + (".en.html" if lang == "en" else ".html"))
        self.url = str(self.out_rel).replace(os.sep, "/")
        self.depth = len(self.out_rel.parts) - 1
        self.title = read_title(src)
        self.toc = []
        self.body = ""
        self.text = ""
        self.updated = last_updated(str(rel))
        if section["dir"] == "." and rel.name in {"README.md", "README.en.md"}:
            self.kind = "home"
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
                 "group": sec.get("group", "reference"), "pages": []}
        for f in files:
            zh = Page(f, entry, "zh")
            en_src = f.with_name(f.stem + ".en.md")
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
    NAV.update(nav)
    NAV["_sections"] = sections
    return pages, sections


def write_redirects(nav, known):
    """Generate small redirect pages for published URLs moved into topic folders."""
    redirects = nav.get("redirects", {})
    for old_url, target_url in redirects.items():
        if target_url not in known:
            raise ValueError(f"redirect target is not a generated page: {target_url}")
        old_path = Path(old_url)
        relative_target = os.path.relpath(target_url, old_path.parent).replace(os.sep, "/")
        escaped_target = html.escape(relative_target, quote=True)
        script_target = json.dumps(relative_target)
        page = f"""<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={escaped_target}">
<link rel="canonical" href="{escaped_target}">
<title>Moved</title>
<script>location.replace({script_target} + location.hash)</script>
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


def render_markdown(text):
    text, math_stash = protect_math(text)
    md = markdown.Markdown(extensions=[
        "fenced_code", "tables", "footnotes", "attr_list", "sane_lists", "md_in_html",
        TocExtension(anchorlink=False, permalink=False, toc_depth="2-3"),
        "codehilite",
    ], extension_configs={"codehilite": {"guess_lang": False, "css_class": "hl"}})
    body = md.convert(text)
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
    page.title = m.group(1).strip() if m else read_title(page.src)
    raw = re.sub(r"^#\s+.+$", "", raw, count=1, flags=re.M)
    if page.kind == "home":
        # Useful on GitHub, but redundant (and oddly circular) on the site itself.
        raw = re.sub(
            r"^###\s+[^\n]*(?:阅读请到|Read online|Read it at)[^\n]*\n(?:\n)?",
            "",
            raw,
            count=1,
            flags=re.M,
        )
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
    used = []
    if page.lang == "zh":
        body, used = annotate(body, terms)
    body = rewrite_links(body, page, repo, known)
    # stable targets for the question bank, whatever the heading slug turned out to be
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
    pair = NAV.get("label", {}).get(key)
    if pair:
        return pair[0] if page.lang == "zh" else pair[1]
    head = re.split(r"[：:]", page.title, maxsplit=1)[0].strip()
    return head if 1 < len(head) < len(page.title) else page.title


def section_html(page, sec, show_head=True):
    """One section: an optional small heading and its page links, each under its short label
    with the full title on hover. The heading is left out where the block name already says it."""
    label = html.escape(sec["zh" if page.lang == "zh" else "en"])
    items, has_active = [], False
    for pair in sec["pages"]:
        target = pair[page.lang] or pair["zh"]
        active = target.url == page.url
        has_active = has_active or active
        cls = ' class="active" aria-current="page"' if active else ""
        items.append(f'<li><a{cls} href="{page.rel(target.url)}" title="{html.escape(target.title, quote=True)}">'
                     f'{html.escape(nav_label(target))}</a></li>')
    head = f'<span class="sec-name">{label}</span>' if show_head else ""
    return (f'<li class="sec{"" if show_head else " sec-flat"}">{head}'
            f'<ul>{"".join(items)}</ul></li>'), has_active


def sidebar_html(page, sections, groups):
    """Sections bucketed into collapsible topic groups.

    Built on <details>, so collapsing still works with JavaScript disabled. The
    group holding the current page ships open; the rest ship closed and their
    state is remembered client-side.
    """
    by_group = {}
    for sec in sections:
        by_group.setdefault(sec.get("group", "reference"), []).append(sec)

    out, seen_cat = [], None
    cats = {c["id"]: c for c in NAV.get("category", [])}
    scope = page_category(page)
    if scope in cats:
        groups = [g for g in groups if g.get("category") == scope]
    ordered = sorted(groups, key=lambda g: list(cats).index(g["category"]) if g.get("category") in cats else -1)
    for g in ordered:
        if not g.get("category"):          # "start" is the home page itself; the top bar links it
            continue
        secs = [s for s in by_group.get(g["id"], []) if s["pages"]]
        if not secs:
            continue
        cat = cats.get(g.get("category"))
        if cat and cat["id"] != seen_cat:
            seen_cat = cat["id"]
            home = category_home(cat)
            target = home and (home.get(page.lang) or home["zh"])
            label_cat = html.escape(cat["zh" if page.lang == "zh" else "en"])
            out.append(f'<li class="cat"><a href="{page.rel(target.url)}">{label_cat}</a></li>' if target
                       else f'<li class="cat"><span>{label_cat}</span></li>')
        # a heading only where it adds something: several sections in the block, and
        # more than one note in this one (a lone note's label already names it)
        rendered = [section_html(page, s, len(secs) > 1 and len(s["pages"]) > 1) for s in secs]
        body = "".join(h for h, _ in rendered)
        is_active = any(a for _, a in rendered)
        n_pages = sum(len(s["pages"]) for s in secs)
        label = html.escape(g["zh" if page.lang == "zh" else "en"])
        out.append(
            f'<li class="grp"><details data-grp="{g["id"]}"'
            f'{" open" if is_active else ""}>'
            f'<summary><svg class="chev" viewBox="0 0 12 12" width="11" height="11" '
            f'aria-hidden="true"><path d="M4 2.5 L7.5 6 L4 9.5" fill="none" '
            f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
            f'stroke-linejoin="round"/></svg>'
            f'<span class="grp-name">{label}</span>'
            f'<span class="grp-count">{n_pages}</span></summary>'
            f'<ul class="grp-body">{body}</ul></details></li>')
    # the way into the contributors page from any note, not just the page foot
    crew = page.rel("contributors.html" if page.lang == "zh" else "contributors.en.html")
    out.append(f'<li class="nav-crew"><a href="{crew}">'
               f'<span aria-hidden="true">✧</span>'
               f'{"幕后" if page.lang == "zh" else "Behind the notes"}</a></li>')
    return f'<ul class="nav">{"".join(out)}</ul>'


def page_category(page):
    """Which top-level direction a page belongs to ("home" for the roadmap page)."""
    if page.url in {"index.html", "index.en.html"}:
        return "home"
    groups = {g["id"]: g for g in NAV.get("group", [])}
    return groups.get(page.section.get("group"), {}).get("category")


def group_target(page, group):
    """The page a block tab lands on: the group's `home`, else the first page of its first section."""
    pair = BY_SRC.get(group.get("home", ""))
    if not pair:
        secs = [s for s in NAV.get("_sections", []) if s["group"] == group["id"] and s["pages"]]
        pair = secs[0]["pages"][0] if secs else None
    return pair and (pair.get(page.lang) or pair["zh"])


def tabs_html(page):
    """Level 1, in the top bar on every page: the roadmap plus the big directions."""
    zh = page.lang == "zh"
    current = page_category(page)
    home = {"id": "home", "zh": "首页", "en": "Home"}
    items = [(home, page.rel("index.html" if zh else "index.en.html"))]
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


def subtabs_html(page):
    """Level 2, at the top of the content column: the blocks inside the chosen direction."""
    zh = page.lang == "zh"
    current = page_category(page)
    if current == "home":
        return ""                         # the home page is one short page; the top bar is enough
    cat = next((c for c in NAV.get("category", []) if c["id"] == current), None)
    if not cat:
        return ""
    here = page.section.get("group")
    links = ""
    for group in (g for g in NAV.get("group", []) if g.get("category") == current):
        target = group_target(page, group)
        if not target:
            continue
        cls = ' class="active" aria-current="true"' if group["id"] == here else ""
        links += f'<a href="{page.rel(target.url)}"{cls}><span>{both(group, zh)[0]}</span></a>'
    label = both(cat, zh)[0]
    if not links:
        return ""
    return (f'<nav class="subtabs" aria-label="{"板块" if zh else "Blocks"}">'
            f'<span class="subtabs-label">{label}</span>{links}</nav>')


def hero_html(page) -> str:
    """The home page opens on the night sky with one line; other pages have no hero."""
    if page.url not in {"index.html", "index.en.html"}:
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


def page_header_html(page):
    zh = page.lang == "zh"
    if page.url in {"contributors.html", "contributors.en.html"}:
        return (f'<header class="crew-heading"><p>BEHIND THE NOTES</p>'
                f'<h1>{html.escape(page.title)}</h1></header>')
    section = page.section["zh" if zh else "en"]
    labels = {
        "home": "学习笔记" if zh else "Study notes",
        "index": "章节概览" if zh else "Chapter overview",
        "workshop": "实践" if zh else "Practice",
        "guide": "参考资料" if zh else "Reference",
        "article": "概念笔记" if zh else "Concept note",
    }
    read = f"约 {page.read_minutes} 分钟" if zh else f"{page.read_minutes} min read"
    position = (f"{page.position:02d} / {page.section_count:02d}"
                if page.kind != "home" and page.section_count > 1 else "")
    reviewed = (("审阅于 " if zh else "Reviewed ") + page.reviewed
                if page.reviewed else "")
    number_match = re.match(r"^(\d\d)-", page.section["dir"])
    chapter_mark = (f'<em class="chapter-mark" aria-hidden="true">'
                    f'{number_match.group(1)}</em>' if number_match else "")
    bits = [f'<span>{html.escape(read)}</span>']
    if reviewed:
        bits.append(f'<span>{html.escape(reviewed)}</span>')
    if position:
        bits.append(f'<span class="page-position">{position}</span>')
    return f"""
<header class="article-head">
  {chapter_mark}
  <div class="article-kicker"><span>{html.escape(labels[page.kind])}</span>
    <i>{html.escape(section)}</i></div>
  <h1>{html.escape(page.title)}</h1>
  <div class="article-meta">{'<b aria-hidden="true"></b>'.join(bits)}</div>
</header>"""


def mobile_toc_html(page):
    toc = toc_html(page)
    if not toc:
        return ""
    label = "展开本页路线" if page.lang == "zh" else "Open this page's route"
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
    return f'<nav class="page-turn" aria-label="{"继续阅读" if zh else "Continue reading"}">' + "".join(items) + "</nav>"


def footer_html(page, people, repo, built):
    when = page.updated[:10] if page.updated else built[:10]
    src = f"https://github.com/{repo}/blob/main/{page.src.relative_to(ROOT)}"
    edit = f"https://github.com/{repo}/edit/main/{page.src.relative_to(ROOT)}"
    zh = page.lang == "zh"
    portal = page.rel("contributors.html" if zh else "contributors.en.html")
    contributor_portal = "" if page.url in {"contributors.html", "contributors.en.html"} else f"""
  <div class="crew-footer">
    <a class="contributor-portal" href="{portal}">
      <span aria-hidden="true">✧</span>
      <span>{'幕后' if zh else 'Behind the notes'}</span>
      <span aria-hidden="true">↗</span>
    </a>
  </div>"""
    return f"""
<footer class="page-foot">
  <div class="foot-meta">
    <span class="updated">{'最后更新' if zh else 'Last updated'}
      <time datetime="{page.updated or built}">{when}</time></span>
    <span class="sep">·</span>
    <a href="{src}">{'查看源文件' if zh else 'View source'}</a>
    <span class="sep">·</span>
    <a href="{edit}">{'提交修改' if zh else 'Suggest an edit'}</a>
  </div>
  {contributor_portal}
</footer>"""


def contributor_universe_html(people, page, site):
    """The contributors page: the 1-bit sky, then the credits, the board and the map.

    Avatars are dithered to two colours in the browser (static/app.js), so the page keeps
    the same black-and-white language as the sky it floats in. Everything below the first
    screen is plain HTML: it reads fine with no JavaScript at all."""
    zh = page.lang == "zh"
    today = datetime.now(timezone.utc).date()
    week_start = today - timedelta(days=today.weekday())
    countries = world_dots().get("countries", {})

    crew, board, credits, tally, weight = [], [], {}, {}, {}
    ranked = sorted(people, key=lambda p: (-p.get("commits", 0), p["name"]))
    for index, person in enumerate(people):
        login = person.get("login")
        is_owner = bool(login and login.lower() == site.get("owner_login", "").lower())
        url = site.get("owner_url") if is_owner else person.get("url")
        handle = f"@{login}" if login else person["name"]
        role = (("AI 协作者" if zh else "AI collaborator") if person.get("kind") == "ai"
                else ("笔记与代码" if zh else "Notes & code"))
        title = html.escape(person["name"])
        avatar = person.get("avatar") or ""
        seed = html.escape(person["name"][:2].upper(), quote=True)
        face = (f'<img class="crew-src" src="{html.escape(avatar, quote=True)}" alt="" '
                f'crossorigin="anonymous" loading="lazy" decoding="async">' if avatar else "")
        link = (f'<a href="{html.escape(url, quote=True)}">{title} <span aria-hidden="true">↗</span></a>'
                if url else f'<strong>{title}</strong>')
        active = week_start.isoformat() <= person["last"] <= today.isoformat()
        crew.append(
            f'<div class="crew-drifter{" is-recent" if active else ""}" '
            f'data-crew-id="{html.escape(login or person["name"], quote=True)}" '
            f'data-last="{person["last"]}" style="--start-x:{(17 + index * 37) % 74 + 13}%;'
            f'--start-y:{(23 + index * 29) % 56 + 22}%">'
            f'<button class="crew-pilot" type="button" aria-label="{html.escape(handle, quote=True)}" '
            f'aria-expanded="false" aria-controls="crew-label-{index}">'
            f'<span class="crew-face" data-initials="{seed}">{face}'
            f'<canvas class="crew-bits" width="72" height="72" aria-hidden="true"></canvas></span></button>'
            f'<div class="crew-label" id="crew-label-{index}">'
            f'<span class="crew-handle">{person["crew_id"]} · {html.escape(handle)}</span>'
            f'{link}<small>{role}</small></div></div>')
        # the credits roll, grouped by what someone did, in the order the roles are listed
        credits.setdefault(role, []).append(
            f'<li><span class="credit-name">{link}</span>'
            f'<span class="credit-note">{person["crew_id"]} · {html.escape(handle)} · '
            f'{person["first"][:7]} {"起" if zh else "onwards"}</span></li>')
        mine = [c for c in person.get("countries", ()) if c in countries]
        for code in mine:                      # two places: half the weight to each
            tally.setdefault(code, []).append(person["name"])
            weight[code] = weight.get(code, 0.0) + max(person.get("commits", 0), 1) / len(mine)

    top = max([p.get("commits", 0) for p in people] + [1])
    for person in ranked:
        login = person.get("login")
        is_owner = bool(login and login.lower() == site.get("owner_login", "").lower())
        url = site.get("owner_url") if is_owner else person.get("url")
        handle = f"@{login}" if login else person["name"]
        title = html.escape(person["name"])
        link = (f'<a href="{html.escape(url, quote=True)}">{title}</a>' if url
                else f'<strong>{title}</strong>')
        role = (("AI 协作者" if zh else "AI collaborator") if person.get("kind") == "ai"
                else ("笔记与代码" if zh else "Notes & code"))
        active = week_start.isoformat() <= person["last"] <= today.isoformat()
        commits = person.get("commits", 0)
        week = (f'<span class="crew-weekly">{"本周" if zh else "This week"}</span>'
                if active else "")
        board.append(
            f'<tr class="{"is-recent" if active else ""}">'
            f'<td class="crew-id">{person["crew_id"]}</td>'
            f'<th scope="row">{link}<small>{html.escape(handle)}</small></th>'
            f'<td class="role">{role}</td>'
            f'<td class="count"><span class="bar" style="--fill:{commits / top:.3f}" '
            f'aria-hidden="true"></span><b>{commits}</b></td>'
            f'<td><time datetime="{person["first"]}">{person["first"]}</time></td>'
            f'<td><time datetime="{person["last"]}">{person["last"]}</time>{week}</td></tr>')

    # five steps of brightness: a single commit already lights a country, more burns
    # brighter up to a ceiling, and the first few stay clearly ahead of the rest
    order = sorted(weight, key=lambda c: (-weight[c], countries[c]["en"]))
    peak = weight[order[0]] if order else 1.0

    def step(rank, w):
        if rank == 0:
            return 5
        if rank <= 2:
            return 4
        share = w / peak
        return 3 if share >= 0.5 else (2 if share >= 0.2 else 1)

    lit = [f"{c}:{step(i, weight[c])}" for i, c in enumerate(order)]
    map_list = "".join(
        f'<li><b>{html.escape(countries[c]["zh" if zh else "en"])}</b>'
        f'<span>{len(tally[c])} {"位" if zh else ("person" if len(tally[c]) == 1 else "people")}</span></li>'
        for c in order)
    crew_issue = (f'https://github.com/{html.escape(site["repo"], quote=True)}'
                  f'/issues/new?template=add-me-to-the-crew.yml')
    home = page.rel("index.html" if zh else "index.en.html")
    language = page.rel("contributors.en.html" if zh else "contributors.html")
    prefix = "../" * page.depth
    roles = [r for r in (("笔记与代码" if zh else "Notes & code"),
                         ("AI 协作者" if zh else "AI collaborator")) if r in credits]
    roll = "".join(f'<div class="credit-group"><p class="credit-role">{role}</p>'
                   f'<ul>{"".join(credits[role])}</ul></div>' for role in roles)
    return f"""
<section class="contributor-universe" aria-label="{'贡献者' if zh else 'Contributors'}"
  style="background-image:url('{prefix}static/crew-sky-1bit.png')">
  <nav class="orbit-nav" aria-label="{'页面导航' if zh else 'Page navigation'}">
    <a href="{home}">← {'回到笔记' if zh else 'Back to notes'}</a>
    <div><a href="#crew-board">{'榜单' if zh else 'Board'}</a>
    <button class="crew-motion" type="button" aria-pressed="false" hidden
      data-pause="{'暂停' if zh else 'Pause'}" data-play="{'继续' if zh else 'Resume'}">{'暂停' if zh else 'Pause'}</button>
    <a href="{language}">{'EN' if zh else '中文'}</a></div>
  </nav>
  <header class="orbit-title">
    <p>THE PEOPLE BEHIND THE NOTES</p>
    <h1>{'这一小片宇宙，谢谢你来过。' if zh else 'A little universe, made together.'}</h1>
  </header>
  <div class="crew-field">{"".join(crew)}</div>
  <footer class="orbit-footer">
    <p class="orbit-status" role="status" aria-live="polite"></p>
    <p><span class="orbit-desktop">{'鼠标靠近，打个招呼' if zh else 'Hover over someone to say hello'}</span>
    <span class="orbit-touch">{'轻点头像，打个招呼' if zh else 'Tap someone to say hello'}</span></p>
    <a class="orbit-more" href="#crew-credits">{'往下看 ↓' if zh else 'Scroll down ↓'}</a>
  </footer>
</section>
<section class="crew-credits" id="crew-credits" aria-label="{'演职员表' if zh else 'Credits'}">
  <p class="credits-kicker">{'演职员表' if zh else 'Credits'}</p>
  <div class="credits-roll">{roll}</div>
  <p class="credits-end">{'谢谢每一个来过、又留下过点什么的人。' if zh else 'Thank you to everyone who came through and left something behind.'}</p>
</section>
<section class="crew-board" id="crew-board" aria-labelledby="board-title">
  <div class="board-head">
    <h2 id="board-title">{'贡献榜单' if zh else 'Contribution board'}</h2>
    <p>{'按提交数排，数字来自 main 分支的提交和 Co-authored-by 署名。' if zh else 'By commits, from main-branch commits and Co-authored-by credits.'}</p>
  </div>
  <div class="board-scroll"><table class="board-table">
    <thead><tr><th class="crew-id">{'编号' if zh else 'No.'}</th><th>{'贡献者' if zh else 'Contributor'}</th>
      <th>{'参与方式' if zh else 'Role'}</th><th class="count">{'提交' if zh else 'Commits'}</th>
      <th>{'第一次' if zh else 'First'}</th><th>{'最近' if zh else 'Latest'}</th></tr></thead>
    <tbody>{"".join(board)}</tbody>
  </table></div>
  <p class="board-note">{'更新于' if zh else 'Updated'} {today.isoformat()}</p>
</section>
<section class="crew-map" id="crew-map" aria-labelledby="map-title"
  data-world="{prefix}static/world-dots.json" data-lit="{",".join(lit)}">
  <div class="board-head">
    <h2 id="map-title">{'大家在哪儿' if zh else 'Where the crew is'}</h2>
  </div>
  <canvas class="world-dots" width="800" height="400" role="img"
    aria-label="{'点亮了 ' + str(len(order)) + ' 个国家或地区的世界地图' if zh else f'A world map with {len(order)} countries or regions lit up'}"></canvas>
  <ul class="map-list">{map_list or f'<li class="map-empty">{"还没人填。" if zh else "Nobody yet."}</li>'}</ul>
  <p class="map-how"><a href="{crew_issue}">{'开个 issue 告诉我' if zh else 'Open an issue'}</a>{'，我给你点上。' if zh else " and I'll light one up."}</p>
</section>
<footer class="crew-end">
  <a href="https://github.com/{html.escape(site["repo"], quote=True)}/blob/main/CONTRIBUTING.md">{'下一个位置，也许是你 ↗' if zh else 'Room for one more ↗'}</a>
  <a href="{home}">← {'回到笔记' if zh else 'Back to notes'}</a>
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


def assemble(page, sections, people, nav, built, template):
    site = nav["site"]
    zh = page.lang == "zh"
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
        image_match = re.search(r'<img\s+[^>]*src="([^"]+\.(?:png|jpe?g|webp))"', page.body, re.I)
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
    content = page.body
    if 'data-contributors-universe' in content:
        content = contributor_universe_html(people, page, site)
        template = re.sub(r'<header class="topbar">.*?</header>', "", template, count=1, flags=re.S)
        # this page has its own still sky, so the animated one is never loaded here
        template = template.replace('<canvas class="sky" aria-hidden="true"></canvas>\n', "")
        template = re.sub(r'<script defer src="\{\{prefix\}\}static/sky\.js[^>]*></script>\n?', "", template)
        template = re.sub(
            r'<div class="shell">.*?(?=<script>window.SITE)',
            '<main id="content" class="crew-space">{{content}}</main>\n',
            template, count=1, flags=re.S,
        )
    return (template
            .replace("{{lang}}", "zh-Hans" if zh else "en")
            .replace("{{dir_class}}", "lang-zh" if zh else "lang-en")
            .replace("{{page_class}}", "page-crew" if 'data-contributors-universe' in page.body else f"page-{page.kind}")
            .replace("{{title}}", html.escape(page.title))
            .replace("{{site_title}}", html.escape(site["title_zh" if zh else "title_en"]))
            .replace("{{tagline}}", html.escape(site["tagline_zh" if zh else "tagline_en"]))
            .replace("{{description}}", html.escape(description, quote=True))
            .replace("{{canonical}}", html.escape(canonical, quote=True))
            .replace("{{og_type}}", og_type)
            .replace("{{social_image}}", social_image)
            .replace("{{twitter_card}}", twitter_card)
            .replace("{{og_locale}}", "zh_CN" if zh else "en_US")
            .replace("{{home}}", page.rel("index.html" if zh else "index.en.html"))
            .replace("{{prefix}}", prefix)
            .replace("{{sidebar}}", sidebar_html(page, sections, nav.get("group", [])))
            .replace("{{hero}}", hero_html(page))
            .replace("{{tabs}}", tabs_html(page))
            .replace("{{subtabs}}", subtabs_html(page))
            .replace("{{toc}}", toc_html(page))
            .replace("{{mobile_toc}}", mobile_toc_html(page))
            .replace("{{toc_label}}", "本页目录" if zh else "On this page")
            .replace("{{search_ph}}", "搜索笔记…" if zh else "Search notes…")
            .replace("{{lang_href}}", lang_href)
            .replace("{{lang_cls}}", lang_cls)
            .replace("{{lang_label}}", "EN" if zh else "中文")
            .replace("{{crew_href}}", page.rel("contributors.html" if zh else "contributors.en.html"))
            .replace("{{crew_label}}", "幕后" if zh else "Behind the notes")
            .replace("{{crew_aria}}", "幕后" if zh else "Behind the notes")
            .replace("{{repo}}", site["repo"])
            .replace("{{content}}", content)
            .replace("{{page_header}}", page_header_html(page))
            .replace("{{page_nav}}", "" if 'data-contributors-universe' in page.body else page_nav_html(page))
            .replace("{{glossary}}", glossary_html(page))
            .replace("{{footer}}", footer_html(page, people, site["repo"], built))
            .replace("{{built}}", built))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    nav = load_nav()
    terms = load_glossary()
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("discovering pages...")
    pages, sections = discover(nav)
    print(f"  {len(pages)} pages in {len(sections)} sections")

    print("collecting contributors...")
    people = contributors(nav["site"]["repo"])
    print(f"  {len(people)}: " + ", ".join(p["name"] for p in people[:6]))

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
        dest.write_text(assemble(page, sections, people, nav, built, template),
                        encoding="utf-8")
        index.append({"u": page.url, "t": page.title, "l": page.lang,
                      "s": page.section["zh" if page.lang == "zh" else "en"],
                      "x": page.text})
    print(f"  {len(pages)} pages")

    redirect_count = write_redirects(nav, known)
    if redirect_count:
        print(f"  {redirect_count} redirects")

    # static assets + every section's figures
    shutil.copytree(SITE / "static", OUT / "static")
    for d in ROOT.glob("*/assets"):
        shutil.copytree(d, OUT / d.relative_to(ROOT))
        print(f"  assets: {d.relative_to(ROOT)}")

    (OUT / "search-index.json").write_text(json.dumps(index, ensure_ascii=False),
                                           encoding="utf-8")
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
