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
from datetime import datetime, timezone
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


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_nav():
    with open(SITE / "nav.toml", "rb") as f:
        return tomllib.load(f)


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


def contributors(repo: str):
    """Everyone who has committed, most recent first.

    Tries the GitHub API for avatars (CI has a token and it is the accurate
    source), falls back to `git log` so a local build still works offline.
    """
    recency, counts = {}, {}
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
            "last": when[:10],
            "initial": name[:1].upper(),
        })
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
        self.out = []
        self.used = []

    def handle_starttag(self, tag, attrs):
        if tag in PROTECTED:
            self.depth += 1
        self.out.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.out.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag in PROTECTED:
            self.depth = max(0, self.depth - 1)
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

        if path.endswith(".en.md"):
            path = path[:-6] + ".en.html"
        elif path.endswith(".md"):
            path = path[:-3] + suffix
        elif path.endswith("/") or (path and "." not in Path(path).name):
            path = path.rstrip("/") + "/index" + suffix
        elif "assets/" in path:
            return f'{attr}="{path}{frag}"'          # copied verbatim into _site
        else:
            return f'{attr}="{gh(raw, "blob")}"'     # a real file, no page
        if path.endswith("README" + suffix):
            path = path[: -len("README" + suffix)] + "index" + suffix

        resolved = os.path.normpath(str(here / path)).replace(os.sep, "/")
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


def expand_widgets(md_text: str) -> str:
    return re.sub(r"<!--\s*widget:([a-z0-9_-]+)\s*-->",
                  lambda m: WIDGETS.get(m.group(1), ""), md_text)


# --------------------------------------------------------------------------- #
# page model
# --------------------------------------------------------------------------- #
def read_title(src: Path) -> str:
    """First h1 of a file. Needed for every page before any sidebar is built."""
    m = re.search(r"^#\s+(.+)$", src.read_text(encoding="utf-8"), re.M)
    return m.group(1).strip() if m else src.stem


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

    def rel(self, target: str) -> str:
        return ("../" * self.depth) + target if self.depth else target


def discover(nav):
    pages, sections = [], []
    for sec in nav["section"]:
        d = ROOT if sec["dir"] == "." else ROOT / sec["dir"]
        if not d.exists():
            continue
        order = sec.get("order", [])
        files = [p for p in d.glob("*.md") if not p.name.endswith(".en.md")]
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
            pages.append(zh)
            if en:
                pages.append(en)
        if entry["pages"]:
            sections.append(entry)
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
    raw = expand_widgets(raw)
    raw = protect_mermaid(raw)

    m = re.search(r"^#\s+(.+)$", raw, re.M)
    page.title = m.group(1).strip() if m else page.src.stem
    # the lang switcher line right under the title is redundant on the site
    raw = re.sub(r"^\*\*?(中文|English)\*\*?\s*[·|].*$", "", raw, count=1, flags=re.M)
    raw = re.sub(r"^\[中文\]\([^)]*\)\s*[·|].*$", "", raw, count=1, flags=re.M)

    body, toc = render_markdown(raw)
    used = []
    if page.lang == "zh":
        body, used = annotate(body, terms)
    body = rewrite_links(body, page, repo, known)
    page.body = wrap_tables(body)
    page.toc = [{"id": t["id"], "name": strip_tags(t["name"]), "level": t["level"],
                 "children": [{"id": c["id"], "name": strip_tags(c["name"])}
                              for c in t.get("children", [])]}
                for t in toc]
    page.text = strip_tags(body)[:1500]
    page.glossary = used
    return page


# --------------------------------------------------------------------------- #
# html assembly
# --------------------------------------------------------------------------- #
def section_html(page, sec):
    """One section: its number badge, its label, and its page links."""
    label = html.escape(sec["zh" if page.lang == "zh" else "en"])
    items, has_active = [], False
    for pair in sec["pages"]:
        target = pair[page.lang] or pair["zh"]
        active = target.url == page.url
        has_active = has_active or active
        cls = ' class="active"' if active else ""
        items.append(f'<li><a{cls} href="{page.rel(target.url)}">'
                     f'{html.escape(target.title)}</a></li>')
    # only top-level chapters carry the curriculum number; a subdirectory
    # section like 00-foundations/code would otherwise repeat it
    num = sec["dir"].split("-")[0] if re.match(r"^\d\d-[^/]+$", sec["dir"]) else ""
    badge = f'<span class="sec-num">{num}</span>' if num else ""
    return (f'<li class="sec">{badge}<span class="sec-name">{label}</span>'
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

    out = []
    for g in groups:
        secs = by_group.get(g["id"])
        if not secs:
            continue
        rendered = [section_html(page, s) for s in secs]
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
    return f'<ul class="nav">{"".join(out)}</ul>'


def toc_html(page):
    if not page.toc:
        return ""
    li = []
    for t in page.toc:
        li.append(f'<li><a href="#{t["id"]}">{html.escape(t["name"])}</a></li>')
        for c in t["children"]:
            li.append(f'<li class="l3"><a href="#{c["id"]}">{html.escape(c["name"])}</a></li>')
    return f'<ul>{"".join(li)}</ul>'


def footer_html(page, people, repo, built):
    when = page.updated[:10] if page.updated else built[:10]
    src = f"https://github.com/{repo}/blob/main/{page.src.relative_to(ROOT)}"
    edit = f"https://github.com/{repo}/edit/main/{page.src.relative_to(ROOT)}"
    chips = []
    for p in people:
        av = (f'<img src="{p["avatar"]}" alt="" loading="lazy">' if p["avatar"]
              else f'<span class="ini">{html.escape(p["initial"])}</span>')
        inner = (f'{av}<span class="who">{html.escape(p["name"])}</span>'
                 f'<span class="cnt">{p["commits"]}</span>')
        chips.append(f'<a class="person" href="{p["url"]}" title="{html.escape(p["name"])} · '
                     f'{p["commits"]} commits · last {p["last"]}">{inner}</a>'
                     if p["url"] else f'<span class="person">{inner}</span>')
    zh = page.lang == "zh"
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
  <div class="contrib">
    <h3>{'贡献者' if zh else 'Contributors'}
      <small>{'按最近提交排序，自动生成' if zh else 'most recent first, generated at build time'}</small>
    </h3>
    <div class="people">{"".join(chips)}</div>
    <p class="join">{'欢迎参与：改一个错字、补一段解释、加一篇笔记都算。'
                     if zh else 'Everyone is welcome: a typo fix counts.'}
      <a href="https://github.com/{repo}/blob/main/CONTRIBUTING.md">{'怎样参与' if zh else 'How to contribute'} &#8594;</a>
    </p>
  </div>
</footer>"""


def assemble(page, sections, people, nav, built, template):
    site = nav["site"]
    zh = page.lang == "zh"
    sib = getattr(page, "sibling", None)
    lang_href = page.rel(sib.url) if sib else "#"
    lang_cls = "" if sib else " disabled"
    prefix = "../" * page.depth
    return (template
            .replace("{{lang}}", "zh-Hans" if zh else "en")
            .replace("{{dir_class}}", "lang-zh" if zh else "lang-en")
            .replace("{{title}}", html.escape(page.title))
            .replace("{{site_title}}", html.escape(site["title_zh" if zh else "title_en"]))
            .replace("{{tagline}}", html.escape(site["tagline_zh" if zh else "tagline_en"]))
            .replace("{{home}}", page.rel("index.html" if zh else "index.en.html"))
            .replace("{{prefix}}", prefix)
            .replace("{{sidebar}}", sidebar_html(page, sections, nav.get("group", [])))
            .replace("{{toc}}", toc_html(page))
            .replace("{{toc_label}}", "本页目录" if zh else "On this page")
            .replace("{{search_ph}}", "搜索笔记…" if zh else "Search notes…")
            .replace("{{lang_href}}", lang_href)
            .replace("{{lang_cls}}", lang_cls)
            .replace("{{lang_label}}", "EN" if zh else "中文")
            .replace("{{repo}}", site["repo"])
            .replace("{{content}}", page.body)
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
