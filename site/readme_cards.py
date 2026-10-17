"""README cards: what's inside, the roadmap, and the people behind the notes.

GitHub shows images, not scripts or site CSS, so the build writes standalone SVGs (light and dark,
both languages) into the deployed site and the README embeds them from there, like the topic
coverage chart. The site rebuilds on every push to main and once a day, so the cards follow the
notes, site/milestones.toml and GitHub without anyone editing the README.

People data comes from the GitHub API when it is reachable (CI has a token); avatars are embedded
as data URIs because an SVG shown as an image cannot load anything else. Offline, initials stand in.
"""
import base64
import html
import itertools
import json
import os
import re
import tomllib
import urllib.error
import urllib.parse
import urllib.request


FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', 'PingFang SC', "
        "'Hiragino Sans GB', 'Microsoft YaHei', sans-serif")
WIDTH, PAD, INSET = 640, 2, 22
DRAFT_MARK = "草稿，待审核"
OTHER_CATEGORIES = (("interview", "面试准备", "Interview prep"),
                    ("practice", "工程实践", "Engineering practice"),
                    ("history", "技术编年史", "Tech history"))
# GitHub's own page colours underneath, the site's amber on top
PALETTES = {
    "light": {"ink": "#1f2328", "muted": "#59636e", "accent": "#98512b", "rule": "#d1d9e0",
              "top": "#ffffff", "bottom": "#f6f8fa", "tile": "#f6f8fa", "tile_rule": "#e1e6eb",
              "glow": "#e8a672", "glow_opacity": ".16", "soft": "#98512b"},
    "dark": {"ink": "#e6edf3", "muted": "#9198a1", "accent": "#e8a672", "rule": "#30363d",
             "top": "#161b22", "bottom": "#0d1117", "tile": "#1c2128", "tile_rule": "#2d333b",
             "glow": "#e8a672", "glow_opacity": ".13", "soft": "#e8a672"},
}
esc = lambda value: html.escape(str(value), quote=True)


def _pick(zh, chinese, english):
    return chinese if zh else english


def _width(text, size):
    """Rough rendered width, generous for Latin text because Windows and Linux fonts run wider
    than the Mac's; CJK glyphs are square."""
    return sum(size if ord(char) > 0x2E80 else size * 0.6 for char in text)


TOKEN = r"[A-Za-z0-9&().,:;/'’+\-·µ%]+|\s+|."


def wrap(text, max_width, size):
    """Greedy line wrap using the same rough glyph widths as the coverage chart."""
    tokens = re.findall(TOKEN, text)
    lines, line = [], ""
    for token in tokens:
        candidate = line + token
        if line and _width(candidate.rstrip(), size) > max_width and token not in CLOSING:
            lines.append(line.rstrip())
            line = token.lstrip()
        else:
            line = candidate
    if line.strip():
        lines.append(line.rstrip())
    return lines


CLOSING = set("，。、；：！？）》」,.;:!?)")
BREAK_AFTER = CLOSING | set("，、；：,;:")


def wrap_even(text, max_width, size):
    """As many lines as wrap() needs, but evenly long and broken after punctuation where possible
    (for short labels in narrow columns, where "Deep / RL" would read badly)."""
    tokens = re.findall(TOKEN, text)
    n, best = len(wrap(text, max_width, size)), None
    for cuts in itertools.combinations(range(1, len(tokens)), n - 1):
        bounds = (0, *cuts, len(tokens))
        lines = ["".join(tokens[a:b]).strip() for a, b in zip(bounds, bounds[1:])]
        widest = max(_width(line, size) for line in lines)
        if not all(lines) or widest > max_width or any(tokens[c] in CLOSING for c in cuts):
            continue
        score = widest + sum(0 if line[-1] in BREAK_AFTER else 4 * size for line in lines[:-1])
        if best is None or score < best[0]:
            best = (score, lines)
    return best[1] if best else wrap(text, max_width, size)


def _text(out, x, y, value, fill, size=13, weight=400, anchor="start", spacing=None, opacity=None):
    extra = (f' letter-spacing="{spacing}"' if spacing else "") + (f' opacity="{opacity}"' if opacity else "")
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
               f'text-anchor="{anchor}"{extra}>{esc(value)}</text>')


def _panel(width, height, colors, title, desc, body, glow=(0.12, 0.0)):
    """The shared frame: a rounded panel with a soft amber glow in one corner."""
    gx, gy = glow
    defs = (f'<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{colors["top"]}"/>'
            f'<stop offset="1" stop-color="{colors["bottom"]}"/></linearGradient>'
            f'<radialGradient id="glow" cx="{gx}" cy="{gy}" r="0.75"><stop offset="0" stop-color="{colors["glow"]}" '
            f'stop-opacity="{colors["glow_opacity"]}"/><stop offset="1" stop-color="{colors["glow"]}" stop-opacity="0"/></radialGradient></defs>')
    frame = (f'<rect x="{PAD}" y="{PAD}" width="{width - 2 * PAD}" height="{height - 2 * PAD:.0f}" rx="18" fill="url(#bg)"/>'
             f'<rect x="{PAD}" y="{PAD}" width="{width - 2 * PAD}" height="{height - 2 * PAD:.0f}" rx="18" fill="url(#glow)"/>'
             f'<rect x="{PAD + .5}" y="{PAD + .5}" width="{width - 2 * PAD - 1}" height="{height - 2 * PAD - 1:.0f}" rx="17.5" '
             f'fill="none" stroke="{colors["rule"]}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" width="{width}" '
            f'height="{height:.0f}" role="img" aria-labelledby="t d" font-family="{esc(FONT)}">'
            f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>{defs}{frame}' + "".join(body) + "</svg>\n")


def _kicker(out, x, y, label, colors):
    _text(out, x, y, label.upper(), colors["accent"], 11.5, 700, spacing=".12em")


def _stats(out, x, y, stats, colors, width):
    """Big numbers with small labels, in equal columns; labels wrap to two lines."""
    col = width / len(stats)
    for i, (value, label) in enumerate(stats):
        cx = x + i * col
        _text(out, cx, y, value, colors["ink"], 26, 700, spacing="-.02em")
        for j, line in enumerate(wrap(label, col - 14, 12.5)[:2]):
            _text(out, cx, y + 20 + j * 15, line, colors["muted"], 12.5)
    return y + 20 + 15 * 2


def _tile(out, x, y, w, h, colors, accent=True, dashed=False):
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="11" fill="{colors["tile"]}" '
               f'stroke="{colors["tile_rule"]}"/>')
    if accent:
        dash = ' stroke-dasharray="3 3"' if dashed else ""
        out.append(f'<line x1="{x + 1.5:.1f}" y1="{y + 11:.1f}" x2="{x + 1.5:.1f}" y2="{y + h - 11:.1f}" '
                   f'stroke="{colors["accent"]}" stroke-width="3" stroke-linecap="round"{dash}/>')


# ---------------------------------------------------------------- what's inside

def _plain(zh, en):
    """Labels without the sidebar's draft suffix: the card marks drafts itself."""
    return zh.replace("（待审核）", "").replace("，待审核）", "）"), en.replace(" (draft)", "")


def topic_rows(nav, sections):
    """{"zones": [(zh, en, [(zh, en, notes, drafts)])], "others": [(zh, en, notes)]}.
    A zone made of a single group (the frontier) lists that group's series instead."""
    counts, by_section = {}, []
    for section in sections:
        tally = [0, 0]
        for pair in section["pages"]:
            page = pair.get("zh") or pair.get("en")
            if page.kind in {"index", "home"}:
                continue
            tally[0] += 1
            tally[1] += DRAFT_MARK in page.src.read_text(encoding="utf-8")[:600]
        notes, drafts = counts.get(section["group"], (0, 0))
        counts[section["group"]] = (notes + tally[0], drafts + tally[1])
        if tally[0]:
            by_section.append((section, *tally))
    zones = []
    for zone in nav.get("zone", []):
        ids = [g["id"] for g in nav["group"] if g.get("zone") == zone["id"] and counts.get(g["id"], (0, 0))[0]]
        if len(ids) == 1:
            rows = [(*_plain(s["zh"], s["en"]), n, d) for s, n, d in by_section if s["group"] == ids[0]]
        else:
            rows = [(*_plain(g["zh"], g["en"]), *counts[g["id"]]) for g in nav["group"] if g["id"] in ids]
        if rows:
            zones.append((zone["zh"], zone["en"], rows))
    others = []
    for category, zh, en in OTHER_CATEGORIES:
        ids = {g["id"] for g in nav["group"] if g.get("category") == category}
        notes = sum(counts.get(i, (0, 0))[0] for i in ids)
        if notes:
            others.append((zh, en, notes))
    return {"zones": zones, "others": others}


def topics_svg(data, lang, palette, built, signals):
    """signals: {"reviewed", "figures", "paper"}: what the notes are checked and illustrated with,
    shown instead of a note count (the point is quality, not volume)."""
    zh, colors = lang == "zh", PALETTES[palette]
    out = []
    notes_total = sum(n for *_, groups in data["zones"] for *_, n, _ in groups) + sum(n for *_, n in data["others"])
    drafts_total = sum(d for *_, groups in data["zones"] for *_, d in groups)
    inner = WIDTH - 2 * INSET
    y = INSET + 14
    _kicker(out, INSET, y, _pick(zh, "内容一览", "What's inside"), colors)
    y = _stats(out, INSET, y + 36, [(str(signals["reviewed"]), _pick(zh, "篇中英文逐篇复核", "notes re-read in both languages")),
                                    (str(signals["figures"]), _pick(zh, "张交互与分步图解", "interactive figures")),
                                    (str(signals["paper"]), _pick(zh, "张论文原图", "figures from papers")),
                                    (str(drafts_total), _pick(zh, "篇草稿待审核", "drafts in review"))], colors, inner) + 10
    gap = 14
    col_w = (inner - gap) / 2
    zones = data["zones"]
    def row_height(lines, drafts):
        return 14 + 19 * len(lines) + (16 if drafts else 0)

    for row in range(0, len(zones), 2):
        tops, pair = [], []
        for zone_zh, zone_en, groups in zones[row:row + 2]:
            rows = [(wrap(_pick(zh, name_zh, name_en), col_w - 74, 14), notes, drafts)
                    for name_zh, name_en, notes, drafts in groups]
            pair.append((zone_zh, zone_en, rows))
        h = max(sum(row_height(lines, drafts) for lines, _, drafts in rows) for *_, rows in pair) + 8
        for k, (zone_zh, zone_en, rows) in enumerate(pair):
            x, yy = INSET + k * (col_w + gap), y
            label = _pick(zh, zone_zh, zone_en)
            number, _, name = label.partition(" · ")
            out.append(f'<circle cx="{x + 11}" cy="{yy + 10}" r="11" fill="none" stroke="{colors["accent"]}" stroke-opacity=".6"/>')
            _text(out, x + 11, yy + 14, number, colors["accent"], 10.5, 700, "middle")
            _text(out, x + 30, yy + 15, name or label, colors["ink"], 14, 650)
            yy += 30
            _tile(out, x, yy, col_w, h, colors, accent=False)
            ty = yy + 8
            for i, (lines, notes, drafts) in enumerate(rows):
                row_h = row_height(lines, drafts)
                if i:
                    out.append(f'<line x1="{x + 12:.1f}" x2="{x + col_w - 12:.1f}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="{colors["tile_rule"]}"/>')
                if drafts:
                    dash = ' stroke-dasharray="3 3"' if drafts == notes else ""
                    out.append(f'<line x1="{x + 1.5:.1f}" y1="{ty + 8:.1f}" x2="{x + 1.5:.1f}" y2="{ty + row_h - 8:.1f}" '
                               f'stroke="{colors["accent"]}" stroke-width="3" stroke-linecap="round"{dash}/>')
                ly = ty + 21
                for line in lines:
                    _text(out, x + 14, ly, line, colors["ink"], 14, 600)
                    ly += 19
                if drafts:
                    tag = _pick(zh, "草稿" if drafts == notes else f"{drafts} 篇草稿",
                                ("draft" if notes == 1 else "drafts") if drafts == notes else f"{drafts} draft" + ("s" if drafts > 1 else ""))
                    _text(out, x + 14, ly - 3, tag, colors["muted"], 12)
                _text(out, x + col_w - 14, ty + 21, _pick(zh, f"{notes} 篇", str(notes)), colors["ink"], 14, 700, "end")
                ty += row_h
            tops.append(yy + h)
        y = max(tops) + 16
    x = INSET
    y += 20
    for name_zh, name_en, notes in data["others"]:
        label, count = _pick(zh, name_zh, name_en), _pick(zh, f"{notes} 篇", str(notes))
        w = _width(label, 13) + _width(count, 13) + 36
        if x + w > WIDTH - INSET:
            x, y = INSET, y + 38
        out.append(f'<rect x="{x:.1f}" y="{y - 19}" width="{w:.0f}" height="28" rx="14" fill="{colors["tile"]}" stroke="{colors["tile_rule"]}"/>')
        _text(out, x + 13, y, label, colors["ink"], 13, 600)
        _text(out, x + w - 13, y, count, colors["muted"], 13, anchor="end")
        x += w + 8
    y += 30
    for line in wrap(_pick(zh, f"{built} 构建 · 左侧色条是待审核草稿 · 随网站每次部署更新",
                             f"Built {built} · an amber bar marks drafts pending review · updated on every deploy"), inner, 11.5):
        _text(out, INSET, y, line, colors["muted"], 11.5)
        y += 15
    title = _pick(zh, "内容一览", "What's inside")
    desc = _pick(zh, f"按板块列出的主题和篇数，共 {notes_total} 篇。", f"Topics by area with note counts, {notes_total} notes in all.")
    return _panel(WIDTH, y + INSET - 10, colors, title, desc, out, glow=(0.05, 0.0))


def load_milestones(path):
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _icon(out, kind, cx, cy, colors):
    a = colors["accent"]
    if kind == "done":
        out.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{a}"/><path d="M{cx - 3.6} {cy + .2}l2.4 2.4 4.8-5" '
                   f'fill="none" stroke="{colors["bottom"]}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>')
    elif kind == "now":
        out.append(f'<circle cx="{cx}" cy="{cy}" r="10" fill="{a}" fill-opacity=".18"/><circle cx="{cx}" cy="{cy}" r="5" fill="{a}"/>')
    elif kind == "next":
        out.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="none" stroke="{a}" stroke-width="1.6"/>')
    else:
        out.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="{colors["muted"]}" stroke-dasharray="2 2"/>'
                   f'<path d="M{cx - 3.5} {cy}h7M{cx} {cy - 3.5}v7" stroke="{colors["muted"]}" stroke-width="1.5" stroke-linecap="round"/>')


def roadmap_svg(data, lang, palette, built):
    zh, colors = lang == "zh", PALETTES[palette]
    out, inner = [], WIDTH - 2 * INSET
    y = INSET + 14
    _kicker(out, INSET, y, _pick(zh, "路线图", "Roadmap"), colors)
    y += 26
    # now and next, side by side
    gap = 14
    col_w = (inner - gap) / 2
    bottoms = []
    for k, (key, label_zh, label_en) in enumerate((("now", "现在", "Now"), ("next", "接下来", "Next"))):
        x, yy = INSET + k * (col_w + gap), y
        _icon(out, key, x + 10, yy + 8, colors)
        _text(out, x + 28, yy + 13, _pick(zh, label_zh, label_en), colors["ink"], 15, 700)
        yy += 26
        items = [wrap(_pick(zh, item["zh"], item["en"]), col_w - 46, 14) for item in data.get(key, [])]
        h = sum(16 + 19 * len(lines) for lines in items) + 10
        if key == "now":
            out.append(f'<rect x="{x - 2:.1f}" y="{yy - 2:.1f}" width="{col_w + 4:.1f}" height="{h + 4}" rx="13" fill="{colors["accent"]}" fill-opacity=".08"/>'
                       f'<rect x="{x:.1f}" y="{yy:.1f}" width="{col_w:.1f}" height="{h}" rx="11" fill="{colors["tile"]}" stroke="{colors["accent"]}" stroke-opacity=".75"/>')
        else:
            out.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{col_w:.1f}" height="{h}" rx="11" fill="{colors["tile"]}" '
                       f'stroke="{colors["tile_rule"]}" stroke-dasharray="4 3"/>')
        ty = yy + 8
        for lines in items:
            out.append(f'<circle cx="{x + 18:.1f}" cy="{ty + 15:.1f}" r="3" fill="{colors["accent"] if key == "now" else colors["muted"]}"/>')
            ly = ty + 20
            for line in lines:
                _text(out, x + 30, ly, line, colors["ink"], 14)
                ly += 19
            ty += 16 + 19 * len(lines)
        bottoms.append(yy + h)
    y = max(bottoms) + 30
    # done: a timeline, oldest on the left
    _icon(out, "done", INSET + 10, y - 5, colors)
    _text(out, INSET + 28, y, _pick(zh, "已完成", "Done"), colors["ink"], 15, 700)
    y += 22
    done = data.get("done", [])
    step = inner / max(1, len(done))
    out.append(f'<line x1="{INSET + step / 2:.1f}" x2="{INSET + inner - step / 2:.1f}" y1="{y:.1f}" y2="{y:.1f}" '
               f'stroke="{colors["accent"]}" stroke-opacity=".55" stroke-width="2"/>')
    low = y
    for i, item in enumerate(done):
        cx = INSET + step * (i + .5)
        out.append(f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="5" fill="{colors["accent"]}"/>')
        _text(out, cx, y + 24, item["date"], colors["accent"], 12, 700, "middle")
        ly = y + 42
        for line in wrap_even(_pick(zh, item["zh"], item["en"]), step - 8, 12.5):
            _text(out, cx, ly, line, colors["ink"], 12.5, anchor="middle")
            ly += 16
        low = max(low, ly)
    y = low + 22
    # help wanted
    _icon(out, "open", INSET + 10, y - 5, colors)
    _text(out, INSET + 28, y, _pick(zh, "欢迎参与", "Help wanted"), colors["ink"], 15, 700)
    y += 30
    x = INSET
    for item in data.get("open", []):
        label = _pick(zh, item["zh"], item["en"])
        w = _width(label, 13) + 28
        if x + w > WIDTH - INSET:
            x, y = INSET, y + 38
        out.append(f'<rect x="{x:.1f}" y="{y - 19}" width="{w:.0f}" height="28" rx="14" fill="none" '
                   f'stroke="{colors["tile_rule"]}" stroke-dasharray="4 3"/>')
        _text(out, x + w / 2, y, label, colors["ink"], 13, anchor="middle")
        x += w + 8
    y += 32
    _text(out, INSET, y, _pick(zh, f"{built} 构建 · 内容来自 site/milestones.toml", f"Built {built} · from site/milestones.toml"),
          colors["muted"], 11.5)
    title = _pick(zh, "路线图", "Roadmap")
    desc = _pick(zh, "现在、接下来、已完成和欢迎参与的方向。", "What is in progress, next, done, and open to contributors.")
    return _panel(WIDTH, y + INSET - 8, colors, title, desc, out, glow=(0.2, 0.0))


# ---------------------------------------------------------------- people

def _get(url, token=None, accept="application/vnd.github+json"):
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "cooking-agi-site-build"})
    if token and url.startswith("https://api.github.com/"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def avatar_uri(login, cache):
    if not login:
        return None
    if login not in cache:
        try:
            data = _get(f"https://github.com/{urllib.parse.quote(login)}.png?size=96", accept="image/*")
            cache[login] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            cache[login] = None
    return cache[login]


def review_counts(repo, logins):
    """PRs each person has reviewed in the repo (GitHub search), or {} when the API is unreachable."""
    token, out = os.environ.get("GITHUB_TOKEN"), {}
    for login in logins:
        q = urllib.parse.quote(f"repo:{repo} is:pr reviewed-by:{login}")
        try:
            out[login] = json.loads(_get(f"https://api.github.com/search/issues?q={q}&per_page=1", token))["total_count"]
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
            return {}
    return out


def next_stops(outcomes):
    """[(company name, count, logo data URI or None)] from community/outcomes.json, consented records only."""
    import company_icons
    from pathlib import Path
    counts, names = {}, outcomes.get("companies", {})
    for record in outcomes.get("records", []):
        if record.get("consent"):
            counts[record["company"]] = counts.get(record["company"], 0) + 1
    out = []
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        entry = company_icons.resolve(key, names.get(key, key))
        logo = None
        if entry and entry.get("asset") and entry.get("native"):
            svg = (Path(company_icons.ROOT) / "site/static/company-icons" / entry["asset"]).read_bytes()
            logo = "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")
        out.append(((entry or {}).get("name") or names.get(key, key), n, logo))
    return out


def people_data(people, collaboration, repo, outcomes=None):
    """Humans by commits, AI collaborators by name, reviewers with their areas and review counts."""
    humans = sorted((p for p in people if p["kind"] == "human" and p.get("commits")),
                    key=lambda p: -(p["commits"] or 0))
    ais = [p["name"] for p in people if p["kind"] == "ai"]
    areas = {}
    for area in collaboration.get("area", []):
        for login in area.get("reviewers", []):
            areas.setdefault(login, []).append((area["zh"], area["en"]))
    counts = review_counts(repo, list(areas))
    cache = {}
    contributors = [{"name": p["name"], "login": p.get("login"), "commits": p["commits"],
                     "avatar": avatar_uri(p.get("login"), cache)} for p in humans]
    reviewers = [{"login": login, "areas": names, "reviews": counts.get(login),
                  "avatar": avatar_uri(login, cache)} for login, names in areas.items()]
    return {"contributors": contributors, "ai": ais, "reviewers": reviewers, "next": next_stops(outcomes or {})}


def _avatar(out, x, y, r, person, colors, key, ring=False):
    cx, cy = x + r, y + r
    if ring:
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r + 3.5}" fill="none" stroke="{colors["accent"]}" stroke-opacity=".55" stroke-width="1.5"/>')
    if person.get("avatar"):
        out.append(f'<clipPath id="{key}"><circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath>'
                   f'<image href="{person["avatar"]}" x="{x}" y="{y}" width="{2 * r}" height="{2 * r}" clip-path="url(#{key})"/>'
                   f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{colors["tile_rule"]}"/>')
    else:
        name = person.get("name") or person.get("login") or "?"
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{colors["tile"]}" stroke="{colors["tile_rule"]}"/>')
        _text(out, cx, cy + 5, name[:1].upper(), colors["muted"], 15, 600, "middle")


def people_svg(data, lang, palette, built):
    zh, colors = lang == "zh", PALETTES[palette]
    out, inner = [], WIDTH - 2 * INSET
    y = INSET + 14
    _kicker(out, INSET, y, _pick(zh, "幕后", "Behind the notes"), colors)
    commits = sum(p["commits"] for p in data["contributors"])
    humans, reviewers = len(data["contributors"]), len(data["reviewers"])
    y = _stats(out, INSET, y + 36, [(str(humans), _pick(zh, "位贡献者", "contributor" + "s" * (humans != 1))),
                                    (str(commits), _pick(zh, "次提交", "commit" + "s" * (commits != 1))),
                                    (str(reviewers), _pick(zh, "位审稿人", "reviewer" + "s" * (reviewers != 1)))], colors, inner * 3 / 4) + 4
    _text(out, INSET, y, _pick(zh, "贡献者", "Contributors"), colors["ink"], 15, 700)
    y += 12
    gap = 12
    per_row = 3
    card_w, card_h = (inner - (per_row - 1) * gap) / per_row, 64
    for i, person in enumerate(data["contributors"]):
        x = INSET + (i % per_row) * (card_w + gap)
        row_y = y + (i // per_row) * (card_h + gap)
        out.append(f'<g><title>{esc(person["name"])}</title>')
        _tile(out, x, row_y, card_w, card_h, colors, accent=False)
        _avatar(out, x + 12, row_y + 12, 20, person, colors, f"c{i}", ring=i == 0)
        _text(out, x + 62, row_y + 28, person["name"][:16], colors["ink"], 14, 650)
        _text(out, x + 62, row_y + 47, _pick(zh, f"{person['commits']} 次提交", f"{person['commits']} commits"), colors["muted"], 12.5)
        out.append("</g>")
    y += max(1, -(-len(data["contributors"]) // per_row)) * (card_h + gap)
    if data["ai"]:
        x = INSET
        label = _pick(zh, "一起写的 AI：", "Written with AI:")
        _text(out, x, y + 16, label, colors["muted"], 13)
        x += _width(label, 13) + 10
        for name in data["ai"]:
            w = _width(name, 13) + 26
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" rx="12" fill="none" stroke="{colors["tile_rule"]}"/>')
            _text(out, x + w / 2, y + 17, name, colors["ink"], 13, 600, "middle")
            x += w + 8
        y += 36
    y += 22
    _text(out, INSET, y, _pick(zh, "审稿人", "Reviewers"), colors["ink"], 15, 700)
    y += 12
    card_w = (inner - gap) / 2
    for i, person in enumerate(data["reviewers"]):
        x = INSET + (i % 2) * (card_w + gap)
        row_y = y + (i // 2) * 82
        names = [_pick(zh, a, b) for a, b in person["areas"]]
        out.append(f'<g><title>{esc(", ".join(names))}</title>')
        _tile(out, x, row_y, card_w, 70, colors, accent=True)
        _avatar(out, x + 16, row_y + 13, 22, {**person, "name": person["login"]}, colors, f"r{i}")
        _text(out, x + 70, row_y + 31, person["login"], colors["ink"], 14, 650)
        detail = _pick(zh, f"{len(names)} 个领域", f"{len(names)} areas")
        if person.get("reviews"):
            detail += _pick(zh, f" · 审过 {person['reviews']} 个 PR", f" · {person['reviews']} PRs reviewed")
        _text(out, x + 70, row_y + 51, detail, colors["muted"], 12.5)
        out.append("</g>")
    y += max(1, -(-len(data["reviewers"]) // 2)) * 82 - 4
    if data.get("next"):
        y += 26
        _text(out, INSET, y, _pick(zh, "读者的下一站", "Where readers went next"), colors["ink"], 15, 700)
        y += 18
        _text(out, INSET, y, _pick(zh, "读者自愿分享，不代表笔记促成了这些机会", "Self-reported and voluntary; not a claim that the notes got anyone a job"),
              colors["muted"], 12)
        y += 14
        x = INSET
        for name, count, logo in data["next"]:
            label = f"{name} · {count}"
            w = _width(label, 13) + (54 if logo else 28)
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="36" rx="18" fill="{colors["tile"]}" stroke="{colors["tile_rule"]}"/>')
            tx = x + 14
            if logo:
                out.append(f'<image href="{logo}" x="{x + 11}" y="{y + 9}" width="18" height="18"/>')
                tx = x + 37
            _text(out, tx, y + 23, label, colors["ink"], 13, 600)
            x += w + 8
        y += 44
    y += 22
    for line in wrap(_pick(zh, f"{built} 构建 · 来自 git 记录、GitHub 与 site/collaboration.toml，随网站每次部署更新",
                             f"Built {built} · from git history, GitHub and site/collaboration.toml, updated on every deploy"), inner, 11.5):
        _text(out, INSET, y, line, colors["muted"], 11.5)
        y += 15
    title = _pick(zh, "贡献者与审稿人", "Contributors and reviewers")
    desc = _pick(zh, f"{len(data['contributors'])} 位贡献者，{len(data['reviewers'])} 位审稿人。",
                 f"{len(data['contributors'])} contributors and {len(data['reviewers'])} reviewers.")
    return _panel(WIDTH, y + INSET - 10, colors, title, desc, out, glow=(0.9, 0.0))
