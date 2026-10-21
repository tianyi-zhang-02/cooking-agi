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
import random
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
# The README cards share the site's night sky in both GitHub themes: black, faint stars, amber accents.
NIGHT = {"ink": "#f0f0f0", "muted": "#a4a4a4", "accent": "#e8a672", "rule": "#262626",
         "top": "#000000", "bottom": "#000000", "tile": "#0b0b0b", "tile_top": "#151515", "tile_rule": "#262626",
         "shadow": "#000000", "shadow_opacity": ".6", "glow": "#e8a672", "glow_opacity": ".10", "soft": "#e8a672"}
PALETTES = {"light": NIGHT, "dark": NIGHT}
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
            f'stop-opacity="{colors["glow_opacity"]}"/><stop offset="1" stop-color="{colors["glow"]}" stop-opacity="0"/></radialGradient>'
            f'<linearGradient id="tileg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{colors["tile_top"]}"/>'
            f'<stop offset="1" stop-color="{colors["tile"]}"/></linearGradient>'
            f'<filter id="lift" x="-10%" y="-20%" width="120%" height="150%"><feDropShadow dx="0" dy="1.5" stdDeviation="1.6" '
            f'flood-color="{colors["shadow"]}" flood-opacity="{colors["shadow_opacity"]}"/></filter>'
            '<radialGradient id="dawnlight" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#fff" stop-opacity=".5"/>'
            '<stop offset=".6" stop-color="#fff" stop-opacity=".18"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>'
            '<filter id="dawn" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">'
            '<feColorMatrix type="matrix" values=".3 .59 .11 0 0 .3 .59 .11 0 0 .3 .59 .11 0 0 0 0 0 1 0" result="g"/>'
            '<feTurbulence type="fractalNoise" baseFrequency="1.9" numOctaves="1" seed="9" result="n"/>'
            '<feColorMatrix in="n" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="ng"/>'
            '<feComposite in="g" in2="ng" operator="arithmetic" k2="1" k3="1.15" k4="-.575" result="d"/>'
            '<feComponentTransfer in="d"><feFuncR type="discrete" tableValues="0 1"/><feFuncG type="discrete" tableValues="0 1"/>'
            '<feFuncB type="discrete" tableValues="0 1"/><feFuncA type="identity"/></feComponentTransfer>'
            '<feColorMatrix type="matrix" values="0 0 0 0 .91 0 0 0 0 .65 0 0 0 0 .45 1 0 0 0 0"/></filter></defs>'
            '<style>.tw{animation:tw 4s ease-in-out infinite}@keyframes tw{50%{opacity:.08}}'
            '@media (prefers-reduced-motion: reduce){.tw{animation:none}}</style>')
    rng = random.Random(int(height))
    stars = "".join(f'<rect x="{rng.uniform(8, width - 8):.1f}" y="{rng.uniform(8, height - 8):.1f}" width="{s}" height="{s}" fill="#fff" '
                    f'opacity="{rng.uniform(.18, .6):.2f}"{" class=\"tw\"" if rng.random() < .1 else ""}/>'
                    for s in (rng.choice((1, 1, 1, 1.5)) for _ in range(int(width * height / 2400))))
    # a dithered band of dawn light along the bottom edge, as on the site's home page
    dawn = (f'<g clip-path="url(#card)" filter="url(#dawn)"><ellipse cx="{width / 2}" cy="{height + 64:.0f}" rx="{width * .7:.0f}" ry="84" '
            f'fill="url(#dawnlight)"/></g>')
    frame = (f'<clipPath id="card"><rect x="{PAD}" y="{PAD}" width="{width - 2 * PAD}" height="{height - 2 * PAD:.0f}" rx="18"/></clipPath>'
             f'<rect x="{PAD}" y="{PAD}" width="{width - 2 * PAD}" height="{height - 2 * PAD:.0f}" rx="18" fill="url(#bg)"/>'
             f'{dawn}{stars}'
             f'<rect x="{PAD}" y="{PAD}" width="{width - 2 * PAD}" height="{height - 2 * PAD:.0f}" rx="18" fill="url(#glow)"/>'
             f'<rect x="{PAD + .5}" y="{PAD + .5}" width="{width - 2 * PAD - 1}" height="{height - 2 * PAD - 1:.0f}" rx="17.5" '
             f'fill="none" stroke="{colors["rule"]}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" width="{width}" '
            f'height="{height:.0f}" role="img" aria-labelledby="t d" font-family="{esc(FONT)}">'
            f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>{defs}{frame}' + "".join(body) + "</svg>\n")


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
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="11" fill="url(#tileg)" '
               f'stroke="{colors["tile_rule"]}" filter="url(#lift)"/>')
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
    stats = [(str(signals["reviewed"]), _pick(zh, "篇中英文逐篇复核", "notes re-read in both languages")),
             (str(signals["figures"]), _pick(zh, "张交互与分步图解", "interactive figures")),
             (str(signals["paper"]), _pick(zh, "张论文原图", "figures from papers"))]
    if drafts_total:  # only while some notes still wait for the owner's review
        stats.append((str(drafts_total), _pick(zh, "篇草稿待审核", "drafts in review")))
    y = _stats(out, INSET, INSET + 30, stats, colors, inner) + 10
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
            out.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="26" height="19" rx="9.5" fill="{colors["accent"]}" fill-opacity=".14"/>')
            _text(out, x + 13, yy + 13.5, number, colors["accent"], 10.5, 700, "middle")
            _text(out, x + 34, yy + 14.5, name or label, colors["ink"], 14, 650)
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
        out.append(f'<rect x="{x:.1f}" y="{y - 19}" width="{w:.0f}" height="28" rx="14" fill="url(#tileg)" '
                   f'stroke="{colors["tile_rule"]}" filter="url(#lift)"/>')
        _text(out, x + 13, y, label, colors["ink"], 13, 600)
        _text(out, x + w - 13, y, count, colors["muted"], 13, anchor="end")
        x += w + 8
    y += 30
    marks = _pick(zh, " · 左侧色条是待审核草稿", " · an amber bar marks drafts pending review") if drafts_total else ""
    for line in wrap(_pick(zh, f"{built} 构建{marks} · 随网站每次部署更新",
                             f"Built {built}{marks} · updated on every deploy"), inner, 11.5):
        _text(out, INSET, y, line, colors["muted"], 11.5)
        y += 15
    title = _pick(zh, "内容一览", "What's inside")
    desc = _pick(zh, f"按板块列出的主题和篇数，共 {notes_total} 篇。", f"Topics by area with note counts, {notes_total} notes in all.")
    return _panel(WIDTH, y + INSET + 14, colors, title, desc, out, glow=(0.05, 0.0))


def load_milestones(path):
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _icon(out, kind, cx, cy, colors):
    """Status badges for the roadmap: the same soft circle, a different glyph."""
    a = colors["accent"]
    out.append(f'<circle cx="{cx}" cy="{cy}" r="10" fill="{a}" fill-opacity="{".9" if kind == "done" else ".15"}"/>')
    line = f'fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
    if kind == "done":
        out.append(f'<path d="M{cx - 4} {cy + .3}l2.7 2.7 5.3-5.6" stroke="{colors["bottom"]}" {line}/>')
    elif kind == "now":
        out.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{a}"/>')
    elif kind == "next":
        out.append(f'<path d="M{cx - 4} {cy}h8M{cx + .8} {cy - 3.4}l3.4 3.4-3.4 3.4" stroke="{a}" {line}/>')
    else:
        out.append(f'<path d="M{cx - 4} {cy}h8M{cx} {cy - 4}v8" stroke="{a}" {line}/>')


def roadmap_svg(data, lang, palette, built):
    zh, colors = lang == "zh", PALETTES[palette]
    out, inner = [], WIDTH - 2 * INSET
    y = INSET + 4
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
    done = data.get("done", [])[-5:]  # the latest five; older ones stay in site/milestones.toml
    step = inner / max(1, len(done))
    out.append(f'<line x1="{INSET + step / 2:.1f}" x2="{INSET + inner - step / 2:.1f}" y1="{y:.1f}" y2="{y:.1f}" '
               f'stroke="{colors["accent"]}" stroke-opacity=".55" stroke-width="2"/>')
    low = y
    for i, item in enumerate(done):
        cx = INSET + step * (i + .5)
        out.append(f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="9" fill="{colors["accent"]}" fill-opacity=".15"/>'
                   f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="4.5" fill="{colors["accent"]}"/>')
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
    return _panel(WIDTH, y + INSET + 16, colors, title, desc, out, glow=(0.2, 0.0))


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
    """[{"name", "logo": SVG text or None, "native": full colour?}] for each company in
    community/outcomes.json with a consented record, most frequent first. The card shows marks only,
    no counts; a company without a local mark is shown by name."""
    import company_icons
    from pathlib import Path
    counts, names = {}, outcomes.get("companies", {})
    for record in outcomes.get("records", []):
        if record.get("consent"):
            counts[record["company"]] = counts.get(record["company"], 0) + 1
    out = []
    for key, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        entry = company_icons.resolve(key, names.get(key, key)) or {}
        logo = None
        if entry.get("asset"):
            logo = (Path(company_icons.ROOT) / "site/static/company-icons" / entry["asset"]).read_text(encoding="utf-8")
        out.append({"name": entry.get("name") or names.get(key, key), "logo": logo, "native": bool(entry.get("native"))})
    return out


# AI collaborators have no avatar; their marks are drawn on a shaded disc (site/static/crew-icons/SOURCES.md)
AI_MARKS = {"Claude Code": "site/static/crew-icons/claude.svg", "Codex": "site/static/company-icons/openai.svg"}


def people_data(people, collaboration, repo, outcomes=None, today=None):
    """{"crew", "reviewers", "next"}. The crew is everyone with a commit or co-author credit, numbered
    CG 001, CG 002 ... by first participation and shown by GitHub username (AI collaborators by name)."""
    from datetime import date, datetime, timedelta, timezone
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    today = today or datetime.now(timezone.utc).date()
    monday = (today - timedelta(days=today.weekday())).isoformat()  # this week, UTC Monday to Sunday
    cache, crew = {}, []
    members = [p for p in people if p["kind"] == "ai" or p.get("commits")]
    for number, p in enumerate(sorted(members, key=lambda p: (p.get("first") or "", p["name"])), 1):
        login = p.get("login") if p["kind"] == "human" else None
        member = {"name": p["name"], "login": login, "kind": p["kind"], "commits": p.get("commits"),
                  "first": p.get("first") or "", "last": p.get("last") or "", "cg": f"CG {number:03d}",
                  "handle": f"@{login}" if login else p["name"], "this_week": (p.get("last") or "") >= monday}
        if p["kind"] == "human":
            member["avatar"] = avatar_uri(login, cache)
        elif p["name"] in AI_MARKS and (root / AI_MARKS[p["name"]]).exists():
            member["mark"] = (root / AI_MARKS[p["name"]]).read_text(encoding="utf-8")
        crew.append(member)
    areas = {}
    for area in collaboration.get("area", []):
        for login in area.get("reviewers", []):
            areas.setdefault(login, []).append((area["zh"], area["en"]))
    counts = review_counts(repo, list(areas))
    reviewers = [{"login": login, "handle": f"@{login}", "kind": "human", "areas": names, "reviews": counts.get(login),
                  "avatar": avatar_uri(login, cache)} for login, names in areas.items()]
    return {"crew": crew, "reviewers": reviewers, "next": next_stops(outcomes or {})}


# ---------------------------------------------------------------- the crew card: night sky, end credits, roster
# The same card in both themes: the site's old contributors page drew people as 1-bit figures drifting over a
# dithered planet, followed by film-style credits and a table. Dithering here is an SVG filter (grey + noise +
# threshold), so the build needs no image library.

CREW_INK, CREW_MUTED, CREW_FAINT, CREW_AMBER, CREW_RULE = "#f0f0f0", "#a4a4a4", "#6f6f6f", "#e8a672", "#262626"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"


def _say(out, x, y, value, fill=CREW_INK, size=13, weight=400, anchor="middle", mono=False, spacing=None):
    family = f' font-family="{esc(MONO)}"' if mono else ""
    spacing = f' letter-spacing="{spacing}"' if spacing else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
               f'text-anchor="{anchor}"{family}{spacing}>{esc(value)}</text>')


def _disc(out, cx, cy, r, member, key, ring=False):
    """A 1-bit disc: the avatar, or an AI mark on a shaded disc, through the dither filter."""
    out.append(f'<clipPath id="{key}"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"/></clipPath><g clip-path="url(#{key})" filter="url(#dither)">')
    if member.get("avatar"):
        out.append(f'<image href="{member["avatar"]}" x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" '
                   f'preserveAspectRatio="xMidYMid slice"/></g>')
    else:
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="url(#shade)"/>')
        if member.get("mark"):
            mark = re.sub(r'\sfill="[^"]*"', "", member["mark"]).replace("<svg ", '<svg fill="#111" ', 1)
            uri = "data:image/svg+xml;base64," + base64.b64encode(mark.encode("utf-8")).decode("ascii")
            out.append(f'<image href="{uri}" x="{cx - r * .6:.1f}" y="{cy - r * .6:.1f}" width="{r * 1.2:.1f}" height="{r * 1.2:.1f}"/></g>')
        else:
            out.append("</g>")
            _say(out, cx, cy + r * .35, (member.get("login") or member["name"])[:1].upper(), "#111", round(r * .9), 700)
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="none" stroke="#000"/>')
    if ring:  # committed this week
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 4}" fill="none" stroke="{CREW_AMBER}" stroke-width="1.2" '
                   f'stroke-dasharray="2 3" opacity=".9"/>')


def people_svg(data, lang, palette, built):
    """palette is accepted for symmetry with the other cards; the night sky is the same in both themes."""
    zh = lang == "zh"
    say = lambda chinese, english: chinese if zh else english
    out, css, rng = [], [], random.Random(7)
    crew = data["crew"]
    # night sky
    sky = 330
    stars = []
    for i in range(260):
        size = rng.choice((1, 1, 1, 1.5, 2))
        twinkle = f' class="tw{i % 3}"' if rng.random() < .12 else ""
        stars.append(f'<rect x="{rng.uniform(4, WIDTH - 4):.1f}" y="{rng.uniform(4, sky - 40):.1f}" width="{size}" height="{size}" '
                     f'fill="#fff" opacity="{rng.uniform(.35, .95):.2f}"{twinkle}/>')
    out.append("".join(stars))
    radius, horizon = 900, sky - 96
    out.append(f'<clipPath id="sky"><rect width="{WIDTH}" height="{sky}"/></clipPath><g clip-path="url(#sky)">'
               f'<g filter="url(#dither)"><circle cx="{WIDTH / 2}" cy="{horizon + radius}" r="{radius + 40}" fill="url(#earth)"/></g>'
               f'<circle cx="{WIDTH / 2}" cy="{horizon + radius}" r="{radius}" fill="none" stroke="#fff" stroke-width="1.3" opacity=".9"/></g>')
    _say(out, WIDTH / 2, 46, "THE PEOPLE BEHIND THE NOTES", CREW_MUTED, 10.5, 600, mono=True, spacing=".28em")
    _say(out, WIDTH / 2, 80, say("谢谢一起参与的你们。", "Thank you, everyone who pitched in."), CREW_INK, 21, 700)
    spots = [(.27, .44), (.72, .38), (.52, .58), (.12, .56), (.88, .55), (.40, .32), (.64, .66), (.20, .30), (.82, .28)]
    for i, member in enumerate(crew[:len(spots)]):
        cx, cy = WIDTH * spots[i][0], sky * spots[i][1]
        r = 27 if member["kind"] == "human" else 23
        dx, dy = (6, -7, 5, -5)[i % 4], (-8, 6, -6, 7)[i % 4]
        css.append(f".f{i}{{animation:f{i} {13 + 4 * (i % 3)}s ease-in-out infinite alternate}}"
                   f"@keyframes f{i}{{from{{transform:translate(0,0)}}to{{transform:translate({dx}px,{dy}px)}}}}")
        out.append(f'<g class="f{i}"><title>{esc(member["handle"])}</title>')
        _disc(out, cx, cy, r, member, f"s{i}", ring=member["this_week"])
        _say(out, cx, cy + r + 15, member["handle"], CREW_MUTED, 10.5, mono=True)
        out.append("</g>")
    week = sum(member["this_week"] for member in crew)
    if week:
        _say(out, WIDTH / 2, sky + 6, say(f"本周有 {week} 位留下了提交", f"{week} left commits this week"), CREW_AMBER, 11.5, 600)
    # end credits
    y = sky + 56
    _say(out, WIDTH / 2, y, say("演 职 员 表", "C R E D I T S"), CREW_MUTED, 11, 600, spacing=".3em")
    y += 34
    since = lambda m: say(f'{m["cg"]} · {m["first"][:7]} 起', f'{m["cg"]} · since {m["first"][:7]}')
    reviewed = lambda m: (say(f'{len(m["areas"])} 个领域', f'{len(m["areas"])} areas') +
                          (say(f' · 审过 {m["reviews"]} 个 PR', f' · {m["reviews"]} PRs reviewed') if m.get("reviews") else ""))
    for heading, members, meta in ((say("笔记与代码", "NOTES & CODE"), [m for m in crew if m["kind"] == "human"], since),
                                   (say("AI 协作者", "AI COLLABORATORS"), [m for m in crew if m["kind"] == "ai"], since),
                                   (say("审稿人", "REVIEWERS"), data["reviewers"], reviewed)):
        if not members:
            continue
        _say(out, WIDTH / 2, y, heading, CREW_FAINT, 10.5, 600, spacing=".22em")
        y += 22
        for row in range(0, len(members), 3):
            chunk = members[row:row + 3]
            step = WIDTH / (len(chunk) + 1)
            for k, member in enumerate(chunk):
                cx = step * (k + 1)
                _disc(out, cx, y + 22, 20, member, f"c{row + k}{heading[:2].encode().hex()}", ring=member.get("this_week"))
                _say(out, cx, y + 64, member["handle"], CREW_INK, 15, 700)
                _say(out, cx, y + 82, meta(member), CREW_MUTED, 10.5, mono=True)
            y += 118
        y += 8
    if data.get("next"):
        _say(out, WIDTH / 2, y, say("读者的下一站", "WHERE READERS WENT NEXT"), CREW_FAINT, 10.5, 600, spacing=".22em")
        y += 30
        # marks only; a company without a local mark is written out by name, never drawn as an invented logo
        widths = [26 if c["logo"] else _width(c["name"], 14) for c in data["next"]]
        x = WIDTH / 2 - (sum(widths) + 30 * (len(widths) - 1)) / 2
        for company, w in zip(data["next"], widths):
            out.append(f'<g><title>{esc(company["name"])}</title>')
            if company["logo"]:
                mark = re.sub(r'\sfill="[^"]*"', "", company["logo"]).replace("<svg ", f'<svg fill="{CREW_INK}" ', 1)
                uri = "data:image/svg+xml;base64," + base64.b64encode(mark.encode("utf-8")).decode("ascii")
                out.append(f'<image href="{uri}" x="{x:.1f}" y="{y - 20}" width="26" height="26"/>')
            else:
                _say(out, x, y, company["name"], CREW_INK, 14, 600, anchor="start")
            out.append("</g>")
            x += w + 30
        y += 30
        _say(out, WIDTH / 2, y, say("读者自愿分享，不代表笔记促成了这些机会", "Self-reported; not a claim that the notes got anyone a job"),
             CREW_FAINT, 10.5)
        y += 34
    # roster
    out.append(f'<line x1="40" x2="{WIDTH - 40}" y1="{y}" y2="{y}" stroke="{CREW_RULE}"/>')
    y += 34
    _say(out, 40, y, say("贡献榜单", "Roster"), CREW_INK, 15, 700, anchor="start")
    y += 30
    for x, heading in ((40, say("编号", "No.")), (110, say("贡献者", "Contributor")), (300, say("提交", "Commits")),
                       (410, say("第一次", "First")), (500, say("最近", "Latest"))):
        _say(out, x, y, heading, CREW_FAINT, 10.5, 600, anchor="start", spacing=".08em")
    y += 12
    out.append(f'<line x1="40" x2="{WIDTH - 40}" y1="{y}" y2="{y}" stroke="{CREW_RULE}"/>')
    top = max([m["commits"] or 0 for m in crew] + [1])
    for i, member in enumerate(crew):
        row = y + 40
        _say(out, 40, row, member["cg"], CREW_FAINT, 11, mono=True, anchor="start")
        _disc(out, 124, row - 4, 12, member, f"t{i}")
        _say(out, 144, row, member["handle"], CREW_INK, 13, 600, anchor="start")
        commits = member["commits"]
        if commits:
            out.append("".join(f'<rect x="{300 + 4 * k}" y="{row - 16}" width="2" height="5" fill="{CREW_AMBER}"/>'
                               for k in range(max(1, round(24 * commits / top)))))
        _say(out, 300, row + 2, str(commits) if commits else "—", CREW_INK, 13, 700, anchor="start")
        _say(out, 410, row, member["first"] or "—", CREW_MUTED, 11, mono=True, anchor="start")
        _say(out, 500, row, member["last"] or "—", CREW_MUTED, 11, mono=True, anchor="start")
        if member["this_week"]:
            _say(out, WIDTH - 38, row, say("本周", "this week"), CREW_AMBER, 10, 600, anchor="end")
        y += 56
        out.append(f'<line x1="40" x2="{WIDTH - 40}" y1="{y}" y2="{y}" stroke="#1a1a1a"/>')
    y += 30
    _say(out, 40, y, say(f"{built} 更新 · 提交数不是质量排名，也不决定审核权限",
                         f"Updated {built} · commit counts are not a ranking of quality or review rights"), CREW_FAINT, 10.5, anchor="start")
    height = y + 26
    defs = ('<defs><filter id="dither" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">'
            '<feColorMatrix type="matrix" values=".3 .59 .11 0 0 .3 .59 .11 0 0 .3 .59 .11 0 0 0 0 0 1 0" result="g"/>'
            '<feTurbulence type="fractalNoise" baseFrequency="1.9" numOctaves="1" seed="5" result="n"/>'
            '<feColorMatrix in="n" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="ng"/>'
            '<feComposite in="g" in2="ng" operator="arithmetic" k2="1" k3="1.15" k4="-.575" result="d"/>'
            '<feComponentTransfer in="d"><feFuncR type="discrete" tableValues="0 1"/><feFuncG type="discrete" tableValues="0 1"/>'
            '<feFuncB type="discrete" tableValues="0 1"/></feComponentTransfer><feComposite in2="SourceAlpha" operator="in"/></filter>'
            '<radialGradient id="earth" cx=".5" cy=".5" r=".5"><stop offset=".86" stop-color="#fff" stop-opacity="0"/>'
            '<stop offset=".94" stop-color="#fff" stop-opacity=".45"/><stop offset=".957" stop-color="#fff" stop-opacity=".95"/>'
            '<stop offset=".962" stop-color="#fff" stop-opacity=".35"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>'
            '<radialGradient id="shade" cx=".38" cy=".32" r=".75"><stop offset="0" stop-color="#fff"/><stop offset=".7" stop-color="#cfcfcf"/>'
            '<stop offset="1" stop-color="#7a7a7a"/></radialGradient></defs>')
    style = ("<style>" + "".join(css) + ".tw0{animation:tw 3.2s ease-in-out infinite}.tw1{animation:tw 4.1s ease-in-out infinite .8s}"
             ".tw2{animation:tw 5.3s ease-in-out infinite 1.6s}@keyframes tw{50%{opacity:.15}}"
             "@media (prefers-reduced-motion: reduce){*{animation:none!important}}</style>")
    humans = sum(m["kind"] == "human" for m in crew)
    title = say("贡献者与审稿人", "Contributors and reviewers")
    desc = say(f"{humans} 位贡献者、{len(crew) - humans} 个 AI 协作者、{len(data['reviewers'])} 位审稿人。",
               f"{humans} contributors, {len(crew) - humans} AI collaborators and {len(data['reviewers'])} reviewers.")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:.0f}" width="{WIDTH}" height="{height:.0f}" '
            f'role="img" aria-labelledby="t d" font-family="{esc(FONT)}"><title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
            f'{style}{defs}<rect width="{WIDTH}" height="{height:.0f}" rx="18" fill="#000"/>' + "".join(out) + "</svg>\n")
