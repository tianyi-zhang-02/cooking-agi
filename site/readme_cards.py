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
import json
import os
import re
import tomllib
import urllib.error
import urllib.parse
import urllib.request

from topic_coverage import _width

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', 'PingFang SC', "
        "'Hiragino Sans GB', 'Microsoft YaHei', sans-serif")
WIDTH, PAD, INSET = 960, 2, 26
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


def wrap(text, max_width, size):
    """Greedy line wrap using the same rough glyph widths as the coverage chart."""
    tokens = re.findall(r"[A-Za-z0-9&().,:;/'’+\-·µ%]+|\s+|.", text)
    lines, line = [], ""
    for token in tokens:
        candidate = line + token
        if line and _width(candidate.rstrip(), size) > max_width:
            lines.append(line.rstrip())
            line = token.lstrip()
        else:
            line = candidate
    if line.strip():
        lines.append(line.rstrip())
    return lines


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


def _stats(out, x, y, stats, colors):
    """Big numbers with small labels, side by side."""
    for value, label in stats:
        _text(out, x, y, value, colors["ink"], 28, 700, spacing="-.02em")
        _text(out, x, y + 20, label, colors["muted"], 12)
        x += max(_width(value, 28), _width(label, 12)) + 40
    return x


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
    x0, y = INSET, INSET + 14
    _kicker(out, x0, y, _pick(zh, "内容一览", "What's inside"), colors)
    _stats(out, x0, y + 40, [(str(signals["reviewed"]), _pick(zh, "篇已中英文逐篇复核", "notes re-read in both languages")),
                             (str(signals["figures"]), _pick(zh, "张交互与分步图解", "interactive and step-by-step figures")),
                             (str(signals["paper"]), _pick(zh, "张论文原图", "figures from the papers")),
                             (str(drafts_total), _pick(zh, "篇草稿待审核", "drafts in review"))], colors)
    cols, gap = len(data["zones"]), 14
    col_w = (WIDTH - 2 * INSET - (cols - 1) * gap) / cols
    top, bottoms = y + 92, []
    for i, (zone_zh, zone_en, groups) in enumerate(data["zones"]):
        x, yy = INSET + i * (col_w + gap), top
        label = _pick(zh, zone_zh, zone_en)
        number, _, name = label.partition(" · ")
        out.append(f'<circle cx="{x + 11}" cy="{yy - 4}" r="11" fill="none" stroke="{colors["accent"]}" stroke-opacity=".6"/>')
        _text(out, x + 11, yy, number, colors["accent"], 10.5, 700, "middle")
        _text(out, x + 30, yy, name or label, colors["ink"], 12.5, 650)
        yy += 16
        for name_zh, name_en, notes, drafts in groups:
            name_lines = wrap(_pick(zh, name_zh, name_en), col_w - 74, 13)
            h = max(52, 20 + 17 * len(name_lines) + 14)
            tip = _pick(zh, f"{name_zh}：{notes} 篇" + (f"，其中 {drafts} 篇待审核" if drafts else ""),
                        f"{name_en}: {notes} notes" + (f", {drafts} pending review" if drafts else ""))
            out.append(f"<g><title>{esc(tip)}</title>")
            _tile(out, x, yy, col_w, h, colors, accent=True, dashed=bool(drafts) and drafts == notes)
            ty = yy + 23
            for line in name_lines:
                _text(out, x + 14, ty, line, colors["ink"], 13, 600)
                ty += 17
            if drafts:
                tag = _pick(zh, "草稿" if drafts == notes else f"{drafts} 篇草稿", "draft" + ("s" if drafts > 1 else "") if drafts == notes else f"{drafts} draft" + ("s" if drafts > 1 else ""))
                _text(out, x + 14, ty + 1, tag, colors["muted"], 11)
            _text(out, x + col_w - 12, yy + 26, str(notes), colors["ink"], 20, 700, "end", spacing="-.02em")
            _text(out, x + col_w - 12, yy + 41, _pick(zh, "篇", "notes"), colors["muted"], 10.5, anchor="end")
            out.append("</g>")
            yy += h + 8
        bottoms.append(yy)
    y = max(bottoms) + 10
    out.append(f'<line x1="{INSET}" x2="{WIDTH - INSET}" y1="{y:.1f}" y2="{y:.1f}" stroke="{colors["tile_rule"]}"/>')
    y += 28
    x = INSET
    for name_zh, name_en, notes in data["others"]:
        label, count = _pick(zh, name_zh, name_en), _pick(zh, f"{notes} 篇", f"{notes} notes")
        w = _width(label, 12.5) + _width(count, 12) + 40
        out.append(f'<rect x="{x}" y="{y - 17}" width="{w:.0f}" height="26" rx="13" fill="{colors["tile"]}" stroke="{colors["tile_rule"]}"/>')
        _text(out, x + 13, y, label, colors["ink"], 12.5, 600)
        _text(out, x + w - 13, y, count, colors["muted"], 12, anchor="end")
        x += w + 10
    y += 32
    _text(out, INSET, y, _pick(zh, f"{built} 构建 · 虚线是待审核草稿 · 随网站每次部署更新",
                               f"Built {built} · dashed accents mark drafts pending review · updated on every deploy"),
          colors["muted"], 11)
    height = y + INSET - 6
    title = _pick(zh, "内容一览", "What's inside")
    desc = _pick(zh, f"按板块列出的主题和篇数，共 {notes_total} 篇。", f"Topics by area with note counts, {notes_total} notes in all.")
    return _panel(WIDTH, height, colors, title, desc, out, glow=(0.05, 0.0))


# ---------------------------------------------------------------- roadmap

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
    columns = [("done", "已完成", "Done"), ("now", "现在", "Now"), ("next", "接下来", "Next"),
               ("open", "欢迎参与", "Help wanted")]
    out, gap = [], 14
    y0 = INSET + 14
    _kicker(out, INSET, y0, _pick(zh, "路线图", "Roadmap"), colors)
    col_w = (WIDTH - 2 * INSET - 3 * gap) / 4
    # the rail: one line through the four column heads
    rail = y0 + 34
    for i, (key, label_zh, label_en) in enumerate(columns[:-1]):
        start = INSET + i * (col_w + gap) + 28 + _width(_pick(zh, label_zh, label_en), 13.5) + 10
        end = INSET + (i + 1) * (col_w + gap) - 2
        done = key in ("done", "now") and i == 0
        out.append(f'<line x1="{start:.1f}" x2="{end:.1f}" y1="{rail}" y2="{rail}" stroke="{colors["accent"] if done else colors["tile_rule"]}" '
                   f'stroke-width="2"' + (' stroke-opacity=".7"' if done else ' stroke-dasharray="4 4"' if key == "next" else "") + '/>')
    bottoms = []
    for i, (key, label_zh, label_en) in enumerate(columns):
        x, y = INSET + i * (col_w + gap), rail
        _icon(out, key, x + 10, y, colors)
        _text(out, x + 28, y + 4.5, _pick(zh, label_zh, label_en), colors["ink"], 13.5, 700)
        y += 24
        for item in data.get(key, []):
            lines = wrap(_pick(zh, item["zh"], item["en"]), col_w - 30, 12.5)
            dated = "date" in item
            h = 16 + (17 if dated else 0) + 17 * len(lines) + 8
            if key == "now":
                out.append(f'<rect x="{x - 2:.1f}" y="{y - 2:.1f}" width="{col_w + 4:.1f}" height="{h + 4}" rx="13" '
                           f'fill="{colors["accent"]}" fill-opacity=".08"/>')
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{col_w:.1f}" height="{h}" rx="11" fill="{colors["tile"]}" '
                           f'stroke="{colors["accent"]}" stroke-opacity=".75"/>')
            else:
                _tile(out, x, y, col_w, h, colors, accent=key == "done", dashed=key in ("next", "open"))
                if key in ("next", "open"):
                    out[-2] = out[-2].replace(f'stroke="{colors["tile_rule"]}"', f'stroke="{colors["tile_rule"]}" stroke-dasharray="4 3"')
            ty = y + 21
            if dated:
                _text(out, x + 14, ty, item["date"], colors["accent"], 11, 700, spacing=".04em")
                ty += 17
            for line in lines:
                _text(out, x + 14, ty, line, colors["ink"], 12.5)
                ty += 17
            y += h + 8
        bottoms.append(y)
    y = max(bottoms) + 16
    _text(out, INSET, y, _pick(zh, f"{built} 构建 · 内容来自 site/milestones.toml，随网站每次部署更新",
                               f"Built {built} · from site/milestones.toml, updated on every deploy"), colors["muted"], 11)
    title = _pick(zh, "路线图", "Roadmap")
    desc = _pick(zh, "已完成、正在做、接下来和欢迎参与的方向。", "What is done, in progress, next, and open to contributors.")
    return _panel(WIDTH, y + INSET - 6, colors, title, desc, out, glow=(0.32, 0.0))


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
    out, y = [], INSET + 14
    _kicker(out, INSET, y, _pick(zh, "幕后", "Behind the notes"), colors)
    commits = sum(p["commits"] for p in data["contributors"])
    _stats(out, INSET, y + 40, [(str(len(data["contributors"])), _pick(zh, "位贡献者", "contributors")),
                                (str(commits), _pick(zh, "次提交", "commits")),
                                (str(len(data["reviewers"])), _pick(zh, "位审稿人", "reviewers"))], colors)
    y += 92
    _text(out, INSET, y, _pick(zh, "贡献者", "Contributors"), colors["ink"], 13.5, 700)
    _text(out, WIDTH - INSET, y, _pick(zh, "按提交数排序", "by commits"), colors["muted"], 11.5, anchor="end")
    y += 14
    card_w, card_h, gap = 168, 64, 12
    per_row = int((WIDTH - 2 * INSET + gap) // (card_w + gap))
    for i, person in enumerate(data["contributors"]):
        x = INSET + (i % per_row) * (card_w + gap)
        row_y = y + (i // per_row) * (card_h + gap)
        out.append(f'<g><title>{esc(person["name"])}</title>')
        _tile(out, x, row_y, card_w, card_h, colors, accent=False)
        _avatar(out, x + 12, row_y + 12, 20, person, colors, f"c{i}", ring=i == 0)
        _text(out, x + 62, row_y + 28, person["name"][:15], colors["ink"], 12.5, 650)
        _text(out, x + 62, row_y + 45, _pick(zh, f"{person['commits']} 次提交", f"{person['commits']} commits"), colors["muted"], 11)
        out.append("</g>")
    y += max(1, -(-len(data["contributors"]) // per_row)) * (card_h + gap)
    if data["ai"]:
        x = INSET
        label = _pick(zh, "一起写的 AI：", "Written with AI:")
        _text(out, x, y + 13, label, colors["muted"], 12)
        x += _width(label, 12) + 10
        for name in data["ai"]:
            w = _width(name, 12) + 24
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" rx="11" fill="none" stroke="{colors["tile_rule"]}"/>')
            _text(out, x + w / 2, y + 15, name, colors["ink"], 12, 600, "middle")
            x += w + 8
        y += 34
    y += 22
    _text(out, INSET, y, _pick(zh, "审稿人", "Reviewers"), colors["ink"], 13.5, 700)
    _text(out, WIDTH - INSET, y, _pick(zh, "每个 PR 都要一位非作者的领域审稿人通过", "every PR needs a non-author reviewer for its area"),
          colors["muted"], 11.5, anchor="end")
    y += 14
    card_w = (WIDTH - 2 * INSET - 2 * gap) / 3
    for i, person in enumerate(data["reviewers"]):
        x = INSET + (i % 3) * (card_w + gap)
        row_y = y + (i // 3) * 82
        names = [_pick(zh, a, b) for a, b in person["areas"]]
        out.append(f'<g><title>{esc(", ".join(names))}</title>')
        _tile(out, x, row_y, card_w, 70, colors, accent=True)
        _avatar(out, x + 16, row_y + 13, 22, {**person, "name": person["login"]}, colors, f"r{i}")
        _text(out, x + 70, row_y + 30, person["login"], colors["ink"], 13, 650)
        detail = _pick(zh, f"{len(names)} 个领域", f"{len(names)} areas")
        if person.get("reviews"):
            detail += _pick(zh, f" · 审过 {person['reviews']} 个 PR", f" · {person['reviews']} PRs reviewed")
        _text(out, x + 70, row_y + 49, detail, colors["muted"], 11.5)
        out.append("</g>")
    y += max(1, -(-len(data["reviewers"]) // 3)) * 82
    if data.get("next"):
        y += 22
        _text(out, INSET, y, _pick(zh, "读者的下一站", "Where readers went next"), colors["ink"], 13.5, 700)
        _text(out, WIDTH - INSET, y, _pick(zh, "读者自愿分享，不代表笔记促成了这些机会", "self-reported and voluntary; not a claim that the notes got anyone a job"),
              colors["muted"], 11.5, anchor="end")
        y += 14
        x = INSET
        for name, count, logo in data["next"]:
            label = _pick(zh, f"{name} · {count}", f"{name} · {count}")
            w = _width(label, 12.5) + (52 if logo else 28)
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="34" rx="17" fill="{colors["tile"]}" stroke="{colors["tile_rule"]}"/>')
            tx = x + 14
            if logo:
                out.append(f'<image href="{logo}" x="{x + 10}" y="{y + 8}" width="18" height="18"/>')
                tx = x + 36
            _text(out, tx, y + 22, label, colors["ink"], 12.5, 600)
            x += w + 10
        y += 48
    y += 14
    _text(out, INSET, y, _pick(zh, f"{built} 构建 · 来自 git 记录、GitHub 与 site/collaboration.toml，随网站每次部署更新",
                               f"Built {built} · from git history, GitHub and site/collaboration.toml, updated on every deploy"),
          colors["muted"], 11)
    title = _pick(zh, "贡献者与审稿人", "Contributors and reviewers")
    desc = _pick(zh, f"{len(data['contributors'])} 位贡献者，{len(data['reviewers'])} 位审稿人。",
                 f"{len(data['contributors'])} contributors and {len(data['reviewers'])} reviewers.")
    return _panel(WIDTH, y + INSET - 6, colors, title, desc, out, glow=(0.9, 0.0))
