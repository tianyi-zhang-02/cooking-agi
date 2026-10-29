"""README cards: an overview of what is inside (with the roadmap's now and next), and the people behind
the notes.

GitHub shows images, not scripts or site CSS, so the build writes standalone SVGs (light and dark,
both languages) into the deployed site and the README embeds them from there. The site rebuilds on
every push to main and once a day, so the cards follow the notes, site/milestones.toml and GitHub
without anyone editing the README.

Both cards are kept plain on purpose: one column, sentence-case sans type, bars instead of nested
tiles, and the site's night sky behind them.

People data comes from the GitHub API when it is reachable (CI has a token); avatars are embedded
as data URIs because an SVG shown as an image cannot load anything else. Offline, initials stand in.
"""
import base64
import html
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


def _rule(out, y, colors):
    out.append(f'<line x1="{INSET}" x2="{WIDTH - INSET}" y1="{y:.1f}" y2="{y:.1f}" stroke="{colors["rule"]}"/>')


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


def overview_svg(data, milestones, lang, palette, built, signals):
    """One card for the README's "What's inside": what the notes are checked and illustrated with
    (signals: {"reviewed", "figures", "paper"}; quality, not volume), one bar per area with its
    topics underneath, and the roadmap's now and next. Drafts are counted only while there are some."""
    zh, colors = lang == "zh", PALETTES[palette]
    out, inner = [], WIDTH - 2 * INSET
    stats = [(str(signals["reviewed"]), _pick(zh, "篇中英文逐篇复核", "notes re-read in both languages")),
             (str(signals["figures"]), _pick(zh, "张交互与分步图解", "interactive figures")),
             (str(signals["paper"]), _pick(zh, "张论文原图", "figures from papers"))]
    y = _stats(out, INSET, INSET + 30, stats, colors, inner) + 8
    _rule(out, y, colors)
    y += 34
    # areas: a name, a bar, a count; the topics inside in a muted line below
    rows = []
    for zone_zh, zone_en, groups in data["zones"]:
        label = _pick(zh, zone_zh, zone_en).partition(" · ")
        rows.append((label[2] or label[0], sum(n for *_, n, _ in groups), sum(d for *_, d in groups),
                     " · ".join(_pick(zh, g_zh, g_en) for g_zh, g_en, *_ in groups)))
    rows += [(_pick(zh, name_zh, name_en), notes, 0, "") for name_zh, name_en, notes in data["others"]]
    top = max([n for _, n, _, _ in rows] + [1])
    bar_x, count_x = INSET + 196, WIDTH - INSET
    bar_w = count_x - 44 - bar_x
    for name, notes, drafts, topics in rows:
        _text(out, INSET, y, name, colors["ink"], 14, 650)
        out.append(f'<rect x="{bar_x}" y="{y - 8.5:.1f}" width="{bar_w}" height="5" rx="2.5" fill="{colors["rule"]}"/>'
                   f'<rect x="{bar_x}" y="{y - 8.5:.1f}" width="{max(5, bar_w * notes / top):.1f}" height="5" rx="2.5" fill="{colors["accent"]}"/>')
        _text(out, count_x, y, _pick(zh, f"{notes} 篇", str(notes)), colors["ink"], 14, 700, "end")
        y += 19
        detail = topics + (_pick(zh, f"（{drafts} 篇草稿待审核）", f" ({drafts} in review)") if drafts else "")
        for line in wrap(detail, inner, 12):
            _text(out, INSET, y, line, colors["muted"], 12)
            y += 16
        y += 14
    # the roadmap's now and next
    y += 2
    _rule(out, y - 18, colors)
    y += 10
    for key, label_zh, label_en in (("now", "现在", "Now"), ("next", "接下来", "Next")):
        items = [_pick(zh, item["zh"], item["en"]) for item in milestones.get(key, [])]
        if not items:
            continue
        _text(out, INSET, y, _pick(zh, label_zh, label_en), colors["accent"], 12.5, 700)
        for item in items:  # one per line; labels in site/milestones.toml are kept short
            for line in wrap(item, inner - 72, 13):
                _text(out, INSET + 72, y, line, colors["ink"], 13)
                y += 18
        y += 8
    y += 10
    _text(out, INSET, y, _pick(zh, f"{built} 构建 · 随网站每次部署更新", f"Built {built} · updated on every deploy"), colors["muted"], 11.5)
    notes_total = sum(n for _, n, _, _ in rows)
    title = _pick(zh, "内容一览", "What's inside")
    desc = _pick(zh, f"按板块列出的主题和篇数，共 {notes_total} 篇；以及现在和接下来在做的事。",
                 f"Topics by area with note counts, {notes_total} notes in all, and what is being worked on now and next.")
    return _panel(WIDTH, y + INSET + 6, colors, title, desc, out, glow=(0.05, 0.0))


def load_milestones(path):
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


# ---------------------------------------------------------------- the people card
# People are drawn as 1-bit discs (their avatar through a dither filter: grey + noise + threshold, so the
# build needs no image library), on the same night sky as the overview.

DITHER = ('<defs><filter id="dither" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">'
          '<feColorMatrix type="matrix" values=".3 .59 .11 0 0 .3 .59 .11 0 0 .3 .59 .11 0 0 0 0 0 1 0" result="g"/>'
          '<feTurbulence type="fractalNoise" baseFrequency="1.9" numOctaves="1" seed="5" result="n"/>'
          '<feColorMatrix in="n" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="ng"/>'
          '<feComposite in="g" in2="ng" operator="arithmetic" k2="1" k3="1.15" k4="-.575" result="d"/>'
          '<feComponentTransfer in="d"><feFuncR type="discrete" tableValues="0 1"/><feFuncG type="discrete" tableValues="0 1"/>'
          '<feFuncB type="discrete" tableValues="0 1"/></feComponentTransfer><feComposite in2="SourceAlpha" operator="in"/></filter>'
          '<radialGradient id="shade" cx=".38" cy=".32" r=".75"><stop offset="0" stop-color="#fff"/><stop offset=".7" stop-color="#cfcfcf"/>'
          '<stop offset="1" stop-color="#7a7a7a"/></radialGradient></defs>')


def _disc(out, cx, cy, r, member, key):
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
            _text(out, cx, cy + r * .35, (member.get("login") or member["name"])[:1].upper(), "#111", round(r * .9), 700, "middle")
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="none" stroke="#000"/>')


def people_svg(data, milestones, lang, palette, built):
    """Everyone with a commit or co-author credit in one row (wrapping), then the reviewers, where readers
    went next (marks only, never counts), and what still needs people (site/milestones.toml "open")."""
    zh, colors = lang == "zh", PALETTES[palette]
    out, inner = [DITHER], WIDTH - 2 * INSET
    crew = data["crew"]
    per_row = min(max(1, len(crew)), 5)
    step = inner / per_row
    y = INSET + 20
    for row in range(0, len(crew), per_row):
        for k, member in enumerate(crew[row:row + per_row]):
            cx = INSET + step * (k + .5)
            out.append(f'<g><title>{esc(member["handle"])}</title>')
            _disc(out, cx, y + 24, 24, member, f"c{row + k}")
            out.append("</g>")
            _text(out, cx, y + 70, member["handle"], colors["ink"], 13, 650, "middle")
            commits = member["commits"]
            meta = (_pick(zh, f"{commits} 次提交", f"{commits} commit" + ("s" if commits != 1 else "")) if commits else "")
            if member["kind"] == "ai":
                meta = _pick(zh, "AI 协作者", "AI collaborator") + (f" · {commits}" if commits else "")
            _text(out, cx, y + 88, meta, colors["muted"], 11.5, anchor="middle")
        y += 112
    _rule(out, y, colors)
    y += 34
    label_w = 132

    def line_label(text):
        _text(out, INSET, y, text, colors["muted"], 12.5)

    if data["reviewers"]:
        line_label(_pick(zh, "审稿人", "Reviewers"))
        x = INSET + label_w
        for i, member in enumerate(data["reviewers"]):
            _disc(out, x + 10, y - 5, 10, member, f"r{i}")
            meta = _pick(zh, f' · {len(member["areas"])} 个领域', f' · {len(member["areas"])} areas')
            if member.get("reviews"):
                meta += _pick(zh, f' · 审过 {member["reviews"]} 个 PR', f' · {member["reviews"]} PRs reviewed')
            _text(out, x + 26, y, member["handle"], colors["ink"], 13, 650)
            _text(out, x + 26 + _width(member["handle"], 13) + 2, y, meta, colors["muted"], 12.5)
            x += 26 + _width(member["handle"] + meta, 13) + 22
        y += 30
    if data.get("next"):
        line_label(_pick(zh, "读者的下一站", "Readers went on to"))
        x = INSET + label_w
        for company in data["next"]:
            out.append(f'<g><title>{esc(company["name"])}</title>')
            if company["logo"]:
                mark = re.sub(r'\sfill="[^"]*"', "", company["logo"]).replace("<svg ", f'<svg fill="{colors["ink"]}" ', 1)
                uri = "data:image/svg+xml;base64," + base64.b64encode(mark.encode("utf-8")).decode("ascii")
                out.append(f'<image href="{uri}" x="{x:.1f}" y="{y - 15}" width="19" height="19"/>')
                x += 19 + 16
            else:
                _text(out, x, y, company["name"], colors["ink"], 13, 650)
                x += _width(company["name"], 13) + 16
            out.append("</g>")
        _text(out, x + 2, y, _pick(zh, "读者自愿分享", "self-reported"), colors["muted"], 11.5)
        y += 30
    wanted = [_pick(zh, item["zh"], item["en"]) for item in milestones.get("open", [])]
    if wanted:
        line_label(_pick(zh, "欢迎参与", "Help wanted"))
        for line in wrap(" · ".join(wanted), inner - label_w, 13):
            _text(out, INSET + label_w, y, line, colors["ink"], 13)
            y += 18
        y += 12
    y += 8
    _text(out, INSET, y, _pick(zh, f"{built} 更新 · 提交数不是质量排名，也不决定审核权限",
                               f"Updated {built} · commit counts are not a ranking of quality or review rights"), colors["muted"], 11.5)
    humans = sum(m["kind"] == "human" for m in crew)
    title = _pick(zh, "贡献者与审稿人", "Contributors and reviewers")
    desc = _pick(zh, f"{humans} 位贡献者、{len(crew) - humans} 个 AI 协作者、{len(data['reviewers'])} 位审稿人。",
                 f"{humans} contributors, {len(crew) - humans} AI collaborators and {len(data['reviewers'])} reviewers.")
    return _panel(WIDTH, y + INSET + 6, colors, title, desc, out, glow=(0.95, 0.0))
