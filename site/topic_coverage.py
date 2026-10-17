"""Topic coverage: how many notes each topic has, and how many the current review pass has read.

Built from the navigation on every build, so it changes as notes are added or reviewed. The site
shows it inline (`<!-- widget:topic-coverage -->` in the README); GitHub can only show images, so
the build also writes light and dark SVGs that the README embeds from the deployed site."""
import html
import tomllib

CATEGORIES = ("learn", "interview", "practice")
SKIP_GROUPS = {"study", "practice"}          # navigation pages, not topics
# Areas we would like contributors to start (community/roadmap.md#future); none has notes yet.
# Vision, driving and robotics now have draft notes in 13-embodied/.
PROPOSED = (("Jetson 与边缘部署", "Jetson & edge deployment"), ("物理仿真", "Physical simulation"))
PALETTES = {
    "light": {"ink": "#1f2328", "muted": "#59636e", "bar": "#98512b", "rule": "#d1d9e0"},
    "dark": {"ink": "#e6edf3", "muted": "#9198a1", "bar": "#e8a672", "rule": "#3d444d"},
}
PENDING_OPACITY = 0.3


def load_reviewed(path, by_src):
    """Chinese source paths whose two languages were read and fixed in this pass."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    reviewed = data.get("reviewed", [])
    unknown = [source for source in reviewed if source not in by_src]
    if unknown:
        raise SystemExit(f"{path.name}: not a note: {', '.join(unknown)}")
    if len(set(reviewed)) != len(reviewed):
        raise SystemExit(f"{path.name}: a note is listed twice")
    return data.get("pass", ""), set(reviewed)


def rows(nav, sections, root, reviewed):
    """[(category label pair, [(topic label pair, notes, reviewed)])]; section home pages don't count."""
    counts = {}
    for section in sections:
        for pair in section["pages"]:
            page = pair.get("zh") or pair.get("en")
            if page.kind in {"index", "home"}:
                continue
            notes, done = counts.get(section["group"], (0, 0))
            counts[section["group"]] = (notes + 1, done + (page.src.relative_to(root).as_posix() in reviewed))
    result = []
    for category in (c for c in nav["category"] if c["id"] in CATEGORIES):
        topics = [((group["zh"], group["en"]), *counts[group["id"]]) for group in nav["group"]
                  if group.get("category") == category["id"] and group["id"] not in SKIP_GROUPS
                  and counts.get(group["id"], (0, 0))[0]]
        result.append(((category["zh"], category["en"]), topics))
    result.append((("欢迎共建", "Open to contributors"), [(labels, 0, 0) for labels in PROPOSED]))
    return result


def totals(data):
    topics = [topic for _, group in data for topic in group]
    return sum(t[1] for t in topics), sum(t[2] for t in topics)


def _text(zh, chinese, english):
    return chinese if zh else english


def _width(text, size):
    """Rough rendered width: CJK glyphs are square, Latin ones about half as wide."""
    return sum(size if ord(char) > 0x2E80 else size * 0.56 for char in text)


def svg(data, lang, palette, built):
    """A standalone SVG for the README (GitHub shows images, not scripts or site CSS)."""
    zh = lang == "zh"
    colors = PALETTES[palette]
    esc = lambda value: html.escape(str(value), quote=True)
    font = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', 'PingFang SC', "
            "'Hiragino Sans GB', 'Microsoft YaHei', sans-serif")
    width, pad, row, size = 720, 16, 24, 13
    label = max(_width(labels[0 if zh else 1], size) for _, group in data for labels, *_ in group) + 20
    bar_x, bar_w = pad + label, width - pad - label - 76
    most = max(notes for _, group in data for _, notes, _ in group)
    notes_total, done_total = totals(data)
    out, y = [], pad + 14

    def text(x, y, value, fill, anchor="start", weight=400, small=False):
        out.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{11.5 if small else size}" '
                   f'font-weight="{weight}" text-anchor="{anchor}">{esc(value)}</text>')

    # legend and totals
    out.append(f'<rect x="{pad}" y="{y - 9}" width="10" height="10" rx="2" fill="{colors["bar"]}"/>')
    text(pad + 15, y, _text(zh, "本轮已复核", "Reviewed this pass"), colors["muted"], small=True)
    shift = pad + 15 + _width(_text(zh, "本轮已复核", "Reviewed this pass"), 11.5) + 18
    out.append(f'<rect x="{shift:.1f}" y="{y - 9}" width="10" height="10" rx="2" fill="{colors["bar"]}" fill-opacity="{PENDING_OPACITY}"/>')
    text(shift + 15, y, _text(zh, "待复核", "Not yet reviewed"), colors["muted"], small=True)
    shift += 15 + _width(_text(zh, "待复核", "Not yet reviewed"), 11.5) + 18
    out.append(f'<rect x="{shift:.1f}" y="{y - 9}" width="10" height="10" rx="2" fill="none" '
               f'stroke="{colors["muted"]}" stroke-dasharray="2 2"/>')
    text(shift + 15, y, _text(zh, "还没有笔记", "No notes yet"), colors["muted"], small=True)
    text(width - pad, y, _text(zh, f"已复核 {done_total} / 共 {notes_total} 篇", f"{done_total} of {notes_total} notes reviewed"),
         colors["ink"], anchor="end", weight=600, small=True)
    y += 14
    for (category, topics) in data:
        y += 34
        out.append(f'<line x1="{pad}" x2="{width - pad}" y1="{y - 19}" y2="{y - 19}" stroke="{colors["rule"]}"/>')
        text(pad, y, category[0 if zh else 1], colors["ink"], weight=600)
        y += 6
        for labels, notes, done in topics:
            y += row
            name = labels[0 if zh else 1]
            tip = (_text(zh, f"{name}：{notes} 篇，本轮已复核 {done} 篇", f"{name}: {notes} notes, {done} reviewed this pass")
                   if notes else _text(zh, f"{name}：还没有笔记，欢迎共建", f"{name}: no notes yet, open to contributors"))
            out.append(f'<g><title>{esc(tip)}</title>')
            text(pad, y, name, colors["ink"])
            top = y - 9
            if notes:
                full = bar_w * notes / most
                out.append(f'<rect x="{bar_x:.1f}" y="{top}" width="{full:.1f}" height="10" rx="2" fill="{colors["bar"]}" fill-opacity="{PENDING_OPACITY}"/>')
                if done:
                    out.append(f'<rect x="{bar_x:.1f}" y="{top}" width="{bar_w * done / most:.1f}" height="10" rx="2" fill="{colors["bar"]}"/>')
                text(width - pad, y, f"{done} / {notes}", colors["muted"], anchor="end", small=True)
            else:
                out.append(f'<rect x="{bar_x:.1f}" y="{top}" width="40" height="10" rx="2" fill="none" '
                           f'stroke="{colors["muted"]}" stroke-dasharray="2 2"/>')
                text(width - pad, y, _text(zh, "欢迎共建", "Help wanted"), colors["muted"], anchor="end", small=True)
            out.append('</g>')
    y += 30
    text(pad, y, _text(zh, f"{built} 构建 · 随网站每次部署更新", f"Built {built} · updated on every site deploy"), colors["muted"], small=True)
    height = y + pad
    title = _text(zh, "主题覆盖", "Topic coverage")
    desc = _text(zh, f"各主题的笔记数和本轮已复核的篇数：共 {notes_total} 篇，已复核 {done_total} 篇。",
                 f"Notes per topic and how many this review pass has covered: {done_total} of {notes_total} reviewed.")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-labelledby="t d" font-family="{esc(font)}">'
            f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>' + "".join(out) + '</svg>\n')


def figure(data, lang, link):
    """The same chart as HTML for the site: real text, theme colours, and it reflows on a phone."""
    zh = lang == "zh"
    esc = lambda value: html.escape(str(value), quote=True)
    most = max(notes for _, group in data for _, notes, _ in group)
    notes_total, done_total = totals(data)
    out = ['<figure class="topic-coverage">',
           '<div class="tc-head"><div class="tc-legend" aria-hidden="true">'
           f'<span><i class="tc-done"></i>{_text(zh, "本轮已复核", "Reviewed this pass")}</span>'
           f'<span><i class="tc-todo"></i>{_text(zh, "待复核", "Not yet reviewed")}</span>'
           f'<span><i class="tc-none"></i>{_text(zh, "还没有笔记", "No notes yet")}</span></div>'
           f'<strong>{_text(zh, f"已复核 {done_total} / 共 {notes_total} 篇", f"{done_total} of {notes_total} notes reviewed")}</strong></div>']
    for category, topics in data:
        out.append(f'<p class="tc-category">{esc(category[0 if zh else 1])}</p><ul class="tc-rows">')
        for labels, notes, done in topics:
            name = esc(labels[0 if zh else 1])
            if notes:
                bar = (f'<span class="tc-bar" aria-hidden="true"><i class="tc-todo" style="width:{100 * notes / most:.1f}%"></i>'
                       f'<i class="tc-done" style="width:{100 * done / most:.1f}%"></i></span>')
                value = f'{done} / {notes}'
                label = _text(zh, f"{notes} 篇，已复核 {done} 篇", f"{notes} notes, {done} reviewed")
            else:
                bar = '<span class="tc-bar" aria-hidden="true"><i class="tc-none"></i></span>'
                value = _text(zh, "欢迎共建", "Help wanted")
                label = _text(zh, "还没有笔记", "no notes yet")
            out.append(f'<li><span class="tc-name">{name}</span>{bar}'
                       f'<span class="tc-value" aria-label="{esc(label)}">{value}</span></li>')
        out.append('</ul>')
    out.append(f'<figcaption><a href="{esc(link)}">{_text(zh, "逐项查漏：内容覆盖与待办", "Item by item: coverage and gaps")}</a></figcaption></figure>')
    return "".join(out)
