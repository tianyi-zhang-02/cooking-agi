import html
import os
import tomllib


def localized(record, field, language):
    return record.get(f"{field}_{language}", "")


def note_link(page, source, catalog, label=None):
    pair = catalog.get(source)
    if not pair or not pair.get(page.lang):
        raise ValueError(f"Missing curriculum target: {source} ({page.lang})")
    target = pair[page.lang]
    destination = os.path.relpath(target.src, page.src.parent).replace(os.sep, '/')
    return (f'<a href="{html.escape(destination, quote=True)}">'
            f'{html.escape(label or target.title)}</a>')


def study_atlas(page, navigation, sections, catalog):
    source = page.src.relative_to(page.src.parents[page.depth]).as_posix()
    source = source.replace('.en.md', '.md')
    selections = {
        '00-foundations/deep-dives/README.md': {'pretraining', 'inference'},
        'learn/pretraining/README.md': {'pretraining'},
        'learn/inference/README.md': {'inference'},
    }
    selected = selections.get(source)
    chinese = page.lang == 'zh'
    labels = navigation.get('label', {})
    groups = [group for group in navigation['group']
              if group.get('category') == 'learn' and group['id'] != 'study'
              and (selected is None or group['id'] in selected)]
    cards = {}
    heading_level = 'h3' if selected else 'h4'
    for position, group in enumerate(groups, 1):
        chapters = [section for section in sections if section['group'] == group['id']]
        count = sum(len(section['pages']) for section in chapters)
        if not count:
            continue
        title = group['zh' if chinese else 'en']
        heading = note_link(page, group['home'], catalog, title)
        description = localized(group, 'summary', page.lang)
        rows = []
        for chapter in chapters:
            entries = []
            for pair in chapter['pages']:
                target = pair.get(page.lang)
                if not target:
                    raise ValueError(f"Missing translation in {chapter['dir']}")
                key = target.src.relative_to(target.src.parents[target.depth]).as_posix().replace('.en.md', '.md')
                short = labels.get(key, [target.title, target.title])[0 if chinese else 1]
                entries.append(f'<li>{note_link(page, key, catalog, short)}</li>')
            chapter_title = html.escape(chapter['zh' if chinese else 'en'])
            intro = html.escape(localized(chapter, 'intro', page.lang))
            explanation = f'<p class="atlas-chapter-intro">{intro}</p>' if intro else ''
            expanded = ' open' if selected is not None else ''
            rows.append(f'<details class="atlas-chapter"{expanded}>'
                        f'<summary><span>{chapter_title}</span><small>{len(entries)}</small></summary>'
                        f'{explanation}<ol>{"".join(entries)}</ol></details>')
        count_label = f'{count} 篇' if chinese else f'{count} notes'
        card = (f'<section class="atlas-subject" id="subject-{html.escape(group["id"])}">'
                     f'<header><span class="atlas-number">{position:02}</span>'
                     f'<{heading_level}>{heading}</{heading_level}><span class="atlas-count">{count_label}</span></header>'
                     f'<p class="atlas-description">{html.escape(description)}</p>'
                     f'<div class="atlas-chapters">{"".join(rows)}</div></section>')
        cards.setdefault(group.get('zone', ''), []).append(card)
    label = '完整章节目录' if chinese else 'Complete chapter catalog'
    if selected:
        return f'<div class="study-atlas" aria-label="{label}">{"".join(card for area in cards.values() for card in area)}</div>'
    areas = []
    zones = {zone['id']: zone for zone in navigation.get('zone', [])}
    for zone_id, subjects in cards.items():
        if zone_id not in zones:
            raise ValueError(f'Missing study area: {zone_id}')
        zone = zones[zone_id]
        title = html.escape(zone['zh' if chinese else 'en'])
        description = html.escape(localized(zone, 'summary', page.lang))
        anchor = f'area-{html.escape(zone_id, quote=True)}'
        areas.append(f'<section class="atlas-area" aria-labelledby="{anchor}">'
                     f'<header><h3 id="{anchor}">{title}</h3><p>{description}</p></header>'
                     f'<div class="study-atlas">{"".join(subjects)}</div></section>')
    return f'<div class="study-catalog" aria-label="{label}">{"".join(areas)}</div>'


def chapter_context(page, catalog):
    section = page.section
    intro = localized(section, 'intro', page.lang)
    if not intro or page.kind in {'home', 'index'}:
        return ''
    chinese = page.lang == 'zh'
    label = '这篇放在哪里' if chinese else 'Where this fits'
    return (f'<aside class="chapter-context" aria-label="{label}">'
            f'<strong>{label}</strong><p>{html.escape(intro)}</p></aside>')


def validate_catalog(navigation, sections, catalog):
    claimed = [source for section in navigation['section'] for source in section.get('include', [])]
    if len(claimed) != len(set(claimed)):
        raise ValueError('A note is assigned to more than one chapter')
    for source in claimed:
        if source not in catalog:
            raise ValueError(f'Missing included note: {source}')
    for section in sections:
        if section.get('home') and section['home'] not in catalog:
            raise ValueError(f'Missing chapter overview: {section["home"]}')
    for group in navigation['group']:
        if group.get('home') and group['home'] not in catalog:
            raise ValueError(f'Missing subject overview: {group["home"]}')


def reference_coverage(page, catalog, source):
    data = tomllib.loads(source.read_text(encoding='utf-8'))
    chinese = page.lang == 'zh'
    states = {
        'article': '有正文' if chinese else 'Article available',
        'overview': '部分讲解' if chinese else 'Partial explanation',
        'pending': '待补' if chinese else 'To add',
    }
    blocks = []
    for chapter in data['chapter']:
        rows = []
        for topic in chapter['topics']:
            if topic['state'] not in states:
                raise ValueError('Unknown coverage state')
            links = [note_link(page, target, catalog) for target in topic['notes']]
            if topic.get('source'):
                if not topic['source'].startswith('https://arxiv.org/abs/'):
                    raise ValueError('Unexpected primary-source URL')
                label = '原论文' if chinese else 'Paper'
                links.append(f'<a href="{html.escape(topic["source"], quote=True)}">{label}</a>')
            destination = '<br>'.join(links) if links else ('尚无独立正文' if chinese else 'No standalone article yet')
            rows.append(f'<tr><td>{html.escape(topic["zh" if chinese else "en"])}</td>'
                        f'<td>{states[topic["state"]]}</td><td>{destination}</td></tr>')
        headings = ['知识点', '本站内容', '入口'] if chinese else ['Topic', 'Site coverage', 'Read']
        blocks.append(f'<section class="coverage-chapter"><h3>{html.escape(chapter["zh" if chinese else "en"])}</h3>'
                      f'<table><thead><tr>{"".join(f"<th scope=\"col\">{label}</th>" for label in headings)}</tr></thead>'
                      f'<tbody>{"".join(rows)}</tbody></table></section>')
    return '<div class="reference-coverage">' + ''.join(blocks) + '</div>'


def appendix_coverage(page, catalog, source):
    data = tomllib.loads(source.read_text(encoding='utf-8'))
    chinese = page.lang == 'zh'
    statuses = {
        'article': ('已有专门讲解', 'Dedicated explanation'),
        'related': ('有相关内容，需补充或对照', 'Related material; expansion or comparison needed'),
        'missing': ('缺少专题讲解', 'Dedicated coverage missing'),
    }
    reading = {
        'body': ('参考正文已读', 'Reference prose read'),
        'title': ('只确认标题，正文未核对', 'Title identified; source prose unchecked'),
    }
    identifiers = set()
    blocks = []
    counts = dict.fromkeys(statuses, 0)
    for chapter in data['chapter']:
        entries = []
        for topic in chapter['topics']:
            identifier = topic['id']
            if identifier in identifiers:
                raise ValueError(f'Duplicate appendix topic: {identifier}')
            identifiers.add(identifier)
            if topic['state'] not in statuses or topic['read'] not in reading:
                raise ValueError(f'Unknown appendix review state: {identifier}')
            counts[topic['state']] += 1
            title = html.escape(topic[page.lang])
            status = statuses[topic['state']][not chinese]
            read = reading[topic['read']][not chinese]
            gap = html.escape(localized(topic, 'gap', page.lang))
            if not gap:
                raise ValueError(f'Missing appendix explanation: {identifier}')
            links = ' · '.join(note_link(page, target, catalog) for target in topic['notes'])
            destination = links or ('暂无本站入口' if chinese else 'No site article yet')
            entries.append(f'<li id="coverage-{html.escape(identifier, quote=True)}">'
                           f'<strong>{title}</strong><span class="coverage-status">{status}</span>'
                           f'<p>{gap}</p><p>{destination}</p>'
                           f'<small>{read}</small></li>')
        title = html.escape(chapter[page.lang])
        blocks.append(f'<details class="appendix-group"><summary>{title} · {len(entries)}</summary>'
                      f'<ul class="appendix-topics">{"".join(entries)}</ul></details>')
    if chinese:
        summary = (f'本次登记 {len(identifiers)} 个入口：{counts["article"]} 项已有专门讲解，'
                   f'{counts["related"]} 项需补充或对照，{counts["missing"]} 项缺少专题讲解。'
                   '这是当前盘点，不是全库完成率；每项另列参考阅读状态。')
    else:
        summary = (f'This inventory tracks {len(identifiers)} entries: {counts["article"]} with dedicated explanations, '
                   f'{counts["related"]} needing expansion or comparison, and {counts["missing"]} lacking dedicated coverage. '
                   'These are inventory counts, not a library completion rate; source-reading status is recorded separately.')
    return '<div class="appendix-coverage"><p>' + summary + '</p>' + ''.join(blocks) + '</div>'
