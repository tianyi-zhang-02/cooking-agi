import html
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


DATA = Path(__file__).parent / "static/technology-tree-data.json"
MARKER = '<div data-ai-chronicle></div>'


def validate(data):
    def require(condition, message):
        if not condition:
            raise ValueError(f"Technology chronicle: {message}")

    def identifier(value):
        return isinstance(value, str) and re.fullmatch(r"[a-z0-9-]+", value)

    def translated(value):
        return isinstance(value, dict) and set(value) == {"zh", "en"} and all(
            isinstance(text, str) and text.strip() for text in value.values())

    require(data.get("version") == 3, "unsupported data version")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", data.get("reviewed", "")), "missing review date")
    require(isinstance(data.get("nodes"), list) and data["nodes"], "missing works")
    require(isinstance(data.get("eras"), list) and data["eras"], "missing eras")
    require(isinstance(data.get("layers"), list) and data["layers"], "missing layers")
    require(isinstance(data.get("topics"), list) and data["topics"], "missing topics")
    layers, topics = set(), set()
    for layer in data["layers"]:
        require(identifier(layer.get("id")) and layer["id"] not in layers, "invalid layer")
        require(translated(layer.get("title")), "missing layer translation")
        layers.add(layer["id"])
    for topic in data["topics"]:
        require(identifier(topic.get("id")) and topic["id"] not in topics, "invalid topic")
        require(topic.get("layer") in layers, "unknown layer")
        require(all(translated(topic.get(field)) for field in ("title", "subtitle", "description", "next")), "missing topic translation")
        topics.add(topic["id"])
    nodes = {}
    for node in data["nodes"]:
        name = node.get("id")
        require(identifier(name) and name not in nodes, "duplicate or invalid work ID")
        require(isinstance(node.get("year"), int) and 1800 <= node["year"] <= int(data["reviewed"][:4]), f"invalid year: {name}")
        require(node.get("topic") in topics, f"unknown topic: {name}")
        require(node.get("kind") in {"research", "system", "product", "open"}, f"invalid kind: {name}")
        require(node.get("status") in {"historical", "adopted", "evolving"}, f"invalid status: {name}")
        require(all(translated(node.get(field)) for field in ("title", "body", "limit")), f"missing translation: {name}")
        require("detail" not in node or translated(node["detail"]), f"missing detail translation: {name}")
        require("hook" not in node or translated(node["hook"]), f"missing hook translation: {name}")
        require("detail" not in node or len(node["detail"]["zh"].split("\n\n")) == len(node["detail"]["en"].split("\n\n")),
                f"detail paragraphs differ between languages: {name}")
        require(isinstance(node.get("sources"), list) and node["sources"], f"missing sources: {name}")
        for source in node["sources"]:
            require(isinstance(source, list) and len(source) == 2 and all(isinstance(part, str) and part.strip() for part in source), f"invalid source: {name}")
            url = urlsplit(source[1])
            require(url.scheme == "https" and url.hostname and not url.username and not url.password, f"unsafe source: {name}")
        nodes[name] = node
    previous_end = 0
    era_ids, branch_ids, used = set(), set(), []
    for era in data["eras"]:
        name = era.get("id")
        require(identifier(name) and name not in era_ids, "duplicate or invalid era ID")
        era_ids.add(name)
        require(isinstance(era.get("start"), int) and isinstance(era.get("end"), int) and previous_end < era["start"] <= era["end"], f"overlapping or unordered era: {name}")
        previous_end = era["end"]
        require(all(translated(era.get(field)) for field in ("title", "summary")), f"missing era translation: {name}")
        require(isinstance(era.get("branches"), list) and era["branches"], f"missing branches: {name}")
        era_works = []
        for branch in era["branches"]:
            branch_id = f'{name}-{branch.get("id")}'
            require(identifier(branch.get("id")) and branch_id not in branch_ids, f"invalid branch: {name}")
            branch_ids.add(branch_id)
            require(all(translated(branch.get(field)) for field in ("title", "summary")), f"missing branch translation: {branch_id}")
            require(isinstance(branch.get("works"), list) and branch["works"], f"empty branch: {branch_id}")
            years = []
            for work in branch["works"]:
                require(work in nodes, f"unknown work: {work}")
                year = nodes[work]["year"]
                require(era["start"] <= year <= era["end"], f"work outside era: {work}")
                years.append(year)
            require(years == sorted(years), f"unordered branch: {branch_id}")
            era_works.extend(branch["works"])
        require(isinstance(era.get("featured"), list) and 1 <= len(era["featured"]) <= 3, f"use 1–3 featured works: {name}")
        require(len(set(era["featured"])) == len(era["featured"]) and set(era["featured"]).issubset(era_works), f"invalid featured works: {name}")
        used.extend(era_works)
    require(len(used) == len(set(used)) and set(used) == set(nodes), "each work needs one canonical branch")
    require(set(node["topic"] for node in nodes.values()) == topics, "each topic needs at least one work")
    for topic in data["topics"]:
        tracks = topic.get("tracks", [])
        require(isinstance(tracks, list) and all(isinstance(track, dict) and identifier(track.get("id")) and translated(track.get("title")) for track in tracks)
                and len({track["id"] for track in tracks}) == len(tracks), f"invalid tracks: {topic['id']}")
        known = {track["id"] for track in tracks}
        for node in nodes.values():
            if node["topic"] == topic["id"]:
                require(node.get("track") in known if known else "track" not in node, f"unknown track: {node['id']}")
        if "root" in topic:
            root = topic["root"]
            require(isinstance(root, dict) and nodes.get(root.get("from"), {}).get("topic") == topic["id"], f"aerial root must start on its own field: {topic['id']}")
            require(translated(root.get("label")) and translated(root.get("note")), f"missing aerial root translation: {topic['id']}")
    connections = set()
    for connection in data.get("connections", []):
        endpoints = (connection.get("from"), connection.get("to"))
        require(all(endpoint in nodes for endpoint in endpoints) and endpoints[0] != endpoints[1], "invalid connection endpoints")
        require(endpoints not in connections, "duplicate connection")
        require(connection.get("type") in {"uses", "adapts", "complements"}, "invalid connection type")
        require(translated(connection.get("description")), "missing connection translation")
        require(connection.get("source") in [source[1] for source in nodes[endpoints[0]]["sources"]], "connection needs evidence from its source work")
        connections.add(endpoints)
    return nodes


LEGACY_BRANCHES = {"transformer": "language", "alexnet": "vision", "qlearning": "rl", "sac": "robotics", "zero": "systems"}


def split_title(title):
    for separator in (" · ", "：", ": "):
        if separator in title:
            name, rest = title.split(separator, 1)
            return name, rest
    return title, ""


def render(data, language):
    """A tree for atmosphere, a vertical timeline for the information.

    chronicle.js draws the tree on a background canvas; this markup carries no
    tree. Below the opening space: a sticky field filter, then the nine periods
    from top to bottom, each with a short header and its milestones on a spine.
    The full text of each milestone lives in a library that chronicle.js opens
    in a side panel; without JavaScript everything simply stacks in order.
    """
    nodes = validate(data)
    escape = html.escape
    chinese = language == "zh"
    say = lambda zh, en: zh if chinese else en
    text = lambda value: escape(value[language])
    topics = {topic["id"]: topic for topic in data["topics"]}
    kinds = {"research": say("研究", "Research"), "system": say("系统", "System"),
             "product": say("产品", "Product"), "open": say("开放问题", "Open question")}
    outgoing_label = {"uses": say("用到", "Uses"), "adapts": say("借用", "Adapts"), "complements": say("互补", "Complements")}
    incoming_label = {"uses": say("被用于", "Used by"), "adapts": say("被借用于", "Adapted by"), "complements": say("互补", "Complements")}
    connections = data.get("connections", [])
    canonical = [name for era in data["eras"] for branch in era["branches"] for name in branch["works"]]
    position = {name: index for index, name in enumerate(canonical)}
    era_of = {name: era["id"] for era in data["eras"] for branch in era["branches"] for name in branch["works"]}
    short = lambda node: escape(split_title(node["title"][language])[0])
    hook = lambda node: node.get("hook", {}).get(language) or split_title(node["title"][language])[1]

    first, last = data["eras"][0]["start"], data["eras"][-1]["end"]
    latest = data["eras"][-1]
    # most readers come for the last few years: one link straight there
    jump = (f'<a class="ch-jump" href="#era-{latest["id"]}">'
            f'{say(f"直接看最近：{latest["start"]} – {latest["end"]}", f"Jump to the latest: {latest["start"]} – {latest["end"]}")} ↓</a>')
    stage = (f'<div class="ch-stage" data-tree-stage data-first="{first}" data-last="{last}"><p class="ch-year" data-year aria-hidden="true"></p>'
             f'<p class="ch-hint">{say(f"{first} 年到今天，往下读。", f"From {first} to today. Scroll to read.")}{jump}</p></div>')
    ongoing = int(data["reviewed"][:4])

    count = lambda topic: sum(node["topic"] == topic for node in data["nodes"])
    chips = ''.join(f'<button type="button" class="ch-field" id="field-{topic["id"]}" data-layer="{topic["layer"]}" data-field="{topic["id"]}" '
                    f'aria-pressed="false" aria-controls="chronicle-content" disabled><i aria-hidden="true"></i>{text(topic["title"])}<small>{count(topic["id"])}</small></button>'
                    for layer in data["layers"] for topic in data["topics"] if topic["layer"] == layer["id"])
    bar = (f'<div class="ch-bar"><div class="ch-fields" role="group" aria-label="{say("按领域看", "Filter by field")}">'
           f'<button type="button" class="ch-field ch-field-all" data-filter-reset aria-pressed="true" disabled>{say("全部", "All")}</button>{chips}</div></div>')

    def item(node, key):
        name = node["id"]
        layer = topics[node["topic"]]["layer"]
        title = split_title(node["title"][language])[0]
        line = hook(node)
        line = f'<span class="ch-item-hook">{escape(line)}</span>' if line else ''
        return (f'<li data-work-topic="{node["topic"]}" data-layer="{layer}"{" data-key" if key else ""}><a class="ch-item" href="#event-{name}" data-node="{name}" data-topic="{node["topic"]}" data-layer="{layer}">'
                f'<span class="ch-item-top"><time>{node["year"]}</time><span class="ch-item-name">{escape(title)}</span></span>{line}</a></li>')

    periods = []
    for era in data["eras"]:
        works = sorted((nodes[name] for branch in era["branches"] for name in branch["works"]), key=lambda node: (node["year"], position[node["id"]]))
        # Three milestones show before the period is opened: the featured ones first, then the earliest others.
        keys = [name for name in era["featured"]]
        keys += [node["id"] for node in works if node["id"] not in keys][:max(0, 3 - len(keys))]
        twigs = ''.join(f'<span id="twig-{era["id"]}-{branch["id"]}"></span>' for branch in era["branches"])
        more = (f'<button type="button" class="ch-more" data-more hidden data-more-label="{escape(say(f"展开这个年代的全部 {len(works)} 个", f"Show all {len(works)} in this period"), quote=True)}" '
                f'data-less-label="{escape(say("收起", "Show fewer"), quote=True)}"></button>') if len(works) > len(keys) else ''
        periods.append(f'<section class="ch-era" id="era-{era["id"]}" data-era="{era["id"]}" aria-labelledby="era-{era["id"]}-title">'
                       f'<header class="ch-era-head">{twigs}<p class="ch-era-years">{era["start"]} – {era["end"]}'
                       f'{f"<span class=\"ch-era-live\">{say("还在发生", "Still unfolding")}</span>" if era["end"] >= ongoing else ""}</p>'
                       f'<h2 id="era-{era["id"]}-title">{text(era["title"])}</h2><p class="ch-era-summary">{text(era["summary"])}</p></header>'
                       f'<div class="ch-era-body"><ol class="ch-spine">{"".join(item(node, node["id"] in keys) for node in works)}</ol>{more}</div></section>')
    empty = f'<p class="ch-empty" data-filter-empty hidden>{say("这个领域还没有节点。", "No milestones in this field yet.")}</p>'

    track_names = {(topic["id"], track["id"]): track["title"][language] for topic in data["topics"] for track in topic.get("tracks", [])}

    def article(node):
        name = node["id"]
        topic = topics[node["topic"]]
        outgoing = [connection for connection in connections if connection["from"] == name]
        incoming = [connection for connection in connections if connection["to"] == name]
        groups = {}
        for connection in outgoing:
            groups.setdefault(outgoing_label[connection["type"]], []).append(nodes[connection["to"]])
        for connection in incoming:
            groups.setdefault(incoming_label[connection["type"]], []).append(nodes[connection["from"]])
        quick = ''.join(f'<span class="ch-link-group"><span class="ch-link-type">{label}</span>'
                        + ''.join(f'<a href="#event-{other["id"]}" data-node-link="{other["id"]}">{short(other)}</a>' for other in others) + '</span>'
                        for label, others in groups.items())
        quick = f'<p class="ch-links">{quick}</p>' if quick else ''
        detail = ''.join(f'<p class="chronicle-explanation">{escape(paragraph)}</p>'
                         for paragraph in node["detail"][language].split("\n\n")) if "detail" in node else ''
        evidence = ''.join(f'<li><span>{outgoing_label[connection["type"]]} · </span><a href="#event-{connection["to"]}">{short(nodes[connection["to"]])}</a>'
                           f'<p>{text(connection["description"])}</p></li>' for connection in outgoing)
        evidence = f'<ul class="chronicle-connections">{evidence}</ul>' if evidence else ''
        sources = ''.join(f'<li><a href="{escape(url, quote=True)}"><span>{escape(label)}</span><span aria-hidden="true">↗</span></a></li>' for label, url in node["sources"])
        status = f'<span class="ch-unsettled">{say("仍在探索", "Still evolving")}</span>' if node["status"] == "evolving" else ''
        alias = f'<span id="branch-{LEGACY_BRANCHES[name]}"></span>' if name in LEGACY_BRANCHES else ''
        title, phrase = split_title(node["title"][language])
        phrase = f'<span class="ch-sub">{escape(phrase)}</span>' if phrase else ''
        track = track_names.get((node["topic"], node.get("track")))
        track = f' · {escape(track)}' if track else ''
        return (f'<article class="chronicle-work" id="event-{name}" data-work-topic="{node["topic"]}" data-layer="{topic["layer"]}" data-era="{era_of[name]}" aria-labelledby="event-{name}-title">{alias}'
                f'<p class="ch-work-meta"><time>{node["year"]}</time><span class="ch-topic"><i aria-hidden="true"></i>{text(topic["title"])}{track}</span>'
                f'<span>{kinds[node["kind"]]}</span>{status}</p>'
                f'<h3 id="event-{name}-title" data-work-title><span class="ch-name">{escape(title)}</span>{phrase}</h3>'
                f'<p class="chronicle-lead">{text(node["body"])}</p>{quick}'
                f'<details class="chronicle-more"><summary>{say("细节", "Details")}</summary>'
                f'<div class="chronicle-more-body">{detail}<p class="ch-fold-label">{say("留下的问题", "Left open")}</p>'
                f'<p class="chronicle-limit">{text(node["limit"])}</p>{evidence}</div></details>'
                f'<details class="chronicle-references"><summary>{say("来源", "Sources")} <span class="chronicle-source-count">{len(node["sources"])}</span></summary>'
                f'<ol class="chronicle-sources">{sources}</ol></details></article>')

    library = (f'<div class="ch-library" data-library aria-label="{say("全部节点", "All milestones")}">'
               f'{"".join(article(nodes[name]) for name in canonical)}</div>')
    return ('<section class="technology-chronicle" id="tree-map" data-chronicle>'
            '<span id="branch-overview"></span><span id="selection"></span><span id="connections"></span><span id="turning-points"></span>'
            f'{stage}{bar}'
            f'<div class="ch-periods" id="chronicle-content">{"".join(periods)}{empty}</div>{library}</section>')


def expand(body, language):
    marker = re.search(r'<div data-ai-chronicle(?:="data-ai-chronicle")?></div>', body)
    if marker is None:
        return body
    data = json.loads(DATA.read_text(encoding="utf-8"))
    body = body.replace('class="technology-tree"', 'class="technology-tree tree-chronicle-ready"', 1)
    body = body.replace('class="tree-scene" id="tree-map"', 'class="tree-scene"', 1)
    return re.sub(r'<div data-ai-chronicle(?:="data-ai-chronicle")?></div>', lambda match: render(data, language), body, count=1)
