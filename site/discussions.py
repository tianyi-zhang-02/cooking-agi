import html
import re
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with (ROOT / "site/discussions.toml").open("rb") as source:
        config = tomllib.load(source)
    validate(config)
    return config


def validate(config, root=ROOT):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config.get("repo", "")):
        raise ValueError("Invalid discussion repository")
    for field in ("repo_id", "category_id"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", config.get(field, "")):
            raise ValueError(f"Missing or invalid {field}")
    if not isinstance(config.get("embed_enabled"), bool) or not config.get("category"):
        raise ValueError("Explicit comment mode and category required")
    collection_ids = set()
    for collection in config.get("collection", []):
        identifier = collection.get("id", "")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", identifier) or identifier in collection_ids:
            raise ValueError("Invalid or duplicate discussion collection")
        for language in ("zh", "en"):
            for field in ("title", "summary"):
                value = collection.get(f"{field}_{language}")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Missing {language} collection {field}")
        collection_ids.add(identifier)
    seen_slugs, seen_numbers = set(), set()
    for topic in config.get("topic", []):
        slug, number = topic["slug"], topic["number"]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", slug) or slug in seen_slugs:
            raise ValueError("Invalid or duplicate discussion slug")
        if topic.get("collection") not in collection_ids:
            raise ValueError("Unknown discussion collection")
        if type(number) is not int or number <= 0 or number in seen_numbers:
            raise ValueError("Use a unique, existing GitHub discussion number")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", topic["author"]):
            raise ValueError("Invalid discussion author")
        date.fromisoformat(topic["date"])
        for language, suffix in (("zh", ".md"), ("en", ".en.md")):
            for field in ("title", "tag", "summary"):
                if not isinstance(topic.get(f"{field}_{language}"), str) or not topic[f"{field}_{language}"].strip():
                    raise ValueError(f"Missing {language} {field}")
            if not (root / "discussions" / (slug + suffix)).is_file():
                raise ValueError(f"Missing {language} topic: {slug}")
        seen_slugs.add(slug)
        seen_numbers.add(number)


def topic_for(page, config):
    if page.section.get("dir") != "discussions":
        return None
    slug = page.src.name.removesuffix(".md").removesuffix(".en")
    return next((topic for topic in config.get("topic", []) if topic["slug"] == slug), None)


def thread_url(topic, config):
    return f'https://github.com/{config["repo"]}/discussions/{topic["number"]}'


def header(page, config):
    chinese = page.lang == "zh"
    topic = topic_for(page, config)
    if not topic:
        eyebrow = "写在笔记之外" if chinese else "BEYOND THE NOTES"
        count = len(config.get("topic", []))
        total = f"{count} 篇随笔" if chinese else f"{count} essays"
        return f'''<header class="talk-heading talk-home-heading">
<div><p class="talk-eyebrow">{eyebrow}<span class="talk-total">{total}</span></p>
<h1>{"聊聊" if chinese else "Discussions"}</h1></div>
<svg class="talk-sketch" viewBox="0 0 150 110" fill="none" aria-hidden="true">
<ellipse cx="74" cy="57" rx="63" ry="25" transform="rotate(-23 74 57)" stroke="currentColor" opacity=".4"/>
<circle cx="75" cy="56" r="32" stroke="currentColor" opacity=".7"/>
<path d="m75 34 4 17 17 5-17 4-4 18-5-18-17-4 17-5Z" stroke="currentColor" stroke-linejoin="round"/>
<circle cx="25" cy="76" r="3" fill="currentColor"/>
<path d="M123 21v10m-5-5h10M38 15v6m-3-3h6" stroke="currentColor" stroke-linecap="round"/>
</svg></header>'''
    home = "index.html" if chinese else "index.en.html"
    back = "← 聊聊" if chinese else "← Discussions"
    join = "聊两句 ↓" if chinese else "Join the conversation ↓"
    collection = next(entry for entry in config["collection"] if entry["id"] == topic["collection"])
    category = html.escape(collection[f"title_{page.lang}"])
    return (f'<header class="talk-heading talk-article-heading">'
            f'<div class="talk-breadcrumb"><a class="talk-back" href="{home}">{back}</a>'
            f'<span aria-hidden="true">/</span><a href="{home}#collection-{collection["id"]}">{category}</a></div>'
            f'<p class="talk-eyebrow">{html.escape(topic[f"tag_{page.lang}"])}</p>'
            f'<h1>{html.escape(topic[f"title_{page.lang}"])}</h1>'
            f'<div class="talk-byline"><p class="talk-meta">'
            f'<a href="https://github.com/{topic["author"]}">@{topic["author"]}</a>'
            f'<span aria-hidden="true">·</span><time datetime="{topic["date"]}">{topic["date"]}</time></p>'
            f'<a class="talk-comment-jump" href="#comments">{join}</a></div></header>')


def topic_card(page, topic, prefix="topic"):
    language = page.lang
    suffix = ".html" if language == "zh" else ".en.html"
    href = html.escape(page.rel("discussions/" + topic["slug"] + suffix), quote=True)
    identifier = f'{prefix}-{topic["slug"]}'
    return (f'<a class="talk-card" href="{href}" aria-labelledby="{identifier}">'
            f'<div class="talk-card-copy"><h3 id="{identifier}">{html.escape(topic[f"title_{language}"])}</h3>'
            f'<p>{html.escape(topic[f"summary_{language}"])}</p></div>'
            '<span class="talk-card-arrow" aria-hidden="true">→</span></a>')


def topic_cards(page, config):
    language = page.lang
    chinese = language == "zh"
    jump_label = "按话题阅读" if chinese else "Browse by topic"
    jumps, sections = [], []
    for collection in config.get("collection", []):
        topics = [topic for topic in config.get("topic", []) if topic["collection"] == collection["id"]]
        if not topics:
            continue
        identifier = "collection-" + collection["id"]
        title = html.escape(collection[f"title_{language}"])
        summary = html.escape(collection[f"summary_{language}"])
        count = len(topics)
        jumps.append(f'<a href="#{identifier}">{title}<span class="talk-count">{count:02d}</span></a>')
        rows = "".join(f'<li>{topic_card(page, topic)}</li>' for topic in
                       sorted(topics, key=lambda value: value["date"], reverse=True))
        sections.append(f'''<section class="talk-collection" id="{identifier}" aria-labelledby="{identifier}-title">
<div class="talk-collection-heading"><h2 id="{identifier}-title">{title}</h2><p>{summary}</p></div>
<ul class="talk-list">{rows}</ul></section>''')
    return (f'<div class="talk-topics"><nav class="talk-jump" aria-label="{jump_label}">{"".join(jumps)}</nav>'
            f'{"".join(sections)}</div>')


def related(page, config):
    current = topic_for(page, config)
    if not current:
        return ""
    candidates = [topic for topic in config["topic"]
                  if topic["collection"] == current["collection"] and topic["slug"] != current["slug"]]
    candidates = sorted(candidates, key=lambda value: value["date"], reverse=True)[:2]
    if not candidates:
        return ""
    title = "还可以读读" if page.lang == "zh" else "Keep reading"
    rows = "".join(f'<li>{topic_card(page, topic, prefix="related")}</li>' for topic in candidates)
    return (f'<section class="talk-related" aria-labelledby="related-title">'
            f'<h2 id="related-title">{title}</h2><ul class="talk-list">{rows}</ul></section>')


def comments(page, config):
    topic = topic_for(page, config)
    if not topic:
        return ""
    chinese = page.lang == "zh"
    title = "接着聊" if chinese else "Your turn"
    privacy = ("留言公开，需 GitHub 登录。分享自己愿意公开的部分就好。" if chinese else
               "Comments are public and require a GitHub account. Share only what you want to make public.")
    link = "看留言，接着聊 ↗" if chinese else "Read & join in ↗"
    setup = ("目前在 GitHub 留言，中英文共用同一条讨论。" if chinese else
             "Join in on GitHub; both languages share the same conversation.")
    widget = ""
    if config["embed_enabled"]:
        setup = ("点击后会连接 giscus / GitHub，加载公开留言。" if chinese else
                 "Loading comments connects to giscus / GitHub to display the public thread.")
        attrs = {"repo": config["repo"], "repo-id": config["repo_id"], "category": config["category"],
                 "category-id": config["category_id"], "number": str(topic["number"]),
                 "lang": "zh-CN" if chinese else "en"}
        attributes = " ".join(f'data-{key}="{html.escape(value, quote=True)}"' for key, value in attrs.items())
        label = "加载留言" if chinese else "Load comments"
        widget = (f'<div class="talk-widget" data-discussion-widget {attributes}>'
                  f'<button type="button" class="talk-load" data-load-comments hidden>{label}</button>'
                  '<p class="talk-load-status" role="status" aria-live="polite"></p>'
                  '<div class="giscus"></div></div>')
    return (f'<section class="talk-comments" id="comments" aria-labelledby="comments-title">'
            f'<h2 id="comments-title">{title}</h2><p>{privacy}</p><p class="talk-caption">{setup}</p>{widget}'
            f'<a class="talk-external" href="{thread_url(topic, config)}">{link}</a></section>')
