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
    seen_slugs, seen_numbers = set(), set()
    for topic in config.get("topic", []):
        slug, number = topic["slug"], topic["number"]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", slug) or slug in seen_slugs:
            raise ValueError("Invalid or duplicate discussion slug")
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
        return (f'<header class="talk-heading"><p class="talk-eyebrow">OFF THE CLOCK</p>'
                f'<h1>{"聊聊" if chinese else "Discussions"}</h1></header>')
    home = "index.html" if chinese else "index.en.html"
    back = "← 所有话题" if chinese else "← All topics"
    return (f'<header class="talk-heading"><a class="talk-back" href="{home}">{back}</a>'
            f'<p class="talk-eyebrow">{html.escape(topic[f"tag_{page.lang}"])}</p>'
            f'<h1>{html.escape(topic[f"title_{page.lang}"])}</h1>'
            f'<p class="talk-meta"><a href="https://github.com/{topic["author"]}">@{topic["author"]}</a>'
            f'<span aria-hidden="true">·</span><time datetime="{topic["date"]}">{topic["date"]}</time></p></header>')


def intro(page):
    chinese = page.lang == "zh"
    title = "认真做事，也随便聊聊。" if chinese else "Room for a different kind of conversation."
    caption = "不用每个问题都有标准答案。" if chinese else "Not every question needs a right answer."
    return f'''<div class="talk-intro">
<div><p class="talk-lead">{title}</p><p class="talk-caption">{caption}</p></div>
<svg class="talk-orbit" viewBox="0 0 180 140" fill="none" aria-hidden="true">
<ellipse cx="90" cy="72" rx="77" ry="39" stroke="currentColor" stroke-dasharray="2 7" transform="rotate(-20 90 72)"/>
<g class="talk-bubble-a"><path d="M30 30h69a12 12 0 0 1 12 12v27a12 12 0 0 1-12 12H60L43 94V81H30a12 12 0 0 1-12-12V42a12 12 0 0 1 12-12Z" fill="var(--paper)" stroke="currentColor"/>
<circle cx="46" cy="56" r="2" fill="currentColor"/><circle cx="64" cy="56" r="2" fill="currentColor"/><circle cx="82" cy="56" r="2" fill="currentColor"/></g>
<g class="talk-bubble-b"><path d="M98 75h42a10 10 0 0 1 10 10v20a10 10 0 0 1-10 10h-7v12l-15-12H98a10 10 0 0 1-10-10V85a10 10 0 0 1 10-10Z" fill="var(--paper)" stroke="currentColor"/>
<path d="M109 94q10 12 20 0" stroke="currentColor" stroke-linecap="round"/></g>
<path d="m141 21 2 6 6 2-6 2-2 6-2-6-6-2 6-2Z" fill="currentColor"/>
</svg></div>'''


def topic_cards(page, config):
    language = page.lang
    heading = "最近聊到" if language == "zh" else "On the table"
    cards = []
    for topic in sorted(config.get("topic", []), key=lambda value: value["date"], reverse=True):
        suffix = ".html" if language == "zh" else ".en.html"
        href = html.escape(page.rel("discussions/" + topic["slug"] + suffix), quote=True)
        label = "读一读，接着聊" if language == "zh" else "Read & join in"
        cards.append(f'''<a class="talk-card" href="{href}">
<div class="talk-meta"><span>{html.escape(topic[f'tag_{language}'])}</span><time datetime="{topic['date']}">{topic['date']}</time></div>
<h3>{html.escape(topic[f'title_{language}'])}</h3><p>{html.escape(topic[f'summary_{language}'])}</p>
<span class="talk-card-link">{label} <span aria-hidden="true">↗</span></span></a>''')
    return f'<section class="talk-topics" aria-label="{heading}"><p class="talk-eyebrow">{heading}</p>{"".join(cards)}</section>'


def comments(page, config):
    topic = topic_for(page, config)
    if not topic:
        return ""
    chinese = page.lang == "zh"
    title = "接着聊" if chinese else "Your turn"
    privacy = ("中英文共用这条讨论。留言公开，发言需 GitHub 登录；不想公开的经历，不用勉强分享。" if chinese else
               "Both languages share one thread. Comments are public and require a GitHub account. Share only what you are comfortable making public.")
    link = "去 GitHub 看留言 / 回复 ↗" if chinese else "Read & reply on GitHub ↗"
    setup = ("目前在 GitHub 留言；站内留言接通后，也会显示同一条讨论。" if chinese else
             "For now, join in on GitHub. The in-page comments will use this same thread once connected.")
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
