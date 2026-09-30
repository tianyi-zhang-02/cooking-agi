"""Validated bilingual chapter self-checks with a no-JavaScript fallback."""

import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".", "community", "templates"}


def load_config():
    return json.loads((ROOT / "site/review.json").read_text(encoding="utf-8"))


def validate(config, nav, root=ROOT):
    expected = {section["dir"] for section in nav["section"]
                if section["dir"] not in EXCLUDED
                and any((root / section["dir"]).glob("*.md"))}
    seen_sections, seen_ids = set(), set()
    for deck in config["decks"]:
        section = deck["section"]
        if section not in expected or section in seen_sections:
            raise ValueError(f"Unknown or duplicate review section: {section}")
        seen_sections.add(section)
        if not 2 <= len(deck["cards"]) <= 8:
            raise ValueError(f"Use 2–8 focused cards per chapter: {section}")
        for card in deck["cards"]:
            card_id = card["id"]
            if not re.fullmatch(r"[a-z][a-z0-9-]+", card_id) or card_id in seen_ids:
                raise ValueError(f"Invalid or duplicate card ID: {card_id}")
            seen_ids.add(card_id)
            for field in ("question", "answer", "pitfall"):
                if set(card[field]) != {"zh", "en"} or not all(
                        isinstance(value, str) and value.strip() for value in card[field].values()):
                    raise ValueError(f"Missing bilingual {field}: {card_id}")
            source = card["source"]
            source_path = root / source
            if (Path(source).is_absolute()
                    or not re.fullmatch(r"[A-Za-z0-9_/-]+\.md", source)
                    or not source_path.resolve().is_relative_to(root.resolve())
                    or not source_path.is_file()
                    or not source_path.with_suffix(".en.md").is_file()):
                raise ValueError(f"Missing or unsafe bilingual source: {card_id}")
    if expected != seen_sections:
        raise ValueError(f"Chapters without review cards: {sorted(expected - seen_sections)}")


def fingerprint(card):
    payload = json.dumps(card, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def copy_html(values, language, tag="span"):
    return "".join(f'<{tag} data-review-copy="{locale}" lang="{locale}"'
                   f'{" hidden" if locale != language else ""}>{html.escape(text)}</{tag}>'
                   for locale, text in values.items())


def render(page, config, pages_by_source):
    deck = next((item for item in config["decks"]
                 if item["section"] == page.section["dir"]), None)
    if not deck:
        return ""
    language = page.lang
    copy = lambda zh, en: copy_html({"zh": zh, "en": en}, language)
    cards = []
    for card in deck["cards"]:
        links = []
        for locale in ("zh", "en"):
            target = pages_by_source[card["source"]][locale]
            links.append(f'<a href="{html.escape(page.rel(target.url), quote=True)}"'
                         f' data-review-copy="{locale}" lang="{locale}"'
                         f'{" hidden" if locale != language else ""}>'
                         f'{"回到相关笔记 →" if locale == "zh" else "Revisit the note →"}</a>')
        cards.append(f'''<div class="review-card" data-card-id="{card['id']}" data-version="{fingerprint(card)}">
<h3 class="review-question">{copy_html(card['question'], language)}</h3>
<details class="review-answer">
<summary>{copy("展开思路", "Reveal reasoning")}</summary>
{copy_html(card['answer'], language, "p")}
<div class="review-pitfall"><strong>{copy("容易漏掉：", "Watch out: ")}</strong>{copy_html(card['pitfall'], language)}</div>
<p>{''.join(links)}</p>
</details>
<p class="review-mark" aria-live="polite"></p>
</div>''')
    return f'''<section class="chapter-review" id="chapter-review" data-review-section="{deck['section']}" data-review-language="{language}" aria-labelledby="review-heading">
<div class="review-heading"><h2 id="review-heading">{copy("合上笔记，试着讲一遍", "Close the notes. Try explaining it.")}</h2>
<button type="button" data-review-action="language" hidden>{"EN" if language == "zh" else "中文"}</button></div>
<p class="review-intro">{copy("本章复习 · 先自己回答，再展开思路。不用背答案，能讲清前提和理由就好。", "Chapter review · Answer before revealing the reasoning. Explain assumptions rather than memorizing the wording.")}</p>
<div class="review-toolbar" data-review-controls hidden>
<button type="button" data-review-action="filter" aria-pressed="false">{copy("只看未掌握", "Needs practice only")}</button>
<button type="button" data-review-action="shuffle">{copy("打乱顺序", "Shuffle")}</button>
<button type="button" data-review-action="reset">{copy("重置本章记录", "Reset chapter marks")}</button>
</div>
<p class="review-progress" aria-live="polite" aria-atomic="true"></p>
<div class="review-cards">{''.join(cards)}</div>
<p class="review-empty" hidden>{copy("这章都标记过了。可以取消筛选，再试一次。", "All cards are marked understood. Turn off the filter to revisit them.")}</p>
<div class="review-actions" data-review-controls hidden>
<button type="button" data-review-action="previous">{copy("上一张", "Previous")}</button>
<button type="button" data-review-action="again" disabled>{copy("再练一次", "Try again")}</button>
<button type="button" data-review-action="understood" disabled>{copy("讲清楚了", "Understood")}</button>
<button type="button" data-review-action="next">{copy("下一张", "Next")}</button>
</div>
<p class="review-storage">{copy("这是自测，不是考试。记录只保存在当前浏览器，中英文共用，不会上传。", "Self-checks, not an exam. Marks stay in this browser, shared across languages, never uploaded.")}</p>
</section>'''
