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


def deck_for(page, config):
    if page.section.get("review") is False:
        return None
    return next((item for item in config["decks"]
                 if item["section"] == page.section["dir"]), None)


def render(page, config, pages_by_source):
    deck = deck_for(page, config)
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
<h3 class="review-question" tabindex="-1">{copy_html(card['question'], language)}</h3>
<details class="review-answer">
<summary><span class="review-show-label">{copy("看看思路", "Show reasoning")}</span><span class="review-hide-label">{copy("收起思路", "Hide reasoning")}</span><span class="review-reveal-arrow" aria-hidden="true">↓</span></summary>
<div class="review-answer-body">
{copy_html(card['answer'], language, "p")}
<div class="review-pitfall"><strong>{copy("别漏掉", "Keep in mind")}</strong>{copy_html(card['pitfall'], language)}</div>
<p class="review-source">{''.join(links)}</p>
</div>
</details>
<p class="review-mark" aria-live="polite"></p>
</div>''')
    return f'''<section class="chapter-review" id="chapter-review" data-review-section="{deck['section']}" data-review-language="{language}" aria-labelledby="review-heading">
<div class="review-heading"><div><h2 id="review-heading">{copy("回顾一下", "Quick recap")}</h2>
<p class="review-intro">{copy("先想一想，再看参考思路。", "Think it through, then compare your reasoning.")}</p></div>
<div class="review-utilities">
<button type="button" data-review-action="language" hidden>{"EN" if language == "zh" else "中文"}</button>
<details class="review-options" data-review-controls hidden>
<summary>{copy("更多", "More")}<span aria-hidden="true"> ···</span></summary>
<div class="review-toolbar">
<button type="button" data-review-action="filter" aria-pressed="false">{copy("只看未掌握", "Needs practice only")}</button>
<button type="button" data-review-action="shuffle">{copy("打乱顺序", "Shuffle")}</button>
<button type="button" data-review-action="reset">{copy("重置本章记录", "Reset chapter marks")}</button>
</div></details></div></div>
<div class="review-status" data-review-controls hidden>
<p class="review-progress" aria-live="polite" aria-atomic="true"></p><p class="review-mastery"></p>
</div>
<div class="review-meter" data-review-controls hidden aria-hidden="true"><span></span></div>
<div class="review-cards">{''.join(cards)}</div>
<div class="review-empty" hidden><p>{copy("这一组都掌握了。想再过一遍，随时回来。", "You've covered this set. Come back whenever you want a refresher.")}</p>
<button type="button" data-review-action="all">{copy("查看全部题目", "View all questions")}</button></div>
<div class="review-assessment" data-review-controls hidden>
<p>{copy("这题感觉怎么样？", "How did that go?")}</p><div class="review-ratings">
<button type="button" data-review-action="again" aria-pressed="false" disabled>{copy("还需练习", "Keep practicing")}</button>
<button type="button" data-review-action="understood" aria-pressed="false" disabled>{copy("能讲清了", "Got it")}</button>
</div>
</div>
<div class="review-actions" data-review-controls hidden>
<button type="button" data-review-action="previous"><span aria-hidden="true">←</span> {copy("上一题", "Previous")}</button>
<button type="button" data-review-action="next">{copy("下一题", "Next")} <span aria-hidden="true">→</span></button>
</div>
<p class="review-storage">{copy("进度只保存在此浏览器，中英文共用。", "Progress stays in this browser, shared across languages.")}</p>
</section>'''
