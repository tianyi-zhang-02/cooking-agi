from functools import lru_cache
from pathlib import Path


SOURCE = "00-foundations/deep-dives/kv-cache-and-inference.md"
DIRECTORY = Path(__file__).parent / "learning-views"


def supports(page, root):
    return page.src.relative_to(root).as_posix().replace(".en.md", ".md") == SOURCE


@lru_cache(maxsize=2)
def panels(language):
    if language not in {"zh", "en"}:
        raise ValueError("Unsupported learning-view language")
    return (DIRECTORY / f"kv-cache.{language}.html").read_text(encoding="utf-8")


def render(language, full_body):
    chinese = language == "zh"
    names = (("速览", "先抓住要点"), ("图解实验", "一步步看计算"), ("完整笔记", "推导、例子与细节")) if chinese else (
        ("Quick read", "Get the idea"), ("Visual lab", "Follow each step"), ("Full note", "Derivations and details"))
    label = "学习方式" if chinese else "Learning view"
    buttons = "".join(
        f'<button type="button" id="kv-{mode}-tab" role="tab" aria-selected="{str(mode == "full").lower()}" '
        f'aria-controls="kv-{mode}" tabindex="{0 if mode == "full" else -1}" data-learning-mode="{mode}">'
        f'<strong>{title}</strong><span>{hint}</span></button>'
        for mode, (title, hint) in zip(("quick", "lab", "full"), names))
    return (f'<div class="learning-views" data-learning-views="kv-cache" data-lang="{language}">'
            f'<nav class="learning-view-nav" data-learning-nav hidden aria-label="{label}">'
            f'<div role="tablist" aria-label="{label}">{buttons}</div></nav>'
            f'{panels(language)}'
            f'<div id="kv-full" data-learning-panel="full">{full_body}</div></div>')
