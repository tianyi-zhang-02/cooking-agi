"""Local company marks, independent of self-reported career outcomes."""

import html
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "tech": ("科技与产品", "Tech & products"),
    "ai": ("AI 与模型", "AI & models"),
    "software": ("软件与云服务", "Software & cloud"),
    "quant": ("量化与金融科技", "Quant & finance"),
}


def normalize(value):
    return re.sub(r"[\W_]+", "", value.casefold())


@lru_cache(maxsize=1)
def load():
    data = json.loads((ROOT / "site/company-icons.json").read_text(encoding="utf-8"))
    aliases = {}
    for entry in data["companies"]:
        if not re.fullmatch(r"[a-z0-9-]+", entry["id"]) or entry["group"] not in GROUPS:
            raise ValueError("Invalid company preset")
        if entry.get("asset"):
            if not re.fullmatch(r"[a-z0-9-]+\.svg", entry["asset"]):
                raise ValueError("Company icons must use local SVG filenames")
            if not (ROOT / "site/static/company-icons" / entry["asset"]).is_file():
                raise ValueError("Missing company icon asset")
        for name in [entry["id"], entry["name"], *entry.get("aliases", [])]:
            key = normalize(name)
            if key in aliases and aliases[key]["id"] != entry["id"]:
                raise ValueError(f"Ambiguous company alias: {name}")
            aliases[key] = entry
    return data["companies"], aliases


def resolve(company_id, name):
    aliases = load()[1]
    return aliases.get(normalize(company_id)) or aliases.get(normalize(name))


def mark(company_id, name, prefix):
    entry = resolve(company_id, name)
    display_name = entry["name"] if entry else name
    fallback = f'<span class="company-wordmark">{html.escape(display_name)}</span>'
    image = ""
    if entry and entry.get("asset"):
        treatment = "native" if entry.get("native") else "mono"
        image = f'<img class="company-mark-{treatment}" src="{prefix}static/company-icons/{entry["asset"]}" alt="" width="40" height="40" loading="lazy" decoding="async">'
    text_class = "" if image else " company-mark-text"
    return f'<span class="company-mark{text_class}" aria-hidden="true">{fallback}{image}</span>'


def identity(company_id, name, prefix):
    entry = resolve(company_id, name)
    icon = mark(company_id, name, prefix) if entry and entry.get("asset") else ""
    return f'{icon}<span>{html.escape(name)}</span>'


def catalog(page, request_url):
    zh = page.lang == "zh"
    entries = load()[0]
    prefix = "../" * page.depth
    groups = []
    for group, labels in GROUPS.items():
        items = []
        for entry in sorted((entry for entry in entries if entry["group"] == group), key=lambda entry: entry["name"].casefold()):
            search = html.escape(" ".join([entry["name"], entry["id"], *entry.get("aliases", [])]), quote=True)
            label = "文字标识" if zh else "Text mark"
            fallback = f'<small>{label}</small>' if not entry.get("asset") else ""
            icon = mark(entry["id"], entry["name"], prefix) if entry.get("asset") else ""
            text_class = "company-preset-wordmark" if not icon else ""
            items.append(f'<li class="{text_class}" data-company-preset data-search="{search}">{icon}<span>{html.escape(entry["name"])}{fallback}</span></li>')
        groups.append(f'<section data-company-group><h3>{labels[0 if zh else 1]}</h3><ul>{"".join(items)}</ul></section>')
    return f'''<div class="company-library" id="company-icons">
      <div class="company-request"><div><h2>{"没找到你的公司？" if zh else "Missing your company?"}</h2>
      <p>{"Tech、AI、量化，或其他行业都欢迎。带上公司名、官网和 logo 资源，开个 issue ping 我一下，我确认后就加 :)" if zh else "Tech, AI, quant, or another field—all welcome. Send the company name, website, and logo resources in an issue and ping me. I’ll review it and add it :)"}</p></div>
      <a href="{request_url}">{"申请新公司图标 ↗" if zh else "Request a company icon ↗"}</a></div>
      <details class="company-presets"><summary><span>{"看看已有的公司标识" if zh else "Browse company presets"}</span><span>{len(entries)}</span></summary>
        <p class="company-disclaimer">{"这里只是预设图标，不是去向名单，也不表示合作或背书。暂缺 logo 的直接显示完整公司名，不用单个字母代替；公司不在名单里，也照样可以分享。" if zh else "An icon library, not a list of reader destinations or endorsements. Missing logos use the full company name, not initials. You can share an update even if your company is not listed."}</p>
        <label class="company-search" data-company-search-wrap hidden><span>{"查找公司" if zh else "Find a company"}</span><input type="search" data-company-search placeholder="{"公司名或简称，如 NVIDIA / 英伟达" if zh else "Company name or alias, e.g. NVIDIA / HRT"}" autocomplete="off"></label>
        <p class="company-search-status" data-company-status role="status" aria-live="polite"></p>
        <div class="company-preset-groups">{"".join(groups)}</div>
        <p class="company-search-empty" data-company-no-match hidden>{"还没收录，欢迎用上面的入口告诉我。" if zh else "Not listed yet? Let me know using the request link above."}</p>
      </details></div>'''
