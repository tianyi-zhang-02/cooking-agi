#!/usr/bin/env python3
"""中英对照检查：每篇笔记的 .md 与 .en.md 在结构上是否一致。

    python3 site/paritycheck.py            # 报告
    python3 site/paritycheck.py --strict   # 有缺口就返回非零（给 CI 用）

它不比较措辞，只比较**结构**：有没有英文版、标题数、代码块、公式块、表格行、
折叠块、交互 widget、图片。结构对得上，不代表翻译得好；结构对不上，几乎一定
是某一边少了内容。规范见 EDITORIAL.md 的“双语阅读原则”。
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"templates"}                       # 模板不是给读者看的页面

COUNTERS = {
    "h2": r"(?m)^## ", "h3": r"(?m)^### ", "code": r"(?m)^```", "math": r"(?m)^\$\$",
    "table rows": r"(?m)^\|", "details": r"<details", "widgets": r"<!--\s*widget:", "images": r"!\[|<img ",
}


def shape(text: str) -> dict:
    fenced = re.sub(r"```.*?```", "```\n```", text, flags=re.S)     # headings inside code are not headings
    out = {k: len(re.findall(rx, fenced if k in ("h2", "h3", "table rows") else text)) for k, rx in COUNTERS.items()}
    out["code"] //= 2
    out["math"] //= 2
    return out


def main() -> int:
    nav = tomllib.loads((ROOT / "site" / "nav.toml").read_text(encoding="utf-8"))
    dirs = [s["dir"] for s in nav["section"]]
    missing, drift = [], []
    seen = set()
    for d in dirs:
        base = ROOT if d == "." else ROOT / d
        if not base.exists() or d.split("/")[0] in SKIP:
            continue
        for zh in sorted(base.glob("*.md")):
            if zh.name.endswith(".en.md") or zh in seen:
                continue
            if d == "." and zh.name not in {"README.md", "EDITORIAL.md"}:
                continue
            seen.add(zh)
            en = zh.with_name(zh.stem + ".en.md")
            rel = zh.relative_to(ROOT).as_posix()
            if not en.exists():
                missing.append(rel)
                continue
            a, b = shape(zh.read_text(encoding="utf-8")), shape(en.read_text(encoding="utf-8"))
            diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
            if diff:
                drift.append((rel, diff))

    print(f"检查了 {len(seen)} 篇笔记\n")
    print(f"缺英文版：{len(missing)} 篇")
    for rel in missing:
        print(f"  - {rel}")
    print(f"\n结构不一致：{len(drift)} 篇   （中文 → 英文）")
    for rel, diff in sorted(drift, key=lambda x: -sum(abs(a - b) for a, b in x[1].values())):
        print(f"  - {rel}: " + ", ".join(f"{k} {a}→{b}" for k, (a, b) in diff.items()))
    if not missing and not drift:
        print("\n✓ 中英两版结构一致")
    return 1 if ("--strict" in sys.argv and (missing or drift)) else 0


if __name__ == "__main__":
    sys.exit(main())
