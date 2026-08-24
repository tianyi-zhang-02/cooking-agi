#!/usr/bin/env python3
"""公开仓库的泄漏兜底扫描。构建时跑，命中即让 CI 失败。

## 为什么这份清单和私有仓库那份不一样

私有仓库的 forbidden.txt 里写着具体要挡的值——真实百分比、真实卡数。
那份清单**本身就是机密**：一份公开的、写着 `98\\.8|17\\.9` 的正则表，
等于告诉所有人你在藏哪两个数字。

所以这里只放**类别级**的模式：形如超参的东西、形如内部代码路径的东西。
它认不出某个具体数值，但认得出"这看起来像是不该出现在这儿的配置"。

两份清单，两种失效模式，故意的：
    私有那份漏了 → 公开这份可能靠类别拦下
    公开这份漏了 → 私有那份在 promote 时已经拦过
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RULES = [
    # 雇主与平台内部叫法（用户在推荐语境里被叫成什么，是最容易漏的指纹）
    ("employer / internal term", r"linkedin|领英|字节跳动|bytedance"),
    ("internal term for users", r"practice/.*", r"\bmembers?\b|会员"),
    # 形如模型配置
    ("model config", r"practice/.*", r"qwen[\w.-]*|(?<!sub-)\b\d+(\.\d+)?B\b"),
    # 形如集群规模与超参
    ("cluster size / hyperparam", r"\b\d+\s*(张卡|块卡|GPUs?|devices)\b|world_size\s*=\s*\d+"),
    ("internal config key", r"per_device\w*\s*[:=]\s*\d+|in_batch_negatives"),
    # 形如内部代码路径
    ("internal code path", r"\b\w+\.py:\d+"),
]
EXEMPT = [r"sub-1B", r"pytorch\.org", r"^\s*<!--"]

SCAN = ["00-foundations", "01-data-and-feedback", "02-memory",
        "03-multimodal-learning", "04-search", "05-post-training", "06-systems",
        "07-evaluation", "08-model-experience", "09-personal-agi", "ai-infra",
        "papers", "practice"]


def main():
    exempt = [re.compile(p, re.IGNORECASE) for p in EXEMPT]
    rules = []
    for r in RULES:
        label, scope, rx = (r[0], None, r[1]) if len(r) == 2 else r
        rules.append((label, re.compile(scope) if scope else None,
                      re.compile(rx, re.IGNORECASE)))

    hits = []
    for d in SCAN:
        for f in (ROOT / d).rglob("*.md"):
            rel = f.relative_to(ROOT).as_posix()
            for i, line in enumerate(f.read_text(encoding="utf-8",
                                                 errors="replace").splitlines(), 1):
                if any(e.search(line) for e in exempt):
                    continue
                for label, scope, rx in rules:
                    if scope and not scope.search(rel):
                        continue
                    m = rx.search(line)
                    if m:
                        hits.append((rel, i, label, m.group(0), line.strip()[:88]))

    if hits:
        print(f"\n✗ 泄漏检查未通过：{len(hits)} 处\n", file=sys.stderr)
        for rel, i, label, matched, line in hits:
            print(f"  {rel}:{i}  [{label}]  «{matched}»", file=sys.stderr)
            print(f"      {line}\n", file=sys.stderr)
        print("这是公开仓库。要么改写这一行，要么——如果确属误报——"
              "在 site/leakcheck.py 的 EXEMPT 里加一条并说明理由。", file=sys.stderr)
        return 1
    print("✓ 泄漏检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
