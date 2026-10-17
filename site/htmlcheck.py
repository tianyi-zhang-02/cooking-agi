#!/usr/bin/env python3
"""构建产物的脚本注入检查。Build 之后跑，命中即让 CI 失败。

笔记是 Markdown，但 Markdown 允许原始 HTML，attr_list 还能给标题加属性。
所以一篇 PR 里的笔记可以夹带 <script>、onclick=、javascript: 链接或 iframe。
审核者未必每行都看得出来，这里对 _site 里每个页面做最后一道检查：

- 内联 <script> 必须在本页 CSP 里有哈希（build.py 只给模板自己的脚本算哈希）；
  跳转小页面例外，它只有一行 location.replace。
- 外部脚本只能来自本站或固定版本的 KaTeX。
- 不允许 on* 事件属性、javascript:/vbscript: 链接、非图片的 data: 地址、
  iframe / object / embed / base / srcdoc。
- 每个正式页面都要有 CSP。
"""

import base64
import hashlib
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"

SCRIPT_ORIGINS = ("https://cdn.jsdelivr.net/npm/katex@0.16.9/",)
URL_ATTRIBUTES = {"href", "src", "action", "formaction", "xlink:href", "poster", "data", "background", "cite"}
BANNED_TAGS = {"iframe", "frame", "object", "embed", "base", "applet"}
# a redirect to a section of the new page has no use for the old page's anchor
REDIRECT_SCRIPT = re.compile(r'location\.replace\("[^"<>\\]*"(?: \+ location\.hash)?\)')
HASH = re.compile(r"'sha256-([A-Za-z0-9+/=]+)'")


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.problems, self.inline, self.csp, self.refresh = [], [], None, False
        self.in_script, self.buffer = False, []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in BANNED_TAGS:
            self.problems.append(f"<{tag}> is not allowed")
        for name, value in attrs:
            if name.startswith("on"):
                self.problems.append(f"event handler {name}= on <{tag}>")
            if name == "srcdoc":
                self.problems.append(f"srcdoc on <{tag}>")
            if name in URL_ATTRIBUTES and value:
                scheme = re.sub(r"[\s\x00-\x1f]", "", value).lower()
                if scheme.startswith(("javascript:", "vbscript:")):
                    self.problems.append(f"{name}={value[:40]!r} on <{tag}>")
                elif scheme.startswith("data:") and not (
                        scheme.startswith("data:image/") and tag in {"img", "link", "source"}):
                    self.problems.append(f"data: URL in {name} on <{tag}>")
        if tag == "meta":
            equiv = (values.get("http-equiv") or "").lower()
            if equiv == "content-security-policy":
                self.csp = values.get("content") or ""
            elif equiv == "refresh":
                self.refresh = True
        if tag == "script":
            src = values.get("src")
            if src is None:
                self.in_script, self.buffer = True, []
            elif re.match(r"^([a-z][a-z0-9+.-]*:|//)", src, re.I) and not src.startswith(SCRIPT_ORIGINS):
                self.problems.append(f"external script {src[:80]}")

    def handle_endtag(self, tag):
        if tag == "script" and self.in_script:
            self.inline.append("".join(self.buffer))
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.buffer.append(data)


def digest(text):
    return base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii")


def check(text):
    page = Page()
    page.feed(text)
    page.close()
    problems = list(page.problems)
    redirect_stub = page.refresh and page.csp is None
    if page.csp is None and not redirect_stub:
        problems.append("missing Content-Security-Policy")
    allowed = set(HASH.findall(page.csp or ""))
    for script in page.inline:
        if redirect_stub and REDIRECT_SCRIPT.fullmatch(script.strip()):
            continue
        if digest(script) not in allowed:
            problems.append(f"inline script without a CSP hash: {script.strip()[:60]!r}")
    return problems


def main():
    if not OUT.is_dir():
        sys.exit("_site 不存在：先运行 python site/build.py")
    failures = 0
    for path in sorted(OUT.rglob("*.html")):
        for problem in check(path.read_text(encoding="utf-8")):
            failures += 1
            print(f"✗ {path.relative_to(OUT)}: {problem}")
    if failures:
        sys.exit(f"\n{failures} 处可能的脚本注入。笔记里不要写原始 <script>、事件属性或 javascript: 链接；"
                 "交互图请走 <!-- widget:NAME --> 和 site/static/。")
    print("✓ 页面脚本检查通过")


if __name__ == "__main__":
    main()
