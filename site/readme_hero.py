"""The animated hero at the top of the README: real pages of the site in a browser window, one
scene at a time, with a caption beside them. GitHub plays CSS animations inside an SVG image, and
an image cannot load anything else, so the screenshots are embedded as WebP data URIs.

Two steps, run by hand when the site changes enough to be worth re-filming (not part of the build):

    python3 site/build.py --serve                      # in one terminal: the site on :8000
    python3 site/readme_hero.py capture --base http://127.0.0.1:8000/ --shots /tmp/hero-shots
    python3 site/readme_hero.py compose --shots /tmp/hero-shots   # writes .github/media/hero.{en,zh}.svg
    python3 site/readme_hero.py tour --shots /tmp/hero-shots      # writes .github/media/tour/*.webp

`capture` drives a local headless Chrome over the DevTools protocol (needs Google Chrome and the
`websocket-client` package); `compose` needs Pillow. Neither is needed to build the site.
"""
import argparse
import base64
import io
import json
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".github" / "media"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H = 960, 540
WIN = {"x": 384, "y": 62, "w": 544, "bar": 30}
WIN["screen_h"] = round(WIN["w"] / (540 / 372))
PHONE = {"x": 784, "y": 196, "w": 150, "h": 314}
SCENE = 4.0  # seconds per scene

SCENES = [  # name, page path (en, zh), caption (number, title lines, subtitle lines) per language
    ("figure", ("12-infra/performance.en.html", "12-infra/performance.html"),
     {"en": ("01 · Learn by example", ["Watch a worked", "example, step by step"],
             ["Every figure replays the note's own", "numbers and ends on its result."]),
      "zh": ("01 · 跟着例子学", ["一步一步，", "看完一个算例"], ["每张动画都用笔记自己的数字，", "停在笔记算出的结果上。"])}),
    ("paper", ("13-embodied/driving/bev-occupancy.en.html", "13-embodied/driving/bev-occupancy.html"),
     {"en": ("02 · From the papers", ["The original figure,", "where it's explained"],
             ["Openly licensed figures are embedded", "with credit; others link to the paper."]),
      "zh": ("02 · 原文图", ["论文原图，", "就放在讲它的地方"], ["开放许可的图直接嵌入并注明出处，", "其余链接到论文原处。"])}),
    ("frontier", ("11-frontier/self-improvement/index.en.html", "11-frontier/self-improvement/index.html"),
     {"en": ("03 · Frontier watch", ["Settled, still moving,", "and what to watch"],
             ["World models, self-improvement, test-time", "compute and more, with dates and sources."]),
      "zh": ("03 · 前沿观察", ["哪些已经定了，", "哪些还在变"], ["世界模型、递归自我改进（RSI）、测试时计算……", "每条都有日期和出处。"])}),
    ("bilingual", ("00-foundations/transformer-lab.en.html", "00-foundations/transformer-lab.html"),
     {"en": ("04 · Two languages", ["Every note in", "English and Chinese"],
             ["Written for each language,", "not translated line by line."]),
      "zh": ("04 · 中英双语", ["每一篇，", "中文和英文都有"], ["两种语言分别写，", "不是逐句翻译。"])}),
    ("interview", ("interview/questions.en.html", "interview/questions.html"),
     {"en": ("05 · Interview prep", ["Review by topic,", "in the time you have"],
             ["Questions with short answers that link", "back to the notes they come from."]),
      "zh": ("05 · 面试准备", ["按知识板块，", "过一遍考题"], ["每道题都有简短答案，", "并链接回出处的笔记。"])}),
    ("search", ("index.en.html", "index.zh.html"),
     {"en": ("06 · Find it fast", ["Search every note,", "in either language"],
             ["Full-text search runs in your browser;", "your reading history stays there too."]),
      "zh": ("06 · 全文搜索", ["所有笔记，", "中英文都能搜"], ["搜索在浏览器里完成，", "阅读记录也只存在你自己的浏览器。"])}),
]
FOOTER = {"en": "Free to read · open source · tianyi-zhang-02.github.io/cooking-agi",
          "zh": "免费阅读 · 开源 · tianyi-zhang-02.github.io/cooking-agi"}
BRAND = {"en": "Cooking AGI", "zh": "AGI 大锅烩"}


# ---------------------------------------------------------------- capture

class Chrome:
    """Just enough of the DevTools protocol: navigate, evaluate, screenshot."""

    def __init__(self, port=9333):
        import websocket
        self.proc = subprocess.Popen(
            [CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={tempfile.mkdtemp()}",
             "--no-first-run", "--hide-scrollbars", "--force-color-profile=srgb", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except (OSError, StopIteration, ValueError):
                time.sleep(0.2)
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)
        self.n = 0
        self.call("Page.enable")
        self.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"},
                                                         {"name": "prefers-reduced-motion", "value": "reduce"}])

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def js(self, expr, wait=0.4):
        value = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        time.sleep(wait)
        return value.get("result", {}).get("value")

    def go(self, url, wait=2.0):
        self.call("Page.navigate", url=url)
        time.sleep(wait)

    def viewport(self, width, height, scale, mobile=False):
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale, mobile=mobile)

    def shot(self, path, clip):
        data = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True, clip={**clip, "scale": 1})
        Path(path).write_bytes(base64.b64decode(data["data"]))

    def close(self):
        self.ws.close()
        self.proc.terminate()


HIDE = ("(() => { const s = document.createElement('style'); s.textContent = '.language-guide,.study-buddy,"
        "[class*=buddy],.reading-progress{display:none!important}' + (%s ? '.topbar{display:none!important}' : '');"
        " document.head.append(s); document.documentElement.dataset.theme = '%s'; return 1 })()")
TOUR = ("figure", "paper", "frontier")  # the three-column table under the hero, in both themes
ACTIONS = {
    "figure": """(() => { const w = document.querySelector('[data-widget="step-roofline"]');
        [...w.querySelectorAll('button')].slice(2, -2)[3].click(); w.id = 'hero-target'; return 1 })()""",
    "paper": "(() => { document.querySelector('figure.paper-fig').id = 'hero-target'; return 1 })()",
    "frontier": """(() => { const t = [...document.querySelectorAll('table')].find(t => /Settled|比较确定|已经定/.test(
        t.querySelector('thead')?.textContent || '')); t.id = 'hero-target'; return 1 })()""",
    "interview": """(() => { const ds = [...document.querySelectorAll('article details.qa')];
        const i = Math.max(0, ds.findIndex(d => /KV|attention|DPO|PPO|注意力/.test(d.querySelector('summary').textContent)));
        ds[i].open = true; ds[i].id = 'hero-target'; return i })()""",
}
RATIO = 540 / 372


def clip_around(chrome, pad, min_w, center):
    r = chrome.js("""(() => { const el = document.getElementById('hero-target'); el.scrollIntoView({block: 'center'});
        const b = el.getBoundingClientRect(); return {x: b.left + scrollX, y: b.top + scrollY, w: b.width, h: b.height} })()""", 0.8)
    w = min(1280, max(min_w, r["w"] + 2 * pad))
    h = w / RATIO
    x = max(0, min(r["x"] + r["w"] / 2 - w / 2, 1280 - w))
    y = max(0, r["y"] + r["h"] / 2 - h / 2) if center else max(0, r["y"] - pad)
    return {"x": x, "y": y, "width": w, "height": h}


def capture(base, shots):
    shots.mkdir(parents=True, exist_ok=True)
    chrome = Chrome()
    try:
        for li, lang in enumerate(("en", "zh")):
            chrome.viewport(1280, 800, 2)
            for name, paths, _ in SCENES:
                chrome.go(base + paths[li])
                chrome.js(HIDE % ("true" if name not in ("bilingual", "search") else "false", "dark"))
                full = {"x": 0, "y": 0, "width": 1280, "height": 1280 / RATIO}
                if name == "bilingual":
                    clip = full
                elif name == "search":
                    chrome.js("(() => { const b = document.querySelector('[data-search-open]'); b.hidden = false; b.click(); return 1 })()", 0.8)
                    chrome.js("""(() => { const i = document.querySelector('#note-search input'); i.value = 'KV cache';
                        i.dispatchEvent(new Event('input', {bubbles: true})); document.querySelector('#note-search').id = 'note-search';
                        document.querySelector('#note-search').setAttribute('data-hero', 1); return 1 })()""", 1.2)
                    chrome.js("(() => { document.querySelector('#note-search').id = 'hero-target'; return 1 })()", 0.1)
                    clip = clip_around(chrome, 24, 760, True)
                else:
                    chrome.js(ACTIONS[name], 0.8)
                    clip = clip_around(chrome, 110 if name == "interview" else 36, 780 if name == "figure" else 900, name == "paper")
                chrome.shot(shots / f"{lang}-{name}.png", clip)
                print(lang, name)
            chrome.viewport(390, 844, 3, mobile=True)
            other = SCENES[3][1][1 - li]
            chrome.go(base + other)
            chrome.js(HIDE % ("false", "dark"))
            chrome.shot(shots / f"{lang}-phone.png", {"x": 0, "y": 0, "width": 390, "height": 844})
            print(lang, "phone")
        chrome.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"},
                                                           {"name": "prefers-reduced-motion", "value": "reduce"}])
        for li, lang in enumerate(("en", "zh")):
            chrome.viewport(1280, 800, 2)
            for name, paths, _ in SCENES:
                if name not in TOUR:
                    continue
                chrome.go(base + paths[li])
                chrome.js(HIDE % ("true", "light"))
                chrome.js(ACTIONS[name], 0.8)
                chrome.shot(shots / f"{lang}-{name}-light.png", clip_around(chrome, 36, 780 if name == "figure" else 900, name == "paper"))
                print(lang, name, "light")
    finally:
        chrome.close()


# ---------------------------------------------------------------- compose

def data_uri(path, width, height, quality=70):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im = im.resize((width, round(width * im.height / im.width)), Image.LANCZOS)
    if im.height > height:
        im = im.crop((0, 0, width, height))
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def keyframes(name, start, end, total, fade=0.45, body="opacity:{o}"):
    """Visible from start to end (seconds) in a loop of total seconds, with short fades."""
    pct = lambda s: f"{100 * max(0.0, min(total, s)) / total:.3f}%"
    hidden, shown = body.format(o=0), body.format(o=1)
    frames = []
    if start <= 0:
        frames += [f"0%{{{shown}}}", f"{pct(end - fade)}{{{shown}}}", f"{pct(end)}{{{hidden}}}",
                   f"{pct(total - fade)}{{{hidden}}}", f"100%{{{shown}}}"]
    else:
        frames += [f"0%{{{hidden}}}", f"{pct(start)}{{{hidden}}}", f"{pct(start + fade)}{{{shown}}}",
                   f"{pct(end - fade)}{{{shown}}}", f"{pct(end)}{{{hidden}}}", f"100%{{{hidden}}}"]
    return f"@keyframes {name}{{{''.join(frames)}}}"


def compose(shots, lang):
    total = SCENE * len(SCENES)
    css, body = [], []
    rng = random.Random(7)
    accent, ink, muted = "#E8A672", "#F3ECE2", "#A89E92"
    # stage
    body.append(f'<rect width="{W}" height="{H}" fill="url(#sky)"/>')
    for _ in range(70):
        x, y, r = rng.uniform(0, W), rng.uniform(0, H * 0.8), rng.choice((0.6, 0.8, 1.0, 1.3))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#fff" opacity="{rng.uniform(0.12, 0.55):.2f}"/>')
    body.append(f'<ellipse cx="180" cy="{H + 40}" rx="520" ry="220" fill="url(#dawn)"/>')
    body.append(f'<ellipse cx="660" cy="300" rx="380" ry="250" fill="url(#glow)"/>')
    # brand
    body.append('<g transform="translate(56 56)"><rect width="32" height="32" rx="9" fill="#1d1a17" stroke="#3a332c"/>'
                f'<g transform="translate(4 4) scale(0.86)" fill="none" stroke="{accent}" stroke-width="1.8" stroke-linejoin="round">'
                '<path d="M4.5 6.5h6.8c1.5 0 2.7 1.2 2.7 2.7v12.3c0-1.5-1.2-2.7-2.7-2.7H4.5z"/>'
                '<path d="M23.5 6.5h-6.8c-1.5 0-2.7 1.2-2.7 2.7v12.3c0-1.5 1.2-2.7 2.7-2.7h6.8z"/></g>'
                f'<text x="44" y="22" font-size="18" font-weight="650" fill="{ink}" letter-spacing="-.01em">{esc(BRAND[lang])}</text></g>')
    # window shadow and frame
    x, y, w, bar, sh = WIN["x"], WIN["y"], WIN["w"], WIN["bar"], WIN["screen_h"]
    body.append(f'<rect x="{x + 18}" y="{y + 40}" width="{w - 36}" height="{bar + sh}" rx="18" fill="#000" opacity=".55" filter="url(#shadow)"/>')
    body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{bar + sh}" rx="14" fill="url(#frame)" stroke="#ffffff" stroke-opacity=".10"/>')
    for i, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        body.append(f'<circle cx="{x + 18 + i * 16}" cy="{y + bar / 2}" r="5" fill="{color}" opacity=".85"/>')
    body.append(f'<rect x="{x + 92}" y="{y + 7}" width="{w - 184}" height="{bar - 14}" rx="8" fill="#0d0c0b" stroke="#ffffff" stroke-opacity=".06"/>')
    body.append(f'<clipPath id="scr"><rect x="{x}" y="{y + bar}" width="{w}" height="{sh}" rx="0"/></clipPath>')
    body.append(f'<g clip-path="url(#scr)"><rect x="{x}" y="{y + bar}" width="{w}" height="{sh}" fill="#0c0a08"/>')
    li = 0 if lang == "en" else 1
    for i, (name, paths, captions) in enumerate(SCENES):
        start, end = i * SCENE, (i + 1) * SCENE + 0.45
        uri = data_uri(shots / f"{lang}-{name}.png", round(w * 1.7), round(sh * 1.7))
        css.append(keyframes(f"s{i}", start, end, total))
        css.append(f".s{i}{{animation:s{i} {total}s linear infinite both}}")
        css.append(f"@keyframes k{i}{{0%{{transform:scale(1)}}{100 * start / total:.3f}%{{transform:scale(1)}}"
                   f"{100 * min(end, total) / total:.3f}%{{transform:scale(1.045) translateY(-6px)}}100%{{transform:scale(1)}}}}")
        css.append(f".k{i}{{animation:k{i} {total}s linear infinite both;transform-box:view-box;transform-origin:{x + w / 2}px {y + bar}px}}")
        body.append(f'<g class="scene s{i}"><image class="k{i}" href="{uri}" x="{x}" y="{y + bar}" width="{w}" height="{sh}" preserveAspectRatio="xMidYMin slice"/></g>')
    # sheen sweeping across the screen at each scene change
    css.append(f"@keyframes sheen{{0%{{transform:translateX(-{w}px)}}{100 * 1.2 / SCENE:.2f}%{{transform:translateX({w + 200}px)}}100%{{transform:translateX({w + 200}px)}}}}")
    css.append(f".sheen{{animation:sheen {SCENE}s ease-in-out infinite both}}")
    body.append(f'<rect class="sheen" x="{x - 160}" y="{y + bar}" width="160" height="{sh}" fill="url(#sheenfill)" opacity=".5"/></g>')
    # url text per scene
    for i, (name, paths, _) in enumerate(SCENES):
        path = paths[li].replace("index.en.html", "").replace("index.zh.html", "")
        url = "…/cooking-agi/" + (path if len(path) <= 44 else path[:43] + "…")
        body.append(f'<text class="scene s{i}" x="{x + w / 2}" y="{y + bar / 2 + 4}" text-anchor="middle" font-size="10.5" fill="{muted}">{esc(url)}</text>')
    # phone for the bilingual scene
    px, py, pw, ph = PHONE["x"], PHONE["y"], PHONE["w"], PHONE["h"]
    phone_uri = data_uri(shots / f"{lang}-phone.png", round((pw - 12) * 2.2), round((ph - 12) * 2.2))
    i = 3
    start, end = i * SCENE + 0.35, (i + 1) * SCENE + 0.2
    pct = lambda s: f"{100 * s / total:.3f}%"
    css.append(f"@keyframes ph{{0%{{opacity:0;transform:translateX(28px)}}{pct(start)}{{opacity:0;transform:translateX(28px)}}"
               f"{pct(start + 0.6)}{{opacity:1;transform:translateX(0)}}{pct(end - 0.45)}{{opacity:1;transform:translateX(0)}}"
               f"{pct(end)}{{opacity:0;transform:translateX(0)}}100%{{opacity:0;transform:translateX(28px)}}}}")
    css.append(f".ph{{animation:ph {total}s cubic-bezier(.2,.7,.2,1) infinite both}}")
    body.append(f'<g class="ph"><rect x="{px + 10}" y="{py + 24}" width="{pw - 20}" height="{ph - 10}" rx="26" fill="#000" opacity=".6" filter="url(#shadow)"/>'
                f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="26" fill="url(#frame)" stroke="#ffffff" stroke-opacity=".14"/>'
                f'<clipPath id="pscr"><rect x="{px + 6}" y="{py + 6}" width="{pw - 12}" height="{ph - 12}" rx="21"/></clipPath>'
                f'<image href="{phone_uri}" x="{px + 6}" y="{py + 6}" width="{pw - 12}" height="{ph - 12}" clip-path="url(#pscr)" preserveAspectRatio="xMidYMin slice"/>'
                f'<rect x="{px + pw / 2 - 22}" y="{py + 12}" width="44" height="11" rx="5.5" fill="#000"/></g>')
    # captions
    for i, (_, _, captions) in enumerate(SCENES):
        number, title, sub = captions[lang]
        start, end = (i * SCENE + 0.15 if i else 0), (i + 1) * SCENE - 0.1
        css.append(keyframes(f"c{i}", start, end, total, 0.5, "opacity:{o}"))
        css.append(f".c{i}{{animation:c{i} {total}s ease-in-out infinite both}}")
        size = 34 if lang == "en" else 33
        lines = "".join(f'<tspan x="56" y="{224 + j * (size + 8)}">{esc(t)}</tspan>' for j, t in enumerate(title))
        subs = "".join(f'<tspan x="56" y="{224 + len(title) * (size + 8) + 22 + j * 23}">{esc(t)}</tspan>' for j, t in enumerate(sub))
        body.append(f'<g class="cap c{i}"><text x="56" y="176" fill="{accent}" font-size="13" font-weight="650" letter-spacing=".05em">{esc(number)}</text>'
                    f'<text fill="{ink}" font-size="{size}" font-weight="650" letter-spacing="-.02em">{lines}</text>'
                    f'<text fill="{muted}" font-size="14.5">{subs}</text></g>')
    # progress bars
    for i in range(len(SCENES)):
        bx = 56 + i * 44
        body.append(f'<rect x="{bx}" y="452" width="36" height="3" rx="1.5" fill="#fff" opacity=".14"/>')
        css.append(f"@keyframes p{i}{{0%{{transform:scaleX(0)}}{100 * i * SCENE / total:.3f}%{{transform:scaleX(0)}}"
                   f"{100 * (i + 1) * SCENE / total:.3f}%{{transform:scaleX(1)}}99.95%{{transform:scaleX(1)}}100%{{transform:scaleX(0)}}}}")
        css.append(f".p{i}{{animation:p{i} {total}s linear infinite both;transform-box:fill-box;transform-origin:left center}}")
        body.append(f'<rect class="pg p{i}" x="{bx}" y="452" width="36" height="3" rx="1.5" fill="{accent}"/>')
    body.append(f'<text x="56" y="486" font-size="12" fill="{muted}">{esc(FOOTER[lang])}</text>')
    body.append(f'<path d="M0 {H - 0.5}H{W}" stroke="#fff" stroke-opacity=".06"/>')
    defs = ('<defs>'
            '<radialGradient id="sky" cx="70%" cy="30%" r="90%"><stop offset="0" stop-color="#1b1f2a"/>'
            '<stop offset=".55" stop-color="#0f1015"/><stop offset="1" stop-color="#090807"/></radialGradient>'
            '<radialGradient id="dawn" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#ffa860" stop-opacity=".30"/>'
            '<stop offset="1" stop-color="#ffa860" stop-opacity="0"/></radialGradient>'
            '<radialGradient id="glow" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#6b7fa8" stop-opacity=".22"/>'
            '<stop offset="1" stop-color="#6b7fa8" stop-opacity="0"/></radialGradient>'
            '<linearGradient id="frame" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3b3631"/>'
            '<stop offset=".35" stop-color="#1a1816"/><stop offset=".7" stop-color="#26221e"/><stop offset="1" stop-color="#121110"/></linearGradient>'
            '<linearGradient id="sheenfill" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/>'
            '<stop offset=".5" stop-color="#fff" stop-opacity=".22"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>'
            '<filter id="shadow" x="-40%" y="-20%" width="180%" height="160%"><feGaussianBlur stdDeviation="18"/></filter>'
            '</defs>')
    font = "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif"
    label = {"en": "A tour of Cooking AGI: step-by-step figures, the papers' own figures, frontier watch, two languages, interview prep and search",
             "zh": "AGI 大锅烩一览：分步动画、论文原图、前沿观察、中英双语、面试准备与全文搜索"}[lang]
    # without animation (reduced motion, or a renderer that ignores CSS) only the first scene shows
    base = (".scene,.cap,.ph{opacity:0}.s0,.c0{opacity:1}"
            ".pg{transform:scaleX(0);transform-box:fill-box;transform-origin:left center}.p0{transform:scaleX(1)}")
    style = "<style>" + base + "".join(css) + "@media (prefers-reduced-motion: reduce){*{animation:none!important}}</style>"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="{esc(label)}" font-family="{esc(font)}">{style}{defs}{"".join(body)}</svg>\n')


def tour(shots, out):
    """Stills for the three-column table: dark and light, both languages, as WebP files."""
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "zh"):
        for name in TOUR:
            for theme, suffix in (("dark", ""), ("light", "-light")):
                im = Image.open(shots / f"{lang}-{name}{suffix}.png").convert("RGB")
                im = im.resize((640, round(640 * im.height / im.width)), Image.LANCZOS)
                path = out / f"{lang}-{name}-{theme}.webp"
                im.save(path, "WEBP", quality=74, method=6)
                print(path, f"{path.stat().st_size / 1024:.0f} KB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["capture", "compose", "tour"])
    ap.add_argument("--base", default="http://127.0.0.1:8000/")
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.step == "capture":
        capture(args.base, args.shots)
    elif args.step == "tour":
        tour(args.shots, args.out / "tour")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for lang in ("en", "zh"):
            svg = compose(args.shots, lang)
            path = args.out / f"hero.{lang}.svg"
            path.write_text(svg, encoding="utf-8")
            print(path, f"{len(svg.encode()) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
