#!/usr/bin/env python3
"""Render static/og.png: the card chat apps and social sites show when a page of
this site is shared. Run it by hand after changing the site title or tagline:

    python3 site/og_card.py

The PNG is committed, so the CI build needs neither Pillow nor a CJK font. The
card repeats the site's own look: warm near-black, a star field, the planet limb
with its sunrise, and the title in unbleached ink with one amber accent.
"""
from __future__ import annotations

import glob
import random
import tomllib
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

SITE = Path(__file__).resolve().parent
OUT = SITE / "static" / "og.png"
W, H = 1200, 630

PAPER = (12, 10, 8)          # --paper
BODY = (7, 6, 5)             # the planet's night side
INK = (239, 230, 218)        # --ink
MUTED = (165, 154, 140)      # --muted
EMBER = (232, 166, 114)      # --ember

FONT_PATHS = (
    glob.glob("/System/Library/AssetsV2/**/PingFang.ttc", recursive=True)
    + ["/System/Library/Fonts/PingFang.ttc",
       "/System/Library/Fonts/Hiragino Sans GB.ttc",
       "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
       "/System/Library/Fonts/Helvetica.ttc",
       "/System/Library/Fonts/Supplemental/Arial.ttf",
       "/System/Library/Fonts/Menlo.ttc"]
)


def font(size: int, families: tuple[str, ...], styles: tuple[str, ...]) -> ImageFont.FreeTypeFont:
    """The first installed face whose family and style match, probing every face of each collection."""
    for path in FONT_PATHS:
        if not Path(path).exists():
            continue
        for style in styles:
            for index in range(48):
                try:
                    face = ImageFont.truetype(path, size, index=index)
                except (OSError, IndexError, ValueError):
                    break
                family, face_style = face.getname()
                if any(f in family for f in families) and face_style == style:
                    return face
    return ImageFont.load_default(size)


def layer() -> Image.Image:
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def radial(center, radius, color, peak, steps=72) -> Image.Image:
    """A soft radial glow: concentric discs, inner ones drawn last, then blurred."""
    out = layer()
    d = ImageDraw.Draw(out)
    for i in range(steps, 0, -1):
        t = i / steps
        r = radius * t
        a = int(255 * peak * (1 - t) ** 1.7)
        d.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], fill=color + (a,))
    return out.filter(ImageFilter.GaussianBlur(radius / steps * 2.2))


def disc(center, r, fill, blur=0) -> Image.Image:
    out = layer()
    ImageDraw.Draw(out).ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], fill=fill)
    return out.filter(ImageFilter.GaussianBlur(blur)) if blur else out


def stars(seed=20260921) -> Image.Image:
    rng = random.Random(seed)
    out = layer()
    d = ImageDraw.Draw(out)
    warm, cool, amber = (250, 243, 230), (205, 215, 240), (240, 205, 170)
    for _ in range(1100):
        x, y = rng.random() * W, rng.random() * H * 0.86
        m = rng.random()
        tone = warm if rng.random() < 0.72 else (cool if rng.random() < 0.7 else amber)
        if m < 0.80:                                  # faint: single pixels
            d.point((x, y), fill=tone + (rng.randint(40, 120),))
        elif m < 0.965:                               # mid
            d.ellipse([x - .9, y - .9, x + .9, y + .9], fill=tone + (rng.randint(150, 220),))
        else:                                         # bright, with a bloom
            r = 1.4 + rng.random() * .9
            d.ellipse([x - r, y - r, x + r, y + r], fill=tone + (255,))
    bloom = layer()
    b = ImageDraw.Draw(bloom)
    rng = random.Random(seed + 1)
    for _ in range(14):
        x, y = rng.random() * W, rng.random() * H * 0.8
        b.ellipse([x - 5, y - 5, x + 5, y + 5], fill=warm + (110,))
    bloom = bloom.filter(ImageFilter.GaussianBlur(5))
    return Image.alpha_composite(bloom, out)


def milky_way() -> Image.Image:
    band = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(band)
    d.ellipse([W * 0.30, -H * 0.55, W * 1.35, H * 0.75], fill=52)
    d.ellipse([W * 0.50, -H * 0.35, W * 1.15, H * 0.55], fill=34)
    band = band.rotate(-22, resample=Image.BICUBIC, center=(W * 0.8, H * 0.15)).filter(ImageFilter.GaussianBlur(70))
    out = Image.new("RGBA", (W, H), (236, 226, 212, 0))
    out.putalpha(band)
    return out


def planet() -> Image.Image:
    R = 1900
    cx, cy = W * 0.5, 478 + R                     # the limb's crown at y = 478
    haze = disc((cx, cy), R + 120, (150, 190, 255, 95), blur=42)
    edge = disc((cx, cy), R + 14, (255, 250, 240, 230), blur=5)
    glare = radial((110, 530), 520, (255, 168, 96), 0.62)
    atmosphere = Image.alpha_composite(Image.alpha_composite(glare, haze), edge)
    # terminator: the atmosphere thins towards the night side on the right
    ramp = Image.linear_gradient("L").rotate(90, expand=True).resize((W, H))
    ramp = ramp.point(lambda v: 255 - int(v * 0.62))
    atmosphere.putalpha(ImageChops.multiply(atmosphere.getchannel("A"), ramp))
    body = disc((cx, cy), R, BODY + (255,))
    inner = disc((cx, cy), R, (120, 160, 230, 60), blur=26)
    inner.putalpha(ImageChops.multiply(inner.getchannel("A"), body.getchannel("A")))
    return Image.alpha_composite(Image.alpha_composite(atmosphere, body), inner)


def spaced(draw, xy, text, fnt, fill, tracking):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + tracking


def main() -> None:
    site = tomllib.loads((SITE / "nav.toml").read_text(encoding="utf-8"))["site"]
    img = Image.new("RGBA", (W, H), PAPER + (255,))
    for part in (milky_way(), stars(), planet()):
        img = Image.alpha_composite(img, part)

    d = ImageDraw.Draw(img)
    mono = font(21, ("Menlo", "SF Mono", "Courier"), ("Regular",))
    cjk_bold = font(102, ("PingFang SC", "Hiragino Sans GB", "Noto Sans CJK"), ("Semibold", "Medium", "Bold", "Regular"))
    cjk = font(38, ("PingFang SC", "Hiragino Sans GB", "Noto Sans CJK"), ("Regular", "Light"))
    latin = font(26, ("Helvetica", "Arial"), ("Regular",))

    x0 = 96
    d.rounded_rectangle([x0, 132, x0 + 56, 135], radius=2, fill=EMBER)
    spaced(d, (x0, 152), f"{site['title_en'].upper()} · {site['author_en'].upper()}", mono, EMBER + (215,), 4.2)
    d.text((x0 - 5, 190), site["title_zh"], font=cjk_bold, fill=INK)
    d.text((x0, 336), site["tagline_zh"], font=cjk, fill=MUTED)
    d.text((x0, 394), site["tagline_en"], font=latin, fill=MUTED + (205,))

    out = img.convert("RGB")
    out.save(OUT, optimize=True)
    size = OUT.stat().st_size
    if size > 320_000:                             # keep chat apps happy; dithered palette hides the banding
        out.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(OUT, optimize=True)
        size = OUT.stat().st_size
    print(f"wrote {OUT.relative_to(SITE.parent)} ({size // 1024} KB)")


if __name__ == "__main__":
    main()
