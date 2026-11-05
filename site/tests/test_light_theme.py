from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
STYLE = (ROOT / "site/static/style.css").read_text()
SEARCH = (ROOT / "site/static/search.css").read_text()


def declarations(source, selector):
    values = {}
    for match in re.finditer(r"(?m)^[ \t]*" + re.escape(selector) + r"\s*\{([^{}]*)\}", source):
        for name, value in re.findall(r"([\w-]+)\s*:\s*([^;]+);", match[1]):
            values[name] = value.strip()
    return values


def luminance(color):
    channels = [int(color[offset:offset + 2], 16) / 255 for offset in (1, 3, 5)]
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4
              for value in channels]
    return sum(value * weight for value, weight in zip(linear, (.2126, .7152, .0722)))


def contrast(foreground, background):
    lower, higher = sorted((luminance(foreground), luminance(background)))
    return (higher + .05) / (lower + .05)


class LightThemeTests(unittest.TestCase):
    def test_reading_colors_have_normal_text_contrast(self):
        tokens = declarations(STYLE, ':root[data-theme="light"]')
        for foreground in ("--ink", "--ink-soft", "--muted", "--faint", "--link", "--ember"):
            for background in ("--paper", "--panel", "--code-bg"):
                with self.subTest(foreground=foreground, background=background):
                    self.assertGreaterEqual(contrast(tokens[foreground], tokens[background]), 4.5)

    def test_both_themes_keep_the_three_colour_roles_readable(self):
        for selector in (":root", ':root[data-theme="light"]'):
            tokens = declarations(STYLE, selector)
            pairs = [("--band-ink", "--band"), ("--band-muted", "--band"), ("--accent-ink", "--accent-fill"),
                     ("--ink", "--side-bg"), ("--muted", "--side-bg"), ("--faint", "--side-bg"), ("--ember", "--side-bg")]
            for foreground, background in pairs:
                with self.subTest(theme=selector, foreground=foreground, background=background):
                    self.assertGreaterEqual(contrast(tokens[foreground], tokens[background]), 4.5)

    def test_dark_reading_colors_have_normal_text_contrast(self):
        tokens = declarations(STYLE, ":root")
        for foreground in ("--ink", "--ink-soft", "--muted", "--faint", "--link", "--ember"):
            for background in ("--paper", "--panel", "--code-bg"):
                with self.subTest(foreground=foreground, background=background):
                    self.assertGreaterEqual(contrast(tokens[foreground], tokens[background]), 4.5)

    def test_palette_is_near_black_grey_band_and_one_green(self):
        tokens = declarations(STYLE, ":root")
        self.assertEqual(tokens["--paper"], "#040804")
        self.assertEqual(tokens["--band"], "#181A17")
        self.assertEqual(tokens["--accent-fill"], "#7FEE64")

    def test_cover_sits_on_the_band_in_both_themes(self):
        self.assertEqual(declarations(STYLE, ".hero-title")["color"], "var(--band-ink)")
        self.assertEqual(declarations(STYLE, ".hero-kicker")["color"], "var(--band-muted)")
        self.assertEqual(declarations(STYLE, ':root[data-theme="light"] .sky'), {})
        self.assertEqual(declarations(STYLE, ':root[data-theme="light"] .hero'), {})
        self.assertIn("cover ? '--band' : '--paper'", (ROOT / "site/static/sky.js").read_text())

    def test_search_uses_a_separate_light_overlay(self):
        dialog = declarations(SEARCH, ':root[data-theme="light"] .note-search')
        backdrop = declarations(SEARCH, ':root[data-theme="light"] .note-search::backdrop')
        self.assertEqual(dialog["background"], "var(--panel)")
        self.assertIn(".14", dialog["box-shadow"])
        self.assertIn(".20", backdrop["background"])


if __name__ == "__main__":
    unittest.main()
