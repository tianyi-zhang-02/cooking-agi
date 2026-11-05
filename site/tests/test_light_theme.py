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

    def test_light_hero_does_not_keep_the_night_scrim(self):
        self.assertEqual(declarations(STYLE, ':root[data-theme="light"] .hero::before')["display"], "none")
        title = declarations(STYLE, ':root[data-theme="light"] .hero-title')
        self.assertEqual(title["color"], "var(--ink)")
        self.assertEqual(title["text-shadow"], "none")
        self.assertEqual(declarations(STYLE, ':root[data-theme="light"] .hero-kicker')["color"], "var(--muted)")
        self.assertEqual(declarations(STYLE, ':root[data-theme="light"] .sky')["display"], "none")

    def test_hero_text_also_contrasts_with_the_horizon(self):
        tokens = declarations(STYLE, ':root[data-theme="light"]')
        hero = declarations(STYLE, ':root[data-theme="light"] .hero')["background"]
        for color in re.findall(r"#[0-9a-fA-F]{6}", hero):
            for token in ("--ink", "--muted"):
                self.assertGreaterEqual(contrast(tokens[token], color), 4.5)

    def test_dark_theme_keeps_its_existing_palette_and_cover(self):
        tokens = declarations(STYLE, ":root")
        self.assertEqual(tokens["--paper"], "#0C0A08")
        self.assertEqual(tokens["--ink"], "#EFE6DA")
        self.assertEqual(declarations(STYLE, ".hero-title")["color"], "#F7F1E8")
        self.assertIn("rgba(0,0,0,.62)", declarations(STYLE, ".hero::before")["background"])

    def test_search_uses_a_separate_light_overlay(self):
        dialog = declarations(SEARCH, ':root[data-theme="light"] .note-search')
        backdrop = declarations(SEARCH, ':root[data-theme="light"] .note-search::backdrop')
        self.assertEqual(dialog["background"], "var(--panel)")
        self.assertIn(".14", dialog["box-shadow"])
        self.assertIn(".20", backdrop["background"])


if __name__ == "__main__":
    unittest.main()
