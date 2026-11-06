"""Reader colour options: every palette stays readable, and the list in the settings matches palettes.css."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
PALETTES = (ROOT / "site/static/palettes.css").read_text()
HUB = (ROOT / "site/static/reader-hub.js").read_text()
TEMPLATE = (ROOT / "site/template.html").read_text()


def luminance(color):
    channels = [int(color[offset:offset + 2], 16) / 255 for offset in (1, 3, 5)]
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in channels]
    return sum(value * weight for value, weight in zip(linear, (.2126, .7152, .0722)))


def contrast(foreground, background):
    lower, higher = sorted((luminance(foreground), luminance(background)))
    return (higher + .05) / (lower + .05)


def tokens(selector):
    match = re.search(re.escape(selector) + r" \{([^{}]*)\}", PALETTES)
    return dict(re.findall(r"--([\w-]+): (#[0-9A-Fa-f]{6});", match[1])) if match else None


KEYS = re.findall(r':root\[data-palette="([a-z]+)"\] \{', PALETTES)


class PaletteTests(unittest.TestCase):
    def test_every_text_colour_reads_on_every_surface(self):
        self.assertGreaterEqual(len(KEYS), 5)
        for key in KEYS:
            for theme in ("", '[data-theme="light"]'):
                values = tokens(f':root[data-palette="{key}"]{theme}')
                self.assertIsNotNone(values, (key, theme))
                for foreground in ("ink", "ink-soft", "muted", "faint", "link", "ember"):
                    for background in ("paper", "panel", "code-bg"):
                        with self.subTest(palette=key, theme=theme or "dark", pair=(foreground, background)):
                            self.assertGreaterEqual(contrast(values[foreground], values[background]), 4.5)
                with self.subTest(palette=key, theme=theme or "dark", pair="body"):
                    self.assertGreaterEqual(contrast(values["ink-soft"], values["paper"]), 7)

    def test_settings_list_matches_the_stylesheet(self):
        listed = re.findall(r"\['([a-z]*)', (?:say\(|')", HUB[HUB.index("const PALETTES"):HUB.index("]];", HUB.index("const PALETTES"))])
        self.assertEqual(listed[0], "")                      # the night sky comes first and is the default
        self.assertEqual(sorted(listed[1:]), sorted(KEYS))
        for option in ["night", *KEYS]:
            with self.subTest(option=option):
                self.assertIn(f'[data-palette-option="{option}"] .reader-hub-swatch', PALETTES)

    def test_settings_ask_app_js_to_apply_and_remember_the_choice(self):
        app = (ROOT / "site/static/app.js").read_text()
        self.assertIn("scope.dispatchEvent(new CustomEvent('notes:palette', {detail: value}))", HUB)
        self.assertIn('window.addEventListener("notes:palette"', app)
        self.assertIn('localStorage.setItem("palette", value)', app)

    def test_choice_is_restored_before_first_paint(self):
        head = TEMPLATE[:TEMPLATE.index("</head>")]
        self.assertIn('localStorage.getItem("palette")', head)
        self.assertIn("/^[a-z]+$/.test(p)", head)
        self.assertLess(head.index("static/style.css"), head.index("static/palettes.css"))


if __name__ == "__main__":
    unittest.main()
