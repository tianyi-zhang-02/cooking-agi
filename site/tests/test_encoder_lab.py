from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


class LabMarkup(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.labs = []
        self.tables = 0
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "data-encoder-lab" in attrs:
            self.labs.append(attrs)
        if tag == "table":
            self.tables += 1


class EncoderLabTests(unittest.TestCase):
    def test_numeric_and_visibility_contracts(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable")
        result = subprocess.run([node, str(ROOT / "site/tests/encoder-lab-behavior.cjs")],
                                capture_output=True, text=True, check=True)
        self.assertIn("invalid inputs pass", result.stdout)

    def test_bilingual_pages_have_readable_fallbacks(self):
        for chapter, kind in [("bert", "mlm"), ("vanilla-transformer", "attention")]:
            for suffix, language in [("", "zh"), (".en", "en")]:
                with self.subTest(chapter=chapter, language=language):
                    source = (ROOT / f"00-foundations/core/{chapter}{suffix}.md").read_text()
                    body, _ = build.render_markdown(source)
                    markup = LabMarkup(body)
                    self.assertEqual(len(markup.labs), 1)
                    self.assertEqual(markup.labs[0]["data-encoder-lab"], kind)
                    self.assertEqual(markup.labs[0]["data-lang"], language)
                    self.assertGreater(markup.tables, 1)
                    self.assertIn("JavaScript", body)
                    self.assertIn("0.805" if kind == "mlm" else "[BOS]", body)

    def test_assets_only_load_on_pages_with_a_lab(self):
        old_nav, old_sources = build.NAV.copy(), build.BY_SRC.copy()
        try:
            nav = build.load_nav()
            pages, sections = build.discover(nav)
            known = {page.url for page in pages}
            for source, expected in [("00-foundations/core/bert.md", True),
                                     ("00-foundations/core/vanilla-transformer.md", True),
                                     ("career/journey.md", False)]:
                for language in ["zh", "en"]:
                    page = build.BY_SRC[source][language]
                    build.build_page(page, [], nav["site"]["repo"], known)
                    output = build.assemble(page, sections, [], nav, "test",
                                            (ROOT / "site/template.html").read_text())
                    self.assertEqual("static/encoder-lab.js" in output, expected)
                    self.assertEqual("static/encoder-lab.css" in output, expected)
        finally:
            build.NAV.clear()
            build.NAV.update(old_nav)
            build.BY_SRC.clear()
            build.BY_SRC.update(old_sources)


if __name__ == "__main__":
    unittest.main()
