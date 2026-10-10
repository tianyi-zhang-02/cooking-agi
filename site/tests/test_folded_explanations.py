from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


class FoldedContent(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            self.depth += 1
        if self.depth:
            self.tags.append(tag)

    def handle_endtag(self, tag):
        if tag == "details":
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.text.append(data)


class FoldedExplanationTests(unittest.TestCase):
    def test_bilingual_derivations_render_markdown_inside_details(self):
        for chapter in ("core/bert", "deep-dives/nope-and-order"):
            for suffix in ("", ".en"):
                with self.subTest(chapter=chapter, language=suffix):
                    source = ROOT / f"00-foundations/{chapter}{suffix}.md"
                    body, _ = build.render_markdown(source.read_text())
                    folded = FoldedContent()
                    folded.feed(body)
                    self.assertEqual(folded.tags.count("details"), 6 if chapter == "core/bert" else 3)
                    self.assertGreater(folded.tags.count("p"), 3)
                    self.assertNotRegex("".join(folded.text), r"\[[^\]\n]+\]\([^\n)]+\)")
                    if chapter == "core/bert":
                        self.assertEqual(folded.tags.count("table"), 2)
                        self.assertIn("code", folded.tags)
                        self.assertIn("strong", folded.tags)

    def test_only_html_details_may_omit_markdown_attribute(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            pages, _ = build.discover(build.load_nav())
        for page in pages:
            source = page.src.read_text()
            for matched in re.finditer(r"<details\b([^>]*)>(.*?)</details>", source, re.S):
                if 'markdown=' in matched.group(1):
                    continue
                content = re.sub(r"<summary\b[^>]*>.*?</summary>", "", matched.group(2), flags=re.S)
                self.assertNotRegex(content, r"\[[^\]\n]+\]\([^\n)]+\)|\*\*[^*]+\*\*|\|\s*---", str(page.src))


if __name__ == "__main__":
    unittest.main()
