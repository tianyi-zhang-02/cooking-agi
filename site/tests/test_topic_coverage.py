from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import topic_coverage

CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")


class TopicCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            cls.nav = build.load_nav()
            cls.pages, cls.sections = build.discover(cls.nav)
            cls.by_src = dict(build.BY_SRC)

    def data(self, reviewed=()):
        return topic_coverage.rows(self.nav, self.sections, build.ROOT, set(reviewed))

    def test_counts_every_topic_note_once_and_skips_section_homes(self):
        data = self.data()
        groups = {group["id"]: group for group in self.nav["group"]}
        expected = 0
        for section in self.sections:
            group = groups.get(section["group"], {})
            if group.get("category") in topic_coverage.CATEGORIES and group["id"] not in topic_coverage.SKIP_GROUPS:
                expected += sum(1 for pair in section["pages"] if (pair.get("zh") or pair.get("en")).kind not in {"index", "home"})
        self.assertEqual(topic_coverage.totals(data), (expected, 0))
        self.assertEqual([labels for labels, *_ in data[-1][1]], list(topic_coverage.PROPOSED))
        self.assertTrue(all(notes == 0 for _, notes, _ in data[-1][1]))
        self.assertTrue(all(notes > 0 for _, topics in data[:-1] for _, notes, _ in topics))

    def test_reviewed_notes_count_toward_their_topic(self):
        source = "00-foundations/core/multi-head-attention.md"
        data = self.data({source})
        self.assertEqual(topic_coverage.totals(data)[1], 1)
        reviewed = [(labels[1], done) for _, topics in data for labels, _, done in topics if done]
        self.assertEqual(reviewed, [("Foundations", 1)])

    def test_review_list_rejects_unknown_and_repeated_notes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "expression-review.toml"
            path.write_text('pass = "2026-10"\nreviewed = ["00-foundations/core/bert.md"]\n')
            self.assertEqual(topic_coverage.load_reviewed(path, self.by_src), ("2026-10", {"00-foundations/core/bert.md"}))
            path.write_text('reviewed = ["missing.md"]\n')
            with self.assertRaisesRegex(SystemExit, "missing.md"):
                topic_coverage.load_reviewed(path, self.by_src)
            path.write_text('reviewed = ["00-foundations/core/bert.md", "00-foundations/core/bert.md"]\n')
            with self.assertRaisesRegex(SystemExit, "twice"):
                topic_coverage.load_reviewed(path, self.by_src)
        self.assertEqual(topic_coverage.load_reviewed(build.SITE / "expression-review.toml", self.by_src)[0], "2026-10")

    def test_svg_is_valid_and_english_has_no_chinese(self):
        data = self.data({"00-foundations/core/bert.md"})
        for language in ("zh", "en"):
            for palette in topic_coverage.PALETTES:
                markup = topic_coverage.svg(data, language, palette, "2026-10-14")
                root = ElementTree.fromstring(markup)
                self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
                self.assertIn(topic_coverage.PALETTES[palette]["bar"], markup)
                if language == "en":
                    self.assertIsNone(CJK.search(markup))
                    self.assertIn("1 of", markup)

    def test_site_figure_scales_bars_and_english_has_no_chinese(self):
        data = self.data()
        for language in ("zh", "en"):
            markup = topic_coverage.figure(data, language, "learn/coverage.html")
            widths = [float(value) for value in re.findall(r'width:([0-9.]+)%', markup)]
            self.assertAlmostEqual(max(widths), 100.0)
            self.assertTrue(all(0 <= width <= 100 for width in widths))
            self.assertIn('href="learn/coverage.html"', markup)
            if language == "en":
                self.assertIsNone(CJK.search(markup))

    def test_readmes_link_merged_pull_requests(self):
        for filename, language in (("README.zh.md", "zh"), ("README.md", "en")):
            source = (build.ROOT / filename).read_text()
            self.assertIn("## 更新记录" if language == "zh" else "## Changelog", source)
            numbers = re.findall(r"\[#(\d+)\]\(https://github\.com/tianyi-zhang-02/cooking-agi/pull/(\d+)\)", source)
            self.assertTrue(numbers)
            self.assertTrue(all(label == target for label, target in numbers))
            if language == "en":
                self.assertIsNone(CJK.search(source.replace("中文", "")))


if __name__ == "__main__":
    unittest.main()
