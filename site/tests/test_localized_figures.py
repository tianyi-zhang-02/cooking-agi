from html.parser import HTMLParser
from pathlib import Path
import hashlib
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "00-foundations/code"))
sys.path.insert(0, str(ROOT / "site"))
import build
from figure_languages import TITLES, localized_svg, translated_label
from svgkit import write


NAMESPACE = {"svg": "http://www.w3.org/2000/svg"}


class MethodMapMarkup(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.regions = []
        self.cards = 0
        self.feedback = 0
        self.labels = []
        self.feed(body)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("class") == "method-map":
            self.regions.append(attrs)
        if attrs.get("class") == "method-map-card":
            self.cards += 1
        if attrs.get("class") == "method-map-feedback":
            self.feedback += 1
        if "id" in attrs:
            self.labels.append(attrs["id"])


class LocalizedFigureTests(unittest.TestCase):
    def test_localized_assets_have_matching_geometry_and_readable_labels(self):
        for stem in TITLES:
            with self.subTest(figure=stem):
                roots = [ET.parse(ROOT / f"00-foundations/assets/{stem}{suffix}.svg").getroot()
                         for suffix in ("", ".en")]
                chinese, english = roots
                self.assertEqual(chinese.get("lang"), "zh-CN")
                self.assertEqual(english.get("lang"), "en")
                self.assertEqual(chinese.get("viewBox"), english.get("viewBox"))
                for root in roots:
                    title = root.find("svg:title", NAMESPACE)
                    self.assertTrue(title.text)
                    self.assertEqual(root.get("aria-labelledby"), title.get("id"))
                geometry = lambda root: [(node.tag, node.attrib) for node in root.iter()
                                         if node.tag.rsplit("}", 1)[-1] in
                                         {"path", "polyline", "rect", "circle", "line"}]
                self.assertEqual(geometry(chinese), geometry(english))
                zh_labels = [node.text or "" for node in chinese.findall("svg:text", NAMESPACE)]
                en_labels = [node.text or "" for node in english.findall("svg:text", NAMESPACE)]
                self.assertEqual(len(zh_labels), len(en_labels))
                self.assertRegex(" ".join(zh_labels), r"[\u4e00-\u9fff]")
                self.assertNotRegex(" ".join(en_labels), r"[\u4e00-\u9fff]")
                for zh_label, en_label in zip(zh_labels, en_labels):
                    self.assertEqual(zh_label, translated_label(en_label, "zh"))
                    if not re.search(r"[\u4e00-\u9fff]", zh_label):
                        self.assertNotRegex(zh_label, r"[A-Za-z]{3,}\s+[A-Za-z]{3,}",
                                            "Untranslated prose in a Chinese diagram")

    def test_article_images_match_the_page_language(self):
        references = 0
        for section in ("00-foundations", "05-post-training"):
            for source in (ROOT / section).rglob("*.md"):
                for relative in re.findall(r"!\[[^\]]*\]\(([^)\s]+\.svg)\)", source.read_text()):
                    if "://" in relative:
                        continue
                    path = (source.parent / relative).resolve()
                    root = ET.parse(path).getroot()
                    labels = " ".join(node.text or "" for node in root.findall("svg:text", NAMESPACE))
                    if path.name == "order-state.svg":
                        self.assertNotRegex(labels, r"[A-Za-z\u4e00-\u9fff]")
                        continue
                    references += 1
                    if source.name.endswith(".en.md"):
                        self.assertTrue(path.name.endswith(".en.svg"), str(source))
                        self.assertNotRegex(labels, r"[\u4e00-\u9fff]", str(source))
                    else:
                        self.assertFalse(path.name.endswith(".en.svg"), str(source))
                        self.assertRegex(labels, r"[\u4e00-\u9fff]", str(source))
        self.assertGreaterEqual(references, 20)

    def test_generator_keeps_both_languages(self):
        original = (ROOT / "00-foundations/assets/attention-sites.en.svg").read_text()
        with tempfile.TemporaryDirectory() as directory:
            write(directory, "attention-sites.svg", original)
            for suffix, language in [("", "zh-CN"), (".en", "en")]:
                root = ET.parse(Path(directory) / f"attention-sites{suffix}.svg").getroot()
                self.assertEqual(root.get("lang"), language)
                self.assertEqual(len(root.findall("svg:title", NAMESPACE)), 1)
        for language in ("fr", "", None):
            with self.assertRaises(ValueError):
                localized_svg("attention-sites.svg", original, language)

    def test_numeric_results_are_not_replaced_by_fixed_translation_values(self):
        for label, expected in [
            ("33 params · 97% accuracy", "33 个参数 · 准确率 97%"),
            ("This initialization: plain-stack gradient shrinks 2e+03× from layer 20 to layer 1.",
             "这次初始化中，无残差网络的梯度从第 20 层传到第 1 层，缩小约 2e+03 倍。"),
        ]:
            self.assertEqual(translated_label(label, "zh"), expected)

    def test_comparison_is_localized_readable_without_javascript(self):
        for suffix, language in [("", "zh-CN"), (".en", "en")]:
            with self.subTest(language=language):
                source = (ROOT / f"05-post-training/rlhf/after-rlhf{suffix}.md").read_text()
                body, _ = build.render_markdown(source)
                markup = MethodMapMarkup(body)
                self.assertEqual(len(markup.regions), 1)
                self.assertEqual(markup.regions[0]["lang"], language)
                self.assertEqual(markup.regions[0]["id"], "method-comparison")
                self.assertIn(markup.regions[0]["aria-labelledby"], markup.labels)
                self.assertEqual(markup.cards, 3)
                self.assertEqual(markup.feedback, 1)
                region = re.search(r'<section class="method-map".*?</section>', body, re.S).group()
                self.assertNotIn("<img", region)
                self.assertNotIn("<script", region)
                self.assertNotIn("<button", region)
                self.assertEqual(region.count("<dt>"), 6)
                self.assertEqual(region.count("<dd>"), 6)
                if language == "en":
                    self.assertNotRegex(region, r"[\u4e00-\u9fff]")
                else:
                    self.assertIn("模型怎么学", region)
                    self.assertNotIn("How does", region)

    def test_comparison_style_is_loaded_only_where_needed(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            nav = build.load_nav()
            pages, sections = build.discover(nav)
            known = {page.url for page in pages}
            template = (ROOT / "site/template.html").read_text()
            for source, expected in [("05-post-training/rlhf/after-rlhf.md", True),
                                     ("00-foundations/core/bert.md", False)]:
                for language in ("zh", "en"):
                    with self.subTest(source=source, language=language):
                        page = build.BY_SRC[source][language]
                        build.build_page(page, [], nav["site"]["repo"], known)
                        output = build.assemble(page, sections, [], nav, "test", template)
                        self.assertEqual("static/method-map.css" in output, expected)

    def test_local_diagram_versions_follow_content_not_build_time(self):
        page = SimpleNamespace(lang="zh", out_rel=Path("00-foundations/core/normalization.html"))
        image = '<img src="../assets/norm-axes.svg?size=large&amp;v=old#figure-title">'
        digest = hashlib.sha256((ROOT / "00-foundations/assets/norm-axes.svg").read_bytes()).hexdigest()[:12]
        expected = f'<img src="../assets/norm-axes.svg?size=large&amp;v={digest}#figure-title">'
        self.assertEqual(build.rewrite_links(image, page, "owner/repo", set()), expected)
        self.assertEqual(build.rewrite_links(expected, page, "owner/repo", set()), expected)
        for unchanged in ['<img src="https://example.com/assets/plot.svg">',
                          '<img src="../assets/missing.svg">',
                          '<a href="../assets/norm-axes.svg">',
                          '<img src="../assets/photo.png">']:
            self.assertEqual(build.rewrite_links(unchanged, page, "owner/repo", set()), unchanged)


if __name__ == "__main__":
    unittest.main()
