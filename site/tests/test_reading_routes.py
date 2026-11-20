from pathlib import Path
import re
import sys
import unittest
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


class ReadingRoutesTests(unittest.TestCase):
    def test_entry_links_resolve_in_both_languages(self):
        for directory in ("learn", "05-post-training/deep-rl"):
            for suffix in ("", ".en"):
                source = ROOT / directory / f"README{suffix}.md"
                rendered, _ = build.render_markdown(source.read_text())
                entry = re.search(r'<nav class="study-route reading-entry"[^>]*>(.*?)</nav>', rendered, re.S)
                with self.subTest(source=source):
                    self.assertIsNotNone(entry)
                    links = re.findall(r'href="([^"]+)"', entry[1])
                    self.assertEqual(len(links), 3)
                    for link in links:
                        target = urlsplit(link)
                        if target.path:
                            destination = source.parent / target.path
                            self.assertTrue(destination.is_file(), link)
                            self.assertEqual(destination.name.endswith(".en.md"), bool(suffix))
                            destination_html, _ = build.render_markdown(destination.read_text())
                        else:
                            destination_html = rendered
                        if target.fragment:
                            self.assertIn(f'id="{target.fragment}"', destination_html)

    def test_existing_overview_anchors_are_preserved(self):
        expected = {
            "learn/README.md": {"_1", "_2", "_3", "_4"},
            "learn/README.en.md": {"study-notes", "how-the-sections-fit-together", "start-with-the-foundations", "follow-the-question-you-have", "all-chapters", "reading-and-updates"},
            "05-post-training/deep-rl/README.md": {"deep-rl", "_1", "_2", "3", "_3", "_4", "_5", "_6", "reward-review", "assumption-review"},
            "05-post-training/deep-rl/README.en.md": {"deep-rl-from-one-decision-to-a-training-loop", "a-route-through-the-material", "different-reading-goals-not-separate-sets-of-notes", "find-these-3-parts-in-any-algorithm", "prerequisites-and-notation", "change-one-assumption-at-a-time", "start-with-an-experiment-if-you-prefer", "public-references", "reward-review", "assumption-review"},
        }
        for filename, anchors in expected.items():
            rendered, _ = build.render_markdown((ROOT / filename).read_text())
            with self.subTest(filename=filename):
                self.assertFalse(anchors - set(re.findall(r'id="([^"]+)"', rendered)))

    def test_update_history_follows_the_reading_routes(self):
        for suffix in ("", ".en"):
            source = (ROOT / "learn" / f"README{suffix}.md").read_text()
            history = re.search(r'<details markdown="1">\s*<summary>[^<]*2026[^<]*</summary>(.*?)</details>', source, re.S)
            with self.subTest(suffix=suffix):
                self.assertIsNotNone(history)
                self.assertGreater(history.start(), source.index("<!-- widget:study-atlas -->"))
                for topic in ("scaling", "diffusion-lm", "quantization", "ope-estimators"):
                    self.assertIn(topic, history[1])

    def test_review_examples_remain_in_the_page_body(self):
        for suffix in ("", ".en"):
            source = (ROOT / "05-post-training/deep-rl" / f"README{suffix}.md").read_text()
            rendered, _ = build.render_markdown(source)
            with self.subTest(suffix=suffix):
                self.assertNotIn("<details", rendered)
                for anchor in ("#why-leave-one-out", "#gradient", "#js", "#pairs-versus-ranking", "#alignment", "#reduction", "#combo-example", "#ranking-versus-selection", "#finite-exploration-evidence"):
                    self.assertIn(anchor, rendered)
                self.assertIn("$1.6-10/7$", source)
                self.assertIn("$5/3$", source)


if __name__ == "__main__":
    unittest.main()
