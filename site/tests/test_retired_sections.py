import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import collaboration


class RetiredSectionsTests(unittest.TestCase):
    def test_discussions_are_not_published_or_assigned_reviewers(self):
        nav = build.load_nav()
        config = collaboration.load_config()
        collaboration.validate(config, nav)
        for kind in ("category", "group"):
            self.assertNotIn("discussions", {entry["id"] for entry in nav[kind]})
        self.assertNotIn("discussions", {entry["dir"] for entry in nav["section"]})
        self.assertNotIn("discussions", {entry["id"] for entry in config["area"]})
        self.assertFalse(any(path.startswith("discussions/") for path in nav.get("label", {})))
        self.assertFalse(list((build.ROOT / "discussions").glob("*.md")))
        for filename in ("discussions.py", "discussions.toml", "DISCUSSIONS.md",
                         "static/discussions.js", "static/discussions.css"):
            self.assertFalse((build.SITE / filename).exists())

    def test_career_section_is_retired_and_old_urls_go_home(self):
        # The career (求职) notes were removed on 2026-11-05.
        nav = build.load_nav()
        config = collaboration.load_config()
        collaboration.validate(config, nav)
        for kind in ("category", "group"):
            self.assertNotIn("career", {entry["id"] for entry in nav[kind]})
        self.assertNotIn("career", {entry["dir"] for entry in nav["section"]})
        self.assertNotIn("career", {entry["id"] for entry in config["area"]})
        self.assertFalse(any(path.startswith("career/") for path in nav.get("label", {})))
        self.assertFalse(list((build.ROOT / "career").glob("*.md")))
        self.assertFalse((build.SITE / "static/career.css").exists())
        published = json.loads((build.SITE / "catalog-baseline.json").read_text())
        redirects = {old: target for old, target in nav["redirects"].items() if old.startswith("career/")}
        self.assertEqual(set(redirects), {url for url in published if url.startswith("career/")})
        for old, target in redirects.items():
            self.assertEqual(target, "index.html" if old.endswith(".en.html") else "index.zh.html", old)

    def test_bilingual_navigation_keeps_learning_and_collaboration(self):
        nav = build.load_nav()
        previous_sources = dict(build.BY_SRC)
        try:
            pages, sections = build.discover(nav)
            known = {page.url for page in pages}
            self.assertFalse(any(url.startswith("discussions/") for url in known))
            template = (build.SITE / "template.html").read_text()
            selected = {"learn/index.html", "learn/index.en.html",
                        "05-post-training/rlhf/ppo-clipping.html",
                        "05-post-training/rlhf/ppo-clipping.en.html"}
            self.assertTrue(selected.issubset(known))
            self.assertFalse(any(url.startswith("quant/") for url in known))
            self.assertFalse(any(url.startswith("career/") for url in known))
            for page in pages:
                if page.url not in selected:
                    continue
                with self.subTest(url=page.url):
                    build.build_page(page, [], nav["site"]["repo"], known)
                    output = build.assemble(page, sections, nav, "test", template)
                    for marker in ("discussions/", "discussions.js", "discussions.css",
                                   "data-discussion-widget", "page-talk"):
                        self.assertNotIn(marker, output)
                    self.assertIn(page.body, output)
                    self.assertIn('class="lang-btn"', output)
                    self.assertIn('id="note-search"', output)
                    self.assertIn(collaboration.page_panel(page, collaboration.load_config(),
                                                          nav["site"]["repo"]), output)
        finally:
            build.BY_SRC.clear()
            build.BY_SRC.update(previous_sources)


if __name__ == "__main__":
    unittest.main()
