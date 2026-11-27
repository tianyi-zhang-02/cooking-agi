from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


class ReaderHubTests(unittest.TestCase):
    def test_counting_summary_and_store_validation(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable")
        subprocess.run([node, "--test", str(ROOT / "site/tests/reader-hub.test.cjs")], check=True, capture_output=True)

    def test_settings_and_storage_failure_messages(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable")
        subprocess.run([node, "--test", str(ROOT / "site/tests/reader-hub-dom.test.cjs")], check=True, capture_output=True)

    def test_reading_workspace_and_legacy_backup_controls(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable")
        subprocess.run([node, "--test", str(ROOT / "site/tests/reader-hub-workspace.test.cjs")], check=True, capture_output=True)

    def test_reads_count_on_the_note_the_page_shows(self):
        # The counter reads the note's own tools row; sidebar links carry other notes' IDs.
        script = (ROOT / "site/static/reader-hub.js").read_text()
        self.assertIn("document.querySelector('.note-tools[data-progress-id]')?.dataset.progressId", script)
        self.assertNotIn("[data-prep-note]", script)
        built = ROOT / "_site/00-foundations/deep-dives/kv-cache-and-inference.en.html"
        if built.exists():
            self.assertRegex(built.read_text(), r'<div class="note-tools"[^>]*data-progress-id="note-[a-f0-9]{16}"')

    def test_my_reading_is_a_page_in_both_languages(self):
        nav = build.load_nav()
        pages, sections = build.discover(nav)
        known = {page.url for page in pages}
        for language, url, title in (("zh", "my-reading.html", "我的阅读"), ("en", "my-reading.en.html", "My reading")):
            with self.subTest(language=language):
                page = build.BY_SRC["my-reading.md"][language]
                self.assertEqual((page.url, page.title, page.kind), (url, title, "reader"))
                build.build_page(page, [], nav["site"]["repo"], known)
                output = build.assemble(page, sections, nav, "test", (build.SITE / "template.html").read_text())
                self.assertIn('class="reader-page" data-reader-page data-tracks="', output)
                self.assertIn("<noscript>", output)
                self.assertNotIn('class="note-tools"', output)   # no "Save this page" on the page of saved pages
                self.assertNotIn("min read", output)
                self.assertNotIn("分钟", output.split('<div class="article-meta">', 1)[1].split("</div>", 1)[0])

    def test_reader_page_shares_the_drawer_colours(self):
        styles = (ROOT / "site/static/reader-hub.css").read_text()
        palettes = (ROOT / "site/static/palettes.css").read_text()
        self.assertIn(":is(.reader-hub, .reader-page) {", styles)
        self.assertNotIn(" .reader-hub {", palettes)
        self.assertIn("Filled stops are ones you have opened in this browser, not ones you have finished.",
                      (ROOT / "site/static/reader-hub.js").read_text())

    def test_drawer_layout_matches_header_and_keeps_quiet_statistics(self):
        styles = (ROOT / "site/static/reader-hub.css").read_text()
        self.assertIn("top: 88px", styles)
        self.assertIn("top: 64px", styles)
        self.assertIn("@media (max-width: 1080px)", styles)
        self.assertIn(".reader-hub-label { display: none; }", styles)
        self.assertIn(".reader-hub summary:focus-visible", styles)
        self.assertIn("@media (prefers-reduced-motion: reduce)", styles)
        for effect in ("drop-shadow", "text-shadow", "--rh-glow", "backdrop-filter"):
            self.assertNotIn(effect, styles)

    def test_script_stays_local(self):
        script = (ROOT / "site/static/reader-hub.js").read_text()
        for forbidden in ("innerHTML", "sessionStorage", "XMLHttpRequest", "eval("):
            self.assertNotIn(forbidden, script)
        self.assertNotIn("localStorage", script)  # all writes go through the shared progress controller
        self.assertEqual(script.count("fetch("), 1)  # only the public note catalog
        self.assertIn("static/prep-progress-catalog.json", script)

    def test_every_page_loads_the_hub_after_the_progress_controller(self):
        source = (ROOT / "site/build.py").read_text()
        controller = source.index("static/prep-progress.js?v=")
        self.assertLess(controller, source.index("static/reader-hub.js?v="))
        self.assertLess(controller, source.index("static/reader-hub.css?v="))


if __name__ == "__main__":
    unittest.main()
