import importlib.util
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck
import prep_progress
from test_mlsys_runtime_notes import Markup, diagram_edges


DIRECTORY = ROOT / "practice" / "runtime-reading"
CHAPTERS = ("README", "prefill-decode", "nemo-rl", "training-stability")
SPEC = importlib.util.spec_from_file_location("runtime_checks", DIRECTORY / "code" / "runtime_checks.py")
EXAMPLES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLES)


class RuntimeSourceReadingTests(unittest.TestCase):
    def test_bilingual_shape_math_and_no_chinese_in_english_body(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                chinese = (DIRECTORY / f"{chapter}.md").read_text()
                english = (DIRECTORY / f"{chapter}.en.md").read_text()
                self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
                self.assertEqual(re.findall(r"\{#([^}]+)\}", chinese), re.findall(r"\{#([^}]+)\}", english))
                self.assertEqual(re.findall(r"\$\$(.*?)\$\$", chinese, re.S), re.findall(r"\$\$(.*?)\$\$", english, re.S))
                self.assertEqual(diagram_edges(chinese), diagram_edges(english))
                self.assertNotRegex(re.sub(r"\[中文\]\([^)]*\)", "", english), r"[\u3400-\u9fff]")

    def test_links_preserve_language_and_resolve_anchors(self):
        for chapter in CHAPTERS:
            for suffix in ("", ".en"):
                for label, href in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", (DIRECTORY / f"{chapter}{suffix}.md").read_text()):
                    parsed = urlsplit(href)
                    if parsed.scheme or parsed.netloc:
                        continue
                    target = (DIRECTORY / unquote(parsed.path)).resolve()
                    with self.subTest(chapter=chapter, suffix=suffix, href=href):
                        self.assertTrue(target.is_file())
                        if target.suffix == ".md" and label not in ("中文", "English"):
                            self.assertEqual(target.name.endswith(".en.md"), suffix == ".en")
                        if parsed.fragment:
                            raw = re.sub(r"^#\s+.+$", "", target.read_text(), count=1, flags=re.M)
                            markup, _ = build.render_markdown(build.protect_mermaid(raw))
                            self.assertIn(unquote(parsed.fragment), Markup(markup).ids)

    def test_navigation_and_reading_ids(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            pages, sections = build.discover(build.load_nav())
            section = next(item for item in sections if item["dir"] == "practice/runtime-reading")
            self.assertEqual(section["group"], "ml-systems")
            self.assertEqual([pair["zh"].src.stem for pair in section["pages"]], list(CHAPTERS))
            catalog = prep_progress.catalog(build.BY_SRC)["notes"]
            for chapter in CHAPTERS:
                source = f"practice/runtime-reading/{chapter}.md"
                note = catalog[prep_progress.identifier(source)]
                stem = "index" if chapter == "README" else chapter
                self.assertEqual(note["pages"]["zh"]["url"], f"practice/runtime-reading/{stem}.html")
                self.assertEqual(note["pages"]["en"]["url"], f"practice/runtime-reading/{stem}.en.html")
                self.assertEqual(sum(page.src == DIRECTORY / f"{chapter}.md" for page in pages), 1)

    def test_source_links_are_pinned_not_floating_main(self):
        for source in DIRECTORY.glob("*.md"):
            for href in re.findall(r"https://github\.com/[^\s)]+", source.read_text()):
                self.assertRegex(href, r"/(?:blob|tree)/[a-f0-9]{40}(?:/|$)")

    def test_cached_attention_matches_causal_full_pass(self):
        full = EXAMPLES.full_attention(EXAMPLES.TOKENS)
        for boundary in range(1, len(EXAMPLES.TOKENS)):
            chunks = [EXAMPLES.TOKENS[:boundary], EXAMPLES.TOKENS[boundary:]]
            self.assertLess(EXAMPLES.max_difference(full, EXAMPLES.cached_attention(chunks)), 1e-12)

    def test_removing_causal_mask_is_detected(self):
        self.assertGreater(EXAMPLES.max_difference(EXAMPLES.full_attention(EXAMPLES.TOKENS), EXAMPLES.full_attention(EXAMPLES.TOKENS, causal=False)), 1e-3)

    def test_normalization_and_distributed_scaling(self):
        token_mean, batch_mean, reduced, direct = EXAMPLES.normalization_example()
        self.assertAlmostEqual(token_mean, 13 / 11)
        self.assertEqual(batch_mean, 2)
        self.assertAlmostEqual(reduced, direct)
        self.assertNotAlmostEqual(token_mean, batch_mean)

    def test_demonstrations_execute_without_accelerator_dependencies(self):
        report = EXAMPLES.run_checks()
        self.assertEqual(report["kv_per_token_bytes"], 131072)
        self.assertEqual(report["sixteen_requests_kv_gib"], 8)

    def test_response_mask_includes_first_answer_prediction(self):
        rows = EXAMPLES.response_alignment_example()
        self.assertEqual([row for row in rows if row[2]], [("D", "E", 1), ("E", "F", 1), ("F", "G", 1)])
        self.assertEqual([row[2] for row in rows], [0, 0, 0, 1, 1, 1])

    def test_generation_timing_separates_first_token_and_intervals(self):
        arrivals = EXAMPLES.generation_timing_example()
        self.assertEqual(arrivals[0], 100)
        self.assertEqual([later - earlier for earlier, later in zip(arrivals, arrivals[1:])], [20, 20])
        self.assertEqual(arrivals[-1], 140)

    def test_norm_clipping_preserves_direction(self):
        horizontal, vertical = EXAMPLES.norm_clipping_example()
        self.assertAlmostEqual(horizontal, 0.6)
        self.assertAlmostEqual(vertical, 0.8)
        self.assertAlmostEqual(horizontal / vertical, 3 / 4)
        self.assertAlmostEqual(horizontal**2 + vertical**2, 1)

    def test_reward_centering_keeps_questions_separate(self):
        groups = EXAMPLES.grouped_rewards_example()
        self.assertEqual(groups[0], [0.5, -0.5, 0.5, -0.5])
        self.assertEqual(groups[1], [0, 0, 0, 0])
        self.assertNotEqual(groups[1][0], 1 - 6 / 8)


if __name__ == "__main__":
    unittest.main()
