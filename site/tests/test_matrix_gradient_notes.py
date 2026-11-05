import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import paritycheck
from test_mlsys_runtime_notes import Markup


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "00-foundations/deep-dives"
STEM = "matrix-gradients-to-systems"


class MatrixGradientNotesTests(unittest.TestCase):
    def test_bilingual_structure_derivations_and_code_match(self):
        chinese = (DIRECTORY / f"{STEM}.md").read_text()
        english = (DIRECTORY / f"{STEM}.en.md").read_text()
        self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
        for pattern in (r"\{#([^}]+)\}", r"\$\$(.*?)\$\$", r"```python\n(.*?)```"):
            self.assertEqual(re.findall(pattern, chinese, re.S), re.findall(pattern, english, re.S))
        self.assertNotRegex(re.sub(r"\[中文\]\([^)]*\)", "", english), r"[\u3400-\u9fff]")

    def test_published_finite_difference_script(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "00-foundations/code/matrix_gradient_checks.py")],
            check=True, capture_output=True, text=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 3)
        for case, before, after in zip(report["cases"], (1, 2, 20.6875), (0.25, 1.14, 11.1128125)):
            self.assertAlmostEqual(case["loss_before"], before)
            self.assertAlmostEqual(case["loss_after"], after)
            self.assertLess(case["max_central_difference_error"], 1e-8)
        self.assertEqual(report["cases"][1]["dW"], [[1, 1], [1, 3]])
        self.assertEqual(len(report["cases"][2]["weights"]), 3)
        self.assertEqual(len(report["cases"][2]["weights"][0]), 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_both_published_pytorch_examples_execute(self):
        for suffix in ("", ".en"):
            text = (DIRECTORY / f"{STEM}{suffix}.md").read_text()
            blocks = re.findall(r"```python\n(.*?)```", text, re.S)
            self.assertTrue(blocks)
            for block in blocks:
                with self.subTest(language=suffix or "zh"):
                    exec(compile(block, f"{STEM}{suffix}.md", "exec"), {})

    def test_note_links_resolve_and_keep_language(self):
        for suffix in ("", ".en"):
            source = DIRECTORY / f"{STEM}{suffix}.md"
            for label, href in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", source.read_text()):
                parsed = urlsplit(href)
                if parsed.scheme or parsed.netloc:
                    continue
                target = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
                with self.subTest(source=source.name, href=href):
                    self.assertTrue(target.is_file())
                    if target.suffix == ".md" and label not in ("中文", "English"):
                        self.assertEqual(target.name.endswith(".en.md"), bool(suffix))
                    if parsed.fragment:
                        raw = re.sub(r"^#\s+.+$", "", target.read_text(), count=1, flags=re.M)
                        markup, _ = build.render_markdown(build.protect_mermaid(raw))
                        self.assertIn(unquote(parsed.fragment), Markup(markup).ids)


if __name__ == "__main__":
    unittest.main()
