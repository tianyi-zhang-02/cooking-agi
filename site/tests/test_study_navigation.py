import contextlib
import importlib.util
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "recsys_evaluation", ROOT / "practice/recommender-systems/code/evaluation.py")
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class StudyNavigationTests(unittest.TestCase):
    def setUp(self):
        self.old_nav = dict(build.NAV)
        self.old_sources = dict(build.BY_SRC)
        build.NAV.clear()
        build.BY_SRC.clear()
        self.nav = build.load_nav()
        self.pages, self.sections = build.discover(self.nav)
        self.chinese = {page.src.relative_to(ROOT).as_posix(): page
                        for page in self.pages if page.lang == "zh"}

    def tearDown(self):
        build.NAV.clear()
        build.NAV.update(self.old_nav)
        build.BY_SRC.clear()
        build.BY_SRC.update(self.old_sources)

    def test_technical_notes_are_learning_not_career(self):
        technical = [page for source, page in self.chinese.items()
                     if source.startswith("interview/") or source in {
                         "00-foundations/interview-basics.md",
                         "00-foundations/hand-write-kit.md",
                         "00-foundations/ml-math-interview.md"}]
        self.assertGreater(len(technical), 10)
        for page in technical:
            self.assertEqual(build.page_category(page), "learn", str(page.src))
        career = [page for page in self.pages if build.page_category(page) == "career"]
        self.assertTrue(career)
        self.assertTrue(all(page.src.relative_to(ROOT).as_posix().startswith("career/")
                            for page in career))

    def test_all_old_technical_urls_remain(self):
        for source, url in {
            "interview/README.md": "interview/index.html",
            "interview/leetcode.md": "interview/leetcode.html",
            "interview/system-design.md": "interview/system-design.html",
            "interview/python.md": "interview/python.html",
            "00-foundations/hand-write-kit.md": "00-foundations/hand-write-kit.html",
        }.items():
            self.assertEqual(self.chinese[source].url, url)
            self.assertIsNotNone(self.chinese[source].sibling)

    def test_global_navigation_has_one_search_and_no_duplicate_subtabs(self):
        template = (ROOT / "site/template.html").read_text()
        self.assertEqual(template.count("data-search-open"), 1)
        self.assertNotIn("{{subtabs}}", template)
        self.assertNotIn("search-launch", template)
        markup = build.tabs_html(self.chinese["interview/leetcode.md"])
        self.assertIn('href="../learn/index.html" class="active"', markup)
        self.assertIn('href="../practice/index.html"', markup)

    def test_new_series_and_exercises_are_bilingual(self):
        additions = [page for source, page in self.chinese.items()
                     if source.startswith(("learn/", "practice/recommender-systems/"))]
        self.assertGreaterEqual(len(additions), 15)
        for lesson in ("06-why-two-towers", "07-component-choices", "08-serving-lifecycle"):
            self.assertIn(f"practice/recommender-systems/{lesson}.md", self.chinese)
        for page in additions:
            self.assertIsNotNone(page.sibling, str(page.src))
        for source, page in self.chinese.items():
            if source.startswith("practice/recommender-systems/"):
                self.assertEqual(build.page_category(page), "practice")

    def test_includes_have_a_single_owner(self):
        included = [source for section in self.nav["section"] for source in section.get("include", [])]
        self.assertEqual(len(included), len(set(included)))
        urls = [page.url for page in self.pages]
        self.assertEqual(len(urls), len(set(urls)))


class RecommendationLabTests(unittest.TestCase):
    def test_candidate_pool_changes_difficulty(self):
        self.assertEqual(EVALUATION.recall_at_k(["P", "N1", "N2"], {"P"}, 2), 1.0)
        self.assertEqual(EVALUATION.recall_at_k(["N3", "N4", "P"], {"P"}, 2), 0.0)

    def test_recall_denominator_and_empty_labels(self):
        self.assertEqual(EVALUATION.recall_at_k(["P1", "P2"], {"P1", "P2", "P3", "P4"}, 2), .5)
        self.assertIsNone(EVALUATION.recall_at_k(["P"], set(), 2))
        self.assertEqual(EVALUATION.recall_at_k([], {"P"}, 2), 0)
        for invalid in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                EVALUATION.recall_at_k(["P"], {"P"}, invalid)
        with self.assertRaises(ValueError):
            EVALUATION.recall_at_k(["P", "P"], {"P"}, 2)

    def test_bilingual_examples_match_and_run(self):
        directory = ROOT / "practice/recommender-systems"
        for chinese in directory.glob("*.md"):
            if chinese.name.endswith(".en.md"):
                continue
            pattern = r"```python\n(.*?)\n```"
            examples = re.findall(pattern, chinese.read_text(), re.S)
            self.assertEqual(examples, re.findall(
                pattern, chinese.with_suffix(".en.md").read_text(), re.S), str(chinese))
            self.assertNotIn(chr(92) + "`", chinese.read_text(), str(chinese))
            for example in examples:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(example, str(chinese), "exec"), {})
                self.assertTrue(output.getvalue())

    @unittest.skipUnless(shutil.which("node"), "Node required for ranking lab")
    def test_ranking_controls_and_metrics(self):
        script = """
const assert = require('node:assert/strict');
const {candidates, cosine, rerank, summarize} = require(process.argv[1]);
const before = JSON.stringify(candidates);
const baseline = rerank(candidates, 0, false);
assert.deepEqual(baseline.map(item => item.id), ['A', 'B', 'C']);
assert.equal(summarize(baseline).topics, 1);
const varied = rerank(candidates, 1, false);
assert.deepEqual(varied.map(item => item.id), ['A', 'E', 'D']);
assert.equal(summarize(varied).topics, 3);
assert(summarize(varied).ild > summarize(baseline).ild);
assert(summarize(varied).mean < summarize(baseline).mean);
assert.equal(rerank(candidates, 1, false)[0].gain, candidates[0].score);
assert.deepEqual(rerank(candidates, 0, false, 0), []);
assert.deepEqual(summarize([]), {mean: null, topics: 0, ild: null});
assert.equal(summarize([candidates[0]]).ild, null);
for (const penalty of [0, .2, .5, 1]) {
  const chosen = rerank(candidates, penalty, true);
  assert.equal(new Set(chosen.map(item => item.author)).size, chosen.length);
  assert(chosen.every(item => Number.isFinite(item.gain)));
  assert.deepEqual(rerank(candidates, penalty, true), chosen);
}
assert.equal(JSON.stringify(candidates), before);
assert.throws(() => cosine([0, 0], [1, 0]));
assert.throws(() => cosine([1], [1, 2]));
assert.throws(() => rerank(candidates, -1, false));
assert.throws(() => rerank(candidates, 0, false, -1));
"""
        result = subprocess.run([shutil.which("node"), "-e", script,
                                 str(ROOT / "site/static/recsys-lab.js")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
