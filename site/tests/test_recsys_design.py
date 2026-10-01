from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paritycheck


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "site/static/recsys-design.js"


class RecsysDesignTests(unittest.TestCase):
    def run_js(self, script):
        if not shutil.which("node"):
            self.skipTest("Node.js is required for the interactive experiments")
        result = subprocess.run(
            [shutil.which("node"), "-e", "const assert = require('node:assert/strict');\n"
             "const lab = require(process.argv[1]);\n" + script, str(SCRIPT)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_architecture_has_complete_bilingual_module_contracts(self):
        self.run_js("""
const visited = new Set();
for (const lane of Object.values(lab.lanes)) {
  assert.equal(lane.label.length, 2);
  assert.equal(lane.nodes.length, new Set(lane.nodes).size);
  for (const identifier of lane.nodes) {
    visited.add(identifier);
    const stage = lab.stages[identifier];
    assert(stage);
    for (const field of ['label', 'input', 'output', 'why', 'options', 'check']) {
      assert.equal(stage[field].length, 2);
      assert(stage[field].every(value => typeof value === 'string' && value.length > 0));
    }
    assert.match(stage.lesson, /^0[1-8]-[a-z-]+$/);
  }
}
assert.equal(visited.size, Object.keys(lab.stages).length);
""")

    def test_trace_preserves_deletion_and_failure_semantics(self):
        self.run_js("""
const normal = lab.traceRequest('normal');
assert.deepEqual(normal.merged, ['A', 'B', 'C', 'D']);
assert.deepEqual(normal.returned, ['C', 'D']);
assert.deepEqual(normal.removed, ['B']);
const timeout = lab.traceRequest('timeout');
const mismatch = lab.traceRequest('mismatch');
assert.deepEqual(timeout.returned, ['A']);
assert.deepEqual(mismatch.returned, timeout.returned);
assert.notEqual(timeout.denseStatus, mismatch.denseStatus);
for (const scenario of Object.keys(lab.scenarios)) {
  const trace = lab.traceRequest(scenario);
  assert(!trace.returned.includes('B'));
  assert.equal(trace.returned.length, new Set(trace.returned).size);
  assert(trace.returned.length <= 2);
}
const unseen = lab.traceRequest('unseen');
assert.deepEqual(unseen.returned, normal.returned);
assert.deepEqual(unseen.exposed, []);
assert.throws(() => lab.traceRequest('unknown'));
""")

    def test_vectors_match_hand_check_and_budget(self):
        self.run_js("""
assert.deepEqual(lab.vectorSearch(.5, 4, 'mixed').unique, ['C', 'D', 'E', 'B']);
assert.deepEqual(lab.vectorSearch(.5, 4, 'split').unique, ['A', 'B', 'G', 'F']);
assert.equal(lab.vectorSearch(1, 8, 'split').duplicates, 1);
const before = JSON.stringify(lab.vectorItems);
for (const mode of ['mixed', 'split']) {
  for (const weight of [0, .2, .5, .8, 1]) {
    for (const budget of [2, 4, 8]) {
      const result = lab.vectorSearch(weight, budget, mode);
      assert.equal(result.results.flat().length, budget);
      assert.equal(result.queries.reduce((sum, query) => sum + query.limit, 0), budget);
      assert.equal(result.duplicates, budget - result.unique.length);
      assert.equal(result.comparisons, mode === 'mixed' ? 8 : 16);
      assert.equal(new Set(result.unique).size, result.unique.length);
      assert(result.results.flat().every(item => Number.isFinite(item.score)));
      for (const query of result.queries) assert(Math.abs(Math.hypot(...query.vector) - 1) < 1e-12);
      assert.deepEqual(lab.vectorSearch(weight, budget, mode), result);
    }
  }
}
assert.equal(JSON.stringify(lab.vectorItems), before);
for (const args of [[-1,4,'mixed'], [NaN,4,'mixed'], [1,1,'split'], [.5,4.5,'mixed'], [.5,4,'none']]) {
  assert.throws(() => lab.vectorSearch(...args));
}
""")

    def test_evaluation_distinguishes_pool_definition_and_missing_labels(self):
        self.run_js("""
assert.equal(lab.evaluateRecall('small', 2, 'standard').score, 1);
assert.equal(lab.evaluateRecall('larger', 2, 'standard').score, 0);
assert.equal(lab.evaluateRecall('multi', 2, 'standard').score, .5);
assert.equal(lab.evaluateRecall('multi', 2, 'capped').score, 1);
for (const denominator of ['standard', 'capped']) {
  assert.equal(lab.evaluateRecall('none', 2, denominator).score, null);
  for (const caseId of Object.keys(lab.evaluationCases)) {
    for (let cutoff = 1; cutoff <= 5; cutoff += 1) {
      const result = lab.evaluateRecall(caseId, cutoff, denominator);
      assert(result.score === null || (result.score >= 0 && result.score <= 1));
      assert(result.hits <= cutoff);
    }
  }
}
assert.throws(() => lab.evaluateRecall('small', 0, 'standard'));
assert.throws(() => lab.evaluateRecall('unknown', 2, 'standard'));
assert.throws(() => lab.evaluateRecall('small', 2, 'unknown'));
""")

    def test_lesson_pairs_and_progressive_fallbacks(self):
        folder = ROOT / "practice/recommender-systems"
        for source in folder.glob("*.md"):
            if source.name.endswith(".en.md"):
                continue
            english = source.with_name(source.stem + ".en.md")
            self.assertTrue(english.exists(), source.name)
            chinese_text, english_text = source.read_text(), english.read_text()
            for name, pattern in paritycheck.COUNTERS.items():
                self.assertEqual(len(re.findall(pattern, chinese_text)),
                                 len(re.findall(pattern, english_text)), f"{source.name}: {name}")
            for marker in ("architecture", "vectors", "evaluation"):
                pattern = rf'<div[^>]*data-recsys-{marker}[^>]*><p>(.+?)</p></div>'
                chinese_fallbacks = re.findall(pattern, chinese_text)
                english_fallbacks = re.findall(pattern, english_text)
                self.assertEqual(len(chinese_fallbacks), len(english_fallbacks))
                self.assertTrue(all(len(text) > 20 for text in chinese_fallbacks + english_fallbacks))
            headings = re.findall(r"(?m)^## (.+)$", english_text)
            self.assertEqual(len(headings), len(set(headings)), english.name)

    def test_assets_and_architecture_navigation(self):
        template = (ROOT / "site/template.html").read_text()
        for asset in ("recsys-design.js", "recsys-design.css"):
            self.assertIn(f"static/{asset}", template)
        nav = (ROOT / "site/nav.toml").read_text()
        self.assertIn('"README.md", "00-architecture.md", "01-feed-pipeline.md"', nav)
        script = SCRIPT.read_text()
        self.assertNotIn("localStorage", script)
        self.assertNotIn("fetch(", script)
        self.assertNotIn("setInterval", script)
        self.assertIn('data-vector-weight aria-label=', script)
        self.assertIn('data-vector-budget aria-label=', script)
        self.assertIn('data-eval-cutoff type="range" aria-label="Top-K"', script)


if __name__ == "__main__":
    unittest.main()
