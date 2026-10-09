import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import paritycheck

SPEC = importlib.util.spec_from_file_location(
    "judge_eval_reference", ROOT / "07-evaluation/llm-as-a-judge/code/judge_eval.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
SCRIPT = ROOT / "site/static/judge-lab.js"


class JudgeLabTests(unittest.TestCase):
    def test_probability_example_distinguishes_mass_from_conditional_mean(self):
        snippets = []
        for suffix in ('.md', '.en.md'):
            source = (ROOT / ('07-evaluation/llm-as-a-judge/probability-scores' + suffix)).read_text()
            snippets.append(re.findall(r'```python\n(.*?)```', source, re.S))
        self.assertEqual(snippets[0], snippets[1])
        self.assertEqual(len(snippets[0]), 1)
        namespace = {}
        exec(compile(snippets[0][0], 'probability-scores', 'exec'), namespace)
        self.assertAlmostEqual(namespace['rating_mass'], 0.8)
        self.assertAlmostEqual(namespace['weighted_sum'], 3.28)
        self.assertAlmostEqual(namespace['conditional_mean'], 4.1)

    def run_js(self, script):
        if not shutil.which("node"):
            self.skipTest("Node.js required")
        result = subprocess.run(
            [shutil.which("node"), "-e",
             "const assert = require('node:assert/strict'); const lab = require(process.argv[1]);\n" + script,
             str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_walkthrough_separates_judgment_and_decision(self):
        self.run_js("""
const fluent = lab.walkthroughVerdicts('fluent', true);
assert.deepEqual(fluent, {grounding:'fail', completeness:'pass', tone:'pass'});
assert.equal(lab.walkthroughDecision(fluent,false).decision, 'pass');
assert.equal(lab.walkthroughDecision(fluent,true).decision, 'fail');
const brief = lab.walkthroughVerdicts('brief',true);
assert.equal(lab.walkthroughDecision(brief,false).decision, 'pass');
assert.equal(lab.walkthroughDecision(brief,true).decision, 'fail');
const complete = lab.walkthroughVerdicts('complete',true);
assert.equal(lab.walkthroughDecision(complete,true).decision, 'pass');
assert.deepEqual(fluent, {grounding:'fail', completeness:'pass', tone:'pass'});
for (const answer of Object.values(lab.walkthroughAnswers)) {
  for (const field of ['label','text']) assert.equal(answer[field].length,2);
  for (const reasons of Object.values(answer.reasons)) assert.equal(reasons.length,2);
}
""")

    def test_walkthrough_missing_evidence_never_approves(self):
        self.run_js("""
for (const answer of Object.keys(lab.walkthroughAnswers)) {
  const verdicts = lab.walkthroughVerdicts(answer,false);
  assert.equal(verdicts.grounding,'unknown');
  for (const hardGates of [false,true]) {
    assert.notEqual(lab.walkthroughDecision(verdicts,hardGates).decision, 'pass');
  }
}
assert.equal(lab.walkthroughDecision(lab.walkthroughVerdicts('complete',false),true).decision, 'review');
assert.equal(lab.walkthroughDecision(lab.walkthroughVerdicts('brief',false),true).decision, 'fail');
assert.throws(()=>lab.walkthroughVerdicts('missing',true));
assert.throws(()=>lab.walkthroughVerdicts('complete','yes'));
assert.throws(()=>lab.walkthroughDecision({},true));
assert.throws(()=>lab.walkthroughDecision({grounding:'unknown',completeness:'pass',tone:'maybe'},true));
""")

    def test_calibration_walkthrough_matches_explanations(self):
        self.run_js("""
const lower = lab.calibrate(lab.calibrationRows,3.5,0);
const higher = lab.calibrate(lab.calibrationRows,4,0);
assert.equal(lower.accuracy, higher.accuracy);
assert.equal(lower.matrix.fp, 3);
assert.equal(higher.matrix.fp, 2);
assert.equal(lower.matrix.fn, 2);
assert.equal(higher.matrix.fn, 3);
assert.equal(lab.calibrate(lab.calibrationRows,4,.5).review,6);
""")

    def test_same_mean_different_variance_and_empty(self):
        self.run_js("""
const center = lab.scoreSummary([0,0,20,0,0]);
const split = lab.scoreSummary([10,0,0,0,10]);
assert.equal(center.mean, 3);
assert.equal(split.mean, 3);
assert.equal(center.variance, 0);
assert.equal(split.variance, 4);
assert.equal(split.entropy, 1);
assert.equal(lab.scoreSummary([0,0,4,12,4]).mean, 4);
assert.equal(lab.scoreSummary([0,0,0,0,0]).mean, null);
for (const counts of [[1], [-1,0,0,0,0], [NaN,0,0,0,0], [1.2,0,0,0,0]]) {
  assert.throws(() => lab.scoreSummary(counts));
}
""")

    def test_pairwise_tracks_identity_not_position(self):
        self.run_js("""
const order = ['answer-A','answer-B'];
const reversed = [...order].reverse();
assert.equal(lab.choosePair(order, 'first'), 'answer-A');
assert.equal(lab.choosePair(reversed, 'first'), 'answer-B');
for (const policy of ['policy','longer']) {
  assert.equal(lab.choosePair(order, policy), lab.choosePair(reversed, policy));
}
assert.equal(lab.choosePair(order, 'longer'), 'answer-B');
assert.equal(lab.choosePair(order, 'policy'), 'answer-A');
assert.deepEqual(order, ['answer-A','answer-B']);
assert.throws(() => lab.choosePair(['answer-A','answer-A'], 'first'));
assert.throws(() => lab.choosePair(order, 'unknown'));
""")

    def test_missing_evidence_is_criterion_specific(self):
        self.run_js("""
assert.equal(lab.evidenceVerdict('returns','grounding',false).verdict, 'unknown');
assert.equal(lab.evidenceVerdict('returns','grounding',true).verdict, 'fail');
assert.equal(lab.evidenceVerdict('returns','relevance',false).verdict, 'pass');
assert.equal(lab.evidenceVerdict('agent','permission',true).verdict, 'pass');
assert.equal(lab.evidenceVerdict('agent','completion',true).verdict, 'fail');
assert.equal(lab.evidenceVerdict('memory','current',false).verdict, 'fail');
assert.throws(() => lab.evidenceVerdict('missing','criterion',true));
assert.throws(() => lab.evidenceVerdict('returns','missing',true));
for (const scene of Object.values(lab.scenarios)) {
  for (const field of ['label','task','candidate','evidence']) {
    assert.equal(scene[field].length, 2);
    assert(scene[field].every(value => typeof value === 'string' && value.length));
  }
}
""")

    def test_calibration_denominators_and_review(self):
        self.run_js("""
const before = JSON.stringify(lab.calibrationRows);
const result = lab.calibrate(lab.calibrationRows, 3.5, 0);
assert.deepEqual(result.matrix, {tp:4,fp:3,fn:2,tn:3});
assert.equal(result.approvalError, 3/7);
assert.equal(result.accuracy, 7/12);
assert.equal(result.coverage, 1);
const reviewed = lab.calibrate(lab.calibrationRows, 4, .5);
assert.equal(reviewed.review, 6);
assert.equal(reviewed.coverage, .5);
const allReview = lab.calibrate(lab.calibrationRows, 3, 4);
assert.equal(allReview.accuracy, null);
assert.equal(allReview.approvalError, null);
assert.equal(allReview.coverage, 0);
const noApprovals = lab.calibrate(lab.calibrationRows, 5, 0);
assert.equal(noApprovals.approvalError, null);
assert.equal(lab.calibrate([],3,0).coverage, null);
assert.equal(JSON.stringify(lab.calibrationRows), before);
for (const threshold of [NaN,0,6]) assert.throws(() => lab.calibrate([],threshold,0));
assert.throws(() => lab.calibrate([lab.calibrationRows[0],lab.calibrationRows[0]],3,0));
assert.throws(() => lab.calibrate([{id:'bad',human:'other',score:4}],3,0));
for (const threshold of [1,2,3,4,5]) {
  for (const band of [0,.5]) {
    const audit = lab.calibrate(lab.calibrationRows,threshold,band);
    assert.equal(Object.values(audit.matrix).reduce((sum,value)=>sum+value,0), audit.decided);
    assert.equal(audit.decided + audit.review, audit.total);
  }
}
""")

    def test_notes_bilingual_and_all_demos_present(self):
        folder = ROOT / "07-evaluation/llm-as-a-judge"
        widgets = []
        for path in folder.glob("*.md"):
            if path.name.endswith(".en.md"):
                continue
            english = path.with_name(path.stem + ".en.md")
            self.assertTrue(english.exists())
            self.assertEqual(paritycheck.shape(path.read_text()), paritycheck.shape(english.read_text()))
            for name in ("walkthrough", "evidence", "pairwise", "distribution", "calibration"):
                if f'data-judge-lab="{name}"' in path.read_text():
                    widgets.append(name)
                    self.assertIn(f'data-judge-lab="{name}"', english.read_text())
        self.assertCountEqual(widgets, ["walkthrough", "evidence", "pairwise", "distribution", "calibration"])
        script = SCRIPT.read_text()
        for network_api in ("fetch(", "XMLHttpRequest", "localStorage", "sessionStorage"):
            self.assertNotIn(network_api, script)


class JudgeReferenceTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "07-evaluation/llm-as-a-judge/code/example-results.jsonl"
        self.records = [json.loads(line) for line in path.read_text().splitlines()]
        self.output = json.loads(self.records[0]["judge_raw"])

    def parse(self, payload):
        return REFERENCE.parse_judgment(json.dumps(payload), {"policy-1"}, "policy-grounding-v1")

    def test_demo_keeps_failures_in_denominator(self):
        report = REFERENCE.evaluate(self.records)
        overall = report["overall"]
        self.assertEqual(overall["total_cases"], 7)
        self.assertEqual(overall["decided"], 4)
        self.assertEqual(overall["coverage"], 4 / 7)
        self.assertEqual(overall["accuracy_on_decided"], .5)
        self.assertEqual(overall["error_among_approvals"], .5)
        self.assertEqual(overall["confusion_matrix"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual([overall[key] for key in ("unknown", "invalid", "error")], [1, 1, 1])
        self.assertEqual(sum(row["total_cases"] for row in report["slices"].values()), 7)

    def test_unknown_is_not_invalid(self):
        self.output.update(verdict="unknown", evidence_ids=[])
        self.assertEqual(self.parse(self.output)["status"], "unknown")
        self.output["verdict"] = "maybe"
        self.assertEqual(self.parse(self.output)["status"], "invalid")

    def test_contract_rejects_bad_outputs(self):
        variants = [
            {"verdict": True}, {"verdict": []}, {"criterion_id": "old"},
            {"evidence_ids": ["made-up"]}, {"evidence_ids": []},
            {"evidence_ids": ["policy-1", "policy-1"]},
            {"reason": ""}, {"reason": "x" * 601}, {"extra": "not contracted"},
        ]
        for update in variants:
            with self.subTest(update=update):
                self.assertEqual(self.parse({**self.output, **update})["status"], "invalid")
        for raw in ("not json", "[]", "null", '{"verdict":"pass","verdict":"fail"}'):
            self.assertEqual(REFERENCE.parse_judgment(raw, {"policy-1"}, "policy-grounding-v1")["status"], "invalid")
        self.assertEqual(REFERENCE.parse_judgment(None, set(), "policy-grounding-v1")["status"], "error")

    def test_duplicate_ids_and_bad_dataset_are_rejected(self):
        for records in ([self.records[0], self.records[0]], [{"case_id": "missing"}],
                        [{**self.records[0], "human": "unknown"}]):
            with self.assertRaises(ValueError):
                REFERENCE.evaluate(records)

    def test_empty_report_has_no_fake_zeros(self):
        report = REFERENCE.evaluate([])["overall"]
        self.assertIsNone(report["coverage"])
        self.assertIsNone(report["accuracy_on_decided"])
        self.assertIsNone(report["error_among_approvals"])


if __name__ == "__main__":
    unittest.main()
