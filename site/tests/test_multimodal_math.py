import importlib.util
import json
import math
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("multimodal_math", ROOT / "00-foundations/code/multimodal_math.py")
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


class MultimodalMathTests(unittest.TestCase):
    def test_dot_and_cosine_differ(self):
        self.assertEqual(LAB.dot([1, 0], [2, 2]), 2)
        self.assertAlmostEqual(LAB.cosine([1, 0], [2, 2]), 1 / math.sqrt(2))
        self.assertAlmostEqual(LAB.cosine([1, 0], [20, 20]), 1 / math.sqrt(2))
        with self.assertRaises(ValueError):
            LAB.cosine([0, 0], [1, 0])

    def test_softmax_is_shift_invariant_and_stable(self):
        original = LAB.log_softmax([0.8, 0.2, 0.1])
        shifted = LAB.log_softmax([1000.8, 1000.2, 1000.1])
        for first, second in zip(original, shifted):
            self.assertAlmostEqual(first, second)
        self.assertAlmostEqual(sum(math.exp(value) for value in original), 1)
        self.assertAlmostEqual(math.exp(original[0]), 0.48890265771885366)

    def test_uniform_symmetric_and_single_pair(self):
        self.assertAlmostEqual(LAB.contrastive_loss([[0] * 3 for _ in range(3)]), math.log(3))
        self.assertAlmostEqual(LAB.contrastive_loss(LAB.SCORES), LAB.contrastive_loss(list(zip(*LAB.SCORES))))
        self.assertEqual(LAB.contrastive_loss([[1]]), 0)

    def test_temperature_and_duplicate_candidates(self):
        self.assertLess(LAB.contrastive_loss(LAB.SCORES, 0.1), LAB.contrastive_loss(LAB.SCORES, 1))
        duplicate = [0.8, 0.8, 0.1]
        probabilities = [math.exp(value) for value in LAB.log_softmax(duplicate)]
        self.assertEqual(probabilities[0], probabilities[1])
        self.assertLess(probabilities[0], 0.5)
        wrong = [0.2, 0.8, 0.1]
        self.assertGreater(-LAB.log_softmax([value / 0.1 for value in wrong])[0], -LAB.log_softmax(wrong)[0])

    def test_invalid_inputs(self):
        for temperature in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                LAB.contrastive_loss(LAB.SCORES, temperature)
        for scores in ([], [[1, 2]], [[float("nan")]]):
            with self.assertRaises(ValueError):
                LAB.contrastive_loss(scores)

    def test_next_token_shift_and_mask(self):
        logits = [[0.0] * 6 for _ in range(6)]
        logits[2][4] = 2
        logits[3][5] = 3
        tokens = [1, 2, 3, 4, 5, 0]
        mask = [False, False, False, True, True, False]
        expected = (-LAB.log_softmax(logits[2])[4] - LAB.log_softmax(logits[3])[5]) / 2
        self.assertAlmostEqual(LAB.next_token_loss(logits, tokens, mask), expected)
        logits[0] = [100.0] * 6
        logits[4] = [-100.0] * 6
        self.assertAlmostEqual(LAB.next_token_loss(logits, tokens, mask), expected)
        self.assertAlmostEqual(LAB.next_token_loss(logits + [[0] * 6], tokens + [0], mask + [False]), expected)
        with self.assertRaises(ValueError):
            LAB.next_token_loss(logits, tokens, [False] * 6)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch comparison")
    def test_torch_parity_and_update(self):
        import torch
        import torch.nn.functional as functional
        before, after = LAB.torch_demo()
        self.assertLess(after, before)
        tokens = torch.tensor([1, 2, 3, 4, 5, 0])
        logits = torch.arange(36, dtype=torch.float64).reshape(6, 6) / 10
        mask = torch.tensor([False, False, False, True, True, False])
        expected = functional.cross_entropy(logits[:-1][mask[1:]], tokens[1:][mask[1:]])
        self.assertAlmostEqual(LAB.next_token_loss(logits.tolist(), tokens.tolist(), mask.tolist()), expected.item())

    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser-math parity")
    def test_browser_math_parity(self):
        script = """
const lab = require(process.argv[1]);
const results = ['paired', 'wrong', 'duplicate'].flatMap(scenario => [0.05, 0.5, 1.5].map(temperature => {
  const scores = lab.scenarioScores(scenario);
  return {scores, temperature, result: lab.calculate(scores, temperature)};
}));
process.stdout.write(JSON.stringify(results));
"""
        result = subprocess.check_output([shutil.which("node"), "-e", script, str(ROOT / "site/static/clip-lab.js")], text=True)
        for item in json.loads(result):
            self.assertAlmostEqual(item["result"]["loss"], LAB.contrastive_loss(item["scores"], item["temperature"]))
            for direction in ("rows", "columns"):
                for row in item["result"][direction]:
                    self.assertAlmostEqual(sum(math.exp(value) for value in row), 1)


if __name__ == "__main__":
    unittest.main()
