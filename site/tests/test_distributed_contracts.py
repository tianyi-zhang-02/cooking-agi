from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DistributedContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = []
        for suffix in (".md", ".en.md"):
            text = (ROOT / "06-systems" / ("distributed-training" + suffix)).read_text()
            cls.blocks.append(re.findall(r"```python\n(.*?)```", text, re.S))
        namespace = {}
        for block in cls.blocks[0]:
            exec(block, namespace)
        cls.traffic = staticmethod(namespace["ring_traffic"])

    def test_bilingual_code(self):
        self.assertEqual(*self.blocks)

    def test_ring_volume_and_rounds(self):
        self.assertEqual(self.traffic(120, 4), {"sent_per_rank": 180, "received_per_rank": 180, "rounds": 6})
        self.assertEqual(self.traffic(120, 1)["sent_per_rank"], 0)
        self.assertEqual(self.traffic(0, 4)["sent_per_rank"], 0)
        for args in [(120, 0), (120, True), (-1, 4), (float("nan"), 4)]:
            with self.assertRaises(ValueError):
                self.traffic(*args)

    def test_stale_gradient_changes_the_update(self):
        parameter, rate = 2.0, 0.5
        old_gradient = parameter
        parameter -= rate * old_gradient
        self.assertEqual(parameter - rate * parameter, 0.5)
        self.assertEqual(parameter - rate * old_gradient, 0)

    def test_stashed_derivative_matches_finite_difference(self):
        step = 1e-6
        numerical = ((3 * (2 + step)) ** 2 / 2 - (3 * (2 - step)) ** 2 / 2) / (2 * step)
        self.assertAlmostEqual(numerical, 18, places=6)
        self.assertNotAlmostEqual(numerical, 6 * 4)
        self.assertAlmostEqual(2 / 25 + 2 / 25, 0.16)


if __name__ == "__main__":
    unittest.main()
