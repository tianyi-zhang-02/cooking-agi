from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FunctionApproximationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = []
        for suffix in (".md", ".en.md"):
            text = (ROOT / "00-foundations" / ("from-linear-to-neural" + suffix)).read_text()
            cls.blocks.append(next(block for block in re.findall(r"```python\n(.*?)```", text, re.S) if "def square_interpolant" in block))
        namespace = {}
        exec(cls.blocks[0], namespace)
        cls.interpolate = staticmethod(namespace["square_interpolant"])

    def test_bilingual_code(self):
        self.assertEqual(*self.blocks)

    def test_uniform_bound_and_endpoints(self):
        for segments in (1, 2, 4, 10, 25):
            for position in range(1001):
                value = position / 1000
                error = self.interpolate(value, segments) - value ** 2
                self.assertGreaterEqual(error, -1e-12)
                self.assertLessEqual(error, 1 / (4 * segments ** 2) + 1e-12)
            self.assertEqual(self.interpolate(0, segments), 0)
            self.assertAlmostEqual(self.interpolate(1, segments), 1)

    def test_invalid_configuration(self):
        for args in [(-0.1, 4), (1.1, 4), (0.5, 0), (0.5, True)]:
            with self.assertRaises(ValueError):
                self.interpolate(*args)


if __name__ == "__main__":
    unittest.main()
