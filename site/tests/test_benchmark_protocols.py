from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BenchmarkProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = {}
        for suffix in (".md", ".en.md"):
            text = (ROOT / "07-evaluation" / ("benchmark-protocols" + suffix)).read_text()
            cls.blocks[suffix] = re.findall(r"```python\n(.*?)```", text, re.S)
        namespace = {}
        for block in cls.blocks[".md"]:
            exec(compile(block, "benchmark-protocols", "exec"), namespace)
        cls.lookup = staticmethod(namespace["lookup_case"])

    def test_bilingual_code(self):
        self.assertEqual(self.blocks[".md"], self.blocks[".en.md"])

    def test_evidence_moves_without_changing_other_records(self):
        cases = [self.lookup(15, position) for position in range(15)]
        expected_records = sorted(cases[0][0].splitlines())
        for position, (context, answer) in enumerate(cases):
            self.assertEqual(context.splitlines()[position], "target: maple-47")
            self.assertEqual(sorted(context.splitlines()), expected_records)
            self.assertEqual(answer, "maple-47")

    def test_unanswerable_control_does_not_contain_answer(self):
        context, answer = self.lookup(9, 4, False)
        self.assertIsNone(answer)
        self.assertNotIn("maple-47", context)
        self.assertNotIn("target:", context)
        self.assertEqual(len(context.splitlines()), 9)

    def test_invalid_parameters(self):
        for args in [(1, 0), (2, -1), (2, 2), (2, True), (2, 1, 1)]:
            with self.assertRaises(ValueError):
                self.lookup(*args)


if __name__ == "__main__":
    unittest.main()
