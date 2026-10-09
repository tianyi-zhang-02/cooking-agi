from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ResearchWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = []
        for suffix in (".md", ".en.md"):
            text = (ROOT / "10-agents" / ("deep-research" + suffix)).read_text()
            cls.blocks.append(re.findall(r"```python\n(.*?)```", text, re.S))
        namespace = {}
        for block in cls.blocks[0]:
            exec(block, namespace)
        cls.status = staticmethod(namespace["research_status"])
        cls.citations = staticmethod(namespace["check_citation_spans"])

    def test_bilingual_code(self):
        self.assertEqual(*self.blocks)

    def test_support_requires_evidence(self):
        self.assertEqual(self.status({"q": {"status": "supported", "evidence_ids": []}}, 2), ("needs_research", ["q"]))
        self.assertEqual(self.status({"q": {"status": "supported", "evidence_ids": ["a"]}}, 0), ("ready_for_review", []))

    def test_budget_exhaustion_is_not_completion(self):
        self.assertEqual(self.status({"q": {"status": "unavailable"}}, 0), ("budget_exhausted", ["q"]))

    def test_invalid_status(self):
        with self.assertRaises(ValueError):
            self.status({"q": {"status": "guessed"}}, 1)

    def test_bad_citations(self):
        sources = {"a": {"read": True, "text": "abc"}}
        self.assertEqual(self.citations([{"source_id": "b", "span": (0, 1), "quote": "a"}], sources), ["missing_or_unread_source"])
        self.assertEqual(self.citations([{"source_id": "a", "span": (0, 4), "quote": "abcd"}], sources), ["invalid_span"])
        self.assertEqual(self.citations([{"source_id": "a", "span": (0, 1), "quote": "z"}], sources), ["quote_mismatch"])
        self.assertEqual(self.citations([{"source_id": "a", "span": (0, 3), "quote": "abc"}], sources), [])


if __name__ == "__main__":
    unittest.main()
