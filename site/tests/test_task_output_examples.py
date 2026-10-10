from contextlib import redirect_stdout
from html.parser import HTMLParser
import io
import itertools
import math
from pathlib import Path
import random
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


class ExampleMarkup(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.depth = 0
        self.figures = {}
        self.visible = []
        self.folded = []
        self.feed(body)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "details":
            self.depth += 1
        if tag == "figure":
            self.figures[attrs["id"]] = (attrs, self.depth)

    def handle_endtag(self, tag):
        if tag == "details":
            self.depth -= 1

    def handle_data(self, text):
        (self.folded if self.depth else self.visible).append(text)


class TaskOutputExamplesTests(unittest.TestCase):
    def source(self, path, suffix=""):
        return (ROOT / f"{path}{suffix}.md").read_text()

    def code(self, path, anchor, suffix=""):
        body = self.source(path, suffix).split(f"{{#{anchor}}}", 1)[1]
        return re.search(r"```python\n(.*?)```", body, re.S).group(1)

    def run_example(self, path, anchor):
        chinese = self.code(path, anchor)
        self.assertEqual(chinese, self.code(path, anchor, ".en"))
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(compile(chinese, anchor, "exec"), namespace)
        return namespace

    def decoder(self):
        return self.run_example("00-foundations/core/bert", "answer-span")["best_answer_span"]

    def test_bilingual_qa_example_matches_visible_boundaries(self):
        result = self.run_example("00-foundations/core/bert", "answer-span")
        self.assertEqual(result["answer"], (9, 3, 4))
        slots = {"": ["会议", "改到", "了", "周五", "下午", "。"],
                 ".en": ["Meeting", "moved", "to", "Friday", "afternoon", "."]}
        for suffix, tokens in slots.items():
            with self.subTest(language=suffix):
                source = self.source("00-foundations/core/bert", suffix)
                self.assertIn(" / ".join(tokens), source)
                chosen = tokens[result["answer"][1]:result["answer"][2] + 1]
                self.assertEqual(chosen, ["Friday", "afternoon"] if suffix else ["周五", "下午"])

    def test_start_and_end_cannot_be_selected_independently(self):
        start_scores, end_scores = [0, 6], [8, 0]
        naive_start = max(range(2), key=start_scores.__getitem__)
        naive_end = max(range(2), key=end_scores.__getitem__)
        self.assertGreater(naive_start, naive_end)
        self.assertEqual(self.decoder()(start_scores, end_scores, [True, True], 2), (8, 0, 0))

    def test_masked_positions_cannot_be_endpoints_or_inside_a_span(self):
        decode = self.decoder()
        self.assertEqual(decode([100, 2, 1], [100, 1, 3], [False, True, True], 3), (5, 1, 2))
        self.assertEqual(decode([9, 0, 1], [1, 0, 9], [True, False, True], 3), (10, 0, 0))

    def test_max_length_is_inclusive_and_ties_are_stable(self):
        decode = self.decoder()
        self.assertEqual(decode([3, 2], [1, 4], [True, True], 1), (6, 1, 1))
        self.assertEqual(decode([3, 2], [1, 4], [True, True], 2), (7, 0, 1))
        self.assertEqual(decode([0, 0], [0, 0], [True, True], 2), (0, 0, 0))

    def test_no_candidate_is_not_semantic_abstention(self):
        decode = self.decoder()
        self.assertIsNone(decode([], [], [], 1))
        self.assertIsNone(decode([3], [4], [False], 1))
        self.assertEqual(decode([-100], [-100], [True], 1), (-200, 0, 0))

    def test_qa_input_checks(self):
        decode = self.decoder()
        inputs = [([1], [], [True], 1), ([1], [1], [], 1),
                  ([1], [1], [1], 1), ([1], [1], [True], 0),
                  ([1], [1], [True], -1), ([1], [1], [True], 1.5),
                  ([1], [1], [True], True), ([math.nan], [1], [True], 1),
                  ([1], [math.inf], [True], 1)]
        for arguments in inputs:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    decode(*arguments)

    def test_decoder_matches_independent_enumeration(self):
        decode = self.decoder()
        generator = random.Random(42)
        for length in range(1, 6):
            start_scores = [generator.uniform(-5, 5) for _ in range(length)]
            end_scores = [generator.uniform(-5, 5) for _ in range(length)]
            for allowed in itertools.product([False, True], repeat=length):
                for max_length in range(1, length + 2):
                    candidates = [(start_scores[start] + end_scores[end], start, end)
                                  for start in range(length) for end in range(start, length)
                                  if end - start + 1 <= max_length and all(allowed[start:end + 1])]
                    expected = max(candidates, key=lambda item: item[0], default=None)
                    self.assertEqual(decode(start_scores, end_scores, allowed, max_length), expected)

    def test_grpo_example_and_loss_sign(self):
        result = self.run_example("05-post-training/rlhf/after-rlhf", "reward-to-update")
        self.assertAlmostEqual(result["ratio"], 1.3)
        self.assertAlmostEqual(result["objective"], 2.0784609690826525)
        self.assertEqual(result["loss_term"], -result["objective"])
        log_ratio = math.exp(math.log(result["current_probability"]) - math.log(result["old_probability"]))
        self.assertAlmostEqual(log_ratio, result["ratio"])

    def test_clipping_handles_both_advantage_signs(self):
        term = self.run_example("05-post-training/rlhf/after-rlhf", "reward-to-update")["clipped_policy_term"]
        self.assertAlmostEqual(term(1.3, 2), term(1.6, 2))
        self.assertAlmostEqual(term(0.4, -2), term(0.7, -2))
        self.assertLess(term(1.6, -2), term(1.3, -2))
        self.assertLess(term(0.4, 2), term(0.7, 2))
        self.assertEqual(term(1.5, 0), 0)

    def test_figures_are_localized_and_visible_code_is_optional(self):
        for path, identifier, cards in [("00-foundations/core/bert", "answer-boundaries", 2),
                                        ("05-post-training/rlhf/after-rlhf", "grpo-signal-path", 3)]:
            for suffix, language in [("", "zh-CN"), (".en", "en")]:
                with self.subTest(path=path, language=language):
                    body, _ = build.render_markdown(self.source(path, suffix))
                    parsed = ExampleMarkup(body)
                    attributes, depth = parsed.figures[identifier]
                    self.assertEqual(attributes["lang"], language)
                    self.assertEqual(depth, 0)
                    figure = re.search(rf'<figure[^>]*id="{identifier}".*?</figure>', body, re.S).group()
                    self.assertEqual(figure.count("<li>"), cards)
                    self.assertNotIn("<button", figure)
                    if suffix:
                        self.assertNotRegex(figure, r"[\u4e00-\u9fff]")
                    else:
                        self.assertRegex(figure, r"[\u4e00-\u9fff]")
                    function = "best_answer_span" if cards == 2 else "clipped_policy_term"
                    self.assertIn(f"def {function}", "".join(parsed.folded))
                    self.assertNotIn(f"def {function}", "".join(parsed.visible))


if __name__ == "__main__":
    unittest.main()
