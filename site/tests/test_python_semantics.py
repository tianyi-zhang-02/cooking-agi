from collections import defaultdict
from functools import cache
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck


class FigureMarkup(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.details_depth = 0
        self.figures = {}
        self.feed(body)

    def handle_starttag(self, tag, attributes):
        if tag == "details":
            self.details_depth += 1
        if tag == "figure":
            attrs = dict(attributes)
            self.figures[attrs.get("id")] = (attrs.get("lang"), self.details_depth)

    def handle_endtag(self, tag):
        if tag == "details":
            self.details_depth -= 1


class PythonSemanticsTests(unittest.TestCase):
    def source(self, chapter="python-objects", suffix=""):
        return (ROOT / f"interview/{chapter}{suffix}.md").read_text()

    def snippets(self, chapter="python-objects", suffix=""):
        return re.findall(r"```python\n(.*?)```", self.source(chapter, suffix), re.S)

    def example(self, marker):
        code = next(code for code in self.snippets() if marker in code)
        namespace = {}
        exec(compile(code, marker, "exec"), namespace)
        return namespace

    def test_each_snippet_runs_independently_in_both_languages(self):
        for chapter in ("python", "python-objects"):
            for suffix in ("", ".en"):
                for number, code in enumerate(self.snippets(chapter, suffix), 1):
                    with self.subTest(chapter=chapter, language=suffix, number=number):
                        exec(compile(code, f"{chapter}{suffix}:{number}", "exec"), {})

    def test_bilingual_examples_and_structure_match(self):
        for chapter in ("python", "python-objects"):
            self.assertEqual(self.snippets(chapter), self.snippets(chapter, ".en"))
            self.assertEqual(paritycheck.shape(self.source(chapter)),
                             paritycheck.shape(self.source(chapter, ".en")))

    def test_aliasing_and_rebinding(self):
        namespace = self.example('backup = labels')
        self.assertEqual(namespace["labels"], ["draft", "checked"])
        self.assertIsNot(namespace["labels"], namespace["backup"])

    def test_tuple_immutability_does_not_freeze_nested_list(self):
        bundle = (["draft"],)
        bundle[0].append("checked")
        self.assertEqual(bundle[0], ["draft", "checked"])
        with self.assertRaises(TypeError):
            bundle[0] = []
        with self.assertRaises(TypeError):
            hash(bundle)

    def test_copy_boundaries(self):
        namespace = self.example('edited = record.copy()')
        self.assertIsNot(namespace["edited"], namespace["record"])
        self.assertIs(namespace["edited"]["tags"], namespace["record"]["tags"])
        namespace = self.example('edited = {**record')
        self.assertIsNot(namespace["edited"]["tags"], namespace["record"]["tags"])

    def test_deepcopy_preserves_internal_sharing_and_cycles(self):
        namespace = self.example('from copy import deepcopy')
        self.assertIs(namespace["cloned"][0]["tags"], namespace["cloned"][1]["tags"])
        self.assertIsNot(namespace["cloned"][0]["tags"], namespace["shared_tags"])
        self.assertIs(namespace["copied_cycle"][0], namespace["copied_cycle"])

    def test_mutation_survives_local_parameter_rebinding(self):
        namespace = self.example('def append_then_replace')
        original = []
        self.assertEqual(namespace["append_then_replace"](original), ["replacement"])
        self.assertEqual(original, ["checked"])
        alias = original
        original += ["in-place"]
        self.assertIs(alias, original)
        original = original + ["new"]
        self.assertIsNot(alias, original)
        self.assertNotIn("new", alias)

    def test_argument_collection_and_errors(self):
        summarize = self.example('def summarize')["summarize"]
        self.assertEqual(summarize(), (0, (), {}))
        self.assertEqual(summarize(2, 5, 3), (10, (2, 5, 3), {}))
        with self.assertRaises(TypeError):
            summarize(2, scale=1, **{"scale": 2})
        with self.assertRaises(TypeError):
            summarize(**{3: "bad-key"})

    def test_fresh_defaults_but_explicit_list_is_mutated(self):
        add_tag = self.example('def add_tag')["add_tag"]
        first = add_tag("one")
        second = add_tag("two")
        self.assertIsNot(first, second)
        self.assertEqual(first, ["one"])
        self.assertIs(add_tag("three", first), first)
        self.assertEqual(first, ["one", "three"])
        self.assertEqual(second, ["two"])

    def test_definition_time_capture_is_not_deepcopy(self):
        namespace = self.example('fixed_readers')
        self.assertEqual([reader() for reader in namespace["readers"]], [2, 2, 2])
        self.assertEqual([reader() for reader in namespace["fixed_readers"]], [0, 1, 2])
        values = []
        reader = lambda saved=values: saved
        values.append("later")
        self.assertIs(reader(), values)
        self.assertEqual(reader(), ["later"])

    def test_all_short_circuits_and_exhaustion_does_not_restart(self):
        namespace = self.example('def passing_scores')
        namespace["events"].clear()
        checks = namespace["passing_scores"]([80, 40, 90])
        self.assertFalse(all(checks))
        self.assertEqual(namespace["events"], [80, 40])
        self.assertEqual(list(checks), [True])
        self.assertEqual(list(checks), [])
        self.assertTrue(all([]))
        self.assertFalse(any([]))

    def test_running_mean_handles_batches_and_independent_instances(self):
        namespace = self.example('class RunningMean')
        meter = namespace["RunningMean"]()
        with self.assertRaises(ValueError):
            meter.value()
        with self.assertRaises(ValueError):
            namespace["average"]([])
        for batch in ([2, 4], [9], [-3, 8]):
            for value in batch:
                meter.add(value)
        self.assertEqual(meter.value(), namespace["average"](iter([2, 4, 9, -3, 8])))
        self.assertEqual(namespace["RunningMean"]().count, 0)

    def test_mixed_sort_is_not_whole_tuple_reverse(self):
        records = [("beta", 3), ("alpha", 2), ("beta", 1), ("alpha", 1)]
        expected = [("beta", 1), ("beta", 3), ("alpha", 1), ("alpha", 2)]
        ordered = sorted(sorted(records, key=lambda record: record[1]),
                         key=lambda record: record[0], reverse=True)
        self.assertEqual(ordered, expected)
        self.assertNotEqual(sorted(records, reverse=True), expected)

    def test_zip_strict_detects_length_mismatch(self):
        self.assertEqual(list(zip(["amy", "bob"], [90])), [("amy", 90)])
        with self.assertRaises(ValueError):
            list(zip(["amy", "bob"], [90], strict=True))

    def test_defaultdict_get_does_not_insert(self):
        groups = defaultdict(list)
        self.assertIsNone(groups.get("missing"))
        self.assertNotIn("missing", groups)
        self.assertEqual(groups["missing"], [])
        self.assertIn("missing", groups)

    def test_cache_requires_hashability_and_keeps_returned_object(self):
        @cache
        def labels(name):
            return [name]

        first = labels("draft")
        first.append("checked")
        self.assertIs(labels("draft"), first)
        self.assertEqual(labels("draft"), ["draft", "checked"])
        self.assertIsNone(labels.cache_parameters()["maxsize"])
        with self.assertRaises(TypeError):
            labels([])

    def test_default_visible_figure_uses_page_language(self):
        for suffix, language in (("", "zh-CN"), (".en", "en")):
            body, _ = build.render_markdown(self.source(suffix=suffix))
            self.assertEqual(FigureMarkup(body).figures["copy-boundary"], (language, 0))
            figure = re.search(r'<figure\b.*?</figure>', body, re.S).group()
            if suffix:
                self.assertNotRegex(figure, r"[\u4e00-\u9fff]")
            else:
                self.assertIn("替换标题", figure)

    def test_navigation_and_evidence_status(self):
        navigation = tomllib.loads((ROOT / "site/nav.toml").read_text())
        section = next(section for section in navigation["section"] if section["dir"] == "interview")
        self.assertLess(section["order"].index("python.md"), section["order"].index("python-objects.md"))
        evidence = tomllib.loads((ROOT / "site/appendix-coverage.toml").read_text())
        topic = next(topic for chapter in evidence["chapter"] for topic in chapter["topics"]
                     if topic["id"] == "python")
        self.assertEqual((topic["state"], topic["read"]), ("article", "body"))
        self.assertIn("interview/python-objects.md", topic["notes"])


if __name__ == "__main__":
    unittest.main()
