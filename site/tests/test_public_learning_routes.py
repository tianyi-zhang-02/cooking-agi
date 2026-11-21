import copy
import importlib.util
import math
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = "14-production-ml-and-recs"
INFRA = "15-ai-infrastructure-and-distributed-systems"


def load_example(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = load_example("route_data_checks", f"{PRODUCTION}/code/data_checks.py")
capacity = load_example("route_capacity_checks", f"{INFRA}/code/capacity_checks.py")


class DataExampleTests(unittest.TestCase):
    def test_late_snapshot_is_not_available(self):
        results = data.example_results()
        self.assertEqual(results["available_feature"], 3)
        self.assertEqual(results["event_time_only_feature"], 7)

    def test_earlier_arrival_changes_selection(self):
        records = copy.deepcopy(data.FEATURES)
        records[1]["available"] = 599
        self.assertEqual(data.feature_at(records, 600, 60), 7)
        self.assertEqual(data.FEATURES[1]["available"], 604)

    def test_ttl_and_empty_history(self):
        self.assertIsNone(data.feature_at(data.FEATURES, 600, 5))
        self.assertEqual(data.feature_at(data.FEATURES, 600, 10), 3)
        self.assertIsNone(data.feature_at([], 600, 60))
        with self.assertRaises(ValueError):
            data.feature_at(data.FEATURES, 600, -1)

    def test_backfill_and_future_event(self):
        records = [
            {"event": 590, "available": 591, "value": 3},
            {"event": 590, "available": 610, "value": 99},
            {"event": 601, "available": 599, "value": 88},
        ]
        self.assertEqual(data.feature_at(records, 600, 60), 3)
        self.assertEqual(data.feature_at(records[:2], 610, 60), 99)

    def test_probability_correction_and_inverse(self):
        self.assertAlmostEqual(data.original_probability(0.5, 0.1), 1 / 11)
        for original in (0, 0.02, 0.5, 1):
            for keep in (0.1, 0.4, 1):
                sampled = original / (original + keep * (1 - original))
                self.assertAlmostEqual(data.original_probability(sampled, keep), original)
        for probability, keep in ((-0.1, 0.1), (1.1, 0.1), (0.5, 0), (0.5, 2), (math.nan, 1)):
            with self.subTest(probability=probability, keep=keep), self.assertRaises(ValueError):
                data.original_probability(probability, keep)

    def test_overall_and_segment_changes(self):
        self.assertEqual(data.example_results()["overall_ctr"], [0.05, 0.06])
        self.assertEqual(data.example_results()["new_user_ctr"], [0.05, 0.03])
        self.assertAlmostEqual(data.click_rate(1140, 18000), 0.06333333333333334)
        for clicks, impressions in ((1, 0), (3, 2), (-1, 2), (1.5, 2)):
            with self.subTest(clicks=clicks), self.assertRaises(ValueError):
                data.click_rate(clicks, impressions)


class CapacityExampleTests(unittest.TestCase):
    def test_documented_capacity_and_cache(self):
        self.assertEqual(capacity.example_results(), {
            "backlog": 200, "drain_seconds": 10, "capacity_replicas": 3,
            "replicas_with_one_failure_reserve": 4,
            "short_request_kv_mib": 320, "long_request_kv_mib": 1088,
            "short_batch_kv_gib": 10, "long_batch_kv_gib": 34,
            "token_weighted_loss": 2.5,
        })

    def test_queue_boundaries(self):
        self.assertEqual(capacity.backlog_after(200, 80, 100, 20), 0)
        self.assertEqual(capacity.backlog_after(200, 100, 100, 20), 200)
        self.assertEqual(capacity.drain_seconds(200, 100, 100), math.inf)
        self.assertEqual(capacity.drain_seconds(200, 120, 100), math.inf)
        self.assertEqual(capacity.drain_seconds(0, 80, 100), 0)
        for invalid in (-1, math.inf, math.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                capacity.backlog_after(invalid, 80, 100, 10)

    def test_replica_headroom_is_not_failure_reserve(self):
        replicas = capacity.replica_count(120, 50)
        self.assertLess((replicas - 1) * 50, 120)
        self.assertGreaterEqual(replicas * 0.8 * 50, 120)
        self.assertEqual(capacity.replica_count(0, 50), 0)
        for target in (0, -1, 2, math.nan):
            with self.subTest(target=target), self.assertRaises(ValueError):
                capacity.replica_count(120, 50, target)

    def test_kv_units_and_validation(self):
        self.assertEqual(capacity.kv_bytes(32, 8, 128, 2, 1), 128 * 1024)
        for length in (0, -1, 1.5, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                capacity.kv_bytes(32, 8, 128, 2, length)

    def test_weighting_and_empty_rank(self):
        self.assertEqual(capacity.token_weighted_mean([0, 3], [0, 6]), 3)
        self.assertEqual(capacity.token_weighted_mean([1, 3], [4, 4]), 2)
        for means, counts in (([], []), ([1], [0]), ([1, 2], [2]), ([1], [-1])):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                capacity.token_weighted_mean(means, counts)

    def test_other_worked_arithmetic(self):
        self.assertEqual(5 + max(30, 40) + 8 + 22 + 5 + 5, 85)
        self.assertEqual(300 + 200 + 200 - 100, 600)
        self.assertEqual(6 + 80 + 60 + 10 + 99 * 20, 2136)
        self.assertEqual(1 / 8 * 1000, 125)
        self.assertAlmostEqual(2000000 * (1 - 0.999), 2000)
        self.assertEqual(80 * 0.25, 20)


class RouteContentTests(unittest.TestCase):
    def test_bilingual_structure_examples_and_sources(self):
        sources = [ROOT / "practice/learning-routes.md"]
        for directory in (PRODUCTION, INFRA):
            sources.extend(source for source in (ROOT / directory).glob("*.md")
                           if not source.name.endswith(".en.md"))
        for source in sources:
            with self.subTest(source=source.name):
                chinese = source.read_text()
                english = source.with_name(source.stem + ".en.md").read_text()
                self.assertEqual(re.findall(r"\{#([^}]+)\}", chinese),
                                 re.findall(r"\{#([^}]+)\}", english))
                self.assertEqual(re.findall(r"\\\[(.*?)\\\]", chinese, re.S),
                                 re.findall(r"\\\[(.*?)\\\]", english, re.S))
                self.assertEqual(re.findall(r"```(?:bash|yaml)\n(.*?)```", chinese, re.S),
                                 re.findall(r"```(?:bash|yaml)\n(.*?)```", english, re.S))
                self.assertEqual(set(re.findall(r"https?://[^)\s]+", chinese)),
                                 set(re.findall(r"https?://[^)\s]+", english)))
                self.assertNotRegex(english.replace("中文", ""), r"[\u4e00-\u9fff]")
                self.assertNotRegex(chinese + english, r"/Users/|/home/|file://")
                self.assertEqual(chinese.count('class="system-flow"'), english.count('class="system-flow"'))
                self.assertEqual(chinese.count('<li>'), english.count('<li>'))
                self.assertEqual(chinese.count('<h3>'), english.count('<h3>'))

    def test_navigation_is_practice_and_stays_additive(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        for group_id, directory in (("production-ml", PRODUCTION), ("ai-platforms", INFRA)):
            group = next(group for group in nav["group"] if group["id"] == group_id)
            self.assertEqual(group["category"], "practice")
            section = next(section for section in nav["section"] if section["dir"] == directory)
            actual = {source.name for source in (ROOT / directory).glob("*.md")
                      if not source.name.endswith(".en.md")}
            self.assertEqual(actual, set(section["order"]))
        self.assertTrue((ROOT / "practice/recommender-systems/README.md").exists())
        self.assertTrue((ROOT / "06-systems/inference-frameworks.md").exists())


if __name__ == "__main__":
    unittest.main()
