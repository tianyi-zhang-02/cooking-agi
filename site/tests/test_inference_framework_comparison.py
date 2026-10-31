from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import paritycheck


class InferenceFrameworkComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chinese = (ROOT / "06-systems/inference-frameworks.md").read_text()
        cls.english = (ROOT / "06-systems/inference-frameworks.en.md").read_text()

    def test_structure_and_stable_anchors_match(self):
        self.assertEqual(paritycheck.shape(self.chinese), paritycheck.shape(self.english))
        anchors = re.findall(r"\{#([^}]+)\}", self.chinese)
        self.assertEqual(anchors, re.findall(r"\{#([^}]+)\}", self.english))
        self.assertEqual(len(anchors), len(set(anchors)))
        self.assertTrue({"ecosystem", "layers", "engine-map", "runtime-stack",
                         "deployment-map", "eight-gpus", "moe-deployment", "pp-kv",
                         "batching", "prefix", "hicache", "bottlenecks", "benchmark",
                         "interview-walkthrough", "further-reading"} <= set(anchors))

    def test_sources_and_examples_match(self):
        sources = lambda text: set(re.findall(r"https?://[^\s)<>]+", text))
        self.assertEqual(sources(self.chinese), sources(self.english))
        for text in (self.chinese, self.english):
            section = text.split("{#ecosystem}", 1)[1].split("## ", 1)[0]
            for source in (
                "https://docs.vllm.ai/en/latest/deployment/integrations/production-stack/",
                "https://docs.sglang.io/docs/advanced_features/sgl_model_gateway",
                "https://docs.vllm.ai/en/latest/deployment/integrations/llm-d/",
                "https://docs.sglang.io/docs/advanced_features/llm-d",
                "https://verl.readthedocs.io/en/latest/start/install.html",
            ):
                self.assertIn(source, section)
            for quantity in ("2,000-token", "100-token", "48 KiB", "192 MiB", "12 GiB/s"):
                self.assertIn(quantity, text)

    def test_english_body_has_no_chinese(self):
        body = re.sub(r"^\[中文\].*$", "", self.english, flags=re.M)
        self.assertNotRegex(body, r"[\u3400-\u9fff]")
