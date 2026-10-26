from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


class PosttrainingReadabilityTests(unittest.TestCase):
    def source(self, name, suffix=""):
        return (ROOT / "05-post-training" / f"{name}{suffix}.md").read_text(encoding="utf-8")

    def test_sft_card_translations_match_the_english_chapter(self):
        chinese = self.source("sft-and-its-ceiling")
        english = self.source("sft-and-its-ceiling", ".en")
        faces = re.findall(
            r'<div class="concept-face concept-en" data-concept-en markdown="1">\s*'
            r'<div[^>]*>.*?</div>\s*(.*?)\s*</div>\s*</section>',
            chinese,
            re.S,
        )
        sections = re.split(r"^### [123]\. .*$", english, flags=re.M)[1:]
        self.assertEqual(len(faces), 3)
        self.assertEqual(len(sections), 3)
        for number, (face, section) in enumerate(zip(faces, sections), 1):
            with self.subTest(card=number):
                section = re.split(r"^## ", section, flags=re.M)[0]
                self.assertEqual(" ".join(face.split()), " ".join(section.split()))

    def test_explicit_sft_glosses_are_not_repeated_by_the_glossary(self):
        source = self.source("sft-and-its-ceiling")
        markup, _ = build.render_markdown(source)
        body, _ = build.annotate(markup, build.load_glossary())
        for gloss in (
            "supervised fine-tuning，SFT",
            "reinforcement learning，RL",
            "loss mask",
            "attention mask",
            "teacher forcing",
        ):
            with self.subTest(gloss=gloss):
                self.assertEqual(body.count(f"（{gloss}）"), 1)
        for term in ("监督微调", "强化学习", "教师强制"):
            self.assertNotRegex(body, rf'>{term}<span class="term-en">')

    def test_chinese_sft_heading_changes_keep_existing_anchors(self):
        markup, _ = build.render_markdown(self.source("sft-and-its-ceiling"))
        self.assertIn('<h3 id="mask-user">用户消息不算损失，模型怎么学着理解问题？</h3>', markup)
        self.assertIn('<h3 id="_5">多轮对话和工具调用，哪些内容参与训练？</h3>', markup)

    def test_adaptation_diagrams_have_the_same_connections(self):
        diagrams = [
            re.search(r"```mermaid\n(.*?)```", self.source("model-adaptation", suffix), re.S).group(1)
            for suffix in ("", ".en")
        ]
        edges = [re.findall(r"^\s*(\w+)(?:\[.*?\]|\{.*?\})?\s*-->\s*(\w+)", diagram, re.M)
                 for diagram in diagrams]
        self.assertEqual(edges[0], edges[1])
        self.assertEqual(len(edges[0]), 12)
        self.assertIn("用什么训练信号", diagrams[0])
        self.assertIn("允许更新哪些参数", diagrams[0])
        self.assertNotIn("Learning objective", diagrams[0])
        self.assertNotRegex(diagrams[1], r"[\u4e00-\u9fff]")


if __name__ == "__main__":
    unittest.main()
