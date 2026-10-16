from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


def render(relative):
    source = (ROOT / relative).read_text()
    source = re.sub(r"^#\s+.+$", "", source, count=1, flags=re.M)
    rendered, unused_toc = build.render_markdown(build.protect_mermaid(source))
    return rendered


class ReviewAnchorSemanticsTests(unittest.TestCase):
    def test_numbered_links_still_target_the_original_topic(self):
        cases = [
            ("05-post-training/README.md", "_7", "这个系列怎么读"),
            ("05-post-training/README.md", "_2", "各阶段的直观分工"),
            ("05-post-training/README.md", "_3", "从基础模型到对齐模型：先分清每一步在学什么"),
            ("05-post-training/README.md", "_4", "为什么数据经常比算法更先成为瓶颈"),
            ("05-post-training/README.md", "_5", "怎样选择方法"),
            ("05-post-training/README.md", "_6", "怎样评估"),
            ("02-memory/memory-lifecycle.md", "_9", "这套办法的代价"),
            ("05-post-training/dapo.md", "_2", "两种平均方式，可以直接算出不同结果"),
            ("05-post-training/dapo.md", "_3", "多卡上还有一次容易漏掉的加权"),
            ("05-post-training/dapo.md", "_4", "生成被截断，不等于推理一定错了"),
            ("05-post-training/deep-rl/planning-and-learning.md", "_7", "这里讲到哪里为止"),
            ("05-post-training/rlhf/rl-for-language-models.md", "_2", "映射到语言模型"),
            ("06-systems/README.md", "_6", "为什么要看完整链路"),
            ("06-systems/README.md", "_7", "排查前，先问自己这九个问题"),
            ("06-systems/README.md", "_8", "两个帮助系统持续改进的模块"),
            ("interview/study-plan.en.md", "inference-ml-systems_1", "Inference &amp; ML systems"),
        ]
        for relative, anchor, heading in cases:
            with self.subTest(source=relative, anchor=anchor):
                self.assertRegex(render(relative), rf'<h[23] id="{re.escape(anchor)}">{re.escape(heading)}')

    def test_aliases_target_the_topic_not_a_later_auto_numbered_heading(self):
        cases = [
            ("02-memory/memory-lifecycle.md", "_7", "deletion-propagation"),
            ("02-memory/memory-lifecycle.md", "_8", "timeline-test"),
            ("05-post-training/dapo.md", "_1", "dynamic-sampling"),
            ("05-post-training/deep-rl/planning-and-learning.md", "_6", "combinations"),
            ("05-post-training/rlhf/rl-for-language-models.md", "_1", "rl-vocabulary"),
        ]
        for relative, alias, target in cases:
            with self.subTest(source=relative, anchor=alias):
                self.assertRegex(render(relative), rf'<span id="{alias}"></span></p>\s*<h2 id="{target}">')

    def test_removed_subheadings_remain_next_to_their_explanations(self):
        for suffix in (".md", ".en.md"):
            distillation = render("05-post-training/model-adaptation" + suffix)
            for anchor, label in (
                ("response-distillation", "Response distillation"),
                ("logit-distribution-distillation", "Logit / distribution distillation"),
                ("on-policy-distillation", "On-policy distillation"),
            ):
                with self.subTest(language=suffix, anchor=anchor):
                    self.assertIn(f'<span id="{anchor}"></span><strong>{label}</strong>', distillation)
            plan = render("interview/study-plan" + suffix)
            self.assertIn('<span id="inference-ml-systems"></span><strong>Inference / ML Systems', plan)
        systems = render("06-systems/README.md")
        self.assertRegex(systems, r'<span id="_5"></span></p>\s*<p>确定性规则、参考答案或执行器')
