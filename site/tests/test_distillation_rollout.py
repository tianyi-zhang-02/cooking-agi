import contextlib
import io
import itertools
import math
from pathlib import Path
import re
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[2]
DISTILLATION = "05-post-training/distillation"
ROLLOUT = "05-post-training/post-training-infrastructure"


def examples(chapter, suffix=".md"):
    return re.findall(r"```python\n(.*?)```", (ROOT / (chapter + suffix)).read_text(), re.S)


def namespace_for(chapter, marker):
    source = next(source for source in examples(chapter) if marker in source)
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, chapter, "exec"), namespace)
    return namespace


def byte_partitions(value):
    for cuts in itertools.product((False, True), repeat=len(value) - 1):
        starts = [0] + [index + 1 for index, cut in enumerate(cuts) if cut] + [len(value)]
        yield [value[left:right] for left, right in zip(starts, starts[1:])]


class DistillationRolloutTests(unittest.TestCase):
    def test_bilingual_examples_match_and_execute(self):
        for chapter in (DISTILLATION, ROLLOUT):
            self.assertEqual(examples(chapter), examples(chapter, ".en.md"))
            for source in examples(chapter):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(source, chapter, "exec"), {})

    def test_figures_follow_page_language(self):
        for chapter in (DISTILLATION, ROLLOUT):
            for suffix in (".md", ".en.md"):
                source = (ROOT / (chapter + suffix)).read_text()
                figures = re.findall(r'<figure class="worked-update.*?</figure>', source, re.S)
                self.assertEqual(len(figures), 1)
                self.assertEqual(figures[0].count("<li>"), 2)
                self.assertEqual(bool(re.search(r"[\u4e00-\u9fff]", figures[0])), suffix == ".md")

    def test_alignment_keeps_every_common_boundary(self):
        align = namespace_for(DISTILLATION, "def aligned_spans")["aligned_spans"]
        for student in byte_partitions(b"abcd"):
            for teacher in byte_partitions(b"abcd"):
                spans = align(student, teacher)
                student_ends = set(itertools.accumulate(map(len, student)))
                teacher_ends = set(itertools.accumulate(map(len, teacher)))
                offsets = []
                previous_student = previous_teacher = 0
                for left, right, start, stop in spans:
                    self.assertEqual((left, start), (previous_student, previous_teacher))
                    self.assertEqual(b"".join(student[left:right]), b"".join(teacher[start:stop]))
                    offsets.append(sum(map(len, student[:right])))
                    previous_student, previous_teacher = right, stop
                self.assertEqual(set(offsets), student_ends & teacher_ends)
                self.assertEqual((previous_student, previous_teacher), (len(student), len(teacher)))

    def test_alignment_preserves_partial_unicode_bytes(self):
        align = namespace_for(DISTILLATION, "def aligned_spans")["aligned_spans"]
        encoded = "雨".encode("utf-8")
        self.assertEqual(align([encoded[:1], encoded[1:]], [encoded]), [(0, 2, 0, 1)])

    def test_alignment_rejects_mismatches_empty_and_text_pieces(self):
        align = namespace_for(DISTILLATION, "def aligned_spans")["aligned_spans"]
        invalid = (([], []), ([b""], [b""]), ([b"ab"], [b"ac"]), (["ab"], [b"ab"]))
        for student, teacher in invalid:
            with self.assertRaises(ValueError):
                align(student, teacher)

    def test_proportional_span_targets_preserve_total_not_conditionals(self):
        old = [-.7, -.9]
        targets = [value * (-1.2 / sum(old)) for value in old]
        self.assertAlmostEqual(sum(targets), -1.2)
        self.assertAlmostEqual(targets[0], -.525)
        self.assertAlmostEqual(targets[1], -.675)
        self.assertAlmostEqual(sum(target - value for target, value in zip(targets, old)), .4)
        self.assertNotEqual(targets, [-.6, -.6])

    def test_canonical_path_is_not_text_marginal(self):
        paths = [(b"ab", .2), (b"a" + b"b", .3), (b"ac", .5)]
        marginal = sum(probability for text, probability in paths if text == b"ab")
        self.assertAlmostEqual(marginal, .5)
        self.assertNotAlmostEqual(marginal, paths[0][1])

    def test_score_gradient_matches_finite_difference_not_naive_gradient(self):
        values = namespace_for(DISTILLATION, "score_gradient")
        self.assertAlmostEqual(values["finite_difference"], values["score_gradient"][0], places=9)
        self.assertAlmostEqual(sum(values["score_gradient"]), 0)
        self.assertEqual(values["naive_gradient"], [0., 0.])

    def test_choosing_one_teacher_mode_has_nonzero_reverse_kl(self):
        self.assertAlmostEqual(math.log(1 / .5), math.log(2))
        self.assertGreater(math.log(1 / .5), 0)

    def test_action_average_ignores_context_and_padding(self):
        average = namespace_for(ROLLOUT, "def action_mean")["action_mean"]
        self.assertEqual(average([999., 2., 4., -999.], [0, 1, 1, 0]), 3.)
        self.assertEqual(average([float("nan"), 2., 4.], [0, 1, 1]), 3.)

    def test_action_average_rejects_missing_and_misaligned_actions(self):
        average = namespace_for(ROLLOUT, "def action_mean")["action_mean"]
        for values, mask in (([], []), ([1], [0]), ([1], [1, 1]), ([1], [.5])):
            with self.assertRaises(ValueError):
                average(values, mask)

    def test_termination_and_collection_cutoff_have_distinct_targets(self):
        reward, discount, next_value = .2, .9, .8
        terminal_target = reward
        truncated_target = reward + discount * next_value
        self.assertAlmostEqual(terminal_target, .2)
        self.assertAlmostEqual(truncated_target, .92)
        for suffix in (".md", ".en.md"):
            self.assertIn("0.92", (ROOT / (ROLLOUT + suffix)).read_text())

    def test_temperature_changes_behavior_ratio(self):
        weights = [math.sqrt(probability) for probability in [.8, .2]]
        behavior = [weight / sum(weights) for weight in weights]
        self.assertAlmostEqual(behavior[1], 1 / 3)
        self.assertAlmostEqual(.25 / behavior[1], .75)
        self.assertAlmostEqual(.25 / .2, 1.25)

    def test_zero_mean_log_difference_does_not_imply_unit_ratios(self):
        differences = [.7, -.7]
        self.assertEqual(sum(differences), 0)
        ratios = [math.exp(value) for value in differences]
        self.assertGreater(ratios[0], 2)
        self.assertLess(ratios[1], .5)
        self.assertGreater(sum(ratios) / 2, 1)

    def test_ideal_pipeline_interval_is_not_first_batch_latency(self):
        stages, synchronization = [30, 10, 20], 5
        latency = sum(stages) + synchronization
        interval = max(stages) + synchronization
        self.assertEqual(latency, 65)
        self.assertEqual(interval, 35)
        self.assertAlmostEqual(latency / interval, 13 / 7)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_forward_kl_penalizes_teacher_zero_token_through_normalization(self):
        logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        loss = -(torch.tensor([1., 0.]) * logits.log_softmax(-1)).sum()
        gradient = torch.autograd.grad(loss, logits)[0]
        torch.testing.assert_close(gradient, torch.tensor([-.5, .5], dtype=torch.float64))

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_full_sequence_gradient_contains_future_cost(self):
        logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        first = logits.softmax(-1)
        second = torch.full((2, 2), .5, dtype=torch.float64)
        teacher = torch.tensor([[.45, .05], [.25, .25]], dtype=torch.float64)
        joint = first[:, None] * second
        log_ratio = joint.log() - teacher.log()
        exact = torch.autograd.grad((joint * log_ratio).sum(), logits, retain_graph=True)[0]
        returns = log_ratio.detach()
        score_loss = (joint.detach() * returns * joint.log()).sum()
        score = torch.autograd.grad(score_loss, logits, retain_graph=True)[0]
        immediate = first.log() - math.log(.5)
        local_loss = (first.detach() * immediate.detach() * first.log()).sum()
        local = torch.autograd.grad(local_loss, logits)[0]
        torch.testing.assert_close(exact, score)
        torch.testing.assert_close(local, torch.zeros_like(local))
        self.assertGreater(exact[0].item(), .1)


if __name__ == "__main__":
    unittest.main()
