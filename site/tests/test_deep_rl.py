import ast
import importlib.util
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "05-post-training/deep-rl"
sys.path.insert(0, str(ROOT / "site"))
import paritycheck

SPEC = importlib.util.spec_from_file_location("rl_checks", NOTES / "code/rl_checks.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class DeepRLTests(unittest.TestCase):
    def test_worked_examples(self):
        REFERENCE.run_checks()

    def test_gae_full_episode_telescopes(self):
        rewards = [1, 0, 2]
        values = [0.4, 0.3, 0.2]
        advantages = REFERENCE.gae(rewards, values, [0.3, 0.2, 0],
                                   [False, False, True], [False] * 3, 0.9, 1.0)
        for step, advantage in enumerate(advantages):
            expected = REFERENCE.discounted_return(rewards[step:], 0.9) - values[step]
            self.assertAlmostEqual(advantage, expected)

    def test_zero_lambda_is_one_step_td(self):
        self.assertEqual(REFERENCE.gae([1], [2], [5], [False], [False], 0.9, 0),
                         [3.5])

    def test_truncated_gae_keeps_bootstrap_tail(self):
        rewards = [1, 0, 2]
        values = [0.4, 0.3, 0.2]
        advantages = REFERENCE.gae(rewards, values, [0.3, 0.2, 5],
                                   [False] * 3, [False, False, True], 0.9, 1.0)
        for step, advantage in enumerate(advantages):
            tail = 0.9 ** (len(rewards) - step) * 5
            expected = REFERENCE.discounted_return(rewards[step:], 0.9) + tail - values[step]
            self.assertAlmostEqual(advantage, expected)

    def test_bandit_gradient_matches_finite_difference(self):
        def expected_reward(logits):
            normalizer = sum(math.exp(logit) for logit in logits)
            return sum(math.exp(logit) * reward for logit, reward in zip(logits, [1, 3])) / normalizer

        step = 1e-5
        for index, expected in enumerate([-0.5, 0.5]):
            positive = [0.0, 0.0]
            negative = [0.0, 0.0]
            positive[index] = step
            negative[index] = -step
            derivative = (expected_reward(positive) - expected_reward(negative)) / (2 * step)
            self.assertAlmostEqual(derivative, expected)

    def test_model_rollout_bound_includes_initial_error(self):
        for initial_error in (0, 1):
            for lipschitz in (0.5, 1, 1.2):
                actual = 0.0
                predicted = float(initial_error)
                error_limit = 0.1
                for horizon in range(1, 7):
                    actual *= lipschitz
                    predicted = lipschitz * predicted + error_limit
                    bound = lipschitz ** horizon * initial_error + error_limit * sum(
                        lipschitz ** offset for offset in range(horizon))
                    self.assertAlmostEqual(abs(predicted - actual), bound)

    def test_published_python_snippets_match_reference(self):
        def function_tree(source, name):
            return ast.dump(next(node for node in ast.parse(source).body
                                 if isinstance(node, ast.FunctionDef) and node.name == name))

        for chapter, name, source_file in (
            ("policy-gradients", "actor_loss", "torch_updates.py"),
            ("actor-critic-gae", "gae", "rl_checks.py"),
            ("dqn", "double_dqn_loss", "torch_updates.py"),
        ):
            reference = function_tree((NOTES / "code" / source_file).read_text(), name)
            for suffix in (".md", ".en.md"):
                snippets = re.findall(r"```python\n(.*?)```", (NOTES / (chapter + suffix)).read_text(), re.S)
                self.assertEqual(function_tree("\n".join(snippets), name), reference, chapter + suffix)

    def test_actor_loss_rejects_broadcastable_or_empty_inputs(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch required")
        import torch
        spec = importlib.util.spec_from_file_location("torch_updates", NOTES / "code/torch_updates.py")
        updates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(updates)
        for log_prob, advantage in ((torch.zeros(2, 1), torch.zeros(2)),
                                     (torch.zeros(2, 1), torch.zeros(2, 1)),
                                     (torch.zeros(0), torch.zeros(0))):
            with self.assertRaises(ValueError):
                updates.actor_loss(log_prob, advantage)
        log_prob = torch.tensor([-0.2, -0.5], requires_grad=True)
        advantage = torch.tensor([1.0, -1.0], requires_grad=True)
        updates.actor_loss(log_prob, advantage).backward()
        self.assertTrue(torch.equal(log_prob.grad, torch.tensor([-0.5, 0.5])))
        self.assertIsNone(advantage.grad)

    def test_finite_gae_mixture_preserves_tail_weight(self):
        residuals = [1, 2, -1]
        gamma = 0.9
        for decay in (0, 0.3, 0.8, 1):
            estimates = [sum(gamma ** offset * value for offset, value in enumerate(residuals[:length]))
                         for length in (1, 2, 3)]
            weights = [1 - decay, (1 - decay) * decay, decay ** 2]
            self.assertAlmostEqual(sum(weights), 1)
            mixture = sum(weight * value for weight, value in zip(weights, estimates))
            trace = sum((gamma * decay) ** offset * value for offset, value in enumerate(residuals))
            self.assertAlmostEqual(mixture, trace)

    def test_hand_calculated_control_examples(self):
        self.assertAlmostEqual(1 + 0.9 * (4 - 0.2 * -0.7), 4.726)
        actions = [-2, -1, 0, 1]
        costs = [(2 + action) ** 2 + 0.1 * action ** 2 for action in actions]
        for actual, expected in zip(costs, [0.4, 1.1, 4, 9.1]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(actions[:2]) / 2, -1.5)
        self.assertAlmostEqual(math.sqrt(sum((action + 1.5) ** 2 for action in actions[:2]) / 2), 0.5)
        for latent in (-3, 0, 3):
            stable = 2 * (math.log(2) - latent - math.log1p(math.exp(-2 * latent)))
            self.assertAlmostEqual(stable, math.log(1 - math.tanh(latent) ** 2))

    def test_terminal_and_truncation_do_not_cross_resets(self):
        for terminated, truncated, expected in [(True, False, 1), (False, True, 5.5)]:
            result = REFERENCE.gae([1, 999], [0, 0], [5, 0],
                                   [terminated, True], [truncated, False], 0.9, 0.8)
            self.assertEqual(result[0], expected)

    def test_invalid_reference_inputs(self):
        with self.assertRaises(ValueError):
            REFERENCE.gae([1], [], [0], [True], [False])
        for weights in ([], [0, 0], [-1, 2], [math.nan]):
            with self.assertRaises(ValueError):
                REFERENCE.effective_sample_size(weights)

    def test_ess_is_scale_invariant_at_extreme_magnitudes(self):
        for scale in (1e-300, 1e-200, 1, 1e200, 1e300):
            self.assertAlmostEqual(REFERENCE.effective_sample_size([scale, scale, 8 * scale]), 100 / 66)
            self.assertEqual(REFERENCE.effective_sample_size([0, scale]), 1)
            self.assertEqual(REFERENCE.effective_sample_size([scale] * 3), 3)

    def test_action_independent_baseline_cancels_in_expected_gradient(self):
        probabilities = [.25, .75]
        score_gradients = [1 - probabilities[0], -probabilities[0]]
        baseline_error = -2
        self.assertAlmostEqual(sum(probability * gradient * baseline_error
                                   for probability, gradient in zip(probabilities, score_gradients)), 0)
        action_dependent = [-3, 1]
        self.assertNotEqual(sum(probability * gradient * error
                               for probability, gradient, error in zip(probabilities, score_gradients, action_dependent)), 0)

    def test_value_targets_use_raw_not_normalized_advantages(self):
        values = [10, 10]
        raw = REFERENCE.gae([11, 13], values, [0, 0], [True, True], [False, False])
        self.assertEqual(raw, [1, 3])
        self.assertEqual([value + advantage for value, advantage in zip(values, raw)], [11, 13])
        centered = [advantage - sum(raw) / len(raw) for advantage in raw]
        self.assertEqual([value + advantage for value, advantage in zip(values, centered)], [9, 11])

    def load_torch_updates(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch required")
        spec = importlib.util.spec_from_file_location("torch_updates", NOTES / "code/torch_updates.py")
        updates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(updates)
        return updates

    def test_dqn_rejects_silent_broadcast_and_wrong_dtypes(self):
        updates = self.load_torch_updates()
        import torch
        network = torch.nn.Linear(2, 2)
        states = torch.ones(2, 2)
        actions = torch.tensor([0, 1])
        rewards = torch.ones(2)
        terminated = torch.tensor([False, True])
        invalid = [(actions, rewards[:, None], terminated),
                   (actions[:, None], rewards, terminated),
                   (actions, rewards, terminated[:, None]),
                   (actions.float(), rewards, terminated),
                   (actions, rewards.long(), terminated),
                   (actions, rewards, terminated.long())]
        for current_actions, current_rewards, current_terminated in invalid:
            with self.assertRaises(ValueError):
                updates.double_dqn_loss(network, network, states, current_actions,
                                        current_rewards, states, current_terminated)
        with self.assertRaises(ValueError):
            updates.double_dqn_loss(network, network, states, actions, rewards,
                                    states, terminated, float('nan'))

    def test_dqn_never_evaluates_terminal_successors(self):
        updates = self.load_torch_updates()
        import torch

        class TableNetwork(torch.nn.Module):
            def __init__(self, table):
                super().__init__()
                self.table = torch.nn.Parameter(torch.tensor(table, dtype=torch.float64))

            def forward(self, states):
                if not torch.isfinite(states).all():
                    raise ValueError("invalid successor was evaluated")
                return self.table[states[:, 0].long()]

        online = TableNetwork([[.5, .25], [5, 4]])
        target = TableNetwork([[0, 0], [2, 6]])
        states = torch.tensor([[0.], [0.]])
        next_states = torch.tensor([[float('nan')], [1.]])
        actions = torch.tensor([0, 1])
        rewards = torch.tensor([1., 1.], dtype=torch.float64)
        loss = updates.double_dqn_loss(online, target, states, actions, rewards,
                                       next_states, torch.tensor([True, False]), .9)
        self.assertAlmostEqual(loss.item(), (.125 + 2.05) / 2)
        loss.backward()
        self.assertIsNone(target.table.grad)
        self.assertEqual(online.table.grad[1].abs().sum().item(), 0)
        self.assertLess(online.table.grad[0, 0], 0)
        self.assertLess(online.table.grad[0, 1], 0)
        for flags, gamma in [(torch.tensor([True, True]), .9), (torch.tensor([False, False]), 0)]:
            result = updates.double_dqn_loss(online, target, states, actions, rewards,
                                             torch.full_like(states, float('nan')), flags, gamma)
            expected = torch.nn.functional.smooth_l1_loss(torch.tensor([.5, .25], dtype=torch.float64), rewards)
            self.assertAlmostEqual(result.item(), expected.item())

    def test_projected_regression_improves_fit_but_grows_values(self):
        parameter = 1.0
        updated = REFERENCE.projected_value_step(parameter)
        target = 1.8
        old_loss = sum((feature * parameter - target) ** 2 for feature in (1, 2)) / 2
        new_loss = sum((feature * updated - target) ** 2 for feature in (1, 2)) / 2
        self.assertAlmostEqual(old_loss, 0.34)
        self.assertAlmostEqual(new_loss, 0.324)
        self.assertLess(new_loss, old_loss)
        for iteration in range(10):
            parameter = REFERENCE.projected_value_step(parameter)
        self.assertAlmostEqual(parameter, 1.08 ** 10)
        self.assertAlmostEqual(REFERENCE.projected_value_step(1, 0.5), 0.6)
        for parameter, gamma in [(math.inf, 0.9), (1, -1), (1, math.nan), (1, 1)]:
            with self.assertRaises(ValueError):
                REFERENCE.projected_value_step(parameter, gamma)

    def test_fisher_step_and_exact_kl(self):
        step = math.sqrt(2 * 0.01 / 0.25)
        probability = 1 / (1 + math.exp(-step))
        self.assertAlmostEqual(0.5 * 0.25 * step ** 2, 0.01)
        self.assertLess(REFERENCE.bernoulli_kl(0.5, probability), 0.01)
        self.assertGreater(REFERENCE.bernoulli_kl(0.5, 1 / (1 + math.exp(-2))), 0.43)
        self.assertAlmostEqual(REFERENCE.bernoulli_kl(0.5, 0.5), 0)
        for probability in (0, 1, math.nan, -0.1, math.inf):
            with self.assertRaises(ValueError):
                REFERENCE.bernoulli_kl(0.5, probability)

    def test_bilingual_structure_and_private_material_excluded(self):
        pages = [path for path in NOTES.glob("*.md") if not path.name.endswith(".en.md")]
        self.assertEqual(len(pages), 17)
        for chinese in pages:
            english = chinese.with_name(chinese.stem + ".en.md")
            self.assertTrue(english.exists(), chinese.name)
            self.assertEqual(paritycheck.shape(chinese.read_text()),
                             paritycheck.shape(english.read_text()), chinese.name)
            for path in (chinese, english):
                text = path.read_text()
                self.assertNotIn("gatech.instructure.com", text)
                self.assertNotIn("docviewer", text)
                self.assertNotIn("data-review-deck", text)
                self.assertNotRegex(text, r"lectures_export|lec-\d-cs8803")
        self.assertFalse(list(NOTES.rglob("*.pdf")))

    def test_three_nav_sections_claim_every_chapter_once(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        group = next(group for group in nav["group"] if group["id"] == "deep-rl")
        self.assertEqual(group["category"], "learn")
        sections = [section for section in nav["section"] if section.get("group") == "deep-rl"]
        self.assertEqual(len(sections), 4)
        claimed = [path for section in sections for path in section.get("include", [])]
        self.assertEqual(len(claimed), len(set(claimed)))
        expected = {path.relative_to(ROOT).as_posix() for path in NOTES.glob("*.md")
                    if not path.name.endswith(".en.md") and path.name != "README.md"}
        self.assertEqual(set(claimed), expected)

    def test_relative_note_links_exist(self):
        for page in NOTES.glob("*.md"):
            for url in re.findall(r"\]\(([^)\s]+)\)", page.read_text()):
                if "://" not in url and not url.startswith("#"):
                    self.assertTrue((page.parent / url.split("#")[0]).exists(),
                                    f"{page.name}: {url}")

    def test_interactive_arithmetic_and_input_validation(self):
        if not shutil.which("node"):
            self.skipTest("Node.js required")
        script = """
const assert = require('node:assert/strict');
const lab = require(process.argv[1]);
assert.deepEqual(lab.bellman(.9, 0), {start:0, charged:0});
assert.deepEqual(lab.bellman(.9, 2), {start:3.6, charged:4});
assert.deepEqual(lab.bellman(.4, 2), {start:2, charged:4});
assert.equal(lab.boundaryTarget(1,5,.9,true), 1);
assert.equal(lab.boundaryTarget(1,5,.9,false), 5.5);
assert.ok(Math.abs(lab.gaeTerms(.9,.8,[1,2,-1]).reduce((sum,value)=>sum+value,0)-1.9216)<1e-10);
assert.equal(lab.doubleTargets([5,4],[2,6],.9).action,0);
assert.equal(lab.doubleTargets([3,4],[2,6],.9).action,1);
assert.equal(lab.doubleTargets([4,4],[2,6],.9).action,0);
assert.ok(lab.entropyPolicy(.1).probabilities[1] > lab.entropyPolicy(1).probabilities[1]);
assert.ok(lab.entropyPolicy(.1).entropy < lab.entropyPolicy(1).entropy);
assert.throws(()=>lab.bellman(-.1,1));
assert.throws(()=>lab.bellman(.9,1.2));
assert.throws(()=>lab.gaeTerms(.9,2,[1]));
assert.throws(()=>lab.doubleTargets([1],[],.9));
assert.throws(()=>lab.boundaryTarget(1,5,.9,'true'));
assert.throws(()=>lab.entropyPolicy(0));
assert.equal(lab.projectionTrace(.9,0).length,1);
assert.ok(Math.abs(lab.projectionTrace(.9,2)[2].parameter-1.1664)<1e-10);
assert.ok(Math.abs(lab.projectionTrace(.5,2)[2].parameter-.36)<1e-10);
for(const row of lab.projectionTrace(.99,20).slice(1)) assert.ok(row.lossAfter<=row.lossBefore+1e-10);
assert.throws(()=>lab.projectionTrace(.9,21));
assert.throws(()=>lab.projectionTrace(1,2));
assert.equal(lab.trustStep(0,.01).exact,0);
assert.equal(lab.trustStep(.3,.01).accepted,false);
assert.equal(lab.trustStep(.3,.02).accepted,true);
assert.ok(Math.abs(lab.trustStep(.3,.01).exact-.011208063908581797)<1e-10);
assert.throws(()=>lab.trustStep(-1,.01));
assert.throws(()=>lab.trustStep(1,0));
assert.deepEqual(lab.planSequence(3,1,1).actions,[-1]);
assert.ok(Math.abs(lab.planSequence(3,3,1).cost-5.3)<1e-10);
for(const horizon of [1,2,3,4,5,6]) {
  for(const gain of [.5,1,1.5]) {
    const plan=lab.planSequence(3,horizon,gain);
    assert.equal(plan.candidates,3**horizon);
    assert.equal(plan.actions.length,horizon);
    assert.equal(plan.states.length,horizon+1);
    let recomputed=0;
    plan.actions.forEach((action,index)=>{
      assert.ok([-1,0,1].includes(action));
      assert.ok(Math.abs(plan.states[index+1]-plan.states[index]-gain*action)<1e-10);
      recomputed+=plan.states[index+1]**2+.1*action**2;
    });
    assert.ok(Math.abs(recomputed-plan.cost)<1e-10);
  }
}
const exact=lab.planningComparison(5,1);
assert.deepEqual(exact.openStates,exact.predicted);
assert.deepEqual(exact.openStates,exact.feedbackStates);
const wrong=lab.planningComparison(5,1.5);
assert.deepEqual(wrong.openStates,[3,2,1,1,1,1]);
assert.deepEqual(wrong.feedbackStates,[3,2,1,0,0,0]);
assert.ok(Math.abs(wrong.openCost-8.2)<1e-10);
assert.ok(Math.abs(wrong.feedbackCost-5.3)<1e-10);
assert.throws(()=>lab.planSequence(3,7,1));
assert.throws(()=>lab.planSequence(3,2,0));
assert.throws(()=>lab.planSequence(NaN,2,1));
"""
        result = subprocess.run([shutil.which("node"), "-e", script,
                                 str(ROOT / "site/static/deep-rl-lab.js")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
