from fractions import Fraction
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = "11-frontier/diffusion-lm/ar-and-diffusion"


def block_attention_mask(length, block_size):
    return [
        [key_position // block_size <= query_position // block_size
         for key_position in range(length)]
        for query_position in range(length)
    ]


def uniform_attention_layer(states, visibility):
    return [
        states[position] + sum(
            value for value, allowed in zip(states, row) if allowed
        ) / sum(row)
        for position, row in enumerate(visibility)
    ]


class DiffusionComparisonNotesTests(unittest.TestCase):
    def test_latency_example_distinguishes_steps_from_time(self):
        ar_latency = 6 * Fraction(1)
        slow_diffusion = 3 * Fraction(3)
        fast_diffusion = 3 * Fraction(3, 2)
        self.assertEqual((ar_latency, slow_diffusion, fast_diffusion), (6, 9, 4.5))
        self.assertGreater(slow_diffusion, ar_latency)
        self.assertLess(fast_diffusion, ar_latency)

    def test_two_block_visibility_matches_the_note(self):
        visibility = block_attention_mask(6, 3)
        self.assertEqual(visibility[:3], [[True] * 3 + [False] * 3] * 3)
        self.assertEqual(visibility[3:], [[True] * 6] * 3)

    def test_completed_prefix_is_invariant_but_active_token_is_not(self):
        visibility = block_attention_mask(6, 3)
        before = list(map(Fraction, [1, 2, 3, 4, 0, 6]))
        after = list(map(Fraction, [1, 2, 3, 4, 5, 6]))
        self.assertEqual(before[3], after[3])
        for _ in range(3):
            before = uniform_attention_layer(before, visibility)
            after = uniform_attention_layer(after, visibility)
            self.assertEqual(before[:3], after[:3])
            self.assertNotEqual(before[3], after[3])

    def test_last_masked_pass_is_not_the_clean_block_cache(self):
        visibility = block_attention_mask(3, 3)
        masked = list(map(Fraction, [1, 2, 0]))
        clean = list(map(Fraction, [1, 2, 3]))
        masked_hidden = uniform_attention_layer(masked, visibility)
        clean_hidden = uniform_attention_layer(clean, visibility)
        self.assertTrue(all(
            masked_value != clean_value
            for masked_value, clean_value in zip(masked_hidden, clean_hidden)
        ))

    def test_bilingual_note_is_in_the_reading_order(self):
        navigation = tomllib.loads((ROOT / "site/nav.toml").read_text(encoding="utf-8"))
        section = next(
            section for section in navigation["section"]
            if section["dir"] == "11-frontier/diffusion-lm"
        )
        order = section["order"]
        self.assertLess(order.index("ar-and-diffusion.md"), order.index("context-reuse.md"))
        self.assertEqual(order.index("context-reuse.md"), order.index("review.md") - 1)
        for suffix in (".md", ".en.md"):
            source = (ROOT / (CHAPTER + suffix)).read_text(encoding="utf-8")
            for reference in ("2503.09573v2", "2509.26328v1", "2510.18087v2", "2610.17590v1"):
                self.assertIn(reference, source)
            self.assertIn("The cats are | on the mat", source)
            overview = (ROOT / "11-frontier/diffusion-lm" / ("README" + suffix)).read_text(encoding="utf-8")
            self.assertIn("ar-and-diffusion" + suffix, overview)

    def test_english_body_has_no_untranslated_chinese(self):
        source = (ROOT / (CHAPTER + ".en.md")).read_text(encoding="utf-8")
        source = source.replace("[中文](ar-and-diffusion.md)", "")
        self.assertIsNone(re.search(r"[\u3400-\u9fff]", source))


if __name__ == "__main__":
    unittest.main()
