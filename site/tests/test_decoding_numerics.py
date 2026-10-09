import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DecodingNumericsTests(unittest.TestCase):
    def test_bilingual_nucleus_examples_and_boundaries(self):
        examples = []
        for suffix in ('.md', '.en.md'):
            source = ROOT / ('00-foundations/core/decoding' + suffix)
            code = re.findall(r'```python\n(.*?)```', source.read_text(), re.S)[0]
            examples.append(code)
            namespace = {}
            exec(compile(code, str(source), 'exec'), namespace)
            nucleus = namespace['nucleus_distribution']
            self.assertEqual(list(nucleus([math.log(.1), math.log(.6), math.log(.3)], .8)), [1, 2])
            self.assertEqual(len(nucleus([0, -1, -2], 1)), 3)
            self.assertEqual(nucleus([1000, -1000], .9), {0: 1.0})
            self.assertEqual(nucleus([3], .1), {0: 1.0})
            self.assertEqual(nucleus([3, 3], 1, .01), {0: .5, 1: .5})
            for threshold in (0, -1, 1.1, float('nan')):
                with self.assertRaises(ValueError):
                    nucleus([0, 1], threshold)
            for temperature in (0, -1, float('inf'), float('nan')):
                with self.assertRaises(ValueError):
                    nucleus([0, 1], temperature=temperature)
            with self.assertRaises(ValueError):
                nucleus([])
            probabilities = [.40, .25, .15, .08, .05, .03, .025, .015]
            logits = [math.log(value) for value in probabilities]
            for temperature, size in ((.7, 4), (1., 5), (1.5, 6)):
                self.assertEqual(len(nucleus(logits, .9, temperature)), size)
                powered = [value ** (1 / temperature) for value in probabilities]
                expected = [value / sum(powered) for value in powered]
                actual = nucleus(logits, 1, temperature)
                for token_id, probability in enumerate(expected):
                    self.assertAlmostEqual(actual[token_id], probability)
        self.assertEqual(*examples)


if __name__ == '__main__':
    unittest.main()
