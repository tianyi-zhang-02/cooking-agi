from pathlib import Path
import re
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class FoundationsNumericsTests(unittest.TestCase):
    def test_logistic_example_is_stable_in_both_languages(self):
        for suffix in ('.md', '.en.md'):
            source = ROOT / ('00-foundations/from-linear-to-neural' + suffix)
            snippet = re.findall(r'```python\n(.*?)```', source.read_text(), re.S)[0]
            namespace = {}
            exec(compile(snippet, str(source), 'exec'), namespace)
            with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
                actual = namespace['sigmoid']([-1000, -1, 0, 1, 1000])
            np.testing.assert_allclose(actual, [0, 1 / (1 + np.e), .5, 1 / (1 + np.exp(-1)), 1])
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertEqual(float(namespace['sigmoid'](0)), .5)


if __name__ == '__main__':
    unittest.main()
