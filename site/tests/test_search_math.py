"""Search excerpts read as prose: formulas never show up as raw TeX."""
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build

# TeX that a reader would see as noise; a Windows path like .venv\\Scripts in code is fine
TEX = r"\$\$|\\(?:t?frac|sqrt|text|operatorname|math[a-z]+|left|right|times|cdot|approx|sum|alpha|theta|pi|log|hat|top)(?![A-Za-z])"

class SearchMathTests(unittest.TestCase):
    def test_inline_formulas_become_readable_text(self):
        cases = {
            r"\sqrt{d_k}": "√d_k", r"\mathcal S": "S", r"W^\top x": "Wᵀ x", r"\hat r_t": "r̂_t",
            r"\tfrac{1}{2}": "1/2", r"\frac{1}{\sqrt{d}}": "1/√d", r"\left(\frac{a+b}{c}\right)": "((a+b)/c)",
            r"\text{softmax}(QK^\top/\sqrt{d_k})": "softmax(QKᵀ/√d_k)", r"x_1,\dots,x_n": "x_1,…,x_n",
            r"\operatorname{clip}(r,1-\epsilon,1+\epsilon)": "clip(r,1-ε,1+ε)", r"j\in\mathcal S": "j∈S",
        }
        for tex, text in cases.items():
            with self.subTest(tex=tex):
                self.assertEqual(build.plain_tex(tex), text)

    def test_display_math_is_left_out_and_code_keeps_its_dollars(self):
        prose = '值 <code>$HOME</code> 是 $A$，$$x^2 + y^2$$ 然后 \\(\\alpha\\) 和 $\\sqrt{d_k}$。'
        self.assertEqual(build.search_prose(prose), "值 $HOME 是 A， 然后 α 和 √d_k。")

    def test_built_indexes_carry_no_tex_commands(self):
        for language in ("zh", "en"):
            index = ROOT / f"_site/search-index.{language}.json"
            if not index.exists():
                self.skipTest("run site/build.py first")
            for entry in json.loads(index.read_text()):
                for _, heading, text, _ in entry["h"]:
                    with self.subTest(page=entry["u"]):
                        self.assertNotRegex(heading + text, TEX)


if __name__ == "__main__":
    unittest.main()
