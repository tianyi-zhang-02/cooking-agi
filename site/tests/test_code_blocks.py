import html
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build


def lines_of(block_html):
    """The text of each numbered line, exactly as copying gives it."""
    return [html.unescape(re.sub(r"<[^>]+>", "", line))
            for line in re.findall(r'<span class="cl"[^>]*>(.*?)</span>(?=<span class="cl"|</code>)', block_html, re.S)]


def render(markdown_text, where="test.md"):
    raw, options = build.stash_code_options(markdown_text)
    body, _ = build.render_markdown(raw)
    return build.code_blocks(body, options, where)


LONG = "```python\n" + "".join(f"x{i} = {i}\n" for i in range(1, 10)) + "```\n"


class CodeLineTests(unittest.TestCase):
    """Long code blocks become numbered lines that copy back to exactly the original code."""

    def test_long_blocks_are_numbered_and_short_ones_left_alone(self):
        self.assertIn('class="hl code-lines"', render(LONG))
        short = render("```python\nx = 1\ny = 2\n```\n")
        self.assertNotIn("code-lines", short)
        plain = render("```\n" + "line\n" * 12 + "```\n")       # program output: no highlighting, no numbers
        self.assertNotIn("code-lines", plain)

    def test_a_docstring_over_several_lines_stays_balanced_and_exact(self):
        source = 'def f(x):\n    """first\n    second\n    third"""\n    return x  # done\n\n\n' + "y = 1\n" * 3
        body = render("```python\n" + source + "```\n")
        lines = lines_of(body)
        self.assertEqual("".join(lines), source)
        for line in re.findall(r'<span class="cl"[^>]*>(.*?)\n</span>', body, re.S):
            self.assertEqual(line.count("<span"), line.count("</span>"), line)

    def test_each_line_carries_its_indent_for_the_wrapped_continuation(self):
        body = render("```python\n" + "def f():\n" + "    if True:\n        pass\n" * 4 + "```\n")
        self.assertEqual(re.findall(r'style="--in:(\d+)"', body)[:3], ["0", "4", "8"])

    def test_folds_and_marks_come_from_the_comment_before_the_fence(self):
        text = '<!-- code fold="2-4: setup" mark="6:4 7:2,3" -->\n' + LONG
        body = render(text)
        self.assertNotIn("<!--", body)
        self.assertNotIn("code-options", body)
        self.assertEqual(len(re.findall(r'data-fold="0"', body)), 3)
        self.assertIn('data-fold-label="setup" data-fold-lines="2–4"', body)
        self.assertIn('data-mark="4"', body)
        self.assertIn('data-mark="2 3"', body)
        self.assertEqual("".join(lines_of(body)), "".join(f"x{i} = {i}\n" for i in range(1, 10)))

    def test_bad_options_fail_the_build(self):
        for text in ('<!-- code fold="5-40: too far" -->\n' + LONG,
                     '<!-- code mark="0:1" -->\n' + LONG,
                     '<!-- code colour="red" -->\n' + LONG,
                     '<!-- code mark="1:1" -->\n\n' + LONG,           # not right before the fence
                     '<!-- code mark="1:1" -->\n```bash\necho hi\n```\n'):
            with self.subTest(text=text[:40]):
                with self.assertRaises(ValueError):
                    render(text)


class ImplementationSetTests(unittest.TestCase):
    """Adjacent code-drop details are one set of implementations; code-step sets mark what changed."""

    def drops(self, codes, step=False):
        kind = " code-step" if step else ""
        return "\n\n".join(f'<details class="code-drop{kind}" markdown="1">\n<summary><b>V{i}</b> · note {i}</summary>\n\n'
                           f"```python\n{code}```\n\n</details>" for i, code in enumerate(codes)) + "\n"

    def test_adjacent_details_become_one_set(self):
        body = render(self.drops(["a = 1\n", "a = 2\n", "a = 3\n"]))
        self.assertEqual(body.count('<div class="code-variants">'), 1)
        self.assertEqual(body.count('<details class="code-drop">'), 3)
        single = render(self.drops(["a = 1\n"]))
        self.assertNotIn("code-variants", single)

    def test_steps_mark_changed_lines_but_not_renames_or_comment_edits(self):
        base = "def one(x):\n    y = x + 1  # add\n    return y\n"
        renamed = "def two(x):\n    y = x + 1  # add one\n    y = y * 2\n    return y\n"
        body = render(self.drops([base, renamed], step=True))
        self.assertIn('<div class="code-variants code-steps">', body)
        blocks = re.findall(r'<div class="hl code-lines( code-diff)?"><pre>(.*?)</pre>', body, re.S)
        self.assertEqual([kind for kind, _ in blocks], ["", " code-diff"])     # the first step has nothing to compare with
        marked = [i + 1 for i, tag in enumerate(re.findall(r'<span class="cl"[^>]*>', blocks[1][1])) if "data-new" in tag]
        self.assertEqual(marked, [3])

    def test_a_set_is_all_steps_or_none(self):
        mixed = self.drops(["a = 1\n"], step=True) + "\n" + self.drops(["a = 2\n"])
        with self.assertRaises(ValueError):
            render(mixed)


class AttentionPagesTests(unittest.TestCase):
    """The pages that use the component: multi-head attention and the five-way ladder."""

    @classmethod
    def setUpClass(cls):
        cls.old_nav, cls.old_sources = build.NAV.copy(), build.BY_SRC.copy()
        cls.nav = build.load_nav()
        cls.pages, _ = build.discover(cls.nav)
        cls.known = {page.url for page in cls.pages}

    @classmethod
    def tearDownClass(cls):
        build.NAV.clear()
        build.NAV.update(cls.old_nav)
        build.BY_SRC.clear()
        build.BY_SRC.update(cls.old_sources)

    def built(self, source, language):
        page = build.BY_SRC[source][language]
        build.build_page(page, [], self.nav["site"]["repo"], self.known)
        return page.body

    def test_attention_offers_three_implementations_with_the_six_places_marked(self):
        for language in ("zh", "en"):
            with self.subTest(language=language):
                body = self.built("00-foundations/core/multi-head-attention.md", language)
                group = re.search(r'<div class="code-variants">(.*?)</details></div>', body, re.S)
                self.assertIsNotNone(group)
                self.assertEqual(re.findall(r"<summary><b>(.*?)</b>", group[1])[2], "SDPA")
                numpy_block = re.search(r'<div class="hl code-lines"><pre>(.*?)</pre>', group[1], re.S)[1]
                marks = re.findall(r'data-mark="([^"]+)"', numpy_block)
                self.assertEqual(sorted(" ".join(marks).split()), ["1", "2", "3", "4", "5", "6"])
                self.assertEqual(len(re.findall(r'data-fold="0"', numpy_block)), 11)
                source = (build.ROOT / ("00-foundations/core/multi-head-attention.md" if language == "zh"
                                        else "00-foundations/core/multi-head-attention.en.md")).read_text(encoding="utf-8")
                first = re.search(r"```python\n(import numpy as np\n\ndef softmax\(logits.*?)```", source, re.S)[1]
                self.assertEqual("".join(lines_of(numpy_block)), first)      # copying gives the note's code exactly

    def test_the_ladder_highlights_only_what_each_step_changes(self):
        for language in ("zh", "en"):
            with self.subTest(language=language):
                body = self.built("learn/ml-exercises/attention-ladder.md", language)
                self.assertEqual(body.count('<details class="code-drop code-step">'), 5)
                blocks = re.findall(r'<div class="hl code-lines code-diff"><pre>(.*?)</pre>', body, re.S)
                changed = [[i + 1 for i, tag in enumerate(re.findall(r'<span class="cl"[^>]*>', block)) if "data-new" in tag]
                           for block in blocks]
                self.assertEqual(changed, [[4], [1, 2, 3], [1, 2, 3], [3, 4, 5, 7]])

    def test_the_script_loads_where_code_needs_it_and_copy_takes_every_line(self):
        template = (build.SITE / "template.html").read_text(encoding="utf-8")
        self.assertIn('<!-- if:code-lines|code-variants --><script defer src="{{prefix}}static/code-blocks.js', template)
        script = (build.SITE / "static/code-blocks.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", script)
        for needed in ("role', 'tablist'", "ArrowRight", "until-found", "aria-expanded"):
            self.assertIn(needed, script)
        app = (build.SITE / "static/app.js").read_text(encoding="utf-8")
        self.assertIn('var lines = pre.querySelectorAll(".cl");', app)


if __name__ == "__main__":
    unittest.main()
