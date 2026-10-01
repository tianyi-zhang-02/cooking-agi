from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SearchTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is required for search tests")
    def test_search_interactions(self):
        result = subprocess.run([shutil.which("node"), str(ROOT / "tests/search-behavior.cjs")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node is required for search tests")
    def test_local_search_ranking_and_safety(self):
        script = r"""
const assert = require('node:assert/strict');
const {normalize, safePath, prepareIndex, rankNotes, excerpt} = require(process.argv[1]);
const notes = prepareIndex([
  {u: 'core/attention.html', t: '注意力', s: '大模型基础', l: 'zh', x: '注意力 attention 使用 mask 和 softmax'},
  {u: 'core/attention.en.html', t: 'Attention mechanisms', s: 'Foundations', l: 'en', x: 'Attention uses masks and softmax'},
  {u: 'other.html', t: 'Other models', s: '基础', l: 'zh', x: 'Also mentions attention'},
  {u: 'clip.html', t: 'CLIP 图文学习', s: '多模态', l: 'zh', x: 'A contrastive objective'},
  {u: 'javascript:alert(1)', t: 'Attention', s: '', l: 'zh', x: ''},
  {u: 'bad.html', t: {}, s: '', l: 'en', x: ''},
  null
]);
assert.equal(notes.length, 4);
assert.equal(normalize(' ＣＬＩＰ '), 'clip');
assert.equal(rankNotes(notes, 'attention', 'zh', false)[0].u, 'core/attention.html');
assert.equal(rankNotes(notes, 'mechanisms', 'zh', false)[0].u, 'core/attention.html');
assert.equal(rankNotes(notes, 'attention', 'en', false).length, 1);
assert.equal(rankNotes(notes, 'attention', 'zh', true).length, 3);
assert.equal(rankNotes(notes, 'attention mask', 'zh', false).length, 1);
assert.equal(rankNotes(notes, 'ＡＴＴＥＮＴＩＯＮ', 'zh', false).length, 2);
assert.equal(rankNotes(notes, '图文', 'zh', false)[0].u, 'clip.html');
assert.equal(rankNotes(notes, 'does-not-exist', 'zh', true).length, 0);
assert.equal(rankNotes(notes, '   ', 'zh', true).length, 0);
assert.equal(rankNotes(notes, 'a+b [', 'zh', true).length, 0);
for (const path of ['../x.html', '/x.html', '//evil/x.html', 'https://evil/x.html', 'x.html?bad', 'x//y.html', 'x%2Fy.html']) assert.equal(safePath(path), false, path);
assert.equal(safePath('core/a-note.en.html'), true);
assert.throws(() => prepareIndex({}));
assert.ok(excerpt('before '.repeat(50) + 'attention is useful', 'attention').includes('attention'));
assert.ok(excerpt('a'.repeat(1000), 'missing').length <= 152);
assert.equal(excerpt(' some\n  whitespace ', 'some'), 'some whitespace');
"""
        result = subprocess.run([shutil.which("node"), "-e", script,
                                 str(ROOT / "static/search.js")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_search_uses_a_labelled_modal_and_progressive_controls(self):
        template = (ROOT / "template.html").read_text()
        self.assertIn('aria-labelledby="note-search-title"', template)
        self.assertIn('aria-haspopup="dialog"', template)
        self.assertIn('aria-describedby="note-search-status"', template)
        self.assertIn('role="status"', template)
        self.assertNotIn('class="search-input"', template)
        script = (ROOT / "static/search.js").read_text()
        self.assertNotIn("innerHTML", script)
        self.assertIn('compositionstart', script)
        self.assertIn('current !== request', script)
        self.assertIn('!response.ok', script)
        self.assertIn('dialog.close()', script)


if __name__ == "__main__":
    unittest.main()
