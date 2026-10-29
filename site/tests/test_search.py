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
const {normalize, fold, safePath, prepareIndex, rankNotes, excerpt} = require(process.argv[1]);
const notes = prepareIndex([
  {u: 'core/attention.html', t: '注意力', s: '大模型基础', l: 'zh', a: 'Attention mechanisms',
   h: [['', '', '注意力 attention 使用 mask 和 softmax', ''], ['kv', 'KV cache', '生成时把 key 和 value 存下来', 'cache = {}']]},
  {u: 'core/attention.en.html', t: 'Attention mechanisms', s: 'Foundations', l: 'en', a: '注意力', h: [['', '', 'Attention uses masks and softmax', '']]},
  {u: 'other.html', t: 'Other models', s: '基础', l: 'zh', h: [['', '', 'Also mentions attention', '']]},
  {u: 'clip.html', t: 'CLIP 图文学习', s: '多模态', l: 'zh', h: [['', '', 'A contrastive objective', '']]},
  {u: 'explore.html', t: '探索', s: 'RL', l: 'zh', h: [['', '', 'exploration and a property of the policy', '']]},
  {u: 'lora.html', t: 'LoRA 与 QLoRA', s: '后训练', l: 'zh', h: [['', '', '低秩适配', ''], ['memory', '显存', 'LoRA 省的是优化器状态', '']]},
  {u: 'adapt.html', t: '模型适配：Full Fine-Tuning、LoRA', s: '后训练', l: 'zh', h: [['', '', 'LoRA LoRA LoRA', '']]},
  {u: 'decoding.html', t: '解码策略', s: '基础', l: 'zh', h: [['', '', 'temperature 和 top-p', ''], ['beam', 'Beam search', '保留 k 条候选', '']]},
  {u: 'javascript:alert(1)', t: 'Attention', s: '', l: 'zh', h: []},
  {u: 'bad.html', t: {}, s: '', l: 'en', h: []},
  {u: 'old.html', t: 'Old format', s: '', l: 'zh', x: 'no sections'},
  null
]);
assert.equal(notes.length, 8);
assert.equal(normalize(' ＣＬＩＰ '), 'clip');
assert.equal(fold(' KV-Cache_x/y '), 'kv cache x y');
assert.equal(fold('kv缓存'), 'kv 缓存');
assert.equal(rankNotes(notes, 'attention', 'zh', false)[0].u, 'core/attention.html');
// the English title finds the Chinese note
assert.equal(rankNotes(notes, 'mechanisms', 'zh', false)[0].u, 'core/attention.html');
assert.equal(rankNotes(notes, 'attention', 'en', false).length, 1);
assert.equal(rankNotes(notes, 'attention', 'zh', true).length, 3);
assert.equal(rankNotes(notes, 'attention mask', 'zh', false).length, 1);
assert.equal(rankNotes(notes, 'ＡＴＴＥＮＴＩＯＮ', 'zh', false).length, 2);
assert.equal(rankNotes(notes, '图文', 'zh', false)[0].u, 'clip.html');
// a Latin term matches where a word starts: not "lora" in "exploration", "rope" in "property"
assert.equal(rankNotes(notes, 'lora', 'zh', false).some(hit => hit.u === 'explore.html'), false);
assert.equal(rankNotes(notes, 'rope', 'zh', false).length, 0);
// the note named for the query comes first and opens at the top
const lora = rankNotes(notes, 'lora', 'zh', false);
assert.deepEqual(lora.map(hit => hit.u), ['lora.html', 'adapt.html']);
assert.equal(lora[0].anchor, '');
// otherwise a hit opens at the section that matched, whichever way the term is written
const beam = rankNotes(notes, 'beam search', 'zh', false)[0];
assert.deepEqual([beam.u, beam.anchor, beam.heading], ['decoding.html', 'beam', 'Beam search']);
assert.equal(rankNotes(notes, 'kv-cache', 'zh', false)[0].anchor, 'kv');
assert.ok(rankNotes(notes, 'kv-cache', 'zh', false)[0].preview.includes('value'));
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
