const test = require('node:test');
const assert = require('node:assert/strict');
const {stepState, locationState, viewURL} = require('../static/learning-views.js');

test('prefill predicts the next token without caching that prediction', () => {
  const state = stepState(0);
  assert.deepEqual(state.tokens, ['A', 'B', 'C']);
  assert.equal(state.prediction, 'D');
  assert.equal(state.input, 'A B C');
  assert.deepEqual(state.projected, {plain: 3, cached: 3});
  assert.deepEqual(state.scores, {plain: 6, cached: 6});
  assert.equal(state.reused, 0);
});

test('each decode step appends the previous prediction and computes one row', () => {
  for (const step of [1, 2, 3]) {
    const previous = stepState(step - 1);
    const current = stepState(step);
    assert.equal(current.input, previous.prediction);
    assert.deepEqual(current.tokens, [...previous.tokens, previous.prediction]);
    assert.equal(current.tokens.includes(current.prediction), false);
    assert.equal(current.projected.cached, 1);
    assert.equal(current.projected.plain, 3 + step);
    assert.equal(current.reused, 2 + step);
    assert.equal(current.scores.cached, 3 + step);
    assert.equal(current.scores.plain, (3 + step) * (4 + step) / 2);
  }
});

test('invalid model states cannot silently become extra tokens', () => {
  for (const step of [-1, 4, 0.5, NaN, Infinity, '1', null]) assert.throws(() => stepState(step), RangeError);
  const first = stepState(1);
  first.tokens.push('X');
  assert.equal(stepState(1).tokens.includes('X'), false);
});

test('existing fragments always open the full note', () => {
  for (const view of ['quick', 'lab', 'full', 'invalid']) {
    assert.equal(locationState(`https://example.org/note.html?view=${view}#_3`).mode, 'full');
    assert.equal(locationState(`https://example.org/note.html?view=${view}#%E6%98%BE%E5%AD%98`).mode, 'full');
  }
  assert.equal(locationState('https://example.org/note.html#%').mode, 'full');
});

test('view and step defaults are bounded and do not depend on storage', () => {
  assert.deepEqual(locationState('https://example.org/note.html'), {mode: 'quick', step: 1});
  assert.deepEqual(locationState('https://example.org/note.html?view=lab&step=0'), {mode: 'lab', step: 0});
  for (const step of ['-1', '4', '0.5', '', 'infinity']) {
    assert.equal(locationState(`https://example.org/note.html?step=${step}`).step, 1);
  }
  assert.equal(locationState('https://example.org/note.html?view=invalid').mode, 'quick');
});

test('view links preserve paths and unrelated parameters, not obsolete fragments', () => {
  const href = 'https://example.org/cooking-agi/00-foundations/note.en.html?ref=bookmark#old-section';
  const lab = new URL(viewURL(href, 'lab', 2));
  assert.equal(lab.pathname, '/cooking-agi/00-foundations/note.en.html');
  assert.equal(lab.searchParams.get('ref'), 'bookmark');
  assert.equal(lab.searchParams.get('view'), 'lab');
  assert.equal(lab.searchParams.get('step'), '2');
  assert.equal(lab.hash, '');
  const full = new URL(viewURL(lab.href, 'full', 2));
  assert.equal(full.searchParams.has('step'), false);
  assert.throws(() => viewURL(href, 'invalid', 1), RangeError);
});
