const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const feedback = require(path.join(__dirname, '../static/page-feedback.js'));

const DATA = {repo: 'tianyi-zhang-02/cooking-agi', area: 'Area: evaluation', source: '07-evaluation/README.md', title: 'Evaluation'};

test('builds a prefilled note-feedback issue', () => {
  const {url, trimmed} = feedback.issueURL(DATA, 'Outdated', 'The leaderboard moved.', 'Old  passage\nhere', 'https://example.test/a.html');
  const parsed = new URL(url);
  assert.equal(parsed.origin + parsed.pathname, 'https://github.com/tianyi-zhang-02/cooking-agi/issues/new');
  assert.equal(parsed.searchParams.get('template'), 'note-feedback.yml');
  assert.equal(parsed.searchParams.get('title'), '[Note] Outdated · Evaluation');
  assert.equal(parsed.searchParams.get('source'), '07-evaluation/README.md');
  assert.equal(parsed.searchParams.get('question'),
    '**Outdated**\n\n> Old passage here\n\nThe leaderboard moved.\n\n—\nhttps://example.test/a.html\n');
  assert.equal(parsed.searchParams.has('labels'), false);
  assert.equal(trimmed, false);
});

test('long notes are trimmed under the URL limit', () => {
  const {url, trimmed} = feedback.issueURL(DATA, '信息过时', '很长的说明'.repeat(400), '', 'https://example.test/');
  assert.ok(url.length <= feedback.LIMIT, String(url.length));
  assert.equal(trimmed, true);
  assert.match(new URL(url).searchParams.get('question'), /…/);
});

test('quotes are collapsed and clipped', () => {
  const quote = feedback.clip('a\n\n b '.repeat(200), 280);
  assert.ok(quote.length <= 280);
  assert.ok(quote.endsWith('…'));
  assert.equal(feedback.clip('  x   y ', 280), 'x y');
});
