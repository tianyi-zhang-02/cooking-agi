const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/language.js'), 'utf8');
const {entryDecision, storageKeys, linkLanguage} = require('../static/language.js');
const base = 'https://example.org/cooking-agi/';
const keys = storageKeys(base);
const {prepareIndex, rankNotes, safePath} = require('../static/search.js');
const homes = prepareIndex([
  {u: 'index.html', t: 'Cooking AGI', s: 'Home', l: 'en', a: 'AGI 大锅烩', h: [['', '', 'Start here', '']]},
  {u: 'index.zh.html', t: 'AGI 大锅烩', s: '首页', l: 'zh', a: 'Cooking AGI', h: [['', '', '从这里开始', '']]}
]);
assert.equal(homes.length, 2);
assert.equal(rankNotes(homes, 'Cooking AGI', 'zh', false)[0].u, 'index.zh.html');
assert.equal(rankNotes(homes, '大锅烩', 'en', false)[0].u, 'index.html');
for (const unsafe of ['../index.zh.html', '/index.zh.html', '//index.zh.html', 'index.zh.html?url=elsewhere']) {
  assert.equal(safePath(unsafe), false);
}

for (const suffix of ['', 'index.html']) {
  assert.equal(entryDecision(base + suffix, base, null).redirect, null);
  assert.equal(entryDecision(base + suffix, base, 'en').redirect, null);
  assert.equal(entryDecision(base + suffix, base, 'zh').redirect, base + 'index.zh.html');
  assert.equal(entryDecision(base + suffix, base, 'not-a-language').redirect, null);
}
assert.equal(entryDecision(base + '?lang=en#start', base, 'zh').redirect, null);
assert.equal(entryDecision(base + '?lang=en', base, 'zh').choice, 'en');
assert.equal(entryDecision(base + '?lang=zh&from=share#start', base, 'en').redirect, base + 'index.zh.html?from=share#start');
for (const suffix of ['index.zh.html', 'learn/index.html', 'learn/index.en.html', 'index.en.html']) {
  assert.equal(entryDecision(base + suffix, base, 'zh').redirect, null);
}
assert.equal(entryDecision('https://elsewhere.org/cooking-agi/', base, 'zh').redirect, null);
assert.equal(entryDecision('https://example.org/another/', base, 'zh').redirect, null);
assert.notEqual(keys.choice, storageKeys('https://example.org/another/').choice);
assert.equal(linkLanguage(base + 'learn/index.html', 'zh', base), 'zh');
assert.equal(linkLanguage(base + 'index.zh.html', undefined, base), 'zh');
assert.equal(linkLanguage(base + 'index.en.html', undefined, base), 'en');
assert.equal(linkLanguage('https://elsewhere.org/', 'zh', base), null);
assert.equal(linkLanguage('https://example.org/elsewhere/', 'en', base), null);
assert.equal(linkLanguage('javascript:alert(1)', 'zh', base), null);
assert.equal(linkLanguage(base + 'learn/index.html', 'bad', base), null);
assert.equal(entryDecision(base + 'index.html?lang=zh#content', base, null).redirect,
  base + 'index.zh.html#content');
assert.equal(entryDecision(base + 'index.html?lang=en#content', base, 'zh').choice, 'en');
assert.equal(entryDecision(base + 'learn/index.html#content', base, 'en').redirect, null);
assert.equal(entryDecision(base + 'learn/index.en.html#content', base, 'zh').redirect, null);

function simulate(options = {}) {
  const local = options.local || new Map();
  const session = options.session || new Map();
  const handlers = {};
  let redirected = null;
  const document = {
    currentScript: {dataset: {siteRoot: options.prefix || ''}},
    readyState: 'complete',
    getElementById: () => null,
    querySelector: () => null,
    addEventListener: (type, handler) => { handlers[type] = handler; }
  };
  const storage = (values, blocked) => ({
    getItem: key => { if (blocked) throw Error('blocked'); return values.get(key) ?? null; },
    setItem: (key, value) => { if (blocked) throw Error('blocked'); values.set(key, value); }
  });
  const window = {
    location: {href: options.href || base, replace: value => { redirected = value; }},
    localStorage: storage(local, options.blockLocal),
    sessionStorage: storage(session, options.blockSession)
  };
  vm.runInNewContext(source, {URL, document, window});
  return {local, session, handlers, redirected};
}

// No first-visit tip any more: the home cover names the language in place, and nothing pops up.
const first = simulate();
assert.equal(first.local.size, 0);
assert.equal(first.redirected, null);
assert.equal(source.includes('language-guide'), false);

const linked = simulate();
const chineseLink = {
  href: base + 'learn/index.html', dataset: {languageSwitch: 'zh'},
  hasAttribute: name => name === 'data-language-switch'
};
linked.handlers.click({target: {closest: () => chineseLink}, button: 0});
assert.equal(linked.local.get(keys.choice), 'zh');
assert.equal(simulate({local: linked.local}).redirected, base + 'index.zh.html');
assert.equal(simulate({href: base + '?lang=en', local: linked.local}).redirected, null);
assert.equal(linked.local.get(keys.choice), 'en');

// the line on the home cover is a plain link into the other home page
const coverLink = {href: base + 'index.zh.html', dataset: {languageSwitch: 'zh'}, hasAttribute: () => true};
const cover = simulate();
cover.handlers.click({target: {closest: () => coverLink}, button: 0});
assert.equal(cover.local.get(keys.choice), 'zh');

const modifiedClick = simulate();
modifiedClick.handlers.click({target: {closest: () => chineseLink}, button: 0, ctrlKey: true});
assert.equal(modifiedClick.local.get(keys.choice), undefined);
const englishHomeLink = {
  href: base + 'index.html?lang=en', dataset: {languageSwitch: 'en'},
  hasAttribute: name => name === 'data-language-switch'
};
for (const gesture of [{button: 0, ctrlKey: true}, {button: 0, metaKey: true},
                       {button: 0, shiftKey: true}, {button: 1}, null]) {
  const sourceTab = simulate({href: base + 'index.zh.html', local: new Map([[keys.choice, 'zh']])});
  if (gesture) sourceTab.handlers.click({target: {closest: () => englishHomeLink}, ...gesture});
  assert.equal(sourceTab.local.get(keys.choice), 'zh');
  const newTab = simulate({href: englishHomeLink.href, local: sourceTab.local});
  assert.equal(newTab.redirected, null);
  assert.equal(newTab.local.get(keys.choice), 'en');
}
const questionBank = simulate({href: base + 'interview/questions.html', prefix: '../',
  local: new Map([[keys.choice, 'zh']])});
questionBank.handlers.click({target: {closest: () => ({href: base + 'learn/index.html',
  dataset: {}, hasAttribute: () => false})}, button: 0});
assert.equal(questionBank.local.get(keys.choice), 'zh');
const fallback = simulate({blockLocal: true});
fallback.handlers.click({target: {closest: () => chineseLink}, button: 0});
assert.equal(fallback.session.get(keys.choice), 'zh');
assert.equal(simulate({blockLocal: true, session: fallback.session}).redirected, base + 'index.zh.html');
assert.doesNotThrow(() => simulate({blockLocal: true, blockSession: true}));
const unavailableStorage = simulate({blockLocal: true, blockSession: true});
unavailableStorage.handlers.click({target: {closest: () => chineseLink}, button: 0});
assert.equal(unavailableStorage.redirected, null);
const deepLink = simulate({href: base + 'interview/ai-roles-prep.en.html', prefix: '../', local: new Map([[keys.choice, 'zh']])});
assert.equal(deepLink.redirected, null);
assert.equal(deepLink.local.get(keys.choice), 'zh');
console.log('Language entry and preferences passed.');
