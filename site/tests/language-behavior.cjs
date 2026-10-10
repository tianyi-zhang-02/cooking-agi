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
  {u: 'index.html', t: 'AGI Study Notes', s: 'Home', l: 'en', x: 'Start here'},
  {u: 'index.zh.html', t: 'AGI 学习笔记', s: '首页', l: 'zh', x: '从这里开始'}
]);
assert.equal(homes.length, 2);
assert.equal(rankNotes(homes, 'study notes', 'zh', false)[0].u, 'index.zh.html');
assert.equal(rankNotes(homes, '学习笔记', 'en', false)[0].u, 'index.html');
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
  const buttonHandlers = {};
  const attributes = {};
  const classes = new Set();
  let redirected = null;
  let focused = false;
  const button = {addEventListener: (type, handler) => { buttonHandlers[type] = handler; }};
  const guide = {
    hidden: true,
    contains: element => element === button,
    querySelector: selector => selector === '[data-language-dismiss]' ? button : null
  };
  const switcher = {
    setAttribute: (key, value) => { attributes[key] = value; },
    removeAttribute: key => { delete attributes[key]; },
    classList: {add: value => classes.add(value), remove: value => classes.delete(value)},
    focus: () => { focused = true; }
  };
  const document = {
    currentScript: {dataset: {siteRoot: options.prefix || ''}},
    readyState: 'complete',
    activeElement: null,
    getElementById: name => name === 'language-guide' ? guide : null,
    querySelector: () => options.noSwitcher ? null : switcher,
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
  return {local, session, guide, handlers, buttonHandlers, attributes, document, button,
    redirected, focused: () => focused};
}

const first = simulate();
assert.equal(first.guide.hidden, false);
assert.equal(first.focused(), false);
assert.equal(first.attributes['aria-describedby'], 'language-guide-description');
assert.equal(first.local.get(keys.guide), 'seen');
first.document.activeElement = first.button;
first.buttonHandlers.click();
assert.equal(first.guide.hidden, true);
assert.equal(first.focused(), true);
assert.equal(first.attributes['aria-describedby'], undefined);
assert.equal(simulate({local: first.local}).guide.hidden, true);

const escape = simulate();
escape.handlers.keydown({key: 'Escape'});
assert.equal(escape.guide.hidden, true);
assert.equal(escape.focused(), false);

const linked = simulate();
const chineseLink = {
  href: base + 'learn/index.html', dataset: {languageSwitch: 'zh'},
  hasAttribute: name => name === 'data-language-switch'
};
linked.handlers.click({target: {closest: () => chineseLink}, button: 0});
assert.equal(linked.local.get(keys.choice), 'zh');
assert.equal(linked.guide.hidden, true);
assert.equal(simulate({local: linked.local}).redirected, base + 'index.zh.html');
assert.equal(simulate({href: base + '?lang=en', local: linked.local}).redirected, null);
assert.equal(linked.local.get(keys.choice), 'en');

const modifiedClick = simulate();
modifiedClick.handlers.click({target: {closest: () => chineseLink}, button: 0, ctrlKey: true});
assert.equal(modifiedClick.local.get(keys.choice), undefined);
assert.equal(simulate({noSwitcher: true}).guide.hidden, true);
const fallback = simulate({blockLocal: true});
assert.equal(fallback.guide.hidden, false);
assert.equal(simulate({blockLocal: true, session: fallback.session}).guide.hidden, true);
assert.doesNotThrow(() => simulate({blockLocal: true, blockSession: true}));
const unavailableStorage = simulate({blockLocal: true, blockSession: true});
unavailableStorage.handlers.click({target: {closest: () => chineseLink}, button: 0});
assert.equal(unavailableStorage.guide.hidden, true);
assert.equal(unavailableStorage.redirected, null);
const deepLink = simulate({href: base + 'career/journey.en.html', prefix: '../', local: new Map([[keys.choice, 'zh']])});
assert.equal(deepLink.redirected, null);
assert.equal(deepLink.local.get(keys.choice), 'zh');
console.log('Language entry, preferences, and first-visit guide passed.');
