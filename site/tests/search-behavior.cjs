const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/search.js'), 'utf8');
const {entryDecision} = require('../static/language.js');

function mount(language = 'zh', prefix = '../') {
  let document;
  function element(tag = 'div', dataset = {}) {
    return { tag, dataset, children: [], listeners: {}, attributes: {}, hidden: false,
      textContent: '', value: '', isConnected: true, open: false,
      append(...nodes) { this.children.push(...nodes); },
      replaceChildren(...nodes) { this.children = nodes; },
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; },
      focus() { document.activeElement = this; }, select() {}, closest() { return null; },
      querySelectorAll(selector) {
        return this.children.flatMap(child => [child, ...child.querySelectorAll(selector)]).filter(child => child.tag === selector);
      },
      click() { this.clicked = true; return this.listeners.click?.({target: this}); },
      showModal() { this.open = true; },
      close() { this.open = false; this.listeners.close?.(); }
    };
  }
  const dialog = element('dialog'), input = element('input'), results = element('ol');
  const empty = element(), status = element(), close = element('button'), body = element();
  const opener = element('button');
  const scopes = ['current', 'all'].map(searchScope => element('button', {searchScope}));
  dialog.querySelector = selector => ({input, ol: results, '.note-search-empty': empty,
    '[role=status]': status, '.note-search-close': close, '.note-search-body': body}[selector]);
  dialog.querySelectorAll = () => scopes;
  document = element('document');
  document.querySelector = selector => selector === '.note-search' ? dialog : null;
  document.querySelectorAll = () => [opener];
  document.createElement = element;
  document.createTextNode = text => Object.assign(element('text'), {textContent: text});
  document.documentElement = {classList: {add() {}, remove() {}}};
  const requests = [], timers = new Map();
  let timerId = 0;
  vm.runInNewContext(source, {document, window: {SITE: {lang: language, prefix, search: {zh: 'z1', en: 'e1'}}},
    fetch: url => new Promise((resolve, reject) => requests.push({url, resolve, reject})),
    setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: identity => timers.delete(identity)});
  function type(text) {
    input.value = text;
    input.listeners.input();
    for (const [identity, callback] of [...timers]) { timers.delete(identity); callback(); }
  }
  return {dialog, input, results, empty, status, close, opener, scopes, requests, type, document};
}

const index = [
  {u: 'attention.html', t: '注意力', s: '基础', l: 'zh', h: [['', '', 'attention', '']]},
  {u: 'attention.en.html', t: 'Attention', s: 'Core', l: 'en', h: [['', '', 'attention', '']]},
  {u: 'clip.html', t: '<CLIP>', s: '多模态', l: 'zh', h: [['', '', 'images', ''], ['zero-shot', '零样本', 'zero shot images', '']]}
];
const tick = () => new Promise(resolve => setImmediate(resolve));
// each language is its own file; answer a request with that language's notes
function answer(request, items = index) {
  const language = request.url.match(/search-index\.(zh|en)\.json\?v=/)[1];
  request.resolve({ok: true, json: async () => items.filter(item => item.l === language)});
}

(async () => {
  const page = mount();
  page.opener.click();
  assert.equal(page.dialog.open, true);
  assert.equal(page.document.activeElement, page.input);
  page.type('attention');
  page.type('CLIP');
  assert.equal(page.requests.length, 1);
  assert.equal(page.requests[0].url, '../search-index.zh.json?v=z1');
  answer(page.requests[0]);
  await tick();
  assert.equal(page.status.textContent, '1 篇笔记');
  assert.equal(page.results.querySelectorAll('a')[0].href, '../clip.html');
  page.type('attention');
  await tick();
  page.scopes[1].click();
  assert.equal(page.requests.length, 2);
  assert.equal(page.requests[1].url, '../search-index.en.json?v=e1');
  answer(page.requests[1]);
  await tick();
  assert.equal(page.status.textContent, '2 篇笔记');
  page.dialog.listeners.keydown({key: 'ArrowDown', preventDefault() {}});
  assert.equal(page.document.activeElement, page.results.querySelectorAll('a')[0]);
  page.dialog.listeners.keydown({key: 'ArrowUp', preventDefault() {}});
  assert.equal(page.document.activeElement, page.input);
  page.input.listeners.compositionstart();
  page.type('not-a-match');
  assert.equal(page.results.children.length, 0);
  page.input.listeners.compositionend();
  await tick();
  assert.equal(page.status.textContent, '0 篇笔记');
  assert.equal(page.results.hidden, true);
  page.close.click();
  assert.equal(page.dialog.open, false);
  assert.equal(page.document.activeElement, page.opener);

  const failed = mount();
  failed.opener.click();
  failed.type('attention');
  failed.requests[0].resolve({ok: false});
  await tick();
  assert.equal(failed.status.textContent, '搜索暂时没加载出来');
  failed.empty.children.at(-1).click();
  assert.equal(failed.requests.length, 2);
  answer(failed.requests[1]);
  await tick();
  assert.equal(failed.status.textContent, '1 篇笔记');

  const closed = mount();
  closed.opener.click();
  closed.type('attention');
  closed.close.click();
  answer(closed.requests[0]);
  await tick();
  assert.equal(closed.results.children.length, 0);

  // a section hit links to its heading and steps the dialog aside, even on the page already open
  const section = mount();
  section.opener.click();
  section.type('zero shot');
  answer(section.requests[0]);
  await tick();
  const sectionLink = section.results.querySelectorAll('a')[0];
  assert.equal(sectionLink.href, '../clip.html#zero-shot');
  assert.ok(sectionLink.children.some(child => child.className === 'note-search-section'));
  sectionLink.click();
  assert.equal(section.dialog.open, false);

  // reaching for the button starts the download; opening the dialog reuses it
  const warm = mount();
  warm.opener.listeners.pointerenter();
  assert.equal(warm.requests.length, 1);
  warm.opener.click();
  warm.type('attention');
  assert.equal(warm.requests.length, 1);
  answer(warm.requests[0]);
  await tick();
  assert.equal(warm.status.textContent, '1 篇笔记');

  const homes = [
    {u: 'index.html', t: 'Cooking AGI', s: 'Home', l: 'en', h: [['', '', 'Study notes', '']]},
    {u: 'index.zh.html', t: 'AGI 大锅烩', s: '首页', l: 'zh', h: [['', '', '学习笔记', '']]}
  ];
  for (const language of ['zh', 'en']) {
    for (const prefix of ['', '../', '../../']) {
      const homeSearch = mount(language, prefix);
      homeSearch.opener.click();
      homeSearch.type('AGI');
      answer(homeSearch.requests[0], homes);
      await tick();
      const ownLink = homeSearch.results.querySelectorAll('a')[0].href;
      assert.equal(ownLink, prefix + (language === 'en' ? 'index.html?lang=en' : 'index.zh.html'));
      homeSearch.scopes[1].click();
      answer(homeSearch.requests[1], homes);
      await tick();
      const links = homeSearch.results.querySelectorAll('a').map(link => link.href);
      assert.equal(links.length, 2);
      const base = 'https://example.org/cooking-agi/';
      const current = new URL('chapter/'.repeat(prefix.split('../').length - 1) + 'note.html', base);
      for (const href of links) {
        const target = new URL(href, current).href;
        assert.equal(entryDecision(target, base, 'zh').redirect, null, href);
      }
      assert.ok(links.includes(prefix + 'index.html?lang=en'));
      assert.ok(links.includes(prefix + 'index.zh.html'));
    }
  }
  console.log('Search interactions passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
