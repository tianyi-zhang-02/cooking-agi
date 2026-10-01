const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/search.js'), 'utf8');

function mount() {
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
  vm.runInNewContext(source, {document, window: {SITE: {lang: 'zh', prefix: '../'}},
    fetch: () => new Promise((resolve, reject) => requests.push({resolve, reject})),
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
  {u: 'attention.html', t: '注意力', s: '基础', l: 'zh', x: 'attention'},
  {u: 'attention.en.html', t: 'Attention', s: 'Core', l: 'en', x: 'attention'},
  {u: 'clip.html', t: '<CLIP>', s: '多模态', l: 'zh', x: 'images'}
];
const tick = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  const page = mount();
  page.opener.click();
  assert.equal(page.dialog.open, true);
  assert.equal(page.document.activeElement, page.input);
  page.type('attention');
  page.type('CLIP');
  assert.equal(page.requests.length, 1);
  page.requests[0].resolve({ok: true, json: async () => index});
  await tick();
  assert.equal(page.status.textContent, '1 篇笔记');
  assert.equal(page.results.querySelectorAll('a')[0].href, '../clip.html');
  page.type('attention');
  await tick();
  page.scopes[1].click();
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
  failed.requests[1].resolve({ok: true, json: async () => index});
  await tick();
  assert.equal(failed.status.textContent, '1 篇笔记');

  const closed = mount();
  closed.opener.click();
  closed.type('attention');
  closed.close.click();
  closed.requests[0].resolve({ok: true, json: async () => index});
  await tick();
  assert.equal(closed.results.children.length, 0);
  console.log('Search interactions passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
