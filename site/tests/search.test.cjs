const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/search.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
const notes = [
  {u: 'core/lora.en.html', t: 'LoRA and QLoRA', s: 'Post-training', l: 'en',
    h: [['', '', 'LoRA adapts weights; exploration is not a matching word. <script>alert(1)</script>', '']]},
  {u: 'core/lora.html', t: 'LoRA 与 QLoRA', s: '后训练', l: 'zh',
    h: [['', '', 'LoRA 使用低秩适配。', '']]},
  {u: 'core/clip.en.html', t: '<CLIP> image models', s: 'Multimodal <img src=x>', l: 'en',
    h: [['', '', 'Image and text encoders.', ''], ['zero-shot', 'Zero-shot evaluation', 'Evaluate unseen classes with prompts.', '']]},
  {u: 'core/regex.en.html', t: 'a+b [literal] expressions', s: 'Engineering', l: 'en',
    h: [['', '', 'a+b [literal] is plain text, not executable markup.', '']]}
];

function mount(language = 'en') {
  let document;
  function element(tag = 'div', dataset = {}) {
    return {
      tag, dataset, children: [], listeners: {}, attributes: {}, hidden: false,
      textContent: '', value: '', isConnected: true, open: false,
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; this.textContent = ''; },
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; },
      focus() { document.activeElement = this; },
      select() {},
      closest(selector) {
        if (selector === '[hidden]') return this.hidden ? this : null;
        return ['input', 'textarea', 'select'].includes(this.tag) ? this : null;
      },
      querySelectorAll(selector) {
        return this.children.flatMap(child => [child, ...child.querySelectorAll('*')])
          .filter(child => selector === '*' || child.tag === selector || child.className === selector.slice(1));
      },
      click() { return this.listeners.click?.({target: this}); },
      showModal() { this.open = true; },
      close() { this.open = false; this.listeners.close?.(); },
      getBoundingClientRect() { return {left: 10, right: 500, top: 10, bottom: 500}; }
    };
  }
  const dialog = element('dialog'), input = element('input'), results = element('ol');
  const empty = element(), status = element(), close = element('button'), body = element();
  const opener = element('button');
  const scopes = ['current', 'all'].map(searchScope => element('button', {searchScope}));
  dialog.querySelector = selector => ({input, ol: results, '.note-search-empty': empty,
    '[role=status]': status, '.note-search-close': close, '.note-search-body': body}[selector]);
  dialog.querySelectorAll = selector => selector === '[data-search-scope]' ? scopes :
    [close, input, ...scopes, ...results.querySelectorAll('a'), ...empty.querySelectorAll('button')];
  document = element('document');
  document.querySelector = selector => selector === '.note-search' ? dialog : null;
  document.querySelectorAll = () => [opener];
  document.createElement = element;
  document.createTextNode = text => Object.assign(element('text'), {textContent: text});
  document.documentElement = {classList: {add() {}, remove() {}}};
  const requests = [], timers = new Map();
  let timerId = 0;
  vm.runInNewContext(source, {document, window: {SITE: {lang: language, prefix: '../', search: {zh: 'z1', en: 'e1'}}},
    fetch: url => new Promise((resolve, reject) => requests.push({url, resolve, reject})),
    setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: identity => timers.delete(identity)});
  function flush() {
    for (const [identity, callback] of [...timers]) { timers.delete(identity); callback(); }
  }
  function type(value, immediate = true) {
    input.value = value;
    input.listeners.input();
    if (immediate) flush();
  }
  function answer(request, items = notes) {
    const targetLanguage = request.url.match(/search-index\.(zh|en)\.json/)[1];
    request.resolve({ok: true, json: async () => items.filter(item => item.l === targetLanguage)});
  }
  return {dialog, input, results, empty, status, close, opener, scopes, requests, document, type, answer, flush};
}

function textOf(node) {
  return node.textContent + node.children.map(textOf).join('');
}

function childWithClass(node, name) {
  return node.querySelectorAll('.' + name)[0];
}

async function search(query, language = 'en', items = notes) {
  const page = mount(language);
  page.opener.click();
  page.type(query);
  page.answer(page.requests[0], items);
  await tick();
  return page;
}

test('results show real section metadata, stable paths and escaped snippets', async () => {
  const page = await search('lora');
  const result = page.results.querySelectorAll('a')[0];
  assert.equal(page.status.textContent, '1 note');
  assert.equal(result.href, '../core/lora.en.html');
  assert.equal(result.lang, 'en');
  assert.equal(textOf(childWithClass(result, 'note-search-category')), 'Post-training');
  assert.equal(textOf(childWithClass(result, 'note-search-title')), 'LoRA and QLoRA');
  assert.equal(textOf(childWithClass(result, 'note-search-path')), 'core/lora.en.html');
  assert.match(textOf(childWithClass(result, 'note-search-excerpt')), /<script>alert\(1\)<\/script>/);
  assert.equal(result.querySelectorAll('script').length, 0);
  assert.deepEqual(result.querySelectorAll('mark').map(textOf), ['LoRA', 'LoRA']);
});

test('section matches keep their anchor and language when opened with Enter', async () => {
  const page = await search('zero-shot');
  const result = page.results.querySelectorAll('a')[0];
  assert.equal(result.href, '../core/clip.en.html#zero-shot');
  assert.equal(textOf(childWithClass(result, 'note-search-section')), 'Zero-shot evaluation');
  assert.equal(textOf(childWithClass(result, 'note-search-path')), 'core/clip.en.html#zero-shot');
  assert.equal(textOf(childWithClass(result, 'note-search-category')), 'Multimodal <img src=x>');
  assert.equal(result.querySelectorAll('img').length, 0);
  page.dialog.listeners.keydown({key: 'Enter', preventDefault() {}});
  assert.equal(page.dialog.open, false);
  assert.equal(page.document.activeElement, page.opener);
});

test('query highlighting treats regex syntax and HTML as literal text', async () => {
  for (const query of ['a+b', '[literal]', '<CLIP>']) {
    const page = await search(query);
    const result = page.results.querySelectorAll('a')[0];
    assert.ok(result, query);
    assert.ok(result.querySelectorAll('mark').some(mark => textOf(mark) === query), query);
    assert.equal(result.querySelectorAll('img').length, 0);
  }
});

test('language scopes load on demand, preserve query and update pressed state', async () => {
  const page = await search('LoRA');
  page.scopes[1].focus();
  page.scopes[1].click();
  assert.equal(page.requests.length, 2);
  assert.equal(page.requests[1].url, '../search-index.zh.json?v=z1');
  page.answer(page.requests[1]);
  await tick();
  assert.equal(page.status.textContent, '2 notes');
  assert.equal(page.input.value, 'LoRA');
  assert.equal(page.scopes[1].attributes['aria-pressed'], 'true');
  assert.equal(page.document.activeElement, page.scopes[1]);
  assert.deepEqual(page.results.querySelectorAll('a').map(result => result.lang), ['en', 'zh']);
  assert.deepEqual(page.results.querySelectorAll('.note-search-language').map(textOf), ['EN', 'ZH']);
  page.scopes[0].click();
  await tick();
  assert.equal(page.status.textContent, '1 note');
  assert.equal(page.scopes[1].attributes['aria-pressed'], 'false');
  assert.equal(page.requests.length, 2);
});

test('Chinese results keep localized copy and the existing note URL', async () => {
  const page = await search('LoRA', 'zh');
  const result = page.results.querySelectorAll('a')[0];
  assert.equal(page.status.textContent, '1 篇笔记');
  assert.equal(result.href, '../core/lora.html');
  assert.equal(textOf(childWithClass(result, 'note-search-category')), '后训练');
  assert.equal(textOf(childWithClass(result, 'note-search-language')), '中文');
});

test('editing a query clears stale counts and stale keyboard targets immediately', async () => {
  const page = await search('LoRA');
  page.type('unmatched', false);
  assert.equal(page.status.textContent, '');
  assert.equal(page.results.hidden, true);
  assert.equal(page.results.querySelectorAll('a').length, 0);
  page.flush();
  await tick();
  assert.equal(page.status.textContent, '0 notes');
  assert.match(textOf(page.empty), /No matching notes/);
  page.type('');
  await tick();
  assert.match(textOf(page.empty), /Start with a question/);
});

test('IME composition blocks navigation and shortcuts, and closing resets composition', async () => {
  const page = await search('LoRA');
  page.input.listeners.compositionstart();
  const block = () => assert.fail('A composing key should not be intercepted');
  page.dialog.listeners.keydown({key: 'Enter', preventDefault: block});
  page.document.listeners.keydown({key: 'k', ctrlKey: true, target: page.input, preventDefault: block});
  assert.equal(page.dialog.open, true);
  page.close.click();
  page.opener.click();
  page.type('zero-shot');
  await tick();
  assert.equal(page.results.querySelectorAll('a')[0].href, '../core/clip.en.html#zero-shot');
});

test('keyboard result navigation wraps and restores the query field', async () => {
  const page = await search('LoRA');
  const result = page.results.querySelectorAll('a')[0];
  page.dialog.listeners.keydown({key: 'ArrowUp', preventDefault() {}});
  assert.equal(page.document.activeElement, result);
  page.dialog.listeners.keydown({key: 'ArrowDown', preventDefault() {}});
  assert.equal(page.document.activeElement, result);
  page.dialog.listeners.keydown({key: 'ArrowUp', preventDefault() {}});
  assert.equal(page.document.activeElement, page.input);
});

test('Tab and Shift+Tab stay inside the modal at either boundary', async () => {
  const page = await search('LoRA');
  const result = page.results.querySelectorAll('a')[0];
  let prevented = 0;
  result.focus();
  page.dialog.listeners.keydown({key: 'Tab', preventDefault() { prevented++; }});
  assert.equal(page.document.activeElement, page.close);
  page.dialog.listeners.keydown({key: 'Tab', shiftKey: true, preventDefault() { prevented++; }});
  assert.equal(page.document.activeElement, result);
  assert.equal(prevented, 2);
});

test('Escape closes the search field without first clearing the query', async () => {
  const page = await search('LoRA');
  let prevented = false;
  page.dialog.listeners.keydown({key: 'Escape', preventDefault() { prevented = true; }});
  assert.equal(prevented, true);
  assert.equal(page.dialog.open, false);
  assert.equal(page.input.value, 'LoRA');
  assert.equal(page.document.activeElement, page.opener);
});

test('result counts reflect the full index rather than only the displayed twelve', async () => {
  const many = Array.from({length: 15}, (_, position) => ({...notes[0], u: `core/lora-${position}.en.html`}));
  const page = await search('LoRA', 'en', many);
  assert.equal(page.status.textContent, '15 notes · Showing the first 12');
  assert.equal(page.results.querySelectorAll('a').length, 12);
});
