const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const store = require('../static/prep-progress-store.js');
const library = require('../static/prep-library-state.js');
const hub = require('../static/reader-hub.js');
const source = fs.readFileSync(require.resolve('../static/reader-hub.js'), 'utf8');
const controller = fs.readFileSync(require.resolve('../static/prep-progress.js'), 'utf8');
const NOTE = 'note-0123456789abcdef';
const OTHER = 'note-fedcba9876543210';
const UNKNOWN = 'note-1111111111111111';
const AT = '2026-10-12T08:00:00.000Z';
const LATER = '2026-10-13T08:00:00.000Z';
const catalog = {notes: {
  [NOTE]: {source: 'chapter.md', pages: {en: {title: 'A real note', url: 'chapter.en.html'}, zh: {title: '一篇笔记', url: 'chapter.html'}}},
  [OTHER]: {source: 'other.md', pages: {en: {title: 'Another note', url: 'other.en.html'}}}
}};

class Element {
  constructor(tag, owner) {
    Object.assign(this, {tag, owner, attributes: {}, children: [], dataset: {}, events: {}, hidden: false, open: false, style: {}, text: ''});
  }
  get isConnected() { return this.tag === 'body' || Boolean(this.parent?.isConnected); }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
  set textContent(value) { this.text = String(value); this.children = []; }
  setAttribute(key, value) {
    this.attributes[key] = String(value);
    if (key.startsWith('data-')) this.dataset[key.slice(5)] = String(value);
    if (key === 'hidden') this.hidden = true;
    if (key === 'open') this.open = true;
  }
  append(...children) { children.forEach(child => { child.parent = this; this.children.push(child); }); }
  replaceChildren(...children) { this.children.forEach(child => { child.parent = null; }); this.children = []; this.append(...children); }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); this.parent = null; }
  insertBefore(child) { this.append(child); }
  addEventListener(name, handler) { this.events[name] = handler; }
  focus() { this.owner.activeElement = this; }
  contains(node) { return node === this || this.children.some(child => child.contains(node)); }
  getClientRects() { return this.hidden || this.parent && !this.parent.getClientRects().length ? [] : [{}]; }
  matches(selector) {
    if (selector.includes(',')) return selector.split(',').some(part => this.matches(part.trim()));
    if (selector.startsWith('.')) return (this.attributes.class || '').split(' ').includes(selector.slice(1));
    const attribute = /^(\w+)?\[([^=\]]+)(?:="([^"]*)")?\]$/.exec(selector);
    return attribute ? (!attribute[1] || this.tag === attribute[1]) && Object.hasOwn(this.attributes, attribute[2])
      && (attribute[3] === undefined || this.attributes[attribute[2]] === attribute[3]) : this.tag === selector;
  }
  querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  closest(selector) { return this.matches(selector) ? this : this.parent?.closest(selector); }
  click() { this.focus(); this.events.click?.({target: this}); }
}

function fixture() {
  return {...store.empty(), entries: {[NOTE]: {status: 'learning', title: 'Retired record', note: 'Private memo', updatedAt: AT}},
    mode: {value: 'plan', updatedAt: AT},
    schedule: {version: 1, sessions: {'session-0123456789abcdef0123456789abcdef': {
      title: 'Old session', date: '2026-10-12', minutes: 30, noteId: NOTE, status: 'done', completedOn: '2026-10-12', actualMinutes: 25, updatedAt: AT
    }}}, workspace: {version: 1, records: {
      ['bookmark-' + NOTE]: {saved: true, updatedAt: AT},
      ['bookmark-' + UNKNOWN]: {saved: true, updatedAt: AT},
      ['position-en-' + NOTE]: {anchor: 'stored-section', updatedAt: AT},
      'legacy-library': {reading: {saved: ['missing.md']}, positions: [], updatedAt: AT},
      'reader-settings': {history: true, mascot: 'hidden', updatedAt: AT},
      ['reads-' + hub.dayKey(new Date())]: {notes: {[NOTE]: 2}, updatedAt: AT}
    }, history: []}};
}

function environment({initial = fixture(), language = 'en', failCatalog = false} = {}) {
  const values = new Map([[store.KEY, store.encode(initial)]]);
  const storage = {getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value)};
  const events = {};
  const documentEvents = {};
  const timers = [];
  const downloads = [];
  const document = {documentElement: {lang: language, dataset: {}}, visibilityState: 'visible', title: 'A note', activeElement: null,
    addEventListener(name, handler) { documentEvents[name] = handler; }};
  document.body = new Element('body', document);
  const right = new Element('div', document);
  document.body.append(right);
  document.createElement = tag => new Element(tag, document);
  document.createElementNS = (namespace, tag) => document.createElement(tag);
  document.createTextNode = text => Object.assign(document.createElement('text'), {textContent: text});
  document.querySelector = selector => selector === '.topbar-right' ? right : null;
  class DownloadURL extends URL {
    static createObjectURL(blob) { downloads.push(blob); return 'blob:test'; }
    static revokeObjectURL() {}
  }
  const window = {PrepProgressStore: store, PrepLibraryState: library, localStorage: storage, SITE: {prefix: '/'},
    addEventListener(name, handler) { (events[name] ||= []).push(handler); },
    dispatchEvent(event) { (events[event.type] || []).forEach(handler => handler(event)); },
    setTimeout(handler) { timers.push(handler); }, setInterval() {}, matchMedia: () => ({matches: false})};
  const context = vm.createContext({window, document, URL: DownloadURL, Blob, Event, CustomEvent, setTimeout: window.setTimeout,
    location: {href: 'https://example.com/index.html'}, fetch: () => Promise.reject(Error('offline'))});
  vm.runInContext(controller, context);
  if (!failCatalog) window.PreparationProgress.getCatalog = () => catalog;
  vm.runInContext(source, context);
  const drawer = document.body.querySelector('.reader-hub');
  const toggle = document.body.querySelector('.reader-hub-toggle');
  const select = name => {
    const tab = drawer.querySelector(`[data-tab="${name}"]`);
    tab.focus();
    drawer.querySelector('.reader-hub-tabs').events.click({target: tab});
  };
  const importFile = async (text, options = {}) => {
    const input = drawer.querySelector('.reader-hub-quick-backup').querySelector('input');
    input.files = [{size: Buffer.byteLength(text), text: async () => text, ...options}];
    await input.events.change();
  };
  toggle.click();
  return {window, document, documentEvents, drawer, toggle, select, storage, values, downloads, importFile,
    status: () => drawer.querySelector('.reader-hub-status').textContent,
    records: () => window.PreparationProgress.getState().workspace.records};
}

test('resume uses only a known position in the current language and never writes or guesses', () => {
  const records = fixture().workspace.records;
  const before = JSON.stringify(records);
  assert.deepEqual(hub.lastPosition(records, catalog.notes, 'en'), {id: NOTE, anchor: 'stored-section', updatedAt: AT});
  assert.equal(hub.lastPosition(records, catalog.notes, 'zh'), null);
  assert.equal(hub.lastPosition({}, catalog.notes, 'en'), null);
  for (const record of [{anchor: '', updatedAt: AT}, {anchor: 'x" onclick="bad', updatedAt: AT}, {anchor: 'part', updatedAt: 'invalid'}]) {
    assert.equal(hub.lastPosition({['position-en-' + NOTE]: record}, catalog.notes, 'en'), null);
  }
  assert.equal(hub.lastPosition({['position-en-' + UNKNOWN]: {anchor: 'chapter', updatedAt: LATER}}, catalog.notes, 'en'), null);
  assert.equal(hub.lastPosition({['position-zh-' + OTHER]: {anchor: 'chapter', updatedAt: LATER}}, catalog.notes, 'zh'), null);
  assert.equal(hub.lastPosition({...records, ['position-en-' + OTHER]: {anchor: 'next', updatedAt: LATER}}, catalog.notes, 'en').id, OTHER);
  assert.equal(JSON.stringify(records), before);
});

test('both languages prioritize actual notes, keep activity collapsed and expose backup outside settings', async () => {
  for (const language of ['en', 'zh']) {
    const context = environment({language});
    await Promise.resolve();
    const panel = context.drawer.querySelector('[id="reader-panel-reading"]');
    const activity = panel.querySelector('.reader-hub-activity');
    assert.equal(activity.open, false);
    assert.ok(panel.children.indexOf(panel.querySelector('.reader-hub-list')) < panel.children.indexOf(activity));
    assert.match(panel.querySelector('.reader-hub-list').textContent, language === 'en' ? /A real note/ : /一篇笔记/);
    assert.equal(context.drawer.querySelector('.reader-hub-quick-backup').hidden, false);
    assert.equal(context.drawer.querySelector('.reader-hub-quick-backup').querySelectorAll('button').length, 2);
    const resume = panel.querySelector('.reader-hub-continue');
    assert.equal(resume?.attributes.href || null, language === 'en' ? 'https://example.com/chapter.en.html#stored-section' : null);
    context.select('saved');
    assert.equal(context.drawer.querySelector('.reader-hub-quick-backup').hidden, false);
    context.select('settings');
    assert.equal(context.drawer.querySelector('.reader-hub-quick-backup').hidden, true);
    assert.ok(context.drawer.querySelector('[id="reader-panel-settings"]').querySelector('[data-backup-row]'));
  }
});

test('empty and history-disabled states do not manufacture dates, positions or mastery', () => {
  const empty = store.empty();
  const context = environment({initial: empty});
  const panel = context.drawer.querySelector('[id="reader-panel-reading"]');
  assert.equal(panel.querySelector('.reader-hub-continue'), null);
  assert.match(panel.textContent, /No reading history yet/);
  assert.doesNotMatch(panel.textContent, /since /);
  const initial = fixture();
  initial.workspace.records['reader-settings'].history = false;
  const disabled = environment({initial});
  assert.match(disabled.drawer.querySelector('[id="reader-panel-reading"]').textContent, /Existing history and saved notes are kept/);
  assert.match(disabled.drawer.querySelector('.reader-hub-list').textContent, /A real note/);
});

test('opening, browsing and exporting preserve the whole legacy envelope and existing keys', async () => {
  const initial = fixture();
  const context = environment({initial});
  const before = context.storage.getItem(store.KEY);
  context.select('saved');
  assert.match(context.drawer.querySelector('[id="reader-panel-saved"]').textContent, /not in the current catalog/);
  context.select('settings');
  context.select('reading');
  context.drawer.querySelector('.reader-hub-quick-backup').querySelector('button').click();
  assert.match(context.status(), /Backup exported/);
  const {exportedAt, ...exported} = store.parse(await context.downloads[0].text());
  assert.ok(Number.isFinite(Date.parse(exportedAt)));
  assert.deepEqual(exported, store.parse(before));
  assert.equal(context.storage.getItem(store.KEY), before);
  assert.deepEqual([...context.values.keys()], [store.KEY]);
});

test('backup controls merge legacy envelopes and bookmarks through the original APIs', async () => {
  const context = environment();
  const original = context.window.PreparationProgress.getState();
  const incoming = fixture();
  incoming.workspace.records['bookmark-' + OTHER] = {saved: true, updatedAt: LATER};
  await context.importFile(store.encode(incoming));
  await context.importFile(store.encode(incoming));
  assert.match(context.status(), /Backup merged/);
  const after = context.window.PreparationProgress.getState();
  assert.deepEqual(after.entries, original.entries);
  assert.deepEqual(after.schedule, original.schedule);
  assert.deepEqual(after.mode, original.mode);
  assert.equal(after.workspace.records['bookmark-' + UNKNOWN].saved, true);
  assert.equal(after.workspace.records['bookmark-' + OTHER].saved, true);
  assert.deepEqual(after.workspace.records['legacy-library'], original.workspace.records['legacy-library']);
  assert.ok([...context.values.keys()].some(key => key.endsWith(':snapshots')));
  const legacy = environment();
  await legacy.importFile(JSON.stringify({format: 'cooking-agi-bookmarks', saved: ['other.md', 'missing.md', {bad: true}]}));
  assert.equal(legacy.records()['bookmark-' + OTHER].saved, true);
  assert.equal(legacy.records()['bookmark-' + NOTE].saved, true);
  assert.match(legacy.status(), /Imported 1 saved notes/);
});

test('malformed, oversized, unreadable and future backups leave persistent records unchanged', async () => {
  for (const [text, options] of [
    ['not json', {}], [JSON.stringify({...fixture(), version: 999}), {}],
    [JSON.stringify({format: 'cooking-agi-bookmarks', saved: []}), {size: 2 * 1024 * 1024 + 1}],
    ['{}', {text: async () => { throw Error('unreadable'); }}],
    ['{"format":"cooking-agi-prep-progress","__proto__":{"polluted":true}}', {}]
  ]) {
    const context = environment();
    const before = context.storage.getItem(store.KEY);
    await context.importFile(text, options);
    assert.match(context.status(), /Could not read this backup/);
    assert.equal(context.storage.getItem(store.KEY), before);
  }
});

test('storage and catalog failures report errors without claiming a successful import or export', async () => {
  const context = environment();
  const before = context.storage.getItem(store.KEY);
  context.storage.setItem = () => { throw Error('quota'); };
  await context.importFile(store.encode(fixture()));
  assert.match(context.status(), /browser could not save/);
  assert.equal(context.storage.getItem(store.KEY), before);
  await context.importFile(JSON.stringify({format: 'cooking-agi-bookmarks', saved: ['other.md']}));
  assert.match(context.status(), /Could not save the imported notes/);
  context.window.PreparationProgress.exportBackup = () => { throw Error('denied'); };
  context.drawer.querySelector('.reader-hub-quick-backup').querySelector('button').click();
  assert.match(context.status(), /Could not export/);
  const offline = environment({failCatalog: true});
  const offlineBefore = offline.storage.getItem(store.KEY);
  await offline.importFile(JSON.stringify({format: 'cooking-agi-bookmarks', saved: ['chapter.md']}));
  assert.match(offline.status(), /Could not read this backup/);
  assert.equal(offline.storage.getItem(store.KEY), offlineBefore);
});

test('Escape returns focus and settings redraws preserve keyboard focus', () => {
  const context = environment();
  context.select('settings');
  const history = context.drawer.querySelector('[role="switch"]');
  history.click();
  assert.equal(context.document.activeElement.attributes['role'], 'switch');
  context.documentEvents.keydown({key: 'Escape'});
  assert.equal(context.drawer.hidden, true);
  assert.equal(context.document.activeElement, context.toggle);
  assert.equal(context.toggle.attributes['aria-expanded'], 'false');
});
