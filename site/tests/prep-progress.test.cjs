// window.PreparationProgress: the small API the save button, the reading drawer,
// and note positions share, plus the whole-envelope backup.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const api = require('../static/prep-progress-store.js');
const libraryState = require('../static/prep-library-state.js');
const source = fs.readFileSync(path.join(__dirname, '../static/prep-progress.js'), 'utf8');
const first = 'note-0123456789abcdef';
const second = 'note-fedcba9876543210';
const early = '2026-10-10T12:00:00.000Z';
const later = '2026-10-12T12:00:00.000Z';

function environment({initial = null, denied = false} = {}) {
  const data = new Map(initial ? [[api.KEY, initial]] : []);
  const storage = {
    getItem(key) { if (denied) throw Error('denied'); return data.get(key) ?? null; },
    setItem(key, value) { if (denied) throw Error('denied'); data.set(key, value); }
  };
  const listeners = {};
  const fired = [];
  const window = {
    PrepProgressStore: api, PrepLibraryState: libraryState, localStorage: storage,
    addEventListener(name, callback) { (listeners[name] ||= []).push(callback); },
    dispatchEvent(event) { fired.push(event.type); (listeners[event.type] || []).forEach(callback => callback(event)); }
  };
  vm.runInNewContext(source, {window, Event});
  return {data, window, listeners, fired, progress: window.PreparationProgress,
    state() { return api.parse(data.get(api.KEY)); }};
}

function withBookmarks(records) {
  const state = api.empty();
  state.workspace = {version: 1, records, history: []};
  return state;
}

test('the API is small: library, backup and health only', () => {
  const {progress} = environment();
  assert.deepEqual(Object.keys(progress).sort(), ['exportBackup', 'getCatalog', 'getHealth', 'getState',
    'importBackup', 'library', 'storageKey', 'updateWorkspace']);
  assert.equal(progress.storageKey, api.KEY);
  assert.equal(typeof progress.library.read, 'function');
});

test('an export carries the whole envelope, including records the site no longer shows', () => {
  const original = withBookmarks({['bookmark-' + first]: {saved: true, updatedAt: early}});
  original.entries[first] = {status: 'learning', title: 'Old study record', note: 'Kept', updatedAt: early};
  const {progress} = environment({initial: api.encode(original)});
  const exported = api.parse(progress.exportBackup());
  assert.equal(exported.workspace.records['bookmark-' + first].saved, true);
  assert.equal(exported.entries[first].note, 'Kept');
});

test('reading changes, reload and backup restoration retain retired plans and check-ins', () => {
  const original = withBookmarks({
    ['bookmark-' + first]: {saved: true, updatedAt: early},
    ['position-' + first]: {anchor: 'input', language: 'zh', updatedAt: early},
    'reads-20261010': {notes: {[first]: 2}, updatedAt: early}
  });
  original.mode = {value: 'plan', updatedAt: early};
  original.entries[first] = {status: 'learning', title: 'Old study record', note: 'Keep my memo', updatedAt: early};
  original.schedule = {version: 1, sessions: {
    'session-0123456789abcdef0123456789abcdef': {
      title: 'Review attention', date: '2026-10-10', minutes: 30, noteId: first,
      status: 'done', completedOn: '2026-10-10', actualMinutes: 25, updatedAt: early
    },
    'session-fedcba9876543210fedcba9876543210': {
      title: 'Read when ready', date: '2026-10-12', minutes: 20, noteId: second,
      status: 'planned', completedOn: '', actualMinutes: null, updatedAt: early
    }
  }};
  const context = environment({initial: api.encode(original)});
  context.progress.updateWorkspace({['bookmark-' + second]: {saved: true, updatedAt: later}});
  const reloaded = environment({initial: context.data.get(api.KEY)});
  const restored = environment();
  assert.equal(restored.progress.importBackup(reloaded.progress.exportBackup()).persisted, true);
  const result = restored.state();
  assert.deepEqual(result.schedule, original.schedule);
  assert.deepEqual(result.entries, original.entries);
  assert.deepEqual(result.mode, original.mode);
  for (const [key, record] of Object.entries(original.workspace.records)) {
    assert.deepEqual(result.workspace.records[key], record);
  }
  assert.equal(result.workspace.records['bookmark-' + second].saved, true);
});

test('an import merges: the newer copy of a record wins and local-only records stay', () => {
  const local = withBookmarks({
    ['bookmark-' + first]: {saved: true, updatedAt: later},
    ['bookmark-' + second]: {saved: false, updatedAt: early}
  });
  const context = environment({initial: api.encode(local)});
  const incoming = withBookmarks({
    ['bookmark-' + first]: {saved: false, updatedAt: early},
    ['bookmark-' + second]: {saved: true, updatedAt: later}
  });
  const result = context.progress.importBackup(api.encode(incoming));
  const records = context.state().workspace.records;
  assert.equal(records['bookmark-' + first].saved, true);
  assert.equal(records['bookmark-' + second].saved, true);
  assert.deepEqual({...result}, {bookmarks: 2, addedBookmarks: 1, persisted: true});
  assert.ok(context.fired.includes('notes:prep-restored'));
});

test('an unreadable or future backup throws and leaves local records untouched', () => {
  const local = api.encode(withBookmarks({['bookmark-' + first]: {saved: true, updatedAt: early}}));
  const context = environment({initial: local});
  assert.throws(() => context.progress.importBackup('{not json'));
  assert.throws(() => context.progress.importBackup(JSON.stringify({...api.empty(), version: 99})));
  assert.equal(context.data.get(api.KEY), local);
  assert.ok(!context.fired.includes('notes:prep-restored'));
});

test('a change from another tab re-reads storage and tells the page', () => {
  const context = environment();
  context.listeners.storage.forEach(callback => callback({key: 'unrelated'}));
  assert.ok(!context.fired.includes('notes:prep-storage'));
  context.listeners.storage.forEach(callback => callback({key: api.KEY}));
  assert.ok(context.fired.includes('notes:prep-storage'));
});
