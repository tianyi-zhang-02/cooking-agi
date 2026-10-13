const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const hub = require(path.join(__dirname, '../static/reader-hub.js'));
const store = require(path.join(__dirname, '../static/prep-progress-store.js'));

const NOTE_A = 'note-0123456789abcdef';
const NOTE_B = 'note-fedcba9876543210';
const AT = '2026-10-12T08:00:00.000Z';
const day = (y, m, d) => new Date(y, m - 1, d);

function memoryStorage() {
  const values = new Map();
  return {getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value)};
}

function readingState(notes, updatedAt = AT) {
  return {...store.empty(), workspace: {version: 1, records: {
    'reads-20261012': {notes, updatedAt}
  }, history: []}};
}

test('clear survives the real store, reload and export without clearing unrelated work', () => {
  const disk = memoryStorage();
  const progress = store.createStore(disk);
  progress.updateEntry(NOTE_A, {title: 'A note', note: 'Try a different mask', status: 'learning', updatedAt: AT});
  progress.updateWorkspace({
    'reads-20261012': {notes: {[NOTE_A]: 2, [NOTE_B]: 1}, updatedAt: AT, future: {keep: true}},
    'reads-20261011': {notes: {[NOTE_B]: 1}, updatedAt: AT},
    ['bookmark-' + NOTE_A]: {saved: true, updatedAt: AT},
    'reader-settings': {history: true, updatedAt: AT}
  });
  const before = progress.read();
  const result = progress.updateWorkspace(hub.clear(before.workspace.records, '2026-10-12T09:00:00.000Z'));
  assert.equal(result.persisted, true);
  const reloaded = store.createStore(disk);
  const after = store.parse(reloaded.serialize());
  assert.deepEqual(after.workspace.records['reads-20261012'].notes, {});
  assert.deepEqual(after.workspace.records['reads-20261011'].notes, {});
  assert.deepEqual(after.workspace.records['reads-20261012'].future, {keep: true});
  assert.deepEqual(after.entries, before.entries);
  assert.deepEqual(after.workspace.records['bookmark-' + NOTE_A], before.workspace.records['bookmark-' + NOTE_A]);
  assert.deepEqual(after.workspace.records['reader-settings'], before.workspace.records['reader-settings']);
  assert.equal(hub.summarize(after.workspace.records, day(2026, 10, 12)).span, 0);
  reloaded.updateWorkspace(hub.increment(after.workspace.records, NOTE_A, day(2026, 10, 12), AT));
  assert.deepEqual(reloaded.read().workspace.records['reads-20261012'].notes, {[NOTE_A]: 1});
});

test('90-day retention clears persisted counts rather than only hiding them', () => {
  const disk = memoryStorage();
  const progress = store.createStore(disk);
  progress.updateWorkspace({
    'reads-20260601': {notes: {[NOTE_A]: 3}, updatedAt: AT},
    'reads-20261011': {notes: {[NOTE_B]: 1}, updatedAt: AT}
  });
  progress.updateWorkspace(hub.increment(progress.read().workspace.records, NOTE_A, day(2026, 10, 12), AT));
  const records = store.createStore(disk).read().workspace.records;
  assert.deepEqual(records['reads-20260601'].notes, {});
  assert.deepEqual(records['reads-20261011'].notes, {[NOTE_B]: 1});
});

test('backup merge chooses a whole reading snapshot, including an empty one', () => {
  const earlier = readingState({[NOTE_A]: 3});
  const cleared = readingState({}, '2026-10-12T09:00:00.000Z');
  for (const policy of ['keep-local', 'newer']) {
    assert.deepEqual(store.merge(cleared, earlier, policy).state.workspace.records['reads-20261012'].notes, {});
  }
  for (const policy of ['newer', 'prefer-backup']) {
    assert.deepEqual(store.merge(earlier, cleared, policy).state.workspace.records['reads-20261012'].notes, {});
  }
  assert.deepEqual(store.merge(earlier, cleared, 'keep-local').state.workspace.records['reads-20261012'].notes, {[NOTE_A]: 3});
  assert.deepEqual(store.merge(cleared, earlier, 'prefer-backup').state.workspace.records['reads-20261012'].notes, {[NOTE_A]: 3});
  const replacement = readingState({[NOTE_B]: 1}, '2026-10-12T10:00:00.000Z');
  assert.deepEqual(store.merge(earlier, replacement, 'newer').state.workspace.records['reads-20261012'].notes, {[NOTE_B]: 1});
  const progress = store.createStore(memoryStorage());
  progress.applyImport(earlier, 'newer');
  progress.applyImport(cleared, 'newer');
  assert.deepEqual(store.parse(progress.serialize()).workspace.records['reads-20261012'].notes, {});
});

test('a failed clear stays cleared in memory and persists when storage recovers', () => {
  const disk = memoryStorage();
  const progress = store.createStore(disk);
  progress.applyImport(readingState({[NOTE_A]: 3}), 'newer');
  const persist = disk.setItem;
  disk.setItem = () => { throw Error('quota'); };
  const cleared = progress.updateWorkspace(hub.clear(progress.read().workspace.records, '2026-10-12T09:00:00.000Z'));
  assert.equal(cleared.persisted, false);
  assert.deepEqual(progress.read().workspace.records['reads-20261012'].notes, {});
  assert.deepEqual(store.parse(progress.serialize()).workspace.records['reads-20261012'].notes, {});
  assert.deepEqual(store.parse(disk.getItem(store.KEY)).workspace.records['reads-20261012'].notes, {[NOTE_A]: 3});
  disk.setItem = persist;
  const anotherTab = store.createStore(disk);
  anotherTab.updateWorkspace({['bookmark-' + NOTE_B]: {saved: true, updatedAt: AT}});
  progress.updateWorkspace({'reader-settings': {history: false, updatedAt: AT}});
  const reloaded = store.createStore(disk).read().workspace.records;
  assert.deepEqual(reloaded['reads-20261012'].notes, {});
  assert.equal(reloaded['bookmark-' + NOTE_B].saved, true);
});

test('new offline reading days keep complete snapshots when writes are retried', () => {
  const disk = memoryStorage();
  const persist = disk.setItem;
  disk.setItem = () => { throw Error('quota'); };
  const progress = store.createStore(disk);
  progress.updateWorkspace(hub.increment({}, NOTE_A, day(2026, 10, 12), AT));
  progress.updateWorkspace(hub.clear(progress.read().workspace.records, '2026-10-12T09:00:00.000Z'));
  assert.deepEqual(progress.read().workspace.records['reads-20261012'].notes, {});
  disk.setItem = persist;
  progress.updateWorkspace({'reader-settings': {history: false, updatedAt: AT}});
  assert.deepEqual(store.createStore(disk).read().workspace.records['reads-20261012'].notes, {});
});

test('a read lands in the local day record and counts up', () => {
  let records = {};
  for (let i = 0; i < 3; i++) Object.assign(records, hub.increment(records, NOTE_A, day(2026, 10, 12), AT));
  Object.assign(records, hub.increment(records, NOTE_B, day(2026, 10, 12), AT));
  assert.deepEqual(records['reads-20261012'].notes, {[NOTE_A]: 3, [NOTE_B]: 1});
  assert.equal(records['reads-20261012'].updatedAt, AT);
  assert.deepEqual(records['reading-total'], {total: 4, since: '2026-10-12', updatedAt: AT});
});

test('the all-time total keeps counting after old days are emptied', () => {
  let records = {'reads-20260601': {notes: {[NOTE_A]: 2}, updatedAt: AT}, 'reading-total': {total: 40, since: '2026-03-01', updatedAt: AT}};
  Object.assign(records, hub.increment(records, NOTE_B, day(2026, 10, 12), AT));
  assert.deepEqual(records['reads-20260601'].notes, {});
  assert.deepEqual(records['reading-total'], {total: 41, since: '2026-03-01', updatedAt: AT});
  assert.deepEqual(hub.lifetime(records, day(2026, 10, 12)), {total: 41, since: '2026-03-01'});
});

test('without a stored total, the days on record seed it', () => {
  const records = {'reads-20261010': {notes: {[NOTE_A]: 2}, updatedAt: AT}, 'reads-20261012': {notes: {[NOTE_B]: 1}, updatedAt: AT}};
  assert.deepEqual(hub.lifetime(records, day(2026, 10, 12)), {total: 3, since: '2026-10-10'});
  assert.deepEqual(hub.lifetime({}, day(2026, 10, 12)), {total: 0, since: '2026-10-12'});
  // A stale total never reads lower than the days still kept.
  assert.equal(hub.lifetime(Object.assign({'reading-total': {total: 1, since: '2026-10-12', updatedAt: AT}}, records), day(2026, 10, 12)).total, 3);
});

test('invalid note IDs are ignored', () => {
  assert.deepEqual(hub.increment({}, 'https://example.com', day(2026, 10, 12), AT), {});
});

test('days past the 90-day window are emptied, not deleted', () => {
  const records = {'reads-20260601': {notes: {[NOTE_A]: 2}, updatedAt: AT}, 'reads-20261001': {notes: {[NOTE_B]: 1}, updatedAt: AT}};
  const changes = hub.increment(records, NOTE_A, day(2026, 10, 12), AT);
  assert.deepEqual(changes['reads-20260601'], {notes: {}, updatedAt: AT});
  assert.equal(changes['reads-20261001'], undefined);
});

test('summary covers today, this week, 13 weeks, the streak and recent notes', () => {
  // 2026-10-14 is a Wednesday, so the week runs Monday 12th to Sunday 18th.
  const records = {
    'reads-20261014': {notes: {[NOTE_A]: 2}, updatedAt: AT},
    'reads-20261013': {notes: {[NOTE_B]: 1, [NOTE_A]: 1}, updatedAt: AT},
    'reads-20261011': {notes: {[NOTE_B]: 4}, updatedAt: AT},
    'reads-20261007': {notes: {[NOTE_A]: 3}, updatedAt: AT},
    'reads-20260901': {notes: {[NOTE_B]: 5}, updatedAt: AT},
    'reader-settings': {history: true, updatedAt: AT}
  };
  const summary = hub.summarize(records, day(2026, 10, 14));
  assert.equal(summary.today.total, 2);
  assert.deepEqual(summary.today.notes, [[NOTE_A, 2]]);
  assert.deepEqual(summary.week.days.map(item => item.total), [0, 2, 2, 0, 0, 0, 0]);
  assert.deepEqual(summary.week.days.map(item => item.future), [false, false, false, true, true, true, true]);
  assert.equal(summary.week.days[0].date.getDate(), 12);
  assert.equal(summary.week.total, 4);
  assert.equal(summary.week.notes, 2);
  assert.equal(summary.week.last, 7);
  assert.equal(summary.heat.length, hub.WEEKS);
  assert.ok(summary.heat.every(column => column.length === 7 && column[0].date.getDay() === 1));
  assert.equal(summary.heat[hub.WEEKS - 1][2].total, 2);
  assert.equal(summary.span, 16);
  assert.equal(summary.streak, 2);
  assert.deepEqual(summary.lifetime, {total: 16, since: '2026-09-01'});
  const b = summary.recent.find(item => item.id === NOTE_B);
  assert.equal(b.lastAge, 1);
  assert.equal(b.total, 10);
});

test('a streak survives a day with no reads yet, and Sunday closes the week', () => {
  const records = {'reads-20261017': {notes: {[NOTE_A]: 1}, updatedAt: AT}, 'reads-20261016': {notes: {[NOTE_A]: 1}, updatedAt: AT}};
  assert.equal(hub.summarize(records, day(2026, 10, 18)).streak, 2);
  const sunday = hub.summarize(records, day(2026, 10, 18));
  assert.ok(sunday.week.days.every(item => !item.future));
  assert.equal(sunday.week.total, 2);
});

test('clearing empties every reading day and leaves other records alone', () => {
  const records = {'reads-20261012': {notes: {[NOTE_A]: 2}, updatedAt: AT}, 'bookmark-x': {saved: true, updatedAt: AT}};
  assert.deepEqual(hub.clear(records, AT, day(2026, 10, 12)), {
    'reads-20261012': {notes: {}, updatedAt: AT},
    'reading-total': {total: 0, since: '2026-10-12', updatedAt: AT}
  });
});

test('settings default to counting on and the Star notice allowed', () => {
  assert.deepEqual(hub.settings({}), {history: true, mascot: 'collapsed', buddy: 'shown', starAskedOn: '', starDone: false, snoozedUntil: ''});
  assert.equal(hub.settings({'reader-settings': {buddy: 'hidden', updatedAt: AT}}).buddy, 'hidden');
  assert.equal(hub.settings({'reader-settings': {buddy: 'gone', updatedAt: AT}}).buddy, 'shown');
  assert.equal(hub.settings({'reader-settings': {history: false, mascot: 'hidden', updatedAt: AT}}).mascot, 'hidden');
  assert.equal(hub.settings({'reader-settings': {mascot: 'dancing', updatedAt: AT}}).mascot, 'collapsed');
});

test('the shared store accepts these records and rejects malformed ones', () => {
  const base = {format: 'cooking-agi-prep-progress', version: 1, site: 'cooking-agi', entries: {}, mode: {value: 'review', updatedAt: AT}};
  const ok = Object.assign({}, base, {workspace: {version: 1, records: {
    'reads-20261012': {notes: {[NOTE_A]: 2}, updatedAt: AT},
    'reading-total': {total: 12, since: '2026-10-01', updatedAt: AT},
    'reader-settings': {history: false, mascot: 'open', buddy: 'hidden', starAskedOn: '2026-10-12', starDone: true, snoozedUntil: AT, updatedAt: AT}}, history: []}});
  assert.doesNotThrow(() => store.parse(JSON.stringify(ok)));
  for (const bad of [
    {'reads-2026-10-12': {notes: {}, updatedAt: AT}},
    {'reads-20261012': {notes: {[NOTE_A]: -1}, updatedAt: AT}},
    {'reads-20261012': {notes: {'not-a-note': 1}, updatedAt: AT}},
    {'reading-total': {total: -1, since: '', updatedAt: AT}},
    {'reading-total': {total: 1.5, since: '', updatedAt: AT}},
    {'reading-total': {total: 3, since: '2026-02-30', updatedAt: AT}},
    {'reading-total': {total: 3, updatedAt: AT}},
    {'reader-settings': {mascot: 'dancing', updatedAt: AT}},
    {'reader-settings': {buddy: 'gone', updatedAt: AT}},
    {'reader-settings': {starAskedOn: 'tomorrow', updatedAt: AT}},
    {'reader-settings': {starDone: 'yes', updatedAt: AT}},
    {'reader-settings': {snoozedUntil: 'later', updatedAt: AT}}
  ]) {
    const value = Object.assign({}, base, {workspace: {version: 1, records: bad, history: []}});
    assert.throws(() => store.parse(JSON.stringify(value)));
  }
});
