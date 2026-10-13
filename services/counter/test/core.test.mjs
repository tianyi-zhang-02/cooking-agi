import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';
import {CounterCore, Catalog, handle, ONLINE_MS} from '../src/core.js';

/* A real SQLite behind the same sql.exec() shape the Durable Object uses. */
function sqlite() {
  const child = spawn(process.env.PYTHON || 'python3', [fileURLToPath(new URL('./sqlite_bridge.py', import.meta.url))], {stdio: ['pipe', 'pipe', 'inherit']});
  const lines = createInterface({input: child.stdout});
  const waiting = [];
  lines.on('line', line => waiting.shift()(JSON.parse(line)));
  const log = [];
  return {
    log,
    exec(sql, ...params) {
      log.push({sql, params});
      return new Promise((resolve, reject) => {
        waiting.push(reply => (reply.error ? reject(new Error(reply.error)) : resolve({toArray: () => reply.rows})));
        child.stdin.write(JSON.stringify({sql, params}) + '\n');
      });
    },
    close() { child.stdin.end(); }
  };
}

function setup(start = '2026-10-13T15:00:00Z') {
  const sql = sqlite();
  let time = new Date(start).getTime();
  let seed = 0;
  const core = new CounterCore(sql, {clock: () => new Date(time), random: () => new Uint8Array(16).fill(++seed)});
  return {sql, core, advance: ms => { time += ms; }, at: iso => { time = new Date(iso).getTime(); }};
}
const A = {ip: '203.0.113.1', ua: 'Firefox'};
const B = {ip: '203.0.113.2', ua: 'Safari'};
const NOTE = 'note-0123456789abcdef';
const TOKEN = 'a'.repeat(64);    // what the Worker passes: a peppered hash
const TOKEN2 = 'b'.repeat(64);

test('presence: here now and today, once per person per day', async t => {
  const {sql, core, advance, at} = setup();
  t.after(() => sql.close());
  await core.init();
  assert.deepEqual(await core.presence(true, A), {version: 1, day: '2026-10-13', now: 1, today: 1});
  advance(60_000);
  assert.deepEqual(await core.presence(true, A), {version: 1, day: '2026-10-13', now: 1, today: 1});
  assert.deepEqual(await core.presence(true, B), {version: 1, day: '2026-10-13', now: 2, today: 2});
  // looking without a heartbeat changes nothing
  assert.deepEqual(await core.presence(false, {ip: '198.51.100.9', ua: 'x'}), {version: 1, day: '2026-10-13', now: 2, today: 2});
  // A goes quiet; B keeps beating
  advance(ONLINE_MS - 30_000); await core.presence(true, B);
  advance(40_000);
  assert.deepEqual(await core.presence(true, B), {version: 1, day: '2026-10-13', now: 1, today: 2});
  // a new UTC day starts from zero with a new salt
  at('2026-10-14T00:00:05Z');
  assert.deepEqual(await core.presence(false, A), {version: 1, day: '2026-10-14', now: 0, today: 0});
  assert.deepEqual(await core.presence(true, A), {version: 1, day: '2026-10-14', now: 1, today: 1});
});

test('no IP or user agent is ever written to storage, and salts rotate daily', async t => {
  const {sql, core, at} = setup();
  t.after(() => sql.close());
  await core.init();
  await core.presence(true, A);
  const first = (await sql.exec('SELECT id FROM visitors')).toArray()[0].id;
  at('2026-10-14T09:00:00Z');
  await core.presence(true, A);
  const second = (await sql.exec('SELECT id FROM visitors')).toArray();
  assert.equal(second.length, 1, 'yesterday is gone');
  assert.notEqual(second[0].id, first, 'the same person gets an unrelated id the next day');
  const written = JSON.stringify(sql.log.filter(entry => /^\s*(INSERT|UPDATE)/i.test(entry.sql)).map(entry => entry.params));
  for (const secret of [A.ip, A.ua]) assert.ok(!written.includes(secret), secret);
});

test('rate limits per visitor', async t => {
  const {sql, core} = setup();
  t.after(() => sql.close());
  await core.init();
  for (let k = 0; k < 6; k++) assert.equal((await core.presence(true, A)).version, 1);
  const refused = await core.presence(true, A);
  assert.equal(refused.refused.status, 429);
  assert.ok(refused.refused.retry > 0);
  assert.equal((await core.presence(true, B)).version, 1, 'others are unaffected');
});

test('note reads: once per browser, note and UTC day; months roll over', async t => {
  const {sql, core, at} = setup('2026-10-30T10:00:00Z');
  t.after(() => sql.close());
  await core.init();
  assert.deepEqual(await core.snapshot(NOTE, A), {version: 1, noteId: NOTE, month: '2026-10', mode: 'public', reads: 0, monthReads: 0, saves: 0, monthSaves: 0});
  await core.record(NOTE, 'view', TOKEN, null, A);
  const twice = await core.record(NOTE, 'view', TOKEN, null, A);
  assert.equal(twice.reads, 1);
  assert.equal((await core.record(NOTE, 'view', TOKEN2, null, B)).reads, 2);
  at('2026-10-31T10:00:00Z');
  assert.equal((await core.record(NOTE, 'view', TOKEN, null, A)).reads, 3);
  at('2026-11-01T00:10:00Z');
  const november = await core.record(NOTE, 'view', TOKEN, null, A);
  assert.deepEqual([november.month, november.reads, november.monthReads], ['2026-11', 4, 1]);
});

test('note saves are a state: repeat does nothing, removal subtracts from its month', async t => {
  const {sql, core, at} = setup('2026-10-20T10:00:00Z');
  t.after(() => sql.close());
  await core.init();
  await core.record(NOTE, 'bookmark', TOKEN, true, A);
  let now = await core.record(NOTE, 'bookmark', TOKEN, true, A);
  assert.deepEqual([now.saves, now.monthSaves], [1, 1]);
  await core.record(NOTE, 'bookmark', TOKEN2, true, B);
  at('2026-11-02T10:00:00Z');
  now = await core.snapshot(NOTE, A);
  assert.deepEqual([now.saves, now.monthSaves], [2, 0]);
  // removing an October save leaves November alone
  now = await core.record(NOTE, 'bookmark', TOKEN, false, A);
  assert.deepEqual([now.saves, now.monthSaves], [1, 0]);
  now = await core.record(NOTE, 'bookmark', TOKEN, false, A);
  assert.deepEqual([now.saves, now.monthSaves], [1, 0], 'never below zero, never double-removed');
  now = await core.record(NOTE, 'bookmark', TOKEN, true, A);
  assert.deepEqual([now.saves, now.monthSaves], [2, 1]);
  assert.ok(now.monthSaves <= now.saves && now.monthReads <= now.reads);
});

/* ---------------------------------------------------------------- the Worker */
const ENV = {ALLOWED_ORIGINS: 'https://tianyi-zhang-02.github.io', TOKEN_PEPPER: 'p'.repeat(40)};
const SITE = 'https://tianyi-zhang-02.github.io';
const catalogOf = notes => new Catalog('https://example.test/catalog.json', async () => new Response(JSON.stringify({notes: Object.fromEntries(notes.map(id => [id, {}]))})));
const request = (path, {method = 'GET', origin = SITE, body, type = 'application/json', headers = {}} = {}) =>
  new Request('https://counter.example.workers.dev' + path, {method, body: body === undefined ? undefined : (typeof body === 'string' ? body : JSON.stringify(body)),
    headers: {...(origin ? {Origin: origin} : {}), ...(body !== undefined ? {'Content-Type': type} : {}), 'CF-Connecting-IP': '203.0.113.7', 'User-Agent': 'test', ...headers}});

test('Worker: CORS only for the site, writes only from the site', async t => {
  const {sql, core} = setup();
  t.after(() => sql.close());
  await core.init();
  const catalog = catalogOf([NOTE]);
  let response = await handle(request('/api/presence', {method: 'OPTIONS'}), ENV, core, catalog);
  assert.equal(response.status, 204);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), SITE);
  response = await handle(request('/api/presence', {method: 'OPTIONS', origin: 'https://evil.example'}), ENV, core, catalog);
  assert.equal(response.status, 403);
  response = await handle(request('/api/presence', {method: 'POST', origin: 'https://evil.example'}), ENV, core, catalog);
  assert.equal(response.status, 403);
  response = await handle(request('/api/presence', {method: 'POST', origin: ''}), ENV, core, catalog);
  assert.equal(response.status, 403);
  response = await handle(request('/api/presence', {method: 'POST'}), ENV, core, catalog);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {version: 1, day: '2026-10-13', now: 1, today: 1});
  assert.equal(response.headers.get('Cache-Control'), 'no-store');
  assert.equal(response.headers.get('X-Content-Type-Options'), 'nosniff');
  // reading counts from elsewhere works, but the browser gets no CORS grant
  response = await handle(request('/api/presence', {origin: 'https://evil.example'}), ENV, core, catalog);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), null);
  assert.equal((await handle(request('/api/presence', {method: 'DELETE'}), ENV, core, catalog)).status, 405);
  assert.equal((await handle(request('/elsewhere'), ENV, core, catalog)).status, 404);
});

test('Worker: note events are validated exactly and tokens are hashed before storage', async t => {
  const {sql, core} = setup();
  t.after(() => sql.close());
  await core.init();
  const catalog = catalogOf([NOTE]);
  const post = (body, extra) => handle(request('/api/note-metrics/' + NOTE, {method: 'POST', body, ...extra}), ENV, core, catalog);
  const raw = '0123456789abcdef0123456789abcdef';
  assert.equal((await handle(request('/api/note-metrics/note-ffffffffffffffff'), ENV, core, catalog)).status, 404, 'not in the catalog');
  assert.equal((await handle(request('/api/note-metrics/../../etc'), ENV, core, catalog)).status, 404);
  assert.equal((await post({kind: 'view', token: raw}, {type: 'text/plain'})).status, 415);
  assert.equal((await post('{"kind":"view","token":"' + raw + '","pad":"' + 'x'.repeat(2000) + '"}')).status, 413);
  for (const bad of [{kind: 'view'}, {kind: 'view', token: 'short'}, {kind: 'view', token: raw, extra: 1}, {kind: 'bookmark', token: raw},
    {kind: 'bookmark', token: raw, saved: 'yes'}, {kind: 'like', token: raw}, [1, 2], 'not json{']) {
    assert.equal((await post(bad)).status, 400, JSON.stringify(bad));
  }
  const ok = await post({kind: 'view', token: raw});
  assert.equal(ok.status, 200);
  assert.deepEqual(await ok.json(), {version: 1, noteId: NOTE, month: '2026-10', mode: 'public', reads: 1, monthReads: 1, saves: 0, monthSaves: 0});
  assert.equal((await (await post({kind: 'bookmark', token: raw, saved: true})).json()).saves, 1);
  const stored = JSON.stringify(sql.log.map(entry => entry.params));
  assert.ok(!stored.includes(raw), 'the raw token never reaches storage');
  assert.equal((await handle(request('/api/note-metrics/' + NOTE, {method: 'POST', body: {kind: 'view', token: raw}}), {...ENV, TOKEN_PEPPER: ''}, core, catalog)).status, 503);
});

test('Worker: when the catalog cannot be fetched at all, notes are unavailable rather than open', async t => {
  const {sql, core} = setup();
  t.after(() => sql.close());
  await core.init();
  const broken = new Catalog('https://example.test/catalog.json', async () => new Response('nope', {status: 500}));
  assert.equal((await handle(request('/api/note-metrics/' + NOTE), ENV, core, broken)).status, 503);
});
