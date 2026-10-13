/* Anonymous counters for cooking-agi.

   Two things are counted:
   - presence: how many people have the site open right now, and how many
     different people opened it today (UTC). Shown by the study buddy.
   - note metrics: reads and saves per note, exactly as specified in
     site/tools/note-metrics.md (the page only sends them after opting in).

   Nothing here stores an IP address or sets a cookie. A visitor is a hash of
   IP + user agent with a salt that is random for each UTC day and replaced
   the next day, so the same person cannot be linked across days. Note tokens
   arrive already hashed with a secret pepper by the Worker. The code has no
   logging of requests, IPs or tokens.

   This module has no Cloudflare imports so it can be tested in Node. */

export const NOTE = /^note-[a-f0-9]{16}$/;
export const TOKEN = /^[a-f0-9]{32}$/;
export const ONLINE_MS = 150_000;             // a heartbeat every 2 minutes keeps you "here"
export const LIMITS = {beat: [6, 60], look: [120, 60], write: [30, 60]};   // requests per seconds, per visitor
const MAX_BODY = 1024;

export const utcDay = date => date.toISOString().slice(0, 10);
export const utcMonth = date => date.toISOString().slice(0, 7);
const hex = bytes => Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('');
export async function sha256(text) {
  return hex(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))));
}

export const SCHEMA = [
  'CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID',
  'CREATE TABLE IF NOT EXISTS visitors (day TEXT NOT NULL, id TEXT NOT NULL, PRIMARY KEY (day, id)) WITHOUT ROWID',
  'CREATE TABLE IF NOT EXISTS days (day TEXT PRIMARY KEY, visitors INTEGER NOT NULL) WITHOUT ROWID',
  'CREATE TABLE IF NOT EXISTS reads (note TEXT NOT NULL, day TEXT NOT NULL, token TEXT NOT NULL, PRIMARY KEY (note, day, token)) WITHOUT ROWID',
  'CREATE TABLE IF NOT EXISTS saves (note TEXT NOT NULL, token TEXT NOT NULL, month TEXT NOT NULL, PRIMARY KEY (note, token)) WITHOUT ROWID',
  'CREATE TABLE IF NOT EXISTS totals (note TEXT PRIMARY KEY, reads INTEGER NOT NULL DEFAULT 0, saves INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID',
  'CREATE TABLE IF NOT EXISTS months (note TEXT NOT NULL, month TEXT NOT NULL, reads INTEGER NOT NULL DEFAULT 0, saves INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (note, month)) WITHOUT ROWID'
];

const refuse = (status, error, retry) => ({refused: {status, error, ...(retry ? {retry} : {})}});

/* The state behind one Durable Object. `sql.exec(query, ...params)` returns a
   cursor with toArray() (Cloudflare) or a promise of one (tests). */
export class CounterCore {
  constructor(sql, {clock = () => new Date(), random = () => crypto.getRandomValues(new Uint8Array(16))} = {}) {
    this.sql = sql;
    this.clock = clock;
    this.random = random;
    this.online = new Map();    // visitor -> last heartbeat (ms); memory only
    this.windows = new Map();   // visitor:kind -> {start, count}; memory only
    this.day = '';
    this.salt = '';
  }
  async rows(query, ...params) { return (await this.sql.exec(query, ...params)).toArray(); }
  async init() {
    for (const statement of SCHEMA) await this.rows(statement);
    const meta = Object.fromEntries((await this.rows('SELECT key, value FROM meta')).map(row => [row.key, row.value]));
    this.day = meta.day || '';
    this.salt = meta.salt || '';
  }
  // A new UTC day: a fresh salt, and yesterday's visitor hashes and read keys are dropped.
  async roll(day) {
    if (day === this.day && this.salt) return;
    this.day = day;
    this.salt = hex(this.random());
    this.online.clear();
    await this.rows('DELETE FROM visitors WHERE day <> ?', day);
    await this.rows('DELETE FROM days WHERE day <> ?', day);
    await this.rows('DELETE FROM reads WHERE day <> ?', day);
    await this.rows("INSERT INTO meta (key, value) VALUES ('day', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", day);
    await this.rows("INSERT INTO meta (key, value) VALUES ('salt', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", this.salt);
  }
  async visitor(client) {
    return (await sha256(`${this.salt}\n${client?.ip || ''}\n${client?.ua || ''}`)).slice(0, 32);
  }
  allow(visitor, kind, now) {
    const [limit, seconds] = LIMITS[kind];
    const key = visitor + ':' + kind;
    const window = this.windows.get(key);
    if (!window || now - window.start >= seconds * 1000) {
      if (this.windows.size > 20000) for (const [old, value] of this.windows) if (now - value.start >= 120000) this.windows.delete(old);
      this.windows.set(key, {start: now, count: 1});
      return 0;
    }
    window.count += 1;
    return window.count > limit ? Math.ceil((window.start + seconds * 1000 - now) / 1000) : 0;
  }
  async enter(client, kind) {
    const now = this.clock();
    await this.roll(utcDay(now));
    const visitor = await this.visitor(client);
    const wait = this.allow(visitor, kind, now.getTime());
    return {now, visitor, wait};
  }

  async presence(beat, client) {
    const {now, visitor, wait} = await this.enter(client, beat ? 'beat' : 'look');
    if (wait) return refuse(429, 'rate', wait);
    const ms = now.getTime();
    if (beat) {
      this.online.set(visitor, ms);
      const added = await this.rows('INSERT OR IGNORE INTO visitors (day, id) VALUES (?, ?) RETURNING id', this.day, visitor);
      if (added.length) await this.rows('INSERT INTO days (day, visitors) VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET visitors = visitors + 1', this.day);
    }
    for (const [id, seen] of this.online) if (ms - seen > ONLINE_MS) this.online.delete(id);
    const today = (await this.rows('SELECT visitors FROM days WHERE day = ?', this.day))[0]?.visitors || 0;
    const here = this.online.size;
    return {version: 1, day: this.day, now: here, today: Math.max(today, here)};
  }

  async snapshot(note, client) {
    const {now, wait} = await this.enter(client, 'look');
    if (wait) return refuse(429, 'rate', wait);
    return this.counts(note, utcMonth(now));
  }
  async counts(note, month) {
    const total = (await this.rows('SELECT reads, saves FROM totals WHERE note = ?', note))[0] || {reads: 0, saves: 0};
    const current = (await this.rows('SELECT reads, saves FROM months WHERE note = ? AND month = ?', note, month))[0] || {reads: 0, saves: 0};
    return {version: 1, noteId: note, month, mode: 'public',
      reads: total.reads, monthReads: current.reads, saves: total.saves, monthSaves: current.saves};
  }
  // token is already a peppered hash; kind and saved were validated by the Worker
  async record(note, kind, token, saved, client) {
    const {now, wait} = await this.enter(client, 'write');
    if (wait) return refuse(429, 'rate', wait);
    const month = utcMonth(now);
    const bump = async (field, by, inMonth) => {
      await this.rows(`INSERT INTO totals (note, ${field}) VALUES (?, ?) ON CONFLICT(note) DO UPDATE SET ${field} = MAX(0, ${field} + ?)`, note, Math.max(0, by), by);
      await this.rows(`INSERT INTO months (note, month, ${field}) VALUES (?, ?, ?) ON CONFLICT(note, month) DO UPDATE SET ${field} = MAX(0, ${field} + ?)`, note, inMonth, Math.max(0, by), by);
    };
    if (kind === 'view') {
      // once per browser, note and UTC day
      if ((await this.rows('INSERT OR IGNORE INTO reads (note, day, token) VALUES (?, ?, ?) RETURNING note', note, this.day, token)).length) await bump('reads', 1, month);
    } else if (saved) {
      // a save sets a state; sending it twice changes nothing
      if ((await this.rows('INSERT OR IGNORE INTO saves (note, token, month) VALUES (?, ?, ?) RETURNING note', note, token, month)).length) await bump('saves', 1, month);
    } else {
      // removing a save takes it off the month it was added in
      const removed = await this.rows('DELETE FROM saves WHERE note = ? AND token = ? RETURNING month', note, token);
      if (removed.length) await bump('saves', -1, removed[0].month);
    }
    return this.counts(note, month);
  }
}

/* The public catalog decides which note IDs exist. Cached per isolate. */
export class Catalog {
  constructor(url, fetcher = (...args) => fetch(...args), ttl = 3600_000, clock = () => Date.now()) {
    Object.assign(this, {url, fetcher, ttl, clock, notes: null, loaded: 0});
  }
  async has(note) {
    if (!this.notes || this.clock() - this.loaded > this.ttl) {
      try {
        const response = await this.fetcher(this.url, {cf: {cacheTtl: 300}});
        if (!response.ok) throw Error('catalog');
        const data = await response.json();
        this.notes = new Set(Object.keys(data.notes || {}).filter(id => NOTE.test(id)));
        this.loaded = this.clock();
      } catch (error) {
        if (!this.notes) throw error;     // keep serving the last good list
      }
    }
    return this.notes.has(note);
  }
}

/* Exactly the event shapes the page sends; anything else is refused. */
export async function readEvent(request) {
  if ((request.headers.get('Content-Type') || '').split(';')[0].trim() !== 'application/json') return refuse(415, 'content type');
  const declared = Number(request.headers.get('Content-Length') || 0);
  if (declared > MAX_BODY) return refuse(413, 'size');
  const text = await request.text();
  if (!text || new TextEncoder().encode(text).length > MAX_BODY) return refuse(413, 'size');
  let event;
  try { event = JSON.parse(text); } catch { return refuse(400, 'event'); }
  if (!event || typeof event !== 'object' || Array.isArray(event) || !['view', 'bookmark'].includes(event.kind)) return refuse(400, 'event');
  const keys = Object.keys(event).sort().join(',');
  if (keys !== (event.kind === 'view' ? 'kind,token' : 'kind,saved,token') || typeof event.token !== 'string' || !TOKEN.test(event.token) ||
      (event.kind === 'bookmark' && typeof event.saved !== 'boolean')) return refuse(400, 'event');
  return {event};
}

/* The Worker: routing, origin checks, validation. `counters` is the Durable
   Object stub (or a CounterCore in tests); `catalog` is a Catalog. */
export async function handle(request, env, counters, catalog) {
  const url = new URL(request.url);
  const origin = request.headers.get('Origin') || '';
  const allowed = String(env.ALLOWED_ORIGINS || '').split(',').map(item => item.trim()).filter(Boolean);
  const trusted = allowed.includes(origin);
  const headers = extra => ({'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
    'Vary': 'Origin', ...(trusted ? {'Access-Control-Allow-Origin': origin} : {}), ...extra});
  const json = (status, value, extra = {}) => new Response(JSON.stringify(value), {status, headers: headers(extra)});
  const answer = result => (result && result.refused
    ? json(result.refused.status, {error: result.refused.error}, result.refused.retry ? {'Retry-After': String(result.refused.retry)} : {})
    : json(200, result));

  if (request.method === 'OPTIONS') {
    if (!trusted) return json(403, {error: 'origin'});
    return new Response(null, {status: 204, headers: headers({'Access-Control-Allow-Methods': 'GET, POST', 'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Max-Age': '86400'})});
  }
  if (request.method !== 'GET' && request.method !== 'POST') return json(405, {error: 'method'});
  // writes only from the site's own pages
  if (request.method === 'POST' && !trusted) return json(403, {error: 'origin'});
  const client = {ip: request.headers.get('CF-Connecting-IP') || '', ua: (request.headers.get('User-Agent') || '').slice(0, 300)};
  try {
    if (url.pathname === '/api/presence') return answer(await counters.presence(request.method === 'POST', client));
    const match = url.pathname.match(/^\/api\/note-metrics\/([^/]+)$/);
    if (!match) return json(404, {error: 'path'});
    const note = match[1];
    if (!NOTE.test(note) || !(await catalog.has(note))) return json(404, {error: 'note'});
    if (request.method === 'GET') return answer(await counters.snapshot(note, client));
    if (String(env.TOKEN_PEPPER || '').length < 32) return json(503, {error: 'not configured'});
    const parsed = await readEvent(request);
    if (parsed.refused) return answer(parsed);
    const {event} = parsed;
    const token = await sha256(`${env.TOKEN_PEPPER}:${note}:${event.token}`);
    return answer(await counters.record(note, event.kind, token, event.kind === 'bookmark' ? event.saved : null, client));
  } catch {
    return json(503, {error: 'unavailable'});
  }
}
