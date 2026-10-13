(function (scope) {
  'use strict';
  const KEY = 'cooking-agi:prep-progress';
  const RECOVERY_KEY = KEY + ':before-import';
  const SNAPSHOTS_KEY = KEY + ':snapshots';
  const FORMAT = 'cooking-agi-prep-progress';
  const MAX_BYTES = 2 * 1024 * 1024;
  const statuses = ['planned', 'learning', 'revisit', 'practiced'];
  const sessionStatuses = ['planned', 'done', 'cancelled'];
  const SESSION_ID = /^session-[a-f0-9]{32}$/;
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const plain = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const copy = value => JSON.parse(JSON.stringify(value));

  function empty() {
    return {format: FORMAT, version: 1, site: 'cooking-agi', entries: {},
      mode: {value: 'review', updatedAt: '1970-01-01T00:00:00.000Z'}};
  }

  function timestamp(value) {
    return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(value) &&
      Number.isFinite(Date.parse(value)) && new Date(value).toISOString() === value;
  }

  function civilDate(value) {
    if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const [year, month, day] = value.split('-').map(Number);
    if (year < 1900 || year > 2200 || month < 1 || month > 12) return false;
    const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
    const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    return day >= 1 && day <= days[month - 1];
  }

  function validate(value) {
    if (!plain(value) || value.format !== FORMAT || value.site !== 'cooking-agi') throw Error('invalid');
    if (value.version !== 1) throw Error('version');
    if (value.schedule !== undefined) {
      if (!plain(value.schedule)) throw Error('invalid');
      if (value.schedule.version !== 1) throw Error('version');
    }
    if (value.workspace !== undefined) {
      if (!plain(value.workspace)) throw Error('invalid');
      if (value.workspace.version !== 1) throw Error('version');
      if (!plain(value.workspace.records) || Object.keys(value.workspace.records).length > 3000) throw Error('limits');
      for (const [key, record] of Object.entries(value.workspace.records)) {
        if (!/^[a-z][a-z0-9-]{0,80}$/.test(key) || !plain(record) || !timestamp(record.updatedAt)) throw Error('invalid');
        if (key === 'preferences' && typeof record.autoAdd !== 'boolean') throw Error('invalid');
        if (key.startsWith('bookmark-') && typeof record.saved !== 'boolean') throw Error('invalid');
        if (key.startsWith('visit-') && record.visitedAt !== null && !timestamp(record.visitedAt)) throw Error('invalid');
        if (key.startsWith('position-') && (typeof record.anchor !== 'string' || !/^[\p{L}\p{N}_:.-]{0,180}$/u.test(record.anchor))) throw Error('invalid');
        if ((key === 'baseline' || key.startsWith('follow-')) && (!Number.isSafeInteger(record.edition) || record.edition < 1)) throw Error('invalid');
        if (key.startsWith('follow-') && typeof record.enabled !== 'boolean') throw Error('invalid');
        if (key.startsWith('notice-') && (!['new', 'updated'].includes(record.kind) ||
            !['queued', 'available', 'dismissed', 'reviewed', 'scheduled'].includes(record.status))) throw Error('invalid');
        if ((key.startsWith('notice-') || key.startsWith('seen-')) && (!Number.isSafeInteger(record.revision) || record.revision < 1)) throw Error('invalid');
        if (key.startsWith('reads-') && (!/^reads-\d{8}$/.test(key) || !plain(record.notes) || Object.keys(record.notes).length > 200 ||
            Object.entries(record.notes).some(([note, count]) => !/^note-[a-f0-9]{16}$/.test(note) || !Number.isInteger(count) || count < 0 || count > 9999))) throw Error('invalid');
        if (key === 'reading-total' && (!Number.isSafeInteger(record.total) || record.total < 0 || record.total > 9999999 ||
            typeof record.since !== 'string' || (record.since !== '' && !civilDate(record.since)))) throw Error('invalid');
        if (key === 'reader-settings' && ((record.history !== undefined && typeof record.history !== 'boolean') ||
            (record.mascot !== undefined && !['open', 'collapsed', 'hidden'].includes(record.mascot)) ||
            (record.buddy !== undefined && !['shown', 'hidden'].includes(record.buddy)) ||
            (record.starAskedOn !== undefined && record.starAskedOn !== '' && !civilDate(record.starAskedOn)) ||
            (record.starDone !== undefined && typeof record.starDone !== 'boolean') ||
            (record.snoozedUntil !== undefined && record.snoozedUntil !== '' && !timestamp(record.snoozedUntil)))) throw Error('invalid');
      }
      if (value.workspace.history !== undefined && (!Array.isArray(value.workspace.history) || value.workspace.history.length > 100 ||
          value.workspace.history.some(event => !plain(event) || !timestamp(event.at) || typeof event.action !== 'string' ||
            event.action.length > 100 || typeof event.id !== 'string' || event.id.length > 120))) throw Error('invalid');
    }
    let count = 0;
    function inspect(node, depth) {
      if (depth > 12 || ++count > 30000) throw Error('limits');
      if (node && typeof node === 'object') {
        for (const key of Object.keys(node)) {
          if (['__proto__', 'constructor', 'prototype'].includes(key)) throw Error('invalid');
          inspect(node[key], depth + 1);
        }
      }
    }
    inspect(value, 0);
    if (!plain(value.entries) || Object.keys(value.entries).length > 1000) throw Error('limits');
    for (const [identifier, entry] of Object.entries(value.entries)) {
      if (!/^note-[a-f0-9]{16}$/.test(identifier) || !plain(entry) ||
          !statuses.includes(entry.status) || typeof entry.note !== 'string' || entry.note.length > 2000 ||
          typeof entry.title !== 'string' || entry.title.length > 300 || !timestamp(entry.updatedAt)) throw Error('invalid');
      if (entry.position !== undefined && (!plain(entry.position) ||
          !['zh', 'en'].includes(entry.position.language) || typeof entry.position.anchor !== 'string' ||
          !/^[\p{L}\p{N}_:.-]{1,180}$/u.test(entry.position.anchor))) throw Error('invalid');
    }
    if (!plain(value.mode) || !['review', 'plan'].includes(value.mode.value) || !timestamp(value.mode.updatedAt)) throw Error('invalid');
    if (value.schedule !== undefined) {
      if (!plain(value.schedule.sessions)) throw Error('invalid');
      if (Object.keys(value.schedule.sessions).length > 300) throw Error('limits');
      for (const [identifier, session] of Object.entries(value.schedule.sessions)) {
        if (!SESSION_ID.test(identifier) || !plain(session) ||
            typeof session.title !== 'string' || !session.title.trim() || session.title.length > 160 ||
            !civilDate(session.date) || !Number.isInteger(session.minutes) || session.minutes < 1 || session.minutes > 1440 ||
            typeof session.noteId !== 'string' || (session.noteId !== '' && !/^note-[a-f0-9]{16}$/.test(session.noteId)) ||
            !sessionStatuses.includes(session.status) || !timestamp(session.updatedAt)) throw Error('invalid');
        if (session.status === 'done' ? !civilDate(session.completedOn) : session.completedOn !== '') throw Error('invalid');
        if (session.exerciseId !== undefined && (typeof session.exerciseId !== 'string' || !/^exercise-[a-z0-9-]{1,80}$/.test(session.exerciseId))) throw Error('invalid');
        if (session.actualMinutes !== null && (session.status !== 'done' || !Number.isInteger(session.actualMinutes) ||
            session.actualMinutes < 0 || session.actualMinutes > 1440)) throw Error('invalid');
      }
    }
    if (new TextEncoder().encode(JSON.stringify(value)).length > MAX_BYTES) throw Error('limits');
    return copy(value);
  }

  function parse(text) {
    if (typeof text !== 'string' || new TextEncoder().encode(text).length > MAX_BYTES) throw Error('limits');
    return validate(JSON.parse(text));
  }

  function encode(state) {
    return JSON.stringify(validate(state));
  }

  function overlay(base, preferred, recordKey = '') {
    const result = {...base};
    for (const [key, value] of Object.entries(preferred)) {
      if (['__proto__', 'constructor', 'prototype'].includes(key)) throw Error('invalid');
      const readingNotes = key === 'notes' && /^reads-\d{8}$/.test(recordKey);
      result[key] = !readingNotes && plain(base[key]) && plain(value) ? overlay(base[key], value, key) : value;
    }
    return result;
  }

  function difference(base, next, recordKey = '') {
    const changes = {};
    for (const key of new Set([...Object.keys(base), ...Object.keys(next)])) {
      if (!own(next, key)) changes[key] = undefined;
      else if (plain(base[key]) && plain(next[key]) && !(key === 'notes' && /^reads-\d{8}$/.test(recordKey))) {
        const nested = difference(base[key], next[key], key);
        if (Object.keys(nested).length) changes[key] = nested;
      } else if (JSON.stringify(base[key]) !== JSON.stringify(next[key])) changes[key] = next[key];
    }
    return changes;
  }

  function merge(local, incoming, policy = 'keep-local') {
    local = validate(local);
    incoming = validate(incoming);
    if (!['keep-local', 'newer', 'prefer-backup'].includes(policy)) throw Error('invalid');
    const state = overlay(incoming, local);
    let added = 0, conflicts = 0, replaced = 0;
    for (const [identifier, entry] of Object.entries(incoming.entries)) {
      if (!own(local.entries, identifier)) {
        added++;
      } else if (JSON.stringify(local.entries[identifier]) !== JSON.stringify(entry)) {
        conflicts++;
        if (policy === 'prefer-backup' || (policy === 'newer' && Date.parse(entry.updatedAt) > Date.parse(local.entries[identifier].updatedAt))) {
          state.entries[identifier] = overlay(local.entries[identifier], entry);
          replaced++;
        }
      }
    }
    if (local.mode.updatedAt === '1970-01-01T00:00:00.000Z' || policy === 'prefer-backup' ||
        (policy === 'newer' && Date.parse(incoming.mode.updatedAt) > Date.parse(local.mode.updatedAt))) {
      state.mode = overlay(local.mode, incoming.mode);
    } else {
      state.mode = overlay(incoming.mode, local.mode);
    }
    let addedSessions = 0, conflictingSessions = 0, replacedSessions = 0;
    const localSessions = local.schedule?.sessions || {};
    for (const [identifier, session] of Object.entries(incoming.schedule?.sessions || {})) {
      if (!own(localSessions, identifier)) {
        addedSessions++;
      } else if (JSON.stringify(localSessions[identifier]) !== JSON.stringify(session)) {
        conflictingSessions++;
        if (policy === 'prefer-backup' || (policy === 'newer' && Date.parse(session.updatedAt) > Date.parse(localSessions[identifier].updatedAt))) {
          state.schedule.sessions[identifier] = overlay(localSessions[identifier], session);
          replacedSessions++;
        }
      }
    }
    let workspaceConflicts = 0;
    for (const [key, record] of Object.entries(incoming.workspace?.records || {})) {
      const current = local.workspace?.records?.[key];
      if (current && JSON.stringify(current) !== JSON.stringify(record)) {
        workspaceConflicts++;
        if (policy === 'prefer-backup' || (policy === 'newer' && record.updatedAt > current.updatedAt)) {
          state.workspace.records[key] = overlay(current, record, key);
        }
      }
    }
    if (state.workspace) {
      const events = [...(local.workspace?.history || []), ...(incoming.workspace?.history || [])];
      state.workspace.history = [...new Map(events.map(event => [event.id, event])).values()]
        .sort((left, right) => right.at.localeCompare(left.at)).slice(0, 100);
    }
    return {state: validate(state), added, conflicts, replaced, addedSessions, conflictingSessions, replacedSessions, workspaceConflicts};
  }

  function createStore(storage) {
    let committed = empty();
    let memory = copy(committed);
    let dirty = {};
    let unavailable = false;
    let blocked = false;
    let fault = '';
    let readable = false;
    let rebaseFault = '';
    let unreadableRaw = null;
    let snapshotFault = false;
    let snapshotCacheRaw = null;
    let snapshotCache = [];
    function snapshots(clone = true) {
      try {
        const raw = storage.getItem(SNAPSHOTS_KEY);
        if (!raw || raw.length > MAX_BYTES * 4) return [];
        if (raw === snapshotCacheRaw) return clone ? copy(snapshotCache) : snapshotCache;
        const list = JSON.parse(raw);
        if (!Array.isArray(list) || list.length > 3) return [];
        snapshotCache = list.filter(item => {
          try { return timestamp(item.at) && Boolean(validate(item.state)); } catch { return false; }
        });
        snapshotCacheRaw = raw;
        return clone ? copy(snapshotCache) : snapshotCache;
      } catch { return []; }
    }
    function checkpoint(state, force = false) {
      if (!Object.keys(state.entries).length && !state.schedule && !state.workspace) return;
      try {
        const previous = snapshots(false);
        const at = new Date().toISOString();
        if (previous[0] && ((!force && Date.parse(at) - Date.parse(previous[0].at) < 60000) || encode(previous[0].state) === encode(state))) return;
        const list = [{at, state: validate(state)}, ...previous].slice(0, 3);
        while (list.length > 1 && new TextEncoder().encode(JSON.stringify(list)).length > MAX_BYTES) list.pop();
        storage.setItem(SNAPSHOTS_KEY, JSON.stringify(list));
        snapshotFault = false;
      } catch { snapshotFault = true; }
    }
    function read() {
      let raw;
      try { raw = storage.getItem(KEY); }
      catch { readable = false; unavailable = true; return copy(memory); }
      readable = true;
      if (raw === null && snapshots(false).length) {
        blocked = true;
        fault = 'missing';
        unreadableRaw = null;
        return copy(memory);
      }
      let snapshot;
      try { snapshot = raw === null ? empty() : parse(raw); }
      catch (error) {
        blocked = true;
        fault = error.message;
        unreadableRaw = raw;
        return copy(memory);
      }
      blocked = false;
      unreadableRaw = null;
      try {
        const pending = overlay(snapshot, dirty);
        for (const identifier of Object.keys(dirty.entries || {})) {
          if (!own(snapshot.entries, identifier)) {
            pending.entries[identifier] = overlay(memory.entries[identifier], pending.entries[identifier]);
          }
        }
        if (dirty.schedule) {
          if (!snapshot.schedule) pending.schedule = overlay({...memory.schedule, sessions: {}}, pending.schedule);
          for (const identifier of Object.keys(dirty.schedule.sessions || {})) {
            if (!own(snapshot.schedule?.sessions || {}, identifier)) {
              pending.schedule.sessions[identifier] = overlay(memory.schedule.sessions[identifier], pending.schedule.sessions[identifier]);
            }
          }
        }
        if (dirty.workspace) {
          if (!snapshot.workspace) pending.workspace = overlay(memory.workspace, pending.workspace);
          for (const key of Object.keys(dirty.workspace.records || {})) {
            if (!own(snapshot.workspace?.records || {}, key)) pending.workspace.records[key] = overlay(memory.workspace.records[key], pending.workspace.records[key], key);
          }
        }
        const rebased = validate(pending);
        committed = snapshot;
        memory = rebased;
        rebaseFault = fault = '';
        unavailable = Object.keys(dirty).length > 0;
      } catch (error) {
        rebaseFault = fault = error.message;
        unavailable = true;
      }
      return copy(memory);
    }
    function write(next) {
      if (blocked) throw Error('blocked');
      if (rebaseFault) throw Error(rebaseFault);
      next = validate(next);
      try {
        if (!readable) throw Error('storage');
        checkpoint(committed);
        storage.setItem(KEY, JSON.stringify(next));
        committed = next;
        dirty = {};
        unavailable = false;
        if (!snapshots(false).length) checkpoint(next);
      } catch {
        dirty = difference(committed, next);
        for (const [identifier, changes] of Object.entries(dirty.schedule?.sessions || {})) {
          const completionFields = ['status', 'completedOn', 'actualMinutes'];
          if (completionFields.some(field => own(changes, field))) {
            for (const field of completionFields) changes[field] = next.schedule.sessions[identifier][field];
          }
        }
        unavailable = true;
      }
      memory = next;
      return {state: copy(memory), persisted: !unavailable};
    }
    function updateEntry(identifier, changes) {
      const state = read();
      const previous = state.entries[identifier] || {};
      state.entries[identifier] = overlay(previous, changes);
      return write(state);
    }
    function updateSession(identifier, changes) {
      if (typeof identifier !== 'string' || !SESSION_ID.test(identifier) || !plain(changes)) throw Error('invalid');
      const state = read();
      if (!state.schedule) state.schedule = {version: 1, sessions: {}};
      const previous = state.schedule.sessions[identifier] || {};
      state.schedule.sessions[identifier] = overlay(previous, changes);
      return write(state);
    }
    function addSessions(records) {
      if (!plain(records) || !Object.keys(records).length || Object.keys(records).length > 24) throw Error('invalid');
      const state = read();
      if (!state.schedule) state.schedule = {version: 1, sessions: {}};
      const active = Object.values(state.schedule.sessions).filter(entry => entry.status !== 'cancelled');
      const noteIds = new Set(active.map(entry => entry.noteId).filter(Boolean));
      const legacyNoteIds = new Set(active.filter(entry => !entry.exerciseId).map(entry => entry.noteId).filter(Boolean));
      const exerciseIds = new Set(active.map(entry => entry.exerciseId).filter(Boolean));
      for (const [identifier, entry] of Object.entries(records)) {
        if (!SESSION_ID.test(identifier) || !plain(entry) || entry.status !== 'planned') throw Error('invalid');
        if (own(state.schedule.sessions, identifier) || (entry.exerciseId ? exerciseIds.has(entry.exerciseId) || legacyNoteIds.has(entry.noteId) : entry.noteId && noteIds.has(entry.noteId))) throw Error('conflict');
        state.schedule.sessions[identifier] = entry;
        if (entry.noteId) noteIds.add(entry.noteId);
        if (entry.exerciseId) exerciseIds.add(entry.exerciseId);
        else if (entry.noteId) legacyNoteIds.add(entry.noteId);
      }
      return write(state);
    }
    function setMode(value, updatedAt) {
      const state = read();
      state.mode = {...state.mode, value, updatedAt};
      return write(state);
    }
    function updateWorkspace(records, action = '') {
      const state = read();
      state.workspace = overlay(state.workspace || {version: 1, records: {}, history: []}, {records});
      if (action) {
        const at = new Date().toISOString();
        const event = {at, action, id: at + '-' + Math.random().toString(36).slice(2)};
        state.workspace.history = [event, ...(state.workspace.history || [])].slice(0, 100);
      }
      return write(state);
    }
    function restoreMissing(backup) {
      read();
      if (!blocked || fault !== 'missing') throw Error('blocked');
      const restored = validate(backup);
      storage.setItem(KEY, encode(restored));
      blocked = false;
      dirty = {};
      memory = committed = restored;
      return read();
    }
    function applyImport(backup, policy) {
      const state = read();
      if (blocked) throw Error('blocked');
      if (rebaseFault) throw Error(rebaseFault);
      const imported = merge(state, backup, policy);
      checkpoint(state, true);
      let recoverySaved = false;
      try { storage.setItem(RECOVERY_KEY, encode(state)); recoverySaved = true; } catch {}
      const saved = write(imported.state);
      return {...imported, ...saved, recoverySaved, previous: state};
    }
    function serialize() {
      read();
      if (blocked) {
        if (unreadableRaw === null) throw Error('blocked');
        return unreadableRaw;
      }
      try { return encode({...memory, exportedAt: new Date().toISOString()}); }
      catch (error) {
        if (error.message !== 'limits') throw error;
        return encode(memory);
      }
    }
    function recovery() {
      try { const raw = storage.getItem(RECOVERY_KEY); return raw ? parse(raw) : null; }
      catch { return null; }
    }
    read();
    return {read, updateEntry, updateSession, addSessions, setMode, updateWorkspace, applyImport, serialize, recovery, snapshots, restoreMissing,
      health: () => ({unavailable, blocked, fault, snapshotFault})};
  }

  const api = {KEY, RECOVERY_KEY, SNAPSHOTS_KEY, FORMAT, MAX_BYTES, statuses, sessionStatuses, empty, validate, parse, encode, merge, createStore};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else scope.PrepProgressStore = api;
})(typeof window === 'undefined' ? globalThis : window);
