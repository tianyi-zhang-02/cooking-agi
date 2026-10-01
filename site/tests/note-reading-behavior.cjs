const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = path.join(__dirname, '../static/note-reading.js');
const source = fs.readFileSync(script, 'utf8');
const {cleanPositions, rememberPosition} = require(script);
const entry = {note: 'core/attention.md', lang: 'zh', anchor: 'attention-matrix'};
assert.deepEqual(cleanPositions(null), []);
assert.deepEqual(cleanPositions([null, {...entry, anchor: 'javascript:alert(1)'}, entry, entry]), [entry]);
assert.equal(cleanPositions(Array.from({length:120}, (_, index) => ({...entry, note:`core/note-${index}.md`}))).length, 100);
assert.deepEqual(rememberPosition([entry], {...entry, anchor:'next'}), [{...entry, anchor:'next'}]);
assert.equal(rememberPosition([entry], {...entry, lang:'en'}).length, 2);

function mount(saved, blocked = false) {
  const events = {}, documentEvents = {}, timers = new Map(), writes = [];
  let value = JSON.stringify(saved), nextTimer = 0;
  const resume = {hidden: true, listeners: {}, removeAttribute(name) { delete this[name]; },
    addEventListener(name, callback) { this.listeners[name] = callback; }};
  const headings = [
    {id: 'intro', textContent: '开始', top: 700},
    {id: 'attention-matrix', textContent: 'Attention Matrix <test>', top: 1400}
  ].map(heading => Object.assign(heading, {getClientRects: () => [1], getBoundingClientRect: () => ({top: heading.top})}));
  const tools = {dataset:{readingNote:entry.note}, querySelector: () => resume};
  const document = {documentElement:{lang:'zh-Hans'}, hidden:false,
    querySelector: selector => selector === '[data-reading-note]' ? tools : {querySelectorAll: () => headings},
    addEventListener: (name, callback) => { documentEvents[name] = callback; }};
  const storage = {getItem: () => { if (blocked) throw Error('blocked'); return value; },
    setItem: (key, data) => { if (blocked) throw Error('blocked'); value = data; writes.push(JSON.parse(data)); }};
  vm.runInNewContext(source, {document, window:{addEventListener:(name, callback) => { events[name] = callback; }},
    localStorage:storage, setTimeout: callback => { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearTimeout: id => timers.delete(id)});
  function scroll() { events.scroll(); Array.from(timers.values()).forEach(callback => callback()); timers.clear(); }
  return {resume, headings, writes, events, scroll, document, documentEvents, setStorage(data) { value = JSON.stringify(data); }};
}
const fresh = mount([]);
assert.equal(fresh.resume.hidden, true);
assert.equal(fresh.writes.length, 0);
fresh.headings[0].top = -500;
fresh.headings[1].top = 110;
fresh.scroll();
assert.equal(fresh.writes[0][0].anchor, 'attention-matrix');
fresh.scroll();
assert.equal(fresh.writes.length, 1);
const reopened = mount(fresh.writes[0]);
assert.equal(reopened.resume.hidden, false);
assert.equal(reopened.resume.href, '#attention-matrix');
assert.equal(reopened.resume.textContent, '继续上次位置：Attention Matrix <test>');
assert.equal(reopened.writes.length, 0);
assert.equal(reopened.headings[0].top, 700);
reopened.resume.listeners.click();
assert.equal(reopened.resume.hidden, true);
assert.equal(mount([{...entry, lang:'en'}]).resume.hidden, true);
assert.equal(mount([{...entry, anchor:'removed-section'}]).resume.hidden, true);
const cleared = mount([entry]);
cleared.setStorage([]);
cleared.events['notes:reading-cleared']();
assert.equal(cleared.resume.hidden, true);
cleared.events.pagehide();
assert.equal(cleared.writes.length, 0);
const crossTab = mount([entry]);
crossTab.setStorage([]);
crossTab.events.storage({key:'cooking-agi:positions:v1'});
assert.equal(crossTab.resume.hidden, true);
const blocked = mount([], true);
blocked.headings[0].top = 0;
assert.doesNotThrow(() => blocked.scroll());
assert.equal(blocked.writes.length, 0);
const closing = mount([]);
closing.headings[0].top = 20;
closing.events.scroll();
closing.events.pagehide();
assert.equal(closing.writes[0][0].anchor, 'intro');
console.log('Reading positions: persistence, explicit resume, language isolation, clearing, safe fallback OK');
