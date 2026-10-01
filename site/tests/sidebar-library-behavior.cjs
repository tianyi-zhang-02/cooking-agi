const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/sidebar.js'), 'utf8');

function mount(blocked = false) {
  function element() {
    return {hidden:true, textContent:'', dataset:{}, listeners:{}, attributes:{}, children:[],
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; },
      replaceChildren() { this.children = []; }, append(child) { this.children.push(child); },
      click() { return this.listeners.click && this.listeners.click(); }, remove() {}, focus() {}};
  }
  const selectors = Object.fromEntries(['.side-tools','#side-reading','.side-scroll','[data-side-save]',
    '.side-status','.side-local','[data-reading-export]','[data-reading-import]','[data-reading-file]',
    '[data-side-saved]','[data-side-recent]','[data-side-clear]'].map(selector => [selector, element()]));
  const active = Object.assign(element(), {dataset:{note:'learn/one.md'}, title:'First note',
    getAttribute:() => '../learn/one.html', getBoundingClientRect: () => ({top:0,bottom:100}),
    closest: selector => selector === '[data-grp]' ? {querySelector:() => ({textContent:'Basics'})} : {dataset:{category:'learn'}}});
  const directory = {querySelectorAll: selector => selector === 'a[data-note]' ? [active] : [], querySelector: () => active};
  const inlineSave = element(), inlineStatus = element(), noteTools = element();
  const side = {querySelector: selector => selector === '#side-directory' ? directory : selectors[selector], querySelectorAll:() => []};
  const storage = new Map(), windowEvents = {}, blobs = [], downloads = [];
  selectors['.side-scroll'].getBoundingClientRect = () => ({top:0,bottom:200,height:200});
  const document = {documentElement:{lang:'en'}, body: {append: child => downloads.push(child)},
    querySelector: selector => selector === '#side' ? side : selector === '.note-tool-status' ? inlineStatus : noteTools,
    querySelectorAll: () => [selectors['[data-side-save]'], inlineSave], createElement:element};
  vm.runInNewContext(source, {document, localStorage:{
    getItem: key => { if (blocked) throw Error('blocked'); return storage.get(key) || null; },
    setItem: (key, value) => { if (blocked) throw Error('blocked'); storage.set(key,value); }},
    window:{addEventListener:(name, callback) => { windowEvents[name] = callback; }, dispatchEvent:event => { windowEvents.dispatched = event.type; }},
    requestAnimationFrame:callback => callback(), Event:class {constructor(type) {this.type=type;}},
    Blob:class {constructor(parts) {blobs.push(parts.join(''));}}, URL:{createObjectURL:() => 'blob:backup', revokeObjectURL() {}},
    setTimeout: callback => callback()});
  return {selectors, inlineSave, inlineStatus, storage, blobs, downloads, windowEvents,
    async import(text, size = text.length) {
      selectors['[data-reading-file]'].files = [{size, text:async () => text}];
      await selectors['[data-reading-file]'].listeners.change();
    }};
}

(async function () {
  const page = mount();
  assert.equal(page.inlineSave.textContent, '+ Save this page');
  page.inlineSave.click();
  assert.equal(page.inlineSave.attributes['aria-pressed'], 'true');
  assert.equal(page.selectors['[data-side-save]'].attributes['aria-pressed'], 'true');
  assert.match(page.inlineStatus.textContent, /Note saved/);
  page.selectors['[data-reading-export]'].click();
  assert.deepEqual(JSON.parse(page.blobs[0]), {format:'cooking-agi-bookmarks',version:1,saved:['learn/one.md']});
  assert.equal(page.downloads[0].download, 'agi-notes-bookmarks.json');
  page.selectors['[data-side-save]'].click();
  assert.equal(page.inlineSave.attributes['aria-pressed'], 'false');
  await page.import(page.blobs[0]);
  assert.equal(page.inlineSave.attributes['aria-pressed'], 'true');
  await page.import(page.blobs[0]);
  assert.match(page.inlineStatus.textContent, /Added 0/);
  const before = page.storage.get('cooking-agi:reading:v1');
  await page.import('not json');
  assert.equal(page.storage.get('cooking-agi:reading:v1'), before);
  assert.match(page.inlineStatus.textContent, /Existing bookmarks are unchanged/);
  await page.import('{}', 131073);
  assert.equal(page.storage.get('cooking-agi:reading:v1'), before);
  page.selectors['[data-side-clear]'].click();
  const cleared = JSON.parse(page.storage.get('cooking-agi:reading:v1'));
  assert.deepEqual(cleared, {saved:['learn/one.md'],recent:[]});
  assert.equal(page.storage.get('cooking-agi:positions:v1'), '[]');
  assert.equal(page.windowEvents.dispatched, 'notes:reading-cleared');
  const blocked = mount(true);
  blocked.inlineSave.click();
  assert.match(blocked.inlineStatus.textContent, /storage is unavailable/);
  blocked.selectors['[data-reading-export]'].click();
  assert.deepEqual(JSON.parse(blocked.blobs[0]).saved, ['learn/one.md']);
  console.log('Bookmarks: both buttons, safe import/merge, export, limits, clearing, storage failure OK');
})().catch(error => { console.error(error); process.exitCode = 1; });
