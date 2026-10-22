const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../static/mobile-reading.js'), 'utf8');
const app = fs.readFileSync(require.resolve('../static/app.js'), 'utf8');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.events = {};
    this.className = '';
    this.hidden = false;
    this.nodeType = 1;
    this.isConnected = true;
    this.style = {setProperty: (key, value) => { this.style[key] = value; }};
    this.classList = {
      contains: name => this.className.split(' ').includes(name),
      add: name => { if (!this.classList.contains(name)) this.className += ' ' + name; },
      remove: name => { this.className = this.className.split(' ').filter(value => value !== name).join(' '); },
      toggle: (name, force) => {
        const enabled = force === undefined ? !this.classList.contains(name) : force;
        this.classList[enabled ? 'add' : 'remove'](name);
        return enabled;
      }
    };
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key] ?? null; }
  removeAttribute(key) { delete this.attributes[key]; }
  append(...children) { children.forEach(child => { child.parentElement = this; this.children.push(child); }); }
  after(child) { child.parentElement = this.parentElement; this.parentElement.children.splice(this.parentElement.children.indexOf(this) + 1, 0, child); }
  addEventListener(name, callback) { this.events[name] = callback; }
  matches(selectors) { return selectors.split(',').some(selector => selector.startsWith('.') ? this.classList.contains(selector.slice(1)) : selector === this.tagName); }
  querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector); }
  focus() { this.focused = true; }
  scrollTo(...values) { this.scrolled = values; }
  getBoundingClientRect() { return {left: 0, top: 0, right: 390, bottom: 700}; }
}

function mount(language = 'en', options = {}) {
  const root = new Element('html');
  root.lang = language;
  const body = new Element('body');
  const article = new Element('div');
  article.className = 'article-body';
  body.append(article);
  const region = new Element();
  region.className = 'table-wrap';
  region.clientWidth = 288;
  region.scrollWidth = 600;
  if (options.label) region.setAttribute('aria-label', options.label);
  if (options.tabindex) region.setAttribute('tabindex', options.tabindex);
  const table = new Element('table');
  table.rows = [{cells: [{colSpan: 2}, {colSpan: 1}]}];
  region.append(table);
  region.firstElementChild = table;
  article.append(region);
  const image = new Element('img');
  image.src = '/assets/architecture.svg';
  image.alt = 'Architecture <not markup>';
  image.naturalWidth = options.small ? 64 : 900;
  image.complete = !options.lazy;
  article.append(image);
  const events = {};
  let mutation;
  let scans = 0;
  const query = article.querySelectorAll.bind(article);
  article.querySelectorAll = selector => { scans++; return query(selector); };
  const document = {
    documentElement: root, body,
    querySelector: selector => selector === '.article-body' ? article : null,
    createElement: tag => {
      const element = new Element(tag);
      if (tag === 'dialog' && !options.noDialog) {
        element.showModal = () => { element.open = true; };
        element.close = () => { element.open = false; element.events.close(); };
      }
      return element;
    }
  };
  class Mutation {
    constructor(callback) { mutation = callback; }
    observe() {}
  }
  vm.runInNewContext(source, {document, window: {addEventListener: (name, callback) => { events[name] = callback; }}, MutationObserver: Mutation});
  return {root, body, article, region, image, events, mutation, scans: () => scans};
}

test('wide tables retain their columns and get localized, keyboard-accessible scrolling', () => {
  for (const language of ['zh-Hans', 'en']) {
    const page = mount(language);
    assert.equal(page.region.style['--reading-table-width'], '432px');
    assert.equal(page.region.getAttribute('tabindex'), '0');
    assert.equal(page.region.getAttribute('aria-label'), language === 'en' ? 'Scroll sideways to see the rest' : '左右滑动查看完整内容');
    assert.equal(page.article.querySelector('.reading-scroll-hint').hidden, false);
    page.region.clientWidth = 900;
    page.events.resize();
    assert.equal(page.article.querySelector('.reading-scroll-hint').hidden, true);
    assert.equal(page.region.getAttribute('tabindex'), null);
    assert.equal(page.region.getAttribute('aria-label'), null);
  }
});

test('existing labels and tab order are preserved after resize', () => {
  const page = mount('en', {label: 'Results table', tabindex: '-1'});
  page.region.clientWidth = 900;
  page.events.resize();
  assert.equal(page.region.getAttribute('aria-label'), 'Results table');
  assert.equal(page.region.getAttribute('tabindex'), '-1');
});

test('figure viewer keeps the source and description, supports fitting and returns focus', () => {
  const page = mount();
  const button = page.article.querySelector('.figure-expand');
  assert.equal(button.textContent, 'Enlarge figure');
  button.events.click();
  const viewer = page.body.querySelector('dialog');
  assert.equal(viewer.open, true);
  assert.equal(page.root.classList.contains('figure-is-open'), true);
  assert.equal(viewer.querySelector('img').src, page.image.src);
  assert.equal(viewer.querySelector('img').alt, page.image.alt);
  assert.equal(viewer.querySelector('.figure-viewer-credit').textContent, page.image.alt);
  assert.equal(viewer.querySelector('img').style['--figure-width'], '900px');
  const buttons = viewer.querySelectorAll('button');
  buttons[0].events.click();
  assert.equal(viewer.classList.contains('is-fit'), true);
  assert.equal(buttons[0].textContent, 'Original size');
  buttons[1].events.click();
  assert.equal(viewer.open, false);
  assert.equal(page.root.classList.contains('figure-is-open'), false);
  assert.equal(button.focused, true);
});

test('lazy images are enhanced once; icons and unsupported dialogs are left alone', () => {
  const page = mount('zh-Hans', {lazy: true});
  assert.equal(page.article.querySelector('.figure-expand'), null);
  page.image.events.load();
  assert.equal(page.article.querySelector('.figure-expand').textContent, '放大查看');
  page.mutation([{addedNodes: [page.image]}]);
  assert.equal(page.article.querySelectorAll('.figure-expand').length, 1);
  assert.equal(mount('en', {small: true}).article.querySelector('.figure-expand'), null);
  assert.equal(mount('en', {noDialog: true}).article.querySelector('.figure-expand'), null);
});

test('animated SVG updates do not repeatedly rescan the article', () => {
  const page = mount();
  const scans = page.scans();
  page.mutation([{addedNodes: [new Element('text'), new Element('rect')]}]);
  assert.equal(page.scans(), scans);
});

test('mobile drawer locks the background and closes on section links', () => {
  const start = app.indexOf('  var side = $("#side")');
  const end = app.indexOf('  /* ---------------------------------------------------------- code copy */', start);
  assert.ok(start > 0 && end > start);
  for (const language of ['zh', 'en']) {
    const root = new Element('html');
    const side = new Element('aside');
    const menu = new Element('button');
    const scrim = new Element();
    side.contains = () => false;
    const media = {matches: true, addEventListener: (name, callback) => { media.change = callback; }};
    vm.runInNewContext(app.slice(start, end), {
      root, LANG: language, document: {activeElement: null, addEventListener() {}},
      matchMedia: () => media, $: selector => ({'#side': side, '.side-scrim': scrim, '.icon-btn.menu': menu})[selector], $$: () => []
    });
    assert.equal(side.inert, true);
    menu.events.click();
    assert.equal(root.classList.contains('drawer-is-open'), true);
    assert.equal(menu.getAttribute('aria-label'), language === 'en' ? 'Close navigation' : '关闭目录');
    side.events.click({target: {closest: () => true}});
    assert.equal(root.classList.contains('drawer-is-open'), false);
    assert.equal(side.inert, true);
    assert.equal(scrim.hidden, true);
    menu.events.click();
    media.matches = false;
    media.change();
    assert.equal(root.classList.contains('drawer-is-open'), false);
    assert.equal(side.inert, false);
  }
});
