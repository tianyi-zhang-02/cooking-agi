const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/next-stop.js'), 'utf8');

function mount(language = 'en', empty = false, reduced = false) {
  function element(dataset = {}) {
    return {dataset, hidden: false, listeners: {}, attributes: {}, style: {}, children: [], textContent: '',
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; },
      toggleAttribute(name) { if (name in this.attributes) { delete this.attributes[name]; return false; } this.attributes[name] = ''; return true; },
      focus() { this.focused = true; },
      click() { this.listeners.click(); }
    };
  }
  const records = empty ? [] : [
    {company: 'alpha', role: 'ml', employment: 'full-time', year: '2027'},
    {company: 'beta', role: 'software', employment: 'internship', year: '2027'},
    {company: 'beta', role: 'ml', employment: 'full-time', year: '2026'},
  ].map(element);
  const years = ['2027', '2026'].map(year => Object.assign(element(), {
    querySelectorAll: () => records.filter(record => record.dataset.year === year)
  }));
  const companies = empty ? [] : ['alpha', 'beta'].map(name => {
    const count = element(), unit = element(), bar = element();
    return Object.assign(element({stopCompany: name, name}), {
      count, unit, bar, querySelector: selector => selector === '[data-stop-count]' ? count : selector === '[data-stop-unit]' ? unit : bar
    });
  });
  const filters = ['role', 'employment', 'year'].map(stopFilter => Object.assign(element({stopFilter}), {value: 'all'}));
  const views = ['timeline', 'companies'].map(stopView => element({stopView}));
  const links = records.map(() => element());
  const search = Object.assign(element(), {value: ''});
  const presets = ['NVIDIA 英伟达', 'ServiceNow Service Now', 'Hudson River Trading HRT'].map(search => element({search}));
  const group = Object.assign(element(), {querySelectorAll: () => presets});
  const logo = Object.assign(element(), {complete: true, naturalWidth: 0});
  const selectors = Object.fromEntries(['controls', 'status', 'clear', 'timeline', 'companies', 'no-match', 'motion'].map(name => [`[data-stop-${name}]`, element()]));
  ['search-wrap', 'status', 'no-match'].forEach(name => { selectors[`[data-company-${name}]`] = element(); });
  selectors['[data-company-search]'] = search;
  const companyList = selectors['[data-stop-companies]'];
  companyList.children = companies.slice();
  companyList.appendChild = child => { companyList.children = companyList.children.filter(node => node !== child).concat(child); };
  const preference = Object.assign(element(), {matches: reduced});
  const root = Object.assign(element({lang: language}), {
    querySelector: selector => selectors[selector],
    querySelectorAll: selector => ({'[data-stop-record]': records, '[data-stop-filter]': filters, '[data-stop-view]': views, '[data-stop-year]': years, '[data-stop-open-record]': links, '[data-company-preset]': presets, '[data-company-group]': [group], '.company-mark img': [logo]}[selector] || [])
  });
  vm.runInNewContext(source, {document: {querySelector: () => root}, window: {matchMedia: () => preference}});
  function filter(field, value) {
    const input = filters.find(candidate => candidate.dataset.stopFilter === field);
    input.value = value;
    input.listeners.change();
  }
  return {root, records, companies, companyList, years, filters, views, links, selectors, preference, filter, search, presets, group, logo};
}

const page = mount();
assert.equal(page.logo.hidden, true);
assert.equal(page.selectors['[data-company-search-wrap]'].hidden, false);
page.search.value = 'service now';
page.search.listeners.input();
assert.deepEqual(page.presets.map(preset => preset.hidden), [true, false, true]);
page.search.value = '英伟达';
page.search.listeners.input();
assert.deepEqual(page.presets.map(preset => preset.hidden), [false, true, true]);
page.search.value = 'not listed';
page.search.listeners.input();
assert.equal(page.group.hidden, true);
assert.equal(page.selectors['[data-company-no-match]'].hidden, false);
page.search.value = '';
page.search.listeners.input();
assert.equal(page.presets.every(preset => !preset.hidden), true);
assert.equal(page.group.hidden, false);
assert.match(page.selectors['[data-stop-status]'].textContent, /^3 published updates/);
assert.match(page.selectors['[data-stop-status]'].textContent, /newest start dates first/);
page.views[1].click();
assert.equal(page.views[1].attributes['aria-pressed'], 'true');
assert.equal(page.selectors['[data-stop-timeline]'].hidden, true);
assert.equal(page.companyList.hidden, false);
assert.equal(page.companyList.children[0].dataset.stopCompany, 'beta');
page.filter('role', 'ml');
assert.equal(page.companyList.children[0].dataset.stopCompany, 'alpha');
assert.equal(page.companies[1].count.textContent, '1');
assert.equal(page.companies[1].unit.textContent, 'update');
page.filter('year', '2027');
assert.equal(page.companies[1].hidden, true);
assert.equal(page.years[1].hidden, true);
assert.match(page.selectors['[data-stop-status]'].textContent, /^1 published update ·/);
page.filter('employment', 'internship');
assert.equal(page.selectors['[data-stop-no-match]'].hidden, false);
assert.equal(page.companies[0].hidden, true);
page.selectors['[data-stop-clear]'].click();
assert.equal(page.filters[0].focused, true);
assert.equal(page.selectors['[data-stop-no-match]'].hidden, true);
assert.equal(page.selectors['[data-stop-clear]'].hidden, true);
assert.equal(page.companies[1].count.textContent, '2');
assert.equal(page.companies[1].unit.textContent, 'updates');
page.views[0].click();
assert.equal(page.companyList.hidden, true);
assert.equal(page.records.every(record => !record.hidden), true);
page.views[1].click();
page.filter('role', 'product');
page.links[0].click();
assert.equal(page.companyList.hidden, true);
assert.equal(page.selectors['[data-stop-timeline]'].hidden, false);
assert.equal(page.records.every(record => !record.hidden), true);
assert.equal(page.selectors['[data-stop-no-match]'].hidden, true);
const chinese = mount('zh', true, true);
assert.match(chinese.selectors['[data-stop-status]'].textContent, /^0 条公开分享/);
assert.match(chinese.selectors['[data-stop-status]'].textContent, /按开始时间由近到远/);
assert.equal(chinese.selectors['[data-stop-no-match]'].hidden, true);
chinese.views[1].click();
assert.match(chinese.selectors['[data-stop-status]'].textContent, /公司分享数量/);
console.log('Next-stop filters, counts, view switching, and empty states: OK');
