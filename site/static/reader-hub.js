/* Reader hub: reading history, saved notes, settings, and the Star notice.

   Everything is stored in the shared progress envelope (workspace records), so
   it travels with the existing JSON backup and never leaves the browser:
     reads-YYYYMMDD   {notes: {note-id: count}, updatedAt}   one record per local day
     reading-total    {total, since, updatedAt}               every counted read, all time
     reader-settings  {history, mascot, buddy, starAskedOn, starDone, snoozedUntil, updatedAt}
   A page counts as read once per load, after it has been visible for 15 seconds.
   Days older than 90 are emptied (not deleted), so merges stay timestamp-based;
   the all-time total keeps counting past that window.
   No network request except the site's own public note catalog. */
(function (scope) {
  'use strict';
  const DAY = 'reads-';
  const DAY_KEY = /^reads-\d{8}$/;
  const TOTAL = 'reading-total';
  const SETTINGS = 'reader-settings';
  const KEEP_DAYS = 90;
  const WEEKS = 13;
  const MAX_PER_DAY = 200;
  const MAX_TOTAL = 9999999;
  const NOTE = /^note-[a-f0-9]{16}$/;
  const MASCOT_STATES = ['open', 'collapsed', 'hidden'];

  const pad = value => String(value).padStart(2, '0');
  function dayKey(date) {
    return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}`;
  }
  const civil = date => `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  function fromKey(key) {
    const digits = key.slice(DAY.length);
    return new Date(Number(digits.slice(0, 4)), Number(digits.slice(4, 6)) - 1, Number(digits.slice(6, 8)));
  }
  function daysBetween(earlier, later) {
    return Math.round((new Date(later.getFullYear(), later.getMonth(), later.getDate()) -
      new Date(earlier.getFullYear(), earlier.getMonth(), earlier.getDate())) / 86400000);
  }
  const sum = notes => Object.values(notes || {}).reduce((total, count) => total + (Number.isInteger(count) && count > 0 ? count : 0), 0);

  function settings(records) {
    const record = records?.[SETTINGS] || {};
    return {
      history: record.history !== false,
      mascot: MASCOT_STATES.includes(record.mascot) ? record.mascot : 'collapsed',
      buddy: record.buddy === 'hidden' ? 'hidden' : 'shown',
      starAskedOn: typeof record.starAskedOn === 'string' ? record.starAskedOn : '',
      starDone: record.starDone === true,
      snoozedUntil: typeof record.snoozedUntil === 'string' ? record.snoozedUntil : ''
    };
  }

  /* All-time reads and the day counting started. Never less than the days still
     on record, so an import of older days cannot make the total look smaller. */
  function lifetime(records, today) {
    let kept = 0, earliest = '';
    for (const [key, record] of Object.entries(records || {})) {
      if (!DAY_KEY.test(key) || !sum(record?.notes)) continue;
      kept += sum(record.notes);
      const day = civil(fromKey(key));
      if (!earliest || day < earliest) earliest = day;
    }
    const record = records?.[TOTAL];
    const stored = Number.isSafeInteger(record?.total) && record.total > 0 ? record.total : 0;
    const since = typeof record?.since === 'string' && record.since && (!earliest || record.since <= earliest) ? record.since : (earliest || civil(today));
    return {total: Math.max(stored, kept), since};
  }

  /* Changes that add one read of noteId on date, and empty days past the window. */
  function increment(records, noteId, date, at) {
    if (!NOTE.test(noteId)) return {};
    const key = DAY + dayKey(date);
    const notes = Object.assign({}, records?.[key]?.notes || {});
    if (!(noteId in notes) && Object.keys(notes).length >= MAX_PER_DAY) return {};
    notes[noteId] = Math.min(9999, (Number.isInteger(notes[noteId]) ? notes[noteId] : 0) + 1);
    const changes = {[key]: {notes, updatedAt: at}};
    for (const [other, record] of Object.entries(records || {})) {
      if (!DAY_KEY.test(other) || other === key || !Object.keys(record?.notes || {}).length) continue;
      if (daysBetween(fromKey(other), date) >= KEEP_DAYS) changes[other] = {notes: {}, updatedAt: at};
    }
    const total = lifetime(records, date);
    changes[TOTAL] = {total: Math.min(MAX_TOTAL, total.total + 1), since: total.since, updatedAt: at};
    return changes;
  }

  /* Changes that empty every reading day and restart the all-time total. */
  function clear(records, at, date = new Date()) {
    const changes = Object.fromEntries(Object.keys(records || {})
      .filter(key => DAY_KEY.test(key) && Object.keys(records[key]?.notes || {}).length)
      .map(key => [key, {notes: {}, updatedAt: at}]));
    changes[TOTAL] = {total: 0, since: civil(date), updatedAt: at};
    return changes;
  }

  /* Today, this week (Monday first), the last 13 weeks, the streak, the all-time
     total and recently read notes, from the records alone. */
  function summarize(records, today) {
    const days = new Map();
    for (const [key, record] of Object.entries(records || {})) {
      if (!DAY_KEY.test(key)) continue;
      const age = daysBetween(fromKey(key), today);
      if (age < 0 || age >= KEEP_DAYS) continue;
      days.set(age, record?.notes || {});
    }
    const reads = age => (age >= 0 ? sum(days.get(age)) : 0);
    const dateAt = age => new Date(today.getFullYear(), today.getMonth(), today.getDate() - age);
    const weekday = (today.getDay() + 6) % 7;
    const cell = age => ({date: dateAt(age), total: reads(age), future: age < 0, old: age >= KEEP_DAYS});

    const weekNotes = new Set();
    for (let age = 0; age <= weekday; age++) {
      for (const [id, count] of Object.entries(days.get(age) || {})) if (count > 0) weekNotes.add(id);
    }
    const weekDays = Array.from({length: 7}, (_, index) => cell(weekday - index));
    let lastWeek = 0;
    for (let age = weekday + 1; age <= weekday + 7; age++) lastWeek += reads(age);

    const heat = Array.from({length: WEEKS}, (_, column) =>
      Array.from({length: 7}, (_, row) => cell(weekday - row + (WEEKS - 1 - column) * 7)));
    let streak = 0;
    for (let age = reads(0) ? 0 : 1; age < KEEP_DAYS && reads(age) > 0; age++) streak++;

    const todayNotes = Object.entries(days.get(0) || {}).filter(([, count]) => count > 0).sort((a, b) => b[1] - a[1]);
    const recent = new Map();
    [...days.entries()].sort((a, b) => a[0] - b[0]).forEach(([age, notes]) => {
      for (const [id, count] of Object.entries(notes)) {
        if (!count) continue;
        const item = recent.get(id) || {id, lastAge: age, total: 0};
        item.total += count;
        item.lastAge = Math.min(item.lastAge, age);
        recent.set(id, item);
      }
    });
    return {
      today: {total: sum(Object.fromEntries(todayNotes)), notes: todayNotes},
      week: {total: weekDays.reduce((total, day) => total + day.total, 0), notes: weekNotes.size, days: weekDays, last: lastWeek},
      heat,
      span: heat.flat().reduce((total, day) => total + day.total, 0),
      streak,
      lifetime: lifetime(records, today),
      recent: [...recent.values()].sort((a, b) => a.lastAge - b.lastAge || b.total - a.total)
    };
  }

  const api = {DAY, TOTAL, SETTINGS, KEEP_DAYS, WEEKS, dayKey, settings, lifetime, increment, clear, summarize};
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }
  scope.ReaderHub = api;

  /* ---------------------------------------------------------------- page UI */
  const progress = scope.PreparationProgress;
  if (!progress) return;
  const zh = (document.documentElement.lang || '').startsWith('zh');
  const say = (chinese, english) => (zh ? chinese : english);
  const lang = zh ? 'zh' : 'en';
  const base = new URL((scope.SITE && scope.SITE.prefix) || './', location.href);
  const repoLink = document.querySelector('.repo-btn');
  const repo = repoLink ? repoLink.href : 'https://github.com/tianyi-zhang-02/cooking-agi';
  const pageNote = document.querySelector('[data-prep-note]')?.dataset.prepNote || '';
  const reduce = () => scope.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const records = () => progress.getState()?.workspace?.records || {};
  const now = () => new Date().toISOString();
  const todayKey = () => civil(new Date());
  let catalog = null;
  let catalogLoading = null;

  function write(changes, action) {
    if (!Object.keys(changes).length && !action) return true;
    let persisted;
    try { persisted = progress.updateWorkspace(changes, action)?.persisted !== false; } catch { return false; }
    if (persisted) {
      for (const [key, record] of Object.entries(changes)) {
        if (/^bookmark-note-[a-f0-9]{16}$/.test(key) && typeof record.saved === 'boolean')
          scope.dispatchEvent(new CustomEvent('notes:bookmark-action', {detail: {noteId: key.slice(9), saved: record.saved}}));
      }
    }
    return persisted;
  }
  function saveSettings(patch) {
    const current = settings(records());
    const saved = write({[SETTINGS]: Object.assign({}, current, patch, {updatedAt: now()})});
    // other widgets on this page (the study buddy) follow these settings
    scope.dispatchEvent(new CustomEvent('notes:reader-settings'));
    return saved;
  }
  function loadCatalog() {
    if (catalog) return Promise.resolve(catalog);
    const shared = progress.getCatalog && progress.getCatalog();
    if (shared) { catalog = shared; return Promise.resolve(catalog); }
    if (!catalogLoading) {
      catalogLoading = fetch(new URL('static/prep-progress-catalog.json', base)).then(response => {
        if (!response.ok) throw Error('catalog');
        return response.json();
      }).then(value => { catalog = value; return value; }).catch(() => null);
    }
    return catalogLoading;
  }
  function pageOf(id) {
    const note = catalog?.notes?.[id];
    const page = note?.pages?.[lang] || note?.pages?.zh || note?.pages?.en;
    return page ? {title: page.title, href: new URL(page.url, base).href} : null;
  }
  const el = (tag, attributes = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes)) {
      if (key === 'text') node.textContent = value;
      else if (value !== false && value !== null && value !== undefined) node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of [].concat(children)) if (child) node.append(child);
    return node;
  };
  // Class toggles are visual only, so environments without classList simply skip them.
  const mark = (node, name, on = true) => { if (node && node.classList) node.classList.toggle(name, on); };
  const svg = (tag, attributes = {}) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  };

  /* Count this page once it has been visible for 15 seconds. The visible time
     keeps running afterwards so the drawer can show it live. */
  const COUNT_AFTER = 15000;
  const reading = {visible: 0, since: document.visibilityState === 'visible' ? Date.now() : 0, counted: false, failed: false};
  const visibleFor = () => reading.visible + (reading.since ? Date.now() - reading.since : 0);
  function tickReading() {
    if (!pageNote) return;
    if (document.visibilityState === 'visible') { if (!reading.since) reading.since = Date.now(); }
    else if (reading.since) { reading.visible += Date.now() - reading.since; reading.since = 0; }
    if (reading.counted || reading.failed || !settings(records()).history || visibleFor() < COUNT_AFTER) return;
    const changes = increment(records(), pageNote, new Date(), now());
    reading.counted = Object.keys(changes).length > 0 && write(changes);
    reading.failed = !reading.counted;
    render();
  }

  /* ------------------------------------------------------------- the drawer */
  const toggle = el('button', {type: 'button', class: 'icon-btn reader-hub-toggle', 'aria-label': say('我的阅读', 'My reading'),
    'aria-haspopup': 'dialog', 'aria-controls': 'reader-hub', 'aria-expanded': 'false'});
  const icon = svg('svg', {viewBox: '0 0 20 20', width: '18', height: '18', 'aria-hidden': 'true'});
  icon.append(svg('path', {d: 'M5.5 3.5h9v13l-4.5-3-4.5 3z', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.6', 'stroke-linejoin': 'round'}));
  toggle.append(icon, el('span', {class: 'reader-hub-badge', hidden: true, 'aria-hidden': 'true'}));
  const right = document.querySelector('.topbar-right');
  const themeButton = document.querySelector('.icon-btn.theme');
  if (right) right.insertBefore(toggle, themeButton || null);

  const tabs = [['reading', say('阅读记录', 'Reading')], ['saved', say('收藏', 'Saved')], ['settings', say('设置', 'Settings')]];
  const drawer = el('aside', {id: 'reader-hub', class: 'reader-hub', role: 'dialog', 'aria-labelledby': 'reader-hub-title', hidden: true});
  const close = el('button', {type: 'button', class: 'reader-hub-close', 'aria-label': say('关闭', 'Close'), text: '×'});
  const tabList = el('div', {class: 'reader-hub-tabs', role: 'tablist'});
  const panels = {};
  for (const [name, label] of tabs) {
    tabList.append(el('button', {type: 'button', role: 'tab', id: `reader-tab-${name}`, 'aria-controls': `reader-panel-${name}`,
      'aria-selected': name === 'reading' ? 'true' : 'false', 'data-tab': name, text: label}));
    panels[name] = el('section', {class: 'reader-hub-panel', role: 'tabpanel', id: `reader-panel-${name}`,
      'aria-labelledby': `reader-tab-${name}`, hidden: name !== 'reading'});
  }
  const status = el('p', {class: 'reader-hub-status', role: 'status', 'aria-live': 'polite'});
  drawer.append(el('header', {class: 'reader-hub-head'}, [el('h2', {id: 'reader-hub-title', text: say('我的', 'Mine')}), close]),
    tabList, ...Object.values(panels), status,
    el('p', {class: 'reader-hub-local', text: say('个人记录保存在这个浏览器里；换设备时可在学习台导出、导入。公开计数另行开启，只统计后续操作，不上传已有记录。',
      'Personal records stay in this browser; use the study workspace to export or import them. Public counting is a separate opt-in for subsequent activity, not an upload of existing records.')}));
  document.body.append(drawer);

  // One small tooltip for every chart mark in the drawer.
  const tip = el('div', {class: 'reader-hub-tip', role: 'tooltip', hidden: true});
  drawer.append(tip);
  drawer.addEventListener('pointerover', event => {
    const mark = event.target.closest?.('[data-tip]');
    if (!mark) { tip.hidden = true; return; }
    tip.textContent = mark.getAttribute('data-tip');
    tip.hidden = false;
    const box = mark.getBoundingClientRect(), frame = drawer.getBoundingClientRect();
    const x = Math.min(frame.width - 12 - tip.offsetWidth / 2, Math.max(12 + tip.offsetWidth / 2, box.left + box.width / 2 - frame.left));
    tip.style.left = `${x}px`;
    tip.style.top = `${box.top - frame.top + drawer.scrollTop - 8}px`;
  });
  drawer.addEventListener('pointerleave', () => { tip.hidden = true; });
  drawer.addEventListener('scroll', () => { tip.hidden = true; }, {passive: true});

  let active = 'reading';
  function select(name) {
    active = name;
    tabList.querySelectorAll('[role="tab"]').forEach(tab => {
      const on = tab.dataset.tab === name;
      tab.setAttribute('aria-selected', String(on));
      tab.tabIndex = on ? 0 : -1;
    });
    for (const [key, panel] of Object.entries(panels)) panel.hidden = key !== name;
    render();
  }
  tabList.addEventListener('click', event => { const tab = event.target.closest('[data-tab]'); if (tab) select(tab.dataset.tab); });
  tabList.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    const order = tabs.map(([name]) => name);
    const next = order[(order.indexOf(active) + (event.key === 'ArrowRight' ? 1 : order.length - 1)) % order.length];
    select(next);
    tabList.querySelector(`[data-tab="${next}"]`).focus();
  });
  function open(tab) {
    if (tab) select(tab);
    drawer.hidden = false;
    toggle.setAttribute('aria-expanded', 'true');
    loadCatalog().then(render);
    render();
    close.focus();
  }
  function shut() {
    drawer.hidden = true;
    toggle.setAttribute('aria-expanded', 'false');
    toggle.focus();
  }
  toggle.addEventListener('click', () => (drawer.hidden ? open() : shut()));
  close.addEventListener('click', shut);
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !drawer.hidden) shut(); });
  document.addEventListener('click', event => {
    // Use the path from dispatch time: a button that re-rendered itself is no longer inside the drawer.
    const path = event.composedPath();
    if (!drawer.hidden && !path.includes(drawer) && !path.includes(toggle) && !path.includes(mascot) && !(themeButton && path.includes(themeButton))) shut();
  });

  const dateLabel = date => (zh ? `${date.getMonth() + 1}月${date.getDate()}日` : date.toLocaleDateString('en', {month: 'short', day: 'numeric'}));
  const longDate = date => (date.getFullYear() === new Date().getFullYear() ? dateLabel(date)
    : zh ? `${date.getFullYear()}年${dateLabel(date)}` : date.toLocaleDateString('en', {year: 'numeric', month: 'short', day: 'numeric'}));
  const WEEKDAYS = zh ? ['一', '二', '三', '四', '五', '六', '日'] : ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
  const fullDay = date => (zh ? `${dateLabel(date)} 周${WEEKDAYS[(date.getDay() + 6) % 7]}` : date.toLocaleDateString('en', {weekday: 'short', month: 'short', day: 'numeric'}));
  const times = count => say(`${count} 次`, `${count} ${count === 1 ? 'read' : 'reads'}`);
  const number = value => value.toLocaleString(zh ? 'zh-CN' : 'en');
  function noteLink(id, extra) {
    const page = pageOf(id);
    const item = el('li');
    item.append(page ? el('a', {href: page.href, text: page.title}) : el('span', {class: 'reader-hub-unknown', text: say('（目录里没有这篇）', '(not in the current catalog)')}));
    if (extra) item.append(el('small', {text: extra}));
    return item;
  }

  /* Numbers count up from what was last shown, so a new read visibly lands. */
  const shown = {};
  function countTo(node, key, value) {
    const from = shown[key] ?? 0;
    shown[key] = value;
    if (from === value || reduce() || typeof scope.requestAnimationFrame !== 'function' || typeof performance === 'undefined') { node.textContent = number(value); return; }
    node.textContent = number(from);
    const start = performance.now();
    const step = time => {
      const progress = Math.min(1, (time - start) / 700);
      node.textContent = number(Math.round(from + (value - from) * (1 - Math.pow(1 - progress, 3))));
      if (progress < 1 && node.isConnected) scope.requestAnimationFrame(step);
    };
    scope.requestAnimationFrame(step);
  }

  function tile(key, label, value, note) {
    const strong = el('strong');
    countTo(strong, key, value);
    return el('div', {class: `reader-hub-tile is-${key}`}, [el('span', {text: label}), strong, el('small', {text: note})]);
  }

  const currentTitle = () => (pageOf(pageNote)?.title || document.querySelector('main h1, h1')?.textContent || document.title).trim();

  /* The live card on a note page: a ring that fills over the first 15 seconds,
     then this note's count for today and the time spent on this visit. */
  const live = {};
  function liveCard() {
    const ring = svg('svg', {class: 'reader-hub-ring', viewBox: '0 0 36 36', 'aria-hidden': 'true'});
    live.arc = svg('circle', {class: 'arc', cx: '18', cy: '18', r: '15', pathLength: '100', 'stroke-dasharray': '100', 'stroke-dashoffset': '100'});
    ring.append(svg('circle', {class: 'track', cx: '18', cy: '18', r: '15'}), live.arc, svg('path', {class: 'check', d: 'M12.5 18.5l3.8 3.8 7.4-8'}));
    live.line = el('span', {class: 'reader-hub-live-line'});
    live.clock = el('span', {class: 'reader-hub-live-clock', 'aria-label': say('这次在这页停留的时间', 'Time on this page this visit')});
    live.card = el('div', {class: 'reader-hub-live'}, [ring,
      el('div', {class: 'reader-hub-live-text'}, [el('small', {text: say('正在读', 'Reading now')}), el('strong', {text: currentTitle()}), live.line]),
      live.clock]);
    return live.card;
  }
  function updateLive() {
    if (!live.card?.isConnected) return;
    const spent = visibleFor();
    const fill = reading.counted ? 1 : Math.min(1, spent / COUNT_AFTER);
    live.arc.setAttribute('stroke-dashoffset', String(100 - fill * 100));
    live.arc.style.opacity = fill > .03 ? '1' : '0';
    mark(live.card, 'is-counted', reading.counted);
    const seconds = Math.floor(spent / 1000);
    live.clock.textContent = `${Math.floor(seconds / 60)}:${pad(seconds % 60)}`;
    if (reading.failed) {
      live.line.textContent = say('本次记录未保存，请检查浏览器存储。', 'This visit was not saved; please check browser storage.');
    } else if (!reading.counted) {
      const left = Math.max(1, Math.ceil((COUNT_AFTER - spent) / 1000));
      live.line.textContent = say(`再读 ${left} 秒记一次`, `Counts in ${left}s`);
    } else {
      const count = summarize(records(), new Date()).today.notes.find(([id]) => id === pageNote)?.[1] || 0;
      live.line.textContent = count ? say(`已记一次 · 这篇今天读了 ${count} 次`, `Counted · ${count}× today`) : say('这次打开没有计入', 'Not counted this visit');
    }
  }

  /* This week, Monday to Sunday: one bar per day, today in the accent. */
  function weekChart(week, fresh) {
    const width = 336, height = 112, base = 86, slot = width / 7, bar = 22;
    const average = week.last / 7;
    const peak = Math.max(1, average, ...week.days.map(day => day.total));
    const chart = svg('svg', {class: 'reader-hub-week', viewBox: `0 0 ${width} ${height}`, role: 'img',
      'aria-label': say(`本周每天的阅读次数，共 ${week.total} 次；上周日均 ${average.toFixed(1)} 次`,
        `Reads each day this week: ${week.total} in total; last week averaged ${average.toFixed(1)} a day`)});
    chart.append(svg('line', {class: 'baseline', x1: '0', x2: String(width), y1: String(base + .5), y2: String(base + .5)}));
    week.days.forEach((day, index) => {
      const x = index * slot + (slot - bar) / 2;
      const today = !day.future && index === week.days.findLastIndex(item => !item.future);
      const group = svg('g', {class: ['day', today ? 'is-today' : '', day.future ? 'is-future' : '', day.total ? 'has-reads' : '', today && fresh ? 'is-new' : ''].join(' ').trim(),
        'data-tip': day.future ? fullDay(day.date) : `${fullDay(day.date)} · ${times(day.total)}`});
      group.append(svg('rect', {class: 'hit', x: String(index * slot), y: '0', width: String(slot), height: String(height)}));
      if (day.total) {
        const h = Math.max(6, (base - 18) * day.total / peak), r = 4, top = base - h;
        group.append(svg('path', {class: 'bar', d: `M${x} ${base}V${top + r}q0 -${r} ${r} -${r}h${bar - 2 * r}q${r} 0 ${r} ${r}V${base}z`}));
        const value = svg('text', {class: 'value', x: String(x + bar / 2), y: String(top - 5), 'text-anchor': 'middle'});
        value.textContent = String(day.total);
        group.append(value);
      } else {
        group.append(svg('rect', {class: 'stub', x: String(x), y: String(base - 2), width: String(bar), height: '2', rx: '1'}));
      }
      const label = svg('text', {class: 'label', x: String(x + bar / 2), y: String(height - 6), 'text-anchor': 'middle'});
      label.textContent = WEEKDAYS[index];
      group.append(label);
      chart.append(group);
    });
    if (average > 0) {
      const y = String(base - (base - 18) * average / peak);
      chart.append(svg('line', {class: 'average', x1: '0', x2: String(width), y1: y, y2: y}));
    }
    return chart;
  }

  /* The last 13 weeks as a contribution-style grid: columns are weeks, rows
     Monday to Sunday, darker means more reads. */
  function heatmap(summary, fresh) {
    const cell = 18, gap = 4, left = 22, top = 16, step = cell + gap;
    const width = left + summary.heat.length * step - gap, height = top + 7 * step - gap;
    const peak = Math.max(1, ...summary.heat.flat().map(day => day.total));
    const grid = svg('svg', {class: 'reader-hub-heat', viewBox: `0 0 ${width} ${height}`, role: 'img',
      'aria-label': say(`近 ${WEEKS} 周的阅读，共 ${summary.span} 次`, `Reading over the last ${WEEKS} weeks: ${summary.span} reads`)});
    [0, 2, 4].forEach(row => {
      const label = svg('text', {class: 'axis', x: '0', y: String(top + row * step + cell - 5)});
      label.textContent = zh ? WEEKDAYS[row] : ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][row];
      grid.append(label);
    });
    let lastLabel = -3;
    summary.heat.forEach((column, index) => {
      const monday = column[0].date;
      const previous = summary.heat[index - 1]?.[0].date;
      if ((!previous || previous.getMonth() !== monday.getMonth()) && index - lastLabel >= 3 && index <= summary.heat.length - 2) {
        const month = svg('text', {class: 'axis', x: String(left + index * step), y: '10'});
        month.textContent = zh ? `${monday.getMonth() + 1}月` : monday.toLocaleDateString('en', {month: 'short'});
        grid.append(month);
        lastLabel = index;
      }
      column.forEach((day, row) => {
        if (day.future) return;
        const level = day.total ? Math.min(4, Math.ceil(4 * day.total / peak)) : 0;
        const today = index === summary.heat.length - 1 && row === column.findLastIndex(item => !item.future);
        const square = svg('rect', {class: `level-${level}${day.old ? ' is-old' : ''}${today ? ' is-today' : ''}${today && fresh ? ' is-new' : ''}`,
          x: String(left + index * step), y: String(top + row * step), width: String(cell), height: String(cell), rx: '4',
          'data-tip': `${fullDay(day.date)} · ${day.old ? say('超出 90 天，已清理', 'older than 90 days') : times(day.total)}`});
        grid.append(square);
      });
    });
    return grid;
  }

  let lastToday = null;
  function renderReading(panel, state) {
    panel.replaceChildren();
    const config = settings(state);
    if (!config.history) {
      const enable = el('button', {type: 'button', class: 'reader-hub-button', text: say('打开阅读记录', 'Turn on reading history')});
      enable.addEventListener('click', () => { saveSettings({history: true}); render(); });
      panel.append(el('p', {class: 'reader-hub-empty', text: say('阅读记录已关闭，读过的笔记不会被计数。', 'Reading history is off, so pages are not counted.')}), enable);
      return;
    }
    const summary = summarize(state, new Date());
    const fresh = lastToday !== null && summary.today.total > lastToday;
    lastToday = summary.today.total;
    const since = new Date(...summary.lifetime.since.split('-').map((part, index) => Number(part) - (index === 1 ? 1 : 0)));
    if (pageNote) panel.append(liveCard());
    panel.append(el('div', {class: 'reader-hub-tiles'}, [
      tile('today', say('今天', 'Today'), summary.today.total, say(`${summary.today.notes.length} 篇`, `${summary.today.notes.length} ${summary.today.notes.length === 1 ? 'note' : 'notes'}`)),
      tile('week', say('本周', 'This week'), summary.week.total, say(`上周 ${summary.week.last} 次`, `${summary.week.last} last week`)),
      tile('total', say('累计', 'All time'), summary.lifetime.total, say(`自 ${longDate(since)}`, `since ${longDate(since)}`))
    ]));

    const section = (title, aside, body) => el('section', {class: 'reader-hub-card'}, [
      el('header', {}, [el('h3', {text: title}), aside ? el('span', {text: aside}) : null]), ...[].concat(body)]);
    const average = summary.week.last / 7;
    const key = average > 0 ? el('span', {class: 'reader-hub-key'}, [el('i', {'aria-hidden': 'true'}),
      document.createTextNode(say(`上周日均 ${average.toFixed(1)}`, `last week ${average.toFixed(1)}/day`))]) : null;
    panel.append(el('section', {class: 'reader-hub-card'}, [
      el('header', {}, [el('h3', {text: say('本周每天', 'Each day this week')}), key]), weekChart(summary.week, fresh)]));
    const legend = el('div', {class: 'reader-hub-legend', 'aria-hidden': 'true'}, [el('span', {text: say('少', 'Less')}),
      ...[0, 1, 2, 3, 4].map(level => el('i', {class: `level-${level}`})), el('span', {text: say('多', 'More')})]);
    panel.append(section(say(`近 ${WEEKS} 周`, `Last ${WEEKS} weeks`),
      [say(`${summary.span} 次`, `${summary.span} reads`), summary.streak > 1 ? say(`连续 ${summary.streak} 天`, `${summary.streak}-day streak`) : ''].filter(Boolean).join(' · '),
      [heatmap(summary, fresh), legend]));

    if (summary.today.notes.length) {
      panel.append(el('h3', {text: say('今天读过', 'Read today')}),
        el('ul', {class: 'reader-hub-list'}, summary.today.notes.slice(0, 12).map(([id, count]) => noteLink(id, say(`${count} 次`, `${count}×`)))));
    }
    const earlier = summary.recent.filter(item => item.lastAge > 0).slice(0, 10);
    if (earlier.length) {
      panel.append(el('h3', {text: say('最近读过', 'Recently read')}),
        el('ul', {class: 'reader-hub-list'}, earlier.map(item => noteLink(item.id,
          say(`${item.lastAge === 1 ? '昨天' : item.lastAge + ' 天前'} · 共 ${item.total} 次`, `${item.lastAge === 1 ? 'yesterday' : item.lastAge + ' days ago'} · ${item.total} total`)))));
    }
    if (!summary.today.notes.length && !earlier.length) {
      panel.append(el('p', {class: 'reader-hub-empty', text: say('还没有阅读记录。打开一篇笔记读上 15 秒，这里就会记一次。', 'No reading yet. Spend 15 seconds on a note and it will show up here.')}));
    }
    updateLive();
  }

  const BOOKMARK = 'M5.5 3.5h9v13l-4.5-3-4.5 3z';
  function bookmarkIcon() {
    const icon = svg('svg', {viewBox: '0 0 20 20', width: '16', height: '16', 'aria-hidden': 'true'});
    icon.append(svg('path', {d: BOOKMARK}));
    return icon;
  }
  const group = (title, rows, aside) => [
    el('div', {class: 'reader-hub-group-head'}, [el('h3', {text: title}), aside ? el('span', {text: aside}) : null]),
    el('div', {class: 'reader-hub-group'}, rows)];

  function renderSaved(panel, state) {
    panel.replaceChildren();
    const setSaved = (id, value) => {
      const ok = write({['bookmark-' + id]: {saved: value, updatedAt: now()}});
      status.textContent = ok ? (value ? say('已收藏。', 'Saved.') : say('已取消收藏。', 'Removed from saved notes.')) : say('没能保存，浏览器可能禁止了本地存储。', 'Could not save; browser storage may be blocked.');
      scope.dispatchEvent(new Event('notes:prep-storage'));
      render();
    };
    if (pageNote) {
      const saved = Boolean(state['bookmark-' + pageNote]?.saved);
      const button = el('button', {type: 'button', class: 'reader-hub-save' + (saved ? ' is-on' : ''), 'aria-pressed': String(saved)},
        [bookmarkIcon(), el('span', {text: saved ? say('已收藏', 'Saved') : say('收藏', 'Save')})]);
      button.addEventListener('click', () => setSaved(pageNote, !saved));
      panel.append(el('div', {class: 'reader-hub-current'}, [
        el('div', {class: 'reader-hub-live-text'}, [el('small', {text: say('当前这篇', 'This page')}), el('strong', {text: currentTitle()})]), button]));
    }
    const saved = Object.entries(state).filter(([key, record]) => key.startsWith('bookmark-') && record?.saved)
      .sort((a, b) => String(b[1].updatedAt).localeCompare(String(a[1].updatedAt)));
    if (!saved.length) {
      panel.append(el('p', {class: 'reader-hub-empty', text: say('还没有收藏。在笔记页点「收藏」，或用侧边栏的收藏按钮。', 'Nothing saved yet. Use “Save” on a note page or the sidebar.')}));
      return;
    }
    panel.append(...group(say('收藏的笔记', 'Saved notes'), [el('ul', {class: 'reader-hub-list'}, saved.slice(0, 100).map(([key, record]) => {
      const id = key.slice('bookmark-'.length);
      const when = Date.parse(record.updatedAt);
      const item = noteLink(id, !Number.isFinite(when) ? '' : dayKey(new Date(when)) === dayKey(new Date()) ? say('今天', 'today') : longDate(new Date(when)));
      const remove = el('button', {type: 'button', class: 'reader-hub-remove', 'aria-label': say('取消收藏', 'Remove'), text: '×'});
      remove.addEventListener('click', () => setSaved(id, false));
      item.append(remove);
      return item;
    }))], String(saved.length)));
  }

  function renderSettings(panel, state) {
    panel.replaceChildren();
    const config = settings(state);
    const row = (label, hint, control, tag = 'div', attributes = {}) => el(tag, Object.assign({class: 'reader-hub-setting'}, attributes),
      [el('div', {class: 'reader-hub-setting-text'}, [el('strong', {text: label}), hint ? el('small', {text: hint}) : null]), control]);
    const history = el('button', {type: 'button', role: 'switch', class: 'reader-hub-switch', 'aria-checked': String(config.history), 'aria-label': say('记录阅读次数', 'Count my reading')});
    history.addEventListener('click', () => { saveSettings({history: !config.history}); render(); });
    const segment = (label, options, current, choose) => {
      const box = el('div', {class: 'reader-hub-segment', role: 'radiogroup', 'aria-label': label});
      for (const [value, text] of options) {
        const option = el('button', {type: 'button', role: 'radio', 'aria-checked': String(value === current), text});
        option.addEventListener('click', () => choose(value));
        box.append(option);
      }
      return box;
    };
    const theme = segment(say('主题', 'Theme'), [['dark', say('深色', 'Dark')], ['light', say('浅色', 'Light')]],
      document.documentElement.dataset.theme === 'light' ? 'light' : 'dark',
      value => { if ((value === 'light') !== (document.documentElement.dataset.theme === 'light')) themeButton?.click(); render(); });
    const star = segment(say('Star 提示', 'Star notice'), [['collapsed', say('显示', 'Show')], ['hidden', say('隐藏', 'Hide')]],
      config.mascot === 'hidden' ? 'hidden' : 'collapsed',
      value => {
        saveSettings({mascot: value, snoozedUntil: '', ...(value === 'collapsed' ? {starDone: false} : {})});
        renderMascot({show: value === 'collapsed'});
        render();
      });
    const buddy = segment(say('学习小伙伴', 'Study buddy'), [['shown', say('显示', 'Show')], ['hidden', say('隐藏', 'Hide')]],
      config.buddy, value => { saveSettings({buddy: value}); render(); });
    const chevron = el('span', {class: 'reader-hub-chevron', 'aria-hidden': 'true', text: '›'});
    const backup = row(say('导出 / 导入备份', 'Export or import a backup'), say('在学习台里，换设备时用。', 'In the study workspace, for moving devices.'), chevron, 'a',
      {href: new URL(say('interview/dashboard.html#progress', 'interview/dashboard.en.html#progress'), base).href});
    mark(backup, 'is-link');
    let armed = false;
    const wipeLabel = el('strong', {text: say('清空阅读记录', 'Clear reading history')});
    const wipe = el('button', {type: 'button', class: 'reader-hub-setting is-danger reader-hub-danger'}, [el('div', {class: 'reader-hub-setting-text'}, [wipeLabel,
      el('small', {text: say('包括累计次数；收藏和学习进度不受影响。', 'Includes the all-time total; saved notes and study progress stay.')})])]);
    wipe.addEventListener('click', () => {
      if (!armed) { armed = true; wipeLabel.textContent = say('再点一次确认清空', 'Select again to confirm'); mark(wipe, 'is-armed'); return; }
      const saved = write(clear(records(), now()), say('清空阅读记录', 'Cleared reading history'));
      status.textContent = saved
        ? say('阅读记录和累计次数已清空，收藏和学习进度没有变。', 'Reading history cleared, including the all-time total. Saved notes and study progress are unchanged.')
        : say('没能保存清空操作。浏览器存储可能不可用或已满，请检查后重试。', 'Could not save the cleared history. Browser storage may be unavailable or full; please check and try again.');
      render();
    });
    panel.append(
      ...group(say('阅读', 'Reading'), [row(say('记录阅读次数', 'Count my reading'), say('一篇笔记在前台读满 15 秒记一次，只存在这个浏览器。', 'A note counts after 15 visible seconds; kept in this browser only.'), history)]),
      ...group(say('外观与提示', 'Look and notices'), [
        row(say('主题', 'Theme'), say('和顶栏的切换按钮是同一个设置。', 'Same as the toggle in the top bar.'), theme),
        row(say('Star 提示', 'Star notice'), say('偶尔在右下角提醒一次。', 'An occasional notice in the corner.'), star),
        row(say('学习小伙伴', 'Study buddy'), say('右下角的小窗，跟着时间和节日变化。', 'The little window in the corner that follows the time of day.'), buddy)]),
      ...group(say('数据', 'Data'), [backup, wipe]));
  }

  let renderedDay = dayKey(new Date());
  let badgeCount = null;
  function render() {
    const state = records();
    const summary = summarize(state, new Date());
    renderedDay = dayKey(new Date());
    const badge = toggle.querySelector('.reader-hub-badge');
    badge.hidden = !(settings(state).history && summary.today.total);
    badge.textContent = summary.today.total > 99 ? '99+' : String(summary.today.total);
    if (badgeCount !== null && summary.today.total > badgeCount && !reduce()) {
      mark(badge, 'is-bump', false);
      void badge.offsetWidth;
      mark(badge, 'is-bump');
    }
    badgeCount = summary.today.total;
    if (!drawer.hidden) {
      if (active === 'reading') renderReading(panels.reading, state);
      if (active === 'saved') renderSaved(panels.saved, state);
      if (active === 'settings') renderSettings(panels.settings, state);
    }
  }

  // Once a second: advance this page's visible time, refresh the live card,
  // and redraw after midnight so "today" and "this week" roll over.
  if (scope.setInterval) scope.setInterval(() => {
    tickReading();
    if (dayKey(new Date()) !== renderedDay) render();
    else if (!drawer.hidden) updateLive();
  }, 1000);
  document.addEventListener('visibilitychange', () => { tickReading(); if (document.visibilityState === 'visible') render(); });

  /* ------------------------------------------------------- asking for a star
     The way most project sites do it: a quiet "Star" pill next to the GitHub
     icon in the top bar, plus at most one small notice a day that the reader
     can snooze or switch off. Hovering the pill lets a few tiny hearts rise. */
  const snoozed = config => Boolean(config.snoozedUntil) && Date.parse(config.snoozedUntil) > Date.now();
  const STAR_ICON = 'M8 1.6l1.9 3.9 4.3.6-3.1 3 .7 4.3L8 11.4l-3.8 2 .7-4.3-3.1-3 4.3-.6z';
  const HEART = 'M12 21s-7.2-4.6-9.6-8.8C.7 9 2.6 4.8 6.7 4.8c2.2 0 3.6 1.2 5.3 3.1 1.7-1.9 3.1-3.1 5.3-3.1 4.1 0 6 4.2 4.3 7.4C19.2 16.4 12 21 12 21z';
  let pill = null;
  if (repoLink) {
    pill = repoLink;
    mark(pill, 'repo-pill');
    pill.setAttribute('aria-label', say('在 GitHub 上给这个项目点 Star', 'Star this project on GitHub'));
    const star = svg('svg', {class: 'repo-pill-star', viewBox: '0 0 16 16', width: '14', height: '14', 'aria-hidden': 'true'});
    star.append(svg('path', {d: STAR_ICON}));
    const hearts = el('span', {class: 'repo-pill-hearts', 'aria-hidden': 'true'});
    pill.append(el('span', {class: 'repo-pill-divider', 'aria-hidden': 'true'}), star, el('span', {class: 'repo-pill-label', text: 'Star'}), hearts);
    let last = 0;
    const rise = () => {
      if (reduce() || Date.now() - last < 900) return;
      last = Date.now();
      for (let i = 0; i < 3; i++) {
        const heart = el('span', {class: 'repo-pill-heart'});
        heart.style.setProperty('--drift', `${Math.round((Math.random() - .5) * 18)}px`);
        heart.style.animationDelay = `${i * 110}ms`;
        const shape = svg('svg', {viewBox: '0 0 24 24'});
        shape.append(svg('path', {d: HEART}));
        heart.append(shape);
        heart.addEventListener('animationend', () => heart.remove());
        hearts.append(heart);
      }
    };
    pill.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') rise(); });
    pill.addEventListener('click', () => saveSettings({starDone: true, starAskedOn: todayKey()}));
  }

  const mascot = el('aside', {class: 'star-toast', role: 'dialog', 'aria-labelledby': 'star-toast-text', hidden: true});
  const github = svg('svg', {class: 'star-toast-mark', viewBox: '0 0 16 16', width: '20', height: '20', 'aria-hidden': 'true'});
  github.append(svg('path', {d: 'M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.4 7.4 0 012-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z'}));
  const message = el('p', {class: 'star-toast-text', id: 'star-toast-text', text: say('喜欢这些笔记的话，可以去 GitHub 点个 Star。', 'If you like these notes, you can star the repo on GitHub.')});
  const actions = el('div', {class: 'star-toast-actions'});
  const toastClose = el('button', {type: 'button', class: 'star-toast-close', 'aria-label': say('这次先不显示', 'Not now'), text: '×'});
  mascot.append(github, el('div', {class: 'star-toast-body'}, [message, actions]), toastClose);
  document.body.append(mascot);
  function hideToast() { mascot.hidden = true; }
  const later = () => { saveSettings({snoozedUntil: new Date(Date.now() + 12 * 3600 * 1000).toISOString(), starAskedOn: todayKey()}); hideToast(); };
  const cta = el('a', {class: 'star-toast-cta', href: repo, target: '_blank', rel: 'noopener'});
  const ctaStar = svg('svg', {viewBox: '0 0 16 16', width: '13', height: '13', 'aria-hidden': 'true'});
  ctaStar.append(svg('path', {d: STAR_ICON}));
  cta.append(ctaStar, document.createTextNode(say('去点 Star', 'Star on GitHub')));
  cta.addEventListener('click', () => { saveSettings({starDone: true, starAskedOn: todayKey()}); hideToast(); });
  const notNow = el('button', {type: 'button', class: 'star-toast-choice', text: say('这次先不显示', 'Not now')});
  notNow.addEventListener('click', later);
  const never = el('button', {type: 'button', class: 'star-toast-choice star-toast-quiet', text: say('不再显示', 'Don’t show again')});
  never.addEventListener('click', () => { saveSettings({mascot: 'hidden'}); hideToast(); });
  actions.append(cta, notNow, never);
  toastClose.addEventListener('click', later);
  function renderMascot({show = false} = {}) {
    const config = settings(records());
    if (config.mascot === 'hidden' || config.starDone || snoozed(config)) hideToast();
    else if (show) {
      saveSettings({starAskedOn: todayKey()});
      mascot.hidden = false;
    }
  }
  // At most once a day, after the reader has been on the page for a while.
  const initial = settings(records());
  if (initial.mascot !== 'hidden' && !initial.starDone && !snoozed(initial) && initial.starAskedOn !== todayKey()) {
    scope.setTimeout(() => {
      const now = settings(records());
      if (now.mascot === 'hidden' || now.starDone || snoozed(now) || document.visibilityState !== 'visible') return;
      saveSettings({starAskedOn: todayKey()});
      mascot.hidden = false;
    }, 20000);
  }

  scope.addEventListener('notes:prep-storage', render);
  scope.addEventListener('notes:prep-restored', render);
  render();
})(typeof window === 'undefined' ? globalThis : window);
