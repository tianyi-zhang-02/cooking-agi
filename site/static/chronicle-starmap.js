/* Chronicle star map: the whole chronicle as one rotating galaxy, the other view beside the timeline.

   Every milestone is a star. Fields are the arms, coloured as on the timeline; distance from the
   centre is order in time, spread by rank so each ring holds about as many milestones as the next and
   the crowded recent years get room. The documented connections are the lines between stars. Faint
   rings mark the start of four periods. Drag to turn it; hover a star to light up its connections and
   its name; click (or tap) to open the milestone, the same panel the timeline opens. The field filter
   applies here too.

   Everything is read from the page the server already rendered (the milestone library), so there is
   nothing to fetch. With reduced motion the map stays still until dragged. */
(function () {
  'use strict';
  const root = document.querySelector('[data-chronicle]');
  const box = root && root.querySelector('[data-starmap]');
  const views = root && root.querySelector('[data-views]');
  if (!root || !box || !views) return;
  const canvas = box.querySelector('canvas'), periods = root.querySelector('#chronicle-content');
  const zh = (document.documentElement.lang || '').startsWith('zh');
  const reduce = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  const dark = () => document.documentElement.dataset.theme !== 'light';
  const TAU = Math.PI * 2;

  /* ------------------------------------------------ the milestones, from the rendered library */
  const fields = [...root.querySelectorAll('.ch-field[data-field]')].map((chip) => chip.dataset.field);
  const stars = [...root.querySelectorAll('.chronicle-work')].map((article) => ({
    id: article.id.replace(/^event-/, ''),
    field: article.dataset.workTopic, layer: article.dataset.layer, era: article.dataset.era,
    year: Number(article.querySelector('.ch-work-meta time')?.textContent) || 0,
    name: article.querySelector('.ch-name')?.textContent || '',
    links: [...article.querySelectorAll('.ch-links [data-node-link]')].map((a) => a.dataset.nodeLink),
  }));
  const byId = new Map(stars.map((s) => [s.id, s]));
  const pairs = [];
  for (const s of stars) for (const other of s.links) if (byId.has(other) && s.id < other) pairs.push([s, byId.get(other)]);
  const degree = new Map(stars.map((s) => [s.id, new Set(s.links).size]));
  const featured = new Set([...root.querySelectorAll('.ch-spine li[data-key] [data-node]')].map((a) => a.dataset.node));

  // place each star: its field's arm, at a radius set by its rank in time
  let seed = 11;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const years = stars.map((s) => s.year).sort((a, b) => a - b);
  const reach = (year) => { const below = years.filter((v) => v < year).length, same = years.filter((v) => v === year).length; return .14 + .86 * (below + same / 2) / years.length; };
  for (const s of stars) {
    const arm = Math.max(0, fields.indexOf(s.field)), r = reach(s.year) + (rnd() - .5) * .025;
    const a = arm / Math.max(1, fields.length) * TAU + r * 4.2 + (rnd() - .5) * .16;
    Object.assign(s, {x: Math.cos(a) * r, y: (rnd() - .5) * .14 * (1.2 - r), z: Math.sin(a) * r});
  }
  const dust = Array.from({length: 1200}, () => {
    const r = Math.pow(rnd(), .7), arm = Math.floor(rnd() * Math.max(1, fields.length));
    const a = arm / Math.max(1, fields.length) * TAU + r * 4.2 + (rnd() - .5) * .35;
    return {x: Math.cos(a) * r * 1.05, y: (rnd() - .5) * .12, z: Math.sin(a) * r * 1.05, b: rnd()};
  });
  const rings = [1956, 1986, 2012, 2024].filter((y) => years[0] < y && y <= years[years.length - 1]).map((y) => [y, reach(y)]);

  /* ------------------------------------------------ drawing */
  let yaw = .2, tilt = .95, hover = null, drag = null, raf = 0, last = 0, shown = false, colours = {};
  const rgb = (value) => { const hex = (value || '').trim().replace('#', ''); return hex.length === 6 ? [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) : [201, 133, 0]; };
  function readColours() {
    const style = getComputedStyle(root);
    colours = Object.fromEntries(['physical', 'software', 'knowledge', 'people'].map((k) => [k, rgb(style.getPropertyValue(`--ch-${k}`))]));
  }
  function size() {
    const rect = canvas.getBoundingClientRect(), ratio = Math.min(2, devicePixelRatio || 1);
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  }
  function project(p) {
    const w = canvas.width, h = canvas.height, scale = Math.min(w * .46, h * .62);
    const x1 = p.x * Math.cos(yaw) + p.z * Math.sin(yaw), z1 = -p.x * Math.sin(yaw) + p.z * Math.cos(yaw);
    const y2 = p.y * Math.cos(tilt) - z1 * Math.sin(tilt), z2 = p.y * Math.sin(tilt) + z1 * Math.cos(tilt), k = 3 / (3 + z2);
    return [w / 2 + x1 * k * scale, h * .52 - y2 * k * scale, (1 - z2) / 2, k];
  }
  function draw() {
    const g = canvas.getContext('2d'), w = canvas.width, h = canvas.height, unit = Math.max(1, w / 1200), night = dark();
    if (!w) return;
    g.clearRect(0, 0, w, h);
    const field = root.querySelector('.ch-field[data-field][aria-pressed="true"]')?.dataset.field || null;
    const ink = night ? [214, 204, 191] : [70, 60, 48], white = night ? [255, 246, 226] : [60, 40, 20];
    const core = g.createRadialGradient(w / 2, h * .52, 0, w / 2, h * .52, Math.min(w, h) * .55);
    core.addColorStop(0, `rgba(233,162,59,${night ? .09 : .07})`); core.addColorStop(1, 'rgba(233,162,59,0)');
    g.fillStyle = core; g.fillRect(0, 0, w, h);
    if (night) {
      g.globalCompositeOperation = 'lighter';
      for (const d of dust) { const [x, y, near] = project(d); g.fillStyle = `rgba(255,240,215,${.04 + .1 * d.b * near})`; g.fillRect(x, y, 1.6 * unit, 1.6 * unit); }
    }
    g.globalCompositeOperation = 'source-over';
    g.setLineDash([3 * unit, 9 * unit]); g.lineWidth = 1.2 * unit;
    g.font = `600 ${Math.round(12 * unit)}px ui-monospace, Menlo, monospace`;
    for (const [year, r] of rings) {
      g.strokeStyle = `rgba(${ink},.14)`; g.beginPath();
      for (let i = 0; i <= 96; i += 1) { const a = i / 96 * TAU, [x, y] = project({x: Math.cos(a) * r, y: 0, z: Math.sin(a) * r}); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
      g.stroke();
      const [lx, ly] = project({x: Math.cos(-.9) * r, y: 0, z: Math.sin(-.9) * r});
      g.fillStyle = `rgba(${ink},.5)`; g.fillText(String(year), lx + 6 * unit, ly);
    }
    g.setLineDash([]);
    const lit = hover ? new Set([hover.id, ...hover.links]) : null;
    for (const [a, b] of pairs) {
      const [x1, y1, n1] = project(a), [x2, y2, n2] = project(b), hot = hover && (a === hover || b === hover);
      const dim = field && a.field !== field && b.field !== field;
      g.strokeStyle = hot ? `rgba(${white},.8)` : `rgba(${ink},${(dim ? .025 : .06) + (dim ? 0 : .1) * (n1 + n2) / 2})`;
      g.lineWidth = (hot ? 1.8 : 1) * unit;
      g.beginPath(); g.moveTo(x1, y1); g.quadraticCurveTo((x1 + x2) / 2, (y1 + y2) / 2 - 24 * unit, x2, y2); g.stroke();
    }
    const order = stars.map((s) => [s, project(s)]).sort((a, b) => a[1][2] - b[1][2]);
    if (night) g.globalCompositeOperation = 'lighter';
    for (const [s, [x, y, near, k]] of order) {
      const dim = (field && s.field !== field) || (lit && !lit.has(s.id));
      const base = colours[s.layer] || [201, 133, 0], c = night ? base.map((v) => Math.round(v + (255 - v) * .45)) : base;
      const r = (1.6 + Math.min(4, degree.get(s.id)) * .55 + (featured.has(s.id) ? 1.2 : 0)) * k * unit * 1.3;
      const alpha = dim ? .18 : .55 + .45 * near;
      const halo = g.createRadialGradient(x, y, 0, x, y, r * 5);
      halo.addColorStop(0, `rgba(${c},${alpha * (night ? .32 : .18)})`); halo.addColorStop(1, `rgba(${c},0)`);
      g.fillStyle = halo; g.beginPath(); g.arc(x, y, r * 5, 0, TAU); g.fill();
      g.fillStyle = `rgba(${c},${alpha})`; g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill();
      if ((featured.has(s.id) && !dim) || s === hover) {
        g.strokeStyle = `rgba(${white},${(s === hover ? .8 : .4) * near})`; g.lineWidth = unit;
        g.beginPath(); g.moveTo(x - r * 4, y); g.lineTo(x + r * 4, y); g.moveTo(x, y - r * 4); g.lineTo(x, y + r * 4); g.stroke();
      }
      s.screen = [x, y, r];
    }
    g.globalCompositeOperation = 'source-over';
    // names: the one under the pointer and its links first, then the period highlights, never overlapping
    const taken = [], labelled = [...stars].filter((s) => s === hover || (lit && lit.has(s.id)) || (!lit && featured.has(s.id) && (!field || s.field === field)))
      .sort((a, b) => (b === hover) - (a === hover) || degree.get(b.id) - degree.get(a.id));
    for (const s of labelled) {
      const [x, y] = s.screen, main = s === hover, text = main ? `${s.year} · ${s.name}` : s.name;
      g.font = `${main ? 650 : 500} ${Math.round((main ? 14 : 12) * unit)}px -apple-system, "PingFang SC", sans-serif`;
      const width = g.measureText(text).width, rect = [x + 9 * unit, y - 20 * unit, x + 11 * unit + width, y - 2 * unit];
      if (!main && taken.some((t) => rect[0] < t[2] && rect[2] > t[0] && rect[1] < t[3] && rect[3] > t[1])) continue;
      taken.push(rect);
      g.fillStyle = main ? `rgba(${white},.95)` : `rgba(${ink},.72)`; g.fillText(text, rect[0], y - 6 * unit);
    }
  }
  function frame(now) {
    const dt = Math.min(.05, (now - (last || now)) / 1000); last = now;
    if (!drag && !hover && !reduce()) yaw += dt * .05;
    draw();
    raf = shown && !reduce() ? requestAnimationFrame(frame) : 0;
    if (!raf) last = 0;
  }
  const wake = () => { if (!raf && shown) raf = requestAnimationFrame(frame); if (reduce()) draw(); };

  /* ------------------------------------------------ pointer: turn, hover, open */
  function starAt(event) {
    const rect = canvas.getBoundingClientRect(), ratio = canvas.width / rect.width;
    const x = (event.clientX - rect.left) * ratio, y = (event.clientY - rect.top) * ratio;
    let best = null, bestDistance = 14 * ratio;
    for (const s of stars) { if (!s.screen) continue; const d = Math.hypot(s.screen[0] - x, s.screen[1] - y); if (d < bestDistance) { best = s; bestDistance = d; } }
    return best;
  }
  canvas.addEventListener('pointerdown', (event) => { drag = {x: event.clientX, y: event.clientY, moved: 0}; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointermove', (event) => {
    if (drag) {
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      drag.x = event.clientX; drag.y = event.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
      yaw += dx * .006; tilt = Math.max(.25, Math.min(1.45, tilt + dy * .004));
    } else {
      const next = starAt(event);
      if (next !== hover) { hover = next; canvas.style.cursor = hover ? 'pointer' : 'grab'; }
    }
    wake();
  });
  canvas.addEventListener('pointerleave', () => { if (!drag) { hover = null; wake(); } });
  const release = (event) => {
    if (!drag) return;
    const moved = drag.moved; drag = null;
    if (moved < 5 && event.type === 'pointerup') {
      const s = starAt(event);
      if (s) location.hash = `event-${s.id}`;  // the timeline's own route opens the milestone panel
    }
    wake();
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  /* ------------------------------------------------ the two views */
  const buttons = [...views.querySelectorAll('[data-view]')];
  function show(view) {
    shown = view === 'map';
    buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === view)));
    box.hidden = !shown;
    if (periods) periods.hidden = shown;
    root.classList.toggle('is-starmap', shown);
    if (shown) { readColours(); size(); draw(); wake(); }
  }
  buttons.forEach((b) => { b.disabled = false; b.addEventListener('click', () => show(b.dataset.view)); });
  views.hidden = false;
  root.addEventListener('click', (event) => { if (event.target.closest('.ch-field')) setTimeout(draw, 0); });
  addEventListener('resize', () => { if (shown) { size(); draw(); } });
  new MutationObserver(() => { if (shown) { readColours(); draw(); } }).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
})();
