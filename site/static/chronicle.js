/* Technology chronicle: a tree that grows behind the page on entry and then
   steps back, a vertical timeline that opens one period at a time, field
   filters, a reading panel, and links that survive switching language.
   Without this script the timeline simply shows everything. No storage, no
   network. */
(() => {
  const root = document.querySelector('[data-chronicle]');
  if (!root) return;
  const zh = (document.documentElement.lang || '').startsWith('zh');
  const reduce = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $$ = selector => [...root.querySelectorAll(selector)];
  const eras = $$('.ch-era');
  const items = $$('.ch-spine > li');
  const eraItems = new Map(eras.map(era => [era.dataset.era, [...era.querySelectorAll('.ch-spine > li')]]));
  const chips = $$('[data-field]');
  const reset = root.querySelector('[data-filter-reset]');
  const empty = root.querySelector('[data-filter-empty]');
  const articles = new Map($$('.chronicle-work').map(article => [article.id.replace(/^event-/, ''), article]));
  const topics = new Set(chips.map(chip => chip.dataset.field));
  const languageLinks = [...document.querySelectorAll('[data-language-switch]')];
  const opened = new Set();
  let field = null;
  let current = null;

  root.classList.add('is-enhanced');
  chips.forEach(chip => { chip.disabled = false; });
  if (reset) reset.disabled = false;

  /* ------------------------------------------------------------ address */
  function syncLanguageLinks() {
    languageLinks.forEach(link => {
      const target = new URL(link.href, location.href);
      target.search = location.search;
      if (field) target.searchParams.set('field', field); else target.searchParams.delete('field');
      target.hash = location.hash;
      link.href = target.href;
    });
  }
  function writeAddress(hash) {
    const url = new URL(location.href);
    if (field) url.searchParams.set('field', field); else url.searchParams.delete('field');
    url.hash = hash;
    history.replaceState(history.state, '', url);
    syncLanguageLinks();
  }

  /* ------------------------------------------------------------ the timeline */
  const linkOf = item => (item.querySelector ? item.querySelector('[data-node]') : null);
  function refresh() {
    eras.forEach(era => {
      const name = era.dataset.era;
      const list = eraItems.get(name) || [];
      const expanded = opened.has(name) || Boolean(field);
      let shown = 0, folded = 0;
      list.forEach(item => {
        const match = !field || item.dataset.workTopic === field;
        const visible = match && (expanded || 'key' in item.dataset);
        item.hidden = !visible;
        if (visible) shown++; else if (match) folded++;
      });
      era.hidden = Boolean(field) && shown === 0;
      const more = era.querySelector('[data-more]');
      if (more) {
        more.hidden = Boolean(field) || (!folded && !opened.has(name));
        more.textContent = opened.has(name) ? more.dataset.lessLabel : more.dataset.moreLabel;
        more.setAttribute('aria-expanded', String(opened.has(name)));
      }
    });
    if (empty) empty.hidden = !field || items.some(item => !item.hidden);
  }
  function setField(name, remember = true) {
    if (remember && current) closePanel({write: false, refocus: false});
    field = topics.has(name) ? name : null;
    chips.forEach(chip => chip.setAttribute('aria-pressed', String(chip.dataset.field === field)));
    if (reset) reset.setAttribute('aria-pressed', String(!field));
    refresh();
    if (remember) writeAddress(current ? `#event-${current}` : ''); else syncLanguageLinks();
  }
  chips.forEach(chip => chip.addEventListener('click', () => setField(chip.dataset.field === field ? null : chip.dataset.field)));
  if (reset) reset.addEventListener('click', () => setField(null));
  eras.forEach(era => {
    const more = era.querySelector('[data-more]');
    if (!more) return;
    more.addEventListener('click', () => {
      const name = era.dataset.era;
      if (opened.has(name)) {
        opened.delete(name);
        refresh();
        const box = era.getBoundingClientRect ? era.getBoundingClientRect() : null;
        if (box && box.top < 0) era.scrollIntoView({block: 'start'});
      } else {
        opened.add(name);
        refresh();
      }
    });
  });

  /* ------------------------------------------------------------ the reading panel */
  const panel = document.createElement('aside');
  panel.className = 'ch-panel';
  panel.hidden = true;
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', zh ? '节点详情' : 'Milestone details');
  const button = (name, label, glyph) => {
    const element = document.createElement('button');
    element.type = 'button';
    element.className = name;
    element.setAttribute('aria-label', label);
    element.textContent = glyph;
    return element;
  };
  const prev = button('ch-panel-step', zh ? '上一个' : 'Previous', '←');
  const next = button('ch-panel-step', zh ? '下一个' : 'Next', '→');
  const count = document.createElement('span');
  count.className = 'ch-panel-count';
  const close = button('ch-panel-close', zh ? '关闭' : 'Close', '×');
  const bar = document.createElement('div');
  bar.className = 'ch-panel-bar';
  bar.append(prev, next, count, close);
  const body = document.createElement('div');
  body.className = 'ch-panel-body';
  panel.append(bar, body);
  if (document.body) document.body.append(panel);
  let slot = null, opener = null;

  // Steps go through the whole period (folded milestones included), within the field filter.
  const order = era => (eraItems.get(era) || []).filter(item => !field || item.dataset.workTopic === field).map(linkOf).filter(Boolean).map(link => link.dataset.node);
  function openNode(id, {write = true, focus = true} = {}) {
    const article = articles.get(id);
    if (!article) return false;
    closePanel({write: false, refocus: false});
    if (field && article.dataset.workTopic !== field) setField(null, false);
    opened.add(article.dataset.era);
    refresh();
    if (article.replaceWith && document.createComment) {
      slot = document.createComment('milestone');
      article.replaceWith(slot);
      body.append(article);
    }
    panel.hidden = false;
    current = id;
    root.querySelectorAll('[data-node]').forEach(element => element.classList.toggle('is-current', element.dataset.node === id));
    const ids = order(article.dataset.era), index = ids.indexOf(id);
    prev.disabled = index <= 0;
    next.disabled = index < 0 || index >= ids.length - 1;
    count.textContent = index >= 0 ? `${index + 1} / ${ids.length}` : '';
    body.scrollTop = 0;
    if (write) writeAddress(`#event-${id}`);
    const title = article.querySelector && article.querySelector('[data-work-title]');
    if (focus && title) { title.setAttribute('tabindex', '-1'); title.focus({preventScroll: true}); }
    return true;
  }
  function closePanel({write = true, refocus = true} = {}) {
    if (!current) return;
    const article = articles.get(current);
    if (slot && article) slot.replaceWith(article);
    slot = null;
    panel.hidden = true;
    root.querySelectorAll('.is-current').forEach(element => element.classList.remove('is-current'));
    current = null;
    if (write) writeAddress('');
    if (refocus && opener && opener.focus) opener.focus();
  }
  const step = delta => {
    const article = articles.get(current);
    if (!article) return;
    const ids = order(article.dataset.era), index = ids.indexOf(current);
    if (index >= 0 && ids[index + delta]) openNode(ids[index + delta]);
  };
  prev.addEventListener('click', () => step(-1));
  next.addEventListener('click', () => step(1));
  close.addEventListener('click', () => closePanel());
  addEventListener('keydown', event => {
    if (!current) return;
    if (event.key === 'Escape') closePanel();
    if (!panel.contains(document.activeElement)) return;
    if (event.key === 'ArrowLeft') step(-1);
    if (event.key === 'ArrowRight') step(1);
  });
  if (document.addEventListener) document.addEventListener('click', event => {
    if (event.defaultPrevented || event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest && event.target.closest('a[href^="#event-"]');
    if (!link || !(root.contains(link) || panel.contains(link))) return;
    const id = link.getAttribute('href').slice(7);
    if (!articles.has(id)) return;
    event.preventDefault();
    if (!panel.contains(link)) opener = link;
    openNode(id);
  });

  /* ------------------------------------------------------------ routes, old links included */
  function route() {
    const params = new URLSearchParams(location.search);
    const legacy = params.get('milestone') ? `#event-${params.get('milestone')}` : params.get('era') ? `#era-${params.get('era')}` : '';
    if (legacy && !location.hash) {
      const url = new URL(location.href);
      url.searchParams.delete('milestone');
      url.searchParams.delete('era');
      const target = articles.get(legacy.slice(7));
      if (legacy.startsWith('#event-') && target && params.get('field') && target.dataset.workTopic !== params.get('field')) url.searchParams.delete('field');
      url.hash = legacy;
      history.replaceState(history.state, '', url);
    }
    setField(new URLSearchParams(location.search).get('field'), false);
    let hash;
    try { hash = decodeURIComponent(location.hash.slice(1)); } catch { hash = ''; }
    const legacyArticle = document.getElementById(hash)?.closest?.('.chronicle-work');
    if (hash.startsWith('field-')) {
      closePanel({write: false, refocus: false});
      setField(hash.slice(6), false);
    } else if (hash.startsWith('event-') && articles.has(hash.slice(6))) {
      openNode(hash.slice(6), {write: false, focus: false});
    } else if (legacyArticle && articles.has(legacyArticle.id.slice(6))) {
      openNode(legacyArticle.id.slice(6), {write: false, focus: false});
    } else {
      closePanel({write: false, refocus: false});
    }
    syncLanguageLinks();
  }
  addEventListener('hashchange', route);
  addEventListener('popstate', route);
  addEventListener('beforeprint', () => closePanel({write: false, refocus: false}));
  addEventListener('afterprint', route);
  refresh();
  route();

  /* ------------------------------------------------------------ the tree behind the page */
  function tree() {
    const stage = root.querySelector('[data-tree-stage]');
    const label = root.querySelector('[data-year]');
    if (!stage || !document.body || typeof window.innerWidth !== 'number') return;
    const canvas = document.createElement('canvas');
    if (!canvas.getContext) return;
    canvas.className = 'ch-tree';
    canvas.setAttribute('aria-hidden', 'true');
    document.body.prepend(canvas);
    const g = canvas.getContext('2d');
    if (!g) { canvas.remove(); return; }
    const first = Number(stage.dataset.first) || 1882, last = Number(stage.dataset.last) || 2025;
    let seed = 11, branches = [], nodes = [], W = 0, H = 0, dpr = 1, unit = 1, base = [0, 0];
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const qpoint = (b, u) => [(1 - u) * (1 - u) * b.x + 2 * (1 - u) * u * b.cx + u * u * b.x1, (1 - u) * (1 - u) * b.y + 2 * (1 - u) * u * b.cy + u * u * b.y1];

    // The opening space ends where the first screen ends, so the filter bar sits on the bottom edge
    // and the tree grows from just above it.
    function fitStage() {
      const bar = root.querySelector('.ch-bar');
      const top = stage.getBoundingClientRect().top + scrollY;
      stage.style.height = `${Math.max(220, innerHeight - top - (bar ? bar.offsetHeight : 64))}px`;
      return stage.getBoundingClientRect().bottom + scrollY;
    }
    function build() {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = innerWidth; H = innerHeight;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      const ground = Math.min(H * .98, fitStage() - 6);
      // Keep the crown below the title and intro: shorter trees, pushed right on wide screens.
      unit = Math.max(.35, Math.min((ground - (W >= 900 ? 190 : 230)) / 640, W / 800));
      base = [W >= 900 ? W * .7 : W * .5, ground];
      seed = 11; branches = []; nodes = [];
      const grow = (x, y, angle, len, width, depth, t0) => {
        const x1 = x + Math.cos(angle) * len, y1 = y + Math.sin(angle) * len, bend = (rnd() - .5) * .5;
        const cx = (x + x1) / 2 + Math.cos(angle + Math.PI / 2) * len * bend, cy = (y + y1) / 2 + Math.sin(angle + Math.PI / 2) * len * bend;
        branches.push({x, y, cx, cy, x1, y1, w: width, depth, t0, t1: t0 + len, len});
        if (!depth) return;
        const n = depth >= 5 ? 2 : rnd() < .55 ? 2 : 3, spread = depth >= 5 ? .5 : .62;
        for (let i = 0; i < n; i++) {
          const off = (i / (n - 1) - .5) * spread * 2, lean = (-Math.PI / 2 - angle) * .18;
          grow(x1, y1, angle + off + lean + (rnd() - .5) * .22, len * (.74 + rnd() * .1), width * .68, depth - 1, t0 + len);
        }
      };
      grow(base[0], base[1], -Math.PI / 2, 165 * unit, 24 * unit, 6, 0);
      const maxT = Math.max(...branches.map(b => b.t1));
      branches.forEach(b => { b.t0 /= maxT; b.t1 /= maxT; });
      const weights = branches.map(b => (b.depth < 6 ? b.len : 0)), total = weights.reduce((a, b) => a + b, 0);
      for (let i = 0; i < 143; i++) {
        let pick = rnd() * total, k = 0;
        while (pick > weights[k]) pick -= weights[k++];
        const b = branches[k], u = .3 + rnd() * .7, [x, y] = qpoint(b, u), roll = rnd();
        nodes.push({x, y, t: b.t0 + u * (b.t1 - b.t0), kind: roll < .62 ? 0 : roll < .86 ? 1 : 2, r: (2.2 + rnd() * 2) * Math.max(.7, unit), p: rnd() * 6.28});
      }
    }
    const dark = () => document.documentElement.dataset.theme !== 'light';
    const mix = (a, b, u) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * u)).join(',')})`;
    const wood = depth => mix([122, 112, 98], [243, 232, 210], .35 + .65 * depth / 6);
    const woodLight = depth => mix([226, 212, 190], [158, 116, 66], .3 + .7 * depth / 6);
    // Once the tree has grown, its wood never changes: draw it once (with its glow) and reuse the picture,
    // and let the lights twinkle from small pre-drawn sprites instead of blurring every frame.
    let grownTree = null, sprites = null;
    function sprite(colour, night) {
      const size = 48, c = document.createElement('canvas'); c.width = c.height = size;
      const s = c.getContext('2d');
      s.shadowColor = night ? colour : 'transparent'; s.shadowBlur = night ? 12 : 0;
      s.fillStyle = colour; s.beginPath(); s.arc(size / 2, size / 2, 6, 0, 6.283); s.fill();
      return c;
    }
    function draw(G, time) {
      const night = dark();
      if (G >= 1) return drawGrown(time, night);
      drawGrowing(G, time, night);
    }
    // The glow used to come from shadowBlur on every one of ~2,000 stroke segments, which cost the GPU
    // ~300 ms a frame while the tree grew. Now the segments are batched into one path per depth and
    // position along the branch (they share width and colour), drawn once more into a half-size layer,
    // and that layer is blurred in a single pass and laid underneath. Browsers without canvas filters
    // get the same layer at a quarter size, where scaling it back up does a softer job of the blur.
    const canBlur = typeof g.filter === 'string';
    let glow = null;
    function glowLayer() {
      const k = canBlur ? 2 : 4, w = Math.ceil(canvas.width / k), h = Math.ceil(canvas.height / k);
      if (!glow || glow.width !== w || glow.height !== h) { glow = Object.assign(document.createElement('canvas'), {width: w, height: h}); glow.k = k; }
      return glow;
    }
    function drawGrowing(G, time, night) {
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      if (night) {
        const halo = g.createRadialGradient(base[0], H * .45, 0, base[0], H * .45, H * .62);
        halo.addColorStop(0, `rgba(255,206,140,${.1 * G})`); halo.addColorStop(1, 'rgba(255,206,140,0)');
        g.fillStyle = halo; g.fillRect(0, 0, W, H);
        // An orange copy of the halo: the old code drew the halo with a lights' shadow still switched on.
        const haze = g.createRadialGradient(base[0], H * .45, 0, base[0], H * .45, H * .62);
        haze.addColorStop(0, `rgba(245,184,61,${.1 * G})`); haze.addColorStop(1, 'rgba(245,184,61,0)');
        g.fillStyle = haze; g.fillRect(0, 0, W, H);
      }
      const paths = new Map();  // depth * 8 + segment -> {path, width, depth}
      for (const b of branches) {
        const f = Math.min(1, Math.max(0, (G - b.t0) / (b.t1 - b.t0)));
        if (f <= 0) continue;
        for (let i = 0; i < 8; i++) {
          const key = b.depth * 8 + i;
          if (!paths.has(key)) paths.set(key, {path: new Path2D(), width: Math.max(.6, b.w * (1 - .3 * i / 8)), depth: b.depth});
          const [ax, ay] = qpoint(b, i / 8 * f), [bx, by] = qpoint(b, (i + 1) / 8 * f), path = paths.get(key).path;
          path.moveTo(ax, ay); path.lineTo(bx, by);
        }
      }
      g.lineCap = 'round';
      if (night && paths.size) {
        const layer = glowLayer(), lg = layer.getContext('2d');
        lg.setTransform(dpr / layer.k, 0, 0, dpr / layer.k, 0, 0); lg.clearRect(0, 0, W, H);
        lg.lineCap = 'round'; lg.strokeStyle = 'rgba(255,214,160,.55)';
        for (const {path, width} of paths.values()) { lg.lineWidth = width; lg.stroke(path); }
        g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
        if (canBlur) g.filter = `blur(${(5 * unit).toFixed(1)}px)`;  // shadowBlur 10 was a Gaussian with σ = 5 device pixels
        g.drawImage(layer, 0, 0, canvas.width, canvas.height); g.restore();
      }
      for (const {path, width, depth} of paths.values()) {
        g.lineWidth = width; g.strokeStyle = night ? wood(depth) : woodLight(depth); g.stroke(path);
      }
      if (!nodes.length) return;
      const colors = night ? ['#f5b83d', '#5ad1c0', '#fff6df'] : ['#c88a12', '#2a9c8c', '#8a6a3a'];
      if (!sprites || sprites.night !== night) sprites = Object.assign(colors.map((colour) => sprite(colour, night)), {night});
      for (const n of nodes) {
        if (G < n.t) continue;
        const pop = Math.min(1, (G - n.t) * 14), tw = .7 + .3 * Math.sin(time * 1.6 + n.p), size = n.r * (.6 + .4 * pop) * 2 * 4;
        g.globalAlpha = (night ? tw : .55) * pop;
        g.drawImage(sprites[n.kind], n.x - size / 2, n.y - size / 2, size, size);
      }
      g.globalAlpha = 1;
    }
    function drawGrown(time, night) {
      if (!grownTree || grownTree.night !== night || grownTree.width !== canvas.width || grownTree.height !== canvas.height) {
        // render the full-grown tree once, without its lights, into an offscreen picture
        const keep = nodes; nodes = [];
        drawGrowing(1, time, night);
        nodes = keep;
        const c = document.createElement('canvas'); c.width = canvas.width; c.height = canvas.height;
        const cc = c.getContext('2d');
        // A warm, blurred copy under the whole picture gives the grown tree its haze. It used to be
        // redrawn every frame by a shadow left switched on; baking it in costs one blur, once.
        if (night) { cc.shadowColor = 'rgba(255,214,160,.55)'; cc.shadowBlur = 10 * unit; }
        cc.drawImage(canvas, 0, 0);
        grownTree = Object.assign(c, {night});
        const colors = night ? ['#f5b83d', '#5ad1c0', '#fff6df'] : ['#c88a12', '#2a9c8c', '#8a6a3a'];
        sprites = Object.assign(colors.map((colour) => sprite(colour, night)), {night});
      }
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.clearRect(0, 0, canvas.width, canvas.height);
      g.drawImage(grownTree, 0, 0);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      for (const n of nodes) {
        const tw = .7 + .3 * Math.sin(time * 1.6 + n.p), size = n.r * 2 * 4;
        g.globalAlpha = night ? tw : .55;
        g.drawImage(sprites[n.kind], n.x - size / 2, n.y - size / 2, size, size);
      }
      g.globalAlpha = 1;
    }

    const ease = u => (u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
    let start = null, grown = false, running = false, lastDraw = 0, fade = 1;
    const settle = () => {
      if (!label) return;
      label.textContent = `${first} — ${last}`;
      label.classList.add('is-done');
    };
    function frame(now) {
      if (start === null) start = now;
      const u = Math.min(1, (now - start) / 3200);
      if (!grown && u >= 1) { grown = true; settle(); }
      if (!grown || now - lastDraw > 50) {
        lastDraw = now;
        draw(ease(u), now / 1000);
        if (!grown && label) label.textContent = String(Math.round(first + (last - first) * ease(u)));
      }
      // Keep twinkling only while the tree is clearly on screen.
      running = !grown || (fade > .3 && document.visibilityState !== 'hidden');
      if (running) requestAnimationFrame(frame);
    }
    function onScroll() {
      fade = Math.max(.16, 1 - Math.max(0, scrollY) / (innerHeight * .9) * .84);
      canvas.style.opacity = (fade * (dark() ? 1 : .7)).toFixed(3);
      canvas.style.transform = `translateY(${(-Math.min(scrollY, innerHeight) * .12).toFixed(1)}px)`;
      if (!running && grown && fade > .3 && !reduce()) { running = true; requestAnimationFrame(frame); }
    }
    build();
    if (reduce()) { grown = true; draw(1, 0); settle(); }
    else { running = true; requestAnimationFrame(frame); }
    onScroll();
    addEventListener('scroll', onScroll, {passive: true});
    let resizing = 0;
    addEventListener('resize', () => {
      clearTimeout(resizing);
      resizing = setTimeout(() => { build(); draw(grown ? 1 : 0, 0); onScroll(); }, 150);
    });
    if (typeof MutationObserver !== 'undefined') {
      new MutationObserver(() => draw(grown ? 1 : 0, 0)).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
    }
  }
  tree();
})();
