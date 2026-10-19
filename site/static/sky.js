/* Night sky, from orbit.
   A canvas behind the page: a dense star field drawn from a magnitude distribution with a
   slight spread of colour temperature, a Milky Way arch with a bright core and dust lanes,
   the whole field wheeling very slowly about a pole, and along the bottom the limb of a planet
   the width of the screen, its atmosphere a thin bright edge that fades into the dark.
   Static under reduced motion and when the tab is hidden; hidden in the light theme (CSS).
   Same file on the personal homepage and the notes site. No dependencies, no inline styles. */
(function () {
  'use strict';
  var canvas = document.querySelector('canvas.sky');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d', { alpha: false });
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function rng(seed) {                       // deterministic: every visit is the same sky
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var x = Math.imul(s ^ (s >>> 15), 1 | s);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { return (r() + r() + r() + r() - 2) / 2; }   // roughly normal, |x| < 2

  var W = 0, H = 0, DPR = 1, R = 0, pole = null, stars = [], layer = null, layerScale = 1, faint = [], bright = [], fore = null, halos = {};

  function tint(r) {                         // colour temperature: mostly white, some blue-white, some warm
    var t = r();
    if (t < 0.14) return [196, 210, 255];
    if (t < 0.24) return [255, 228, 196];
    if (t < 0.34) return [255, 243, 226];
    return [255, 255, 255];
  }

  function build() {
    var r = rng(20260921);
    // the field must cover the viewport for any rotation: generated on a disc about the pole
    pole = { x: 0.22 * W, y: -0.10 * H };
    R = 0;
    [[0, 0], [W, 0], [0, H], [W, H]].forEach(function (c) { R = Math.max(R, Math.hypot(c[0] - pole.x, c[1] - pole.y)); });
    R *= 1.03;
    var cssArea = Math.PI * R * R / (DPR * DPR);
    var n = Math.min(16000, Math.round(cssArea / 1000 * 3.4));   // ≈3.4 stars per 1000 css px² of disc
    stars = [];
    for (var i = 0; i < n; i++) {
      var a = r() * 6.2832, d = Math.sqrt(r()) * R, m = r(), size, alpha, tw = 0;
      if (m < 0.80) { size = 0.55 + m * 0.5; alpha = 0.28 + m * 0.55; }            // the dust of faint stars
      else if (m < 0.965) { size = 0.95 + (m - 0.80) * 4.2; alpha = 0.72 + (m - 0.80) * 1.4; }
      else { size = 1.7 + (m - 0.965) * 40; alpha = 1; tw = 0.5 + r() * 1.3; }   // a few bright ones, with halo and twinkle
      var c = tint(r);
      stars.push({ x: Math.cos(a) * d, y: Math.sin(a) * d, s: size * DPR, al: Math.min(1, alpha), c: c, tw: tw, ph: r() * 6.2832,
                   col: c[0] + ',' + c[1] + ',' + c[2], fill: 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + Math.min(1, alpha).toFixed(2) + ')' });
    }
    layer = buildLayer(r);
    // the faint stars never change: draw them into the Milky Way layer's coordinates as one batch per colour
    faint = [];
    var groups = {};                                               // one path per colour and brightness step
    stars.forEach(function (st) {
      if (!(st.s < 1.15 * DPR && !st.tw)) return;
      var key = st.col + '|' + Math.round(st.al * 10);
      if (!groups[key]) { groups[key] = { fill: 'rgba(' + st.col + ',' + (Math.round(st.al * 10) / 10).toFixed(1) + ')', path: typeof Path2D !== 'undefined' ? new Path2D() : null, list: [] }; faint.push(groups[key]); }
      if (groups[key].path) groups[key].path.rect(st.x, st.y, st.s, st.s); else groups[key].list.push(st);
    });
    bright = stars.filter(function (st) { return !(st.s < 1.15 * DPR && !st.tw); });
    fore = buildForeground();
  }

  // The planet, the sunrise and the vignette do not move: render them once per size and lay them on top.
  function buildForeground() {
    var c = document.createElement('canvas');
    c.width = W; c.height = H;
    var g = c.getContext('2d');
    drawForeground(g);
    return c;
  }

  // The Milky Way and the background haze, rendered once at a capped resolution and drawn scaled:
  // soft by construction, cheap to rotate.
  function buildLayer(r) {
    var size = Math.min(3072, Math.ceil(R * 2));
    layerScale = size / (R * 2);
    var c = document.createElement('canvas');
    c.width = size; c.height = size;
    var g = c.getContext('2d');
    var s = layerScale;
    g.translate(size / 2, size / 2);
    // band centred on the viewport centre (in pole coordinates), running upper-left to lower-right,
    // gently arched, with its bright core towards the lower right
    var cx = (W / 2 - pole.x) * s, cy = (H * 0.66 - pole.y) * s;
    var ang = 0.60, len = R * 2.3 * s, half = R * 0.15 * s, curve = 0.10;
    g.translate(cx, cy); g.rotate(ang);
    function width(t) { return half * (0.5 + 0.9 * Math.exp(-Math.pow((t - 0.30) / 0.30, 2))); }
    function core(t) { return Math.exp(-Math.pow((t - 0.30) / 0.24, 2)); }
    function arch(x) { return -curve * x * x / len; }
    // glow: many soft blobs along the arch
    for (var i = 0; i < 160; i++) {
      var t = (i / 160 - 0.5), x = t * len, w = width(t), k = core(t);
      var rb = w * (0.7 + 0.6 * r()), y = arch(x) + gauss(r) * w * 0.35;
      var al = 0.016 * (1 + 2.6 * k) * (0.7 + 0.6 * r());
      var col = k > 0.5 ? '246,238,226' : '206,214,244';
      var grd = g.createRadialGradient(x, y, 0, x, y, rb);
      grd.addColorStop(0, 'rgba(' + col + ',' + al.toFixed(3) + ')');
      grd.addColorStop(1, 'rgba(' + col + ',0)');
      g.fillStyle = grd; g.fillRect(x - rb, y - rb, rb * 2, rb * 2);
    }
    // knots: small brighter clumps along the core
    for (i = 0; i < 70; i++) {
      t = 0.30 + gauss(r) * 0.22; x = t * len; w = width(t);
      var rk = w * (0.10 + 0.16 * r()), yk = arch(x) + gauss(r) * w * 0.45;
      grd = g.createRadialGradient(x, yk, 0, x, yk, rk);
      grd.addColorStop(0, 'rgba(250,244,232,' + (0.05 + 0.06 * r()).toFixed(3) + ')');
      grd.addColorStop(1, 'rgba(250,244,232,0)');
      g.fillStyle = grd; g.fillRect(x - rk, yk - rk, rk * 2, rk * 2);
    }
    // dust lanes: darker blobs threaded along the band, slightly off its axis
    for (i = 0; i < 90; i++) {
      t = (i / 90 - 0.5) * 0.9; x = t * len; w = width(t);
      var rd = w * (0.22 + 0.25 * r()), yd = arch(x) - w * 0.22 + gauss(r) * w * 0.18;
      grd = g.createRadialGradient(x, yd, 0, x, yd, rd);
      grd.addColorStop(0, 'rgba(0,0,0,' + (0.18 + 0.22 * core(t)).toFixed(2) + ')');
      grd.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = grd; g.fillRect(x - rd, yd - rd, rd * 2, rd * 2);
    }
    // grains: the unresolved stars that make the band look like a band
    var grains = Math.min(34000, Math.round(len * half / (14 * s * s)));
    for (i = 0; i < grains; i++) {
      t = (r() - 0.5); x = t * len; w = width(t);
      var yg = arch(x) + gauss(r) * w * 0.95;
      var ag = 0.08 + r() * 0.36 * (0.4 + core(t));
      g.fillStyle = 'rgba(232,236,255,' + ag.toFixed(2) + ')';
      g.fillRect(x, yg, 1, 1);
    }
    g.setTransform(1, 0, 0, 1, 0, 0);
    // a faint haze of very small grains everywhere: the sky is never truly empty
    g.translate(size / 2, size / 2);
    for (i = 0; i < 9000; i++) {
      var aa = r() * 6.2832, dd = Math.sqrt(r()) * size / 2;
      g.fillStyle = 'rgba(220,226,255,' + (0.03 + r() * 0.08).toFixed(2) + ')';
      g.fillRect(Math.cos(aa) * dd, Math.sin(aa) * dd, 1, 1);
    }
    return c;
  }

  function halo(st) {
    var key = st.col + (st.s > 2.9 * DPR ? '+' : '');
    if (halos[key]) return halos[key];
    var n = 64, c = document.createElement('canvas'); c.width = c.height = n;
    var g = c.getContext('2d'), m = n / 2;
    var grd = g.createRadialGradient(m, m, 0, m, m, m);
    grd.addColorStop(0, 'rgba(' + st.col + ',0.42)'); grd.addColorStop(0.35, 'rgba(' + st.col + ',0.10)'); grd.addColorStop(1, 'rgba(' + st.col + ',0)');
    g.fillStyle = grd; g.fillRect(0, 0, n, n);
    if (key.slice(-1) === '+') {                                  // a faint spike on the very brightest
      var h = g.createLinearGradient(0, m, n, m), v = g.createLinearGradient(m, 0, m, n);
      [h, v].forEach(function (s) { s.addColorStop(0, 'rgba(' + st.col + ',0)'); s.addColorStop(0.5, 'rgba(' + st.col + ',0.55)'); s.addColorStop(1, 'rgba(' + st.col + ',0)'); });
      g.fillStyle = h; g.fillRect(0, m - 0.5, n, 1); g.fillStyle = v; g.fillRect(m - 0.5, 0, 1, n);
    }
    return (halos[key] = c);
  }

  function resize() {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(window.innerWidth * DPR); canvas.height = Math.round(window.innerHeight * DPR);
    W = canvas.width; H = canvas.height;
    if (!W || !H) return;                    // a hidden or zero-sized window: draw once it has a size
    build();
    draw(performance.now());
  }

  var ROT = 0.00046 / 1000;   // rad per ms: about a degree every 38 s
  function draw(now) {
    if (!layer) return;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);
    var rot = reduced ? 0 : now * ROT;

    // ---- the sky: haze and Milky Way, then the stars, all wheeling about the pole
    ctx.save();
    ctx.translate(pole.x, pole.y);
    ctx.rotate(rot);
    ctx.drawImage(layer, -R, -R, R * 2, R * 2);
    var k, st;
    for (k = 0; k < faint.length; k++) {                            // the faint stars: a few dozen fills
      var grp = faint[k]; ctx.fillStyle = grp.fill;
      if (grp.path) ctx.fill(grp.path); else for (var q = 0; q < grp.list.length; q++) ctx.fillRect(grp.list[q].x, grp.list[q].y, grp.list[q].s, grp.list[q].s);
    }
    for (k = 0; k < bright.length; k++) {
      st = bright[k];
      var al = st.al;
      if (st.tw && !reduced) al *= 0.8 + 0.2 * Math.sin(now * 0.0012 * st.tw + st.ph);
      if (st.s < 1.15 * DPR) { ctx.globalAlpha = al; ctx.fillStyle = 'rgb(' + st.col + ')'; ctx.fillRect(st.x, st.y, st.s, st.s); continue; }
      if (st.s > 2.0 * DPR) {                                      // bloom on the bright ones, from a cached sprite
        var hr = st.s * 3.2;
        ctx.globalAlpha = al; ctx.drawImage(halo(st), st.x - hr, st.y - hr, hr * 2, hr * 2);
      }
      ctx.globalAlpha = al; ctx.fillStyle = 'rgb(' + st.col + ')';
      ctx.beginPath(); ctx.arc(st.x, st.y, st.s / 2, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    if (fore) ctx.drawImage(fore, 0, 0);
  }

  function drawForeground(g) {
    // ---- the planet: a limb the width of the screen, the sun just behind it on the left
    var Rp = W * 1.45, cx = W / 2, cy = H * 0.80 + Rp;
    function limbY(px) { return cy - Math.sqrt(Math.max(0, Rp * Rp - (px - cx) * (px - cx))); }
    // sunrise glare: warm, low on the left, most of it hidden by the body drawn afterwards
    var sx = W * 0.14, sy = limbY(W * 0.14) + H * 0.02, sr = H * 0.62;
    var sun = g.createRadialGradient(sx, sy, 0, sx, sy, sr);
    sun.addColorStop(0, 'rgba(255,246,232,0.80)'); sun.addColorStop(0.06, 'rgba(255,220,180,0.42)');
    sun.addColorStop(0.20, 'rgba(255,170,110,0.13)'); sun.addColorStop(0.45, 'rgba(200,120,90,0.04)'); sun.addColorStop(1, 'rgba(120,80,80,0)');
    g.fillStyle = sun; g.fillRect(sx - sr, sy - sr, sr * 2, sr * 2);
    // atmosphere, wide layer: blue haze that reaches well above the limb
    var GW = Math.max(150 * DPR, H * 0.26);
    var haze = g.createRadialGradient(cx, cy, Rp - 2 * DPR, cx, cy, Rp + GW);
    haze.addColorStop(0, 'rgba(150,190,255,0.34)'); haze.addColorStop(0.10, 'rgba(110,150,240,0.16)');
    haze.addColorStop(0.35, 'rgba(80,110,210,0.06)'); haze.addColorStop(1, 'rgba(60,90,180,0)');
    g.fillStyle = haze; g.fillRect(0, cy - Rp - GW, W, H - (cy - Rp - GW));
    // atmosphere, thin layer: the bright edge itself
    var EW = Math.max(9 * DPR, H * 0.012);
    var edge = g.createRadialGradient(cx, cy, Rp - 1.5 * DPR, cx, cy, Rp + EW);
    edge.addColorStop(0, 'rgba(255,252,246,0.98)'); edge.addColorStop(0.18, 'rgba(210,228,255,0.75)');
    edge.addColorStop(0.55, 'rgba(150,190,255,0.28)'); edge.addColorStop(1, 'rgba(120,160,240,0)');
    g.fillStyle = edge; g.fillRect(0, cy - Rp - EW, W, H - (cy - Rp - EW));
    // the night side: everything above the limb dies away towards the right; clipped to the ring
    g.save();
    g.beginPath(); g.arc(cx, cy, Rp + GW, 0, 6.2832); g.arc(cx, cy, Rp - 1, 0, 6.2832, true); g.clip('evenodd');
    var term = g.createLinearGradient(0, 0, W, 0);
    term.addColorStop(0, 'rgba(0,0,0,0)'); term.addColorStop(0.5, 'rgba(0,0,0,0.10)'); term.addColorStop(1, 'rgba(0,0,0,0.70)');
    g.fillStyle = term; g.fillRect(0, cy - Rp - GW, W, H - (cy - Rp - GW));
    g.restore();
    // the body: dark, with earthshine on the sunward side and the atmosphere seen from inside the limb
    var body = g.createLinearGradient(0, 0, W, 0);
    body.addColorStop(0, '#0b1222'); body.addColorStop(0.35, '#050912'); body.addColorStop(1, '#010204');
    g.fillStyle = body; g.beginPath(); g.arc(cx, cy, Rp, 0, 6.2832); g.fill();
    var IW = Math.max(70 * DPR, H * 0.10);
    var inner = g.createRadialGradient(cx, cy, Rp - IW, cx, cy, Rp);
    inner.addColorStop(0, 'rgba(60,95,170,0)'); inner.addColorStop(0.7, 'rgba(80,120,200,0.10)'); inner.addColorStop(1, 'rgba(120,160,230,0.30)');
    g.fillStyle = inner; g.beginPath(); g.arc(cx, cy, Rp, 0, 6.2832); g.fill();

    // ---- vignette
    var vg = g.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.hypot(W, H) * 0.62);
    vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(0,0,0,0.55)');
    g.fillStyle = vg; g.fillRect(0, 0, W, H);
  }

  // The sky turns about a degree every 38 s: 12 frames a second keep that and the twinkle smooth.
  // Once it has stepped back behind the reading (12% opacity) it is drawn once and left still.
  var handle = null, last = 0;
  function frame(now) {
    if (document.documentElement.classList.contains('sky-back')) { draw(now); handle = null; return; }
    handle = requestAnimationFrame(frame);
    if (now - last < 83) return;
    last = now; draw(now);
  }
  function start() { if (handle == null && !reduced && !document.hidden) handle = requestAnimationFrame(frame); }
  function stop() { if (handle != null) { cancelAnimationFrame(handle); handle = null; } }

  resize();
  window.addEventListener('resize', resize);
  document.addEventListener('visibilitychange', function () { document.hidden ? stop() : start(); });
  window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', function (event) {
    reduced = event.matches;
    reduced ? stop() : start();
  });
  start();

  // the sky is the whole hero; once the reader is into the content it steps back
  var hero = document.querySelector('.hero');
  if (!hero) document.documentElement.classList.add('sky-back');    // a reading page: the sky stays back
  if (hero && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      document.documentElement.classList.toggle('sky-back', !entries[0].isIntersecting);
      if (entries[0].isIntersecting) start();               // back at the hero: the sky turns again
    }, { threshold: 0.12 }).observe(hero);
  }
})();
