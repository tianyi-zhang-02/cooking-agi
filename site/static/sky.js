/* Night sky, as a star chart.
   A canvas behind the page: the northern sky in stereographic projection about the celestial pole,
   so declination circles stay concentric and the whole field turns about Polaris, as it really does.
   Fine rules for the chart, a sparse field of small stars, and three real constellations.
   The cover draws on the band colour; reading pages draw on the paper, faintly.
   Static under reduced motion, when the tab is hidden, and on reading pages. No dependencies. */
(function () {
  'use strict';
  var canvas = document.querySelector('canvas.sky');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d', { alpha: false });
  var root = document.documentElement;
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Right ascension (hours), declination (degrees). J2000, rounded.
  var CONSTELLATIONS = [
    [[11.062, 61.75], [11.031, 56.38], [11.897, 53.69], [12.257, 57.03], [12.900, 55.96], [13.399, 54.93], [13.792, 49.31]],  // Big Dipper
    [[0.153, 59.15], [0.675, 56.54], [0.945, 60.72], [1.430, 60.24], [1.907, 63.67]],                                         // Cassiopeia
    [[2.530, 89.26], [17.537, 86.59], [16.766, 82.04], [15.734, 77.79], [16.292, 75.76], [15.345, 71.83], [14.845, 74.16], [15.734, 77.79]]  // Little Dipper
  ];
  var POLARIS = [2.530, 89.26];
  var NAMED_STAR = 1.5;                      // css px: the constellation stars, slightly above the field

  var cover = !!document.querySelector('.hero');
  var W = 0, H = 0, DPR = 1, S = 0, pole = null, field = [], ink = '#E9E5DB', paper = '#0E141C', seal = '#E8775A', dark = true;

  function rng(seed) {                       // deterministic: every visit is the same sky
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var x = Math.imul(s ^ (s >>> 15), 1 | s);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function radius(dec) { return S * Math.tan((90 - dec) * Math.PI / 360); }
  function place(ra, dec) {                  // sky seen from below: right ascension runs anticlockwise
    var a = ra * Math.PI / 12, r = radius(dec);
    return { x: -r * Math.sin(a), y: -r * Math.cos(a) };
  }

  function palette() {
    var style = getComputedStyle(root);
    paper = style.getPropertyValue(cover ? '--band' : '--paper').trim() || paper;
    ink = style.getPropertyValue(cover ? '--band-ink' : '--ink').trim() || ink;
    seal = style.getPropertyValue('--accent-fill').trim() || seal;
    dark = cover || root.getAttribute('data-theme') !== 'light';
  }

  function build() {
    var narrow = W / DPR < 720;
    pole = cover
      ? { x: W * (narrow ? 0.66 : 0.74), y: H * (narrow ? 0.30 : 0.36) }
      : { x: W * (narrow ? 0.80 : 0.92), y: H * 0.14 };
    S = Math.max(W, H * 1.15) * (narrow ? 0.52 : 0.62);
    var r = rng(20261104), reach = Math.hypot(Math.max(pole.x, W - pole.x), Math.max(pole.y, H - pole.y)) * 1.04;
    var n = Math.min(900, Math.round(Math.PI * reach * reach / (DPR * DPR) / 6000));
    field = [];
    for (var i = 0; i < n; i++) {
      var a = r() * 6.2832, d = Math.sqrt(r()) * reach, m = r();
      field.push({ x: Math.cos(a) * d, y: Math.sin(a) * d, s: (0.6 + m * m * 0.75) * DPR, al: Math.round((0.22 + m * 0.5) * 10) / 10 });
    }
  }

  function resize() {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(window.innerWidth * DPR); canvas.height = Math.round(window.innerHeight * DPR);
    W = canvas.width; H = canvas.height;
    if (!W || !H) return;                    // a hidden or zero-sized window: draw once it has a size
    palette(); build(); draw(performance.now());
  }

  var ROT = 0.00046 / 1000;                  // rad per ms: about a degree every 38 s
  var START = 0.42;                          // puts the Dipper under the pole and Cassiopeia above it
  function draw(now) {
    if (!pole) return;
    ctx.globalAlpha = 1; ctx.fillStyle = paper; ctx.fillRect(0, 0, W, H);
    ctx.save();
    ctx.translate(pole.x, pole.y);
    ctx.rotate(START + (reduced ? 0 : now * ROT));

    // ---- the chart: declination circles and hour lines
    ctx.strokeStyle = ink; ctx.lineWidth = DPR * 0.75;
    ctx.globalAlpha = cover ? 0.13 : dark ? 0.075 : 0.10;
    ctx.beginPath();
    [80, 70, 60, 50, 40, 30, 20, 10, 0].forEach(function (dec) { var rr = radius(dec); ctx.moveTo(rr, 0); ctx.arc(0, 0, rr, 0, 6.2832); });
    for (var h = 0; h < 12; h++) {
      var a = h * Math.PI / 6, inner = radius(80), outer = Math.hypot(W, H) * 1.2;
      ctx.moveTo(Math.cos(a) * inner, Math.sin(a) * inner); ctx.lineTo(Math.cos(a) * outer, Math.sin(a) * outer);
    }
    ctx.stroke();

    // ---- the field
    ctx.fillStyle = ink;
    for (var i = 0; i < field.length; i++) {
      var st = field[i];
      ctx.globalAlpha = st.al;
      ctx.fillRect(st.x, st.y, st.s, st.s);
    }

    // ---- the constellations: on the cover only, so reading pages keep just the chart and the field
    if (cover) CONSTELLATIONS.forEach(function (stars) {
      var points = stars.map(function (c) { return place(c[0], c[1]); });
      ctx.globalAlpha = 0.42; ctx.lineWidth = DPR * 0.8;
      ctx.beginPath();
      points.forEach(function (p, k) { if (k) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y); });
      ctx.stroke();
      ctx.globalAlpha = 0.95;
      points.forEach(function (p) { ctx.beginPath(); ctx.arc(p.x, p.y, NAMED_STAR * DPR, 0, 6.2832); ctx.fill(); });
    });

    // ---- Polaris, the one point of colour
    if (cover) {
      var north = place(POLARIS[0], POLARIS[1]);
      ctx.globalAlpha = 1; ctx.fillStyle = seal;
      ctx.beginPath(); ctx.arc(north.x, north.y, 2.2 * DPR, 0, 6.2832); ctx.fill();
    }
    ctx.restore();
  }

  // About a degree every 38 s: four frames a second are plenty, and cost almost nothing.
  // Once it has stepped back behind the reading it is drawn once and left still.
  var handle = null, last = 0;
  function frame(now) {
    if (root.classList.contains('sky-back')) { draw(now); handle = null; return; }
    handle = requestAnimationFrame(frame);
    if (now - last < 250) return;
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
  if ('MutationObserver' in window) {        // the theme button swaps the palette without a reload
    new MutationObserver(function () { palette(); draw(performance.now()); }).observe(root, { attributes: true, attributeFilter: ['data-theme', 'data-palette'] });
  }
  start();

  // the sky is the whole hero; once the reader is into the content it steps back
  var hero = document.querySelector('.hero');
  if (!hero) root.classList.add('sky-back');  // a reading page: the sky stays back
  if (hero && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      root.classList.toggle('sky-back', !entries[0].isIntersecting);
      if (entries[0].isIntersecting) start(); // back at the hero: the sky turns again
    }, { threshold: 0.12 }).observe(hero);
  }
})();
