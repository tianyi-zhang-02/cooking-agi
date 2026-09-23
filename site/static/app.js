/* AGI 学习笔记 — site behaviour.
   Everything here is progressive enhancement: the page is fully readable with
   JS disabled. Order: chrome, then the live widgets. */

(function () {
  "use strict";
  var LANG = (window.SITE && window.SITE.lang) === "en" ? "en" : "zh";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ---------------------------------------------------------- reading progress */
  var progress = $(".reading-progress span");
  if (progress) {
    var updateProgress = function () {
      var max = document.documentElement.scrollHeight - innerHeight;
      progress.style.transform = "scaleX(" + (max > 0 ? Math.min(1, scrollY / max) : 0) + ")";
    };
    addEventListener("scroll", updateProgress, { passive: true });
    addEventListener("resize", updateProgress, { passive: true });
    updateProgress();
  }

  /* ---------------------------------------------------------- bilingual bits */
  $$("[data-zh]").forEach(function (el) {
    var v = el.getAttribute(LANG === "en" ? "data-en" : "data-zh");
    if (v) el.textContent = v;
  });

  /* ---------------------------------------------- concept-level bilingual flip */
  $$("[data-concept-card]").forEach(function (card) {
    var zh = $("[data-concept-zh]", card), en = $("[data-concept-en]", card);
    if (!zh || !en) return;

    // Each page opens on its own language; the other face is one click away.
    var home = LANG === "en" ? "en" : "zh";
    card.classList.add("has-flip");
    card.dataset.side = home;
    zh.lang = "zh-Hans";
    en.lang = "en";
    (home === "en" ? zh : en).hidden = true;

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "concept-flip";
    btn.setAttribute("aria-pressed", "false");
    card.insertBefore(btn, card.firstChild);

    function show(side, animate) {
      var english = side === "en";
      card.dataset.side = side;
      card.classList.toggle("is-en", english);
      zh.hidden = english;
      en.hidden = !english;
      btn.setAttribute("aria-pressed", String(english));
      btn.setAttribute("aria-label", english ? "切换回中文" : "查看对应英文");
      btn.innerHTML = "<span>" + (english ? "中文" : "English") +
        "</span><i aria-hidden=\"true\">↻</i>";
      if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches && card.animate) {
        card.animate([
          { opacity: .72, transform: "translateY(2px)" },
          { opacity: 1, transform: "translateY(0)" }
        ], { duration: 170, easing: "ease-out" });
      }
    }

    function flip() { show(card.dataset.side === "zh" ? "en" : "zh", true); }
    btn.addEventListener("click", function (e) { e.stopPropagation(); flip(); });
    card.addEventListener("click", function (e) {
      if (e.target.closest("a, button, pre, code, input, textarea, summary, details")) return;
      if (window.getSelection && window.getSelection().toString()) return;
      flip();
    });
    show(home, false);
  });

  /* ---------------------------------------------------------- theme */
  var root = document.documentElement;
  var themeBtn = $(".icon-btn.theme");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var dark = root.dataset.theme !== "light";   // the site is dark unless light is pinned
    root.dataset.theme = dark ? "light" : "dark";
    localStorage.setItem("theme", root.dataset.theme);
    window.dispatchEvent(new CustomEvent("themechange"));
  });

  /* ---------------------------------------------------------- mobile drawer */
  var side = $("#side"), scrim = $(".side-scrim"), menu = $(".icon-btn.menu");
  function drawer(open) {
    if (!side) return;
    side.classList.toggle("open", open);
    if (scrim) scrim.hidden = !open;
    if (menu) menu.setAttribute("aria-expanded", String(open));
  }
  if (menu) menu.addEventListener("click", function () { drawer(!side.classList.contains("open")); });
  if (scrim) scrim.addEventListener("click", function () { drawer(false); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") drawer(false); });

  /* ---------------------------------------------------------- nav groups */
  /* The group holding the current page ships open from the build. Everything
     else remembers whatever the reader last set, and the active group is never
     closed out from under them. */
  var GKEY = "nav:groups", saved = {};
  try { saved = JSON.parse(localStorage.getItem(GKEY) || "{}"); } catch (e) { saved = {}; }
  $$(".nav details").forEach(function (d) {
    var id = d.dataset.grp, holdsActive = !!$("a.active", d);
    if (!holdsActive && typeof saved[id] === "boolean") d.open = saved[id];
    d.addEventListener("toggle", function () {
      saved[id] = d.open;
      try { localStorage.setItem(GKEY, JSON.stringify(saved)); } catch (e) {}
    });
  });

  /* keep the active sidebar entry in view -- a no-op when it already is */
  var active = $(".nav a.active");
  if (active) active.scrollIntoView({ block: "nearest" });

  /* ------------------------------------------------ contributors, in 1 bit
     Avatars are dithered to two colours on a canvas (Floyd–Steinberg), then drift
     slowly over the dithered sky. Hovering, focusing or tapping holds one still. */
  var universe = $(".contributor-universe");
  if (universe) {
    var field = $(".crew-field", universe);
    var motionButton = $(".crew-motion", universe);
    var statusLine = $(".orbit-status", universe);
    var reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
    var userPaused = false, frame = null, previous = 0, bounds = { w: 0, h: 0 };

    function ditherInto(canvas, source, initials) {
      var size = canvas.width, ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, size, size);
      if (source) { try { ctx.drawImage(source, 0, 0, size, size); } catch (e) { source = null; } }
      if (!source) {                                   // no avatar: a dithered disc with initials
        var g = ctx.createRadialGradient(size * .42, size * .36, 1, size * .5, size * .5, size * .62);
        g.addColorStop(0, "#fff"); g.addColorStop(1, "#111");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(size / 2, size / 2, size * .46, 0, 6.284); ctx.fill();
        ctx.fillStyle = "#000"; ctx.font = "600 " + Math.round(size * .34) + "px ui-monospace, monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(initials || "?", size / 2, size / 2 + 1);
      }
      var image;
      try { image = ctx.getImageData(0, 0, size, size); } catch (e) { return false; }
      var d = image.data, grey = new Float32Array(size * size), i, x, y;
      var radius = size / 2 - 0.5, inside = new Uint8Array(size * size);
      for (i = 0; i < grey.length; i++) {
        var ix = i % size - size / 2 + .5, iy = Math.floor(i / size) - size / 2 + .5;
        inside[i] = ix * ix + iy * iy <= radius * radius ? 1 : 0;
        grey[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
      }
      var bins = new Uint32Array(64), counted = 0;            // auto-levels, over the disc only
      for (i = 0; i < grey.length; i++) {
        if (!inside[i]) continue;
        bins[Math.min(63, Math.floor(grey[i] * 64))]++; counted++;
      }
      var lo = 0, hi = 1, seen = 0;
      for (x = 0; x < 64; x++) { seen += bins[x]; if (seen >= counted * 0.04) { lo = x / 64; break; } }
      for (x = 63, seen = 0; x >= 0; x--) { seen += bins[x]; if (seen >= counted * 0.04) { hi = (x + 1) / 64; break; } }
      if (hi - lo < 0.2) { lo = 0; hi = 1; }                  // a flat image: leave it alone
      for (i = 0; i < grey.length; i++) {
        grey[i] = Math.min(1, Math.max(0, (grey[i] - lo) / (hi - lo)));
        grey[i] = Math.pow(grey[i], 0.8) * 1.04 - 0.02;       // lift the midtones a little
      }
      for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
          var k = y * size + x, old = grey[k], neu = old > 0.5 ? 1 : 0, err = old - neu;
          grey[k] = neu;
          if (x + 1 < size) grey[k + 1] += err * 7 / 16;
          if (y + 1 < size) {
            if (x > 0) grey[k + size - 1] += err * 3 / 16;
            grey[k + size] += err * 5 / 16;
            if (x + 1 < size) grey[k + size + 1] += err * 1 / 16;
          }
        }
      }
      var r = size / 2 - 0.5;
      for (i = 0; i < grey.length; i++) {
        var px = i % size - size / 2 + .5, py = Math.floor(i / size) - size / 2 + .5;
        var on = grey[i] > 0.5 && px * px + py * py <= r * r;      // clip to a disc
        d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = on ? 255 : 0;
        d[i * 4 + 3] = px * px + py * py <= r * r ? 255 : 0;
      }
      ctx.putImageData(image, 0, 0);
      return true;
    }

    var travelers = $$(".crew-drifter", field).map(function (element, index) {
      var face = $(".crew-face", element), canvas = $(".crew-bits", element), img = $(".crew-src", element);
      var traveler = {
        element: element, pilot: $(".crew-pilot", element), canvas: canvas, label: $(".crew-label", element),
        x: 0, y: 0, vx: 0, vy: 0, held: false,
        startX: parseFloat(getComputedStyle(element).getPropertyValue("--start-x")) || 20 + index * 7,
        startY: parseFloat(getComputedStyle(element).getPropertyValue("--start-y")) || 30 + index * 5
      };
      function paint() {
        if (ditherInto(canvas, img && img.complete && img.naturalWidth ? img : null, face.dataset.initials)) {
          element.classList.add("is-dithered");
        }
      }
      if (img) { img.complete ? paint() : (img.onload = paint, img.onerror = paint); } else paint();
      return traveler;
    });

    function measure() {
      bounds.w = field.clientWidth; bounds.h = field.clientHeight;
      var many = travelers.length;                       // fewer people, bigger tokens
      var wanted = many <= 3 ? 104 : many <= 6 ? 88 : many <= 12 ? 72 : 58;
      field.style.setProperty("--crew-size",
        Math.max(46, Math.min(wanted, Math.round(bounds.w / 6))) + "px");
      travelers.forEach(function (t, i) {
        var size = t.element.offsetWidth || 64;
        t.x = Math.min(Math.max(bounds.w * t.startX / 100, 8), Math.max(bounds.w - size - 8, 8));
        t.y = Math.min(Math.max(bounds.h * t.startY / 100, 8), Math.max(bounds.h - size - 8, 8));
        var angle = (i * 2.399) % 6.283;
        t.vx = Math.cos(angle) * 5.5; t.vy = Math.sin(angle) * 3.5;   // px per second: very slow
        place(t);
      });
      field.classList.add("is-ready");
    }
    function place(t) { t.element.style.transform = "translate3d(" + t.x.toFixed(1) + "px," + t.y.toFixed(1) + "px,0)"; }
    function step(now) {
      frame = requestAnimationFrame(step);
      var dt = Math.min(64, now - previous) / 1000; previous = now;
      travelers.forEach(function (t) {
        if (t.held) return;
        var size = t.element.offsetWidth || 64;
        t.x += t.vx * dt; t.y += t.vy * dt;
        if (t.x < 6) { t.x = 6; t.vx = Math.abs(t.vx); }
        if (t.x > bounds.w - size - 6) { t.x = bounds.w - size - 6; t.vx = -Math.abs(t.vx); }
        if (t.y < 6) { t.y = 6; t.vy = Math.abs(t.vy); }
        if (t.y > bounds.h - size - 6) { t.y = bounds.h - size - 6; t.vy = -Math.abs(t.vy); }
        place(t);
      });
    }
    function sync() {
      var stopped = userPaused || reduceMotion.matches || document.hidden;
      universe.classList.toggle("is-paused", stopped);
      if (motionButton) {
        motionButton.hidden = reduceMotion.matches;
        motionButton.setAttribute("aria-pressed", userPaused ? "true" : "false");
        motionButton.textContent = userPaused ? motionButton.dataset.play : motionButton.dataset.pause;
      }
      if (stopped) { if (frame) { cancelAnimationFrame(frame); frame = null; } }
      else if (!frame) { previous = performance.now(); frame = requestAnimationFrame(step); }
    }
    function placeLabel(t) {           // keep the card inside the field: nudge sideways, flip up
      if (!t.label) return;
      var size = t.element.offsetWidth || 64, half = t.label.offsetWidth / 2, centre = t.x + size / 2, shift = 0;
      if (centre - half < 10) shift = 10 - (centre - half);
      else if (centre + half > bounds.w - 10) shift = bounds.w - 10 - (centre + half);
      t.label.style.setProperty("--label-shift", Math.round(shift) + "px");
      t.element.classList.toggle("is-flipped", t.y + size + 22 + t.label.offsetHeight > bounds.h);
    }
    function hold(traveler, on) {
      traveler.held = on;
      if (on) placeLabel(traveler);
      traveler.element.classList.toggle("is-held", on);
      traveler.pilot.setAttribute("aria-expanded", on ? "true" : "false");
    }

    travelers.forEach(function (t) {
      t.element.addEventListener("pointerenter", function () { hold(t, true); });
      t.element.addEventListener("pointerleave", function () { hold(t, false); });
      t.pilot.addEventListener("focus", function () { hold(t, true); });
      t.pilot.addEventListener("blur", function () { hold(t, false); });
      t.pilot.addEventListener("click", function () { hold(t, !t.held); });
    });
    if (motionButton) motionButton.addEventListener("click", function () { userPaused = !userPaused; sync(); });
    reduceMotion.addEventListener("change", sync);
    document.addEventListener("visibilitychange", sync);
    addEventListener("resize", measure);
    if (statusLine) {
      var recent = $$(".crew-drifter.is-recent", field).length;
      statusLine.textContent = recent
        ? (LANG === "zh" ? "本周有 " + recent + " 位留下了提交" : recent + (recent === 1 ? " person" : " people") + " committed this week")
        : "";
    }
    measure(); sync();
  }

  /* ------------------------------------------------ the credits, rolling in
     Each name rises into place as it reaches the middle of the screen, the way
     end credits arrive. Without JavaScript they are simply already there. */
  var roll = $(".credits-roll");
  if (roll && "IntersectionObserver" in window) {
    var lines = $$(".credit-group, .credit-role, .credits-roll li", roll.parentNode);
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      lines.forEach(function (line) { line.classList.add("is-waiting"); });
      var watcher = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          // in view, or already above it: a jump to an anchor must not leave
          // the lines it skipped over sitting there invisible
          if (!entry.isIntersecting && entry.boundingClientRect.top > 0) return;
          entry.target.classList.remove("is-waiting");
          watcher.unobserve(entry.target);
        });
      }, { rootMargin: "0px 0px -18% 0px" });
      lines.forEach(function (line) { watcher.observe(line); });
    }
  }

  /* ------------------------------------------------- the crew map, in 1 bit
     A dot per land cell of an equirectangular grid baked by site/tools/bake_world.py.
     Countries someone has declared in crew.toml burn amber; the rest stay dim. */
  var mapSection = $(".crew-map");
  var mapCanvas = mapSection && $(".world-dots", mapSection);
  if (mapCanvas && mapSection.dataset.world) {
    // "US:5,CN:4" — five steps of brightness, drawn as how many of a country's
    // cells burn and how hard, so one commit shows and a hundred cannot show more
    var STEPS = { 1: [0.40, 0.70], 2: [0.55, 0.80], 3: [0.70, 0.88], 4: [0.86, 0.95], 5: [1, 1] };
    var litSteps = (mapSection.dataset.lit || "").split(",").filter(Boolean).map(function (pair) {
      var bits = pair.split(":");
      return { code: bits[0], step: STEPS[bits[1]] ? bits[1] : 3 };
    });
    var drawMap = function (world) {
      var cols = world.w, rows = world.h;
      var size = Math.floor(mapCanvas.width / cols);          // one land cell, in pixels
      var pad = Math.floor((mapCanvas.width - cols * size) / 2);
      var ctx = mapCanvas.getContext("2d");
      mapCanvas.height = rows * size;
      ctx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
      var paint = function (code, colour, grow, density) {
        var cells = (world.countries[code] || {}).cells || "";
        ctx.fillStyle = colour;
        for (var i = 0; i < cells.length; i += 3) {
          var index = parseInt(cells.substr(i, 3), 36);
          // a fixed pseudo-random order, so the same country always burns the same cells
          if (density < 1 && i > 0 && ((index * 2654435761) % 997) / 997 >= density) continue;
          var x = pad + (index % cols) * size, y = Math.floor(index / cols) * size;
          ctx.fillRect(x, y, Math.max(1, size - 1 + grow), Math.max(1, size - 1 + grow));
        }
      };
      Object.keys(world.countries).forEach(function (code) { paint(code, "#3f3f3f", 0, 1); });
      litSteps.forEach(function (lit) {
        var step = STEPS[lit.step];
        ctx.globalAlpha = step[1];
        paint(lit.code, "#E8A672", lit.step >= 4 ? 1 : 0, step[0]);
        ctx.globalAlpha = 1;
      });
    };
    var loadMap = function () {
      fetch(mapSection.dataset.world)
        .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
        .then(drawMap)
        .catch(function () { mapCanvas.hidden = true; });   // the list below still says it
    };
    if ("IntersectionObserver" in window) {
      var mapWatcher = new IntersectionObserver(function (entries) {
        if (!entries.some(function (e) { return e.isIntersecting; })) return;
        mapWatcher.disconnect(); loadMap();
      }, { rootMargin: "300px" });
      mapWatcher.observe(mapSection);
    } else loadMap();
  }

  /* ---------------------------------------------------------- code copy */
  $$(".prose pre").forEach(function (pre) {
    var wrap = document.createElement("div");
    wrap.className = "code-wrap";
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = LANG === "en" ? "copy" : "复制";
    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(pre.innerText).then(function () {
        btn.textContent = LANG === "en" ? "copied" : "已复制";
        btn.classList.add("done");
        setTimeout(function () {
          btn.textContent = LANG === "en" ? "copy" : "复制";
          btn.classList.remove("done");
        }, 1400);
      });
    });
    wrap.appendChild(btn);
  });

  /* ---------------------------------------------------------- toc scrollspy */
  var links = $$(".toc a");
  if (links.length) {
    var targets = links.map(function (a) { return document.getElementById(a.hash.slice(1)); });
    var spy = function () {
      var best = 0, y = window.scrollY + 90;
      targets.forEach(function (el, i) { if (el && el.offsetTop <= y) best = i; });
      links.forEach(function (a, i) { a.classList.toggle("active", i === best); });
    };
    addEventListener("scroll", spy, { passive: true });
    spy();
  }

  /* ---------------------------------------------------------- search */
  var input = $(".search-input"), box = $(".search-results"), idx = null, sel = -1;
  function load() {
    if (idx) return Promise.resolve(idx);
    return fetch((window.SITE.prefix || "") + "search-index.json")
      .then(function (r) { return r.json(); })
      .then(function (j) { idx = j; return j; });
  }
  function score(item, q) {
    var t = item.t.toLowerCase(), x = item.x.toLowerCase(), s = 0;
    if (t.indexOf(q) >= 0) s += 100 - t.indexOf(q);
    if (item.s.toLowerCase().indexOf(q) >= 0) s += 20;
    var n = x.split(q).length - 1;
    return s + Math.min(n, 8) * 3;
  }
  function render(q) {
    var hits = idx.filter(function (i) { return i.l === LANG; })
      .map(function (i) { return { i: i, s: score(i, q) }; })
      .filter(function (h) { return h.s > 0; })
      .sort(function (a, b) { return b.s - a.s; })
      .slice(0, 8);
    box.innerHTML = hits.length
      ? hits.map(function (h) {
          return '<a href="' + (window.SITE.prefix || "") + h.i.u + '">' +
                 h.i.t.replace(/</g, "&lt;") + "<small>" + h.i.s.replace(/</g, "&lt;") + "</small></a>";
        }).join("")
      : '<div class="search-empty">' + (LANG === "en" ? "No matches" : "没有匹配") + "</div>";
    box.hidden = false;
    sel = -1;
  }
  if (input && box) {
    input.addEventListener("input", function () {
      var q = input.value.trim().toLowerCase();
      if (q.length < 1) { box.hidden = true; return; }
      load().then(function () { render(q); });
    });
    input.addEventListener("keydown", function (e) {
      var items = $$("a", box);
      if (e.key === "Escape") { box.hidden = true; input.blur(); return; }
      if (!items.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        sel = (sel + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length;
        items.forEach(function (a, i) { a.classList.toggle("sel", i === sel); });
      } else if (e.key === "Enter" && sel >= 0) { items[sel].click(); }
    });
    document.addEventListener("click", function (e) {
      if (!e.target.closest(".search")) box.hidden = true;
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== input &&
          !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
        e.preventDefault(); drawer(true); input.focus();
      }
    });
  }

  /* ---------------------------------------------------------- katex + mermaid */
  function typeset() {
    if (!window.renderMathInElement) return;
    // the rail and sidebar carry heading text too, so they need it as well
    $$(".prose, .toc, .side-nav").forEach(function (scope) {
    window.renderMathInElement(scope, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
      throwOnError: false
    });
    });
  }
  if (document.readyState === "complete") typeset();
  else addEventListener("load", typeset);

  if ($(".mermaid")) {
    var s = document.createElement("script");
    s.type = "module";
    s.textContent =
      'import m from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";' +
      'var d=document.documentElement.dataset.theme==="dark"||' +
      '(!document.documentElement.dataset.theme&&matchMedia("(prefers-color-scheme: dark)").matches);' +
      'm.initialize({startOnLoad:true,theme:d?"dark":"neutral",fontFamily:"Charter, Georgia, serif"});';
    document.body.appendChild(s);
  }

  /* ====================================================================== */
  /* XOR playground                                                          */
  /* ====================================================================== */
  function css(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function mulberry(seed) {           // small deterministic PRNG
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeData(n) {
    var rnd = mulberry(7), c = [[1.5, 1.5, 1], [-1.5, -1.5, 1], [1.5, -1.5, 0], [-1.5, 1.5, 0]];
    var pts = [];
    for (var b = 0; b < 4; b++) {
      for (var i = 0; i < n; i++) {
        // Box-Muller for a proper gaussian blob
        var u = Math.max(rnd(), 1e-9), v = rnd(), r = Math.sqrt(-2 * Math.log(u));
        pts.push({ x: c[b][0] + 0.42 * r * Math.cos(2 * Math.PI * v),
                   y: c[b][1] + 0.42 * r * Math.sin(2 * Math.PI * v),
                   t: c[b][2] });
      }
    }
    return pts;
  }

  function Net(H, useAct) {
    var rnd = mulberry(3), g = function () { return (rnd() * 2 - 1) * 0.9; };
    this.H = H; this.act = useAct;
    this.W1 = []; this.b1 = []; this.W2 = []; this.b2 = 0;
    for (var i = 0; i < H; i++) {
      this.W1.push([g(), g()]); this.b1.push(0); this.W2.push(g());
    }
    // Adam state
    this.m = { W1: this.W1.map(function () { return [0, 0]; }), b1: this.b1.map(function () { return 0; }),
               W2: this.W2.map(function () { return 0; }), b2: 0 };
    this.v = { W1: this.W1.map(function () { return [0, 0]; }), b1: this.b1.map(function () { return 0; }),
               W2: this.W2.map(function () { return 0; }), b2: 0 };
    this.t = 0;
  }

  Net.prototype.forward = function (x, y) {
    var z1 = new Float64Array(this.H), h = new Float64Array(this.H), z2 = this.b2;
    for (var i = 0; i < this.H; i++) {
      z1[i] = this.W1[i][0] * x + this.W1[i][1] * y + this.b1[i];
      h[i] = this.act ? Math.max(0, z1[i]) : z1[i];
      z2 += this.W2[i] * h[i];
    }
    return { z1: z1, h: h, z2: z2 };
  };

  Net.prototype.step = function (data, lr) {
    var H = this.H, N = data.length;
    var gW1 = [], gb1 = new Float64Array(H), gW2 = new Float64Array(H), gb2 = 0, loss = 0, ok = 0;
    for (var i = 0; i < H; i++) gW1.push([0, 0]);

    for (var n = 0; n < N; n++) {
      var p = data[n], f = this.forward(p.x, p.y);
      var z = Math.max(-30, Math.min(30, f.z2));
      var s = 1 / (1 + Math.exp(-z));
      loss += -(p.t * Math.log(s + 1e-9) + (1 - p.t) * Math.log(1 - s + 1e-9));
      if ((f.z2 > 0 ? 1 : 0) === p.t) ok++;
      var dz2 = (s - p.t) / N;                       // d(loss)/d(z2)
      gb2 += dz2;
      for (var j = 0; j < H; j++) {
        gW2[j] += dz2 * f.h[j];
        var dh = dz2 * this.W2[j];
        var dz1 = this.act ? (f.z1[j] > 0 ? dh : 0) : dh;
        gW1[j][0] += dz1 * p.x; gW1[j][1] += dz1 * p.y; gb1[j] += dz1;
      }
    }

    this.t++;
    var b1c = 1 - Math.pow(0.9, this.t), b2c = 1 - Math.pow(0.999, this.t), self = this;

    for (var k = 0; k < H; k++) {
      (function (k) {
        var mW = self.m.W1[k], vW = self.v.W1[k];
        for (var d = 0; d < 2; d++) {
          mW[d] = 0.9 * mW[d] + 0.1 * gW1[k][d];
          vW[d] = 0.999 * vW[d] + 0.001 * gW1[k][d] * gW1[k][d];
          self.W1[k][d] -= lr * (mW[d] / b1c) / (Math.sqrt(vW[d] / b2c) + 1e-8);
        }
        self.m.b1[k] = 0.9 * self.m.b1[k] + 0.1 * gb1[k];
        self.v.b1[k] = 0.999 * self.v.b1[k] + 0.001 * gb1[k] * gb1[k];
        self.b1[k] -= lr * (self.m.b1[k] / b1c) / (Math.sqrt(self.v.b1[k] / b2c) + 1e-8);

        self.m.W2[k] = 0.9 * self.m.W2[k] + 0.1 * gW2[k];
        self.v.W2[k] = 0.999 * self.v.W2[k] + 0.001 * gW2[k] * gW2[k];
        self.W2[k] -= lr * (self.m.W2[k] / b1c) / (Math.sqrt(self.v.W2[k] / b2c) + 1e-8);
      })(k);
    }
    this.m.b2 = 0.9 * this.m.b2 + 0.1 * gb2;
    this.v.b2 = 0.999 * this.v.b2 + 0.001 * gb2 * gb2;
    this.b2 -= lr * (this.m.b2 / b1c) / (Math.sqrt(this.v.b2 / b2c) + 1e-8);

    return { loss: loss / N, acc: ok / N };
  };

  /* ---- roadmap: route switcher, expand-all + read-it checkmarks -------------------------
     The markup (site/roadmap.toml -> build.py) is complete without this script:
     every module is already a link. Here we only hide the other tracks and keep
     a private "done" list in localStorage. */
  $$('[data-widget="roadmap"]').forEach(function (root) {
    var tabs = $$(".rm-tab", root), tracks = $$(".rm-track", root);
    var store = {
      get: function (key, fallback) { try { return JSON.parse(localStorage.getItem(key)) || fallback; } catch (e) { return fallback; } },
      set: function (key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) {} }
    };
    var done = store.get("roadmap-done", {});

    function progress(track) {
      var keys = {};
      $$(".rm-mod[data-key]", track).forEach(function (m) { keys[m.dataset.key] = true; });
      var all = Object.keys(keys), n = all.filter(function (k) { return done[k]; }).length;
      $(".rm-bar i", track).style.width = (all.length ? n / all.length * 100 : 0) + "%";
      $(".rm-count", track).textContent = n + " / " + all.length;
    }
    function paint() {
      $$(".rm-mod[data-key]", root).forEach(function (mod) {
        var on = !!done[mod.dataset.key];
        mod.classList.toggle("is-done", on);
        $(".rm-check", mod).setAttribute("aria-pressed", on ? "true" : "false");
      });
      tracks.forEach(progress);
    }
    function select(id, remember) {
      tabs.forEach(function (t) { t.setAttribute("aria-selected", t.dataset.track === id ? "true" : "false"); });
      tracks.forEach(function (t) { t.classList.toggle("is-current", t.dataset.track === id); });
      if (remember) store.set("roadmap-track", id);
    }

    $$(".rm-expand", root).forEach(function (btn) {
      btn.setAttribute("aria-pressed", "false");
      btn.addEventListener("click", function () {
        var open = btn.getAttribute("aria-pressed") !== "true";
        $$(".stack-layer", btn.closest(".rm-track")).forEach(function (layer) { layer.open = open; });
        btn.setAttribute("aria-pressed", open ? "true" : "false");
      });
    });
    root.classList.add("is-live");
    tabs.forEach(function (tab) { tab.addEventListener("click", function () { select(tab.dataset.track, true); }); });
    root.addEventListener("click", function (e) {
      var check = e.target.closest && e.target.closest(".rm-check");
      if (!check) return;
      var key = check.parentNode.dataset.key;
      if (done[key]) delete done[key]; else done[key] = 1;
      store.set("roadmap-done", done);
      paint();
    });
    var fromHash = (location.hash.match(/^#track-([a-z0-9-]+)$/) || [])[1];
    var ids = tracks.map(function (t) { return t.dataset.track; });
    var first = ids.indexOf(fromHash) >= 0 ? fromHash : ids.indexOf(store.get("roadmap-track", "")) >= 0 ? store.get("roadmap-track", "") : ids[0];
    select(first, false);
    paint();
  });

  /* ---- quick-review question bank: open all, close all, draw one at random ---- */
  $$("[data-qbank-tools]").forEach(function (bar) {
    var cards = $$("details.qa");
    bar.addEventListener("click", function (e) {
      var action = e.target.getAttribute && e.target.getAttribute("data-qb");
      if (!action) return;
      if (action === "random") {
        if (!cards.length) return;
        cards.forEach(function (c) { c.classList.remove("is-picked"); });
        var pick = cards[Math.floor(Math.random() * cards.length)];
        pick.open = false;                        // answer in your head first
        pick.classList.add("is-picked");
        pick.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
        return;
      }
      cards.forEach(function (c) { c.open = action === "open"; });
    });
  });

  /* ---- block tabs: on a narrow screen the strip scrolls, so bring the current block into view ---- */
  $$(".subtabs, .topbar-tabs").forEach(function (strip) {
    var current = $("a.active", strip);
    if (current && strip.scrollWidth > strip.clientWidth) {
      strip.scrollLeft = current.offsetLeft - (strip.clientWidth - current.offsetWidth) / 2;
    }
  });

  /* ---- Transformer lab -----------------------------------------------------
     Eight SVG figures (static/tx-lab.js + tx-lab.css). Only pages that embed a
     tx-* widget pay for them. */
  if ($('[data-widget^="tx-"]')) {
    var txBase = ((window.SITE && window.SITE.prefix) || "") + "static/tx-lab";
    var txVer = "?v=" + ((window.SITE && window.SITE.built) || "");
    var txCss = document.createElement("link");
    txCss.rel = "stylesheet"; txCss.href = txBase + ".css" + txVer;
    document.head.appendChild(txCss);
    var txJs = document.createElement("script");
    txJs.src = txBase + ".js" + txVer;
    document.body.appendChild(txJs);
  }

  $$('[data-widget="xor"]').forEach(function (root) {
    var cv = $(".xor-canvas", root), ctx = cv.getContext("2d");
    var actBox = $(".xor-act", root), hidRange = $(".xor-hidden", root);
    var hidOut = $(".xor-hidden-out", root), reset = $(".xor-reset", root);
    var elStep = $(".xor-step", root), elLoss = $(".xor-loss", root), elAcc = $(".xor-acc", root);

    var data = makeData(30), net, hist, raf = null;
    var LIM = 3, CELL = 6;

    function init() {
      net = new Net(parseInt(hidRange.value, 10), actBox.checked);
      hist = [];
      hidOut.textContent = hidRange.value;
    }

    function palette() {
      return {
        ink: css("--ink", "#1b1e23"), rule: css("--rule", "#dee1e5"),
        muted: css("--muted", "#6c727b"), panel: css("--panel", "#fff"),
        c0: css("--link", "#2f5d7c"), c1: css("--ember", "#c8501e")
      };
    }

    function draw() {
      var P = palette(), W = cv.width, H = cv.height, S = H, pad = 0;
      ctx.clearRect(0, 0, W, H);

      /* -- left: decision surface ------------------------------------- */
      var px = function (x) { return pad + (x + LIM) / (2 * LIM) * S; };
      var py = function (y) { return pad + (LIM - y) / (2 * LIM) * S; };
      for (var gx = 0; gx < S; gx += CELL) {
        for (var gy = 0; gy < S; gy += CELL) {
          var x = (gx + CELL / 2) / S * 2 * LIM - LIM;
          var y = LIM - (gy + CELL / 2) / S * 2 * LIM;
          var z = net.forward(x, y).z2;
          var a = Math.min(0.30, 0.06 + Math.abs(Math.tanh(z * 0.8)) * 0.24);
          ctx.fillStyle = (z > 0 ? P.c1 : P.c0);
          ctx.globalAlpha = a;
          ctx.fillRect(pad + gx, pad + gy, CELL, CELL);
        }
      }
      ctx.globalAlpha = 1;
      data.forEach(function (p) {
        ctx.beginPath();
        ctx.arc(px(p.x), py(p.y), 3, 0, 7);
        ctx.fillStyle = p.t ? P.c1 : P.c0;
        ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = P.panel; ctx.stroke();
      });
      ctx.strokeStyle = P.rule; ctx.lineWidth = 1;
      ctx.strokeRect(pad + 0.5, pad + 0.5, S - 1, S - 1);

      /* -- right: loss curve ------------------------------------------ */
      var x0 = S + 26, w = W - x0 - 6, h = S - 34, y0 = 24;
      ctx.strokeStyle = P.rule;
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
      ctx.fillStyle = P.muted;
      ctx.font = '11px ui-sans-serif, -apple-system, system-ui, sans-serif';
      ctx.fillText(LANG === "en" ? "training loss" : "训练 loss", x0, y0 - 8);
      var LN2 = Math.log(2);
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(x0, y0 + h - (LN2 / 0.8) * h); ctx.lineTo(x0 + w, y0 + h - (LN2 / 0.8) * h);
      ctx.strokeStyle = P.rule; ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = P.muted;
      ctx.fillText("ln 2", x0 + w - 26, y0 + h - (LN2 / 0.8) * h - 4);

      if (hist.length > 1) {
        ctx.beginPath();
        hist.forEach(function (l, i) {
          var X = x0 + (i / Math.max(hist.length - 1, 1)) * w;
          var Y = y0 + h - Math.min(l / 0.8, 1) * h;
          i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
        });
        ctx.strokeStyle = P.c1; ctx.lineWidth = 1.8; ctx.stroke();
      }
      if (!net.act) {
        ctx.fillStyle = P.muted;
        ctx.font = 'italic 12px Charter, Georgia, serif';
        var msg = LANG === "en" ? "no activation → stuck at ln 2" : "没有激活函数 → 卡在 ln 2";
        ctx.fillText(msg, x0 + 8, y0 + h - 12);
      }
    }

    function loop() {
      var r;
      for (var i = 0; i < 5; i++) r = net.step(data, 0.06);
      hist.push(r.loss);
      if (hist.length > 260) hist.shift();
      elStep.textContent = net.t;
      elLoss.textContent = r.loss.toFixed(3);
      elAcc.textContent = (r.acc * 100).toFixed(1) + "%";
      draw();
      raf = requestAnimationFrame(loop);
    }

    function restart() {
      if (raf) cancelAnimationFrame(raf);
      init(); draw();
      raf = requestAnimationFrame(loop);
    }

    actBox.addEventListener("change", restart);
    hidRange.addEventListener("input", function () { hidOut.textContent = hidRange.value; });
    hidRange.addEventListener("change", restart);
    reset.addEventListener("click", restart);
    addEventListener("themechange", draw);

    /* only run while on screen */
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting && !raf) raf = requestAnimationFrame(loop);
        else if (!e.isIntersecting && raf) { cancelAnimationFrame(raf); raf = null; }
      });
    }, { rootMargin: "120px" });
    init(); draw(); io.observe(root);
  });
})();
