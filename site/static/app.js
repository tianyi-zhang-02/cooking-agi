/* AGI 大锅烩 — site behaviour.
   Everything here is progressive enhancement: the page is fully readable with
   JS disabled. Order: chrome, then the live widgets. */

(function () {
  "use strict";
  var LANG = (window.SITE && window.SITE.lang) === "en" ? "en" : "zh";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ---------------------------------------------------------- bilingual bits */
  $$("[data-zh]").forEach(function (el) {
    var v = el.getAttribute(LANG === "en" ? "data-en" : "data-zh");
    if (v) el.textContent = v;
  });

  /* ---------------------------------------------------------- theme */
  var root = document.documentElement;
  var themeBtn = $(".icon-btn.theme");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
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
