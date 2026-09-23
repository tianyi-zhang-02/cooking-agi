/* Transformer lab — eight interactive figures for 00-foundations/transformer-lab.md.
   Loaded by app.js only on pages that contain a [data-widget^="tx-"] figure.
   Each figure mounts into the <figure class="widget tx-lab"> that build.py emits;
   everything is inline SVG + DOM, themed through the site's CSS variables. */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var LANG = (window.SITE && window.SITE.lang) === 'en' ? 'en' : 'zh';
  var uid = 0;

  function lang() { return LANG; }
  function t(en, zh) { return LANG === 'zh' && zh != null ? zh : en; }
  // The page language is fixed at build time, so a bilingual node is just its text.
  function bi(node, en, zh) { node.textContent = t(en, zh); return node; }

  function setAttrs(node, attrs) {
    if (!attrs) return node;
    Object.keys(attrs).forEach(function (key) {
      var value = attrs[key];
      if (value == null || value === false) return;
      if (key === 'text') node.textContent = value;
      else if (key === 'bi') bi(node, value[0], value[1]);
      else if (key === 'on') Object.keys(value).forEach(function (type) { node.addEventListener(type, value[type]); });
      else node.setAttribute(key, value === true ? '' : value);
    });
    return node;
  }
  function append(node, kids) {
    (kids || []).forEach(function (kid) {
      if (kid == null) return;
      node.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
    });
    return node;
  }
  function h(tag, attrs, kids) { return append(setAttrs(document.createElement(tag), attrs), kids); }
  function s(tag, attrs, kids) { return append(setAttrs(document.createElementNS(SVGNS, tag), attrs), kids); }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

  // Mount into the figure build.py emitted: title into the head, everything else into the body.
  function figure(name, opts) {
    var root = document.querySelector('[data-widget="' + name + '"]');
    var body = root && root.querySelector('.widget-body');
    if (!body) return null;
    var title = root.querySelector('.widget-title');
    if (title) bi(title, opts.title[0], opts.title[1]);
    clear(body);
    var controls = h('div', { class: 'fig-controls' }), stage = h('div', { class: 'fig-stage' }), foot = h('div', { class: 'fig-foot' });
    append(body, [controls, stage, foot]);
    return { root: root, controls: controls, stage: stage, foot: foot };
  }

  // Segmented control. items: [{ id, en, zh }]
  function seg(items, value, onChange, label) {
    var wrap = h('div', { class: 'seg', role: 'group', 'aria-label': label || null });
    var buttons = items.map(function (item) {
      var button = h('button', { type: 'button', class: 'seg-btn', 'aria-pressed': item.id === value ? 'true' : 'false', bi: [item.en, item.zh] });
      button.addEventListener('click', function () { set(item.id); onChange(item.id); });
      wrap.appendChild(button);
      return button;
    });
    function set(id) {
      buttons.forEach(function (button, i) { button.setAttribute('aria-pressed', items[i].id === id ? 'true' : 'false'); });
    }
    return { el: wrap, set: set };
  }

  function button(en, zh, onClick) {
    return h('button', { type: 'button', class: 'tx-btn', bi: [en, zh], on: { click: onClick } });
  }

  // Labelled range input. opts: { en, zh, min, max, step, value, format, onInput }
  function slider(opts) {
    var id = 'tx-range-' + (++uid);
    var input = h('input', { type: 'range', id: id, min: opts.min, max: opts.max, step: opts.step || 1, value: opts.value });
    var out = h('output', { class: 'ctl-val', for: id });
    function sync() { out.textContent = opts.format ? opts.format(+input.value) : input.value; }
    input.addEventListener('input', function () { sync(); opts.onInput(+input.value); });
    sync();
    return {
      el: h('div', { class: 'ctl' }, [h('label', { class: 'ctl-label', for: id, bi: [opts.en, opts.zh] }), input, out]),
      input: input,
      set: function (v) { input.value = v; sync(); },
      value: function () { return +input.value; }
    };
  }

  function readout(en, zh) {
    var value = h('strong', { class: 'ro-val' });
    return {
      el: h('div', { class: 'ro' }, [h('span', { class: 'ro-label', bi: [en, zh] }), value]),
      set: function (text) { value.textContent = text; }
    };
  }

  // requestAnimationFrame loop with a capped delta, so a background tab does not jump.
  function loop(fn) {
    var handle = null, last = 0;
    function frame(now) {
      handle = requestAnimationFrame(frame);
      fn(now, Math.min(64, now - last));
      last = now;
    }
    return {
      start: function () { if (handle == null) { last = performance.now(); handle = requestAnimationFrame(frame); } },
      stop: function () { if (handle != null) { cancelAnimationFrame(handle); handle = null; } },
      running: function () { return handle != null; }
    };
  }

  function whenVisible(node, onShow, onHide) {
    if (!('IntersectionObserver' in window)) { onShow(); return; }
    new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { (entry.isIntersecting ? onShow : onHide)(); });
    }, { threshold: 0.2 }).observe(node);
  }

  function ease(x) { return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function lerp(a, b, x) { return a + (b - a) * x; }
  function fmtInt(n) { return Math.round(n).toLocaleString('en-US'); }
  function fmtBytes(bytes) {
    var units = ['B', 'KB', 'MB', 'GB', 'TB'], i = 0;
    while (bytes >= 1024 && i < units.length - 1) { bytes /= 1024; i++; }
    return (bytes >= 100 ? bytes.toFixed(0) : bytes >= 10 ? bytes.toFixed(1) : bytes.toFixed(2)) + ' ' + units[i];
  }

  // Small deterministic PRNG so every reader sees the same "random" demo.
  function rng(seed) {
    var state = seed >>> 0;
    return function () {
      state = (state + 0x6D2B79F5) >>> 0;
      var x = Math.imul(state ^ (state >>> 15), 1 | state);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  window.TX = {
    reduced: reduced, lang: lang, t: t, bi: bi, h: h, s: s, clear: clear, append: append,
    figure: figure, seg: seg, button: button, slider: slider, readout: readout,
    loop: loop, whenVisible: whenVisible, ease: ease, clamp: clamp, lerp: lerp,
    fmtInt: fmtInt, fmtBytes: fmtBytes, rng: rng,
    onLang: function () {}
  };
})();

/* 01 · Architecture map. One decoder block that morphs between model families.
   Every optional part (cross-attention, the norms, the position add, the final
   norm) has a presence value in [0,1]; selecting a family tweens those values
   and the layout is recomputed from them each frame. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-arch', { title: ['Architecture map · pick a family', '结构图 · 选择一个模型家族'] });
  if (!fig) return;
  var s = TX.s, h = TX.h;

  function rep(n, a, f) { var out = []; for (var i = 0; i < n; i++) out.push({ a: a, f: f }); return out; }
  function cycle(n, fn) { var out = []; for (var i = 0; i < n; i++) out.push(fn(i)); return out; }

  var DEC = ['Decoder-only', 'Decoder-only（仅解码器）'];
  var PRE_RMS = ['Pre-norm · RMSNorm'];

  var FAMILIES = [
    {
      id: 'vanilla', chip: ['Transformer ’17'], name: 'Transformer (base)', year: '2017', layers: '6',
      cfg: { cross: 1, pre: 0, inner: 0, post: 1, pos: 1, fin: 0 }, norm: 'LayerNorm',
      posLabel: ['+ sinusoidal position', '+ 正弦位置编码'],
      attn: { sub: ['MHA · 8 heads', 'MHA · 8 个 head'], kv: 16 }, tags: [],
      ffn: { sub: ['ReLU · d_ff = 4 × d_model'], experts: 0 },
      stack: rep(6, 'full', 'dense'), stackNote: ['6 decoder layers, plus a 6-layer encoder', '6 层 decoder，另有 6 层 encoder'],
      spec: {
        layout: ['Encoder–decoder · 6 + 6 layers', 'Encoder–decoder · 6 + 6 层'],
        norm: ['Post-LN · LayerNorm'],
        pos: ['Sinusoidal, added to the embedding', '正弦位置编码，加在 embedding 上'],
        attn: ['MHA · 8 heads, plus cross-attention', 'MHA · 8 个 head，另有 cross-attention'],
        ffn: ['Dense · ReLU · d_ff 2,048'],
        ctx: ['Sentence pairs (translation)', '句子对（机器翻译）'],
        size: ['65M · d_model 512'],
        note: ['The baseline. Every later family is a short diff against this block.', '基线。后面的每个家族，都可以看成对这个 block 的一小段 diff。']
      }
    },
    {
      id: 'gpt2', chip: ['GPT-2 / 3'], name: 'GPT-2 / GPT-3', year: '2019–20', layers: '48',
      cfg: { cross: 0, pre: 1, inner: 0, post: 0, pos: 1, fin: 1 }, norm: 'LayerNorm',
      posLabel: ['+ learned position', '+ 可学习位置 embedding'],
      attn: { sub: ['MHA · 25 heads (GPT-2 XL)', 'MHA · 25 个 head（GPT-2 XL）'], kv: 16 }, tags: [],
      ffn: { sub: ['GELU · d_ff = 4 × d_model'], experts: 0 },
      stack: rep(12, 'full', 'dense'), stackNote: ['48 identical layers in GPT-2 XL (12 shown)', 'GPT-2 XL 共 48 层，结构相同（图中画 12 层）'],
      spec: {
        layout: DEC,
        norm: ['Pre-LN · LayerNorm, plus a final LayerNorm', 'Pre-LN · LayerNorm，末尾再加一层 LayerNorm'],
        pos: ['Learned absolute positions', '可学习的绝对位置 embedding'],
        attn: ['MHA · 25 heads (GPT-2 XL) · 96 heads (GPT-3)', 'MHA · 25 个 head（GPT-2 XL）· 96 个（GPT-3）'],
        ffn: ['Dense · GELU · 4×'],
        ctx: ['1,024 (GPT-2) · 2,048 (GPT-3)'],
        size: ['1.5B: 48 layers, d 1,600 · 175B: 96 layers, d 12,288', '1.5B：48 层，d 1,600 · 175B：96 层，d 12,288'],
        note: ['Drops the encoder and cross-attention, and moves the norm inside the residual branch so that deep stacks train stably. GPT-3 also alternates dense and locally banded sparse attention layers.', '去掉 encoder 和 cross-attention，把 norm 移进残差分支，让深层网络训练更稳定。GPT-3 还交替使用 dense 与局部带状的稀疏 attention 层。']
      }
    },
    {
      id: 'llama3', chip: ['Llama 3'], name: 'Llama 3 · 8B', year: '2024', layers: '32',
      cfg: { cross: 0, pre: 1, inner: 0, post: 0, pos: 0, fin: 1 }, norm: 'RMSNorm',
      attn: { sub: ['GQA · 32 query / 8 KV heads', 'GQA · 32 个 query / 8 个 KV head'], kv: 4 }, tags: [['RoPE', '#tx-rope']],
      ffn: { sub: ['SwiGLU · d_ff 14,336'], experts: 0 },
      stack: rep(12, 'full', 'dense'), stackNote: ['32 identical layers (12 shown)', '32 层，结构相同（图中画 12 层）'],
      spec: {
        layout: DEC, norm: PRE_RMS,
        pos: ['RoPE · θ = 500,000'],
        attn: ['GQA · 32 query / 8 KV heads', 'GQA · 32 个 query / 8 个 KV head'],
        ffn: ['Dense · SwiGLU · d_ff 14,336'],
        ctx: ['8K → 128K (Llama 3.1)'],
        size: ['8B: 32 layers, d 4,096 · also 70B, 405B', '8B：32 层，d 4,096 · 另有 70B、405B'],
        note: ['The modern default recipe: RMSNorm, RoPE, SwiGLU and GQA, with no bias terms. Most open models since are variations on it.', '现代默认配方：RMSNorm、RoPE、SwiGLU、GQA，不带 bias。此后大多数开源模型都是它的变体。']
      }
    },
    {
      id: 'mistral', chip: ['Mistral 7B'], name: 'Mistral 7B', year: '2023', layers: '32',
      cfg: { cross: 0, pre: 1, inner: 0, post: 0, pos: 0, fin: 1 }, norm: 'RMSNorm',
      attn: { sub: ['GQA · 32 / 8 · sliding window', 'GQA · 32 / 8 · sliding window'], kv: 4 }, tags: [['RoPE', '#tx-rope'], ['window 4,096', '#tx-windows']],
      ffn: { sub: ['SwiGLU · d_ff 14,336'], experts: 0 },
      stack: rep(12, 'local', 'dense'), stackNote: ['32 layers, all with a 4,096-token window (12 shown)', '32 层，每层都是 4,096 token 的窗口（图中画 12 层）'],
      spec: {
        layout: DEC, norm: PRE_RMS, pos: ['RoPE'],
        attn: ['GQA · 32 / 8 heads · sliding window 4,096', 'GQA · 32 / 8 个 head · sliding window 4,096'],
        ffn: ['Dense · SwiGLU · d_ff 14,336'],
        ctx: ['8K, through a 4,096-token window per layer', '8K，每层只看 4,096 token 的窗口'],
        size: ['7B: 32 layers, d 4,096', '7B：32 层，d 4,096'],
        note: ['The Llama recipe plus a sliding window: each layer looks back 4,096 tokens, and stacking layers extends the reach.', 'Llama 配方加上 sliding window：每层只回看 4,096 个 token，层数叠起来之后可达范围继续扩大。']
      }
    },
    {
      id: 'mixtral', chip: ['Mixtral 8x7B'], name: 'Mixtral 8x7B', year: '2023', layers: '32',
      cfg: { cross: 0, pre: 1, inner: 0, post: 0, pos: 0, fin: 1 }, norm: 'RMSNorm',
      attn: { sub: ['GQA · 32 query / 8 KV heads', 'GQA · 32 个 query / 8 个 KV head'], kv: 4 }, tags: [['RoPE', '#tx-rope']],
      ffn: { sub: ['MoE · 8 experts, top-2', 'MoE · 8 个 expert，选 2 个'], experts: 8, k: 2 },
      stack: rep(12, 'full', 'moe'), stackNote: ['32 layers, every FFN is an MoE (12 shown)', '32 层，每层 FFN 都是 MoE（图中画 12 层）'],
      spec: {
        layout: DEC, norm: PRE_RMS, pos: ['RoPE'],
        attn: ['GQA · 32 / 8 heads · full 32K attention', 'GQA · 32 / 8 个 head · 完整 32K attention'],
        ffn: ['MoE · 8 SwiGLU experts, top-2', 'MoE · 8 个 SwiGLU expert，选 2 个'],
        ctx: ['32K'],
        size: ['≈47B total · ≈13B active · 32 layers', '总参数约 47B · 激活约 13B · 32 层'],
        note: ['Mistral 7B with every FFN swapped for 8 experts. Each token uses 2 of them, so it costs about as much to run as a 13B dense model.', '在 Mistral 7B 的基础上，把每层 FFN 换成 8 个 expert。每个 token 只用其中 2 个，所以推理成本大约相当于 13B 的 dense 模型。']
      }
    },
    {
      id: 'gemma3', chip: ['Gemma 3'], name: 'Gemma 3 · 27B', year: '2025', layers: '62',
      cfg: { cross: 0, pre: 1, inner: 1, post: 0, pos: 0, fin: 1 }, norm: 'RMSNorm',
      attn: { sub: ['GQA · 32 / 16 · 5 local : 1 global', 'GQA · 32 / 16 · 5 层 local : 1 层 global'], kv: 8 }, tags: [['RoPE', '#tx-rope'], ['QK-norm', null], ['window 1,024', '#tx-windows']],
      ffn: { sub: ['GeGLU'], experts: 0 },
      stack: cycle(12, function (i) { return { a: i % 6 === 5 ? 'full' : 'local', f: 'dense' }; }), stackNote: ['62 layers: 5 local (window 1,024), then 1 global, repeating', '62 层：5 层 local（窗口 1,024）后接 1 层 global，循环往复'],
      spec: {
        layout: DEC,
        norm: ['Pre-norm and post-norm · RMSNorm', 'Pre-norm + post-norm · RMSNorm'],
        pos: ['RoPE · θ = 10k on local layers, 1M on global', 'RoPE · local 层 θ = 10k，global 层 θ = 1M'],
        attn: ['GQA · 32 / 16 heads · QK-norm · 5 local (window 1,024) : 1 global', 'GQA · 32 / 16 个 head · QK-norm · 5 层 local（窗口 1,024）: 1 层 global'],
        ffn: ['Dense · GeGLU'],
        ctx: ['128K'],
        size: ['27B: 62 layers, d 5,376 · also 1B, 4B, 12B', '27B：62 层，d 5,376 · 另有 1B、4B、12B'],
        note: ['Only one layer in six keeps a full-length KV cache, which is what makes 128K context affordable. QK-norm replaces the logit soft-capping of Gemma 2.', '每 6 层里只有 1 层保留完整长度的 KV cache，128K 上下文因此才负担得起。QK-norm 取代了 Gemma 2 的 logit soft-capping。']
      }
    },
    {
      id: 'qwen3', chip: ['Qwen3'], name: 'Qwen3 · 235B-A22B', year: '2025', layers: '94',
      cfg: { cross: 0, pre: 1, inner: 0, post: 0, pos: 0, fin: 1 }, norm: 'RMSNorm',
      attn: { sub: ['GQA · 64 query / 4 KV heads', 'GQA · 64 个 query / 4 个 KV head'], kv: 1 }, tags: [['RoPE', '#tx-rope'], ['QK-norm', null]],
      ffn: { sub: ['MoE · 128 experts, top-8', 'MoE · 128 个 expert，选 8 个'], experts: 128, k: 8 },
      stack: rep(12, 'full', 'moe'), stackNote: ['94 layers, every FFN is an MoE (12 shown)', '94 层，每层 FFN 都是 MoE（图中画 12 层）'],
      spec: {
        layout: DEC, norm: PRE_RMS,
        pos: ['RoPE · YaRN for long context', 'RoPE · 长上下文用 YaRN'],
        attn: ['GQA · 64 / 4 heads · QK-norm', 'GQA · 64 / 4 个 head · QK-norm'],
        ffn: ['MoE · 128 experts, top-8, no shared expert', 'MoE · 128 个 expert，选 8 个，没有 shared expert'],
        ctx: ['32K native · 131K with YaRN', '原生 32K · 用 YaRN 扩到 131K'],
        size: ['235B total · 22B active · 94 layers', '总参数 235B · 激活 22B · 94 层'],
        note: ['QK-norm replaces the QKV bias of Qwen2. The dense siblings (0.6B to 32B) use the same block with a SwiGLU FFN.', 'QK-norm 取代了 Qwen2 的 QKV bias。同系列的 dense 模型（0.6B 到 32B）用同样的 block，只是 FFN 换成 SwiGLU。']
      }
    },
    {
      id: 'deepseek', chip: ['DeepSeek-V3'], name: 'DeepSeek-V3', year: '2024', layers: '61',
      cfg: { cross: 0, pre: 1, inner: 0, post: 0, pos: 0, fin: 1 }, norm: 'RMSNorm',
      attn: { sub: ['MLA · 128 heads, latent KV', 'MLA · 128 个 head，latent KV'], kv: 0 }, tags: [['RoPE · decoupled', '#tx-rope']],
      ffn: { sub: ['MoE · 1 shared + 256 routed, top-8', 'MoE · 1 shared + 256 routed，选 8 个'], experts: 256, k: 8, shared: 1 },
      stack: cycle(12, function (i) { return { a: 'full', f: i < 3 ? 'dense' : 'moe' }; }), stackNote: ['61 layers: the first 3 keep a dense FFN, the rest are MoE', '61 层：前 3 层是 dense FFN，其余都是 MoE'],
      spec: {
        layout: DEC, norm: PRE_RMS,
        pos: ['RoPE on a separate 64-dim key', 'RoPE 作用在单独的 64 维 key 上'],
        attn: ['MLA · 128 heads · cached latent 512 + 64', 'MLA · 128 个 head · 缓存 latent 512 + 64'],
        ffn: ['MoE · 1 shared + 256 routed experts, top-8', 'MoE · 1 个 shared + 256 个 routed expert，选 8 个'],
        ctx: ['128K'],
        size: ['671B total · 37B active · 61 layers', '总参数 671B · 激活 37B · 61 层'],
        note: ['MLA caches one 576-number latent per token per layer instead of per-head keys and values. Load balancing uses a bias term rather than an auxiliary loss.', 'MLA 每层每个 token 只缓存一个 576 维的 latent，而不是每个 head 的 key 和 value。负载均衡靠 bias 项，而不是辅助 loss。']
      }
    },
    {
      id: 'gptoss', chip: ['gpt-oss'], name: 'gpt-oss-120b', year: '2025', layers: '36',
      cfg: { cross: 0, pre: 1, inner: 0, post: 0, pos: 0, fin: 1 }, norm: 'RMSNorm',
      attn: { sub: ['GQA · 64 / 8 · alternating window', 'GQA · 64 / 8 · 窗口层与全局层交替'], kv: 2 }, tags: [['RoPE', '#tx-rope'], ['window 128', '#tx-windows'], ['learned sinks', '#tx-windows']],
      ffn: { sub: ['MoE · 128 experts, top-4', 'MoE · 128 个 expert，选 4 个'], experts: 128, k: 4 },
      stack: cycle(12, function (i) { return { a: i % 2 === 0 ? 'local' : 'full', f: 'moe' }; }), stackNote: ['36 layers alternating a 128-token window and full attention', '36 层：128 token 窗口层与全局 attention 层交替'],
      spec: {
        layout: DEC, norm: PRE_RMS,
        pos: ['RoPE · YaRN to 131K', 'RoPE · 用 YaRN 扩到 131K'],
        attn: ['GQA · 64 / 8 heads · window 128 on alternate layers · learned sinks', 'GQA · 64 / 8 个 head · 隔层使用 128 窗口 · learned sinks'],
        ffn: ['MoE · 128 experts, top-4', 'MoE · 128 个 expert，选 4 个'],
        ctx: ['131K'],
        size: ['117B total · 5.1B active · 36 layers', '总参数 117B · 激活 5.1B · 36 层'],
        note: ['Every other layer sees only 128 tokens back. A learned per-head bias in the softmax denominator acts as an attention sink, so a head can choose to attend to nothing.', '每隔一层只回看 128 个 token。softmax 分母里每个 head 有一个可学习的 bias，起到 attention sink 的作用，让 head 可以选择“谁都不看”。']
      }
    }
  ];

  var SPEC_ROWS = [
    ['layout', 'Layout', '整体结构'], ['norm', 'Norm', 'Norm'], ['pos', 'Position', '位置编码'],
    ['attn', 'Attention', 'Attention'], ['ffn', 'FFN', 'FFN'], ['ctx', 'Context', '上下文长度'], ['size', 'Size', '规模']
  ];

  /* ── geometry ─────────────────────────────────────────────────────────── */
  var W = 552, BOTTOM = 790, CX = 335, BW = 230, SKIPX = 494, GAP = 16;
  var HEIGHTS = { tokens: 18, embed: 36, posadd: 22, sa: 64, ca: 46, ff: 64, fin: 24, head: 36, probs: 50, norm: 24, add: 22 };
  var GROUPS = ['sa', 'ca', 'ff'];
  var ORDER = ['tokens', 'embed', 'posadd', '<'].concat(
    GROUPS.reduce(function (list, g) { return list.concat([g + '_pre', g, g + '_in', g + '_add', g + '_post']); }, []),
    ['>', 'fin', 'head', 'probs']
  );

  function presence(key, p) {
    var parts = key.split('_'), scale = parts[0] === 'ca' ? p.cross : 1;
    if (parts.length === 2) return scale * (parts[1] === 'pre' ? p.pre : parts[1] === 'in' ? p.inner : parts[1] === 'post' ? p.post : 1);
    if (key === 'ca') return p.cross;
    if (key === 'posadd') return p.pos;
    if (key === 'fin') return p.fin;
    return 1;
  }
  function heightOf(key) {
    var parts = key.split('_');
    if (parts.length === 2) return parts[1] === 'add' ? HEIGHTS.add : HEIGHTS.norm;
    return HEIGHTS[key];
  }
  function layout(p) {
    var u = 0, pos = {};
    ORDER.forEach(function (key) {
      if (key === '<') { u += 18; pos.frame0 = u; u += 4; return; }
      if (key === '>') { u += 18; pos.frame1 = u; u += 6; return; }
      var pr = presence(key, p);
      u += GAP * pr;
      pos[key] = { u: u, h: heightOf(key), p: pr };
      u += heightOf(key) * pr;
    });
    pos.total = u;
    return pos;
  }

  /* ── build the SVG once ───────────────────────────────────────────────── */
  var svg = s('svg', { viewBox: '0 0 ' + W + ' 800', class: 'arch-svg', role: 'img', 'aria-label': 'Transformer block diagram' });
  svg.appendChild(s('defs', null, [
    s('marker', { id: 'arch-arrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto' }, [s('path', { d: 'M0 0 L8 4 L0 8 Z', class: 'a-arrowhead' })])
  ]));
  var frame = s('rect', { class: 'a-frame', rx: 12 });
  var frameLabel = s('text', { class: 's-sub a-frame-label', 'text-anchor': 'end' });
  var mainLine = s('line', { class: 'a-main', x1: CX, x2: CX });
  svg.appendChild(frame); svg.appendChild(frameLabel); svg.appendChild(mainLine);

  var els = {};   // key -> { g, h }
  function addEl(key, cls, href) {
    var g = s(href ? 'a' : 'g', { class: 'a-el ' + cls, href: href || null });
    svg.appendChild(g);
    els[key] = { g: g };
    return g;
  }
  function rectBox(g, w, hgt) { g.appendChild(s('rect', { x: -w / 2, y: 0, width: w, height: hgt, rx: 7 })); }
  function label(g, cls, y, en, zh) { var node = s('text', { class: cls, x: 0, y: y, 'text-anchor': 'middle' }); if (en != null) TX.bi(node, en, zh); g.appendChild(node); return node; }

  // tokens + embedding + position add
  var gTokens = addEl('tokens', 'a-plain');
  label(gTokens, 's-mono', 13, 'The   cat   sat   on   …');
  var gEmbed = addEl('embed', 'a-box a-emb');
  rectBox(gEmbed, BW, HEIGHTS.embed); label(gEmbed, 's-title', 23, 'Token embedding');
  var gPos = addEl('posadd', 'a-plain');
  gPos.appendChild(s('circle', { class: 'a-add', cx: 0, cy: 11, r: 10 }));
  gPos.appendChild(s('path', { class: 'a-plus', d: 'M-5 11 H5 M0 6 V16' }));
  gPos.appendChild(s('path', { class: 'a-wave', d: 'M-118 11 q7 -12 14 0 t14 0 t14 0 t14 0' }));
  gPos.appendChild(s('line', { class: 'a-side', x1: -60, y1: 11, x2: -12, y2: 11, 'marker-end': 'url(#arch-arrow)' }));
  var posText = s('text', { class: 's-sub', x: 16, y: 15 });
  gPos.appendChild(posText);

  // the three sub-layer groups
  var subText = {}, normTexts = [], skip = {};
  GROUPS.forEach(function (g) {
    skip[g] = { path: s('path', { class: 'a-skip', 'marker-end': 'url(#arch-arrow)' }), dot: s('circle', { class: 'a-dot', r: 4 }) };
    svg.appendChild(skip[g].path);
    ['pre', 'in', 'post'].forEach(function (slot) {
      var gn = addEl(g + '_' + slot, 'a-box a-norm');
      rectBox(gn, BW - 50, HEIGHTS.norm);
      normTexts.push(label(gn, 's-sub', 16));
    });
    var ga = addEl(g + '_add', 'a-plain');
    ga.appendChild(s('circle', { class: 'a-add', cx: 0, cy: 11, r: 10 }));
    ga.appendChild(s('path', { class: 'a-plus', d: 'M-5 11 H5 M0 6 V16' }));
  });
  var gSa = addEl('sa', 'a-box a-attn', '#tx-attention');
  rectBox(gSa, BW, HEIGHTS.sa); label(gSa, 's-title', 19, 'Self-attention (causal)', 'Self-attention（causal）');
  subText.sa = label(gSa, 's-sub', 34);
  var kvGlyph = s('g', { class: 'a-glyph' }); gSa.appendChild(kvGlyph);
  var gCa = addEl('ca', 'a-box a-attn');
  rectBox(gCa, BW, HEIGHTS.ca); label(gCa, 's-title', 19, 'Cross-attention');
  label(gCa, 's-sub', 34, 'queries from here, K/V from the encoder', 'query 来自 decoder，K/V 来自 encoder');
  var gFf = addEl('ff', 'a-box a-ffn', '#tx-moe');
  rectBox(gFf, BW, HEIGHTS.ff); label(gFf, 's-title', 19, 'Feed-forward network', 'Feed-forward network（FFN）');
  subText.ff = label(gFf, 's-sub', 34);
  var ffGlyph = s('g', { class: 'a-glyph' }); gFf.appendChild(ffGlyph);

  var gFin = addEl('fin', 'a-box a-norm');
  rectBox(gFin, BW - 50, HEIGHTS.fin); var finText = label(gFin, 's-sub', 16);
  var gHead = addEl('head', 'a-box a-emb');
  rectBox(gHead, BW, HEIGHTS.head); label(gHead, 's-title', 23, 'Linear → softmax');
  var gProbs = addEl('probs', 'a-plain');
  [[-40, 8], [-20, 22], [0, 30], [20, 12], [40, 5]].forEach(function (bar) {
    gProbs.appendChild(s('rect', { class: 'a-prob', x: bar[0] - 7, y: 32 - bar[1], width: 14, height: bar[1], rx: 2 }));
  });
  label(gProbs, 's-sub', 47, 'next-token probabilities', '下一个 token 的概率');

  // encoder (original Transformer only) and the attention tags
  var gEnc = s('g', { class: 'a-el a-box a-enc' });
  gEnc.appendChild(s('rect', { x: 0, y: 0, width: 150, height: 74, rx: 7 }));
  [['s-title', 24, 'Encoder × 6', 'Encoder × 6'], ['s-sub', 42, 'self-attention + FFN', 'self-attention + FFN'], ['s-sub', 58, 'over the source sentence', '处理源语言句子']].forEach(function (row) {
    gEnc.appendChild(TX.bi(s('text', { class: row[0], x: 75, y: row[1], 'text-anchor': 'middle' }), row[2], row[3]));
  });
  var encArrow = s('line', { class: 'a-side', 'marker-end': 'url(#arch-arrow)' });
  var encLabel = s('text', { class: 's-sub', 'text-anchor': 'middle', text: 'K, V' });
  svg.appendChild(gEnc); svg.appendChild(encArrow); svg.appendChild(encLabel);
  var gTags = s('g', { class: 'a-tags' });
  svg.appendChild(gTags);
  var pulse = s('circle', { class: 'a-dot a-pulse', r: 5, cx: CX });
  GROUPS.forEach(function (g) { svg.appendChild(skip[g].dot); });
  svg.appendChild(pulse);

  /* ── side panel ───────────────────────────────────────────────────────── */
  var side = h('div', { class: 'arch-side' });
  fig.stage.appendChild(h('div', { class: 'arch-layout' }, [h('div', { class: 'fig-scroll arch-diagram' }, [svg]), side]));
  fig.foot.appendChild(h('p', { class: 'fig-note', bi: ['The dot is one token flowing up the residual stream. Click the attention or FFN block to jump to its section.', '圆点是一个 token 沿残差流向上流动。点击 attention 或 FFN 模块，可以跳到对应章节。'] }));

  /* ── state ────────────────────────────────────────────────────────────── */
  var current = FAMILIES[2], previous = null;
  var cur = copy(current.cfg), from = null, tweenStart = 0, pos = layout(cur), pulseU = -10, activeKey = null;
  var rand = TX.rng(7);
  function copy(o) { var out = {}; Object.keys(o).forEach(function (k) { out[k] = o[k]; }); return out; }
  function Y(u, hgt) { return pos.bottom - u - (hgt || 0); }

  function drawGlyphs() {
    TX.clear(kvGlyph);
    var i, n = 16, step = 12, x0 = -(n - 1) * step / 2;
    for (i = 0; i < n; i++) kvGlyph.appendChild(s('circle', { class: 'g-q', cx: x0 + i * step, cy: 45, r: 3 }));
    var kv = current.attn.kv;
    if (kv === 0) kvGlyph.appendChild(s('rect', { class: 'g-k', x: -30, y: 52, width: 60, height: 6, rx: 3 }));
    else for (i = 0; i < kv; i++) kvGlyph.appendChild(s('circle', { class: 'g-k', cx: x0 + ((i + 0.5) * n / kv - 0.5) * step, cy: 55, r: 3 }));
    drawExperts();
  }
  function drawExperts() {
    TX.clear(ffGlyph);
    var f = current.ffn;
    if (!f.experts) { ffGlyph.appendChild(s('rect', { class: 'g-k', x: -70, y: 45, width: 140, height: 9, rx: 4 })); return; }
    var n = f.experts <= 8 ? 8 : 16, k = f.experts <= 8 ? f.k : 1, step = 13, chosen = {};
    while (Object.keys(chosen).length < k) chosen[Math.floor(rand() * n)] = true;
    var x0 = -((n - 1) * step) / 2 + (f.shared ? 10 : 0);
    if (f.shared) ffGlyph.appendChild(s('rect', { class: 'g-k g-shared', x: x0 - 26, y: 44, width: 10, height: 10, rx: 2 }));
    for (var i = 0; i < n; i++) ffGlyph.appendChild(s('rect', { class: chosen[i] ? 'g-k' : 'g-off', x: x0 + i * step - 5, y: 44, width: 10, height: 10, rx: 2 }));
  }
  function drawTags() {
    TX.clear(gTags);
    current.tags.forEach(function (tag, i) {
      var w = tag[0].length * 6.4 + 18;
      var g = s(tag[1] ? 'a' : 'g', { class: 'a-tag', href: tag[1], transform: 'translate(' + (-w) + ',' + (i * 25) + ')' });
      g.appendChild(s('rect', { x: 0, y: 0, width: w, height: 20, rx: 10 }));
      g.appendChild(s('text', { class: 's-tag', x: w / 2, y: 14, 'text-anchor': 'middle', text: tag[0] }));
      gTags.appendChild(g);
    });
  }

  function applyLayout() {
    pos = layout(cur);
    pos.bottom = BOTTOM;
    // Crop the viewBox to the content so short stacks are not drawn small inside a tall box.
    svg.setAttribute('viewBox', '0 ' + (BOTTOM - pos.total - 14).toFixed(1) + ' ' + W + ' ' + (pos.total + 26).toFixed(1));
    Object.keys(els).forEach(function (key) {
      var e = pos[key];
      els[key].g.setAttribute('transform', 'translate(' + CX + ',' + Y(e.u, e.h).toFixed(1) + ')');
      els[key].g.setAttribute('opacity', e.p.toFixed(3));
      els[key].g.setAttribute('visibility', e.p < 0.02 ? 'hidden' : 'visible');
    });
    mainLine.setAttribute('y1', Y(pos.tokens.u + pos.tokens.h + 3));
    mainLine.setAttribute('y2', Y(pos.probs.u - 4));
    var fy = Y(pos.frame1), fx = CX - BW / 2 - 20;
    frame.setAttribute('x', fx); frame.setAttribute('y', fy);
    frame.setAttribute('width', SKIPX + 20 - fx); frame.setAttribute('height', pos.frame1 - pos.frame0);
    frameLabel.setAttribute('x', SKIPX + 20); frameLabel.setAttribute('y', fy - 7);
    GROUPS.forEach(function (g) {
      var pre = pos[g + '_pre'], add = pos[g + '_add'];
      var ys = Y(pre.u - GAP * pre.p + GAP / 2), ya = Y(add.u + 11), r = Math.min(10, Math.abs(ys - ya) / 2);
      skip[g].path.setAttribute('d', 'M' + CX + ' ' + ys + ' H' + (SKIPX - r) + ' Q' + SKIPX + ' ' + ys + ' ' + SKIPX + ' ' + (ys - r) + ' V' + (ya + r) + ' Q' + SKIPX + ' ' + ya + ' ' + (SKIPX - r) + ' ' + ya + ' H' + (CX + 13));
      skip[g].path.setAttribute('opacity', add.p.toFixed(3));
      skip[g].ys = pre.u - GAP * pre.p + GAP / 2; skip[g].ya = add.u + 11;
    });
    var caMid = Y(pos.ca.u + pos.ca.h / 2);
    gEnc.setAttribute('transform', 'translate(12,' + (caMid - 37).toFixed(1) + ')');
    encArrow.setAttribute('x1', 162); encArrow.setAttribute('x2', CX - BW / 2 - 2);
    encArrow.setAttribute('y1', caMid); encArrow.setAttribute('y2', caMid);
    encLabel.setAttribute('x', 190); encLabel.setAttribute('y', caMid - 7);
    [gEnc, encArrow, encLabel].forEach(function (node) { node.setAttribute('opacity', cur.cross.toFixed(3)); node.setAttribute('visibility', cur.cross < 0.02 ? 'hidden' : 'visible'); });
    var tagsH = current.tags.length * 25 - 5;
    gTags.setAttribute('transform', 'translate(' + (CX - BW / 2 - 32) + ',' + (Y(pos.sa.u + pos.sa.h / 2) - tagsH / 2).toFixed(1) + ')');
  }

  function drawPulse() {
    var inside = null;
    Object.keys(els).forEach(function (key) {
      var e = pos[key];
      if (e.p > 0.5 && key !== 'tokens' && key !== 'probs' && pulseU >= e.u && pulseU <= e.u + e.h) inside = key;
    });
    if (inside !== activeKey) {
      if (activeKey) els[activeKey].g.classList.remove('on');
      if (inside) els[inside].g.classList.add('on');
      if (inside === 'ff' && current.ffn.experts) drawExperts();
      activeKey = inside;
    }
    pulse.setAttribute('cy', Y(pulseU));
    pulse.setAttribute('opacity', inside || pulseU < pos.tokens.u + pos.tokens.h || pulseU > pos.probs.u ? 0 : 1);
    GROUPS.forEach(function (g) {
      var sk = skip[g], dot = sk.dot, on = pos[g + '_add'].p > 0.5 && pulseU > sk.ys && pulseU < sk.ya;
      dot.setAttribute('opacity', on ? 0.85 : 0);
      if (!on) return;
      var point = sk.path.getPointAtLength((pulseU - sk.ys) / (sk.ya - sk.ys) * sk.path.getTotalLength());
      dot.setAttribute('cx', point.x); dot.setAttribute('cy', point.y);
    });
  }

  var ticker = TX.loop(function (now, dt) {
    if (from) {
      var x = TX.clamp((now - tweenStart) / 650, 0, 1), k = TX.ease(x);
      Object.keys(cur).forEach(function (key) { cur[key] = TX.lerp(from[key], current.cfg[key], k); });
      if (x === 1) from = null;
      applyLayout();
    }
    pulseU += dt * 0.085;
    if (pulseU > pos.total + 30) pulseU = -10;
    drawPulse();
  });

  function renderSide() {
    TX.clear(side);
    side.appendChild(h('p', { class: 'arch-name' }, [h('strong', { text: current.name }), h('span', { text: current.year })]));
    var dl = h('dl', { class: 'arch-spec' });
    SPEC_ROWS.forEach(function (row) {
      var value = current.spec[row[0]];
      var changed = previous && previous.spec[row[0]][0] !== value[0];
      dl.appendChild(h('div', { class: changed ? 'changed' : null }, [h('dt', { bi: [row[1], row[2]] }), h('dd', { bi: value })]));
    });
    side.appendChild(dl);
    var cells = h('div', { class: 'stack-cells', role: 'img', 'aria-label': current.stackNote[0] });
    current.stack.forEach(function (layer) {
      cells.appendChild(h('span', { class: 'stack-col' }, [h('i', { class: 'st-f st-' + layer.f }), h('i', { class: 'st-a st-' + layer.a })]));
    });
    side.appendChild(h('div', { class: 'arch-stack' }, [
      h('p', { class: 'ro-label', bi: ['Layer stack · bottom row attention, top row FFN', '层堆叠 · 下排 attention，上排 FFN'] }),
      cells,
      h('p', { class: 'stack-legend' }, [
        h('i', { class: 'st-a st-full' }), h('span', { bi: ['full', '全局'] }), h('i', { class: 'st-a st-local' }), h('span', { bi: ['windowed', '窗口'] }),
        h('i', { class: 'st-f st-dense' }), h('span', { bi: ['dense', 'dense'] }), h('i', { class: 'st-f st-moe' }), h('span', { bi: ['MoE', 'MoE'] })
      ]),
      h('p', { class: 'stack-note', bi: current.stackNote })
    ]));
    side.appendChild(h('p', { class: 'arch-note', bi: current.spec.note }));
  }

  function select(id, instant) {
    var next = FAMILIES.filter(function (f) { return f.id === id; })[0];
    if (!next) return;
    previous = current === next ? previous : current;
    current = next;
    chips.set(id);
    normTexts.forEach(function (node) { node.textContent = current.norm; });
    finText.textContent = 'Final ' + current.norm;
    TX.bi(subText.sa, current.attn.sub[0], current.attn.sub[1]);
    TX.bi(subText.ff, current.ffn.sub[0], current.ffn.sub[1]);
    if (current.posLabel) TX.bi(posText, current.posLabel[0], current.posLabel[1]);
    TX.bi(frameLabel, '× ' + current.layers + ' layers', '× ' + current.layers + ' 层');
    drawGlyphs(); drawTags(); renderSide();
    if (instant || TX.reduced) { cur = copy(current.cfg); from = null; applyLayout(); drawPulse(); }
    else { from = copy(cur); tweenStart = performance.now(); }
  }

  var chips = TX.seg(FAMILIES.map(function (f) { return { id: f.id, en: f.chip[0], zh: f.chip[1] }; }), current.id, function (id) { select(id); }, 'Model family');
  fig.controls.appendChild(chips.el);

  select(current.id, true);
  previous = null;
  if (!TX.reduced) TX.whenVisible(fig.root, ticker.start, ticker.stop);
  TX.selectFamily = select;
})();

/* 02 · Self-attention, step by step. A six-token toy example with hand-set
   Q/K/V (non-negative, d_k = 4) so the whole pipeline is real arithmetic. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-attention', { title: ['Toy example · 6 tokens, d_k = 4, hand-set weights', '玩具示例 · 6 个 token，d_k = 4，权重为手工设定'] });
  if (!fig) return;
  var s = TX.s, h = TX.h;

  var TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
  // dims ≈ [noun, verb, function word, generic]. Keys say what a token offers, queries what it looks for.
  var K = [[0.1, 0, 1.0, 0.3], [1.6, 0, 0, 0.2], [0, 1.6, 0.1, 0.2], [0, 0.2, 1.3, 0.2], [0.1, 0, 1.0, 0.3], [1.6, 0, 0, 0.2]];
  var Q = [[0.3, 0.3, 0.3, 0.6], [0.2, 0, 2.6, 0.4], [2.8, 0.2, 0, 0.3], [0.3, 2.8, 0, 0.3], [0.2, 0.5, 2.4, 0.3], [0.4, 1.5, 2.2, 0.4]];
  var V = [[0.2, 0.1, 0.9, 0.3], [1.0, 0.1, 0.1, 0.6], [0.1, 1.0, 0.2, 0.4], [0.1, 0.4, 0.8, 0.1], [0.2, 0.1, 0.9, 0.5], [0.9, 0.2, 0.1, 0.8]];
  var N = TOKENS.length, D = 4;

  var STEPS = [
    { en: 'Tokens', zh: 'Token', cap: ['Six tokens come in. Each one already carries a vector from the layers below.', '六个 token 进来，每个都带着下层传上来的向量。'] },
    { en: 'Project', zh: '投影', cap: ['Three learned matrices turn every token into a query, a key and a value. Here each has 4 numbers; darker means larger.', '三个可学习的矩阵把每个 token 变成 query、key 和 value。这里每个向量 4 个数，颜色越深数值越大。'] },
    { en: 'Score', zh: '打分', cap: ['Every query meets every key: score = q·k / √d_k. A high score means “this key is what I am looking for”.', '每个 query 和每个 key 做一次比较：score = q·k / √d_k。分数高，意思是“这个 key 正是我要找的”。'] },
    { en: 'Mask', zh: 'Mask', cap: ['Causal mask: a token may not look at anything after it, so those scores become −∞. Turn the mask off and you get a BERT-style encoder.', 'Causal mask：token 不能看它后面的东西，所以这些分数被设成 −∞。关掉 mask，就得到 BERT 式的 encoder。'] },
    { en: 'Softmax', zh: 'Softmax', cap: ['Softmax turns every row of scores into weights that sum to 1.', 'Softmax 把每一行分数变成和为 1 的权重。'] },
    { en: 'Mix', zh: '加权求和', cap: ['Each output row is the weighted average of the value vectors. That row is added back to the token’s residual stream.', '每一行输出都是 value 向量的加权平均，这一行会被加回该 token 的残差流。'] }
  ];

  var step = 0, focus = 2, causal = true, autoplay = null;

  /* ── math ─────────────────────────────────────────────────────────────── */
  function dot(a, b) { return a.reduce(function (sum, x, i) { return sum + x * b[i]; }, 0); }
  var RAW = Q.map(function (q) { return K.map(function (k) { return dot(q, k) / Math.sqrt(D); }); });
  var RAW_MAX = Math.max.apply(null, RAW.map(function (row) { return Math.max.apply(null, row); }));
  function weights() {
    return RAW.map(function (row, i) {
      var exps = row.map(function (x, j) { return causal && j > i ? 0 : Math.exp(x); });
      var total = exps.reduce(function (a, b) { return a + b; }, 0);
      return exps.map(function (x) { return x / total; });
    });
  }
  function outputs(wts) {
    return wts.map(function (row) {
      var out = [0, 0, 0, 0];
      row.forEach(function (w, j) { for (var d = 0; d < D; d++) out[d] += w * V[j][d]; });
      return out;
    });
  }

  /* ── build ────────────────────────────────────────────────────────────── */
  var ROW = 34, Y0 = 72, SMALL = 22, BIG = 36;
  var GX = { q: 86, k: 206, s: 336, v: 596, o: 716 };
  var svg = s('svg', { viewBox: '0 0 820 300', class: 'attn-svg', role: 'img', 'aria-label': 'Self-attention computed step by step' });
  var bands = [], rowLabels = [], grids = {};

  TOKENS.forEach(function (tok, i) {
    var band = s('rect', { class: 'at-band', x: 8, y: Y0 + i * ROW - 3, width: 804, height: ROW - 2, rx: 6 });
    svg.appendChild(band); bands.push(band);
  });
  TOKENS.forEach(function (tok, i) {
    var text = s('text', { class: 's-title at-tok', x: 72, y: Y0 + i * ROW + 19, 'text-anchor': 'end', text: tok });
    svg.appendChild(text); rowLabels.push(text);
  });

  function header(x, w, title, cls, sub) {
    svg.appendChild(s('text', { class: 's-title ' + cls, x: x + w / 2, y: 34, 'text-anchor': 'middle', text: title }));
    var node = s('text', { class: 's-sub', x: x + w / 2, y: 50, 'text-anchor': 'middle' });
    TX.bi(node, sub[0], sub[1]); svg.appendChild(node);
  }
  function grid(key, x, cols, size, cls) {
    var g = s('g', { class: 'at-grid ' + cls }), cells = [];
    for (var i = 0; i < N; i++) {
      cells.push([]);
      for (var j = 0; j < cols; j++) {
        var rect = s('rect', { class: 'at-cell', x: x + j * size, y: Y0 + i * ROW, width: size - 3, height: ROW - 8, rx: 4 });
        g.appendChild(rect); cells[i].push({ rect: rect });
      }
    }
    svg.appendChild(g);
    grids[key] = { g: g, cells: cells };
    return g;
  }
  header(GX.q, 4 * SMALL, 'Q', 'fill-q', ['queries', 'query']);
  header(GX.k, 4 * SMALL, 'K', 'fill-k', ['keys', 'key']);
  header(GX.v, 4 * SMALL, 'V', 'fill-v', ['values', 'value']);
  header(GX.o, 4 * SMALL, 'Out', 'fill-v', ['weights × V', '权重 × V']);
  var sTitle = s('text', { class: 's-title', x: GX.s + 3 * BIG, y: 34, 'text-anchor': 'middle' });
  svg.appendChild(sTitle);
  TOKENS.forEach(function (tok, j) {
    svg.appendChild(s('text', { class: 's-sub at-col', x: GX.s + j * BIG + (BIG - 3) / 2, y: 62, 'text-anchor': 'middle', text: tok }));
  });
  grid('q', GX.q, 4, SMALL, 'is-q'); grid('k', GX.k, 4, SMALL, 'is-k');
  grid('s', GX.s, N, BIG, 'is-q'); grid('v', GX.v, 4, SMALL, 'is-v'); grid('o', GX.o, 4, SMALL, 'is-v');
  grids.s.cells.forEach(function (row, i) {
    row.forEach(function (cell, j) {
      cell.text = s('text', { class: 'at-num', x: GX.s + j * BIG + (BIG - 3) / 2, y: Y0 + i * ROW + 17, 'text-anchor': 'middle' });
      grids.s.g.appendChild(cell.text);
    });
  });
  [[GX.q + 4 * SMALL + 14, '·'], [GX.k + 4 * SMALL + 20, '→'], [GX.s + N * BIG + 20, '×'], [GX.v + 4 * SMALL + 14, '=']].forEach(function (op) {
    svg.appendChild(s('text', { class: 'at-op', x: op[0], y: Y0 + 3 * ROW + 2, 'text-anchor': 'middle', text: op[1] }));
  });
  // hover targets, one per row, on top of everything
  TOKENS.forEach(function (tok, i) {
    svg.appendChild(s('rect', { class: 'at-hit', x: 0, y: Y0 + i * ROW - 4, width: 820, height: ROW, on: { pointerenter: function () { setFocus(i); }, click: function () { setFocus(i); } } }));
  });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));

  /* ── controls ─────────────────────────────────────────────────────────── */
  var stepper = TX.seg(STEPS.map(function (st, i) { return { id: String(i), en: (i + 1) + ' ' + st.en, zh: (i + 1) + ' ' + st.zh }; }), '0', function (id) { stop(); go(+id); }, 'Step');
  var maskSeg = TX.seg([{ id: 'on', en: 'Causal mask', zh: 'Causal mask' }, { id: 'off', en: 'No mask (encoder)', zh: '无 mask（encoder）' }], 'on', function (id) { causal = id === 'on'; render(); }, 'Mask');
  var playBtn = TX.button('▶ Play', '▶ 播放', function () { if (autoplay) stop(); else { if (step === STEPS.length - 1) go(0); play(); } });
  fig.controls.appendChild(stepper.el); fig.controls.appendChild(playBtn); fig.controls.appendChild(maskSeg.el);
  var caption = h('p', { class: 'fig-caption' }), insight = h('p', { class: 'fig-insight' });
  fig.foot.appendChild(caption); fig.foot.appendChild(insight);

  /* ── render ───────────────────────────────────────────────────────────── */
  function paint(cell, opacity, delay) {
    cell.rect.style.transitionDelay = (TX.reduced ? 0 : delay) + 'ms';
    cell.rect.style.fillOpacity = opacity;
  }
  function render() {
    var wts = weights(), outs = outputs(wts);
    TX.bi(sTitle, step >= 4 ? 'attention weights' : 'scores  q·k / √d_k', step >= 4 ? 'attention 权重' : '分数  q·k / √d_k');
    [['q', Q, 1], ['k', K, 1], ['v', V, 1], ['o', outs, 5]].forEach(function (spec) {
      var visible = step >= spec[2];
      grids[spec[0]].g.classList.toggle('is-empty', !visible);
      grids[spec[0]].cells.forEach(function (row, i) {
        row.forEach(function (cell, j) { paint(cell, visible ? 0.12 + 0.88 * Math.min(1, spec[1][i][j] / (spec[0] === 'q' ? 2.8 : 1.6)) : 0, i * 45); });
      });
    });
    grids.s.g.classList.toggle('is-empty', step < 2);
    grids.s.cells.forEach(function (row, i) {
      row.forEach(function (cell, j) {
        var masked = causal && j > i && step >= 3, prob = step >= 4;
        var value = prob ? wts[i][j] : RAW[i][j];
        var shade = step < 2 || masked ? 0 : prob ? value : value / RAW_MAX;
        paint(cell, shade * 0.92, i * 45 + j * 12);
        cell.rect.classList.toggle('is-masked', masked);
        cell.text.textContent = step < 2 ? '' : masked ? '−∞' : prob ? value.toFixed(2).replace(/^0/, '') : value.toFixed(1);
        cell.text.setAttribute('class', 'at-num' + (shade > 0.55 ? ' inv' : '') + (masked ? ' muted' : ''));
      });
    });
    // focus row: Q / scores / output rows light up; K and V rows glow by how much they are heard
    bands.forEach(function (band, i) { band.classList.toggle('on', i === focus); });
    rowLabels.forEach(function (node, i) { node.classList.toggle('on', i === focus); });
    ['k', 'v'].forEach(function (key) {
      grids[key].cells.forEach(function (row, j) {
        var heard = step >= 4 ? 0.25 + 0.75 * Math.min(1, wts[focus][j] * 2.2) : (causal && step >= 3 && j > focus ? 0.25 : 1);
        row.forEach(function (cell) { cell.rect.style.opacity = heard; });
      });
    });
    stepper.set(String(step));
    TX.bi(caption, STEPS[step].cap[0], STEPS[step].cap[1]);
    var best = 0;
    wts[focus].forEach(function (w, j) { if (w > wts[focus][best]) best = j; });
    var me = '“' + TOKENS[focus] + '”', them = '“' + TOKENS[best] + '”';
    if (step < 2) TX.bi(insight, 'Hover a row to follow one token through the computation.', '把鼠标放到某一行上，跟着这个 token 走完整个计算。');
    else if (step < 4) TX.bi(insight, 'Row ' + me + ': its largest score is with ' + them + ' (' + RAW[focus][best].toFixed(2) + ').', me + ' 这一行：分数最高的是 ' + them + '（' + RAW[focus][best].toFixed(2) + '）。');
    else TX.bi(insight, me + ' puts ' + Math.round(wts[focus][best] * 100) + '% of its attention on ' + them + (step === 5 ? ', so its output row looks most like the value of ' + them + '.' : '.'),
      me + ' 把 ' + Math.round(wts[focus][best] * 100) + '% 的注意力放在 ' + them + ' 上' + (step === 5 ? '，所以它的输出最像 ' + them + ' 的 value。' : '。'));
  }

  function setFocus(i) { if (i !== focus) { focus = i; render(); } }
  function go(next) { step = TX.clamp(next, 0, STEPS.length - 1); render(); }
  function play() {
    TX.bi(playBtn, '❚❚ Pause', '❚❚ 暂停');
    autoplay = window.setInterval(function () { if (step >= STEPS.length - 1) stop(); else go(step + 1); }, 1900);
  }
  function stop() {
    if (autoplay) window.clearInterval(autoplay);
    autoplay = null;
    TX.bi(playBtn, '▶ Play', '▶ 播放');
  }

  render();
  var started = false;
  TX.whenVisible(fig.root, function () { if (!started && !TX.reduced) { started = true; play(); } }, function () { });
})();

/* 03 · KV cache. Decode ten tokens with and without a cache and count the work. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-kv-cache', { title: ['Decoding 10 tokens · what gets computed at each step', '解码 10 个 token · 每一步到底算了什么'] });
  if (!fig) return;
  var s = TX.s, h = TX.h;

  var TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'purred', 'softly', '.'];
  var N = TOKENS.length, CELL = 22, GX = 96, GY = 58, SX = 410, SLOT = 38;
  var t = 1, cached = true;

  var svg = s('svg', { viewBox: '0 0 820 310', class: 'kv-svg', role: 'img', 'aria-label': 'Attention work per decoding step, with and without a KV cache' });
  svg.appendChild(TX.bi(s('text', { class: 's-title', x: GX + N * CELL / 2, y: 24, 'text-anchor': 'middle' }), 'attention scores (query × key)', 'attention 分数（query × key）'));
  svg.appendChild(TX.bi(s('text', { class: 's-sub', x: GX + N * CELL / 2, y: 42, 'text-anchor': 'middle' }), 'rows: query position · columns: key position', '行：query 位置 · 列：key 位置'));
  var cells = [], rowText = [];
  for (var i = 0; i < N; i++) {
    cells.push([]);
    rowText.push(s('text', { class: 's-sub kv-row', x: GX - 8, y: GY + i * CELL + 15, 'text-anchor': 'end', text: TOKENS[i] }));
    svg.appendChild(rowText[i]);
    for (var j = 0; j <= i; j++) {
      var rect = s('rect', { class: 'kv-cell', x: GX + j * CELL, y: GY + i * CELL, width: CELL - 3, height: CELL - 3, rx: 3 });
      svg.appendChild(rect); cells[i].push(rect);
    }
  }

  var cacheTitle = s('text', { class: 's-title', x: SX + N * SLOT / 2, y: 24, 'text-anchor': 'middle' });
  var cacheSub = s('text', { class: 's-sub', x: SX + N * SLOT / 2, y: 42, 'text-anchor': 'middle' });
  svg.appendChild(cacheTitle); svg.appendChild(cacheSub);
  var slots = { tok: [], k: [], v: [] };
  ['K', 'V'].forEach(function (name, r) {
    svg.appendChild(s('text', { class: 's-title ' + (r ? 'fill-v' : 'fill-k'), x: SX - 14, y: 138 + r * 44 + 21, 'text-anchor': 'end', text: name }));
  });
  for (var c = 0; c < N; c++) {
    slots.tok.push(s('text', { class: 's-sub kv-tok', x: SX + c * SLOT + (SLOT - 4) / 2, y: 118, 'text-anchor': 'middle', text: TOKENS[c] }));
    slots.k.push(s('rect', { class: 'kv-slot is-k', x: SX + c * SLOT, y: 138, width: SLOT - 4, height: 32, rx: 5 }));
    slots.v.push(s('rect', { class: 'kv-slot is-v', x: SX + c * SLOT, y: 182, width: SLOT - 4, height: 32, rx: 5 }));
    svg.appendChild(slots.tok[c]); svg.appendChild(slots.k[c]); svg.appendChild(slots.v[c]);
  }
  var legend = s('g', { transform: 'translate(' + SX + ',250)' });
  [['kv-slot is-k hot', 0, 'computed this step', '这一步计算'], ['kv-slot is-k stored', 170, 'read from the cache', '从 cache 读取']].forEach(function (item) {
    legend.appendChild(s('rect', { class: item[0], x: item[1], y: 0, width: 16, height: 16, rx: 4 }));
    legend.appendChild(TX.bi(s('text', { class: 's-sub', x: item[1] + 23, y: 12.5 }), item[2], item[3]));
  });
  svg.appendChild(legend);
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));

  /* ── controls + readouts ──────────────────────────────────────────────── */
  var mode = TX.seg([{ id: 'off', en: 'No cache', zh: '没有 cache' }, { id: 'on', en: 'With KV cache', zh: '有 KV cache' }], 'on', function (id) { cached = id === 'on'; render(); }, 'Mode');
  var playing = !TX.reduced;
  var playBtn = TX.button('', '', function () { playing = !playing; if (playing) ticker.start(); else ticker.stop(); label(); });
  var stepBtn = TX.button('Step →', '单步 →', function () { playing = false; ticker.stop(); label(); advance(); });
  function label() { TX.bi(playBtn, playing ? '❚❚ Pause' : '▶ Play', playing ? '❚❚ 暂停' : '▶ 播放'); }
  label();
  fig.controls.appendChild(mode.el); fig.controls.appendChild(playBtn); fig.controls.appendChild(stepBtn);

  var roStep = TX.readout('Step', '当前步'), roKv = TX.readout('K/V projections this step', '本步 K/V 投影次数'), roScores = TX.readout('Scores this step', '本步分数个数');
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roStep.el, roKv.el, roScores.el]));
  var bars = {};
  var barWrap = h('div', { class: 'mini-bars' }, [h('p', { class: 'ro-label', bi: ['Total scores computed so far', '到目前为止累计计算的分数个数'] })]);
  [['off', 'No cache', '没有 cache'], ['on', 'With KV cache', '有 KV cache']].forEach(function (row) {
    var fill = h('span', { class: 'mini-fill' }), value = h('strong', { class: 'mini-val' });
    bars[row[0]] = { fill: fill, value: value, row: h('div', { class: 'mini-row' }, [h('span', { class: 'mini-label', bi: [row[1], row[2]] }), h('span', { class: 'mini-track' }, [fill]), value]) };
    barWrap.appendChild(bars[row[0]].row);
  });
  fig.foot.appendChild(barWrap);

  function render() {
    for (var i = 0; i < N; i++) {
      rowText[i].classList.toggle('on', i === t - 1);
      for (var j = 0; j <= i; j++) {
        var state = i >= t ? '' : (!cached || i === t - 1) ? 'hot' : 'done';
        cells[i][j].setAttribute('class', 'kv-cell ' + state);
      }
    }
    for (var c = 0; c < N; c++) {
      var slot = c >= t ? '' : (!cached || c === t - 1) ? 'hot' : 'stored';
      slots.k[c].setAttribute('class', 'kv-slot is-k ' + slot);
      slots.v[c].setAttribute('class', 'kv-slot is-v ' + slot);
      slots.tok[c].setAttribute('class', 's-sub kv-tok' + (c < t ? ' on' : ''));
    }
    TX.bi(cacheTitle, cached ? 'KV cache' : 'keys and values (nothing is kept)', cached ? 'KV cache' : 'key 与 value（什么都不保留）');
    TX.bi(cacheSub, cached ? 'grows by one K/V pair per token' : 'recomputed from scratch at every step', cached ? '每个 token 增加一对 K/V' : '每一步都从头重算');
    roStep.set(t + ' / ' + N + '  ·  “' + TOKENS[t - 1] + '”');
    roKv.set(cached ? '1' : String(t));
    roScores.set(String(cached ? t : t * (t + 1) / 2));
    var totals = { off: t * (t + 1) * (t + 2) / 6, on: t * (t + 1) / 2 }, max = N * (N + 1) * (N + 2) / 6;
    Object.keys(bars).forEach(function (key) {
      bars[key].fill.style.width = (totals[key] / max * 100).toFixed(1) + '%';
      bars[key].value.textContent = TX.fmtInt(totals[key]);
      bars[key].row.classList.toggle('on', (key === 'on') === cached);
    });
  }
  function advance() { t = t >= N ? 1 : t + 1; render(); }

  var elapsed = 0;
  var ticker = TX.loop(function (now, dt) {
    elapsed += dt;
    var wait = t === N ? 2200 : 850;
    if (elapsed >= wait) { elapsed = 0; advance(); }
  });
  render();
  TX.whenVisible(fig.root, function () { if (playing) ticker.start(); }, function () { ticker.stop(); });
})();

/* 04 · MHA → GQA → MQA → MLA. Eight query heads; the K/V heads slide together
   as they are shared, and a calculator turns the idea into gigabytes. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-kv-heads', { title: ['How many K/V heads does each token have to cache?', '每个 token 要缓存多少个 K/V head？'] });
  if (!fig) return;
  var s = TX.s, h = TX.h;

  var H = 8, X0 = 46, PITCH = 94, QW = 72, QY = 26, KVY = 128, LATY = 232;
  var MODES = {
    mha: { groups: 8, en: 'MHA', cache: ['8 × (K + V) = 16 head-vectors per token, per layer. The baseline.', '每层每个 token 缓存 8 ×（K + V）= 16 个 head 向量。这是基线。'] },
    gqa4: { groups: 4, en: 'GQA-4', cache: ['4 × (K + V): half the cache. Every pair of query heads shares one K/V head.', '4 ×（K + V）：cache 减半。每两个 query head 共享一个 K/V head。'] },
    gqa2: { groups: 2, en: 'GQA-2', cache: ['2 × (K + V): a quarter of the cache, and quality stays close to MHA. This 4 : 1 ratio is what Llama 3 8B and Mistral 7B use (32 query heads, 8 KV heads).', '2 ×（K + V）：cache 只有四分之一，质量仍接近 MHA。Llama 3 8B 和 Mistral 7B 用的就是这个 4 : 1 的比例（32 个 query head，8 个 KV head）。'] },
    mqa: { groups: 1, en: 'MQA', cache: ['1 × (K + V): the smallest per-head cache, 8× smaller here. All query heads read the same keys and values, which costs some quality. PaLM and Falcon-7B used it.', '1 ×（K + V）：按 head 共享能做到的最小 cache，这里小了 8 倍。所有 query head 读同一份 key 和 value，质量会有些损失。PaLM 和 Falcon-7B 用过它。'] },
    mla: { groups: 8, en: 'MLA', cache: ['One latent of 4.5 × d_head numbers, however many heads there are: what GQA would need with 2.25 KV heads. Every head still gets its own K and V, rebuilt from the latent. DeepSeek-V2 reports quality above MHA.', '只缓存一个 4.5 × d_head 的 latent，不管有多少个 head：相当于 GQA 只用 2.25 个 KV head。每个 head 仍然有自己的 K 和 V，由 latent 现场还原。DeepSeek-V2 报告它的效果还优于 MHA。'] }
  };
  var mode = 'mha';

  var svg = s('svg', { viewBox: '0 0 820 290', class: 'heads-svg', role: 'img', 'aria-label': 'Query heads and the key/value heads they share' });
  svg.appendChild(TX.bi(s('text', { class: 's-sub', x: 14, y: QY - 8 }), 'query heads', 'query head'));
  var kvLabel = s('text', { class: 's-sub', x: 14, y: KVY - 8 });
  svg.appendChild(kvLabel);
  var lines = [], fan = [], pairs = [];
  for (var i = 0; i < H; i++) {
    lines.push(s('line', { class: 'hd-line', y1: QY + 34, y2: KVY })); svg.appendChild(lines[i]);
    fan.push(s('line', { class: 'hd-fan', y1: LATY, y2: KVY + 34 })); svg.appendChild(fan[i]);
  }
  for (i = 0; i < H; i++) {
    var qx = X0 + i * PITCH;
    svg.appendChild(s('rect', { class: 'hd-q', x: qx, y: QY, width: QW, height: 34, rx: 6 }));
    svg.appendChild(s('text', { class: 's-title hd-qt', x: qx + QW / 2, y: QY + 22, 'text-anchor': 'middle', text: 'Q' + (i + 1) }));
    var g = s('g', { class: 'hd-pair' });
    g.appendChild(s('rect', { class: 'hd-k', x: 0, y: 0, width: 34, height: 34, rx: 6 }));
    g.appendChild(s('rect', { class: 'hd-v', x: 38, y: 0, width: 34, height: 34, rx: 6 }));
    g.appendChild(s('text', { class: 's-title hd-kt', x: 17, y: 22, 'text-anchor': 'middle', text: 'K' }));
    g.appendChild(s('text', { class: 's-title hd-vt', x: 55, y: 22, 'text-anchor': 'middle', text: 'V' }));
    svg.appendChild(g); pairs.push(g);
  }
  var latent = s('g', { class: 'hd-latent' });
  latent.appendChild(s('rect', { class: 'hd-k', x: 270, y: LATY, width: 220, height: 30, rx: 6 }));
  latent.appendChild(TX.bi(s('text', { class: 's-title hd-kt', x: 380, y: LATY + 20, 'text-anchor': 'middle' }), 'latent c · 4 × d_head (512)', 'latent c · 4 × d_head（512）'));
  latent.appendChild(s('rect', { class: 'hd-k', x: 498, y: LATY, width: 92, height: 30, rx: 6 }));
  latent.appendChild(s('text', { class: 's-title hd-kt', x: 544, y: LATY + 20, 'text-anchor': 'middle', text: 'k_rope (64)' }));
  latent.appendChild(TX.bi(s('text', { class: 's-sub', x: 430, y: LATY + 50, 'text-anchor': 'middle' }), 'this is all that is cached; the dashed K and V are rebuilt on the fly', '真正缓存的只有这一条；虚线的 K、V 都是用的时候再还原'));
  svg.appendChild(latent);
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));

  /* ── tweened state ────────────────────────────────────────────────────── */
  var state = { x: [], op: [], ghost: 0 }, from = null, to = null, start = 0;
  function target(id) {
    var groups = MODES[id].groups, size = H / groups, out = { x: [], op: [], ghost: id === 'mla' ? 1 : 0 };
    for (var i = 0; i < H; i++) {
      var group = Math.floor(i / size);
      out.x.push(X0 + (group * size + (size - 1) / 2) * PITCH);
      out.op.push(i % size === 0 ? 1 : 0);
    }
    return out;
  }
  function draw() {
    for (var i = 0; i < H; i++) {
      pairs[i].setAttribute('transform', 'translate(' + state.x[i].toFixed(1) + ',' + KVY + ')');
      pairs[i].setAttribute('opacity', (state.op[i] * (1 - 0.55 * state.ghost)).toFixed(3));
      lines[i].setAttribute('x1', X0 + i * PITCH + QW / 2); lines[i].setAttribute('x2', state.x[i] + QW / 2);
      fan[i].setAttribute('x1', 380 + (i - 3.5) * 22); fan[i].setAttribute('x2', state.x[i] + QW / 2);
      fan[i].setAttribute('opacity', state.ghost.toFixed(3));
    }
    latent.setAttribute('opacity', state.ghost.toFixed(3));
    // the latent row only exists for MLA, so the other modes do not reserve space for it
    svg.setAttribute('viewBox', '0 0 820 ' + (186 + 104 * state.ghost).toFixed(0));
    svg.classList.toggle('is-mla', state.ghost > 0.5);
  }
  var ticker = TX.loop(function (now) {
    var k = TX.clamp((now - start) / 700, 0, 1), e = TX.ease(k);
    for (var i = 0; i < H; i++) { state.x[i] = TX.lerp(from.x[i], to.x[i], e); state.op[i] = TX.lerp(from.op[i], to.op[i], e); }
    state.ghost = TX.lerp(from.ghost, to.ghost, e);
    draw();
    if (k === 1) ticker.stop();
  });

  var note = h('p', { class: 'fig-caption' });
  function select(id) {
    mode = id;
    TX.bi(kvLabel, id === 'mla' ? 'per-head K/V, reconstructed (not cached)' : 'K/V heads (cached)', id === 'mla' ? '每个 head 的 K/V，现场还原（不缓存）' : 'K/V head（需要缓存）');
    TX.bi(note, MODES[id].cache[0], MODES[id].cache[1]);
    to = target(id);
    if (TX.reduced || !state.x.length) { state = to; draw(); }
    else { from = { x: state.x.slice(), op: state.op.slice(), ghost: state.ghost }; start = performance.now(); ticker.start(); }
    renderBars();
  }
  var modeSeg = TX.seg(Object.keys(MODES).map(function (id) { return { id: id, en: MODES[id].en, zh: MODES[id].en }; }), mode, select, 'Attention variant');
  fig.controls.appendChild(modeSeg.el);

  /* ── calculator ───────────────────────────────────────────────────────── */
  var SHAPES = {
    l7: { en: 'Llama-2-7B shape', L: 32, heads: 32, d: 128 },
    l70: { en: 'Llama-3-70B shape', L: 80, heads: 64, d: 128 },
    ds: { en: 'DeepSeek-V3 shape', L: 61, heads: 128, d: 128 }
  };
  var shape = 'l7', ctxExp = 15, batchExp = 0;
  var VARIANTS = [
    { id: 'mha', label: 'MHA', perLayer: function (c) { return 2 * c.heads * c.d; } },
    { id: 'gqa', label: 'GQA · 8 KV heads', perLayer: function (c) { return 2 * 8 * c.d; } },
    { id: 'mqa', label: 'MQA · 1 KV head', perLayer: function (c) { return 2 * c.d; } },
    { id: 'mla', label: 'MLA · 4.5 × d_head', perLayer: function (c) { return 4.5 * c.d; } }
  ];
  function fmtTokens(e) { var n = Math.pow(2, e); return n >= 1048576 ? (n / 1048576) + 'M' : (n / 1024) + 'K'; }
  var shapeSeg = TX.seg(Object.keys(SHAPES).map(function (id) { return { id: id, en: SHAPES[id].en, zh: SHAPES[id].en }; }), shape, function (id) { shape = id; renderBars(); }, 'Model shape');
  var ctxSlider = TX.slider({ en: 'Context length', zh: '上下文长度', min: 10, max: 20, value: ctxExp, format: function (e) { return fmtTokens(e) + ' tokens'; }, onInput: function (v) { ctxExp = v; renderBars(); } });
  var batchSlider = TX.slider({ en: 'Concurrent sequences', zh: '并发序列数', min: 0, max: 6, value: batchExp, format: function (e) { return String(Math.pow(2, e)); }, onInput: function (v) { batchExp = v; renderBars(); } });
  var rows = {};
  var chart = h('div', { class: 'mini-bars calc-bars' });
  VARIANTS.forEach(function (variant) {
    var fill = h('span', { class: 'mini-fill' }), value = h('strong', { class: 'mini-val' }), ratio = h('span', { class: 'mini-ratio' });
    rows[variant.id] = { fill: fill, value: value, ratio: ratio, row: h('div', { class: 'mini-row' }, [h('span', { class: 'mini-label', text: variant.label }), h('span', { class: 'mini-track' }, [fill]), value, ratio]) };
    chart.appendChild(rows[variant.id].row);
  });
  var shapeNote = h('p', { class: 'fig-note' });
  fig.foot.appendChild(note);
  fig.foot.appendChild(h('div', { class: 'calc' }, [
    h('p', { class: 'calc-title', bi: ['KV cache calculator · fp16, every variant on the same model shape', 'KV cache 计算器 · fp16，所有方案用同一个模型形状'] }),
    h('div', { class: 'calc-controls' }, [shapeSeg.el, ctxSlider.el, batchSlider.el]),
    chart, shapeNote
  ]));

  function renderBars() {
    var cfg = SHAPES[shape], tokens = Math.pow(2, ctxExp) * Math.pow(2, batchExp);
    var base = VARIANTS[0].perLayer(cfg) * cfg.L * 2 * tokens;
    VARIANTS.forEach(function (variant) {
      var bytes = variant.perLayer(cfg) * cfg.L * 2 * tokens, row = rows[variant.id];
      row.fill.style.width = Math.max(0.4, bytes / base * 100).toFixed(2) + '%';
      row.value.textContent = TX.fmtBytes(bytes);
      row.ratio.textContent = variant.id === 'mha' ? '' : '÷ ' + (base / bytes >= 10 ? Math.round(base / bytes) : (base / bytes).toFixed(1));
      row.row.classList.toggle('on', mode.indexOf(variant.id) === 0);
    });
    TX.bi(shapeNote,
      cfg.L + ' layers × ' + cfg.heads + ' query heads × d_head ' + cfg.d + '. For scale: one H100 has 80 GB, and the weights need room too.',
      cfg.L + ' 层 × ' + cfg.heads + ' 个 query head × d_head ' + cfg.d + '。作为参照：一张 H100 是 80 GB，而且权重本身也要占显存。');
  }

  select(mode);
})();

/* 05 · Attention patterns: full, sliding window, local:global interleave, sinks. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-windows', { title: ['Who can see whom · 20 tokens', '谁能看到谁 · 20 个 token'] });
  if (!fig) return;
  var s = TX.s, h = TX.h;

  var N = 20, CELL = 16, GX = 44, GY = 30, SINKS = 2;
  var pattern = 'window', w = 5, depth = 1, layer = 0, touched = false;
  var PATTERN_LAYERS = ['local', 'local', 'local', 'local', 'local', 'global'];

  var svg = s('svg', { viewBox: '0 0 380 372', class: 'mask-svg', role: 'img', 'aria-label': 'Attention mask' });
  svg.appendChild(TX.bi(s('text', { class: 's-sub', x: GX + N * CELL / 2, y: 18, 'text-anchor': 'middle' }), 'key position →', 'key 位置 →'));
  svg.appendChild(TX.bi(s('text', { class: 's-sub', x: 16, y: GY + N * CELL / 2, 'text-anchor': 'middle', transform: 'rotate(-90 16 ' + (GY + N * CELL / 2) + ')' }), '← query position', '← query 位置'));
  var cells = [];
  for (var i = 0; i < N; i++) {
    cells.push([]);
    for (var j = 0; j < N; j++) {
      var rect = s('rect', { class: 'mk-cell', x: GX + j * CELL, y: GY + i * CELL, width: CELL - 2, height: CELL - 2, rx: 2.5 });
      svg.appendChild(rect); cells[i].push(rect);
    }
  }
  var legend = s('g', { transform: 'translate(' + GX + ',' + (GY + N * CELL + 12) + ')' });
  [['mk-cell direct', 0, 'attends', '直接可见'], ['mk-cell reach', 84, 'reachable via lower layers', '经由下层可达'], ['mk-cell sink', 252, 'sink', 'sink']].forEach(function (item) {
    legend.appendChild(s('rect', { class: item[0], x: item[1], y: 0, width: 12, height: 12, rx: 2.5 }));
    legend.appendChild(TX.bi(s('text', { class: 's-sub', x: item[1] + 17, y: 10.5 }), item[2], item[3]));
  });
  svg.appendChild(legend);

  /* ── right-hand panel ─────────────────────────────────────────────────── */
  var patternSeg = TX.seg([
    { id: 'full', en: 'Full causal', zh: '完整 causal' }, { id: 'window', en: 'Sliding window', zh: 'Sliding window' },
    { id: 'mix', en: '5 local : 1 global', zh: '5 local : 1 global' }, { id: 'sink', en: 'Sinks + window', zh: 'Sink + 窗口' }
  ], pattern, function (id) { pattern = id; touched = id !== 'mix' ? touched : false; render(); }, 'Pattern');
  fig.controls.appendChild(patternSeg.el);

  var wSlider = TX.slider({ en: 'Window w', zh: '窗口 w', min: 2, max: 12, value: w, format: function (v) { return v + ' tokens'; }, onInput: function (v) { w = v; render(); } });
  var dSlider = TX.slider({ en: 'Stacked layers', zh: '堆叠层数', min: 1, max: 6, value: depth, format: function (v) { return String(v); }, onInput: function (v) { depth = v; render(); } });
  var strip = h('div', { class: 'layer-strip', role: 'group', 'aria-label': 'Layer' });
  var stripButtons = PATTERN_LAYERS.map(function (kind, index) {
    var button = h('button', { type: 'button', class: 'layer-btn is-' + kind, text: kind === 'global' ? 'G' : 'L', title: 'Layer ' + (index + 1), on: { click: function () { touched = true; layer = index; render(); } } });
    strip.appendChild(button);
    return button;
  });
  var stripWrap = h('div', { class: 'ctl-block' }, [h('p', { class: 'ro-label', bi: ['Layer (click, or watch it cycle)', '层（可以点击，也会自动轮播）'] }), strip]);
  var roScores = TX.readout('Scores computed per layer', '每层要算的分数'), roCache = TX.readout('KV cache per layer', '每层 KV cache'), roReach = TX.readout('How far back a token can reach', '一个 token 能回看多远');
  var note = h('p', { class: 'fig-note' });
  var panel = h('div', { class: 'split-side' }, [wSlider.el, dSlider.el, stripWrap, h('div', { class: 'readouts is-stack' }, [roScores.el, roCache.el, roReach.el]), note]);
  fig.stage.appendChild(h('div', { class: 'split' }, [h('div', { class: 'split-main' }, [svg]), panel]));

  var NOTES = {
    full: ['Every token sees the whole prefix. Exact, but compute grows with n² and the cache with n.', '每个 token 都能看到完整前缀。结果精确，但计算量随 n² 增长，cache 随 n 增长。'],
    window: ['Mistral 7B uses w = 4,096 across 32 layers, for a theoretical reach of about 131K tokens.', 'Mistral 7B 用 w = 4,096、共 32 层，理论可达范围约 131K token。'],
    mix: ['Gemma 3: five local layers (window 1,024) for every global layer. Only the global layers keep a full-length cache.', 'Gemma 3：每 5 层 local（窗口 1,024）配 1 层 global，只有 global 层需要保留完整长度的 cache。'],
    sink: ['StreamingLLM keeps the first few tokens forever. Softmax has to put its weight somewhere, and models learn to dump it there; evict them and quality collapses. gpt-oss builds the same escape hatch in as a learned bias.', 'StreamingLLM 永久保留最开头的几个 token。softmax 的权重总得放在某处，模型学会了把多余的权重丢在那里；一旦把它们逐出 cache，效果就会崩。gpt-oss 把同样的“出口”做成了一个可学习的 bias。']
  };

  function render() {
    var local = pattern === 'window' || pattern === 'sink' || (pattern === 'mix' && PATTERN_LAYERS[layer] === 'local');
    var layers = pattern === 'window' || pattern === 'sink' ? depth : 1, direct = 0;
    for (var i = 0; i < N; i++) {
      for (var j = 0; j < N; j++) {
        var cls = 'mk-cell';
        if (j > i) cls += ' future';
        else if (pattern === 'sink' && j < SINKS) { cls += ' sink'; direct++; }
        else if (!local || i - j < w) { cls += ' direct'; direct++; }
        else if (i - j <= layers * (w - 1)) cls += ' reach';
        cells[i][j].setAttribute('class', cls);
      }
    }
    wSlider.el.classList.toggle('is-off', pattern === 'full');
    dSlider.el.classList.toggle('is-off', pattern === 'full' || pattern === 'mix');
    stripWrap.classList.toggle('is-off', pattern !== 'mix');
    stripButtons.forEach(function (button, index) { button.setAttribute('aria-pressed', index === layer ? 'true' : 'false'); });
    var total = N * (N + 1) / 2;
    roScores.set(direct + ' / ' + total + '  (' + Math.round(direct / total * 100) + '%)');
    roCache.set(!local ? TX.t('all n tokens', '全部 n 个 token') : pattern === 'sink' ? TX.t(SINKS + ' sinks + last ' + w + ' tokens', SINKS + ' 个 sink + 最近 ' + w + ' 个 token') : TX.t('last ' + w + ' tokens', '最近 ' + w + ' 个 token'));
    roReach.set(!local ? TX.t('everything', '全部') : pattern === 'mix' ? TX.t(w + ' tokens in this layer', '这一层 ' + w + ' 个 token') : TX.t(layers + ' × (w − 1) + 1 = ' + (layers * (w - 1) + 1) + ' tokens', layers + ' × (w − 1) + 1 = ' + (layers * (w - 1) + 1) + ' 个 token'));
    TX.bi(note, NOTES[pattern][0], NOTES[pattern][1]);
  }
  TX.onLang(render);

  var elapsed = 0;
  var ticker = TX.loop(function (now, dt) {
    if (pattern !== 'mix' || touched) return;
    elapsed += dt;
    if (elapsed > 1300) { elapsed = 0; layer = (layer + 1) % PATTERN_LAYERS.length; render(); }
  });
  render();
  if (!TX.reduced) TX.whenVisible(fig.root, ticker.start, ticker.stop);
})();

/* 06 · RoPE. Three of the 32 rotation planes of a 64-dim head, plus the score
   as a function of relative distance (identical content in q and k). */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-rope', { title: ['Three of the 32 rotation planes in one 64-dim head', '一个 64 维 head 里 32 个旋转平面中的 3 个'] });
  if (!fig) return;
  var s = TX.s, h = TX.h;

  var DIM = 64, PAIRS = [0, 8, 16], MAX_DELTA = 128, MAX_N = 256;
  var base = 10000, n = 0, delta = 12, playing = !TX.reduced;
  function theta(i) { return Math.pow(base, -2 * i / DIM); }
  function score(d) { var sum = 0; for (var i = 0; i < DIM / 2; i++) sum += Math.cos(d * theta(i)); return sum / (DIM / 2); }

  var svg = s('svg', { viewBox: '0 0 820 270', class: 'rope-svg', role: 'img', 'aria-label': 'Rotary position embedding' });
  var R = 58, CY = 112, planes = PAIRS.map(function (pair, index) {
    var cx = 84 + index * 152, g = s('g', { transform: 'translate(' + cx + ',' + CY + ')' });
    g.appendChild(s('circle', { class: 'rp-ring', r: R }));
    g.appendChild(s('line', { class: 'rp-axis', x1: -R - 6, x2: R + 6, y1: 0, y2: 0 }));
    g.appendChild(s('line', { class: 'rp-axis', y1: -R - 6, y2: R + 6, x1: 0, x2: 0 }));
    var arc = s('path', { class: 'rp-arc' }), k = s('line', { class: 'rp-k', x1: 0, y1: 0 }), q = s('line', { class: 'rp-q', x1: 0, y1: 0 });
    var kTip = s('circle', { class: 'rp-k-tip', r: 5 }), qTip = s('circle', { class: 'rp-q-tip', r: 5 });
    [arc, k, q, kTip, qTip].forEach(function (node) { g.appendChild(node); });
    var title = s('text', { class: 's-title', x: 0, y: R + 30, 'text-anchor': 'middle' }), sub = s('text', { class: 's-sub', x: 0, y: R + 47, 'text-anchor': 'middle' });
    g.appendChild(title); g.appendChild(sub);
    svg.appendChild(g);
    return { pair: pair, arc: arc, k: k, q: q, kTip: kTip, qTip: qTip, title: title, sub: sub };
  });
  [['rp-q-tip', 30, 'q at position m = n + Δ', 'q 在位置 m = n + Δ'], ['rp-k-tip', 240, 'k at position n', 'k 在位置 n']].forEach(function (item) {
    svg.appendChild(s('circle', { class: item[0], cx: item[1], cy: 22, r: 5 }));
    svg.appendChild(TX.bi(s('text', { class: 's-sub', x: item[1] + 11, y: 26 }), item[2], item[3]));
  });

  // chart: score vs relative distance
  var PX = 520, PY = 40, PW = 280, PH = 150;
  function sx(d) { return PX + d / MAX_DELTA * PW; }
  function sy(v) { return PY + (1 - (v + 0.25) / 1.25) * PH; }
  svg.appendChild(TX.bi(s('text', { class: 's-title', x: PX + PW / 2, y: 24, 'text-anchor': 'middle' }), 'score vs. distance (same content)', '分数随距离的变化（内容相同）'));
  [0, 0.5, 1].forEach(function (v) {
    svg.appendChild(s('line', { class: v === 0 ? 'ch-zero' : 'ch-grid', x1: PX, x2: PX + PW, y1: sy(v), y2: sy(v) }));
    svg.appendChild(s('text', { class: 's-sub', x: PX - 8, y: sy(v) + 4, 'text-anchor': 'end', text: String(v) }));
  });
  [0, 32, 64, 96, 128].forEach(function (d) { svg.appendChild(s('text', { class: 's-sub', x: sx(d), y: PY + PH + 18, 'text-anchor': 'middle', text: String(d) })); });
  svg.appendChild(TX.bi(s('text', { class: 's-sub', x: PX + PW / 2, y: PY + PH + 38, 'text-anchor': 'middle' }), 'relative distance Δ = m − n (tokens)', '相对距离 Δ = m − n（token）'));
  var curve = s('path', { class: 'ch-line' }), guide = s('line', { class: 'ch-guide', y1: PY, y2: PY + PH }), marker = s('circle', { class: 'ch-marker', r: 5.5 });
  var scrub = s('rect', { class: 'ch-hit', x: PX, y: PY, width: PW, height: PH });
  [curve, guide, marker, scrub].forEach(function (node) { svg.appendChild(node); });
  function scrubTo(event) {
    var box = svg.getBoundingClientRect(), x = (event.clientX - box.left) / box.width * 820;
    delta = Math.round(TX.clamp((x - PX) / PW, 0, 1) * MAX_DELTA);
    deltaSlider.set(delta); render();
  }
  scrub.addEventListener('pointerdown', function (event) { scrub.setPointerCapture(event.pointerId); scrubTo(event); });
  scrub.addEventListener('pointermove', function (event) { if (event.buttons || event.pointerType === 'mouse') scrubTo(event); });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));

  /* ── controls ─────────────────────────────────────────────────────────── */
  var baseSeg = TX.seg([{ id: '10000', en: 'θ = 10k (original)', zh: 'θ = 10k（原始）' }, { id: '500000', en: 'θ = 500k (Llama 3)', zh: 'θ = 500k（Llama 3）' }, { id: '1000000', en: 'θ = 1M (Gemma 3 global)', zh: 'θ = 1M（Gemma 3 global 层）' }], String(base), function (id) { base = +id; drawCurve(); render(); }, 'RoPE base');
  var playBtn = TX.button('', '', function () { playing = !playing; if (playing) ticker.start(); else ticker.stop(); label(); });
  function label() { TX.bi(playBtn, playing ? '❚❚ Pause' : '▶ Slide both positions', playing ? '❚❚ 暂停' : '▶ 同时滑动两个位置'); }
  label();
  fig.controls.appendChild(baseSeg.el); fig.controls.appendChild(playBtn);
  var nSlider = TX.slider({ en: 'Key position n', zh: 'key 位置 n', min: 0, max: MAX_N, value: n, format: function (v) { return 'n = ' + v + ', m = ' + (v + delta); }, onInput: function (v) { n = v; playing = false; ticker.stop(); label(); render(); } });
  var deltaSlider = TX.slider({ en: 'Distance Δ', zh: '距离 Δ', min: 0, max: MAX_DELTA, value: delta, format: function (v) { return 'Δ = ' + v; }, onInput: function (v) { delta = v; render(); } });
  var roScore = TX.readout('Score at this Δ', '当前 Δ 下的分数'), roNote = TX.readout('While n slides', 'n 滑动的时候');
  fig.foot.appendChild(h('div', { class: 'ctl-row' }, [nSlider.el, deltaSlider.el]));
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roScore.el, roNote.el]));

  function drawCurve() {
    var d = '';
    for (var i = 0; i <= MAX_DELTA; i++) d += (i ? 'L' : 'M') + sx(i).toFixed(1) + ' ' + sy(score(i)).toFixed(1);
    curve.setAttribute('d', d);
  }
  function render() {
    var m = n + delta;
    planes.forEach(function (plane) {
      var th = theta(plane.pair), aq = m * th, ak = n * th;
      var qx = R * Math.cos(aq), qy = -R * Math.sin(aq), kx = R * Math.cos(ak), ky = -R * Math.sin(ak);
      plane.q.setAttribute('x2', qx); plane.q.setAttribute('y2', qy); plane.qTip.setAttribute('cx', qx); plane.qTip.setAttribute('cy', qy);
      plane.k.setAttribute('x2', kx); plane.k.setAttribute('y2', ky); plane.kTip.setAttribute('cx', kx); plane.kTip.setAttribute('cy', ky);
      var gap = (delta * th) % (2 * Math.PI), r = 24;
      var ax0 = r * Math.cos(ak), ay0 = -r * Math.sin(ak), ax1 = r * Math.cos(ak + gap), ay1 = -r * Math.sin(ak + gap);
      plane.arc.setAttribute('d', gap < 0.01 ? '' : 'M' + ax0.toFixed(1) + ' ' + ay0.toFixed(1) + ' A' + r + ' ' + r + ' 0 ' + (gap > Math.PI ? 1 : 0) + ' 0 ' + ax1.toFixed(1) + ' ' + ay1.toFixed(1));
      plane.sub.textContent = (th >= 0.01 ? th.toFixed(3) : th.toExponential(1)) + ' rad / token';
    });
    var value = score(delta);
    marker.setAttribute('cx', sx(delta)); marker.setAttribute('cy', sy(value));
    guide.setAttribute('x1', sx(delta)); guide.setAttribute('x2', sx(delta));
    nSlider.set(Math.round(n));
    roScore.set(value.toFixed(3));
  }
  function renderText() { roNote.set(TX.t('both vectors turn, the angle between them does not', '两个向量一起转，它们之间的夹角不变')); }
  TX.onLang(renderText);
  planes.forEach(function (plane, index) {
    TX.bi(plane.title, 'pair ' + plane.pair + ' · ' + ['fast', 'medium', 'slow'][index], '第 ' + plane.pair + ' 对 · ' + ['快', '中', '慢'][index]);
  });

  var ticker = TX.loop(function (now, dt) {
    n += dt * 0.012;
    if (n > MAX_N) n = 0;
    render();
  });
  drawCurve(); render(); renderText();
  TX.whenVisible(fig.root, function () { if (playing) ticker.start(); }, function () { ticker.stop(); });
})();

/* 07 · Mixture of experts. Expert grids are drawn at true scale (8, 128 or 256
   experts); the router here is random, a real one is a learned linear layer. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-moe', { title: ['One token at a time through the FFN · random router for illustration', '一次一个 token 经过 FFN · router 为随机示意'] });
  if (!fig) return;
  var s = TX.s, h = TX.h;

  var PRESETS = {
    dense: { en: 'Dense FFN', zh: 'Dense FFN', n: 1, k: 1, total: 70, active: 70, model: 'Llama 3 70B' },
    mixtral: { en: 'Mixtral · 8, top-2', zh: 'Mixtral · 8 选 2', n: 8, k: 2, total: 47, active: 13, model: 'Mixtral 8x7B' },
    gptoss: { en: 'gpt-oss · 128, top-4', zh: 'gpt-oss · 128 选 4', n: 128, k: 4, total: 116.8, active: 5.1, model: 'gpt-oss-120b' },
    qwen3: { en: 'Qwen3 · 128, top-8', zh: 'Qwen3 · 128 选 8', n: 128, k: 8, total: 235, active: 22, model: 'Qwen3-235B-A22B' },
    deepseek: { en: 'DeepSeek-V3 · 1 + 256, top-8', zh: 'DeepSeek-V3 · 1 + 256 选 8', n: 256, k: 8, shared: true, total: 671, active: 37, model: 'DeepSeek-V3' }
  };
  var preset = 'mixtral', experts = [], counts = [], routed = 0, rand = TX.rng(11);
  var CX = 410, ROUTER_Y = 300, EXP_BOTTOM = 232, MERGE_Y = 46;

  var svg = s('svg', { viewBox: '0 0 820 380', class: 'moe-svg', role: 'img', 'aria-label': 'Tokens routed to experts' });
  var gLines = s('g'), gExperts = s('g');
  svg.appendChild(gLines); svg.appendChild(gExperts);
  svg.appendChild(s('circle', { class: 'moe-merge', cx: CX, cy: MERGE_Y, r: 17 }));
  svg.appendChild(s('text', { class: 's-title', x: CX, y: MERGE_Y + 5, 'text-anchor': 'middle', text: 'Σ' }));
  svg.appendChild(TX.bi(s('text', { class: 's-sub', x: CX + 28, y: MERGE_Y + 4 }), 'weighted sum of the chosen experts’ outputs', '被选中 expert 输出的加权和'));
  var router = s('g', { transform: 'translate(' + CX + ',' + ROUTER_Y + ')' });
  router.appendChild(s('rect', { class: 'moe-router', x: -130, y: 0, width: 260, height: 30, rx: 8 }));
  var routerText = s('text', { class: 's-title', x: 0, y: 20, 'text-anchor': 'middle' });
  router.appendChild(routerText);
  svg.appendChild(router);
  svg.appendChild(s('line', { class: 'moe-stem', x1: CX, x2: CX, y1: ROUTER_Y + 30, y2: 372 }));
  var token = s('circle', { class: 'moe-token', cx: CX, r: 6, cy: 372 });
  svg.appendChild(token);
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));

  var seg = TX.seg(Object.keys(PRESETS).map(function (id) { return { id: id, en: PRESETS[id].en, zh: PRESETS[id].zh }; }), preset, function (id) { preset = id; build(); }, 'MoE preset');
  fig.controls.appendChild(seg.el);
  var roUsed = TX.readout('Experts used per token', '每个 token 用到的 expert'), roLoad = TX.readout('Load so far (tokens per expert)', '目前的负载（每个 expert 的 token 数）');
  var paramFill = h('span', { class: 'mini-fill' }), paramText = h('strong', { class: 'mini-val' }), paramLabel = h('span', { class: 'mini-label' });
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roUsed.el, roLoad.el]));
  fig.foot.appendChild(h('div', { class: 'mini-bars' }, [
    h('p', { class: 'ro-label', bi: ['Parameters touched by one token (filled) out of all parameters (track)', '一个 token 实际用到的参数（填充）占全部参数（底槽）的比例'] }),
    h('div', { class: 'mini-row on' }, [paramLabel, h('span', { class: 'mini-track' }, [paramFill]), paramText])
  ]));

  function build() {
    var cfg = PRESETS[preset], i;
    TX.clear(gExperts); TX.clear(gLines);
    experts = []; counts = []; routed = 0;
    if (cfg.n <= 8) {
      var w = cfg.n === 1 ? 300 : 74, pitch = w + 16, x0 = CX - ((cfg.n - 1) * pitch) / 2;
      for (i = 0; i < cfg.n; i++) {
        var gx = x0 + i * pitch, rect = s('rect', { class: 'moe-expert is-big', x: gx - w / 2, y: EXP_BOTTOM - 64, width: w, height: 64, rx: 8 });
        gExperts.appendChild(rect);
        gExperts.appendChild(s('text', { class: 's-title', x: gx, y: EXP_BOTTOM - 27, 'text-anchor': 'middle', text: cfg.n === 1 ? 'FFN' : 'E' + (i + 1) }));
        experts.push({ rect: rect, x: gx, top: EXP_BOTTOM - 64, bottom: EXP_BOTTOM });
      }
    } else {
      var cols = 32, cell = 13, gap = 3, rows = cfg.n / cols, gridW = cols * (cell + gap) - gap, left = CX - gridW / 2 + (cfg.shared ? 44 : 0), topY = EXP_BOTTOM - rows * (cell + gap) + gap;
      for (i = 0; i < cfg.n; i++) {
        var col = i % cols, row = Math.floor(i / cols), ex = left + col * (cell + gap), ey = topY + row * (cell + gap);
        var small = s('rect', { class: 'moe-expert', x: ex, y: ey, width: cell, height: cell, rx: 3 });
        gExperts.appendChild(small);
        experts.push({ rect: small, x: ex + cell / 2, top: ey, bottom: ey + cell });
      }
      if (cfg.shared) {
        var sharedRect = s('rect', { class: 'moe-expert is-big is-shared', x: left - 100, y: topY, width: 74, height: rows * (cell + gap) - gap, rx: 8 });
        gExperts.appendChild(sharedRect);
        gExperts.appendChild(TX.bi(s('text', { class: 's-title', x: left - 63, y: topY + 58, 'text-anchor': 'middle' }), 'shared', 'shared'));
        gExperts.appendChild(TX.bi(s('text', { class: 's-sub', x: left - 63, y: topY + 75, 'text-anchor': 'middle' }), 'every token', '每个 token'));
        experts.shared = { rect: sharedRect, x: left - 63, top: topY, bottom: topY + rows * (cell + gap) - gap };
      }
    }
    for (i = 0; i < cfg.n; i++) counts.push(0);
    TX.bi(routerText, cfg.n === 1 ? 'no router: one FFN for every token' : 'Router · scores ' + cfg.n + ' experts, keeps top-' + cfg.k, cfg.n === 1 ? '没有 router：所有 token 走同一个 FFN' : 'Router · 给 ' + cfg.n + ' 个 expert 打分，保留前 ' + cfg.k + ' 个');
    roUsed.set(cfg.n === 1 ? '1 / 1  (100%)' : (cfg.k + (cfg.shared ? ' + 1 shared' : '')) + ' / ' + (cfg.n + (cfg.shared ? 1 : 0)) + '  (' + ((cfg.k + (cfg.shared ? 1 : 0)) / (cfg.n + (cfg.shared ? 1 : 0)) * 100).toFixed(1) + '%)');
    paramLabel.textContent = cfg.model;
    paramFill.style.width = (cfg.active / cfg.total * 100).toFixed(1) + '%';
    paramText.textContent = cfg.active + 'B / ' + cfg.total + 'B  (' + Math.round(cfg.active / cfg.total * 100) + '%)';
    route();
  }

  function route() {
    var cfg = PRESETS[preset], chosen = {}, picks = [];
    TX.clear(gLines);
    experts.forEach(function (expert) { expert.rect.classList.remove('hot'); });
    while (picks.length < cfg.k) {
      var index = Math.floor(rand() * cfg.n);
      if (!chosen[index]) { chosen[index] = true; picks.push(experts[index]); counts[index]++; }
    }
    if (experts.shared) picks.push(experts.shared);
    routed++;
    picks.forEach(function (expert) {
      expert.rect.classList.add('hot');
      gLines.appendChild(s('line', { class: 'moe-route', x1: CX, y1: ROUTER_Y, x2: expert.x, y2: expert.bottom }));
      gLines.appendChild(s('line', { class: 'moe-route', x1: expert.x, y1: expert.top, x2: CX, y2: MERGE_Y + 17 }));
    });
    var max = Math.max.apply(null, counts), min = Math.min.apply(null, counts);
    experts.forEach(function (expert, i) { expert.rect.style.setProperty('--load', max ? (counts[i] / max).toFixed(2) : 0); });
    roLoad.set(cfg.n === 1 ? TX.fmtInt(routed) : 'min ' + min + ' · max ' + max + ' · ' + TX.fmtInt(routed) + ' ' + TX.t('tokens', '个 token'));
    if (!TX.reduced) { token.classList.remove('go'); void token.getBoundingClientRect(); token.classList.add('go'); }
  }

  var elapsed = 0;
  var ticker = TX.loop(function (now, dt) { elapsed += dt; if (elapsed > 800) { elapsed = 0; route(); } });
  build();
  if (!TX.reduced) TX.whenVisible(fig.root, ticker.start, ticker.stop);
})();

/* 08 · FlashAttention. Same 16×16 causal attention computed two ways: the
   standard way (materialise S and P in HBM) and tiled (4×4 blocks in SRAM). */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-flash', { title: ['16 tokens · where the attention matrix lives', '16 个 token · attention 矩阵到底放在哪里'] });
  if (!fig) return;
  var s = TX.s, h = TX.h;

  var N = 16, B = 4, CELL = 15, GX = 40, GY = 56, NB = N / B;
  var mode = 'flash', tick = 0, playing = !TX.reduced;

  var svg = s('svg', { viewBox: '0 0 820 330', class: 'flash-svg', role: 'img', 'aria-label': 'Standard attention versus FlashAttention memory traffic' });
  svg.appendChild(TX.bi(s('text', { class: 's-title', x: GX + N * CELL / 2, y: 26, 'text-anchor': 'middle' }), 'n × n scores (causal)', 'n × n 分数矩阵（causal）'));
  var gridSub = s('text', { class: 's-sub', x: GX + N * CELL / 2, y: 44, 'text-anchor': 'middle' });
  svg.appendChild(gridSub);
  var cells = [];
  for (var i = 0; i < N; i++) {
    cells.push([]);
    for (var j = 0; j < N; j++) {
      var rect = s('rect', { class: 'fl-cell', x: GX + j * CELL, y: GY + i * CELL, width: CELL - 2, height: CELL - 2, rx: 2.5 });
      svg.appendChild(rect); cells[i].push(rect);
    }
  }
  var blockFrame = s('rect', { class: 'fl-block', width: B * CELL + 2, height: B * CELL + 2, rx: 4 });
  svg.appendChild(blockFrame);

  // memory hierarchy on the right
  var MX = 330, MW = 470;
  function memBox(y, hgt, en, zh, subEn, subZh, cls) {
    svg.appendChild(s('rect', { class: 'fl-mem ' + cls, x: MX, y: y, width: MW, height: hgt, rx: 10 }));
    svg.appendChild(TX.bi(s('text', { class: 's-title', x: MX + 14, y: y + 22 }), en, zh));
    svg.appendChild(TX.bi(s('text', { class: 's-sub', x: MX + MW - 14, y: y + 22, 'text-anchor': 'end' }), subEn, subZh));
  }
  memBox(20, 176, 'HBM · GPU main memory', 'HBM · GPU 主显存', 'tens of GB, slow to reach', '几十 GB，访问慢', 'is-hbm');
  memBox(214, 100, 'SRAM · on-chip', 'SRAM · 片上缓存', 'tens of MB, very fast', '几十 MB，非常快', 'is-sram');
  function chip(x, y, w, hgt, text, cls) {
    var g = s('g', { class: 'fl-chip ' + cls, transform: 'translate(' + x + ',' + y + ')' });
    g.appendChild(s('rect', { x: 0, y: 0, width: w, height: hgt, rx: 6 }));
    var label = s('text', { class: 's-sub', x: w / 2, y: hgt / 2 + 4, 'text-anchor': 'middle', text: text });
    g.appendChild(label); svg.appendChild(g);
    return { g: g, label: label };
  }
  chip(MX + 14, 56, 40, 34, 'Q', 'is-q'); chip(MX + 60, 56, 40, 34, 'K', 'is-k'); chip(MX + 106, 56, 40, 34, 'V', 'is-v');
  var chipO = chip(MX + 14, 100, 132, 34, 'O  (n × d)', 'is-v');
  var chipS = chip(MX + 170, 56, 136, 122, '', 'is-big'), chipP = chip(MX + 320, 56, 136, 122, '', 'is-big');
  var sramChips = [chip(MX + 14, 252, 110, 40, '', 'is-q'), chip(MX + 134, 252, 130, 40, '', 'is-k'), chip(MX + 274, 252, 180, 40, '', 'is-big')];
  svg.appendChild(TX.bi(s('text', { class: 's-sub', x: MX + 14, y: 162 }), 'inputs and output: n × d each', '输入与输出：各 n × d'));
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));

  var seg = TX.seg([{ id: 'std', en: 'Standard attention', zh: '标准 attention' }, { id: 'flash', en: 'FlashAttention', zh: 'FlashAttention' }], mode, function (id) { mode = id; tick = 0; render(); }, 'Implementation');
  var playBtn = TX.button('', '', function () { playing = !playing; if (playing) ticker.start(); else ticker.stop(); label(); });
  function label() { TX.bi(playBtn, playing ? '❚❚ Pause' : '▶ Play', playing ? '❚❚ 暂停' : '▶ 播放'); }
  label();
  fig.controls.appendChild(seg.el); fig.controls.appendChild(playBtn);
  var roHbm = TX.readout('Attention-matrix cells written to HBM', '写进 HBM 的 attention 矩阵元素'), roPeak = TX.readout('Extra memory', '额外显存'), roSkip = TX.readout('Blocks skipped by the causal mask', '被 causal mask 直接跳过的 block');
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roHbm.el, roPeak.el, roSkip.el]));
  var caption = h('p', { class: 'fig-caption' });
  fig.foot.appendChild(caption);

  // block schedule for the tiled version: for each query block, stream the key/value blocks it may see
  var schedule = [];
  for (var bi = 0; bi < NB; bi++) for (var bj = 0; bj <= bi; bj++) schedule.push([bi, bj]);
  var STD_TICKS = 2 * N + 6, FLASH_TICKS = schedule.length + 3;

  function render() {
    var i, j, flash = mode === 'flash';
    var stepIndex = Math.min(tick, schedule.length - 1), cur = schedule[stepIndex], done = tick >= schedule.length;
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        var cls = 'fl-cell';
        if (j > i) cls += ' future';
        else if (flash) {
          var qb = Math.floor(i / B), kb = Math.floor(j / B), order = qb * (qb + 1) / 2 + kb;
          if (!done && qb === cur[0] && kb === cur[1]) cls += ' hot';
          else if (order < tick) cls += ' gone';
        } else {
          if (i < tick - N) cls += ' prob';
          else if (i < tick) cls += ' stored';
        }
        cells[i][j].setAttribute('class', cls);
      }
    }
    blockFrame.setAttribute('visibility', flash && !done ? 'visible' : 'hidden');
    if (flash && !done) { blockFrame.setAttribute('x', GX + cur[1] * B * CELL - 2); blockFrame.setAttribute('y', GY + cur[0] * B * CELL - 2); }
    var sFrac = flash ? 0 : TX.clamp(tick / N, 0, 1), pFrac = flash ? 0 : TX.clamp((tick - N) / N, 0, 1);
    chipS.g.classList.toggle('is-off', flash); chipP.g.classList.toggle('is-off', flash);
    chipS.g.classList.toggle('hot', !flash && sFrac > 0 && sFrac < 1); chipP.g.classList.toggle('hot', !flash && pFrac > 0 && pFrac < 1);
    chipS.label.textContent = flash ? 'S  ' + TX.t('never stored', '从不存储') : 'S = QKᵀ   ' + Math.round(sFrac * 100) + '%';
    chipP.label.textContent = flash ? 'P  ' + TX.t('never stored', '从不存储') : 'P = softmax(S)   ' + Math.round(pFrac * 100) + '%';
    var outRows = flash ? (done ? N : cur[0] * B) : (tick >= 2 * N ? N : 0);
    chipO.label.textContent = 'O  (n × d)   ' + Math.round(outRows / N * 100) + '%';
    chipO.g.classList.toggle('hot', outRows > 0 && outRows < N);
    sramChips.forEach(function (c) { c.g.classList.toggle('is-off', !flash || done); });
    if (flash && !done) {
      sramChips[0].label.textContent = 'Q ' + TX.t('block', 'block') + ' ' + (cur[0] + 1);
      sramChips[1].label.textContent = 'K, V ' + TX.t('block', 'block') + ' ' + (cur[1] + 1);
      sramChips[2].label.textContent = '4 × 4 ' + TX.t('scores + running softmax', '分数 + running softmax');
    }
    TX.bi(gridSub, flash ? 'one 4 × 4 block at a time, then thrown away' : 'every cell is written out, then read back', flash ? '一次只算一个 4 × 4 的 block，算完就丢' : '每个元素都要写出去，再读回来');
    roHbm.set(flash ? '0' : TX.fmtInt(Math.round((sFrac + pFrac) * N * N)) + ' / ' + TX.fmtInt(2 * N * N));
    roPeak.set(flash ? 'O(n)  ·  ' + TX.t('one block of', '一个 block，') + ' ' + B * B + ' ' + TX.t('scores', '个分数') : 'O(n²)  ·  2 × ' + N * N + ' ' + TX.t('cells', '个元素'));
    roSkip.set(flash ? (NB * NB - schedule.length) + ' / ' + NB * NB : '—');
    TX.bi(caption,
      flash ? 'For each block of queries, stream through the key/value blocks it is allowed to see, keep a running max and normaliser per row, and write only the finished output rows. The n × n matrix never exists.' : 'Compute all of S, write it to HBM, read it back for the softmax, write P, read it back again to multiply by V. The arithmetic is cheap; moving n² numbers twice is not.',
      flash ? '对每个 query block，依次扫过它能看到的 key/value block，每行维护一个 running max 和归一化因子，最后只写回算完的输出行。n × n 矩阵从头到尾都不存在。' : '先算出整个 S 写进 HBM，做 softmax 时读回来，写出 P，再读回来乘以 V。算术本身很便宜，贵的是把 n² 个数搬两遍。');
  }
  TX.onLang(render);

  var elapsed = 0;
  var ticker = TX.loop(function (now, dt) {
    elapsed += dt;
    if (elapsed < (mode === 'flash' ? 520 : 170)) return;
    elapsed = 0;
    tick = tick + 1 > (mode === 'flash' ? FLASH_TICKS : STD_TICKS) ? 0 : tick + 1;
    render();
  });
  if (TX.reduced) tick = mode === 'flash' ? 4 : N;
  render();
  TX.whenVisible(fig.root, function () { if (playing) ticker.start(); }, function () { ticker.stop(); });
})();

/* 09 · RLHF step by step. Stage one and two in one move each; stage three walked one move at a
   time, so it is clear which of the four models acts at each step and that only two ever update. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-rlhf', { title: ['RLHF step by step · three stages, four models', 'RLHF 一步一步 · 三个阶段，四个模型'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  var id = 'rl' + Math.floor(Math.random() * 1e6);

  // [x, y, w, h, kind, title en, title zh, sub en, sub zh]
  var NODES = {
    1: { demo: [40, 118, 170, 64, 'data', 'Demonstrations', '示范数据', '(x, y*) written by people', '人写的 (x, y*)'],
         base: [275, 118, 150, 64, 'plain', 'Pretrained model', '预训练模型', 'next-token prediction', 'next-token prediction'],
         sft:  [490, 118, 170, 64, 'out', 'SFT model', 'SFT 模型', 'follows the instruction format', '会按指令格式回答'] },
    2: { pairs: [40, 118, 170, 64, 'data', 'Preference pairs', '偏好对', 'same prompt: y_w ≻ y_l', '同一 prompt：y_w ≻ y_l'],
         bt:    [275, 118, 150, 64, 'plain', 'Bradley–Terry loss', 'Bradley–Terry 损失', '−log σ(r_w − r_l)', '−log σ(r_w − r_l)'],
         rm:    [490, 118, 170, 64, 'out', 'Reward model r_φ', '奖励模型 r_φ', 'learns the gap only', '只学分差'] },
    3: { prompt: [20, 121, 86, 56, 'data', 'Prompt x', 'Prompt x', 'from the data', '来自数据'],
         actor:  [150, 103, 160, 92, 'train', 'Actor', 'Actor（策略）', 'copy of SFT', 'SFT 副本'],
         reward: [440, 18, 200, 62, 'frozen', 'Reward', 'Reward（奖励）', 'from stage 2', '第二阶段产物'],
         ref:    [440, 118, 200, 62, 'frozen', 'Reference', 'Reference（参考）', 'copy of SFT', 'SFT 副本'],
         critic: [440, 218, 200, 62, 'train', 'Critic', 'Critic（价值）', 'often init. from RM', '常从奖励模型初始化'],
         update: [150, 232, 160, 48, 'plain', 'PPO update', 'PPO 更新', 'clipped objective', 'clipped objective'] }
  };
  // [from xy, to xy, label]
  var EDGES = {
    1: { 'demo-base': [[210, 150], [275, 150], ['fine-tune', '微调']], 'base-sft': [[425, 150], [490, 150], ['', '']] },
    2: { 'pairs-bt': [[210, 150], [275, 150], ['', '']], 'bt-rm': [[425, 150], [490, 150], ['train', '训练']] },
    3: { 'prompt-actor': [[106, 149], [150, 149], ['', '']],
         'actor-reward': [[310, 124], [440, 52], ['y', 'y']],
         'actor-ref': [[310, 149], [440, 149], ['KL', 'KL']],
         'actor-critic': [[310, 174], [440, 246], ['V_t', 'V_t']],
         'update-actor': [[230, 232], [230, 195], ['', '']],
         'update-critic': [[310, 256], [440, 252], ['', '']] }
  };
  var STEPS = [
    { stage: 1, on: ['demo', 'base', 'sft'], edges: ['demo-base', 'base-sft'], upd: t('the model being fine-tuned', '被微调的那个模型'),
      en: 'Stage 1 · SFT. Fine-tune the pretrained model on demonstrations people wrote. The result answers in the instruction format, and it becomes the starting policy for everything that follows.',
      zh: '第一阶段 · SFT。用人写的示范数据微调预训练模型，得到一个会按指令格式回答的起点；后面 RL 的初始策略就是它。' },
    { stage: 2, on: ['pairs', 'bt', 'rm'], edges: ['pairs-bt', 'bt-rm'], upd: t('the reward model', '奖励模型'),
      en: 'Stage 2 · reward model. People see two answers to one prompt and pick the better one; a Bradley–Terry loss pushes r_φ to score the chosen answer above the rejected one. It only learns the gap, never an absolute score.',
      zh: '第二阶段 · 奖励模型。同一个 prompt 的两个回答，人标出哪个更好；Bradley–Terry 损失让 r_φ 给 chosen 打得比 rejected 高。它只学分差，不学绝对分。' },
    { stage: 3, on: ['actor', 'critic', 'reward', 'ref'], edges: [], upd: t('will train: Actor, Critic · frozen: Reward, Reference', '会训练：Actor、Critic · 冻结：Reward、Reference'),
      en: 'Stage 3 begins with four models. Actor and Reference start as two copies of the SFT model; Reward is the model from stage 2; the Critic is often initialised from the reward model.',
      zh: '第三阶段开始时有四个模型：Actor 和 Reference 是 SFT 模型的两份副本，Reward 是第二阶段训好的奖励模型，Critic 常从奖励模型初始化。' },
    { stage: 3, on: ['prompt', 'actor'], edges: ['prompt-actor', 'actor-reward'], upd: t('none, forward pass only', '无，只做前向'),
      en: '① The Actor reads a prompt x and samples an answer y, one token at a time. The Actor is the policy being optimised.',
      zh: '① Actor 读一个 prompt x，一枚一枚 token 采样出回答 y。它就是要被优化的策略。' },
    { stage: 3, on: ['reward'], edges: ['actor-reward'], upd: t('none, forward pass only', '无，只做前向'),
      en: '② The frozen Reward model scores the whole answer, r_φ(x, y): one number, given only when the answer ends.',
      zh: '② 冻结的 Reward 给完整回答打分 r_φ(x, y)：一个数，回答结束时才给。' },
    { stage: 3, on: ['ref', 'actor'], edges: ['actor-ref'], upd: t('none, forward pass only', '无，只做前向'),
      en: '③ The frozen Reference measures how far the Actor has drifted: total reward = r_φ − β · KL(π_θ ‖ π_ref).',
      zh: '③ 冻结的 Reference 衡量 Actor 跑了多远：总奖励 = r_φ − β · KL(π_θ ‖ π_ref)。' },
    { stage: 3, on: ['critic'], edges: ['actor-critic'], upd: t('none, forward pass only', '无，只做前向'),
      en: '④ The Critic estimates V_t for each prefix. The advantage A_t = R_t − V_t says how much better than expected each token turned out.',
      zh: '④ Critic 估计每个前缀的 V_t；优势 A_t = R_t − V_t 说明每个 token 比预期好了多少。' },
    { stage: 3, on: ['actor', 'critic', 'update'], edges: ['update-actor', 'update-critic'], upd: t('Actor and Critic, nothing else', 'Actor 和 Critic，仅此两个'), update: true,
      en: '⑤ One PPO step updates only the Actor and the Critic; Reward and Reference only ever run forward. Then back to ①, sampling again from the new Actor.',
      zh: '⑤ 一次 PPO 更新只改 Actor 和 Critic 的参数；Reward 和 Reference 全程只做前向。然后回到 ①，用新的 Actor 重新采样。' }
  ];
  var step = 0;

  var stageSeg = TX.seg([{ id: 1, en: '1 · SFT', zh: '1 · SFT' }, { id: 2, en: '2 · Reward model', zh: '2 · 奖励模型' }, { id: 3, en: '3 · RL', zh: '3 · RL' }], 1,
    function (v) { step = STEPS.findIndex(function (x) { return x.stage === v; }); render(); }, t('Stage', '阶段'));
  var prev = TX.button('← Back', '← 上一步', function () { step = Math.max(0, step - 1); render(); });
  var next = TX.button('Next →', '下一步 →', function () { step = step + 1 < STEPS.length ? step + 1 : 0; render(); });
  var count = h('span', { class: 'rl-count' });
  fig.controls.appendChild(stageSeg.el);
  fig.controls.appendChild(h('div', { class: 'rl-steps' }, [prev, count, next]));

  var svg = s('svg', { viewBox: '0 0 680 300', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var caption = h('p', { class: 'fig-caption' });
  var roStage = TX.readout('Stage', '阶段'), roUpd = TX.readout('Updating parameters in this step', '这一步更新参数的');
  fig.foot.appendChild(caption);
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roStage.el, roUpd.el]));

  function defs() {
    return s('defs', null, ['', 'on'].map(function (k) {
      return s('marker', { id: id + '-h' + k, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
        [s('path', { d: 'M0,0 L10,5 L0,10 z', class: k ? 'r-head-on' : 'r-head' })]);
    }));
  }
  function box(key, n, cls, stage3) {
    var g = s('g', { class: 'r-box r-' + n[4] + cls });
    g.appendChild(s('rect', { x: n[0], y: n[1], width: n[2], height: n[3], rx: 8 }));
    var cx = n[0] + n[2] / 2, cy = n[1] + n[3] / 2;
    g.appendChild(s('text', { class: 's-title', x: cx, y: cy - 3, 'text-anchor': 'middle', bi: [n[5], n[6]] }));
    g.appendChild(s('text', { class: 's-sub', x: cx, y: cy + 13, 'text-anchor': 'middle', bi: [n[7], n[8]] }));
    if (stage3 && (n[4] === 'train' || n[4] === 'frozen')) {
      var train = n[4] === 'train';
      g.appendChild(s('text', { class: 'r-badge ' + (train ? 'r-badge-train' : 'r-badge-frozen'), x: n[0] + n[2] - 8, y: n[1] + 14, 'text-anchor': 'end',
        bi: train ? ['trains', '训练'] : ['frozen', '冻结'] }));
    }
    return g;
  }
  function edge(e, on) {
    var g = s('g', { class: on ? '' : 'r-dim' });
    g.appendChild(s('line', { x1: e[0][0], y1: e[0][1], x2: e[1][0], y2: e[1][1], class: 'r-edge' + (on ? ' on' : ''), 'marker-end': 'url(#' + id + '-h' + (on ? 'on' : '') + ')' }));
    if (e[2][0]) {
      g.appendChild(s('text', { class: 'r-elabel' + (on ? ' on' : ''), x: (e[0][0] + e[1][0]) / 2 + 4, y: (e[0][1] + e[1][1]) / 2 - 6, bi: e[2] }));
    }
    return g;
  }

  function render() {
    var st = STEPS[step], nodes = NODES[st.stage], edges = EDGES[st.stage], all = st.stage < 3;
    TX.clear(svg);
    svg.appendChild(defs());
    Object.keys(edges).forEach(function (k) { svg.appendChild(edge(edges[k], all || st.edges.indexOf(k) >= 0)); });
    Object.keys(nodes).forEach(function (k) {
      var lit = all || st.on.indexOf(k) >= 0;
      var cls = (lit ? ' on' : ' r-dim') + (st.update && (k === 'actor' || k === 'critic') ? ' r-upd' : '');
      svg.appendChild(box(k, nodes[k], cls, st.stage === 3));
    });
    svg.setAttribute('aria-label', t(st.en, st.zh));
    stageSeg.set(st.stage);
    count.textContent = (step + 1) + ' / ' + STEPS.length;
    TX.bi(caption, st.en, st.zh);
    roStage.set([t('1 · SFT', '1 · SFT'), t('2 · Reward model', '2 · 奖励模型'), t('3 · RL (PPO)', '3 · RL（PPO）')][st.stage - 1]);
    roUpd.set(st.upd);
  }
  render();
})();

/* 10 · PPO clipping explorer. One sampled token: flip the sign of its advantage, drag the
   probability ratio, and see which of the four cases it is in and whether it still gets gradient. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-ppo-clip', { title: ['PPO clipping · drag ρ, flip the sign of A', 'PPO clipping · 拖动 ρ，切换 A 的正负'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  var id = 'pc' + Math.floor(Math.random() * 1e6);
  var sign = 1, rho = 1.3, eps = 0.2;
  var X0 = 58, X1 = 620, Y0 = 26, Y1 = 232, RMIN = 0.4, RMAX = 1.6;
  function sx(r) { return X0 + (r - RMIN) / (RMAX - RMIN) * (X1 - X0); }
  function sy(v) { var lo = sign > 0 ? 0.3 : -1.7, hi = sign > 0 ? 1.7 : -0.3; return Y1 - (v - lo) / (hi - lo) * (Y1 - Y0); }
  function obj(r) { return sign > 0 ? Math.min(r, 1 + eps) : -Math.max(r, 1 - eps); }
  function grad(r) { return sign > 0 ? (r < 1 + eps ? 1 : 0) : (r > 1 - eps ? -1 : 0); }

  var segA = TX.seg([{ id: 1, en: 'A > 0 · a good token', zh: 'A > 0 · 好 token' }, { id: -1, en: 'A < 0 · a bad token', zh: 'A < 0 · 坏 token' }], 1,
    function (v) { sign = v; render(); }, t('Sign of the advantage', 'advantage 的正负'));
  var sRho = TX.slider({ en: 'ratio ρ', zh: 'ratio ρ', min: 0.4, max: 1.6, step: 0.01, value: rho,
    format: function (v) { return v.toFixed(2); }, onInput: function (v) { rho = v; render(); } });
  var sEps = TX.slider({ en: 'clip range ε', zh: '裁剪范围 ε', min: 0.05, max: 0.4, step: 0.01, value: eps,
    format: function (v) { return '±' + v.toFixed(2); }, onInput: function (v) { eps = v; render(); } });
  fig.controls.appendChild(segA.el);
  fig.stage.appendChild(h('div', { class: 'ctl-row pc-sliders' }, [sRho.el, sEps.el]));
  var svg = s('svg', { viewBox: '0 0 640 280', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var caption = h('p', { class: 'fig-caption' });
  var insight = h('p', { class: 'fig-insight', bi: ['A decides the direction, ρ reports how far the probability has moved, and the clip only stops a correct move from going too far.', 'A 决定方向，ρ 报告步幅，clip 只阻止正确方向走得过头。'] });
  var roDir = TX.readout('ρ − 1 (what the new policy did)', 'ρ − 1（新策略做了什么）'), roSign = TX.readout('(ρ − 1) · A', '(ρ − 1) · A'),
      roGrad = TX.readout('∂ℓ / ∂ρ (still pushed?)', '∂ℓ / ∂ρ（还在推吗）'), roClip = TX.readout('Clipped?', '被裁了吗');
  fig.foot.appendChild(caption);
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roDir.el, roSign.el, roGrad.el, roClip.el]));
  fig.foot.appendChild(insight);

  function render() {
    TX.clear(svg);
    svg.appendChild(s('defs', null, [s('marker', { id: id + '-h', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
      [s('path', { d: 'M0,0 L10,5 L0,10 z', class: 'pc-pushhead' })])]));
    var lo = 1 - eps, hi = 1 + eps;
    // the plateau: where this term gives no gradient
    var px0 = sign > 0 ? sx(hi) : X0, px1 = sign > 0 ? X1 : sx(lo);
    svg.appendChild(s('rect', { class: 'pc-plateau', x: px0, y: Y0, width: Math.max(0, px1 - px0), height: Y1 - Y0 }));
    svg.appendChild(s('text', { class: 'pc-plabel', x: (px0 + px1) / 2, y: Y1 - 10, 'text-anchor': 'middle', bi: ['gradient = 0', '梯度 = 0'] }));
    // axes and guides
    svg.appendChild(s('line', { class: 'pc-axis', x1: X0, y1: Y1, x2: X1, y2: Y1 }));
    for (var r = 0.4; r <= 1.6001; r += 0.2) {
      svg.appendChild(s('line', { class: 'pc-axis', x1: sx(r), y1: Y1, x2: sx(r), y2: Y1 + 4 }));
      svg.appendChild(s('text', { class: 'pc-tick', x: sx(r), y: Y1 + 16, 'text-anchor': 'middle', text: r.toFixed(1) }));
    }
    [[lo, '1−ε'], [1, '1'], [hi, '1+ε']].forEach(function (g) {
      svg.appendChild(s('line', { class: 'pc-guide', x1: sx(g[0]), y1: Y0, x2: sx(g[0]), y2: Y1 }));
      svg.appendChild(s('text', { class: 'pc-tick pc-gl', x: sx(g[0]), y: Y1 + 30, 'text-anchor': 'middle', text: g[1] }));
    });
    svg.appendChild(s('text', { class: 'pc-tick', x: X1, y: Y1 + 44, 'text-anchor': 'end', bi: ['ρ = π_θ(a|s) / π_old(a|s), for one sampled token', 'ρ = π_θ(a|s) / π_old(a|s)，同一个已采样 token'] }));
    svg.appendChild(s('text', { class: 'pc-tick', x: X0 - 6, y: Y0 - 8, bi: ['objective ℓ(ρ)', '目标 ℓ(ρ)'] }));
    // unclipped ρ·A, then the clipped objective
    svg.appendChild(s('line', { class: 'pc-raw', x1: sx(RMIN), y1: sy(sign * RMIN), x2: sx(RMAX), y2: sy(sign * RMAX) }));
    svg.appendChild(s('text', { class: 'pc-tick', x: sx(sign > 0 ? 1.52 : 0.48), y: sy(sign * (sign > 0 ? 1.52 : 0.48)) - 8, 'text-anchor': 'middle', bi: ['ρ·A, unclipped', 'ρ·A（不裁剪）'] }));
    var d = '';
    for (var x = RMIN; x <= RMAX + 1e-9; x += 0.01) d += (d ? ' L' : 'M') + sx(x).toFixed(1) + ',' + sy(obj(x)).toFixed(1);
    svg.appendChild(s('path', { class: 'pc-obj', d: d }));
    // where this token is, and which way the gradient pushes ρ
    var mx = sx(rho), my = sy(obj(rho)), g = grad(rho);
    svg.appendChild(s('line', { class: 'pc-drop', x1: mx, y1: my, x2: mx, y2: Y1 }));
    if (g !== 0) {
      var dx = g > 0 ? 48 : -48;
      svg.appendChild(s('line', { class: 'pc-push', x1: mx, y1: my - 16, x2: mx + dx, y2: my - 16, 'marker-end': 'url(#' + id + '-h)' }));
      svg.appendChild(s('text', { class: 'pc-note', x: mx + dx / 2, y: my - 24, 'text-anchor': 'middle', bi: ['pushed', '继续推'] }));
    } else {
      svg.appendChild(s('text', { class: 'pc-note', x: mx, y: my - 14, 'text-anchor': 'middle', bi: ['no push', '不再推'] }));
    }
    svg.appendChild(s('circle', { class: 'pc-dot', cx: mx, cy: my, r: 6 }));

    var up = rho > 1, clipped = g === 0, right = (rho - 1) * sign > 0, flat = Math.abs(rho - 1) < 0.005;
    roDir.set((rho - 1 >= 0 ? '+' : '−') + Math.abs(rho - 1).toFixed(2) + ' · ' + (flat ? t('unchanged', '没变') : up ? t('probability raised', '概率提高了') : t('probability lowered', '概率降低了')));
    roSign.set(flat ? '0' : right ? t('> 0 · right direction', '> 0 · 方向正确') : t('< 0 · wrong direction', '< 0 · 方向错误'));
    roGrad.set(g > 0 ? '+1 · ' + t('raise it', '往上推') : g < 0 ? '−1 · ' + t('lower it', '往下推') : '0 · ' + t('left alone', '不管了'));
    roClip.set(clipped ? t('yes · on the plateau', '是 · 在平台上') : t('no', '否'));
    var c;
    if (sign > 0) {
      if (rho > hi) c = ['It raised the probability of a good token, the right direction, and is already past 1+ε: this term stops rewarding it. It does not pull ρ back into the range, it just stops pushing.', '提高了好 token 的概率，方向正确；已经超过 1+ε，这一项停止继续奖励。它不会把 ρ 拉回区间，只是不再往外推。'];
      else if (!flat && !up) c = ['It lowered the probability of a good token, the wrong direction. There is no lower clip for A > 0, so the gradient stays and pulls the probability back up.', '降低了好 token 的概率，方向错误；A > 0 时没有下界裁剪，梯度保留，把概率拉回来。'];
      else c = ['It raised the probability of a good token (or has not moved yet), and is still inside the range: keep encouraging it.', '提高了好 token 的概率（或还没动），而且还在区间内：继续鼓励。'];
    } else {
      if (rho < lo) c = ['It lowered the probability of a bad token, the right direction, and is already below 1−ε: this term stops rewarding it.', '降低了坏 token 的概率，方向正确；已经低于 1−ε，这一项停止继续奖励。'];
      else if (!flat && up) c = ['It raised the probability of a bad token, the wrong direction. There is no upper clip for A < 0, so the gradient stays and pushes the probability back down.', '提高了坏 token 的概率，方向错误；A < 0 时没有上界裁剪，梯度保留，把概率压下去。'];
      else c = ['It lowered the probability of a bad token (or has not moved yet), and is still inside the range: keep encouraging it.', '降低了坏 token 的概率（或还没动），而且还在区间内：继续鼓励。'];
    }
    TX.bi(caption, c[0], c[1]);
    svg.setAttribute('aria-label', t(c[0], c[1]));
  }
  render();
})();

/* 11 · MoE router. One token, eight experts: router logits → softmax → keep the top k →
   renormalise over the kept ones → the output is the gated sum of the chosen experts. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-moe-router', { title: ['MoE router · one token, eight experts', 'MoE router · 一个 token，八个 expert'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  var N = 8;
  var TOKENS = [
    { id: 0, en: '"return"', zh: '“return”', logits: [2.1, 0.3, -0.4, 1.6, -1.0, 0.2, -0.6, 0.5] },
    { id: 1, en: '"∫"', zh: '“∫”', logits: [-0.5, 2.4, 0.1, -0.8, 1.9, -0.2, 0.4, -1.1] },
    { id: 2, en: '"cat"', zh: '“猫”', logits: [0.2, -0.9, 1.8, 0.1, -0.4, 1.7, 0.9, -0.3] },
    { id: 3, en: '"the"', zh: '“的”', logits: [0.6, 0.4, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4] }
  ];
  var tok = 0, k = 2, noiseSeed = 0;

  var segTok = TX.seg(TOKENS.map(function (x) { return { id: x.id, en: x.en, zh: x.zh }; }), 0, function (v) { tok = v; noiseSeed = 0; render(); }, t('Token', 'Token'));
  var segK = TX.seg([{ id: 1, en: 'top-1', zh: 'top-1' }, { id: 2, en: 'top-2', zh: 'top-2' }, { id: 4, en: 'top-4', zh: 'top-4' }], 2, function (v) { k = v; render(); }, 'k');
  var noiseBtn = TX.button('Add router noise', '加一次 router 噪声', function () { noiseSeed += 1; render(); });
  var clearBtn = TX.button('No noise', '去掉噪声', function () { noiseSeed = 0; render(); });
  fig.controls.appendChild(segTok.el);
  fig.controls.appendChild(segK.el);
  fig.controls.appendChild(h('div', { class: 'rl-steps' }, [noiseBtn, clearBtn]));
  var svg = s('svg', { viewBox: '0 0 640 290', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var formula = h('p', { class: 'fig-insight mr-formula' });
  var caption = h('p', { class: 'fig-caption' });
  var roActive = TX.readout('Experts that run', '真正计算的 expert'), roShare = TX.readout('Share of expert parameters used', '用到的 expert 参数占比');
  fig.foot.appendChild(formula);
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roActive.el, roShare.el]));
  fig.foot.appendChild(caption);

  function render() {
    var base = TOKENS[tok].logits, rnd = TX.rng(1000 + tok * 17 + noiseSeed);
    var logits = base.map(function (v) { return noiseSeed ? v + (rnd() * 2 - 1) * 0.9 : v; });
    var mx = Math.max.apply(null, logits), ex = logits.map(function (v) { return Math.exp(v - mx); });
    var sum = ex.reduce(function (a, b) { return a + b; }, 0), p = ex.map(function (v) { return v / sum; });
    var order = p.map(function (v, i) { return i; }).sort(function (a, b) { return p[b] - p[a]; });
    var sel = order.slice(0, k), selSum = sel.reduce(function (a, i) { return a + p[i]; }, 0);
    var w = {}; sel.forEach(function (i) { w[i] = p[i] / selSum; });

    TX.clear(svg);
    svg.appendChild(s('defs', null, [s('marker', { id: 'mr-h', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, [s('path', { d: 'M0,0 L10,5 L0,10 z', class: 'r-head-on' })])]));
    // token → router
    function box(x, y, w0, h0, cls, a, b, c, d) {
      var g = s('g', { class: 'r-box ' + cls });
      g.appendChild(s('rect', { x: x, y: y, width: w0, height: h0, rx: 8 }));
      g.appendChild(s('text', { class: 's-title', x: x + w0 / 2, y: y + h0 / 2 - 2, 'text-anchor': 'middle', bi: [a, b] }));
      if (c) g.appendChild(s('text', { class: 's-sub', x: x + w0 / 2, y: y + h0 / 2 + 13, 'text-anchor': 'middle', bi: [c, d] }));
      return g;
    }
    svg.appendChild(box(14, 112, 92, 52, 'r-data on', TOKENS[tok].en, TOKENS[tok].zh, 'hidden state x', '隐藏状态 x'));
    svg.appendChild(box(132, 112, 104, 52, 'r-out on', 'Router', 'Router', 'logits = W_r · x', 'logits = W_r · x'));
    svg.appendChild(s('line', { class: 'r-edge on', x1: 106, y1: 138, x2: 132, y2: 138, 'marker-end': 'url(#mr-h)' }));
    // probability bars, one per expert
    var X0 = 268, COL = 45, BASE = 190, HMAX = 150;
    svg.appendChild(s('line', { class: 'pc-axis', x1: X0 - 6, y1: BASE, x2: X0 + N * COL, y2: BASE }));
    svg.appendChild(s('text', { class: 'pc-tick', x: X0 - 6, y: 22, bi: ['softmax over all 8, kept ones renormalised', '8 个一起 softmax，留下的再归一化'] }));
    for (var i = 0; i < N; i++) {
      var on = w[i] != null, x = X0 + i * COL, bh = Math.max(2, p[i] * HMAX);
      svg.appendChild(s('rect', { class: on ? 'mr-bar on' : 'mr-bar', x: x + 6, y: BASE - bh, width: COL - 14, height: bh, rx: 3 }));
      svg.appendChild(s('text', { class: 'pc-tick', x: x + COL / 2 - 1, y: BASE - bh - 6, 'text-anchor': 'middle', text: Math.round(p[i] * 100) + '%' }));
      var g = s('g', { class: 'r-box ' + (on ? 'r-train on' : 'r-dim') });
      g.appendChild(s('rect', { x: x + 3, y: 214, width: COL - 8, height: 34, rx: 6 }));
      g.appendChild(s('text', { class: 's-tag', x: x + COL / 2 - 1, y: 235, 'text-anchor': 'middle', text: 'E' + (i + 1) }));
      svg.appendChild(g);
      if (on) svg.appendChild(s('text', { class: 'pc-note', x: x + COL / 2 - 1, y: 266, 'text-anchor': 'middle', text: 'w=' + w[i].toFixed(2) }));
    }
    svg.appendChild(s('line', { class: 'r-edge on', x1: 236, y1: 138, x2: X0 - 8, y2: 138, 'marker-end': 'url(#mr-h)' }));
    svg.setAttribute('aria-label', t('Router probabilities for eight experts; the top ' + k + ' are kept.', '八个 expert 的 router 概率；保留最高的 ' + k + ' 个。'));

    formula.textContent = 'y = ' + sel.map(function (i) { return w[i].toFixed(2) + '·E' + (i + 1) + '(x)'; }).join(' + ');
    roActive.set(k + ' / ' + N + '  (' + sel.map(function (i) { return 'E' + (i + 1); }).join(', ') + ')');
    roShare.set(Math.round(k / N * 100) + '%');
    var gap = p[order[k - 1]] - p[order[k]];
    TX.bi(caption,
      (tok === 3 ? 'A function word: the router has no strong preference, so the probabilities are nearly flat and a little noise can change which experts run. ' : '') +
      'Only the ' + k + ' chosen experts compute anything; the other ' + (N - k) + ' are skipped for this token. The gap between the last kept and the first dropped expert is ' + (gap * 100).toFixed(1) + ' points' + (noiseSeed ? ', with noise added to the logits (noisy top-k, used in early MoE to spread load).' : '.'),
      (tok === 3 ? '功能词：router 没有明显偏好，概率几乎是平的，一点噪声就可能换掉被选中的 expert。' : '') +
      '只有被选中的 ' + k + ' 个 expert 真正计算，其余 ' + (N - k) + ' 个对这个 token 完全跳过。最后一个留下的和第一个被丢掉的，概率差 ' + (gap * 100).toFixed(1) + ' 个百分点' + (noiseSeed ? '；现在 logits 上加了噪声（noisy top-k，早期 MoE 用它来分散负载）。' : '。'));
  }
  render();
})();

/* 12 · Load balancing, as a toy. Tokens from eight topics of unequal frequency; the router
   reinforces whichever expert already gets a topic, so without balancing it collapses.
   Compare an auxiliary loss (pushes router probabilities towards uniform) with a per-expert
   bias that only changes which expert is selected (DeepSeek-V3's auxiliary-loss-free idea). */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-moe-balance', { title: ['Load balancing · a toy router, eight experts, top-1', '负载均衡 · 玩具 router，8 个 expert，top-1'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  var N = 8, T = 64, MAX_STEPS = 160;
  var freq = (function () { var f = [], z = 0; for (var i = 0; i < N; i++) { f.push(1 / Math.pow(i + 1, 0.7)); z += f[i]; } return f.map(function (v) { return v / z; }); })();
  var mode = 'none', cf = 1.25, A, b, step, rnd, loads, dropped, hist, playing = false;

  function reset() {
    rnd = TX.rng(42); A = []; b = []; step = 0; hist = []; loads = []; dropped = 0;
    for (var i = 0; i < N; i++) { A.push([]); b.push(0); for (var j = 0; j < N; j++) A[i].push((rnd() - 0.5) * 0.3 + (j === 0 ? 0.05 : 0)); }
    tick(true);
  }
  function sampleTopic() { var r = rnd(), c = 0; for (var i = 0; i < N; i++) { c += freq[i]; if (r < c) return i; } return N - 1; }
  function tick(quiet) {
    var cap = Math.ceil(cf * T / N), load = [], kept = [], drop = 0, i, j;
    for (j = 0; j < N; j++) { load.push(0); kept.push(0); }
    var routed = [];
    for (var n = 0; n < T; n++) {
      var topic = sampleTopic(), best = 0, bestScore = -1e9;
      for (j = 0; j < N; j++) { var sc = A[topic][j] + (mode === 'bias' ? b[j] : 0) + (rnd() - 0.5) * 0.4; if (sc > bestScore) { bestScore = sc; best = j; } }
      load[best]++;
      if (kept[best] < cap) { kept[best]++; routed.push([topic, best]); } else drop++;
    }
    // an expert that gets more tokens gets better, at its topics and in general: rich get richer
    routed.forEach(function (r) { A[r[0]][r[1]] += 0.001; });
    var f = load.map(function (v) { return v / T; });
    for (j = 0; j < N; j++) for (i = 0; i < N; i++) A[i][j] += 0.1 * (f[j] - 1 / N);
    if (mode === 'aux') for (j = 0; j < N; j++) for (i = 0; i < N; i++) A[i][j] -= 0.5 * (f[j] - 1 / N);
    if (mode === 'bias') for (j = 0; j < N; j++) b[j] -= 0.015 * Math.sign(f[j] - 1 / N);
    loads = load; dropped = drop; step++;
    var mean = T / N; hist.push(Math.max.apply(null, load) / mean);
    if (!quiet) render();
  }

  var segMode = TX.seg([{ id: 'none', en: 'No balancing', zh: '不做均衡' }, { id: 'aux', en: 'Auxiliary loss', zh: '辅助 loss' }, { id: 'bias', en: 'Bias only (aux-loss-free)', zh: '只调 bias（无辅助 loss）' }], 'none',
    function (v) { mode = v; reset(); render(); }, t('Balancing method', '均衡方式'));
  var play = TX.button('Play', '播放', function () { playing = !playing; if (step >= MAX_STEPS) { reset(); } sync(); });
  var again = TX.button('Reset', '重置', function () { reset(); render(); });
  var sCf = TX.slider({ en: 'capacity factor', zh: 'capacity factor', min: 1, max: 2, step: 0.25, value: cf, format: function (v) { return v.toFixed(2) + ' → ' + Math.ceil(v * T / N) + t(' slots', ' 个位置'); }, onInput: function (v) { cf = v; render(); } });
  fig.controls.appendChild(segMode.el);
  fig.controls.appendChild(h('div', { class: 'rl-steps' }, [play, again]));
  fig.stage.appendChild(h('div', { class: 'ctl-row pc-sliders' }, [sCf.el]));
  var svg = s('svg', { viewBox: '0 0 640 270', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var roStep = TX.readout('Training step', '训练步数'), roImb = TX.readout('Busiest expert ÷ average', '最忙 expert ÷ 平均'), roDrop = TX.readout('Tokens over capacity (dropped)', '超出容量被丢弃的 token');
  var caption = h('p', { class: 'fig-caption' });
  var note = h('p', { class: 'fig-note', bi: ['A toy, not a trained model: 64 tokens per step from eight topics of unequal frequency, and a router that gets better at whatever it is already sent. The shapes are illustrative; the numbers are not measurements.', '这是玩具模拟，不是真实训练：每步 64 个 token，来自出现频率不同的 8 个主题；router 会越来越擅长它已经在处理的东西。曲线形状用来说明问题，数值不是测量结果。'] });
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roStep.el, roImb.el, roDrop.el]));
  fig.foot.appendChild(caption);
  fig.foot.appendChild(note);

  function sync() { TX.bi(play, playing ? 'Pause' : 'Play', playing ? '暂停' : '播放'); }
  function render() {
    var cap = Math.ceil(cf * T / N), X0 = 40, COL = 44, BASE = 200, TOP = 14, SC = 90 / cap;
    TX.clear(svg);
    svg.appendChild(s('line', { class: 'pc-axis', x1: X0 - 4, y1: BASE, x2: X0 + N * COL, y2: BASE }));
    var capY = BASE - cap * SC;
    for (var j = 0; j < N; j++) {
      var L = loads[j] || 0, keep = Math.min(L, cap), over = L - keep, x = X0 + j * COL;
      var yKeep = BASE - keep * SC, yTop = Math.max(TOP, BASE - L * SC);
      svg.appendChild(s('rect', { class: 'mb-bar', x: x + 6, y: yKeep, width: COL - 12, height: keep * SC, rx: 3 }));
      if (over > 0) svg.appendChild(s('rect', { class: 'mb-over', x: x + 6, y: yTop, width: COL - 12, height: Math.max(0, yKeep - yTop), rx: 3 }));
      svg.appendChild(s('text', { class: 'pc-tick', x: x + COL / 2, y: BASE + 15, 'text-anchor': 'middle', text: 'E' + (j + 1) }));
      svg.appendChild(s('text', { class: over > 0 ? 'pc-plabel' : 'pc-tick', x: x + COL / 2, y: Math.max(TOP - 3, yTop - 5), 'text-anchor': 'middle', text: String(L) + (BASE - L * SC < TOP ? '↑' : '') }));
    }
    svg.appendChild(s('line', { class: 'mb-cap', x1: X0 - 4, y1: capY, x2: X0 + N * COL, y2: capY }));
    svg.appendChild(s('text', { class: 'pc-plabel', x: X0 + N * COL, y: capY - 5, 'text-anchor': 'end', bi: ['capacity', '容量'] }));
    // imbalance over time
    var GX = 420, GW = 200, GY = 40, GH = 140;
    svg.appendChild(s('rect', { class: 'mb-frame', x: GX, y: GY, width: GW, height: GH, rx: 6 }));
    svg.appendChild(s('text', { class: 'pc-tick', x: GX, y: GY - 8, bi: ['busiest ÷ average, over steps', '最忙 ÷ 平均，随训练步数'] }));
    var ymax = 8; function gy(v) { return GY + GH - Math.min(v, ymax) / ymax * GH; }
    svg.appendChild(s('line', { class: 'pc-guide', x1: GX, y1: gy(1), x2: GX + GW, y2: gy(1) }));
    svg.appendChild(s('text', { class: 'pc-tick', x: GX + GW - 4, y: gy(1) - 4, 'text-anchor': 'end', bi: ['1 = perfectly even', '1 = 完全均匀'] }));
    if (hist.length > 1) {
      var d = hist.map(function (v, i) { return (i ? 'L' : 'M') + (GX + i / MAX_STEPS * GW).toFixed(1) + ',' + gy(v).toFixed(1); }).join(' ');
      svg.appendChild(s('path', { class: 'pc-obj', d: d }));
    }
    var mean = T / N, imb = loads.length ? Math.max.apply(null, loads) / mean : 1;
    roStep.set(step + ' / ' + MAX_STEPS);
    roImb.set(imb.toFixed(2) + '×');
    roDrop.set(dropped + ' / ' + T + '  (' + Math.round(dropped / T * 100) + '%)');
    var msg = {
      none: ['Nothing pushes back: an expert that happens to get a topic gets better at it and attracts more. A few experts end up doing almost everything, the rest barely train, and tokens past capacity are dropped.', '没有任何反向约束：碰巧拿到某个主题的 expert 越学越擅长，吸走更多 token。最后少数几个 expert 干了几乎所有的活，其余的几乎学不到东西，超出容量的 token 被丢掉。'],
      aux: ['An auxiliary loss adds a penalty that grows with f_i · P_i, so overloaded experts lose router probability. Load evens out, but the penalty acts on the same router scores the language-model loss is trying to learn, and the two can pull against each other.', '辅助 loss 加了一个随 f_i · P_i 增大的惩罚，超载的 expert 会被压低 router 概率。负载会变均匀，但这个惩罚作用在语言模型 loss 想学的同一组 router 分数上，两者可能互相拉扯。'],
      bias: ['Each expert gets a bias that is nudged down when it is overloaded and up when it is idle. The bias only decides which expert is selected; the gating weight still comes from the original score, so no extra gradient reaches the router.', '每个 expert 有一个 bias：超载就调低一点，空闲就调高一点。bias 只影响选谁，门控权重仍然来自原始分数，所以没有额外梯度打到 router 上。']
    }[mode];
    TX.bi(caption, msg[0], msg[1]);
    svg.setAttribute('aria-label', t(msg[0], msg[1]));
  }
  var ticker = TX.loop(function (now, dt) {
    if (!playing) return;
    ticker.acc = (ticker.acc || 0) + dt;
    if (ticker.acc < 90) return;
    ticker.acc = 0;
    if (step >= MAX_STEPS) { playing = false; sync(); return; }
    tick();
  });
  reset();
  if (TX.reduced) { while (step < MAX_STEPS) tick(true); }
  render(); sync();
  TX.whenVisible(fig.root, function () { ticker.start(); }, function () { ticker.stop(); });
})();

/* 13 · Looped Transformer, folded and unrolled. One block of k layers reused L times:
   parameters stay at k layers, while depth, compute and per-token KV grow with L. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-loop-unroll', { title: ['Looping · one block, reused L times', '循环 · 同一个 block，重复用 L 次'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  var K = 4, loops = 3, inject = true;
  var sL = TX.slider({ en: 'loops L', zh: '循环次数 L', min: 1, max: 8, step: 1, value: loops, format: function (v) { return '× ' + v; }, onInput: function (v) { loops = v; render(); } });
  var segInj = TX.seg([{ id: 1, en: 'Re-inject the input each loop', zh: '每圈重新注入输入' }, { id: 0, en: 'Input only at the start', zh: '只在开头输入' }], 1, function (v) { inject = !!v; render(); }, t('Input injection', '输入注入'));
  fig.controls.appendChild(segInj.el);
  fig.stage.appendChild(h('div', { class: 'ctl-row pc-sliders' }, [sL.el]));
  var svg = s('svg', { viewBox: '0 0 640 330', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var roP = TX.readout('Parameters', '参数'), roD = TX.readout('Effective depth', '有效深度'), roC = TX.readout('Compute per token', '每个 token 的计算'), roKV = TX.readout('K/V per token (if every loop keeps its own)', '每个 token 的 K/V（每圈各存一份时）');
  var caption = h('p', { class: 'fig-caption' });
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roP.el, roD.el, roC.el, roKV.el]));
  fig.foot.appendChild(caption);

  function layers(g, x, y, w, hh, n, cls) {
    var lh = (hh - (n - 1) * 3) / n;
    for (var i = 0; i < n; i++) g.appendChild(s('rect', { class: cls, x: x, y: y + i * (lh + 3), width: w, height: Math.max(2, lh), rx: 3 }));
  }
  function render() {
    TX.clear(svg);
    svg.appendChild(s('defs', null, [s('marker', { id: 'lu-h', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, [s('path', { d: 'M0,0 L10,5 L0,10 z', class: 'r-head-on' })])]));
    // folded
    svg.appendChild(s('text', { class: 's-sub', x: 130, y: 18, 'text-anchor': 'middle', bi: ['folded: what is stored', '折叠：实际存下的'] }));
    var g = s('g'); layers(g, 70, 110, 120, 110, K, 'lu-layer'); svg.appendChild(g);
    svg.appendChild(s('text', { class: 's-title', x: 130, y: 102, 'text-anchor': 'middle', bi: ['block θ · ' + K + ' layers', 'block θ · ' + K + ' 层'] }));
    svg.appendChild(s('path', { class: 'r-edge on', d: 'M190,200 C240,200 240,130 190,130', fill: 'none', 'marker-end': 'url(#lu-h)' }));
    svg.appendChild(s('text', { class: 'pc-note', x: 236, y: 170, text: '× ' + loops }));
    svg.appendChild(s('rect', { class: 'lu-emb', x: 80, y: 252, width: 100, height: 30, rx: 6 }));
    svg.appendChild(s('text', { class: 's-sub', x: 130, y: 271, 'text-anchor': 'middle', bi: ['embedding e', 'embedding e'] }));
    svg.appendChild(s('line', { class: 'r-edge on', x1: 130, y1: 252, x2: 130, y2: 222, 'marker-end': 'url(#lu-h)' }));
    svg.appendChild(s('rect', { class: 'lu-emb', x: 80, y: 40, width: 100, height: 30, rx: 6 }));
    svg.appendChild(s('text', { class: 's-sub', x: 130, y: 59, 'text-anchor': 'middle', bi: ['output head', '输出层'] }));
    svg.appendChild(s('line', { class: 'r-edge on', x1: 130, y1: 108, x2: 130, y2: 72, 'marker-end': 'url(#lu-h)' }));
    // unrolled
    svg.appendChild(s('text', { class: 's-sub', x: 470, y: 18, 'text-anchor': 'middle', bi: ['unrolled: what is computed', '展开：实际算的'] }));
    var top = 40, bottom = 290, gap = 8, bh = (bottom - top - gap * (loops - 1)) / loops;
    for (var l = 0; l < loops; l++) {
      var y = bottom - (l + 1) * bh - l * gap, gg = s('g');
      layers(gg, 410, y, 120, bh, K, 'lu-layer');
      svg.appendChild(gg);
      svg.appendChild(s('text', { class: 'pc-tick', x: 540, y: y + bh / 2 + 4, bi: ['loop ' + (l + 1), '第 ' + (l + 1) + ' 圈'] }));
      if (inject) {
        svg.appendChild(s('line', { class: 'lu-inj', x1: 340, y1: y + bh / 2, x2: 406, y2: y + bh / 2, 'marker-end': 'url(#lu-h)' }));
      }
      if (l > 0) svg.appendChild(s('line', { class: 'r-edge on', x1: 470, y1: y + bh + gap - 1, x2: 470, y2: y + bh + 1 }));
    }
    svg.appendChild(s('line', { class: inject ? 'lu-inj' : 'lu-inj r-dim', x1: 340, y1: top, x2: 340, y2: bottom }));
    svg.appendChild(s('text', { class: 'pc-tick', x: 336, y: bottom + 16, 'text-anchor': 'middle', text: 'e' }));
    svg.appendChild(s('text', { class: 'pc-tick', x: 470, y: bottom + 16, 'text-anchor': 'middle', bi: ['same weights θ in every loop', '每一圈都是同一组权重 θ'] }));
    roP.set(K + t(' layers (does not grow with L)', ' 层（不随 L 增长）'));
    roD.set(K * loops + t(' layers', ' 层'));
    roC.set('≈ ' + K * loops + t(' layers of FLOPs', ' 层的计算量'));
    roKV.set(K * loops + t(' layers of K/V', ' 层的 K/V'));
    TX.bi(caption,
      'The stored model is ' + K + ' layers; running it ' + loops + ' times computes like a ' + K * loops + '-layer model. Depth and compute are now a dial you can turn after training; parameter count is not.' + (inject ? ' Re-feeding the embedding e into every loop keeps the original input in view no matter how many loops run.' : ' Without re-injection, the input is only seen at the start and has to survive every loop in the hidden state.'),
      '存下的模型只有 ' + K + ' 层；跑 ' + loops + ' 圈，计算上相当于 ' + K * loops + ' 层。深度和计算量变成了训练后还能拧的旋钮，参数量不变。' + (inject ? '每圈都把 embedding e 重新喂进去，不管转多少圈，原始输入都还看得见。' : '不重新注入时，输入只在开头出现一次，要靠隐藏状态一圈一圈保存下来。'));
  }
  render();
})();

/* 14 · Why loops help with multi-hop problems: an idealised chain where one pass of the block
   can follow one hop. A fixed-depth model stops at its depth; a looped one keeps going. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-loop-reach', { title: ['Multi-hop · each pass follows one more link', '多跳 · 每过一遍多走一跳'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  var hops = 6, loops = 3, FIXED = 4;
  var sH = TX.slider({ en: 'hops needed', zh: '需要几跳', min: 2, max: 10, step: 1, value: hops, format: function (v) { return String(v); }, onInput: function (v) { hops = v; render(); } });
  var sL = TX.slider({ en: 'loops', zh: '循环次数', min: 1, max: 10, step: 1, value: loops, format: function (v) { return '× ' + v; }, onInput: function (v) { loops = v; render(); } });
  fig.stage.appendChild(h('div', { class: 'ctl-row pc-sliders' }, [sH.el, sL.el]));
  var svg = s('svg', { viewBox: '0 0 640 220', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var roNeed = TX.readout('Hops needed', '需要的跳数'), roLoop = TX.readout('Looped block (1-layer params)', '循环 block（1 层的参数）'), roFixed = TX.readout('Fixed ' + FIXED + '-layer model (4× the params)', '固定 ' + FIXED + ' 层模型（4 倍参数）');
  var caption = h('p', { class: 'fig-caption' });
  var note = h('p', { class: 'fig-note', bi: ['Idealised on purpose: it assumes one pass through the block can follow exactly one link, as in pointer chasing or composing facts. Real models are messier, but the dependence of reachable hops on depth is the point results on looped models build on.', '这是刻意理想化的：假设每过一遍 block 正好能多跟一条链接，像 pointer chasing 或者把事实一条条串起来。真实模型没这么整齐，但「能走多少跳取决于深度」正是 looped 模型相关结果的出发点。'] });
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roNeed.el, roLoop.el, roFixed.el]));
  fig.foot.appendChild(caption);
  fig.foot.appendChild(note);

  function row(y, reach, label) {
    svg.appendChild(s('text', { class: 's-sub', x: 16, y: y + 4, bi: label }));
    var X0 = 150, step = Math.min(46, 470 / hops);
    for (var i = 0; i <= hops; i++) {
      var x = X0 + i * step, lit = i <= reach, goal = i === hops;
      if (i > 0) svg.appendChild(s('line', { class: 'r-edge' + (i <= reach ? ' on' : ''), x1: x - step + 9, y1: y, x2: x - 9, y2: y }));
      svg.appendChild(s('circle', { class: 'lr-node' + (lit ? ' on' : '') + (goal ? ' goal' : ''), cx: x, cy: y, r: 8 }));
    }
    var ok = reach >= hops;
    svg.appendChild(s('text', { class: ok ? 'pc-note' : 'pc-plabel', x: X0 + hops * step + 18, y: y + 4, bi: ok ? ['answer reached', '走到答案'] : ['stuck at hop ' + reach, '停在第 ' + reach + ' 跳'] }));
  }
  function render() {
    TX.clear(svg);
    svg.appendChild(s('text', { class: 'pc-tick', x: 150, y: 24, bi: ['start → … → answer', '起点 → … → 答案'] }));
    row(78, Math.min(loops, hops), ['looped block', '循环 block']);
    row(150, Math.min(FIXED, hops), ['fixed 4 layers', '固定 4 层']);
    roNeed.set(String(hops));
    roLoop.set(Math.min(loops, hops) >= hops ? t('reaches it', '能走到') : t('stops at ', '停在第 ') + Math.min(loops, hops) + t('', ' 跳'));
    roFixed.set(FIXED >= hops ? t('reaches it', '能走到') : t('stops at ', '停在第 ') + FIXED + t('', ' 跳'));
    var enough = loops >= hops;
    TX.bi(caption,
      (enough ? 'Enough loops: the looped block follows all ' + hops + ' links with the parameters of a single layer. ' : 'Not enough loops yet: add loops, not parameters. ') + 'The fixed model gets ' + FIXED + ' hops however hard the question is, because its depth was set at training time.',
      (enough ? '圈数够了：循环 block 只用一层的参数就跟完了全部 ' + hops + ' 跳。' : '圈数还不够：加的是圈数，不是参数。') + '固定深度的模型不管问题多难都只能走 ' + FIXED + ' 跳，因为它的深度在训练时就定死了。');
    svg.setAttribute('aria-label', t(caption.textContent, caption.textContent));
  }
  render();
})();

/* 15 · Adaptive depth. Each token has its own confidence curve over loops; it exits at the
   first loop where the exit score clears the threshold, capped at the maximum. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-loop-exit', { title: ['Adaptive depth · each token decides when to stop', '自适应深度 · 每个 token 自己决定转几圈'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  var MAXL = 8, tau = 0.85;
  var WORDS = TX.lang() === 'en'
    ? [['The', .05], ['cat', .25], ['sat', .2], ['on', .05], ['the', .05], ['mat', .3], ['because', .5], ['it', .9], ['was', .15], ['tired', .6]]
    : [['那只', .1], ['猫', .25], ['坐', .2], ['在', .05], ['垫子', .3], ['上', .05], ['因为', .5], ['它', .9], ['很', .1], ['累', .6]];
  function conf(d, l) { return 1 - Math.exp(-l * (0.35 + (1 - d) * 1.6)) * (0.6 + d * 0.4); }
  function exitAt(d) { for (var l = 1; l <= MAXL; l++) if (conf(d, l) >= tau) return l; return MAXL; }
  var sT = TX.slider({ en: 'exit threshold', zh: '退出阈值', min: 0.5, max: 0.99, step: 0.01, value: tau, format: function (v) { return v.toFixed(2); }, onInput: function (v) { tau = v; render(); } });
  fig.stage.appendChild(h('div', { class: 'ctl-row pc-sliders' }, [sT.el]));
  var svg = s('svg', { viewBox: '0 0 640 300', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var roAvg = TX.readout('Average loops per token', '平均每个 token 转几圈'), roSave = TX.readout('Compute vs always ' + MAXL, '相对固定转 ' + MAXL + ' 圈的计算量'), roCap = TX.readout('Tokens that hit the cap', '转满上限的 token');
  var caption = h('p', { class: 'fig-caption' });
  var note = h('p', { class: 'fig-note', bi: ['Toy confidence curves, not a trained gate. Ouro trains its exit gate with an entropy-regularised objective; Mixture-of-Recursions trains a router that assigns each token a depth. Both aim at what this picture shows.', '置信度曲线是玩具，不是训练出来的 gate。Ouro 用带熵正则的目标训练退出 gate；Mixture-of-Recursions 训练一个 router 给每个 token 分配深度。两者想达到的就是这张图的效果。'] });
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roAvg.el, roSave.el, roCap.el]));
  fig.foot.appendChild(caption);
  fig.foot.appendChild(note);

  function render() {
    TX.clear(svg);
    var X0 = 96, CW = 58, Y0 = 34, RH = 24;
    for (var l = 1; l <= MAXL; l++) svg.appendChild(s('text', { class: 'pc-tick', x: X0 + (l - 0.5) * CW, y: Y0 - 10, 'text-anchor': 'middle', bi: ['loop ' + l, '第 ' + l + ' 圈'] }));
    var total = 0, capped = 0;
    WORDS.forEach(function (w, i) {
      var y = Y0 + i * RH, e = exitAt(w[1]);
      total += e; if (e === MAXL && conf(w[1], MAXL) < tau) capped++;
      svg.appendChild(s('text', { class: 's-tag', x: X0 - 12, y: y + 15, 'text-anchor': 'end', text: w[0] }));
      for (var l = 1; l <= MAXL; l++) {
        var used = l <= e;
        svg.appendChild(s('rect', { class: 'le-cell' + (used ? ' on' : '') + (l === e ? ' exit' : ''), x: X0 + (l - 1) * CW + 3, y: y + 3, width: CW - 6, height: RH - 6, rx: 3, 'fill-opacity': used ? (0.35 + 0.65 * conf(w[1], l)).toFixed(2) : null }));
      }
    });
    var avg = total / WORDS.length;
    roAvg.set(avg.toFixed(1) + ' / ' + MAXL);
    roSave.set(Math.round(avg / MAXL * 100) + '%');
    roCap.set(String(capped));
    TX.bi(caption,
      'Function words stop after a loop or two; "it", which has to be resolved to "the cat", keeps going. Raising the threshold buys accuracy with compute; lowering it saves compute and risks stopping before a hard token is settled.',
      '功能词一两圈就停；“它”要回指到“猫”，会一直转下去。阈值调高，是用计算换准确；调低，省计算，但难的 token 可能还没想清楚就停了。');
    svg.setAttribute('aria-label', t(caption.textContent, caption.textContent));
  }
  render();
})();

/* 16 · The agent loop, on one small task: fix a failing test. In agent mode the model picks the
   next step after every observation and keeps going until the tests pass; in workflow mode the
   steps are fixed in code, which is cheaper and predictable but cannot react to a new failure. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-agent-loop', { title: ['The agent loop · fixing a failing test', 'Agent 循环 · 修一个失败的测试'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  // [actor, en, zh, tool call or '', result en, result zh, tests state]
  var AGENT = [
    ['env', 'Read the task and the failing log', '读任务和失败日志', '', 'test_parse_date: expected 2026-09-22, got 2026-22-09', 'test_parse_date：期望 2026-09-22，实际 2026-22-09', '2 failing'],
    ['model', 'Month and day look swapped in a format string', '日期格式里月和日好像写反了', '', '', '', '2 failing'],
    ['tool', 'Search the code', '搜索代码', 'search_code("strftime")', 'utils/dates.py:14  fmt = "%Y-%d-%m"', 'utils/dates.py:14  fmt = "%Y-%d-%m"', '2 failing'],
    ['tool', 'Fix the format', '改掉格式', 'edit_file("utils/dates.py", "%Y-%d-%m" → "%Y-%m-%d")', 'file saved', '已保存', '2 failing'],
    ['tool', 'Run the tests', '跑测试', 'run_tests()', '41 passed, 1 failed: test_parse_date_tz', '41 通过，1 失败：test_parse_date_tz', '1 failing'],
    ['model', 'A second failure: the parser drops the timezone', '还有一个：解析时把时区丢了', '', '', '', '1 failing'],
    ['tool', 'Fix it and run again', '改掉再跑', 'edit_file(...); run_tests()', '42 passed', '42 全部通过', 'all pass'],
    ['stop', 'Stop: the stopping condition is met; report the diff and the test output', '停下：满足停止条件，汇报改动和测试结果', '', '', '', 'all pass']
  ];
  var FLOW = [AGENT[0], AGENT[2], AGENT[3], AGENT[4],
    ['stop', 'Stop: the script has no more steps, one test still fails', '停下：脚本写好的步骤走完了，还有一个测试没过', '', '', '', '1 failing']];
  var mode = 'agent', step = 0;
  var segMode = TX.seg([{ id: 'agent', en: 'Agent: the model picks the next step', zh: 'Agent：模型决定下一步' }, { id: 'flow', en: 'Workflow: fixed steps', zh: 'Workflow：固定步骤' }], 'agent',
    function (v) { mode = v; step = 0; render(); }, t('Mode', '模式'));
  var prev = TX.button('← Back', '← 上一步', function () { step = Math.max(0, step - 1); render(); });
  var next = TX.button('Next →', '下一步 →', function () { var n = (mode === 'agent' ? AGENT : FLOW).length; step = step + 1 < n ? step + 1 : 0; render(); });
  var count = h('span', { class: 'rl-count' });
  fig.controls.appendChild(segMode.el);
  fig.controls.appendChild(h('div', { class: 'rl-steps' }, [prev, count, next]));
  var svg = s('svg', { viewBox: '0 0 640 200', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var trace = h('ol', { class: 'ag-trace' });
  fig.stage.appendChild(trace);
  var roModel = TX.readout('Model calls so far', '到目前为止的模型调用'), roTool = TX.readout('Tool calls so far', '到目前为止的工具调用'), roTests = TX.readout('Tests', '测试');
  var caption = h('p', { class: 'fig-caption' });
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roModel.el, roTool.el, roTests.el]));
  fig.foot.appendChild(caption);

  var NODE = { model: [60, 70, 150, 60, 'Model', '模型', 'decides the next step', '决定下一步'], tool: [260, 70, 150, 60, 'Tools', '工具', 'search · edit · run', '搜索 · 编辑 · 运行'], env: [460, 70, 150, 60, 'Environment', '环境', 'repo · test results', '代码仓库 · 测试结果'] };
  function render() {
    var steps = mode === 'agent' ? AGENT : FLOW, cur = steps[step];
    TX.clear(svg);
    svg.appendChild(s('defs', null, ['', 'on'].map(function (k) {
      return s('marker', { id: 'ag-h' + k, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, [s('path', { d: 'M0,0 L10,5 L0,10 z', class: k ? 'r-head-on' : 'r-head' })]);
    })));
    var lit = { model: cur[0] === 'model' || cur[0] === 'stop', tool: cur[0] === 'tool', env: cur[0] === 'env' || cur[0] === 'tool' };
    if (mode === 'flow') lit.model = false;
    function edge(x1, y1, x2, y2, on, label, curve) {
      var d = curve ? 'M' + x1 + ',' + y1 + ' C' + x1 + ',' + (y1 + 60) + ' ' + x2 + ',' + (y2 + 60) + ' ' + x2 + ',' + y2 : 'M' + x1 + ',' + y1 + ' L' + x2 + ',' + y2;
      svg.appendChild(s('path', { class: 'r-edge' + (on ? ' on' : ''), d: d, fill: 'none', 'marker-end': 'url(#ag-h' + (on ? 'on' : '') + ')' }));
      if (label) svg.appendChild(s('text', { class: 'r-elabel' + (on ? ' on' : ''), x: curve ? (x1 + x2) / 2 : (x1 + x2) / 2, y: curve ? y1 + 52 : y1 - 8, 'text-anchor': 'middle', bi: label }));
    }
    edge(210, 92, 260, 92, cur[0] === 'tool', ['act', '行动']);
    edge(410, 92, 460, 92, cur[0] === 'tool', ['', '']);
    edge(535, 130, 135, 130, cur[0] === 'env' || cur[0] === 'model', ['observe', '观察'], true);
    Object.keys(NODE).forEach(function (k) {
      var n = NODE[k], g = s('g', { class: 'r-box ' + (k === 'model' ? 'r-out' : k === 'tool' ? 'r-train' : 'r-data') + (lit[k] ? ' on' : ' r-dim') });
      g.appendChild(s('rect', { x: n[0], y: n[1], width: n[2], height: n[3], rx: 8 }));
      g.appendChild(s('text', { class: 's-title', x: n[0] + n[2] / 2, y: n[1] + 26, 'text-anchor': 'middle', bi: [n[4], n[5]] }));
      g.appendChild(s('text', { class: 's-sub', x: n[0] + n[2] / 2, y: n[1] + 43, 'text-anchor': 'middle', bi: [n[6], n[7]] }));
      svg.appendChild(g);
    });
    if (mode === 'flow') svg.appendChild(s('text', { class: 'pc-note', x: 135, y: 56, 'text-anchor': 'middle', bi: ['steps fixed in code', '步骤写死在代码里'] }));
    svg.appendChild(s('text', { class: 'pc-tick', x: 320, y: 22, 'text-anchor': 'middle', bi: [cur[1], cur[2]] }));
    // the trace so far
    TX.clear(trace);
    var models = 0, tools = 0;
    steps.forEach(function (st, i) {
      if (i > step) return;
      if (st[0] === 'model' || (mode === 'agent' && (st[0] === 'tool' || st[0] === 'stop'))) models++;
      if (st[0] === 'tool') tools += st[3].indexOf(';') >= 0 ? 2 : 1;
      var li = h('li', { class: i === step ? 'on' : '' }, [h('span', { class: 'ag-who', bi: [{ env: 'observe', model: 'think', tool: 'act', stop: 'stop' }[st[0]], { env: '观察', model: '思考', tool: '行动', stop: '停止' }[st[0]]] }), h('span', { bi: [st[1], st[2]] })]);
      if (st[3]) li.appendChild(h('code', { text: st[3] }));
      if (st[4]) li.appendChild(h('span', { class: 'ag-out', bi: ['→ ' + st[4], '→ ' + st[5]] }));
      trace.appendChild(li);
    });
    count.textContent = (step + 1) + ' / ' + steps.length;
    roModel.set(String(models));
    roTool.set(String(tools));
    roTests.set({ '2 failing': t('2 failing', '2 个失败'), '1 failing': t('1 failing', '1 个失败'), 'all pass': t('all passing', '全部通过') }[cur[6]]);
    var end = step === steps.length - 1;
    TX.bi(caption,
      mode === 'agent' ? (end ? 'The agent found a second failure it was never told about and kept going until the stopping condition held. It also made a model call before every action, which is where the extra cost and variance come from.' : 'After every observation the model chooses what to do next; nothing about the path is fixed in advance.')
                       : (end ? 'The workflow did exactly what it was written to do, cheaply and the same way every time, and stopped with a failure it had no step for. For well-understood tasks that predictability is the point.' : 'Each step is written in code; the model, if used at all, fills in one step and never chooses the next one.'),
      mode === 'agent' ? (end ? 'Agent 发现了一个没人告诉它的新失败，一直做到满足停止条件为止。代价是每次行动前都有一次模型调用，额外的成本和不确定性就来自这里。' : '每看到一次结果，模型就自己决定下一步做什么；路径没有事先写死。')
                       : (end ? 'Workflow 完全按写好的步骤走：便宜、每次都一样，但遇到没写到的失败就停在那里。对已经想清楚的任务，这种可预测正是它的价值。' : '每一步都写在代码里；模型就算用上，也只负责填其中一步，不决定下一步。'));
    svg.setAttribute('aria-label', t(cur[1], cur[2]));
  }
  render();
})();

/* 17 · Frontier API or self-hosted: a rough monthly cost model. Every default is an example to
   replace with real quotes; the point is the shape — a per-token line against a step function
   with a fixed floor. */
(function () {
  'use strict';
  var TX = window.TX;
  if (!TX) return;
  var fig = TX.figure('tx-agent-cost', { title: ['Frontier API or self-hosted · a rough monthly cost model', '用 API 还是自己 serve · 粗略的月成本模型'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;
  var P = { tasks: 3, ktok: 60, pin: 3, pout: 15, gpuHr: 2.5, gpus: 8, tps: 6000, util: 0.5, ops: 12000 };
  function fmt$(v) { return '$' + (v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : v >= 1e4 ? Math.round(v / 1e3) + 'k' : v >= 1000 ? (v / 1e3).toFixed(1) + 'k' : Math.round(v).toString()); }
  function tasksOf(x) { return Math.pow(10, x); }
  function monthTok(tasks) { return tasks * 30 * P.ktok * 1000; }
  function api(tasks) { return monthTok(tasks) * (0.8 * P.pin + 0.2 * P.pout) / 1e6; }
  function replicas(tasks) { return Math.max(1, Math.ceil(monthTok(tasks) / (P.tps * 3600 * 24 * 30 * P.util))); }
  function self(tasks) { return replicas(tasks) * P.gpus * P.gpuHr * 24 * 30 + P.ops; }
  function sl(key, en, zh, min, max, step, f) {
    return TX.slider({ en: en, zh: zh, min: min, max: max, step: step, value: P[key], format: f, onInput: function (v) { P[key] = v; render(); } }).el;
  }
  var usage = h('div', { class: 'calc' }, [h('div', { class: 'calc-title', bi: ['Workload', '负载'] }), h('div', { class: 'calc-controls' }, [
    sl('tasks', 'tasks per day', '每天任务数', 1, 6, 0.1, function (v) { return Math.round(tasksOf(v)).toLocaleString('en-US'); }),
    sl('ktok', 'tokens per task (all steps)', '每个任务的 token（所有步骤）', 5, 400, 5, function (v) { return v + 'k'; })])]);
  var apiBox = h('div', { class: 'calc' }, [h('div', { class: 'calc-title', bi: ['API (example prices)', 'API（示例价格）'] }), h('div', { class: 'calc-controls' }, [
    sl('pin', '$ per 1M input tokens', '每 1M 输入 token 的价格', 0.1, 20, 0.1, function (v) { return '$' + v.toFixed(1); }),
    sl('pout', '$ per 1M output tokens', '每 1M 输出 token 的价格', 0.5, 80, 0.5, function (v) { return '$' + v.toFixed(1); })])]);
  var selfBox = h('div', { class: 'calc' }, [h('div', { class: 'calc-title', bi: ['Self-hosted (example numbers)', '自己 serve（示例数字）'] }), h('div', { class: 'calc-controls' }, [
    sl('gpuHr', '$ per GPU-hour', '每 GPU 小时', 0.5, 10, 0.1, function (v) { return '$' + v.toFixed(1); }),
    sl('gpus', 'GPUs per replica', '每组副本的 GPU 数', 1, 16, 1, function (v) { return String(v); }),
    sl('tps', 'tokens/s per replica', '每组副本每秒 token', 500, 30000, 500, function (v) { return v.toLocaleString('en-US'); }),
    sl('util', 'average utilisation', '平均利用率', 0.1, 0.9, 0.05, function (v) { return Math.round(v * 100) + '%'; }),
    sl('ops', 'people and ops per month', '每月人力与运维', 0, 60000, 1000, function (v) { return fmt$(v); })])]);
  fig.stage.appendChild(h('div', { class: 'ctl-row ac-panels' }, [usage, apiBox, selfBox]));
  var svg = s('svg', { viewBox: '0 0 640 260', role: 'img' });
  fig.stage.appendChild(h('div', { class: 'fig-scroll' }, [svg]));
  var roApi = TX.readout('API per month', 'API 每月'), roSelf = TX.readout('Self-hosted per month', '自己 serve 每月'), roRep = TX.readout('Replicas needed', '需要几组副本'), roBE = TX.readout('Break-even volume', '两者持平的任务量');
  var caption = h('p', { class: 'fig-caption' });
  var note = h('p', { class: 'fig-note', bi: ['A rough model: input is taken as 80% of tokens; throughput lumps prefill and decode together; replicas run around the clock. Replace every default with your own quotes and measurements before using it for a decision.', '这是粗略模型：假设输入占 80% 的 token；吞吐把 prefill 和 decode 合在一起算；副本全天运行。用它做决定之前，把每个默认值换成你自己的报价和实测。'] });
  fig.foot.appendChild(h('div', { class: 'readouts' }, [roApi.el, roSelf.el, roRep.el, roBE.el]));
  fig.foot.appendChild(caption);
  fig.foot.appendChild(note);

  var X0 = 70, X1 = 620, Y0 = 20, Y1 = 214, XMIN = 1, XMAX = 6;
  function sx(x) { return X0 + (x - XMIN) / (XMAX - XMIN) * (X1 - X0); }
  function render() {
    var ys = [], i, x;
    for (x = XMIN; x <= XMAX + 1e-9; x += 0.05) ys.push(api(tasksOf(x)), self(tasksOf(x)));
    var lo = Math.log10(Math.max(1, Math.min.apply(null, ys))), hi = Math.log10(Math.max.apply(null, ys));
    lo = Math.floor(lo); hi = Math.ceil(hi); if (hi - lo < 2) hi = lo + 2;
    function sy(v) { return Y1 - (Math.log10(Math.max(1, v)) - lo) / (hi - lo) * (Y1 - Y0); }
    TX.clear(svg);
    svg.appendChild(s('line', { class: 'pc-axis', x1: X0, y1: Y1, x2: X1, y2: Y1 }));
    svg.appendChild(s('line', { class: 'pc-axis', x1: X0, y1: Y0, x2: X0, y2: Y1 }));
    for (x = XMIN; x <= XMAX; x++) {
      svg.appendChild(s('text', { class: 'pc-tick', x: sx(x), y: Y1 + 16, 'text-anchor': x === XMAX ? 'end' : 'middle', text: Math.round(tasksOf(x)).toLocaleString('en-US') }));
    }
    svg.appendChild(s('text', { class: 'pc-tick', x: X1, y: Y1 + 32, 'text-anchor': 'end', bi: ['tasks per day (log scale)', '每天任务数（对数刻度）'] }));
    for (var e = lo; e <= hi; e++) {
      svg.appendChild(s('line', { class: 'pc-guide', x1: X0, y1: sy(Math.pow(10, e)), x2: X1, y2: sy(Math.pow(10, e)) }));
      svg.appendChild(s('text', { class: 'pc-tick', x: X0 - 6, y: sy(Math.pow(10, e)) + 4, 'text-anchor': 'end', text: fmt$(Math.pow(10, e)) }));
    }
    function path(fn) { var d = ''; for (var xx = XMIN; xx <= XMAX + 1e-9; xx += 0.01) d += (d ? ' L' : 'M') + sx(xx).toFixed(1) + ',' + sy(fn(tasksOf(xx))).toFixed(1); return d; }
    svg.appendChild(s('path', { class: 'ac-api', d: path(api) }));
    svg.appendChild(s('path', { class: 'ac-self', d: path(self) }));
    svg.appendChild(s('text', { class: 'ac-lapi', x: X1 - 4, y: sy(api(tasksOf(XMAX))) - 6, 'text-anchor': 'end', bi: ['API', 'API'] }));
    svg.appendChild(s('text', { class: 'ac-lself', x: sx(XMIN) + 6, y: sy(self(tasksOf(XMIN))) - 6, bi: ['self-hosted', '自己 serve'] }));
    // break-even: the first volume where self-hosting gets cheaper
    var be = null;
    for (x = XMIN; x <= XMAX + 1e-9; x += 0.01) if (self(tasksOf(x)) <= api(tasksOf(x))) { be = x; break; }
    if (be != null) {
      svg.appendChild(s('line', { class: 'pc-drop', x1: sx(be), y1: Y0, x2: sx(be), y2: Y1 }));
      svg.appendChild(s('text', { class: 'pc-note', x: sx(be) + 4, y: Y0 + 12, bi: ['break-even', '持平点'] }));
    }
    var now = tasksOf(P.tasks);
    svg.appendChild(s('circle', { class: 'ac-dot-api', cx: sx(P.tasks), cy: sy(api(now)), r: 5 }));
    svg.appendChild(s('circle', { class: 'ac-dot', cx: sx(P.tasks), cy: sy(self(now)), r: 5 }));
    var a = api(now), b = self(now);
    roApi.set(fmt$(a));
    roSelf.set(fmt$(b));
    roRep.set(String(replicas(now)));
    roBE.set(be == null ? t('none in range', '范围内没有') : Math.round(tasksOf(be)).toLocaleString('en-US') + t(' / day', ' / 天'));
    TX.bi(caption,
      (a < b ? 'At this volume the API is cheaper: self-hosting pays a fixed floor of GPUs that run all day plus the people who run them, whether or not traffic arrives. ' : 'At this volume self-hosting is cheaper: the fixed floor is spread over enough tokens. ') + 'Cost is only one axis: the capability gap, latency, data rules, and whether you need to fine-tune usually decide first.',
      (a < b ? '在这个量级上 API 更便宜：自己 serve 有一块固定的底，全天运行的 GPU 加上维护它的人，不管有没有流量都要付。' : '在这个量级上自己 serve 更便宜：固定成本被足够多的 token 摊薄了。') + '成本只是一个维度：能力差距、延迟、数据合规、要不要微调，往往先一步决定了答案。');
    svg.setAttribute('aria-label', t(caption.textContent, caption.textContent));
  }
  render();
})();

/* --------------------------------------------- 18 three agents, one maze
   The oldest thing the word "agent" pointed at: something that senses a little
   world and moves in it. Three drivers take turns on the same maze — written
   rules, a value function, and a model asked every few steps — so the eras of
   the word sit side by side instead of in a paragraph. */
(function () {
  var fig = TX.figure('tx-agent-maze', { title: ['One maze, three kinds of agent', '同一个迷宫，三种 agent'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;

  var MAZE = [
    '###################',
    '#........#........#',
    '#.##.###.#.###.##.#',
    '#.................#',
    '#.##.#.#####.#.##.#',
    '#....#...#...#....#',
    '##.#####.#.#####.##',
    '#........#........#',
    '#.##.###...###.##.#',
    '#.................#',
    '###################'
  ];
  var COLS = 19, ROWS = 11, CELL = 26, PAD = 10, TICK = 165;
  var W = COLS * CELL + PAD * 2, H = ROWS * CELL + PAD * 2;
  var START = { x: 1, y: 1 }, LAIR = { x: 9, y: 5 };
  var DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

  function open(x, y) { return x >= 0 && y >= 0 && x < COLS && y < ROWS && MAZE[y][x] === '.'; }
  function idx(x, y) { return y * COLS + x; }
  function px(v) { return PAD + v * CELL + CELL / 2; }

  var driver = 'rules', showValues = false, running = true;
  var dots, agent, ghost, eaten, steps, caught, calls, lastChoice;
  var value = new Float32Array(COLS * ROWS);
  var reward = new Float32Array(COLS * ROWS);
  var rand = TX.rng(7);

  function reset(hard) {
    if (hard) { eaten = 0; steps = 0; caught = 0; calls = 0; }
    dots = {};
    for (var y = 0; y < ROWS; y++) for (var x = 0; x < COLS; x++) if (open(x, y)) dots[idx(x, y)] = true;
    delete dots[idx(START.x, START.y)];
    agent = { x: START.x, y: START.y, px: START.x, py: START.y, dir: [1, 0] };
    ghost = { x: LAIR.x, y: LAIR.y, px: LAIR.x, py: LAIR.y };
    lastChoice = [1, 0];
  }

  // ---- the three drivers ----------------------------------------------------
  // Written rules: walk towards the nearest dot as the crow flies. No memory, no
  // lookahead, so a wall between here and there is not its problem.
  function byRules() {
    var best = null, bestD = Infinity, key;
    for (key in dots) {
      var dx = (key % COLS) - agent.x, dy = Math.floor(key / COLS) - agent.y;
      var d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = [dx, dy]; }
    }
    if (!best) return [0, 0];
    var wish = Math.abs(best[0]) > Math.abs(best[1])
      ? [[Math.sign(best[0]), 0], [0, Math.sign(best[1])]]
      : [[0, Math.sign(best[1])], [Math.sign(best[0]), 0]];
    for (var i = 0; i < wish.length; i++) {
      var w = wish[i];
      if ((w[0] || w[1]) && open(agent.x + w[0], agent.y + w[1])) return w;
    }
    var legal = DIRS.filter(function (d) { return open(agent.x + d[0], agent.y + d[1]); });
    return legal.length ? legal[Math.floor(rand() * legal.length)] : [0, 0];
  }

  // A value function: how much is still to be had from each square, if you keep
  // playing well. Dots pay 1, the ghost costs a lot, and γ discounts the future,
  // so the field routes around walls on its own.
  function sweepValues() {
    var i, x, y;
    for (i = 0; i < reward.length; i++) reward[i] = 0;
    for (var key in dots) reward[key] = 1;
    reward[idx(ghost.x, ghost.y)] = -14;
    DIRS.forEach(function (d) {
      if (open(ghost.x + d[0], ghost.y + d[1])) reward[idx(ghost.x + d[0], ghost.y + d[1])] -= 6;
      var far = [ghost.x + d[0] * 2, ghost.y + d[1] * 2];
      if (open(far[0], far[1])) reward[idx(far[0], far[1])] -= 2;
    });
    for (var pass = 0; pass < 70; pass++) {
      for (y = 1; y < ROWS - 1; y++) for (x = 1; x < COLS - 1; x++) {
        if (!open(x, y)) continue;
        var best = -1e9;
        for (var k = 0; k < 4; k++) {
          var nx = x + DIRS[k][0], ny = y + DIRS[k][1];
          if (open(nx, ny) && value[idx(nx, ny)] > best) best = value[idx(nx, ny)];
        }
        value[idx(x, y)] = reward[idx(x, y)] + 0.9 * (best === -1e9 ? 0 : best);
      }
    }
  }
  function byValue() {
    var best = null, bestV = -1e9;
    DIRS.forEach(function (d) {
      var nx = agent.x + d[0], ny = agent.y + d[1];
      if (!open(nx, ny)) return;
      var v = value[idx(nx, ny)] + rand() * 1e-3;     // break ties, never a straight line forever
      if (v > bestV) { bestV = v; best = d; }
    });
    return best || [0, 0];
  }

  // Asking a model: the same judgement, but only every fifth step, because each
  // decision is a network call you wait for and pay for. In between it coasts.
  function byModel() {
    if (steps % 5 === 0) { calls++; lastChoice = byValue(); }
    var straight = [lastChoice[0], lastChoice[1]];
    if (!open(agent.x + straight[0], agent.y + straight[1])) { calls++; lastChoice = byValue(); }
    return lastChoice;
  }

  function chaseStep() {
    var options = DIRS.filter(function (d) { return open(ghost.x + d[0], ghost.y + d[1]); });
    if (!options.length) return [0, 0];
    if (rand() < 0.3) return options[Math.floor(rand() * options.length)];
    var best = options[0], bestD = Infinity;
    options.forEach(function (d) {
      var dx = ghost.x + d[0] - agent.x, dy = ghost.y + d[1] - agent.y;
      var dist = dx * dx + dy * dy;
      if (dist < bestD) { bestD = dist; best = d; }
    });
    return best;
  }

  function tick() {
    sweepValues();
    var move = driver === 'rules' ? byRules() : driver === 'rl' ? byValue() : byModel();
    var wasAgent = { x: agent.x, y: agent.y }, wasGhost = { x: ghost.x, y: ghost.y };
    if (open(agent.x + move[0], agent.y + move[1])) {
      agent.x += move[0]; agent.y += move[1];
      if (move[0] || move[1]) agent.dir = move;
    }
    steps++;
    if (dots[idx(agent.x, agent.y)]) { delete dots[idx(agent.x, agent.y)]; eaten++; }
    var g = chaseStep();
    ghost.x += g[0]; ghost.y += g[1];
    var swapped = agent.x === wasGhost.x && agent.y === wasGhost.y && ghost.x === wasAgent.x && ghost.y === wasAgent.y;
    if (swapped || (agent.x === ghost.x && agent.y === ghost.y)) {
      caught++;
      agent.x = START.x; agent.y = START.y;
      ghost.x = LAIR.x; ghost.y = LAIR.y;
    }
    var left = 0;
    for (var _ in dots) { left++; break; }
    if (!left) reset(false);
  }

  // ---- drawing --------------------------------------------------------------
  var svg = s('svg', { class: 'tx-svg am-svg', viewBox: '0 0 ' + W + ' ' + H, role: 'img' });
  var gValue = s('g', { class: 'am-values' }), gWall = s('g'), gDots = s('g'), gCast = s('g');
  [gValue, gWall, gDots, gCast].forEach(function (g) { svg.appendChild(g); });
  for (var wy = 0; wy < ROWS; wy++) for (var wx = 0; wx < COLS; wx++) {
    if (open(wx, wy)) continue;
    gWall.appendChild(s('rect', { class: 'am-wall', x: PAD + wx * CELL + 3, y: PAD + wy * CELL + 3, width: CELL - 6, height: CELL - 6, rx: 3 }));
  }
  var dotNodes = {}, valueNodes = {};
  for (var dy2 = 0; dy2 < ROWS; dy2++) for (var dx2 = 0; dx2 < COLS; dx2++) {
    if (!open(dx2, dy2)) continue;
    var key2 = idx(dx2, dy2);
    dotNodes[key2] = gDots.appendChild(s('rect', { class: 'am-dot', x: px(dx2) - 2, y: px(dy2) - 2, width: 4, height: 4 }));
    valueNodes[key2] = [0, 1, 2, 3].map(function (n) {
      return gValue.appendChild(s('rect', {
        class: 'am-bit', width: 3, height: 3,
        x: PAD + dx2 * CELL + 5 + (n % 2) * 10, y: PAD + dy2 * CELL + 5 + Math.floor(n / 2) * 10
      }));
    });
  }
  var pac = gCast.appendChild(s('path', { class: 'am-pac' }));
  var spook = gCast.appendChild(s('path', { class: 'am-ghost' }));
  var eyeL = gCast.appendChild(s('circle', { class: 'am-eye', r: 2.2 }));
  var eyeR = gCast.appendChild(s('circle', { class: 'am-eye', r: 2.2 }));

  function drawPac(cx, cy, dir, mouth) {
    var a = Math.atan2(dir[1], dir[0]), r = CELL * 0.42, gap = mouth * 0.9;
    var x1 = cx + r * Math.cos(a + gap), y1 = cy + r * Math.sin(a + gap);
    var x2 = cx + r * Math.cos(a - gap), y2 = cy + r * Math.sin(a - gap);
    pac.setAttribute('d', 'M' + cx.toFixed(1) + ',' + cy.toFixed(1) + ' L' + x1.toFixed(1) + ',' + y1.toFixed(1) +
      ' A' + r + ',' + r + ' 0 1 0 ' + x2.toFixed(1) + ',' + y2.toFixed(1) + ' Z');
  }
  function drawGhost(cx, cy) {
    var r = CELL * 0.38, top = cy - r * 0.9, bot = cy + r * 0.85;
    var d = 'M' + (cx - r) + ',' + bot + ' L' + (cx - r) + ',' + cy + ' A' + r + ',' + r + ' 0 0 1 ' + (cx + r) + ',' + cy + ' L' + (cx + r) + ',' + bot;
    for (var i = 0; i < 3; i++) d += ' l' + (-r * 2 / 3 / 2) + ',' + (-r * 0.28) + ' l' + (-r * 2 / 3 / 2) + ',' + (r * 0.28);
    spook.setAttribute('d', d + ' Z');
    spook.setAttribute('data-top', top);
    eyeL.setAttribute('cx', cx - r * 0.38); eyeL.setAttribute('cy', cy - r * 0.18);
    eyeR.setAttribute('cx', cx + r * 0.38); eyeR.setAttribute('cy', cy - r * 0.18);
  }

  function paint(phase) {
    var key;
    for (key in dotNodes) dotNodes[key].style.display = dots[key] ? '' : 'none';
    if (showValues) {
      var lo = Infinity, hi = -Infinity;
      for (key in valueNodes) { var v = value[key]; if (v < lo) lo = v; if (v > hi) hi = v; }
      var span = hi - lo || 1;
      for (key in valueNodes) {
        var lit = Math.round(((value[key] - lo) / span) * 4);
        valueNodes[key].forEach(function (node, n) { node.style.display = n < lit ? '' : 'none'; });
      }
    } else {
      for (key in valueNodes) valueNodes[key].forEach(function (node) { node.style.display = 'none'; });
    }
    var mouth = 0.18 + 0.32 * Math.abs(Math.sin(phase * Math.PI * 2));
    drawPac(px(agent.px), px(agent.py), agent.dir, mouth);
    drawGhost(px(ghost.px), px(ghost.py));
    roEat.set(String(eaten));
    roStep.set(String(steps));
    roCaught.set(String(caught));
    roCalls.set(driver === 'llm' ? String(calls) : '—');
  }

  // ---- controls -------------------------------------------------------------
  var CAPTIONS = {
    rules: ['Written rules: head for the nearest dot as the crow flies. Fast, free, the same every time — and blind to walls, so it presses against one while the dot sits on the other side. Every maze game before the 1990s was some version of this, and it is what most people still picture when they hear "agent".',
      '写死的规则：哪颗豆直线距离最近就往哪边走。快、不要钱、每次都一样——但它看不见墙，于是会贴着墙一直顶，而豆就在墙那边。九十年代以前的迷宫游戏基本都是这么写的，多数人一听 agent 想到的也还是这个。'],
    rl: ['A value function: how much is still to be had from each square if you keep playing well. Dots pay 1, the ghost costs a lot, and γ = 0.9 discounts what is further away. Nobody told it about walls — the field flows around them because a square behind a wall is simply far in steps. Turn on the value map and watch where it glows.',
      '价值函数：站在每一格，往后还能拿多少分。豆 +1，鬼扣一大笔，γ = 0.9 表示越远的收益越不值钱。没人告诉它墙在哪儿——墙后面的格子「要走很多步才到」，价值自然就低，路线是算出来的。打开价值图，看哪片亮着。'],
    llm: ['Asking a model each step: same judgement, but you wait for a network call and pay for it, so here it only re-decides every fifth step and coasts in between — watch it overshoot corners. What you buy is the thing the other two cannot do: change the goal by saying so, in a sentence, without retraining anything.',
      '每一步问一次模型：判断力差不多，但每次决策都要等一趟网络、花一次钱，所以这里让它每 5 步才重新决定，中间沿着上次的方向滑——会看到它冲过路口。换来的是另外两个做不到的事：想改目标，说一句话就行，不用重训。']
  };
  var caption = h('p', { class: 'fig-note' });
  var seg = TX.seg([
    { id: 'rules', en: 'Written rules', zh: '写死的规则' },
    { id: 'rl', en: 'A value function', zh: '价值函数' },
    { id: 'llm', en: 'Ask a model', zh: '每步问模型' }
  ], driver, function (id) { driver = id; TX.bi(caption, CAPTIONS[id][0], CAPTIONS[id][1]); }, t('Driver', '谁在做决定'));
  var valueSeg = TX.seg([
    { id: 'off', en: 'Maze', zh: '迷宫' },
    { id: 'on', en: 'Value map', zh: '价值图' }
  ], 'off', function (id) { showValues = id === 'on'; }, t('Overlay', '叠加层'));
  var pauseBtn = TX.button('Pause', '暂停', function () {
    running = !running;
    TX.bi(pauseBtn, running ? 'Pause' : 'Resume', running ? '暂停' : '继续');
  });
  var resetBtn = TX.button('Start over', '重来', function () { reset(true); });
  var roEat = TX.readout('Dots', '吃到的豆'), roStep = TX.readout('Steps', '步数');
  var roCaught = TX.readout('Caught', '被抓'), roCalls = TX.readout('Model calls', '模型调用');
  TX.append(fig.controls, [seg.el, valueSeg.el, pauseBtn, resetBtn]);
  fig.stage.appendChild(svg);
  TX.append(fig.foot, [h('div', { class: 'readouts' }, [roEat.el, roStep.el, roCaught.el, roCalls.el]), caption]);
  TX.bi(caption, CAPTIONS.rules[0], CAPTIONS.rules[1]);

  reset(true);
  sweepValues();
  var carry = 0;
  var loop = TX.loop(function (now, dt) {
    if (running) carry += dt;
    while (carry >= TICK) { carry -= TICK; tick(); }
    var phase = carry / TICK;
    agent.px += (agent.x - agent.px) * 0.35;      // ease towards the new square
    agent.py += (agent.y - agent.py) * 0.35;
    ghost.px += (ghost.x - ghost.px) * 0.3;
    ghost.py += (ghost.y - ghost.py) * 0.3;
    paint(phase);
  });
  TX.whenVisible(fig.root, function () { loop.start(); }, function () { loop.stop(); });
  svg.setAttribute('aria-label', t('A maze where an agent eats dots while a ghost chases it; a control chooses whether written rules, a value function or a model decides each step.',
    '一个迷宫：agent 一边吃豆一边被鬼追；上面的开关决定每一步是写死的规则、价值函数，还是模型在做决定。'));
})();

/* --------------------------------------------- 19 who gets the request
   Four ways to spend a request across a small model and a frontier one, on the
   same traffic: everything small, everything frontier, a cascade that escalates
   when it is unsure, and a router that decides up front. The judge's accuracy is
   the slider that matters — it is what makes the middle two worth anything. */
(function () {
  var fig = TX.figure('tx-model-router', { title: ['Who gets the request · small, frontier, cascade, router', '这条请求交给谁 · 小模型、frontier、级联、路由'] });
  if (!fig) return;
  var s = TX.s, h = TX.h, t = TX.t;

  var P = { hard: 35, small: 60, judge: 75, cost: 15 };
  // accuracy of each model on easy and on hard traffic; the small model's ceiling
  // moves with the "small model" slider, the frontier one stays where it is
  function acc() {
    var q = P.small / 100;
    return {
      sEasy: 0.72 + 0.26 * q, sHard: 0.18 + 0.52 * q,
      fEasy: 0.98, fHard: 0.86
    };
  }
  function plans() {
    var a = acc(), hard = P.hard / 100, easy = 1 - hard, j = P.judge / 100, C = P.cost;
    var out = {};
    out.small = { q: easy * a.sEasy + hard * a.sHard, c: 1, lat: 1 };
    out.frontier = { q: easy * a.fEasy + hard * a.fHard, c: C, lat: 2.2 };
    // cascade: the small model answers first, a check escalates what looks wrong.
    // Judging an answer you can see is easier than guessing from the question
    // alone, so the cascade's check is better than the router's by construction.
    var jc = j + (1 - j) * 0.45;
    var upHard = jc, upEasy = 1 - jc;
    var escalated = hard * upHard + easy * upEasy;
    out.cascade = {
      q: easy * ((1 - upEasy) * a.sEasy + upEasy * a.fEasy) + hard * ((1 - upHard) * a.sHard + upHard * a.fHard),
      c: 1 + escalated * C,
      lat: 1 + escalated * 2.2
    };
    // router: one classifier up front, so a request is paid for once
    var sent = hard * j + easy * (1 - j);
    out.router = {
      q: easy * ((1 - (1 - j)) * a.sEasy + (1 - j) * a.fEasy) + hard * ((1 - j) * a.sHard + j * a.fHard),
      c: 0.05 + sent * C + (1 - sent) * 1,
      lat: 0.1 + sent * 2.2 + (1 - sent) * 1
    };
    return out;
  }

  var W = 560, H = 300, X0 = 60, X1 = W - 18, Y0 = H - 40, Y1 = 18;
  var svg = s('svg', { class: 'tx-svg mr-svg', viewBox: '0 0 ' + W + ' ' + H, role: 'img' });
  var gGrid = s('g'), gDots = s('g');
  svg.appendChild(gGrid); svg.appendChild(gDots);
  var PLANS = [
    { id: 'small', en: 'all small', zh: '全用小模型' },
    { id: 'cascade', en: 'cascade', zh: '级联' },
    { id: 'router', en: 'router', zh: '路由' },
    { id: 'frontier', en: 'all frontier', zh: '全用 frontier' }
  ];

  function render() {
    var p = plans(), id;
    TX.clear(gGrid); TX.clear(gDots);
    var maxC = Math.max(p.frontier.c, p.cascade.c) * 1.08, minQ = 1;
    for (id in p) minQ = Math.min(minQ, p[id].q);
    var loQ = Math.max(0, minQ - 0.06), hiQ = 1;
    var sx = function (c) { return X0 + (Math.log(c) / Math.log(maxC)) * (X1 - X0); };
    var sy = function (q) { return Y0 - ((q - loQ) / (hiQ - loQ)) * (Y0 - Y1); };
    [0.25, 0.5, 0.75, 1].forEach(function (frac) {
      var q = loQ + (hiQ - loQ) * frac;
      gGrid.appendChild(s('line', { class: 'pc-guide', x1: X0, y1: sy(q), x2: X1, y2: sy(q) }));
      gGrid.appendChild(s('text', { class: 'pc-tick', x: X0 - 8, y: sy(q) + 4, 'text-anchor': 'end', text: Math.round(q * 100) + '%' }));
    });
    [1, 3, 10, 30].forEach(function (c) {
      if (c > maxC) return;
      gGrid.appendChild(s('text', { class: 'pc-tick', x: sx(c), y: Y0 + 16, 'text-anchor': 'middle', text: '×' + c }));
    });
    gGrid.appendChild(s('text', { class: 'pc-axis', x: (X0 + X1) / 2, y: H - 6, 'text-anchor': 'middle', bi: ['cost per request, relative to the small model', '每条请求的成本，以小模型为 1'] }));
    gGrid.appendChild(s('text', { class: 'pc-axis', x: 14, y: (Y0 + Y1) / 2, 'text-anchor': 'middle', transform: 'rotate(-90 14 ' + (Y0 + Y1) / 2 + ')', bi: ['answers that hold up', '答对的比例'] }));
    // the frontier line: nothing below and to the right of a cheaper, better plan
    PLANS.forEach(function (plan) {
      var d = p[plan.id];
      var dominated = PLANS.some(function (o) {
        return o.id !== plan.id && p[o.id].c <= d.c * 0.999 && p[o.id].q >= d.q * 1.001;
      });
      var g = s('g', { class: 'mr-plan' + (dominated ? ' is-out' : '') });
      g.appendChild(s('circle', { class: 'mr-dot mr-' + plan.id, cx: sx(d.c), cy: sy(d.q), r: 7 }));
      var right = sx(d.c) < (X0 + X1) / 2;
      g.appendChild(s('text', {
        class: 'mr-label', x: sx(d.c) + (right ? 13 : -13), y: sy(d.q) + 4,
        'text-anchor': right ? 'start' : 'end', bi: [plan.en, plan.zh]
      }));
      gDots.appendChild(g);
    });
    roQ.set(Math.round(p.cascade.q * 100) + '% · ' + t('cascade', '级联'));
    roC.set('×' + p.cascade.c.toFixed(1));
    roR.set(Math.round(p.router.q * 100) + '% · ×' + p.router.c.toFixed(1));
    roF.set(Math.round(p.frontier.q * 100) + '% · ×' + p.frontier.c.toFixed(1));
    roL.set('×' + p.cascade.lat.toFixed(1) + ' · ×' + p.router.lat.toFixed(1));
    var lead = p.cascade.q >= p.frontier.q - 0.015
      ? ['With a judge this good, the cascade keeps frontier-level quality for a fraction of the spend: most requests are settled by the small model, and only the doubtful ones are paid for twice. What it cannot avoid is the wait — those requests run two models one after the other.',
         '判断这么准的时候，级联能守住接近 frontier 的质量，只花一部分钱：多数请求小模型就结了，只有拿不准的付两遍。躲不掉的是等待——那些请求要串着跑两个模型。']
      : p.judge < 55
        ? ['The judge is close to guessing, and both middle plans fall apart: the cascade escalates the wrong things and pays twice for them, the router sends hard work to the small model. Below about this line, pick one model and keep the system simple.',
           '判断准头接近瞎猜，中间两种就塌了：级联升级错了对象，还为它们付两遍钱；路由把难题派给小模型。低到这个程度，不如挑一个模型，把系统做简单。']
        : ['The cascade trades some quality for a lot of cost; the router is cheaper still because nothing is paid for twice, but it commits before seeing the answer, so a wrong guess is never caught.',
           '级联用一点质量换掉很多成本；路由更便宜，因为没有一条请求被付两次，但它在看到答案之前就下注，猜错了也没人接住。'];
    TX.bi(note, lead[0], lead[1]);
    svg.setAttribute('aria-label', t('Cost against quality for four ways to serve the same traffic.', '四种服务同一批流量的方式，在成本和质量上的位置。'));
  }

  var note = h('p', { class: 'fig-note' });
  var sHard = TX.slider({ en: 'Hard requests', zh: '难题占比', min: 0, max: 100, step: 5, value: P.hard, format: function (v) { return v + '%'; }, onInput: function (v) { P.hard = v; render(); } });
  var sSmall = TX.slider({ en: 'How good the small model is', zh: '小模型有多强', min: 0, max: 100, step: 5, value: P.small, format: function (v) { return v + '%'; }, onInput: function (v) { P.small = v; render(); } });
  var sJudge = TX.slider({ en: 'How well difficulty is judged', zh: '难度判断的准头', min: 40, max: 98, step: 2, value: P.judge, format: function (v) { return v + '%'; }, onInput: function (v) { P.judge = v; render(); } });
  var sCost = TX.slider({ en: 'Frontier price, ×small', zh: 'Frontier 的价格（小模型的几倍）', min: 2, max: 40, step: 1, value: P.cost, format: function (v) { return '×' + v; }, onInput: function (v) { P.cost = v; render(); } });
  var roQ = TX.readout('Cascade', '级联'), roC = TX.readout('Cascade cost', '级联成本');
  var roR = TX.readout('Router', '路由'), roF = TX.readout('All frontier', '全用 frontier');
  var roL = TX.readout('Wait: cascade · router', '等待：级联 · 路由');
  fig.controls.appendChild(h('div', { class: 'ctl-row' }, [sHard.el, sSmall.el, sJudge.el, sCost.el]));
  fig.stage.appendChild(svg);
  TX.append(fig.foot, [h('div', { class: 'readouts' }, [roQ.el, roC.el, roR.el, roF.el, roL.el]), note]);
  render();
})();
