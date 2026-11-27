(function () {
  'use strict';

  function stepState(step) {
    if (!Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Invalid decoding step');
    const count = 3 + step;
    const tokens = ['A', 'B', 'C', 'D', 'E', 'F'].slice(0, count);
    return {tokens, input: step === 0 ? 'A B C' : tokens.at(-1), prediction: String.fromCharCode(65 + count),
      projected: {plain: count, cached: step === 0 ? count : 1},
      scores: {plain: count * (count + 1) / 2, cached: step === 0 ? count * (count + 1) / 2 : count},
      reused: step === 0 ? 0 : count - 1};
  }

  function locationState(href) {
    const url = new URL(href);
    const requested = url.searchParams.get('view');
    const rawStep = url.searchParams.get('step');
    return {mode: url.hash ? 'full' : ['quick', 'lab', 'full'].includes(requested) ? requested : 'quick',
      step: /^[0-3]$/.test(rawStep || '') ? Number(rawStep) : 1};
  }

  function viewURL(href, mode, step) {
    if (!['quick', 'lab', 'full'].includes(mode)) throw new RangeError('Invalid learning view');
    stepState(step);
    const url = new URL(href);
    url.searchParams.set('view', mode);
    if (mode === 'lab') url.searchParams.set('step', String(step));
    else url.searchParams.delete('step');
    url.hash = '';
    return url.href;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = {stepState, locationState, viewURL};
  if (typeof document === 'undefined') return;
  const root = document.querySelector('[data-learning-views="kv-cache"]');
  if (!root) return;
  const chinese = root.dataset.lang === 'zh';
  const select = selector => root.querySelector(selector);
  const all = selector => [...root.querySelectorAll(selector)];
  const full = select('[data-learning-panel="full"]');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const languageLinks = [...document.querySelectorAll('.topbar a[data-language-switch]')];
  const languageTargets = languageLinks.map(link => link.href);
  let {mode, step} = locationState(location.href);
  let timer = null;

  function stop() {
    clearInterval(timer);
    timer = null;
    select('#kv-play').textContent = chinese ? '播放' : 'Play';
    select('#kv-play').setAttribute('aria-pressed', 'false');
  }

  function syncLinks() {
    languageLinks.forEach((link, index) => { link.href = viewURL(languageTargets[index], mode, step); });
    all('a[data-learning-open]').forEach(link => { link.href = viewURL(location.href, link.dataset.learningOpen, step); });
  }

  function tokenElement(token, state, label) {
    const element = document.createElement('div');
    element.className = `token ${state}`;
    const symbol = document.createElement('strong');
    symbol.textContent = token;
    const detail = document.createElement('small');
    detail.textContent = label;
    element.append(symbol, detail);
    return element;
  }

  function render() {
    const state = stepState(step);
    select('#kv-step-index').textContent = `${step + 1} / 4`;
    select('#kv-current-input').textContent = state.input;
    select('#kv-prediction').textContent = state.prediction;
    select('#kv-no-cache-tokens').replaceChildren(...state.tokens.map((token, index) =>
      tokenElement(token, step > 0 && index === state.tokens.length - 1 ? 'new' : '', chinese ? '计算' : 'compute')));
    select('#kv-cache-tokens').replaceChildren(...state.tokens.map((token, index) => {
      const cached = index < state.reused;
      return tokenElement(token, cached ? 'cached' : 'new', chinese ? cached ? '复用' : '计算' : cached ? 'reuse' : 'compute');
    }));
    select('#kv-no-cache-count').textContent = state.projected.plain;
    select('#kv-cache-count').textContent = state.projected.cached;
    select('#kv-no-cache-scores').textContent = state.scores.plain;
    select('#kv-cache-scores').textContent = state.scores.cached;
    select('#kv-cache-explanation').textContent = step === 0 ? chinese ? '处理输入，建立 A、B、C 的缓存。' : 'Process the input and create its cache.' :
      chinese ? `复用前 ${state.reused} 个 token，只给 ${state.input} 算新的 K、V。` : `Reuse the first ${state.reused} tokens; compute ${state.input}.`;
    select('#kv-step-explanation').textContent = step === 0 ? chinese ? 'Prefill 一次处理 A、B、C，预测出 D。此时缓存里仍然只有 A、B、C。' :
      'Prefill processes A, B, C together. It predicts D, but the cache still only holds A, B, C.' : chinese ?
      `把 ${state.input} 输入模型后，缓存才扩展到 ${state.tokens.join('、')}。这一步预测出 ${state.prediction}，但 ${state.prediction} 的 K、V 要等下一步才会算。` :
      `Feed ${state.input} into the model and append its keys and values. The cache now holds ${state.tokens.join(', ')}. The model predicts ${state.prediction}, whose keys and values will be computed on the next step.`;
    // the code beside the lab: the lines this step runs, and what the cache holds afterwards
    all('[data-code-steps]').forEach(line => line.classList.toggle('is-current', line.dataset.codeSteps.split(' ').includes(String(step))));
    const codeState = select('#kv-code-state');
    if (codeState) codeState.textContent = step === 0
      ? chinese ? 'Prefill：一次算出 A、B、C 的 K、V。缓存里现在是 A、B、C。' : 'Prefill: keys and values for A, B, C in one pass. The cache now holds A, B, C.'
      : chinese ? `解码 ${step}：只投影 ${state.input}，接到缓存后面。缓存里现在是 ${state.tokens.join('、')}。`
        : `Decode ${step}: project ${state.input} only and append it. The cache now holds ${state.tokens.join(', ')}.`;
    select('#kv-previous').disabled = step === 0;
    select('#kv-next').disabled = step === 3;
    all('[data-step]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.step) === step)));
    syncLinks();
  }

  function setMode(nextMode, update = false, focus = false) {
    stop();
    mode = nextMode;
    all('[data-learning-panel]').forEach(panel => { panel.hidden = panel.dataset.learningPanel !== mode; });
    all('[data-learning-mode]').forEach(button => {
      const selected = button.dataset.learningMode === mode;
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
      if (focus && selected) button.focus();
    });
    document.body.dataset.learningView = mode;
    if (update) history.pushState(null, '', viewURL(location.href, mode, step));
    syncLinks();
  }

  function restoreLocation() {
    const state = locationState(location.href);
    step = state.step;
    render();
    setMode(state.mode);
    if (location.hash) requestAnimationFrame(() => {
      let anchor;
      try { anchor = decodeURIComponent(location.hash.slice(1)); } catch { return; }
      document.getElementById(anchor)?.scrollIntoView({block: 'start'});
    });
  }

  function move(nextStep) {
    stop();
    step = nextStep;
    render();
    history.replaceState(null, '', viewURL(location.href, mode, step));
  }

  root.addEventListener('click', event => {
    const opener = event.target.closest('[data-learning-mode], [data-learning-open]');
    if (!opener || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    setMode(opener.dataset.learningMode || opener.dataset.learningOpen, true, true);
    if (opener.hasAttribute('data-learning-open')) select('[data-learning-nav]').scrollIntoView({block: 'start'});
  });
  select('[role="tablist"]').addEventListener('keydown', event => {
    const modes = ['quick', 'lab', 'full'];
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const index = modes.indexOf(mode);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (index + (event.key === 'ArrowLeft' ? 2 : 1)) % 3;
    setMode(modes[next], true, true);
  });
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href]');
    if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const target = new URL(link.href, location.href);
    if (target.origin !== location.origin || target.pathname !== location.pathname || !target.hash) return;
    let anchor;
    try { anchor = decodeURIComponent(target.hash.slice(1)); } catch { return; }
    if (full.contains(document.getElementById(anchor))) setMode('full');
  }, true);
  all('[data-step]').forEach(button => button.addEventListener('click', () => move(Number(button.dataset.step))));
  // one check question after the lab; nothing is stored
  const feedback = select('#kv-check-feedback');
  all('[data-answer]').forEach(button => button.addEventListener('click', () => {
    const right = button.dataset.answer === '6';
    all('[data-answer]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
    feedback.hidden = false;
    feedback.dataset.result = right ? 'right' : 'wrong';
    feedback.textContent = right
      ? chinese ? '对，6 个：prompt 里的 A、B、C，加上每步解码各加一个的 D、E、F。模型每读一个 token，每一层都多存一个 key 和一个 value。'
        : 'Right: six. A, B and C from the prompt, then D, E and F, one per decoding step. Every token the model reads adds one key and one value in each layer.'
      : chinese ? '再数一遍：prompt 给了 A、B、C，之后每步解码再加一个 token：D、E、F。'
        : 'Count again: the prompt gave A, B and C, and each decoding step added one token: D, E, F.';
  }));
  select('#kv-previous').addEventListener('click', () => move(Math.max(0, step - 1)));
  select('#kv-next').addEventListener('click', () => move(Math.min(3, step + 1)));
  select('#kv-reset').addEventListener('click', () => move(0));
  select('#kv-play').addEventListener('click', () => {
    if (timer) return stop();
    if (step === 3) step = 0;
    render();
    select('#kv-play').textContent = chinese ? '暂停' : 'Pause';
    select('#kv-play').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      step += 1;
      render();
      history.replaceState(null, '', viewURL(location.href, mode, step));
      if (step === 3) stop();
    }, 2200);
  });
  function motionPreference() { stop(); select('#kv-play').hidden = reducedMotion.matches; }
  reducedMotion.addEventListener('change', motionPreference);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('pagehide', stop);
  window.addEventListener('popstate', restoreLocation);
  window.addEventListener('hashchange', restoreLocation);
  if (typeof IntersectionObserver !== 'undefined') new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) stop();
  }).observe(select('#kv-lab'));
  render();
  all('[data-learning-panel]').forEach(panel => { panel.tabIndex = 0; });
  full.setAttribute('role', 'tabpanel');
  full.setAttribute('aria-labelledby', 'kv-full-tab');
  select('[data-learning-nav]').hidden = false;
  motionPreference();
  restoreLocation();
})();
