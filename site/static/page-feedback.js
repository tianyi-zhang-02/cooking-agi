(function (scope) {
  'use strict';
  // Page feedback builds a prefilled GitHub issue URL and opens it in a new tab.
  // The site never sends anything itself; the reader confirms on GitHub.
  const LIMIT = 7000;          // stay well under GitHub's URL size limit (414 beyond it)
  const QUOTE_CHARS = 280;
  const QUOTE_FRESH_MS = 120000;

  function clip(value, size) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    return text.length > size ? text.slice(0, size - 1).trimEnd() + '…' : text;
  }

  function body(kind, text, quote, page) {
    const parts = ['**' + kind + '**'];
    if (quote) parts.push('> ' + clip(quote, QUOTE_CHARS));
    if (text && text.trim()) parts.push(text.trim());
    if (page) parts.push('—\n' + page);
    return parts.join('\n\n') + '\n';
  }

  function issueURL(data, kind, text, quote, page) {
    let note = String(text || '');
    let trimmed = false;
    for (;;) {
      const params = new URLSearchParams({
        template: 'note-feedback.yml',
        title: '[Note] ' + kind + ' · ' + data.title,
        area: data.area,
        source: data.source,
        question: body(kind, note + (trimmed ? ' …' : ''), quote, page),
      });
      const url = 'https://github.com/' + data.repo + '/issues/new?' + params;
      if (url.length <= LIMIT || !note) return {url, trimmed};
      note = note.slice(0, Math.floor(note.length * 0.85));
      trimmed = true;
    }
  }

  const api = {LIMIT, clip, body, issueURL};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;

  const root = document.querySelector('[data-page-feedback]');
  if (!root) return;
  const data = {repo: root.dataset.repo, area: root.dataset.area, source: root.dataset.source, title: root.dataset.title};
  if (!/^[\w.-]+\/[\w.-]+$/.test(data.repo || '')) return;

  const kinds = Array.from(root.querySelectorAll('.pf-kind'));
  const form = root.querySelector('.pf-form');
  const field = root.querySelector('[data-pf-text]');
  const quoteRow = root.querySelector('.pf-quote');
  const quoteText = root.querySelector('[data-pf-quote]');
  const tip = root.querySelector('[data-pf-tip]');
  const count = root.querySelector('[data-pf-count]');
  const thanks = root.querySelector('[data-pf-thanks]');
  const article = document.querySelector('.article-body');
  let active = null;
  let quote = '';
  let lastSelection = {text: '', at: 0};

  document.addEventListener('selectionchange', () => {
    const selection = document.getSelection();
    if (!selection || selection.isCollapsed || !article) return;
    const node = selection.anchorNode;
    if (!node || !article.contains(node)) return;
    const text = clip(selection.toString(), QUOTE_CHARS);
    if (text) lastSelection = {text, at: Date.now()};
  });

  function setQuote(text) {
    quote = text;
    quoteText.textContent = text;
    quoteRow.hidden = !text;
    tip.hidden = Boolean(text);
  }

  function updateCount() {
    const left = field.maxLength - field.value.length;
    count.textContent = left < 150 ? field.value.length + ' / ' + field.maxLength : '';
  }

  function open(kind) {
    active = kind;
    kinds.forEach(item => item.setAttribute('aria-pressed', String(item === kind)));
    field.placeholder = kind.dataset.hint || '';
    if (Date.now() - lastSelection.at < QUOTE_FRESH_MS && lastSelection.text) setQuote(lastSelection.text);
    else if (!quote) setQuote('');
    thanks.hidden = true;
    form.hidden = false;
    field.focus({preventScroll: true});
  }

  function close(returnFocus) {
    const previous = active;
    active = null;
    kinds.forEach(item => item.setAttribute('aria-pressed', 'false'));
    form.hidden = true;
    if (returnFocus && previous) previous.focus();
  }

  kinds.forEach(kind => {
    kind.setAttribute('role', 'button');
    kind.setAttribute('aria-pressed', 'false');
    kind.addEventListener('click', event => {
      event.preventDefault();
      if (active === kind) close(false); else open(kind);
    });
    kind.addEventListener('keydown', event => {
      if (event.key === ' ') { event.preventDefault(); kind.click(); }
    });
  });

  root.querySelector('[data-pf-unquote]').addEventListener('click', () => {
    setQuote('');
    lastSelection = {text: '', at: 0};
    field.focus();
  });
  root.querySelector('[data-pf-cancel]').addEventListener('click', () => close(true));
  form.addEventListener('keydown', event => { if (event.key === 'Escape') close(true); });
  field.addEventListener('input', updateCount);

  form.addEventListener('submit', event => {
    event.preventDefault();
    if (!active) return;
    const page = location.origin + location.pathname;
    const {url} = issueURL(data, active.dataset.label, field.value, quote, page);
    scope.open(url, '_blank', 'noopener');
    field.value = '';
    setQuote('');
    lastSelection = {text: '', at: 0};
    updateCount();
    close(false);
    thanks.hidden = false;
  });
})(typeof window === 'undefined' ? globalThis : window);
