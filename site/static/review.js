(() => {
  const deck = document.querySelector('[data-review-section]');
  if (!deck) return;
  const cards = [...deck.querySelectorAll('[data-card-id]')];
  const progress = deck.querySelector('.review-progress');
  const mastery = deck.querySelector('.review-mastery');
  const meter = deck.querySelector('.review-meter span');
  const assessment = deck.querySelector('.review-assessment');
  const navigation = deck.querySelector('.review-actions');
  const options = deck.querySelector('.review-options');
  const storageNote = deck.querySelector('.review-storage');
  const buttons = Object.fromEntries([...deck.querySelectorAll('[data-review-action]')]
    .map(button => [button.dataset.reviewAction, button]));
  const storageKey = `agi-review:v1:${deck.dataset.reviewSection}`;
  const messages = {
    zh: {count: (position, total) => total ? `第 ${position} / ${total} 题` : '本轮已完成',
      mastery: done => `已掌握 ${done} / ${cards.length}`,
      understood: '已记下：能讲清了', again: '已记下：下次再练',
      storage: '浏览器暂时无法保存记录。这次仍可复习，离开页面后标记可能丢失。',
      reset: '再点一次，清除本章记录', resetLabel: '重置本章记录', language: '用英语复习'},
    en: {count: (position, total) => total ? `Question ${position} / ${total}` : 'All caught up',
      mastery: done => `${done} / ${cards.length} understood`,
      understood: 'Saved: got it', again: 'Saved for another try',
      storage: 'Browser storage is unavailable. You can still practice; marks may not survive leaving this page.',
      reset: 'Click again to clear this chapter', resetLabel: 'Reset chapter marks', language: 'Review in Chinese'}
  };
  let language = deck.dataset.reviewLanguage;
  let marks = {};
  let storageAvailable = true;
  let order = [...cards];
  let practiceOnly = false;
  let resetPending = false;
  let current = cards[0];
  const identity = card => `${card.dataset.cardId}:${card.dataset.version}`;
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || '{}');
    if (saved && typeof saved === 'object' && !Array.isArray(saved)) {
      for (const card of cards) {
        const value = saved[identity(card)];
        if (['understood', 'again'].includes(value)) marks[identity(card)] = value;
      }
    }
  } catch { storageAvailable = false; }

  function save() {
    try { localStorage.setItem(storageKey, JSON.stringify(marks)); }
    catch { storageAvailable = false; }
  }

  function queue() {
    return order.filter(card => !practiceOnly || marks[identity(card)] !== 'understood');
  }

  function render() {
    const visible = queue();
    if (!visible.includes(current)) current = visible[0];
    for (const card of cards) {
      card.hidden = card !== current;
      if (card !== current) card.querySelector('details').open = false;
      card.querySelector('.review-mark').textContent = messages[language][marks[identity(card)]] || '';
    }
    const answered = current?.querySelector('details').open;
    assessment.hidden = !answered;
    navigation.hidden = visible.length < 2;
    buttons.again.disabled = !answered;
    buttons.understood.disabled = !answered;
    for (const action of ['again', 'understood']) {
      buttons[action].setAttribute('aria-pressed', String(Boolean(current && marks[identity(current)] === action)));
    }
    buttons.previous.disabled = visible.length < 2;
    buttons.next.disabled = visible.length < 2;
    buttons.shuffle.disabled = visible.length < 2;
    buttons.filter.setAttribute('aria-pressed', String(practiceOnly));
    deck.querySelector('.review-empty').hidden = visible.length !== 0;
    const done = cards.filter(card => marks[identity(card)] === 'understood').length;
    progress.textContent = messages[language].count(current ? visible.indexOf(current) + 1 : 0, visible.length);
    mastery.textContent = messages[language].mastery(done);
    meter.style.transform = `scaleX(${done / cards.length})`;
    buttons.language.textContent = language === 'zh' ? 'EN' : '中文';
    buttons.language.setAttribute('aria-label', messages[language].language);
    buttons.reset.textContent = messages[language][resetPending ? 'reset' : 'resetLabel'];
    if (!storageAvailable) storageNote.textContent = messages[language].storage;
  }

  function move(offset) {
    const visible = queue();
    if (!visible.length) return;
    current.querySelector('details').open = false;
    current = visible[(visible.indexOf(current) + offset + visible.length) % visible.length];
    current.querySelector('details').open = false;
  }

  function closeOptions(restoreFocus) {
    options.open = false;
    resetPending = false;
    if (restoreFocus) options.querySelector('summary').focus();
  }

  deck.addEventListener('click', event => {
    const button = event.target.closest('[data-review-action]');
    if (!button || button.disabled) return;
    const action = button.dataset.reviewAction;
    const previousCard = current;
    if (action !== 'reset') resetPending = false;
    if (action === 'language') {
      language = language === 'zh' ? 'en' : 'zh';
      deck.dataset.reviewLanguage = language;
      deck.querySelectorAll('[data-review-copy]').forEach(node => {
        node.hidden = node.dataset.reviewCopy !== language;
      });
    } else if (action === 'filter') {
      practiceOnly = !practiceOnly;
    } else if (action === 'all') {
      practiceOnly = false;
    } else if (action === 'shuffle') {
      for (let index = order.length - 1; index > 0; index--) {
        const partner = Math.floor(Math.random() * (index + 1));
        [order[index], order[partner]] = [order[partner], order[index]];
      }
      current = queue()[0];
      if (current) current.querySelector('details').open = false;
    } else if (action === 'previous' || action === 'next') {
      move(action === 'next' ? 1 : -1);
    } else if (action === 'reset') {
      if (resetPending) {
        marks = {};
        resetPending = false;
        save();
        current = order[0];
        cards.forEach(card => { card.querySelector('details').open = false; });
      } else {
        resetPending = true;
      }
    } else if (current && current.querySelector('details').open
      && ['again', 'understood'].includes(action)) {
      marks[identity(current)] = action;
      save();
    }
    if (['filter', 'shuffle'].includes(action) || (action === 'reset' && !resetPending)) closeOptions(true);
    render();
    if (['previous', 'next', 'all'].includes(action) || previousCard !== current) {
      if (current) current.querySelector('.review-question').focus({preventScroll: true});
      else buttons.all.focus({preventScroll: true});
    }
  });

  document.addEventListener('click', event => {
    if (options.open && !options.contains(event.target)) {
      closeOptions(false);
      render();
    }
  });
  options.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || !options.open) return;
    closeOptions(true);
    render();
  });
  options.addEventListener('toggle', () => {
    if (!options.open && resetPending) {
      resetPending = false;
      render();
    }
  });

  cards.forEach(card => card.querySelector('details').addEventListener('toggle', render));
  deck.querySelectorAll('[data-review-controls]').forEach(node => { node.hidden = false; });
  buttons.language.hidden = false;
  deck.classList.add('review-enhanced');
  render();
})();
