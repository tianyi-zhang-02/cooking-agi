const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/review.js'), 'utf8');

function element(dataset = {}) {
  return { dataset, hidden: false, disabled: false, open: false, textContent: '', style: {},
    attributes: {}, listeners: {}, focused: false,
    addEventListener(name, callback) { this.listeners[name] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; },
    focus() { this.focused = true; },
    closest() { return this; } };
}

function mount({saved = '{}', blocked = false, language = 'zh'} = {}) {
  const cards = ['first', 'second'].map(name => {
    const card = element({cardId: name, version: 'stable'});
    card.answer = element();
    card.mark = element();
    card.question = element();
    card.querySelector = selector => ({details: card.answer, '.review-mark': card.mark,
      '.review-question': card.question}[selector]);
    return card;
  });
  const buttons = Object.fromEntries(['language', 'filter', 'shuffle', 'reset', 'previous', 'next', 'again', 'understood', 'all']
    .map(action => [action, element({reviewAction: action})]));
  const controls = Object.fromEntries(['progress', 'mastery', 'assessment', 'actions', 'options', 'storage', 'empty']
    .map(name => [name, element()]));
  const meter = element();
  const summary = element();
  controls.options.contains = target => [buttons.filter, buttons.shuffle, buttons.reset, summary].includes(target);
  controls.options.querySelector = () => summary;
  const copies = ['zh', 'en'].map(reviewCopy => element({reviewCopy}));
  const deck = element({reviewLanguage: language, reviewSection: 'test'});
  deck.classList = {add() {}};
  deck.querySelector = selector => selector === '.review-meter span' ? meter : controls[selector.replace('.review-', '')];
  deck.querySelectorAll = selector => ({'[data-card-id]': cards,
    '[data-review-action]': Object.values(buttons), '[data-review-copy]': copies,
    '[data-review-controls]': [controls.assessment, controls.actions, controls.options]}[selector]);
  const document = element();
  document.querySelector = () => deck;
  const writes = [];
  const localStorage = {
    getItem() { if (blocked) throw Error('blocked'); return saved; },
    setItem(key, value) { if (blocked) throw Error('blocked'); writes.push({key, value}); }
  };
  vm.runInNewContext(source, {document, localStorage, Math});
  function click(action) {
    deck.listeners.click({target: buttons[action]});
    document.listeners.click({target: buttons[action]});
  }
  function reveal(index = 0) {
    cards[index].answer.open = true;
    cards[index].answer.listeners.toggle();
  }
  return {cards, buttons, controls, meter, summary, copies, deck, document, writes, click, reveal};
}

const initial = mount();
assert.equal(initial.cards[0].hidden, false);
assert.equal(initial.cards[1].hidden, true);
assert.equal(initial.controls.assessment.hidden, true);
assert.equal(initial.buttons.understood.disabled, true);
assert.equal(initial.controls.progress.textContent, '第 1 / 2 题');
initial.click('understood');
assert.equal(initial.writes.length, 0);
initial.reveal();
assert.equal(initial.controls.assessment.hidden, false);
initial.click('understood');
assert.equal(initial.buttons.understood.attributes['aria-pressed'], 'true');
assert.equal(initial.controls.mastery.textContent, '已掌握 1 / 2');
assert.equal(initial.meter.style.transform, 'scaleX(0.5)');
assert.equal(initial.writes[0].key, 'agi-review:v1:test');
initial.click('next');
assert.equal(initial.cards[1].hidden, false);
assert.equal(initial.cards[0].answer.open, false);
assert.equal(initial.controls.assessment.hidden, true);
assert.equal(initial.cards[1].question.focused, true);
initial.click('previous');
assert.equal(initial.cards[0].hidden, false);

const restored = mount({saved: initial.writes[0].value});
assert.equal(restored.controls.mastery.textContent, '已掌握 1 / 2');
restored.click('filter');
assert.equal(restored.cards[1].hidden, false);
assert.equal(restored.controls.progress.textContent, '第 1 / 1 题');
assert.equal(restored.controls.actions.hidden, true);
restored.reveal(1);
restored.click('understood');
assert.equal(restored.controls.empty.hidden, false);
assert.equal(restored.controls.progress.textContent, '本轮已完成');
assert.equal(restored.controls.assessment.hidden, true);
assert.equal(restored.controls.actions.hidden, true);
assert.equal(restored.buttons.all.focused, true);
restored.click('all');
assert.equal(restored.controls.empty.hidden, true);
assert.equal(restored.buttons.filter.attributes['aria-pressed'], 'false');
assert.equal(restored.cards[0].hidden, false);

restored.controls.options.open = true;
restored.click('reset');
assert.equal(restored.controls.mastery.textContent, '已掌握 2 / 2');
assert.equal(restored.buttons.reset.textContent, '再点一次，清除本章记录');
restored.controls.options.listeners.keydown({key: 'Escape'});
assert.equal(restored.controls.options.open, false);
assert.equal(restored.summary.focused, true);
assert.equal(restored.buttons.reset.textContent, '重置本章记录');
restored.controls.options.open = true;
restored.click('reset');
restored.click('reset');
assert.equal(restored.controls.mastery.textContent, '已掌握 0 / 2');
assert.equal(restored.cards[0].answer.open, false);
assert.equal(restored.controls.options.open, false);

initial.reveal();
initial.click('language');
assert.equal(initial.deck.dataset.reviewLanguage, 'en');
assert.equal(initial.cards[0].answer.open, true);
assert.equal(initial.controls.mastery.textContent, '1 / 2 understood');
assert.equal(initial.copies[0].hidden, true);
assert.equal(initial.copies[1].hidden, false);
assert.equal(initial.buttons.language.attributes['aria-label'], 'Review in Chinese');

const unavailable = mount({blocked: true});
unavailable.reveal();
unavailable.click('again');
assert.equal(unavailable.buttons.again.attributes['aria-pressed'], 'true');
assert(unavailable.controls.storage.textContent.includes('无法保存'));
assert.doesNotThrow(() => mount({saved: '{broken json'}));
assert.equal(mount({saved: '{"first:old":"understood"}'}).controls.mastery.textContent, '已掌握 0 / 2');
assert.equal(mount({language: 'en'}).controls.progress.textContent, 'Question 1 / 2');
const shuffled = mount();
shuffled.reveal();
shuffled.click('shuffle');
assert.equal(shuffled.cards.filter(card => !card.hidden).length, 1);
assert.equal(shuffled.controls.assessment.hidden, true);
