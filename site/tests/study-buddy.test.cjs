const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const buddy = require(path.join(__dirname, '../static/study-buddy.js'));

const at = (hour, minute = 0) => new Date(2026, 9, 13, hour, minute);

test('the day splits into five parts at the expected hours', () => {
  const cases = [[4, 59, 'night'], [5, 0, 'morning'], [10, 59, 'morning'], [11, 0, 'noon'], [13, 59, 'noon'], [14, 0, 'afternoon'],
    [17, 59, 'afternoon'], [18, 0, 'evening'], [22, 59, 'evening'], [23, 0, 'night'], [0, 30, 'night']];
  for (const [hour, minute, period] of cases) assert.equal(buddy.periodAt(at(hour, minute)), period, `${hour}:${minute}`);
});

test('every late-night line reminds the reader to rest', () => {
  for (const text of buddy.LINES.night.zh) assert.match(text, /^夜深了，记得休息哦/);
  for (const text of buddy.LINES.night.en) assert.match(text, /^It’s late\. Remember to rest/);
  for (const [period, lines] of Object.entries(buddy.LINES)) {
    assert.equal(lines.zh.length, lines.en.length, period);
    for (const text of lines.en) assert.doesNotMatch(text, /[㐀-鿿]/, period);
  }
});

test('the line stays the same through a day and changes across days', () => {
  assert.equal(buddy.line('afternoon', 'zh', at(14, 5)), buddy.line('afternoon', 'zh', at(17, 55)));
  const days = new Set(Array.from({length: 3}, (_, k) => buddy.line('afternoon', 'en', new Date(2026, 9, 13 + k, 15))));
  assert.equal(days.size, 3);
});

test('today counts read naturally', () => {
  assert.equal(buddy.todayLine(0, 'morning', 'zh'), '今天还没开始，先挑一篇短的？');
  assert.equal(buddy.todayLine(3, 'night', 'zh'), '今天读了 3 篇，辛苦啦');
  assert.equal(buddy.todayLine(1, 'afternoon', 'en'), '1 note read today');
  assert.equal(buddy.todayLine(2, 'night', 'en'), '2 notes read today. Well done.');
});

test('site-wide counts appear only when they are real numbers', () => {
  assert.equal(buddy.liveLine(null, 'zh'), '');
  assert.equal(buddy.liveLine({now: 0, today: 4}, 'zh'), '');
  assert.equal(buddy.liveLine({now: 5, today: 3}, 'en'), '');
  assert.equal(buddy.liveLine({now: 2.5, today: 9}, 'en'), '');
  assert.equal(buddy.liveLine({now: 3, today: 41}, 'zh'), '此刻 3 人在学，今天共 41 人');
  assert.equal(buddy.liveLine({now: 3, today: 41}, 'en'), '3 studying now · 41 today');
});

test('drawings are plain SVG with ids scoped to their instance', () => {
  for (const period of ['morning', 'noon', 'afternoon', 'evening', 'night']) {
    const svg = buddy.art(period, 'x-test');
    assert.match(svg, /^<svg class="sb-svg sb-/);
    assert.doesNotMatch(svg, /<script|\son[a-z]+=|javascript:|style=/i);
    for (const id of svg.matchAll(/id="([^"]+)"/g)) assert.ok(id[1].startsWith('x-test-'), id[1]);
    for (const ref of svg.matchAll(/url\(#([^)]+)\)/g)) assert.ok(ref[1].startsWith('x-test-'), ref[1]);
  }
});

const day = (y, m, d, h = 15) => new Date(y, m - 1, d, h, 0);
const holiday = (date, lang) => (buddy.holidayAt(date, lang) || {}).id || null;

test('Chinese pages follow Chinese festivals, including lunar ones', () => {
  assert.equal(holiday(day(2026, 1, 1), 'zh'), 'new-year');
  assert.equal(holiday(day(2026, 2, 16), 'zh'), 'chuxi');
  assert.equal(holiday(day(2026, 2, 17), 'zh'), 'spring');
  assert.equal(holiday(day(2026, 2, 23), 'zh'), 'spring');
  assert.equal(holiday(day(2026, 2, 24), 'zh'), null);
  assert.equal(holiday(day(2026, 3, 3), 'zh'), 'lantern');
  assert.equal(holiday(day(2026, 5, 5), 'zh'), 'labor');
  assert.equal(holiday(day(2026, 6, 19), 'zh'), 'dragon-boat');
  assert.equal(holiday(day(2026, 9, 10), 'zh'), 'teachers');
  assert.equal(holiday(day(2026, 9, 25), 'zh'), 'mid-autumn');
  assert.equal(holiday(day(2026, 10, 1), 'zh'), 'national');
  assert.equal(holiday(day(2026, 10, 7), 'zh'), 'national');
  assert.equal(holiday(day(2026, 10, 8), 'zh'), null);
  assert.equal(holiday(day(2026, 10, 24), 'zh'), 'programmers');
  // 2028: Mid-Autumn falls inside the National Day week and wins
  assert.equal(holiday(day(2028, 10, 3), 'zh'), 'mid-autumn');
  assert.equal(holiday(day(2028, 10, 4), 'zh'), 'national');
  assert.equal(holiday(day(2027, 2, 6), 'zh'), 'spring');
  // outside the computed lunar table the lunar festivals stay quiet
  assert.equal(holiday(day(2036, 1, 28), 'zh'), null);
  assert.equal(holiday(day(2026, 10, 31), 'zh'), null);
});

test('English pages follow US holidays', () => {
  assert.equal(holiday(day(2026, 3, 14), 'en'), 'pi');
  assert.equal(holiday(day(2026, 7, 4), 'en'), 'july-4');
  assert.equal(holiday(day(2026, 9, 7), 'en'), 'labor');
  assert.equal(holiday(day(2026, 9, 14), 'en'), null);
  assert.equal(holiday(day(2026, 10, 31), 'en'), 'halloween');
  assert.equal(holiday(day(2026, 11, 26), 'en'), 'thanksgiving');
  assert.equal(holiday(day(2027, 11, 25), 'en'), 'thanksgiving');
  assert.equal(holiday(day(2026, 12, 24), 'en'), 'christmas');
  assert.equal(holiday(day(2026, 12, 31), 'en'), 'new-years-eve');
  assert.equal(holiday(day(2026, 10, 1), 'en'), null);
  assert.equal(holiday(day(2026, 9, 25), 'en'), null);
});

test('holiday words: greeting by day, rest reminder at night, red New Year’s Eve', () => {
  assert.equal(buddy.moment(day(2026, 9, 25, 20), 'zh').text, '中秋快乐！吃块月饼，看看月亮。');
  assert.equal(buddy.moment(day(2026, 9, 25, 23), 'zh').text, '中秋快乐！夜深了，记得休息哦。');
  assert.equal(buddy.moment(day(2026, 2, 16, 23), 'zh').text, '守岁也别熬太晚，新年快乐！');
  assert.equal(buddy.moment(day(2026, 10, 31, 1), 'en').text, 'Happy Halloween! It’s late. Remember to rest.');
  assert.equal(buddy.moment(day(2026, 10, 24, 15), 'zh').label, '程序员节');
  assert.match(buddy.art('night', 'x', 'spring'), /#D9473C/);
  for (const list of Object.values(buddy.HOLIDAYS)) for (const item of list) assert.ok(buddy.LOOKS[item.look], item.id);
  for (const item of buddy.HOLIDAYS.en) assert.doesNotMatch(item.name + item.line + item.greet + (item.night || ''), /[㐀-鿿]/);
});

test('weekday notes on Mondays, Friday evenings and weekends', () => {
  assert.equal(buddy.moment(new Date(2026, 9, 12, 9), 'zh').text, '周一，新的一周，从一篇开始。');
  assert.equal(buddy.moment(new Date(2026, 9, 16, 20), 'en').text, 'Studying on a Friday evening? Give yourself a hand.');
  assert.equal(buddy.moment(new Date(2026, 9, 17, 15), 'zh').text, '周末也来学，节奏慢一点没关系。');
  assert.equal(buddy.moment(new Date(2026, 9, 13, 15), 'zh').text, buddy.line('afternoon', 'zh', new Date(2026, 9, 13, 15)));
});

test('every holiday look is plain SVG too', () => {
  for (const look of Object.keys(buddy.LOOKS)) for (const period of ['afternoon', 'night']) {
    const svg = buddy.art(period, 'h-test', look);
    assert.doesNotMatch(svg, /<script|\son[a-z]+=|javascript:|style=/i, look);
    for (const id of svg.matchAll(/id="([^"]+)"/g)) assert.ok(id[1].startsWith('h-test-'), `${look}: ${id[1]}`);
    for (const ref of svg.matchAll(/url\(#([^)]+)\)/g)) assert.ok(ref[1].startsWith('h-test-'), `${look}: ${ref[1]}`);
  }
});
