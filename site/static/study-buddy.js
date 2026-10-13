/* Study buddy: a small round window in the corner that follows the reader's
   local time of day (sunrise and coffee, a clear noon and lunch, afternoon tea,
   a lamp at dusk, the moon and warm milk late at night) and says how many notes the reader
   has opened today. Lines change from day to day; on holidays it dresses up
   (Chinese festivals on Chinese pages, US holidays on English pages).
   Everything comes from this browser; nothing is sent.

   A site-wide "studying now / today" count needs a counting service. Until one
   is configured, nothing calls StudyBuddy.setLive() and that row stays hidden:
   no placeholder numbers, ever. */
(function (scope) {
  'use strict';
  const KEY = 'cooking-agi:study-buddy';

  function periodAt(date) {
    const hour = date.getHours();
    if (hour >= 5 && hour < 11) return 'morning';
    if (hour >= 11 && hour < 14) return 'noon';
    if (hour >= 14 && hour < 18) return 'afternoon';
    if (hour >= 18 && hour < 23) return 'evening';
    return 'night';
  }
  const LABELS = {
    morning: ['早上', 'Morning'], noon: ['中午', 'Midday'], afternoon: ['下午', 'Afternoon'], evening: ['晚上', 'Evening'], night: ['深夜', 'Late night']
  };
  const pad = value => String(value).padStart(2, '0');
  const LINES = {
    morning: {zh: ['早，今天也慢慢来。', '早上脑子最清楚，挑一篇硬一点的？', '先喝口咖啡，再打开第一篇。', '早起学一会儿，一天都踏实。', '今天想弄懂哪个概念？', '早上好，从昨天没看完的那篇接着来。'],
      en: ['Morning. Take it slow today.', 'A fresh mind: good time for a harder note.', 'Coffee first, then the first note.', 'A little study early makes the day feel steady.', 'Which idea do you want to crack today?', 'Good morning. Pick up where you left off.']},
    noon: {zh: ['吃饱再学，眯一会儿也行。', '午饭时间，看一篇短的就好。', '中午困了就歇十分钟，别硬撑。', '边吃边看？小心别洒到键盘上。', '上午辛苦啦，先好好吃饭。', '午休前看一个小概念，刚刚好。'],
      en: ['Eat first. A short nap is fine too.', 'Lunch break. A short note is plenty.', 'Sleepy at noon? Ten minutes of rest helps.', 'Reading over lunch? Mind the keyboard.', 'Good work this morning. Go eat properly.', 'One small idea before the break is just right.']},
    afternoon: {zh: ['下午茶时间，喝口茶再看一节。', '有点困就起来走走，茶还热着。', '慢慢喝，慢慢看。', '下午容易走神，去交互图里动动手？', '看累了就望望窗外，再回来。', '一杯茶的工夫，刚好读完一小节。'],
      en: ['Tea time. One more section after a sip.', 'Sleepy? Stretch a little; the tea is still warm.', 'Sip slowly, read slowly.', 'Mind wandering? Go play with a figure.', 'Tired eyes? Look out the window, then come back.', 'One cup of tea, one short section.']},
    evening: {zh: ['晚上安静，适合把今天的内容过一遍。', '开着台灯，再看一会儿。', '今天看过的，再翻一遍就熟了。', '晚饭后走一走，回来再学也不迟。', '今天学到一点，就算赚到。', '把今天最有意思的一个点记下来吧。'],
      en: ['A quiet evening, good for going over today.', 'Lamp on. A little more reading.', 'One more look at today’s notes.', 'A walk after dinner, then a little reading.', 'Learned one thing today? That counts.', 'Write down the most interesting idea from today.']},
    // every late-night line starts with the same reminder
    night: {zh: ['夜深了，记得休息哦。', '夜深了，记得休息哦。剩下的明天再看。', '夜深了，记得休息哦。喝杯热牛奶，早点睡。', '夜深了，记得休息哦。今天已经很棒了。', '夜深了，记得休息哦。睡个好觉，明天接着来。'],
      en: ['It’s late. Remember to rest.', 'It’s late. Remember to rest; the rest can wait.', 'It’s late. Remember to rest. Warm milk, then bed.', 'It’s late. Remember to rest. Today was enough.', 'It’s late. Remember to rest. Pick it up tomorrow.']}
  };
  function dayNumber(date) {
    return Math.floor((Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()) - Date.UTC(date.getFullYear(), 0, 0)) / 86400000);
  }
  // one line per period per day, so it does not change on every page
  const line = (period, lang, date) => { const list = LINES[period][lang]; return list[dayNumber(date) % list.length]; };
  // a few days of the week get their own note
  function weekday(date, period, lang) {
    const day = date.getDay(), daytime = ['morning', 'noon', 'afternoon'].includes(period);
    if (day === 1 && daytime) return lang === 'zh' ? '周一，新的一周，从一篇开始。' : 'Monday. A new week starts with one note.';
    if (day === 5 && period === 'evening') return lang === 'zh' ? '周五晚上还在学，给自己鼓个掌。' : 'Studying on a Friday evening? Give yourself a hand.';
    if ((day === 0 || day === 6) && daytime) return lang === 'zh' ? '周末也来学，节奏慢一点没关系。' : 'Weekend study. Slow and steady is fine.';
    return '';
  }

  /* ---------------------------------------------------------------- holidays
     Chinese pages follow Chinese festivals; English pages follow US holidays.
     Lunar dates were computed with Foundation's Chinese calendar (2026–2035);
     outside that range the lunar festivals simply do not show. */
  const LUNAR = {
    2026: '02-16 02-17 03-03 06-19 09-25', 2027: '02-05 02-06 02-20 06-09 09-15', 2028: '01-25 01-26 02-09 05-28 10-03',
    2029: '02-12 02-13 02-27 06-16 09-22', 2030: '02-02 02-03 02-17 06-05 09-12', 2031: '01-22 01-23 02-06 06-24 10-01',
    2032: '02-10 02-11 02-25 06-12 09-19', 2033: '01-30 01-31 02-14 06-01 09-08', 2034: '02-18 02-19 03-05 06-20 09-27',
    2035: '02-07 02-08 02-22 06-10 09-16'
  };
  const LUNAR_NAMES = ['chuxi', 'spring', 'lantern', 'dragon', 'midautumn'];
  const midnight = date => new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const on = (month, day) => date => date.getMonth() + 1 === month && date.getDate() === day;
  const between = (m1, d1, m2, d2) => date => { const v = (date.getMonth() + 1) * 100 + date.getDate(); return v >= m1 * 100 + d1 && v <= m2 * 100 + d2; };
  const nth = (month, weekday, n) => date => date.getMonth() + 1 === month && date.getDay() === weekday && Math.ceil(date.getDate() / 7) === n;
  const lunar = (name, days = 1) => date => {
    const row = LUNAR[date.getFullYear()];
    if (!row) return false;
    const [month, day] = row.split(' ')[LUNAR_NAMES.indexOf(name)].split('-').map(Number);
    const diff = Math.round((midnight(date) - new Date(date.getFullYear(), month - 1, day)) / 86400000);
    return diff >= 0 && diff < days;
  };
  // earlier entries win, so single-day festivals come before longer breaks
  const HOLIDAYS = {
    zh: [
      {id: 'new-year', name: '元旦', when: on(1, 1), look: 'party', greet: '元旦快乐！', line: '元旦快乐！新的一年，从一篇开始。'},
      {id: 'chuxi', name: '除夕', when: lunar('chuxi'), look: 'spring', greet: '除夕快乐！', line: '除夕快乐！今天少学一点，多陪陪家人。', night: '守岁也别熬太晚，新年快乐！'},
      {id: 'lantern', name: '元宵节', when: lunar('lantern'), look: 'lantern', greet: '元宵节快乐！', line: '元宵节快乐！吃碗汤圆，再看一篇。'},
      {id: 'dragon-boat', name: '端午节', when: lunar('dragon'), look: 'zongzi', greet: '端午安康！', line: '端午安康！吃个粽子，再看一节。'},
      {id: 'teachers', name: '教师节', when: on(9, 10), look: 'apple', greet: '教师节快乐！', line: '教师节快乐！谢谢每一位把知识讲明白的人。'},
      {id: 'mid-autumn', name: '中秋节', when: lunar('midautumn'), look: 'moon', greet: '中秋快乐！', line: '中秋快乐！吃块月饼，看看月亮。'},
      {id: 'programmers', name: '程序员节', when: on(10, 24), look: 'code', greet: '1024 快乐！', line: '1024 程序员节快乐！愿今天的代码一次跑通。'},
      {id: 'spring', name: '春节', when: lunar('spring', 7), look: 'spring', greet: '新年快乐！', line: '新年快乐！假期里学一点点，就很棒了。'},
      {id: 'labor', name: '劳动节', when: between(5, 1, 5, 5), look: 'rest', greet: '劳动节快乐！', line: '劳动节快乐！今天允许自己少学一点。'},
      {id: 'national', name: '国庆', when: between(10, 1, 10, 7), look: 'fireworks', greet: '国庆快乐！', line: '国庆快乐！假期也记得出门走走。'}
    ],
    en: [
      {id: 'new-year', name: 'New Year’s Day', when: on(1, 1), look: 'party', greet: 'Happy New Year!', line: 'Happy New Year! Start the year with one note.'},
      {id: 'pi', name: 'Pi Day', when: on(3, 14), look: 'pie', greet: 'Happy Pi Day!', line: 'Happy Pi Day! 3.14159… and one more note.'},
      {id: 'july-4', name: 'Independence Day', when: on(7, 4), look: 'fireworks-us', greet: 'Happy Fourth!', line: 'Happy Fourth of July! Enjoy the fireworks.'},
      {id: 'labor', name: 'Labor Day', when: nth(9, 1, 1), look: 'rest', greet: 'Happy Labor Day!', line: 'Happy Labor Day! Rest counts as progress too.'},
      {id: 'halloween', name: 'Halloween', when: on(10, 31), look: 'pumpkin', greet: 'Happy Halloween!', line: 'Happy Halloween! The scariest thing here is a long derivation.'},
      {id: 'thanksgiving', name: 'Thanksgiving', when: nth(11, 4, 4), look: 'autumn', greet: 'Happy Thanksgiving!', line: 'Happy Thanksgiving! Thanks for reading along.'},
      {id: 'christmas', name: 'Christmas', when: between(12, 24, 12, 25), look: 'christmas', greet: 'Merry Christmas!', line: 'Merry Christmas! Take the day off if you can.'},
      {id: 'new-years-eve', name: 'New Year’s Eve', when: on(12, 31), look: 'party', greet: 'Happy New Year’s Eve!', line: 'New Year’s Eve. Here’s to another year of learning.', night: 'Counting down? Happy New Year! Get some sleep after.'}
    ]
  };
  const holidayAt = (date, lang) => HOLIDAYS[lang].find(item => item.when(date)) || null;
  // what the buddy says and wears right now
  function moment(date, lang) {
    const period = periodAt(date), holiday = holidayAt(date, lang);
    let text;
    if (holiday && period === 'night') text = holiday.night || `${holiday.greet}${lang === 'zh' ? '' : ' '}${LINES.night[lang][0]}`;
    else if (holiday) text = holiday.line;
    else text = weekday(date, period, lang) || line(period, lang, date);
    return {period, holiday, text, label: holiday ? holiday.name : LABELS[period][lang === 'zh' ? 0 : 1]};
  }
  function todayLine(count, period, lang) {
    const late = period === 'night';
    if (lang === 'zh') {
      if (!count) return late ? '今天就到这里，明天见。' : '今天还没开始，先挑一篇短的？';
      return `今天读了 ${count} 篇${late ? '，辛苦啦' : ''}`;
    }
    if (!count) return late ? 'That’s it for today. See you tomorrow.' : 'Nothing read yet today. Start with a short one?';
    return `${count} note${count === 1 ? '' : 's'} read today${late ? '. Well done.' : ''}`;
  }
  function liveLine(live, lang) {
    if (!live || !Number.isInteger(live.now) || !Number.isInteger(live.today) || live.now < 1 || live.today < live.now) return '';
    return lang === 'zh' ? `此刻 ${live.now} 人在学，今天共 ${live.today} 人` : `${live.now} studying now · ${live.today} today`;
  }

  /* ---------------------------------------------------------------- the drawing
     A round window: the sky outside follows the time of day, and the sill holds
     what you would have at hand. No character; holidays change the view and
     the things on the sill. Coordinates: 120 x 120, sill top at y = 84. */
  const FRAME = '#5B3F2D', INK = '#2A201C';
  const VIEW = {
    morning: {sky: ['#A9C9E6', '#FCD8B8'], hills: ['#8E9CC0', '#6E7EA6']},
    noon: {sky: ['#7DB6E4', '#D8ECF7'], hills: ['#8DB496', '#6E9A7B']},
    afternoon: {sky: ['#F1CF98', '#EDA67A'], hills: ['#C99474', '#AD755C']},
    evening: {sky: ['#2A2F59', '#C9767C'], hills: ['#4A3A5C', '#30263F']},
    night: {sky: ['#0B1130', '#1E2353'], hills: ['#1A1F40', '#11142C']}
  };
  const steam = (x, y) => `<g class="sb-steam" transform="translate(${x} ${y})" fill="none" stroke="#FFFFFF" stroke-width="1.2" stroke-linecap="round" opacity=".75">
    <path d="M-3.5 0 C-6 -3 -1 -5 -3.5 -8.5"/><path d="M1 0 C-1.5 -3 3.5 -5 1 -8.5"/><path d="M5.5 0 C3 -3 8 -5 5.5 -8.5"/></g>`;
  const sun = (x, y, r, color) => `<circle class="sb-glow" cx="${x}" cy="${y}" r="${r * 2.6}" fill="url(#ID-glow)"/><circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
  const cloud = (x, y, s) => `<g class="sb-drift" fill="url(#ID-cloud)" opacity=".92"><ellipse cx="${x}" cy="${y}" rx="${10 * s}" ry="${3.8 * s}"/>
    <ellipse cx="${x + 5 * s}" cy="${y - 2.6 * s}" rx="${6.4 * s}" ry="${4.4 * s}"/><ellipse cx="${x - 4 * s}" cy="${y - 1.4 * s}" rx="${4.6 * s}" ry="${3.2 * s}"/></g>`;
  const stars = pts => pts.map(([x, y, r]) => `<circle class="sb-twinkle" cx="${x}" cy="${y}" r="${r}" fill="#FFFFFF"/>`).join('');
  const CRESCENT = `<circle class="sb-glow" cx="86" cy="22" r="20" fill="url(#ID-glow)"/><path d="M86 11 a11 11 0 1 0 7 19.6 a9 9 0 1 1 -7 -19.6 Z" fill="#FFE9B0"/>`;
  const SKYLIFE = {
    morning: sun(30, 66, 8, '#FFE2A6') + cloud(84, 22, 1) + cloud(30, 34, .7),
    noon: sun(86, 18, 7, '#FFF6D2') + cloud(30, 24, 1.1) + cloud(78, 54, .7),
    afternoon: sun(88, 38, 8.5, '#FFE6AC') + cloud(28, 22, .9),
    evening: `<circle cx="90" cy="73" r="7" fill="#FFC27A"/>` + stars([[22, 16, 1], [40, 30, .8], [74, 12, 1.1]]),
    night: CRESCENT + stars([[20, 18, 1.2], [36, 8, .9], [48, 34, .8], [70, 30, 1], [104, 44, .9], [16, 46, .8]])
  };

  /* things on the sill; each sits with its base on y = 89 and casts a soft shadow */
  const shade = (x, w) => `<ellipse cx="${x}" cy="89.3" rx="${w}" ry="1.7" fill="#1C120B" opacity=".32" filter="url(#ID-soft)"/>`;
  const mug = (x, liquid, extra = '') => `<g>${shade(x, 9)}<path d="M${x + 8} ${79} h2.6 a3.6 3.6 0 0 1 0 7.2 h-2.6" stroke="#C3D8E4" stroke-width="2.2" fill="none"/>
    <rect x="${x - 8}" y="75" width="16" height="14" rx="3.4" fill="#E6EEF3"/><rect x="${x - 8}" y="81" width="16" height="2.4" fill="#9FC4DA"/><rect x="${x - 6.6}" y="76.6" width="2.2" height="10.8" rx="1.1" fill="#FFFFFF" opacity=".5"/>
    <rect x="${x + 4.6}" y="75" width="3.4" height="14" rx="1.6" fill="#1C2A33" opacity=".07"/>
    <ellipse cx="${x}" cy="75.6" rx="6.8" ry="1.6" fill="${liquid}"/>${extra}${steam(x - 1, 72)}</g>`;
  const teacup = x => `<g>${shade(x, 12)}<ellipse cx="${x}" cy="89" rx="12" ry="2.4" fill="#F2ECE2"/><path d="M${x + 8} 80.6 a3.4 3.4 0 0 1 0 5.6" stroke="#E6DFD3" stroke-width="2" fill="none"/>
    <path d="M${x - 9} 79.6 h18 v1.6 a9 7.2 0 0 1 -18 0 Z" fill="#FFFFFF"/><path d="M${x - 9} 81.2 h18" stroke="#E2B04E" stroke-width="1"/><path d="M${x - 7} 82.2 q1 2.6 3.2 4" stroke="#E9EEF2" stroke-width="1.3" fill="none" stroke-linecap="round"/>
    <ellipse cx="${x}" cy="79.6" rx="8.6" ry="1.7" fill="#C8773C"/>${steam(x - 1, 76)}</g>`;
  const teapot = x => `<g>${shade(x, 10)}<path d="M${x - 9} 80 C${x - 14} 79 ${x - 16} 75 ${x - 17} 72.6" stroke="#BFD9CB" stroke-width="2.6" fill="none" stroke-linecap="round"/>
    <path d="M${x + 8} 77 a5 5 0 0 1 0 8" stroke="#BFD9CB" stroke-width="2.4" fill="none"/>
    <ellipse cx="${x}" cy="82" rx="10" ry="7.4" fill="#CFE5D8"/><ellipse cx="${x}" cy="75.4" rx="5" ry="1.6" fill="#B7D5C4"/><circle cx="${x}" cy="73.2" r="1.6" fill="#B7D5C4"/>
    <ellipse cx="${x - 3.6}" cy="80" rx="2.6" ry="4" fill="#FFFFFF" opacity=".35"/><rect x="${x - 6}" y="88.2" width="12" height="1.2" rx=".6" fill="#B7D5C4"/></g>`;
  const cookies = x => `<g>${shade(x, 9)}<ellipse cx="${x}" cy="88.4" rx="9" ry="2" fill="#F2ECE2"/><ellipse cx="${x - 3}" cy="86.2" rx="4.4" ry="2" fill="#D9A15E"/>
    <ellipse cx="${x + 3.4}" cy="85.4" rx="4.2" ry="1.9" fill="#E3B172"/><circle cx="${x - 4}" cy="85.8" r=".6" fill="#7A4E2D"/><circle cx="${x + 4}" cy="85" r=".6" fill="#7A4E2D"/></g>`;
  const bowl = x => `<g>${shade(x, 9.5)}<path d="M${x - 10} 80 C${x - 7} 73.6 ${x + 7} 73.6 ${x + 10} 80 Z" fill="#FFFFFF"/>
    <path d="M${x - 11} 80 h22 a11 9 0 0 1 -22 0 Z" fill="#F3E4CF"/><path d="M${x - 10.4} 83 h20.8" stroke="#7FA6C9" stroke-width="1.6"/>
    <path d="M${x + 2} 76 L${x + 16} 68 M${x + 4} 77 L${x + 18} 70" stroke="#9B6B43" stroke-width="1.4" stroke-linecap="round"/>${steam(x - 1, 72)}</g>`;
  const glass = (x, fill = '#BFE2F2') => `<g>${shade(x, 5)}<path d="M${x - 5} 73 L${x + 5} 73 L${x + 4} 89 L${x - 4} 89 Z" fill="#E4F3F9" opacity=".75"/>
    <path d="M${x - 4.6} 78 L${x + 4.6} 78 L${x + 4} 89 L${x - 4} 89 Z" fill="${fill}"/><path d="M${x - 3} 75 L${x - 2.4} 87" stroke="#FFFFFF" stroke-width="1" opacity=".7"/></g>`;
  const plant = x => `<g>${shade(x, 6)}<path d="M${x} 80 C${x - 7} 75 ${x - 9} 68 ${x - 3} 65 C${x - 2} 71 ${x} 75 ${x} 80 Z" fill="#6FA36B"/>
    <path d="M${x} 80 C${x + 7} 76 ${x + 9} 70 ${x + 4} 66.6 C${x + 2.6} 72 ${x + 1} 76 ${x} 80 Z" fill="#5C8F5A"/>
    <path d="M${x} 80 C${x - 1} 74 ${x + 1} 69 ${x + .6} 63" stroke="#4E7F4C" stroke-width="1.2" fill="none"/>
    <path d="M${x - 6} 80 L${x + 6} 80 L${x + 4.6} 89 L${x - 4.6} 89 Z" fill="#C9744A"/><rect x="${x - 6.6}" y="79" width="13.2" height="2.6" rx="1" fill="#B5643E"/></g>`;
  const openBook = x => `<g>${shade(x, 15)}<path d="M${x} 85 L${x - 15} 82 L${x - 15} 88.6 L${x} 90.6 Z" fill="#FFF8EC"/><path d="M${x} 85 L${x + 15} 82 L${x + 15} 88.6 L${x} 90.6 Z" fill="#F7EAD5"/>
    <path class="sb-page" d="M${x} 85 L${x + 13} 82.6 L${x + 13} 88.4 L${x} 90.6 Z" fill="#FFFBF3"/>
    <path d="M${x - 12} 84 L${x - 3} 85.6 M${x - 12} 86.2 L${x - 3} 87.8 M${x + 3} 85.6 L${x + 12} 84" stroke="#D9C7AE" stroke-width=".8"/></g>`;
  const books = x => `<g>${shade(x, 11.5)}<rect x="${x - 11}" y="84" width="22" height="5" rx="1" fill="#5B7DB1"/><rect x="${x - 9}" y="79" width="19" height="5" rx="1" fill="#D98C4A"/>
    <rect x="${x - 10}" y="74.4" width="17" height="4.6" rx="1" fill="#7FAE84"/><path d="M${x + 9.4} 85 v3 M${x + 8.6} 80 v3 M${x + 5.6} 75.4 v2.6" stroke="#FFF7EA" stroke-width="1.2"/></g>`;
  const lamp = x => `<g>${shade(x, 6.5)}<circle class="sb-glow" cx="${x - 12}" cy="80" r="24" fill="url(#ID-lamp)"/>
    <ellipse cx="${x}" cy="88.6" rx="6" ry="1.6" fill="#3D2E26"/><path d="M${x} 88 L${x + 3} 74 L${x - 6} 64" stroke="#3D2E26" stroke-width="1.8" fill="none" stroke-linecap="round"/>
    <path d="M${x - 2} 60 L${x - 14} 64.6 L${x - 10} 70.6 Z" fill="#E8B04A"/><circle cx="${x - 11}" cy="68.6" r="1.6" fill="#FFF3C8"/></g>`;
  function clock(x, date) {
    const h = date.getHours() % 12 + date.getMinutes() / 60, m = date.getMinutes();
    const hand = (len, turns) => { const a = turns * 2 * Math.PI - Math.PI / 2; return `L${(x + Math.cos(a) * len).toFixed(2)} ${(81 + Math.sin(a) * len).toFixed(2)}`; };
    return `<g>${shade(x, 7.5)}<circle cx="${x - 5}" cy="74.6" r="2.4" fill="#C9B79C"/><circle cx="${x + 5}" cy="74.6" r="2.4" fill="#C9B79C"/>
      <path d="M${x - 5} 88 l-1.6 2 M${x + 5} 88 l1.6 2" stroke="#9E8B70" stroke-width="1.4" stroke-linecap="round"/>
      <circle cx="${x}" cy="81" r="7.4" fill="#F1EBDD" stroke="#B9A58C" stroke-width="1.4"/>
      <path d="M${x} 81 ${hand(3.6, h / 12)} M${x} 81 ${hand(5.4, m / 60)}" stroke="${INK}" stroke-width="1.2" stroke-linecap="round"/><circle cx="${x}" cy="81" r=".8" fill="${INK}"/></g>`;
  }
  const plate = (x, food) => `<g>${shade(x, 12.5)}<ellipse cx="${x}" cy="88.4" rx="12" ry="2.4" fill="#F4EEE6"/>${food}</g>`;
  const dumplings = x => plate(x, `<path d="M${x - 10} 87 Q${x - 6} 79.6 ${x - 2} 87 Z M${x - 3.6} 86.6 Q${x} 79 ${x + 3.6} 86.6 Z M${x + 2} 87 Q${x + 6} 79.6 ${x + 10} 87 Z" fill="#FFF8EC" stroke="#E6D3B9" stroke-width=".7"/>`);
  const tangyuan = x => `<g>${shade(x, 9.5)}<path d="M${x - 11} 80 h22 a11 9 0 0 1 -22 0 Z" fill="#F3E4CF"/><path d="M${x - 10.4} 83 h20.8" stroke="#D9473C" stroke-width="1.6"/>
    <circle cx="${x - 5}" cy="79" r="3.6" fill="#FFFFFF"/><circle cx="${x + 1.6}" cy="78" r="3.6" fill="#FFF4F0"/><circle cx="${x + 7}" cy="79.4" r="3" fill="#FFFFFF"/>${steam(x - 1, 73)}</g>`;
  const zongzi = x => plate(x, `<path d="M${x - 9} 87 L${x - 1} 72 L${x + 7} 87 Z" fill="#5FA05A"/><path d="M${x - 9} 87 L${x - 1} 72 L${x + 1.4} 77 L${x - 4.6} 87 Z" fill="#79B86E"/>
    <path d="M${x - 6.4} 82 L${x + 4.4} 82 M${x - 4} 77.6 L${x + 2} 77.6" stroke="#EADBA6" stroke-width="1.3"/>`);
  const apple = x => `<g>${shade(x, 9)}<path d="M${x} 77 C${x - 5} 73 ${x - 10.6} 77 ${x - 10} 82.4 C${x - 9} 88 ${x - 4} 90.4 ${x} 88.8 C${x + 4} 90.4 ${x + 9} 88 ${x + 10} 82.4 C${x + 10.6} 77 ${x + 5} 73 ${x} 77 Z" fill="#E2483D"/>
    <path d="M${x} 77 C${x} 74.6 ${x + .8} 73 ${x + 1.6} 72" stroke="#7A5E44" stroke-width="1.2" fill="none"/><path d="M${x + 1.6} 73.6 C${x + 5} 71 ${x + 7.4} 72.6 ${x + 7.4} 74.2 C${x + 5} 75 ${x + 3.4} 74.4 ${x + 1.6} 73.6 Z" fill="#79B86E"/>
    <ellipse cx="${x - 4}" cy="80.6" rx="1.6" ry="2.4" fill="#FFFFFF" opacity=".35"/></g>`;
  const mooncakes = x => plate(x, `<circle cx="${x - 4}" cy="83.4" r="5.6" fill="#D99A46"/><circle cx="${x - 4}" cy="83.4" r="4" fill="#E8B25E"/>
    <circle cx="${x + 5}" cy="84.4" r="4.8" fill="#D99A46"/><circle cx="${x + 5}" cy="84.4" r="3.4" fill="#E8B25E"/>
    <path d="M${x - 6} 83.4 h4 M${x - 4} 81.4 v4" stroke="#C7843A" stroke-width=".9" stroke-linecap="round"/>`);
  const laptop = x => `<g>${shade(x, 16)}<path d="M${x - 14} 70 h28 v15 h-28 Z" fill="#2B2F3A"/><rect class="sb-screen" x="${x - 12.4}" y="71.6" width="24.8" height="11.8" rx=".8" fill="#14202B"/>
    <text x="${x}" y="80.2" text-anchor="middle" font-family="ui-monospace, monospace" font-size="6.4" font-weight="700" fill="#7CF2B5">1024</text>
    <path d="M${x - 17} 85 h34 l-2 4 h-30 Z" fill="#A3ABB9"/></g>`;
  const pie = x => `<g>${shade(x, 12)}<ellipse cx="${x}" cy="86" rx="12" ry="4" fill="#D9934A"/><ellipse cx="${x}" cy="84.6" rx="9.8" ry="3" fill="#B5462F"/>
    <text x="${x}" y="87" text-anchor="middle" font-family="Georgia, serif" font-size="6.6" font-weight="700" fill="#FFE9C9">π</text></g>`;
  const jack = x => `<g>${shade(x, 10.5)}<ellipse cx="${x}" cy="82" rx="10.6" ry="7.6" fill="#F08A2C"/><path d="M${x - 4} 74.6 C${x - 5} 81 ${x - 5} 85 ${x - 4} 89 M${x + 4} 74.6 C${x + 5} 81 ${x + 5} 85 ${x + 4} 89" stroke="#D9731C" stroke-width="1" fill="none"/>
    <path d="M${x} 74.6 C${x} 72.4 ${x + 1.6} 71 ${x + 3.4} 71" stroke="#5E8A3A" stroke-width="2" fill="none" stroke-linecap="round"/>
    <path class="sb-flicker" d="M${x - 6} 80 L${x - 3.4} 77.4 L${x - .8} 80 Z M${x + .8} 80 L${x + 3.4} 77.4 L${x + 6} 80 Z M${x - 6} 84 Q${x} 88 ${x + 6} 84 Q${x} 85.8 ${x - 6} 84 Z" fill="#FFE27A"/></g>`;
  const candle = x => `<g>${shade(x, 5)}<rect x="${x - 2.6}" y="77" width="5.2" height="11" rx="1" fill="#F4EEE2"/><ellipse cx="${x}" cy="88.6" rx="5" ry="1.3" fill="#B9A58C"/>
    <circle class="sb-glow" cx="${x}" cy="73" r="9" fill="url(#ID-lamp)"/><path class="sb-flicker" d="M${x} 70 C${x - 2} 73 ${x - 1.6} 75.6 ${x} 76 C${x + 1.6} 75.6 ${x + 2} 73 ${x} 70 Z" fill="#FFC75A"/></g>`;
  const slice = x => plate(x, `<path d="M${x - 9} 87 L${x + 8} 81 L${x + 8} 87 Z" fill="#E08A3A"/><path d="M${x + 8} 81 L${x + 10} 81 L${x + 10} 87 L${x + 8} 87 Z" fill="#C9874A"/>
    <ellipse cx="${x + 4}" cy="80.6" rx="2.6" ry="1.6" fill="#FFFFFF"/>`);
  const gift = x => `<g>${shade(x, 9.5)}<rect x="${x - 9}" y="77" width="18" height="12" rx="1.6" fill="#D9473C"/><rect x="${x - 1.6}" y="77" width="3.2" height="12" fill="#F2C14E"/>
    <rect x="${x - 9}" y="81.4" width="18" height="3" fill="#F2C14E"/><path d="M${x} 77 C${x - 5} 71 ${x - 8.6} 76 ${x} 77 C${x + 8.6} 76 ${x + 5} 71 ${x} 77 Z" fill="#F2C14E"/></g>`;
  const shades = x => `<g>${shade(x, 10.5)}<rect x="${x - 10}" y="84" width="8.6" height="5" rx="2.2" fill="#22222A"/><rect x="${x + 1.4}" y="84" width="8.6" height="5" rx="2.2" fill="#22222A"/>
    <path d="M${x - 1.4} 85.4 h2.8" stroke="#22222A" stroke-width="1.2"/><path d="M${x - 8} 85.4 l2 -.8" stroke="#FFFFFF" stroke-width=".8" opacity=".6"/></g>`;
  const juice = x => `<g>${glass(x, '#FFB347')}<path d="M${x + 1} 73 L${x + 4} 66 L${x + 7} 66" stroke="#FF6F91" stroke-width="1.5" fill="none" stroke-linecap="round"/>
    <circle cx="${x - 4}" cy="73.4" r="2.6" fill="#FFE27A"/></g>`;
  const partyHat = x => `<g>${shade(x, 7)}<path d="M${x - 7} 89 L${x} 70 L${x + 7} 89 Z" fill="#7FB3E8"/><path d="M${x - 4.6} 83 L${x + 4.6} 83 M${x - 2.6} 77 L${x + 2.6} 77" stroke="#FFE08A" stroke-width="1.8"/>
    <circle cx="${x}" cy="69.4" r="2.6" fill="#FF8FA3"/></g>`;
  const cocoa = x => mug(x, '#6B3E26', `<circle cx="${x - 2}" cy="74.8" r="1.6" fill="#FFFFFF"/><circle cx="${x + 2}" cy="74.4" r="1.4" fill="#FFF4F0"/>`);

  /* holiday views: what hangs in or happens outside the window */
  const burst = (cx, cy, r, color) => {
    let rays = '';
    for (let k = 0; k < 10; k++) {
      const a = k * Math.PI / 5;
      rays += `<path d="M${(cx + Math.cos(a) * r * .35).toFixed(1)} ${(cy + Math.sin(a) * r * .35).toFixed(1)} L${(cx + Math.cos(a) * r).toFixed(1)} ${(cy + Math.sin(a) * r).toFixed(1)}"/>`;
    }
    return `<g class="sb-burst" stroke="${color}" stroke-width="1.6" stroke-linecap="round">${rays}<circle cx="${cx}" cy="${cy}" r="1.4" fill="${color}" stroke="none"/></g>`;
  };
  const lanterns = [[30, 0], [92, 4]].map(([x, d]) => `<g class="sb-sway"><path d="M${x} 0 V${8 + d}" stroke="#C9A14A" stroke-width="1.1"/>
    <rect x="${x - 3.6}" y="${8 + d}" width="7.2" height="2.4" rx="1" fill="#F2C14E"/><ellipse cx="${x}" cy="${17 + d}" rx="8" ry="7.2" fill="#E2483D"/>
    <path d="M${x - 4.6} ${17 + d} h9.2" stroke="#B83229" stroke-width=".8" opacity=".6"/><rect x="${x - 3.6}" y="${23.4 + d}" width="7.2" height="2.4" rx="1" fill="#F2C14E"/>
    <path d="M${x} ${25.8 + d} v5" stroke="#F2C14E" stroke-width="1.2"/></g>`).join('');
  const falling = (colors, shape) => [[16, 10], [32, 2], [50, 14], [74, 6], [94, 16], [106, 4], [24, 28], [86, 30]]
    .map(([x, y], i) => shape(x, y, colors[i % colors.length])).join('');
  const dot = (x, y, color) => `<circle class="sb-fall" cx="${x}" cy="${y}" r="1.4" fill="${color}"/>`;
  const chip = (x, y, color) => `<rect class="sb-fall" x="${x}" y="${y}" width="3.6" height="2.2" rx=".6" fill="${color}"/>`;
  const leaf = (x, y, color) => `<path class="sb-fall" d="M${x} ${y} c2.6 -2.6 6.2 -1.8 6.2 1.8 c-2.6 2.6 -6.2 1.8 -6.2 -1.8 Z" fill="${color}"/>`;
  const FULL_MOON = `<circle class="sb-glow" cx="84" cy="26" r="30" fill="url(#ID-glow)"/><circle cx="84" cy="26" r="12" fill="#FFF1C1"/>
    <circle cx="80.4" cy="23" r="2" fill="#F1DFA3"/><circle cx="88" cy="30" r="1.5" fill="#F1DFA3"/><circle cx="87" cy="21.4" r="1" fill="#F1DFA3"/>`;
  const LIGHTS = ['#FF6B5A', '#FFD36B', '#7CD6A6', '#6FA8FF'].flatMap((color, i) => [8 + i * 26, 21 + i * 26].map(x =>
    `<circle class="sb-twinkle" cx="${x}" cy="${46.4 + (x % 3)}" r="1.6" fill="${color}"/>`)).join('');
  const BATS = `<path class="sb-flap" d="M20 28 q3.4 -3.4 6.8 0 q1.7 -1.7 3.4 0 q3.4 -3.4 6.8 0 q-3.4 .9 -5.1 3.4 q-1.7 -1.7 -3.4 0 q-1.7 -2.6 -8.5 -3.4 Z" fill="#1A1424"/>
    <path class="sb-flap" d="M58 14 q2.6 -2.6 5.2 0 q1.3 -1.3 2.6 0 q2.6 -2.6 5.2 0 q-2.6 .8 -3.9 2.6 q-1.3 -1.3 -2.6 0 q-1.3 -1.8 -6.5 -2.6 Z" fill="#1A1424"/>`;
  // sky: a view key or [top, bottom]; window: drawn outside; sill: [object, x] pairs
  const LOOKS = {
    party: {window: falling(['#FF8FA3', '#FFE08A', '#7FB3E8', '#9BE3B8'], chip), sill: date => [mug(44, '#6B3E26'), partyHat(84)]},
    spring: {window: lanterns, festiveNight: lanterns + burst(60, 22, 11, '#FFD36B') + burst(40, 34, 7, '#FF6B5A'), sill: () => [dumplings(44), teacup(84)]},
    lantern: {window: lanterns, sill: () => [tangyuan(44), teacup(84)]},
    rest: {sill: () => [juice(46), shades(82)]},
    zongzi: {window: `<path d="M0 84 C6 70 16 62 28 58 C20 68 12 76 6 84 Z M120 84 C114 70 104 62 92 58 C100 68 108 76 114 84 Z" fill="#5FA05A" opacity=".7"/>`, sill: () => [zongzi(44), teacup(84)]},
    apple: {sill: () => [books(42), apple(82)]},
    moon: {sky: 'evening', window: FULL_MOON + stars([[22, 16, 1], [40, 8, .8], [52, 30, .7]]), sill: () => [mooncakes(44), teacup(84)]},
    fireworks: {sky: 'evening', window: burst(28, 24, 12, '#FF6B5A') + burst(88, 20, 14, '#FFD36B') + burst(62, 10, 8, '#FF9F5A')},
    'fireworks-us': {sky: 'evening', window: burst(28, 24, 12, '#FF5A5A') + burst(88, 20, 14, '#FFFFFF') + burst(62, 10, 8, '#6FA8FF')},
    code: {window: `<text class="sb-float" x="16" y="34" font-family="ui-monospace, monospace" font-size="9" font-weight="700" fill="#FFFFFF" opacity=".55">{ }</text>
        <text class="sb-float" x="84" y="62" font-family="ui-monospace, monospace" font-size="8" font-weight="700" fill="#FFFFFF" opacity=".5">&lt;/&gt;</text>`,
      sill: () => [laptop(46), mug(86, '#6B3E26')]},
    pie: {sill: () => [pie(44), mug(84, '#6B3E26')]},
    pumpkin: {sky: ['#2A1E40', '#7A4A66'], window: `<circle cx="88" cy="24" r="10" fill="#FFD68A" opacity=".9"/>` + BATS, sill: () => [jack(44), candle(84)]},
    autumn: {window: falling(['#E0752F', '#C9452B', '#E8A642'], leaf), sill: () => [slice(44), mug(84, '#6B3E26')]},
    christmas: {sky: ['#1F2E4C', '#5A6C9C'], window: falling(['#FFFFFF', '#EAF2FF'], dot), lights: LIGHTS, sill: () => [gift(44), cocoa(84)]}
  };
  const SILL = {
    morning: () => [mug(44, '#6B3E26'), plant(84)],
    noon: () => [bowl(46), glass(84)],
    afternoon: () => [teapot(37), teacup(64), cookies(89)],
    evening: () => [openBook(48), lamp(92)],
    night: date => [mug(44, '#FFF8EE'), clock(84, date)]
  };
  function art(period, id, look = '', date = new Date(2026, 9, 13, 1, 10)) {
    const spec = LOOKS[look] || null;
    const night = period === 'night';
    const festive = night && spec?.festiveNight;
    const viewKey = night ? 'night' : Array.isArray(spec?.sky) ? null : spec?.sky || period;
    const sky = viewKey ? VIEW[viewKey].sky : spec.sky;
    const hills = VIEW[viewKey || (night ? 'night' : 'evening')].hills;
    const outside = festive || (spec?.window ? (night && !spec.sky ? SKYLIFE.night : '') + spec.window : SKYLIFE[night ? 'night' : viewKey || 'evening']);
    const things = (spec?.sill || SILL[period])(date);
    const warm = night || period === 'evening' || viewKey === 'evening';
    return `<svg class="sb-svg sb-${period}${spec ? ` sb-look-${look}` : ''}" viewBox="0 0 120 120" aria-hidden="true" focusable="false">
      <defs><linearGradient id="${id}-sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${sky[0]}"/><stop offset="1" stop-color="${sky[1]}"/></linearGradient>
        <radialGradient id="${id}-glow"><stop offset="0" stop-color="#FFE7B8" stop-opacity=".5"/><stop offset="1" stop-color="#FFE7B8" stop-opacity="0"/></radialGradient>
        <radialGradient id="${id}-lamp"><stop offset="0" stop-color="#FFD58A" stop-opacity=".55"/><stop offset="1" stop-color="#FFD58A" stop-opacity="0"/></radialGradient>
        <linearGradient id="${id}-sill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#A26D45"/><stop offset="1" stop-color="#C48D5F"/></linearGradient>
        <linearGradient id="${id}-front" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#80573A"/><stop offset="1" stop-color="#593A27"/></linearGradient>
        <linearGradient id="${id}-cloud" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#FFFFFF" stop-opacity=".62"/></linearGradient>
        <filter id="${id}-soft" x="-50%" y="-200%" width="200%" height="500%"><feGaussianBlur stdDeviation="1.1"/></filter>
        <clipPath id="${id}-clip"><circle cx="60" cy="60" r="58"/></clipPath></defs>
      <g clip-path="url(#${id}-clip)">
        <rect width="120" height="84" fill="url(#${id}-sky)"/>
        ${outside.replace(/url\(#ID-([a-z]+)\)/g, `url(#${id}-$1)`)}
        <path d="M0 65 C20 59 40 60 60 63 C80 66 100 58 120 59 V84 H0 Z" fill="${hills[0]}" opacity=".42"/>
        <path d="M0 70 C16 63 32 65 48 69 C64 73 82 64 98 62.6 C108 62 115 64 120 66 V84 H0 Z" fill="${hills[0]}" opacity=".8"/>
        <path d="M0 77 C20 72 38 74 56 77 C74 80 92 72 120 74 V84 H0 Z" fill="${hills[1]}"/>
        ${warm ? `<circle cx="28" cy="76" r="36" fill="url(#${id}-lamp)" opacity=".45"/>` : ''}
        <path d="M10 0 L30 0 L4 64 L-16 64 Z" fill="#FFFFFF" opacity=".07"/><path d="M38 0 L41.5 0 L15.5 64 L12 64 Z" fill="#FFFFFF" opacity=".06"/>
        <path d="M60 0 V84 M0 46 H120" stroke="${FRAME}" stroke-width="1.9"/><path d="M60.9 0 V84 M0 46.9 H120" stroke="#A0755A" stroke-width=".6" opacity=".55"/>
        ${spec?.lights || ''}
        <rect y="84" width="120" height="7" fill="url(#${id}-sill)"/><rect y="84" width="120" height="1" fill="#5A3B27" opacity=".35"/>
        <rect y="90.2" width="120" height="1" fill="#E6B587" opacity=".6"/>
        <rect y="91.2" width="120" height="28.8" fill="url(#${id}-front)"/><rect y="91.2" width="120" height="2" fill="#4E3322" opacity=".55"/>
        <path d="M10 99 h38 M62 104 h44 M16 110 h30 M70 114 h26" stroke="#68452E" stroke-width="1" stroke-linecap="round" opacity=".7"/>
        ${warm ? `<rect y="84" width="120" height="36" fill="#FFB866" opacity=".08"/>` : ''}
        <g class="sb-sill" transform="translate(60 89) scale(1.3) translate(-60 -89)">${things.join('').replace(/url\(#ID-([a-z]+)\)/g, `url(#${id}-$1)`)}</g>
      </g>
      <circle cx="60" cy="60" r="56.8" fill="none" stroke="${FRAME}" stroke-width="3"/><circle cx="60" cy="60" r="55.1" fill="none" stroke="#A0755A" stroke-width=".7" opacity=".45"/>
      <circle cx="60" cy="60" r="58.6" fill="none" stroke="rgba(255,255,255,.16)" stroke-width="1.2"/>
    </svg>`;
  }

  const api = {periodAt, line, weekday, holidayAt, moment, todayLine, liveLine, art, LINES, LABELS, HOLIDAYS, LOOKS};
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }
  if (typeof document === 'undefined') return;

  /* ---------------------------------------------------------------- the corner */
  const zh = (document.documentElement.lang || '').startsWith('zh');
  const lang = zh ? 'zh' : 'en';
  const say = (chinese, english) => (zh ? chinese : english);
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } };
  const save = patch => { try { localStorage.setItem(KEY, JSON.stringify({...load(), ...patch})); } catch {} };
  const civil = date => `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  const el = (tag, attrs = {}, kids = []) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) if (key === 'text') node.textContent = value; else node.setAttribute(key, value);
    node.append(...kids);
    return node;
  };

  const root = el('div', {class: 'study-buddy'});
  const card = el('aside', {class: 'sb-card', 'aria-label': say('学习小伙伴', 'Study buddy')});
  const cardArt = el('div', {class: 'sb-art'});
  const kicker = el('p', {class: 'sb-kicker'});
  const message = el('p', {class: 'sb-message'});
  const today = el('p', {class: 'sb-today'});
  const live = el('p', {class: 'sb-live', hidden: ''});
  const fold = el('button', {type: 'button', class: 'sb-fold', 'aria-label': say('收起', 'Fold away'), 'aria-expanded': 'true', text: '›'});
  card.append(cardArt, el('div', {class: 'sb-words'}, [kicker, message, today, live]), fold);
  const avatar = el('button', {type: 'button', class: 'sb-avatar', 'aria-label': say('展开学习小伙伴', 'Open the study buddy'), 'aria-expanded': 'false'});
  const bubble = el('p', {class: 'sb-bubble', role: 'status', hidden: ''});
  root.append(bubble, avatar, card);
  document.body.append(root);

  let scene = '';
  let current = null;
  let liveData = null;
  const records = () => scope.PreparationProgress?.getState?.()?.workspace?.records || {};
  function openNotes() {
    const hub = scope.ReaderHub;
    if (!hub || !scope.PreparationProgress) return null;
    if (hub.settings(records()).history === false) return null;
    return hub.summarize(records(), new Date()).today.notes.length;
  }
  // the reader can switch the buddy off in the reading settings drawer
  const switchedOff = () => Boolean(scope.ReaderHub) && scope.ReaderHub.settings(records()).buddy === 'hidden';
  function render() {
    const now = new Date();
    current = moment(now, lang);
    root.hidden = switchedOff();
    if (root.hidden) return;
    const look = current.holiday ? current.holiday.look : '';
    // late at night the clock on the sill shows the real time
    const key = `${current.period}:${look}:${current.period === 'night' && !look ? now.getHours() * 60 + now.getMinutes() : ''}`;
    if (key !== scene) {
      scene = key;
      cardArt.innerHTML = art(current.period, 'sb-card-art', look, now);
      avatar.innerHTML = art(current.period, 'sb-avatar-art', look, now);
      root.dataset.period = current.period;
    }
    kicker.textContent = `${current.label} · ${pad(now.getHours())}:${pad(now.getMinutes())}`;
    message.textContent = current.text;
    const count = openNotes();
    today.hidden = count === null;
    if (count !== null) today.textContent = todayLine(count, current.period, lang);
    const words = liveLine(liveData, lang);
    live.hidden = !words;
    live.textContent = words;
  }
  function setOpen(open, remember = true) {
    root.classList.toggle('is-open', open);
    fold.setAttribute('aria-expanded', String(open));
    avatar.setAttribute('aria-expanded', String(open));
    if (open) bubble.hidden = true;
    if (remember) save({open});
  }
  fold.addEventListener('click', () => { setOpen(false); avatar.focus(); });
  avatar.addEventListener('click', () => { setOpen(true); fold.focus(); });

  render();
  const stored = load();
  setOpen(typeof stored.open === 'boolean' ? stored.open : scope.matchMedia('(min-width: 900px)').matches, false);
  // Folded away, it still says hello once per part of the day (and holiday), so late at night it always does.
  const seen = `${civil(new Date())}:${current.period}:${current.holiday ? current.holiday.id : ''}`;
  if (!root.hidden && !root.classList.contains('is-open') && stored.seen !== seen) {
    scope.setTimeout(() => {
      if (root.hidden || root.classList.contains('is-open')) return;
      bubble.textContent = current.text;
      bubble.hidden = false;
      save({seen});
      scope.setTimeout(() => { bubble.hidden = true; }, 9000);
    }, 2500);
  }
  scope.setInterval(render, 30000);
  scope.addEventListener('notes:prep-storage', render);
  scope.addEventListener('notes:prep-restored', render);
  scope.addEventListener('notes:reader-settings', render);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') render(); });
  scope.StudyBuddy = {setLive(value) { liveData = value; render(); }};
})(typeof window === 'undefined' ? globalThis : window);
