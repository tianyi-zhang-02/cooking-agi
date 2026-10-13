/* Study buddy: a small companion in the corner that follows the reader's local
   time of day (coffee in the morning, a doze after lunch, afternoon tea, a lamp
   in the evening, a blanket late at night) and says how many notes the reader
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

  /* ---------------------------------------------------------------- the drawing */
  const SKIN = '#F6D3B5', EAR = '#EDBE9C', HAIR = '#3B2A22', HOOD = '#E8A672', HOOD_DARK = '#C9834F', INK = '#4A2E22';
  const SKY = {morning: ['#FFE6BC', '#F4AE86'], noon: ['#D9E9F2', '#F5E6C6'], afternoon: ['#F8D6A2', '#E3955E'], evening: ['#283050', '#4C3A4D'], night: ['#10172F', '#231A3B']};
  const steam = (x, y) => `<g class="sb-steam" transform="translate(${x} ${y})" fill="none" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round">
    <path d="M-5 0 C-8 -4 -2 -6 -5 -10"/><path d="M1 0 C-2 -4 4 -6 1 -10"/><path d="M7 0 C4 -4 10 -6 7 -10"/></g>`;
  function mug(liquid) {
    return `<g class="sb-sip"><g transform="translate(62 104)">
      <path d="M10 -5 h3 a4.2 4.2 0 0 1 0 8.4 h-3" stroke="#EADBC6" stroke-width="2.6" fill="none"/>
      <rect x="-10" y="-9" width="20" height="18" rx="4.5" fill="#FFF4E4"/>
      <rect x="-10" y="-1" width="20" height="3" fill="#F2C48F" opacity=".7"/>
      <ellipse cx="0" cy="-8.3" rx="8.4" ry="2" fill="${liquid}"/>
      <ellipse cx="-11" cy="1" rx="4.6" ry="4" fill="${SKIN}"/><ellipse cx="11.5" cy="2.5" rx="4.4" ry="3.8" fill="${SKIN}"/>
      ${steam(0, -12)}</g></g>`;
  }
  const teacup = `<g class="sb-sip"><g transform="translate(62 104)">
      <ellipse cx="0" cy="7" rx="17" ry="3.6" fill="#F7EBDC"/>
      <path d="M11 -4 a4 4 0 0 1 0 7" stroke="#F7EBDC" stroke-width="2.4" fill="none"/>
      <path d="M-12 -6 h24 v2 a12 10 0 0 1 -24 0 Z" fill="#FFFFFF"/>
      <path d="M-12 -4 h24" stroke="#E8B04A" stroke-width="1.2"/>
      <ellipse cx="0" cy="-6" rx="11.4" ry="2.2" fill="#C8773C"/>
      <ellipse cx="-16" cy="6" rx="4.4" ry="3.6" fill="${SKIN}"/><ellipse cx="16" cy="6" rx="4.4" ry="3.6" fill="${SKIN}"/>
      ${steam(0, -10)}</g></g>`;
  const bowl = `<g transform="translate(60 104)">
      <path d="M-15 -5 C-11 -13 11 -13 15 -5 Z" fill="#FFFFFF"/>
      <path d="M-16 -5 h32 a16 12 0 0 1 -32 0 Z" fill="#F3E4CF"/>
      <path d="M-15 -1 h30" stroke="#7FA6C9" stroke-width="2"/>
      <path d="M8 -9 L24 -24 M11 -8 L27 -22" stroke="#9B6B43" stroke-width="1.8" stroke-linecap="round"/>
      <ellipse cx="-16" cy="2" rx="4.4" ry="3.8" fill="${SKIN}"/><ellipse cx="16" cy="2" rx="4.4" ry="3.8" fill="${SKIN}"/></g>`;
  const book = `<g transform="translate(60 104)">
      <path d="M0 -6 L-22 -10 L-22 8 L0 12 Z" fill="#FFF8EC"/><path d="M0 -6 L22 -10 L22 8 L0 12 Z" fill="#FBEFDC"/>
      <path d="M-18 -5 L-4 -2.5 M-18 -1 L-4 1.5 M-18 3 L-8 4.8 M4 -2.5 L18 -5 M4 1.5 L18 -1" stroke="#D9C7AE" stroke-width="1"/>
      <path class="sb-page" d="M0 -6 L20 -9.4 L20 7.6 L0 12 Z" fill="#FFFBF3"/>
      <path d="M0 -6 V12" stroke="#C9A97F" stroke-width="1.2"/>
      <ellipse cx="-23" cy="3" rx="4.2" ry="3.6" fill="${SKIN}"/><ellipse cx="23" cy="3" rx="4.2" ry="3.6" fill="${SKIN}"/></g>`;
  const EYES = {
    open: `<g class="sb-eyes"><ellipse cx="52" cy="67" rx="2.3" ry="2.9" fill="${INK}"/><ellipse cx="68" cy="67" rx="2.3" ry="2.9" fill="${INK}"/>
      <circle cx="52.8" cy="66" r=".8" fill="#FFFFFF"/><circle cx="68.8" cy="66" r=".8" fill="#FFFFFF"/></g>`,
    down: `<g class="sb-eyes"><path d="M49.5 68 Q52 70.2 54.5 68" stroke="${INK}" stroke-width="1.8" fill="none" stroke-linecap="round"/>
      <path d="M65.5 68 Q68 70.2 70.5 68" stroke="${INK}" stroke-width="1.8" fill="none" stroke-linecap="round"/></g>`,
    sleepy: `<g><path d="M49 67.5 Q52 69.5 55 67.5" stroke="${INK}" stroke-width="1.8" fill="none" stroke-linecap="round"/>
      <path d="M65 67.5 Q68 69.5 71 67.5" stroke="${INK}" stroke-width="1.8" fill="none" stroke-linecap="round"/></g>`
  };
  function head(eyes, cap) {
    return `<g class="sb-head">
      <circle cx="39" cy="67" r="4" fill="${EAR}"/><circle cx="81" cy="67" r="4" fill="${EAR}"/>
      <circle cx="60" cy="64" r="22" fill="${SKIN}"/>
      ${cap ? `<path d="M37 62 C36 44 48 37 61 37 C70 37 77 41 81 47 C86 53 90 61 92 70 L88 71 C85 62 82 57 78 54 C70 58 54 58 45 55 C41 57 39 59 37 62 Z" fill="#5C6CA6"/>
        <path d="M37 61 C44 55 70 55 80 56" stroke="#7F8FC6" stroke-width="3" fill="none" stroke-linecap="round"/><circle cx="90" cy="72" r="3.6" fill="#FFF3DD"/>`
        : `<path d="M38 64 C35 46 47 38 60 38 C73 38 85 46 82 64 C80 57 76 53 70 52 C66 57 57 58 49 55 C44 57 40 60 38 64 Z" fill="${HAIR}"/>
          <path d="M60 39 C58 33 63 30 65 33" stroke="${HAIR}" stroke-width="2.4" fill="none" stroke-linecap="round"/>`}
      ${EYES[eyes]}
      <ellipse cx="47" cy="73" rx="3.6" ry="2" fill="#F08F7E" opacity=".45"/><ellipse cx="73" cy="73" rx="3.6" ry="2" fill="#F08F7E" opacity=".45"/>
      ${eyes === 'sleepy' ? `<ellipse class="sb-yawn" cx="60" cy="76" rx="2.2" ry="2.6" fill="#8A4A3A"/>` : `<path d="M57 74.5 Q60 77.5 63 74.5" stroke="${INK}" stroke-width="1.6" fill="none" stroke-linecap="round"/>`}
    </g>`;
  }
  const hoodie = (color = HOOD, trim = HOOD_DARK) => `<path d="M22 122 C24 100 38 90 60 90 C82 90 96 100 98 122 Z" fill="${color}"/>
    <path d="M48 91 Q60 100 72 91" stroke="${trim}" stroke-width="2.2" fill="none" stroke-linecap="round"/>
    <path d="M55 97 v9 M65 97 v9" stroke="#FBE3C9" stroke-width="1.4" stroke-linecap="round"/>`;
  const blanket = `<path d="M16 122 C16 99 33 86 60 86 C87 86 104 99 104 122 Z" fill="#6E7EB8"/>
    <path d="M24 104 C44 97 76 97 96 104 M20 114 C44 107 76 107 100 114" stroke="#8C9AD0" stroke-width="3" fill="none"/>`;
  const PROPS = {
    morning: `<circle class="sb-sun" cx="94" cy="98" r="15" fill="#FFD27C"/><rect y="104" width="120" height="16" fill="#F2A274" opacity=".5"/>`,
    noon: `<circle cx="95" cy="25" r="10" fill="#FFE39C"/><circle cx="95" cy="25" r="15" fill="#FFE39C" opacity=".25"/>`,
    afternoon: `<circle cx="96" cy="36" r="12" fill="#FFD28A" opacity=".95"/><path d="M0 92 C30 86 60 90 120 84 V120 H0 Z" fill="#D88350" opacity=".35"/>`,
    evening: `<circle class="sb-glow" cx="24" cy="46" r="44" fill="url(#ID-glow)"/>
      <path d="M11 26 L35 26 L30 37 L16 37 Z" fill="#F5B83D"/><rect x="22" y="37" width="2" height="34" fill="#7A5E44"/>`,
    night: `<path d="M99 15 a13 13 0 1 0 7 24 a10.5 10.5 0 1 1 -7 -24 Z" fill="#FFE6A6"/>
      <circle class="sb-twinkle" cx="22" cy="24" r="1.4" fill="#FFF"/><circle class="sb-twinkle" cx="38" cy="14" r="1" fill="#FFF"/>
      <circle class="sb-twinkle" cx="80" cy="46" r="1.1" fill="#FFF"/><circle class="sb-twinkle" cx="14" cy="50" r=".9" fill="#FFF"/>`
  };
  const TEAPOT = `<g transform="translate(28 101)"><path d="M-8 -2 C-8 -11 8 -11 8 -2 C8 5 -8 5 -8 -2 Z" fill="#FFF6EA"/><path d="M8 -4 C13 -6 14 -10 16 -11" stroke="#FFF6EA" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <path d="M-8 -5 C-12 -5 -12 1 -8 1" stroke="#FFF6EA" stroke-width="1.8" fill="none"/><ellipse cx="0" cy="-9.5" rx="3.4" ry="1.4" fill="#E8B04A"/><circle cx="0" cy="-11.4" r="1.4" fill="#E8B04A"/></g>`;
  const ITEMS = {morning: mug('#6B3E26'), noon: bowl, afternoon: TEAPOT + teacup, evening: book, night: mug('#FFF8EE')};
  const Z = `<g class="sb-zs" fill="#FFFFFF" font-family="ui-rounded, system-ui, sans-serif" font-size="9" font-weight="700">
      <text class="sb-z" x="84" y="54">z</text><text class="sb-z" x="90" y="46">z</text></g>`;

  /* holiday looks: a hat, something in hand, and the scene around */
  const hands = (dx = 16, dy = 3) => `<ellipse cx="${-dx}" cy="${dy}" rx="4.4" ry="3.8" fill="${SKIN}"/><ellipse cx="${dx}" cy="${dy}" rx="4.4" ry="3.8" fill="${SKIN}"/>`;
  const held = body => `<g transform="translate(60 104)">${body}</g>`;
  const burst = (cx, cy, r, color) => {
    let rays = '';
    for (let k = 0; k < 10; k++) {
      const a = k * Math.PI / 5, x1 = (cx + Math.cos(a) * r * .35).toFixed(1), y1 = (cy + Math.sin(a) * r * .35).toFixed(1);
      rays += `<path d="M${x1} ${y1} L${(cx + Math.cos(a) * r).toFixed(1)} ${(cy + Math.sin(a) * r).toFixed(1)}"/>`;
    }
    return `<g class="sb-burst" stroke="${color}" stroke-width="1.8" stroke-linecap="round">${rays}<circle cx="${cx}" cy="${cy}" r="1.6" fill="${color}" stroke="none"/></g>`;
  };
  const lanterns = ['24', '98'].map((x, i) => `<g class="sb-sway"><path d="M${x} 0 V${10 + i * 4}" stroke="#C9A14A" stroke-width="1.2"/>
      <rect x="${x - 4}" y="${10 + i * 4}" width="8" height="2.6" rx="1" fill="#F2C14E"/><ellipse cx="${x}" cy="${20 + i * 4}" rx="9" ry="8" fill="#E2483D"/>
      <path d="M${x - 5} ${20 + i * 4} h10" stroke="#B83229" stroke-width=".8" opacity=".6"/><rect x="${x - 4}" y="${27 + i * 4}" width="8" height="2.6" rx="1" fill="#F2C14E"/>
      <path d="M${x} ${29.6 + i * 4} v6" stroke="#F2C14E" stroke-width="1.4"/></g>`).join('');
  const falling = (colors, shape) => [[16, 12], [34, 4], [52, 16], [76, 8], [94, 18], [106, 6], [24, 30], [88, 32]]
    .map(([x, y], i) => shape(x, y, colors[i % colors.length])).join('');
  const dot = (x, y, color) => `<circle class="sb-fall" cx="${x}" cy="${y}" r="1.6" fill="${color}"/>`;
  const chip = (x, y, color) => `<rect class="sb-fall" x="${x}" y="${y}" width="4" height="2.4" rx=".6" fill="${color}"/>`;
  const leaf = (x, y, color) => `<path class="sb-fall" d="M${x} ${y} c3 -3 7 -2 7 2 c-3 3 -7 2 -7 -2 Z" fill="${color}"/>`;
  const HATS = {
    party: `<path d="M49 47 L64 15 L74 44 Z" fill="#7FB3E8"/><path d="M53 38.5 L71 37 M57 29.5 L68 28.8 M60.5 22 L66 21.6" stroke="#FFE08A" stroke-width="2.2" stroke-linecap="round"/><circle cx="64" cy="14" r="3.6" fill="#FF8FA3"/>`,
    santa: `<path d="M38 51 C38 37 50 30 62 30 C74 30 82 36 88 46 C91 51 93 57 94 62 L89 63 C86 55 82 50 76 47 C64 45 50 46 38 51 Z" fill="#D9473C"/>
      <path d="M36 51 C48 44.5 70 43.5 80 46.5" stroke="#FFF7EE" stroke-width="6" stroke-linecap="round" fill="none"/><circle cx="92" cy="64" r="4.2" fill="#FFF7EE"/>`,
    witch: `<path d="M33 51 C45 46 76 46 88 51 C76 55 45 55 33 51 Z" fill="#2B2238"/><path d="M46 49 C52 36 56 24 62 16 C64 14 67 15 66 18 C68 28 72 40 75 49 Z" fill="#3A2D4E"/>
      <path d="M47 45.5 C56 43.5 66 43.5 74 45.5" stroke="#E8A672" stroke-width="2.6" fill="none"/>`
  };
  const SHADES = `<rect x="44.5" y="63" width="13" height="8" rx="3.5" fill="#2B2B35"/><rect x="62.5" y="63" width="13" height="8" rx="3.5" fill="#2B2B35"/>
    <path d="M57.5 66 h5" stroke="#2B2B35" stroke-width="1.6"/><path d="M47 65.5 l3 -1.4 M65 65.5 l3 -1.4" stroke="#FFFFFF" stroke-width="1" opacity=".55"/>`;
  const MOON = `<circle class="sb-glow" cx="88" cy="30" r="28" fill="url(#ID-glow)"/><circle cx="88" cy="30" r="13" fill="#FFF1C1"/>
    <circle cx="84" cy="27" r="2.2" fill="#F1DFA3"/><circle cx="92" cy="34" r="1.6" fill="#F1DFA3"/><circle cx="91" cy="25" r="1.1" fill="#F1DFA3"/>`;
  const LOOKS = {
    party: {hat: HATS.party, props: falling(['#FF8FA3', '#FFE08A', '#7FB3E8', '#9BE3B8'], chip)},
    spring: {hood: ['#D9473C', '#F2C14E'], props: lanterns, festiveNight: lanterns + burst(60, 18, 12, '#FFD36B') + burst(40, 34, 8, '#FF6B5A'),
      item: held(`<ellipse cx="0" cy="5" rx="18" ry="4.6" fill="#FFFFFF"/><ellipse cx="0" cy="4.4" rx="14" ry="3" fill="#F0E6DA"/>
        <path d="M-13 3 Q-8.5 -7 -4 3 Z M-4.5 2 Q0 -8 4.5 2 Z M4 3 Q8.5 -7 13 3 Z" fill="#FFF8EC" stroke="#E6D3B9" stroke-width=".8"/>${hands(18, 5)}`)},
    lantern: {props: lanterns,
      item: held(`<path d="M-16 -5 h32 a16 12 0 0 1 -32 0 Z" fill="#F3E4CF"/><path d="M-15 -1 h30" stroke="#D9473C" stroke-width="2"/>
        <circle cx="-6" cy="-6" r="4.2" fill="#FFFFFF"/><circle cx="2" cy="-7" r="4.2" fill="#FFF6F2"/><circle cx="9" cy="-5.4" r="3.8" fill="#FFFFFF"/>${hands(16, 2)}`)},
    rest: {face: SHADES,
      item: held(`<path d="M-7 -13 L7 -13 L5 8 L-5 8 Z" fill="#DDF1F7" opacity=".85"/><path d="M-6.3 -6 L6.3 -6 L5 8 L-5 8 Z" fill="#FFB347"/>
        <rect x="-4" y="-4" width="3.4" height="3.4" rx=".8" fill="#FFFFFF" opacity=".6"/><rect x="1" y="-1" width="3" height="3" rx=".8" fill="#FFFFFF" opacity=".5"/>
        <path d="M2 -13 L6 -21 L10 -21" stroke="#FF6F91" stroke-width="1.8" fill="none" stroke-linecap="round"/><circle cx="-6" cy="-13" r="3" fill="#FFE27A"/>${hands(9, 4)}`)},
    zongzi: {props: `<path d="M4 120 C12 96 26 84 44 78 C34 92 22 106 4 120 Z" fill="#5FA05A" opacity=".55"/><path d="M116 120 C108 96 94 84 76 78 C86 92 98 106 116 120 Z" fill="#5FA05A" opacity=".55"/>`,
      item: held(`<path d="M-11 6 L0 -13 L11 6 Z" fill="#5FA05A"/><path d="M-11 6 L0 -13 L3 -7 L-5 6 Z" fill="#79B86E"/>
        <path d="M-7.5 0 L7.5 0 M-4.5 -5 L4.5 -5" stroke="#EADBA6" stroke-width="1.5"/>${hands(14, 3)}`)},
    apple: {item: held(`<path d="M0 -6 C-6 -11 -13 -6 -12 1 C-11 8 -5 11 0 9 C5 11 11 8 12 1 C13 -6 6 -11 0 -6 Z" fill="#E2483D"/>
        <path d="M0 -6 C0 -9 1 -11 2 -12" stroke="#7A5E44" stroke-width="1.4" fill="none"/><path d="M2 -10 C6 -13 9 -11 9 -9 C6 -8 4 -9 2 -10 Z" fill="#79B86E"/>
        <ellipse cx="-5" cy="-2" rx="2" ry="3" fill="#FFFFFF" opacity=".35"/>${hands(14, 4)}`)},
    moon: {sky: 'evening', props: MOON + `<path d="M64 46 h28 a5 5 0 0 0 -8 -5 a6 6 0 0 0 -11 1 a4 4 0 0 0 -9 4 Z" fill="#FFFFFF" opacity=".22"/>
        <circle class="sb-twinkle" cx="20" cy="22" r="1.2" fill="#FFF"/><circle class="sb-twinkle" cx="36" cy="12" r="1" fill="#FFF"/>`,
      item: held(`<circle r="11.5" fill="#D99A46"/><circle r="8.6" fill="#E8B25E"/><circle r="8.6" fill="none" stroke="#C7843A" stroke-width="1" stroke-dasharray="2 1.6"/>
        <path d="M-3 -2.5 h6 M0 -5.5 v6.5 M-3 2.5 h6" stroke="#C7843A" stroke-width="1.2" stroke-linecap="round"/>${hands(14, 3)}`)},
    fireworks: {sky: 'evening', props: burst(26, 26, 13, '#FF6B5A') + burst(92, 22, 15, '#FFD36B') + burst(64, 12, 9, '#FF9F5A')},
    'fireworks-us': {sky: 'evening', props: burst(26, 26, 13, '#FF5A5A') + burst(92, 22, 15, '#FFFFFF') + burst(64, 12, 9, '#6FA8FF')},
    code: {props: `<text class="sb-float" x="18" y="34" font-family="ui-monospace, monospace" font-size="10" font-weight="700" fill="#FFFFFF" opacity=".55">{ }</text>
        <text class="sb-float" x="86" y="30" font-family="ui-monospace, monospace" font-size="9" font-weight="700" fill="#FFFFFF" opacity=".5">&lt;/&gt;</text>`,
      item: `<g transform="translate(60 106)"><path d="M-17 -18 h34 v19 h-34 Z" fill="#2B2F3A"/><rect class="sb-screen" x="-15" y="-16" width="30" height="15" rx="1" fill="#14202B"/>
        <text x="0" y="-5.6" text-anchor="middle" font-family="ui-monospace, monospace" font-size="7.5" font-weight="700" fill="#7CF2B5">1024</text>
        <path d="M-21 1 h42 l-3 5 h-36 Z" fill="#A3ABB9"/>${hands(19, 4)}</g>`},
    pie: {item: held(`<ellipse cx="0" cy="3" rx="17" ry="6" fill="#D9934A"/><ellipse cx="0" cy="1" rx="14" ry="4.6" fill="#B5462F"/>
        <text x="0" y="4" text-anchor="middle" font-family="Georgia, serif" font-size="8.5" font-weight="700" fill="#FFE9C9">π</text>${hands(18, 4)}`)},
    pumpkin: {sky: ['#2A1E40', '#5B3A5C'], hat: HATS.witch,
      props: `<circle cx="92" cy="26" r="11" fill="#FFD68A" opacity=".9"/>
        <path class="sb-flap" d="M18 30 q4 -4 8 0 q2 -2 4 0 q4 -4 8 0 q-4 1 -6 4 q-2 -2 -4 0 q-2 -3 -10 -4 Z" fill="#1A1424"/>
        <path class="sb-flap" d="M70 16 q3 -3 6 0 q1.5 -1.5 3 0 q3 -3 6 0 q-3 1 -4.5 3 q-1.5 -1.5 -3 0 q-1.5 -2 -7.5 -3 Z" fill="#1A1424"/>`,
      item: held(`<ellipse rx="13" ry="10" fill="#F08A2C"/><path d="M-5 -10 C-6 0 -6 6 -5 10 M5 -10 C6 0 6 6 5 10" stroke="#D9731C" stroke-width="1.2" fill="none"/>
        <path d="M0 -10 C0 -13 2 -15 4 -15" stroke="#5E8A3A" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <path class="sb-glow" d="M-7 -3 L-4 -6 L-1 -3 Z M1 -3 L4 -6 L7 -3 Z M-7 3 Q0 8 7 3 Q0 5 -7 3 Z" fill="#FFE27A"/>${hands(15, 3)}`)},
    autumn: {props: falling(['#E0752F', '#C9452B', '#E8A642'], leaf),
      item: held(`<ellipse cx="0" cy="6" rx="16" ry="3.4" fill="#F7EBDC"/><path d="M-13 3 L12 -5 L12 3 Z" fill="#E08A3A"/>
        <path d="M12 -5 L14.5 -5 L14.5 3 L12 3 Z" fill="#C9874A"/><ellipse cx="7" cy="-5.5" rx="3.4" ry="2.2" fill="#FFFFFF"/>${hands(17, 5)}`)},
    christmas: {sky: ['#1F2E4C', '#45588A'], hat: HATS.santa, props: falling(['#FFFFFF', '#EAF2FF'], dot),
      item: held(`<rect x="-11" y="-8" width="22" height="16" rx="2" fill="#D9473C"/><rect x="-2" y="-8" width="4" height="16" fill="#F2C14E"/>
        <rect x="-11" y="-2" width="22" height="4" fill="#F2C14E"/><path d="M0 -8 C-6 -15 -10 -9 0 -8 C10 -9 6 -15 0 -8 Z" fill="#F2C14E"/>${hands(14, 3)}`)}
  };
  function art(period, id, look = '') {
    const spec = LOOKS[look] || null;
    const night = period === 'night';
    const sky = night ? SKY.night : Array.isArray(spec?.sky) ? spec.sky : SKY[spec?.sky || period];
    const eyes = night ? 'sleepy' : period === 'evening' && !spec ? 'down' : 'open';
    const festive = night && spec?.festiveNight;   // New Year's Eve stays up in red
    const body = night && !festive ? blanket : hoodie(...(spec?.hood || []));
    const face = spec?.face && !night ? spec.face : '';
    const props = (festive || (spec ? spec.props || PROPS[period] : PROPS[period])).replace(/url\(#ID-glow\)/g, `url(#${id}-glow)`);
    return `<svg class="sb-svg sb-${period}${spec ? ` sb-look-${look}` : ''}" viewBox="0 0 120 120" aria-hidden="true" focusable="false">
      <defs><linearGradient id="${id}-sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${sky[0]}"/><stop offset="1" stop-color="${sky[1]}"/></linearGradient>
        <radialGradient id="${id}-glow"><stop offset="0" stop-color="#FFD9A0" stop-opacity=".55"/><stop offset="1" stop-color="#FFD9A0" stop-opacity="0"/></radialGradient>
        <clipPath id="${id}-clip"><circle cx="60" cy="60" r="58"/></clipPath></defs>
      <g clip-path="url(#${id}-clip)">
        <rect width="120" height="120" fill="url(#${id}-sky)"/>
        ${props}
        <g class="sb-person">${body}${head(eyes, night && !spec?.hat && !festive)}${face}${spec?.hat || ''}${spec?.item || ITEMS[period]}</g>
        ${night || (period === 'noon' && !spec) ? Z : ''}
      </g>
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
  function openNotes() {
    const hub = scope.ReaderHub, progress = scope.PreparationProgress;
    if (!hub || !progress || typeof progress.getState !== 'function') return null;
    const records = progress.getState()?.workspace?.records || {};
    if (hub.settings(records).history === false) return null;
    return hub.summarize(records, new Date()).today.notes.length;
  }
  function render() {
    const now = new Date();
    current = moment(now, lang);
    const look = current.holiday ? current.holiday.look : '';
    if (`${current.period}:${look}` !== scene) {
      scene = `${current.period}:${look}`;
      cardArt.innerHTML = art(current.period, 'sb-card-art', look);
      avatar.innerHTML = art(current.period, 'sb-avatar-art', look);
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
  if (!root.classList.contains('is-open') && stored.seen !== seen) {
    scope.setTimeout(() => {
      if (root.classList.contains('is-open')) return;
      bubble.textContent = current.text;
      bubble.hidden = false;
      save({seen});
      scope.setTimeout(() => { bubble.hidden = true; }, 9000);
    }, 2500);
  }
  scope.setInterval(render, 30000);
  scope.addEventListener('notes:prep-storage', render);
  scope.addEventListener('notes:prep-restored', render);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') render(); });
  scope.StudyBuddy = {setLive(value) { liveData = value; render(); }};
})(typeof window === 'undefined' ? globalThis : window);
