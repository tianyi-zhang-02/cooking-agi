(function () {
  'use strict';

  const candidates = [
    {id: 'A', topic: 'ML', author: 'Lin', score: .95, vector: [1, 0], zh: '注意力图解', en: 'Attention illustrated'},
    {id: 'B', topic: 'ML', author: 'Lin', score: .93, vector: [.99, .01], zh: '再看一次注意力', en: 'Another look at attention'},
    {id: 'C', topic: 'ML', author: 'Mo', score: .90, vector: [.98, .02], zh: '注意力入门', en: 'Getting started with attention'},
    {id: 'D', topic: 'Photo', author: 'Kai', score: .80, vector: [.7, .7], zh: '摄影里的构图', en: 'Composition in photography'},
    {id: 'E', topic: 'Music', author: 'Yu', score: .77, vector: [0, 1], zh: '听懂一段旋律', en: 'Understanding a melody'},
    {id: 'F', topic: 'Travel', author: 'An', score: .72, vector: [-.3, .9], zh: '海边散步', en: 'A walk by the coast'}
  ];

  function cosine(left, right) {
    if (!left.length || left.length !== right.length ||
        [...left, ...right].some(value => !Number.isFinite(value))) {
      throw new Error('Vectors must be finite and have equal, nonzero dimensions');
    }
    const leftScale = Math.max(...left.map(Math.abs));
    const rightScale = Math.max(...right.map(Math.abs));
    if (!leftScale || !rightScale) throw new Error('Zero vectors have no cosine');
    const scaledLeft = left.map(value => value / leftScale);
    const scaledRight = right.map(value => value / rightScale);
    const denominator = Math.hypot(...scaledLeft) * Math.hypot(...scaledRight);
    const dot = scaledLeft.reduce((total, value, index) => total + value * scaledRight[index], 0);
    return Math.max(-1, Math.min(1, dot / denominator));
  }

  function rerank(pool, penalty, uniqueAuthors, limit = 3) {
    if (!Number.isFinite(penalty) || penalty < 0 || !Number.isInteger(limit) || limit < 0) {
      throw new Error('Invalid ranking controls');
    }
    const chosen = [];
    const remaining = [...pool];
    while (remaining.length && chosen.length < limit) {
      const eligible = remaining
        .filter(candidate => !uniqueAuthors || !chosen.some(item => item.author === candidate.author))
        .map(candidate => {
          const similarity = Math.max(0, ...chosen.map(item => cosine(candidate.vector, item.vector)));
          return {...candidate, gain: candidate.score - penalty * similarity};
        })
        .sort((left, right) => right.gain - left.gain || left.id.localeCompare(right.id));
      if (!eligible.length) break;
      const next = eligible[0];
      chosen.push(next);
      remaining.splice(remaining.findIndex(item => item.id === next.id), 1);
    }
    return chosen;
  }

  function summarize(items) {
    let distance = 0;
    let pairs = 0;
    items.forEach((left, index) => items.slice(index + 1).forEach(right => {
      distance += 1 - cosine(left.vector, right.vector);
      pairs += 1;
    }));
    return {
      mean: items.length ? items.reduce((total, item) => total + item.score, 0) / items.length : null,
      topics: new Set(items.map(item => item.topic)).size,
      ild: pairs ? distance / pairs : null
    };
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {candidates, cosine, rerank, summarize};
  }
  if (typeof document === 'undefined') return;

  document.querySelectorAll('[data-recsys-lab]').forEach((root, instance) => {
    const english = document.documentElement.lang.startsWith('en');
    const copy = (zh, en) => english ? en : zh;
    const titleId = 'recsys-title-' + instance;
    root.classList.add('recsys-lab');
    root.setAttribute('role', 'group');
    root.setAttribute('aria-labelledby', titleId);
    root.innerHTML = `<div class="recsys-heading"><strong id="${titleId}">${copy('同一批候选，换一种选法', 'Same candidates, different choices')}</strong><button type="button" data-reset>${copy('重置', 'Reset')}</button></div>
      <p class="recsys-caption">${copy('6 条虚构内容，选出 3 条。模型和基础分数始终不变。', 'Choose 3 of 6 fictional posts. The model and base scores never change.')}</p>
      <label class="recsys-slider"><span>${copy('相似度惩罚 λ', 'Similarity penalty λ')} <output data-penalty>0.00</output></span><input type="range" min="0" max="1" step="0.05" value="0" aria-label="${copy('相似度惩罚', 'Similarity penalty')}"></label>
      <label class="recsys-check"><input type="checkbox"> ${copy('同一作者最多 1 条', 'At most 1 post per author')}</label>
      <div class="recsys-columns">
        <section><h3>${copy('候选池 · 按基础分排序', 'Pool · ordered by base score')}</h3><div data-pool></div></section>
        <section><h3>${copy('最终列表 · 按选出顺序', 'Final list · selection order')}</h3><div data-result></div></section>
      </div>
      <p class="recsys-stats" role="status" aria-live="polite" aria-atomic="true"></p>
      <p class="recsys-caption">${copy('选择分 = 基础分 − λ × 与已选内容的最大非负余弦相似度。第一条不扣分。ILD 是列表内两两余弦距离的平均值，不是真实用户满意度。', 'Selection score = base score − λ × the greatest nonnegative cosine similarity to a selected post. The first post has no penalty. ILD averages pairwise cosine distances; it is not user satisfaction.')}</p>`;
    const slider = root.querySelector('input[type="range"]');
    const checkbox = root.querySelector('input[type="checkbox"]');
    const label = candidate => english ? candidate.en : candidate.zh;

    function draw() {
      const penalty = Number(slider.value);
      const selected = rerank(candidates, penalty, checkbox.checked);
      const stats = summarize(selected);
      root.querySelector('[data-penalty]').value = penalty.toFixed(2);
      root.querySelector('[data-pool]').innerHTML = candidates.map(candidate => {
        const picked = selected.some(item => item.id === candidate.id);
        return `<div class="recsys-item${picked ? ' picked' : ''}"><b>${candidate.id}</b><span>${label(candidate)}<small>${candidate.topic} · ${candidate.author}</small></span><span>${candidate.score.toFixed(2)}<small>${picked ? copy('已选', 'Selected') : '—'}</small></span></div>`;
      }).join('');
      root.querySelector('[data-result]').innerHTML = selected.map((candidate, index) =>
        `<div class="recsys-item picked"><b>${index + 1}</b><span>${candidate.id} · ${label(candidate)}<small>${candidate.topic} · ${candidate.author}</small></span><span>${candidate.gain.toFixed(2)}<small>${copy('选择分', 'Selection')}</small></span></div>`
      ).join('');
      root.querySelector('.recsys-stats').textContent = copy(
        `平均基础分 ${stats.mean.toFixed(3)} · 主题 ${stats.topics} / 3 · ILD ${stats.ild.toFixed(3)}`,
        `Mean base score ${stats.mean.toFixed(3)} · Topics ${stats.topics} / 3 · ILD ${stats.ild.toFixed(3)}`
      );
    }

    slider.addEventListener('input', draw);
    checkbox.addEventListener('change', draw);
    root.querySelector('[data-reset]').addEventListener('click', () => {
      slider.value = '0';
      checkbox.checked = false;
      draw();
    });
    draw();
  });
})();
