(() => {
  "use strict";
  const baseScores = [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.1, 0.9]];
  function logProbabilities(values) {
    const maximum = Math.max(...values);
    const shifted = values.map(value => value - maximum);
    const normalizer = Math.log(shifted.reduce((total, value) => total + Math.exp(value), 0));
    return shifted.map(value => value - normalizer);
  }
  function calculate(scores, temperature) {
    if (!Number.isFinite(temperature) || temperature <= 0 || !scores.length ||
        scores.some(row => row.length !== scores.length || row.some(value => !Number.isFinite(value)))) {
      throw new Error("Expected a finite square matrix and positive temperature");
    }
    const rows = scores.map(row => logProbabilities(row.map(value => value / temperature)));
    const columns = scores.map((_, column) => logProbabilities(scores.map(row => row[column] / temperature)));
    const meanLoss = values => -values.reduce((total, row, index) => total + row[index], 0) / values.length;
    return {rows, columns, loss: (meanLoss(rows) + meanLoss(columns)) / 2};
  }
  function scenarioScores(scenario) {
    const scores = baseScores.map(row => [...row]);
    if (scenario === "wrong") [scores[0][0], scores[0][1]] = [scores[0][1], scores[0][0]];
    if (scenario === "duplicate") scores.forEach(row => { row[1] = row[0]; });
    return scores;
  }
  if (typeof module !== "undefined") module.exports = {calculate, scenarioScores};
  if (typeof document === "undefined") return;

  document.querySelectorAll("[data-clip-lab]").forEach((root, rootIndex) => {
    let language = root.dataset.language || "zh";
    const state = {direction: "rows", sample: 0, scenario: "paired", temperature: 0.5};
    const copy = {
      zh: {
        title: "让 3 张图找自己的描述", subtitle: "虚构分数 · 只改变候选与温度，不会训练模型",
        direction: "检索方向", rows: "图搜文 · 沿行", columns: "文搜图 · 沿列", sample: "看哪一个查询",
        scenario: "换个情况", paired: "正确配对更高", wrong: "猫图误选自行车", duplicate: "两句相同的猫描述",
        temperature: "温度 τ", items: ["猫", "自行车", "咖啡"], duplicateName: "猫（重复）",
        caption: "行是图片，列是描述。每格显示相似度和当前方向的概率；边框圈出数据配对。",
        corner: "图 ↓ / 文 →", probability: "所选查询的配对概率", loss: "所选查询的 loss", total: "整批双向 loss",
        pairedHint: "试着降低温度：固定分数的排序没变，但概率更集中。",
        wrongHint: "猫图的最高分是错配。降低温度只会更确信地选错；换到猫图观察 loss。",
        duplicateHint: "两句描述完全相同，却只把对角线当正例。按行看猫图，两句各得一份概率；这是目标冲突，不是温度没调好。"
      },
      en: {
        title: "Match 3 images with their captions", subtitle: "Synthetic scores · Change candidates and temperature, not model weights",
        direction: "Retrieval direction", rows: "Image → text · rows", columns: "Text → image · columns", sample: "Query to inspect",
        scenario: "Scenario", paired: "Paired scores are higher", wrong: "Cat image favors bicycle", duplicate: "Two identical cat captions",
        temperature: "Temperature τ", items: ["Cat", "Bicycle", "Coffee"], duplicateName: "Cat (copy)",
        caption: "Images are rows; captions are columns. Cells show similarity and directional probability. Outlines mark dataset pairs.",
        corner: "Image ↓ / text →", probability: "Selected query: paired probability", loss: "Selected query: loss", total: "Batch symmetric loss",
        pairedHint: "Lower temperature: fixed scores keep their ranking, but probability becomes more concentrated.",
        wrongHint: "The cat image favors the wrong caption. Lower temperature becomes more confidently wrong; inspect the cat query's loss.",
        duplicateHint: "Identical captions have different diagonal targets. In the cat image row they split probability: conflicting supervision, not a temperature fix."
      }
    };
    function build() {
      const words = copy[language];
      const identity = `clip-${rootIndex}`;
      root.lang = language;
      root.innerHTML = `<div class="clip-heading"><strong>${words.title}</strong><button type="button" data-language-switch>${language === "zh" ? "EN" : "中文"}</button></div>
<p class="clip-subtitle">${words.subtitle}</p>
<div class="clip-controls">
<label>${words.direction}<select data-control="direction"><option value="rows">${words.rows}</option><option value="columns">${words.columns}</option></select></label>
<label>${words.sample}<select data-control="sample">${words.items.map((item, index) => `<option value="${index}">${item}</option>`).join("")}</select></label>
<label>${words.scenario}<select data-control="scenario"><option value="paired">${words.paired}</option><option value="wrong">${words.wrong}</option><option value="duplicate">${words.duplicate}</option></select></label>
<label><span id="${identity}-temperature-label">${words.temperature}</span><output for="${identity}-temperature" data-temperature aria-hidden="true"></output><input id="${identity}-temperature" aria-labelledby="${identity}-temperature-label" data-control="temperature" type="range" min="0.05" max="1.5" step="0.05"></label>
</div><div class="clip-matrix-wrap"><table class="clip-matrix"><caption>${words.caption}</caption><thead></thead><tbody></tbody></table></div>
<div class="clip-metrics" aria-live="polite" aria-atomic="true"></div><p class="clip-hint"></p>`;
      root.querySelectorAll("[data-control]").forEach(control => {
        const key = control.dataset.control;
        control.value = state[key];
        control.addEventListener(control.type === "range" ? "input" : "change", () => {
          state[key] = key === "sample" || key === "temperature" ? Number(control.value) : control.value;
          update();
        });
      });
      root.querySelector("[data-language-switch]").addEventListener("click", () => {
        language = language === "zh" ? "en" : "zh";
        build();
        root.querySelector("[data-language-switch]").focus();
      });
      update();
    }
    function update() {
      const words = copy[language];
      const scores = scenarioScores(state.scenario);
      const result = calculate(scores, state.temperature);
      const captions = [...words.items];
      if (state.scenario === "duplicate") captions[1] = words.duplicateName;
      root.querySelectorAll('[data-control="sample"] option').forEach((option, index) => {
        option.textContent = (state.direction === "columns" ? captions : words.items)[index];
      });
      root.querySelector("[data-temperature]").textContent = state.temperature.toFixed(2);
      root.querySelector("thead").innerHTML = `<tr><th scope="col">${words.corner}</th>${captions.map(label => `<th scope="col">${label}</th>`).join("")}</tr>`;
      root.querySelector("tbody").innerHTML = scores.map((row, rowIndex) => `<tr><th scope="row">${words.items[rowIndex]}</th>${row.map((value, columnIndex) => {
        const logProbability = state.direction === "rows" ? result.rows[rowIndex][columnIndex] : result.columns[columnIndex][rowIndex];
        const selected = (state.direction === "rows" ? rowIndex : columnIndex) === state.sample;
        return `<td class="${rowIndex === columnIndex ? "is-pair " : ""}${selected ? "is-selected" : ""}"><b>${value.toFixed(1)}</b><small>${(Math.exp(logProbability) * 100).toFixed(1)}%</small></td>`;
      }).join("")}</tr>`).join("");
      const pairedLogProbability = result[state.direction][state.sample][state.sample];
      root.querySelector(".clip-metrics").innerHTML = [
        [words.probability, `${(Math.exp(pairedLogProbability) * 100).toFixed(1)}%`],
        [words.loss, (-pairedLogProbability).toFixed(3)],
        [words.total, result.loss.toFixed(3)]
      ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
      root.querySelector(".clip-hint").textContent = words[`${state.scenario}Hint`];
    }
    build();
  });
})();
