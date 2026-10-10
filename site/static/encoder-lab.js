(() => {
  "use strict";

  function mlmExample(replacement = "mask", padding = 1, probability = 0.25) {
    if (!["mask", "random", "unchanged"].includes(replacement) ||
        !Number.isInteger(padding) || padding < 0 || padding > 32 ||
        !Number.isFinite(probability) || probability <= 0 || probability > 1) {
      throw new RangeError("Invalid toy MLM input");
    }
    const input = ["[CLS]", "the", "tea", "is", {mask: "[MASK]", random: "hot", unchanged: "cold"}[replacement], "[SEP]", ...Array(padding).fill("[PAD]")];
    const labels = [-100, -100, 5, -100, 7, -100, ...Array(padding).fill(-100)];
    const terms = [-Math.log(0.8), -Math.log(probability)];
    const total = terms.reduce((sum, value) => sum + value, 0);
    return {input, labels, terms, loss: total / 2, wrongLoss: total / input.length,
      attention: input.map(token => token === "[PAD]" ? 0 : 1)};
  }

  function attentionExample(mode = "encoder", selected = 1) {
    if (!["encoder", "decoder", "cross"].includes(mode) || !Number.isInteger(selected) || selected < 0 || selected > 2) {
      throw new RangeError("Invalid toy attention input");
    }
    const source = ["I", "like", "tea"];
    const target = ["[BOS]", "ich", "mag"];
    const queries = mode === "encoder" ? source : target;
    const keys = [...(mode === "decoder" ? target : source), "[PAD]"];
    const matrix = queries.map((_, query) => keys.map((key, column) => key !== "[PAD]" && (mode !== "decoder" || column <= query)));
    return {queries, keys, matrix, visible: keys.filter((_, column) => matrix[selected][column])};
  }

  if (typeof module !== "undefined") module.exports = {mlmExample, attentionExample};
  if (typeof document === "undefined") return;

  const copy = {
    en: {
      toy: "A small, hand-built example", reset: "Reset", mlmTitle: "Changed input. Same target.",
      mlmIntro: "We chose tea and cold for supervision. Tea stays unchanged. Try changing only what the model sees at cold.",
      replacement: "Input at cold", mask: "Use [MASK]", random: "Use hot (random branch)", unchanged: "Keep cold",
      original: "Original", input: "Input", target: "Target", ignored: "Not scored", scored: "Scored",
      special: "[CLS] and [SEP] are not shown. Padding tokens:", pad: "Padding tokens", probability: "Illustrative p(cold at its position)",
      scope: "No model runs here. Probabilities are hand-set, not predictions. Changing the input keeps them fixed to isolate the label rule; a real model's probabilities may change.",
      terms: "Loss from each selected position", correct: "Mean over 2 targets", incorrect: "Wrong: divide by all positions", fixed: "p(tea) stays at 0.80.",
      hint: "Try keeping cold unchanged: both targets still count. Then add padding: the correct mean stays the same.",
      maskTitle: "Which positions can this query read?", maskIntro: "Translate I like tea → ich mag Tee. The decoder inputs are shifted right: [BOS], ich, mag.",
      encoder: "Encoder self-attention", decoder: "Decoder self-attention", cross: "Cross-attention", mode: "Attention site",
      query: "Query ↓ / Key →", legend: "1 = allowed · 0 = blocked. Select a query row.",
      visibility: "Visibility only—not learned attention weights. Padding query rows are omitted; padding keys are blocked in every mode.",
      encoderHint: "The encoder can read the complete source, including words to the right.",
      decoderHint: "At input {input}, the next-token target is {target}. Reading the current input is allowed; later positions in the shifted input stay blocked.",
      crossHint: "Decoder queries read encoder states. The source is already supplied, so all its valid positions are available—even for the first query.",
      sees: "can read", lossSummary: "Selected-target mean", queryAction: "Inspect query"
    },
    zh: {
      toy: "手工小例子，不运行模型", reset: "重置", mlmTitle: "输入换了，答案没换",
      mlmIntro: "这次选 tea 和 cold 来算 loss。Tea 保持原样；试着只改 cold 位置的输入，看看目标会不会跟着变。",
      replacement: "cold 位置的输入", mask: "换成 [MASK]", random: "换成 hot（随机分支）", unchanged: "保留 cold",
      original: "原文", input: "输入", target: "目标", ignored: "不直接评分", scored: "参与评分",
      special: "图中省略 [CLS] 和 [SEP]。Padding 数量：", pad: "Padding 数量", probability: "手工设定 p(cold｜该位置)",
      scope: "这里只演示字段和算式，不运行模型。切换输入时概率保持不变，是为了单独看标签规则；真实模型的预测概率可能随输入改变。",
      terms: "两个目标分别贡献多少 loss", correct: "按 2 个目标取均值", incorrect: "错误示范：除以全部位置数", fixed: "p(tea) 固定为 0.80。",
      hint: "先选“保留 cold”：两个目标仍然要评分。再增加 padding：正确的平均 loss 不会因此变小。",
      maskTitle: "这个 query 到底能读哪些位置？", maskIntro: "用 I like tea → ich mag Tee 看三处 attention。Decoder 输入右移一位：[BOS]、ich、mag。",
      encoder: "Encoder 自注意力", decoder: "Decoder 自注意力", cross: "Cross-attention", mode: "Attention 类型",
      query: "Query ↓ / Key →", legend: "1 = 可读取 · 0 = 被屏蔽。点行首，换一个 query 看。",
      visibility: "画的是可见范围，不是学到的 attention 权重。省略 padding query 行；所有模式都会屏蔽 padding key。",
      encoderHint: "Encoder 读的是已给出的整句原文，所以右边的词也能看到。",
      decoderHint: "输入为 {input} 时，下一个目标是 {target}。可以看当前输入；右移后的输入序列中，更靠后的位置仍被屏蔽。",
      crossHint: "Query 来自 decoder，key 来自 encoder。原文已经给出，所以即使是第一个 query，也能读到所有有效 source 位置。",
      sees: "可以读取", lossSummary: "选中目标的平均 loss", queryAction: "查看 query"
    }
  };

  function mountMlm(root, text) {
    root.innerHTML = `<header><span class="encoder-kicker">${text.toy}</span><strong class="encoder-title">${text.mlmTitle}</strong><p>${text.mlmIntro}</p></header>
      <div class="encoder-controls"><label>${text.replacement}<select data-replacement><option value="mask">${text.mask}</option><option value="random">${text.random}</option><option value="unchanged">${text.unchanged}</option></select></label><button type="button" data-reset>${text.reset}</button></div>
      <div class="encoder-tokens" data-tokens></div>
      <div class="encoder-controls encoder-padding"><label>${text.special}<select data-padding aria-label="${text.pad}"><option value="1">1</option><option value="5">5</option><option value="9">9</option></select></label></div>
      <div class="encoder-probability"><label>${text.probability} <output data-probability-value>0.25</output><input type="range" data-probability min="0.01" max="0.99" step="0.01" value="0.25"></label></div>
      <p class="encoder-small">${text.fixed}</p><div class="encoder-terms" data-terms aria-label="${text.terms}"></div>
      <div class="encoder-results"><div><span>${text.correct}</span><strong data-loss></strong><small data-calculation></small></div><div class="encoder-wrong"><span>${text.incorrect}</span><strong data-wrong-loss></strong><small data-wrong-calculation></small></div></div>
      <p class="encoder-hint">${text.hint}</p><p class="encoder-small">${text.scope}</p><span class="encoder-sr" data-status role="status" aria-live="polite"></span>`;
    const replacement = root.querySelector("[data-replacement]");
    const padding = root.querySelector("[data-padding]");
    const probability = root.querySelector("[data-probability]");
    const render = (announce = false) => {
      const example = mlmExample(replacement.value, Number(padding.value), Number(probability.value));
      root.querySelector("[data-tokens]").innerHTML = ["the", "tea", "is", "cold"].map((token, index) => {
        const supervised = index === 1 || index === 3;
        const inputToken = example.input[index + 1].replaceAll("<", "&lt;");
        return `<div class="encoder-token ${supervised ? "is-target" : ""}"><span>${text.original} <b>${token}</b></span><small>${text.input}</small><strong>${inputToken}</strong><span class="encoder-target">${text.target}: ${supervised ? token : "—"}</span><small>${supervised ? text.scored : text.ignored}</small></div>`;
      }).join("");
      root.querySelector("[data-probability-value]").value = Number(probability.value).toFixed(2);
      root.querySelector("[data-terms]").innerHTML = example.terms.map((term, index) => `<div><span>${index === 0 ? "tea" : "cold"}</span><span class="encoder-bar" aria-hidden="true"><i style="width:${100 * term / -Math.log(0.01)}%"></i></span><output>${term.toFixed(3)}</output></div>`).join("");
      root.querySelector("[data-loss]").textContent = example.loss.toFixed(3);
      root.querySelector("[data-wrong-loss]").textContent = example.wrongLoss.toFixed(3);
      const sum = example.terms.reduce((total, term) => total + term, 0).toFixed(3);
      root.querySelector("[data-calculation]").textContent = `${sum} ÷ 2`;
      root.querySelector("[data-wrong-calculation]").textContent = `${sum} ÷ ${example.input.length}`;
      if (announce) root.querySelector("[data-status]").textContent = `${text.lossSummary}: ${example.loss.toFixed(3)}. ${text.incorrect}: ${example.wrongLoss.toFixed(3)}.`;
    };
    [replacement, padding].forEach(control => control.addEventListener("change", () => render(true)));
    probability.addEventListener("input", () => render());
    probability.addEventListener("change", () => render(true));
    root.querySelector("[data-reset]").addEventListener("click", () => {
      replacement.value = "mask";
      padding.value = "1";
      probability.value = "0.25";
      render(true);
    });
    render();
  }

  function mountAttention(root, text) {
    let mode = "encoder";
    let selected = 1;
    root.innerHTML = `<header><span class="encoder-kicker">${text.toy}</span><strong class="encoder-title">${text.maskTitle}</strong><p>${text.maskIntro}</p></header>
      <div class="encoder-modes" role="group" aria-label="${text.mode}">${["encoder", "decoder", "cross"].map(value => `<button type="button" data-mode="${value}" aria-pressed="${value === mode}">${text[value]}</button>`).join("")}</div>
      <div class="encoder-matrix-wrap"><table class="encoder-matrix"><caption>${text.legend}</caption><thead data-keys></thead><tbody data-queries></tbody></table></div>
      <div class="encoder-readout" role="status" aria-live="polite" data-readable></div><p class="encoder-hint" data-explanation></p><p class="encoder-small">${text.visibility}</p>`;
    const render = () => {
      const example = attentionExample(mode, selected);
      root.querySelectorAll("[data-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
      root.querySelector("[data-keys]").innerHTML = `<tr><th scope="col">${text.query}</th>${example.keys.map(key => `<th scope="col">${key}</th>`).join("")}</tr>`;
      const activeRow = document.activeElement?.closest("[data-query]");
      const restoreFocus = activeRow && root.contains(activeRow) ? activeRow.dataset.query : null;
      root.querySelector("[data-queries]").innerHTML = example.matrix.map((row, query) => `<tr class="${query === selected ? "is-query" : ""}"><th scope="row"><button type="button" data-query="${query}" aria-label="${text.queryAction} ${example.queries[query]}" aria-pressed="${query === selected}">${example.queries[query]}</button></th>${row.map(allowed => `<td class="${allowed ? "is-readable" : "is-blocked"}">${Number(allowed)}</td>`).join("")}</tr>`).join("");
      root.querySelector("[data-readable]").textContent = `${example.queries[selected]} → ${text.sees}: ${example.visible.join(" · ")}`;
      root.querySelector("[data-explanation]").textContent = text[`${mode}Hint`]
        .replace("{input}", example.queries[selected]).replace("{target}", ["ich", "mag", "Tee"][selected]);
      if (restoreFocus !== null) root.querySelector(`[data-query="${restoreFocus}"]`).focus({preventScroll: true});
    };
    root.addEventListener("click", event => {
      const modeButton = event.target.closest("[data-mode]");
      const rowButton = event.target.closest("[data-query]");
      if (modeButton) mode = modeButton.dataset.mode;
      else if (rowButton) selected = Number(rowButton.dataset.query);
      else return;
      render();
    });
    render();
  }

  document.querySelectorAll("[data-encoder-lab]").forEach(root => {
    const language = root.dataset.lang || (document.documentElement.lang.startsWith("zh") ? "zh" : "en");
    const text = copy[language] || copy.en;
    if (root.dataset.encoderLab === "mlm") mountMlm(root, text);
    if (root.dataset.encoderLab === "attention") mountAttention(root, text);
  });
})();
