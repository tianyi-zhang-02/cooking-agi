(() => {
  "use strict";

  function storyState(step = 0, task = "mlm", query = 2) {
    if (!Number.isInteger(step) || step < 0 || step > 3 ||
        !["mlm", "classify", "tag"].includes(task) ||
        !Number.isInteger(query) || query < 0 || query > 5) {
      throw new RangeError("Invalid BERT walkthrough state");
    }
    const inputs = ["[CLS]", "the", task === "mlm" ? "[MASK]" : "tea", "is", "cold", "[SEP]"];
    const outputs = {mlm: [2], classify: [0], tag: [1, 2, 3, 4]}[task];
    return {step, task, query, inputs, outputs, positions: [0, 1, 2, 3, 4, 5],
      segments: [0, 0, 0, 0, 0, 0], visible: [0, 1, 2, 3, 4, 5]};
  }

  if (typeof module !== "undefined") module.exports = {storyState};
  if (typeof document === "undefined") return;

  const words = {
    en: {
      title: "Follow a sentence through BERT", kicker: "A WORKED EXAMPLE",
      steps: ["Input", "Vectors", "Context", "Task"], navigation: "Walkthrough steps",
      input: "One sentence · six input positions", position: "Position", inspect: "Trace context for",
      embedding: "Token + position + segment", embeddingNote: "Add vectors, then LayerNorm + dropout",
      encoder: "BERT encoder", layers: "12 layers · Base", block: "Attention → Add & Norm → FFN → Add & Norm",
      hidden: "A contextual vector at every position", head: "Task head", headMlm: "MLM head", headClassify: "Classification head", headTag: "Token-label head",
      outputMlm: "Score the vocabulary at h₂ · target: tea", outputClassify: "Pool h₀ ([CLS]) → class scores", outputTag: "Read h₁ … h₄ → a label per word",
      task: "Use the encoder for", mlm: "Restore a token", classify: "Classify text", tag: "Label words",
      stepTitles: ["Hide tea, but keep it as the answer", "Add three embeddings at each position", "Bring in the surrounding words", "Choose what to predict"],
      inputText: "The original word is tea. We replace its input with [MASK], but keep tea as the training target. The words on both sides remain available. This example follows just one selected position.",
      inputTaskText: "For this downstream task, the complete sentence is supplied. [CLS] and [SEP] are special tokens; their positions still receive vectors.",
      vectorText: "At position {position}, we add three vectors: which token is here, where it sits, and which segment it belongs to. Each has 768 values in BERT-Base, and their sum still has 768.",
      contextText: "The state at {token} can receive information from all six input positions. Click another token to trace that query. The paths show allowed connections, not measured attention weights.",
      mlmText: "The MLM head turns the selected state's vector into vocabulary scores. Cross-entropy compares those scores with tea. Other positions supply context even though their outputs are not scored in this example.",
      classifyText: "Now supply the complete sentence and train a classifier with labels such as positive and negative. It reads [CLS] through a pooling layer. We can train just the classifier or update the encoder too.",
      tagText: "For token labeling, read the states at the word positions. This toy skips special tokens; real data also need word/subword alignment. Each task has its own fine-tuning and evaluation.",
      play: "Play walkthrough", pause: "Pause", previous: "Previous", next: "Next", progress: "Step",
      scope: "Schematic, not model inference. Bars stand for vectors. No learned weights or predictions are displayed. NSP is covered below.",
      trace: "Selected query", segment: "Segment A", tokenVector: "Token vector", positionVector: "Position vector", segmentVector: "Segment vector",
      localLink: "Continue with inputs and shapes →", seen: "can read both sides"
    },
    zh: {
      title: "跟着一句话，走一遍 BERT", kicker: "用一个例子看结构",
      steps: ["准备输入", "变成向量", "读上下文", "做预测"], navigation: "图解步骤",
      input: "一句话 · 6 个输入位置", position: "位置", inspect: "查看上下文读取路径",
      embedding: "Token + 位置 + 文本段", embeddingNote: "三个向量相加，再做 LayerNorm 与 dropout",
      encoder: "BERT 编码器", layers: "Base · 12 层", block: "注意力 → 残差与归一化 → FFN → 残差与归一化",
      hidden: "每个位置都有一个带上下文的向量", head: "任务输出层", headMlm: "MLM 输出层", headClassify: "分类输出层", headTag: "逐词标注输出层",
      outputMlm: "用 h₂ 预测原词 · 训练答案是 tea", outputClassify: "读取 h₀（[CLS]）→ 类别分数", outputTag: "读取 h₁ … h₄ → 给每个词打标签",
      task: "用这些表示做什么", mlm: "恢复原词", classify: "文本分类", tag: "逐词标注",
      stepTitles: ["把 tea 藏起来，答案留着", "把三种信息加到一个向量里", "结合周围的词，重新计算表示", "接下来，你想让它做什么？"],
      inputText: "原句里的 tea 被换成了 [MASK]，但训练答案还是 tea。左边的 the、右边的 is cold 都可以提供线索。先跟着这一个位置看，后面再讲怎么选预测位置。",
      inputTaskText: "做这个下游任务时，输入是完整句子。[CLS] 和 [SEP] 是特殊 token；它们的位置也会得到各自的向量。",
      vectorText: "位置 {position} 要用到三个向量：这里是什么 token、排在第几个位置、属于哪段文本。把它们逐维相加。BERT-Base 中每个都是 768 维，加完也还是 768 维。",
      contextText: "计算 {token} 的表示时，左右两边的信息都能用到。点另一个 token，可以看它能从哪里读取信息。这里画的是允许读取的连线，不是真实模型的注意力权重。",
      mlmText: "选中位置的向量交给 MLM 输出层，得到词表中各个 token 的分数，再和原词 tea 算交叉熵。其他位置虽然没有单独算分，仍会影响这次预测。",
      classifyText: "换成评论分类，输入就是完整句子，训练标签是好评或差评。分类层读取经过 pooling 处理的 [CLS] 表示。你可以只训练分类层，也可以连编码器一起微调。",
      tagText: "做逐词标注时，每个词的表示都要交给输出层。若一个词被拆成多个 subword，还得先确定标签怎样对齐。图中只展示普通词的位置，不给特殊 token 打标签。",
      play: "播放一遍", pause: "暂停", previous: "上一步", next: "下一步", progress: "步骤",
      scope: "结构示意，不运行模型。短条代表向量，不展示真实权重或预测。NSP 在后面单独讲。",
      trace: "正在看", segment: "文本段 A", tokenVector: "Token 向量", positionVector: "位置向量", segmentVector: "文本段向量",
      localLink: "继续看输入字段和形状 →", seen: "左右两边都能读"
    }
  };

  document.querySelectorAll("[data-bert-story]").forEach((root, instance) => {
    const language = root.dataset.lang === "zh" ? "zh" : "en";
    const text = words[language];
    const panelId = `bert-story-panel-${instance}`;
    let step = 0;
    let task = "mlm";
    let query = 2;
    let timer = null;
    root.innerHTML = `<header class="bert-story-header"><span>${text.kicker}</span><strong>${text.title}</strong></header>
      <div class="bert-steps" role="group" aria-label="${text.navigation}">${text.steps.map((label, index) => `<button type="button" data-step="${index}" aria-controls="${panelId}" aria-pressed="${index === 0}"><small>0${index + 1}</small><span>${label}</span></button>`).join("")}</div>
      <div class="bert-story-body"><div class="bert-scene" data-scene>
        <div class="bert-row-label"><span>${text.input}</span><span data-trace></span></div>
        <div class="bert-inputs">${Array.from({length: 6}, (_, index) => `<button type="button" data-token="${index}" aria-pressed="${index === query}"><span data-token-label></span><small>${index}</small></button>`).join("")}</div>
        <div class="bert-embedding" data-layer="1"><span class="bert-vector-glyph" aria-hidden="true"><i></i><i></i><i></i><b>+</b><i></i><i></i><i></i><b>+</b><i></i><i></i><i></i></span><strong>${text.embedding}</strong><small>${text.embeddingNote}</small></div>
        <div class="bert-connections"><svg viewBox="0 0 600 76" preserveAspectRatio="none" aria-hidden="true" focusable="false" data-connections></svg><span data-connection-caption></span></div>
        <div class="bert-stack" data-layer="2"><div><strong>${text.encoder}</strong><span>${text.layers}</span></div><small>${text.block}</small></div>
        <div class="bert-hidden-label">${text.hidden}</div><div class="bert-hidden" aria-hidden="true">${Array.from({length: 6}, (_, index) => `<span data-hidden="${index}"><i></i><i></i><i></i><small>h<sub>${index}</sub></small></span>`).join("")}</div>
        <div class="bert-head" data-layer="3"><span data-head-name></span><strong data-head-output></strong></div>
      </div>
      <div class="bert-story-aside"><section class="bert-explanation" id="${panelId}" aria-live="polite" aria-atomic="true"><span data-step-number></span><div><strong data-explanation-title></strong><p data-explanation></p><div class="bert-vector-inspector" data-vector-inspector hidden></div></div></section>
      <div class="bert-task-switch" data-tasks hidden role="group" aria-label="${text.task}">${["mlm", "classify", "tag"].map(value => `<button type="button" data-task="${value}" aria-pressed="${value === task}">${text[value]}</button>`).join("")}</div>
      </div></div>
      <footer class="bert-story-controls"><button type="button" data-play>${text.play}</button><div><button type="button" data-previous>${text.previous}</button><button type="button" data-next>${text.next}</button></div></footer>
      <p class="bert-story-scope">${text.scope}</p>`;

    const render = () => {
      const state = storyState(step, task, query);
      root.dataset.step = String(step);
      root.querySelectorAll("[data-step]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.step) === step)));
      root.querySelectorAll("[data-token]").forEach(button => {
        const index = Number(button.dataset.token);
        button.querySelector("[data-token-label]").textContent = state.inputs[index];
        button.setAttribute("aria-label", `${text.inspect} ${state.inputs[index]} (${index})`);
        button.setAttribute("aria-pressed", String(index === query));
        button.classList.toggle("is-masked", task === "mlm" && index === 2);
      });
      root.querySelectorAll("[data-hidden]").forEach(element => element.classList.toggle("is-output", (step === 2 ? [query] : state.outputs).includes(Number(element.dataset.hidden))));
      root.querySelector("[data-trace]").textContent = step === 2 ? `${text.trace}: ${state.inputs[query]}` : "";
      root.querySelector("[data-connections]").innerHTML = state.visible.map(index => {
        const start = 50 + index * 100;
        const end = step === 2 ? 50 + query * 100 : start;
        return `<path d="M ${start} 0 C ${start} 38, ${end} 38, ${end} 76"/><circle cx="${start}" cy="0" r="2.5"/>`;
      }).join("");
      root.querySelector("[data-connection-caption]").textContent = step === 2 ? text.seen : "";
      root.querySelector("[data-head-name]").textContent = text[{mlm: "headMlm", classify: "headClassify", tag: "headTag"}[task]];
      root.querySelector("[data-head-output]").textContent = text[{mlm: "outputMlm", classify: "outputClassify", tag: "outputTag"}[task]];
      root.querySelector("[data-step-number]").textContent = `0${step + 1}`;
      root.querySelector("[data-explanation-title]").textContent = text.stepTitles[step];
      const explanation = [task === "mlm" ? text.inputText : text.inputTaskText, text.vectorText, text.contextText, text[`${task}Text`]][step];
      root.querySelector("[data-explanation]").textContent = explanation.replaceAll("{position}", String(query)).replaceAll("{token}", state.inputs[query]);
      const inspector = root.querySelector("[data-vector-inspector]");
      inspector.hidden = step !== 1;
      inspector.innerHTML = `<span>${state.inputs[query]}<small>${text.tokenVector}</small></span><b>+</b><span>${query}<small>${text.positionVector}</small></span><b>+</b><span>A<small>${text.segmentVector}</small></span>`;
      root.querySelector("[data-tasks]").hidden = step !== 3;
      root.querySelectorAll("[data-task]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.task === task)));
      root.querySelector("[data-previous]").disabled = step === 0;
      root.querySelector("[data-next]").disabled = step === 3;
      root.querySelector("[data-play]").textContent = timer === null ? text.play : text.pause;
    };

    const stop = () => {
      if (timer !== null) clearInterval(timer);
      timer = null;
      root.querySelector("[data-play]").textContent = text.play;
    };
    root.addEventListener("click", event => {
      const control = event.target.closest("button");
      if (!control || !root.contains(control)) return;
      if (control.hasAttribute("data-play")) {
        if (timer !== null) stop();
        else {
          step = 0;
          timer = setInterval(() => {
            step += 1;
            if (step === 3) stop();
            render();
          }, 3200);
        }
      } else {
        stop();
        if (control.hasAttribute("data-step")) step = Number(control.dataset.step);
        if (control.hasAttribute("data-token")) {
          query = Number(control.dataset.token);
          if (step !== 1) step = 2;
        }
        if (control.hasAttribute("data-task")) {
          task = control.dataset.task;
          query = task === "classify" ? 0 : 2;
        }
        if (control.hasAttribute("data-previous")) step = Math.max(0, step - 1);
        if (control.hasAttribute("data-next")) step = Math.min(3, step + 1);
      }
      render();
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
    if (typeof IntersectionObserver !== "undefined") {
      new IntersectionObserver(entries => { if (!entries[0].isIntersecting) stop(); }).observe(root);
    }
    render();
  });
})();
