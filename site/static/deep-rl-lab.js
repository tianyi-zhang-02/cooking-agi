(function () {
  "use strict";

  function bounded(value, lower, upper) {
    if (!Number.isFinite(value) || value < lower || value > upper) {
      throw new RangeError("Value outside the supported range");
    }
    return value;
  }

  function bellman(gamma, steps) {
    bounded(gamma, 0, 1);
    if (!Number.isInteger(steps) || steps < 0) throw new RangeError("Invalid steps");
    let start = 0;
    let charged = 0;
    for (let step = 0; step < steps; step += 1) {
      start = Math.max(2, gamma * charged);
      charged = 4;
    }
    return { start, charged };
  }

  function boundaryTarget(reward, nextValue, gamma, terminated) {
    bounded(gamma, 0, 1);
    if (![reward, nextValue].every(Number.isFinite) || typeof terminated !== "boolean") {
      throw new TypeError("Invalid transition");
    }
    return reward + gamma * (terminated ? 0 : nextValue);
  }

  function gaeTerms(gamma, traceDecay, residuals) {
    bounded(gamma, 0, 1);
    bounded(traceDecay, 0, 1);
    if (!Array.isArray(residuals) || !residuals.length || !residuals.every(Number.isFinite)) {
      throw new TypeError("Expected finite residuals");
    }
    return residuals.map(function (residual, index) {
      return Math.pow(gamma * traceDecay, index) * residual;
    });
  }

  function doubleTargets(online, target, gamma) {
    bounded(gamma, 0, 1);
    if (!Array.isArray(online) || !Array.isArray(target) || !online.length ||
        online.length !== target.length || !online.concat(target).every(Number.isFinite)) {
      throw new TypeError("Q arrays must be finite and aligned");
    }
    const action = online.indexOf(Math.max.apply(null, online));
    return { action, dqn: 1 + gamma * Math.max.apply(null, target), double: 1 + gamma * target[action] };
  }

  function entropyPolicy(alpha) {
    bounded(alpha, 0.05, 2);
    const high = 1 / (1 + Math.exp(-1 / alpha));
    const low = 1 - high;
    return { probabilities: [low, high], entropy: -low * Math.log(low) - high * Math.log(high) };
  }

  function projectionTrace(gamma, steps) {
    bounded(gamma, 0, 0.99);
    if (!Number.isInteger(steps) || steps < 0 || steps > 20) throw new RangeError("Invalid steps");
    let parameter = 1;
    const rows = [{ step: 0, parameter, lossBefore: null, lossAfter: null }];
    for (let step = 1; step <= steps; step += 1) {
      const target = 2 * gamma * parameter;
      const next = 1.2 * gamma * parameter;
      const loss = function (value) {
        return (Math.pow(value - target, 2) + Math.pow(2 * value - target, 2)) / 2;
      };
      rows.push({ step, parameter: next, lossBefore: loss(parameter), lossAfter: loss(next) });
      parameter = next;
    }
    return rows;
  }

  function trustStep(step, budget) {
    bounded(step, 0, 3);
    bounded(budget, 0.001, 0.2);
    const probability = 1 / (1 + Math.exp(-step));
    const exact = 0.5 * Math.log(0.5 / probability) + 0.5 * Math.log(0.5 / (1 - probability));
    return { probability, exact, quadratic: step * step / 8, accepted: exact <= budget };
  }

  function planSequence(start, horizon, gain) {
    bounded(start, -10, 10);
    bounded(gain, 0.5, 1.5);
    if (!Number.isInteger(horizon) || horizon < 1 || horizon > 6) throw new RangeError("Invalid horizon");
    let best = { actions: [], states: [], cost: Infinity, candidates: 0 };
    let candidates = 0;
    function visit(state, actions, states, cost) {
      if (actions.length === horizon) {
        candidates += 1;
        if (cost < best.cost - 1e-10) best = { actions: actions.slice(), states: states.slice(), cost };
        return;
      }
      [0, -1, 1].forEach(function (action) {
        const next = state + gain * action;
        actions.push(action);
        states.push(next);
        visit(next, actions, states, cost + next * next + 0.1 * action * action);
        actions.pop();
        states.pop();
      });
    }
    visit(start, [], [start], 0);
    best.candidates = candidates;
    return best;
  }

  function planningComparison(horizon, gain) {
    const initial = planSequence(3, horizon, gain);
    const openStates = [3];
    const feedbackStates = [3];
    const feedbackActions = [];
    let openCost = 0;
    let feedbackCost = 0;
    for (let step = 0; step < horizon; step += 1) {
      const nextOpen = openStates[step] + initial.actions[step];
      openStates.push(nextOpen);
      openCost += nextOpen * nextOpen + 0.1 * initial.actions[step] ** 2;
      const replanned = planSequence(feedbackStates[step], horizon, gain);
      const action = replanned.actions[0];
      const nextFeedback = feedbackStates[step] + action;
      feedbackActions.push(action);
      feedbackStates.push(nextFeedback);
      feedbackCost += nextFeedback * nextFeedback + 0.1 * action ** 2;
    }
    return { predicted: initial.states, openStates, feedbackStates, openActions: initial.actions,
      feedbackActions, openCost, feedbackCost, candidatesPerPlan: initial.candidates };
  }

  const api = { bellman, boundaryTarget, gaeTerms, doubleTargets, entropyPolicy,
    projectionTrace, trustStep, planSequence, planningComparison };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;
  const chinese = document.documentElement.lang.startsWith("zh");
  const words = function (zh, en) { return chinese ? zh : en; };
  const number = function (value) { return Number(value.toFixed(4)).toString(); };

  function mount(root, index) {
    const kind = root.dataset.drlLab;
    const panel = document.createElement("section");
    panel.className = "drl-experiment";
    const kicker = document.createElement("p");
    kicker.className = "drl-kicker";
    kicker.textContent = words("动手算一遍 · 自拟算例", "TRY THE CALCULATION · TOY EXAMPLE");
    const heading = document.createElement("h3");
    heading.id = "drl-title-" + index;
    panel.setAttribute("aria-labelledby", heading.id);
    const controls = document.createElement("div");
    controls.className = "drl-controls";
    const result = document.createElement("div");
    result.className = "drl-result";
    result.setAttribute("aria-live", "polite");
    result.setAttribute("aria-atomic", "true");
    const explanation = document.createElement("p");
    explanation.className = "drl-explanation";
    panel.append(kicker, heading, controls, result, explanation);
    let tableExpanded = false;

    function slider(labelText, minimum, maximum, step, initial) {
      const label = document.createElement("label");
      const name = document.createElement("span");
      name.textContent = labelText;
      const input = document.createElement("input");
      input.type = "range";
      input.min = minimum;
      input.max = maximum;
      input.step = step;
      input.value = initial;
      const current = document.createElement("output");
      input.id = "drl-input-" + index + "-" + controls.children.length;
      current.htmlFor = input.id;
      current.textContent = initial;
      input.addEventListener("input", function () { current.textContent = input.value; });
      label.append(name, input, current);
      controls.append(label);
      return input;
    }

    function bars(labels, values, scale) {
      result.replaceChildren();
      values.forEach(function (value, offset) {
        const row = document.createElement("div");
        row.className = "drl-bar-row";
        const label = document.createElement("span");
        label.textContent = labels[offset];
        const track = document.createElement("span");
        track.className = "drl-track";
        track.setAttribute("aria-hidden", "true");
        const fill = document.createElement("span");
        fill.style.width = Math.min(100, 100 * Math.abs(value) / scale) + "%";
        if (value < 0) fill.className = "drl-negative";
        track.append(fill);
        const valueText = document.createElement("strong");
        valueText.textContent = number(value);
        row.append(label, track, valueText);
        result.append(row);
      });
    }

    function dataTable(captionText, headers, rows) {
      const details = document.createElement("details");
      details.className = "drl-data";
      details.open = tableExpanded;
      details.addEventListener("toggle", function () {
        if (details.isConnected) tableExpanded = details.open;
      });
      const summary = document.createElement("summary");
      summary.textContent = words("展开逐步计算表", "Show the calculation table");
      const scroll = document.createElement("div");
      scroll.className = "drl-table-scroll";
      scroll.tabIndex = 0;
      scroll.setAttribute("role", "region");
      scroll.setAttribute("aria-label", captionText);
      const table = document.createElement("table");
      const caption = document.createElement("caption");
      caption.textContent = captionText;
      const head = document.createElement("thead");
      const headingRow = document.createElement("tr");
      headers.forEach(function (text) {
        const cell = document.createElement("th");
        cell.scope = "col";
        cell.textContent = text;
        headingRow.append(cell);
      });
      head.append(headingRow);
      const body = document.createElement("tbody");
      rows.forEach(function (row) {
        const line = document.createElement("tr");
        row.forEach(function (value) {
          const cell = document.createElement("td");
          cell.textContent = typeof value === "number" ? number(value) : value;
          line.append(cell);
        });
        body.append(line);
      });
      table.append(caption, head, body);
      scroll.append(table);
      details.append(summary, scroll);
      result.append(details);
    }

    function plot(series, xLabel, yLabel, ceiling) {
      result.replaceChildren();
      const svgNamespace = "http://www.w3.org/2000/svg";
      function element(tag, attributes, text) {
        const node = document.createElementNS(svgNamespace, tag);
        Object.keys(attributes).forEach(function (key) { node.setAttribute(key, attributes[key]); });
        if (text !== undefined) node.textContent = text;
        return node;
      }
      const chart = element("svg", { viewBox: "0 0 620 280", role: "img", class: "drl-plot" });
      chart.setAttribute("aria-label", yLabel + ": " + series.map(function (line) { return line.name; }).join(", "));
      const values = series.flatMap(function (line) { return line.values; });
      const lower = Math.min(0, Math.min.apply(null, values));
      const upper = ceiling === undefined ? Math.max(1, Math.max.apply(null, values)) : ceiling;
      const count = Math.max.apply(null, series.map(function (line) { return line.values.length; }));
      const mapX = function (offset) { return 54 + 544 * offset / Math.max(1, count - 1); };
      const mapY = function (value) { return 224 - 190 * (value - lower) / (upper - lower); };
      for (let tick = 0; tick <= 4; tick += 1) {
        const value = lower + (upper - lower) * tick / 4;
        chart.append(element("line", { x1: 54, y1: mapY(value), x2: 598, y2: mapY(value), class: "drl-grid" }));
        chart.append(element("text", { x: 45, y: mapY(value) + 5, "text-anchor": "end" }, Number(value.toFixed(2)).toString()));
      }
      chart.append(element("text", { x: 54, y: 18 }, yLabel));
      chart.append(element("text", { x: 598, y: 267, "text-anchor": "end" }, xLabel));
      chart.append(element("text", { x: 54, y: 245 }, "0"));
      if (count > 1) chart.append(element("text", { x: 598, y: 245, "text-anchor": "end" }, String(count - 1)));
      const legend = document.createElement("div");
      legend.className = "drl-legend";
      series.forEach(function (line, lineIndex) {
        const coordinates = line.values.map(function (value, offset) { return mapX(offset) + "," + mapY(value); }).join(" ");
        chart.append(element("polyline", { points: coordinates, class: "drl-line drl-series-" + lineIndex }));
        line.values.forEach(function (value, offset) {
          chart.append(element("circle", { cx: mapX(offset), cy: mapY(value), r: 3,
            class: "drl-point drl-series-" + lineIndex }));
        });
        const key = document.createElement("span");
        key.className = "drl-series-" + lineIndex;
        key.textContent = line.name;
        legend.append(key);
      });
      result.append(chart, legend);
    }

    if (kind === "bellman") {
      heading.textContent = words("多等一步，值得吗？", "Is waiting one more step worth it?");
      const discount = slider("γ", 0, 1, 0.05, 0.9);
      let iteration = 0;
      const advance = document.createElement("button");
      advance.type = "button";
      advance.textContent = words("更新一轮", "One backup");
      const reset = document.createElement("button");
      reset.type = "button";
      reset.textContent = words("从零开始", "Start from zero");
      controls.append(advance, reset);
      const update = function () {
        const values = bellman(Number(discount.value), iteration);
        bars([words("起点", "Start"), words("充好电", "Charged")], [values.start, values.charged], 4);
        explanation.textContent = words("第 ", "Iteration ") + iteration + words(
          " 轮。起点比较立即拿 2 分，与 γ × 上一轮的充电状态估值。这个环境两轮就传播到位。",
          ". Start compares taking 2 now with γ × the previous charged-state value. Two backups suffice in this environment.");
      };
      advance.addEventListener("click", function () { iteration += 1; update(); });
      reset.addEventListener("click", function () { iteration = 0; update(); });
      discount.addEventListener("input", update);
      update();
    } else if (kind === "boundary") {
      heading.textContent = words("同一条数据，结束原因不同", "Same transition, different boundary");
      const label = document.createElement("label");
      label.textContent = words("结束原因", "Boundary");
      const select = document.createElement("select");
      [["continuing", words("继续运行", "Continuing")],
       ["terminated", words("任务真的结束", "True termination")],
       ["truncated", words("外部时限截断", "External truncation")]].forEach(function (entry) {
        const option = document.createElement("option");
        option.value = entry[0];
        option.textContent = entry[1];
        select.append(option);
      });
      label.append(select);
      controls.append(label);
      const update = function () {
        const target = boundaryTarget(1, 5, 0.9, select.value === "terminated");
        bars([words("即时奖励", "Immediate reward"), words("TD 目标值", "TD target")], [1, target], 6);
        explanation.textContent = select.value === "terminated"
          ? words("任务真正终止：不再加未来估值，target = 1。", "True termination: future value is zero. Target = 1.")
          : words("target = 1 + 0.9 × 5 = 5.5。若因外部时限截断，要用重置前的最后观测，不能用下一局的初始状态。",
            "Target = 1 + 0.9 × 5 = 5.5. At truncation, use the final observation, not a reset state.");
      };
      select.addEventListener("change", update);
      update();
    } else if (kind === "gae") {
      heading.textContent = words("哪几步的误差传回了现在？", "Which residuals reach the current step?");
      const decay = slider("λ", 0, 1, 0.05, 0.8);
      const update = function () {
        const terms = gaeTerms(0.9, Number(decay.value), [1, 2, -1]);
        bars(["δ₀", "(γλ) δ₁", "(γλ)² δ₂"], terms, 2);
        explanation.textContent = "A₀ = " + terms.map(function (value) {
          return value < 0 ? "(" + number(value) + ")" : number(value);
        }).join(" + ") + " = " +
          number(terms.reduce(function (sum, value) { return sum + value; }, 0)) +
          words("。条形表示绝对长度，负项用斜线区分；γ 固定为 0.9。", ". Bars show magnitude; hatching marks negative terms. γ = 0.9.");
        dataTable(words("每一步怎样影响 A₀", "How each step contributes to A₀"),
          [words("步", "Step"), "δ", "(γλ)ˡ", words("贡献", "Contribution")],
          terms.map(function (term, offset) { return [offset, [1, 2, -1][offset], Math.pow(0.9 * Number(decay.value), offset), term]; }));
      };
      decay.addEventListener("input", update);
      update();
    } else if (kind === "double-q") {
      heading.textContent = words("选动作的网络，和估值的网络", "The network that chooses versus the one that evaluates");
      const first = slider(words("Online Q · 动作 A", "Online Q · action A"), 0, 8, 0.5, 5);
      const update = function () {
        const values = doubleTargets([Number(first.value), 4], [2, 6], 0.9);
        bars(["DQN target", "Double DQN target"], [values.dqn, values.double], 7);
        explanation.textContent = "Online=[" + first.value + ", 4], target=[2, 6]. " +
          words("Online 选择 ", "Online selects ") + (values.action === 0 ? "A" : "B") +
          words("。值相同时取 A。这里没有真实 Q*，不能据此说哪个更准。",
            ". Ties choose A. True Q* is unknown, so this does not establish which estimate is more accurate.");
      };
      first.addEventListener("input", update);
      update();
    } else if (kind === "entropy") {
      heading.textContent = words("固定 Q，看看温度怎么改变概率", "Hold Q fixed and change the temperature");
      const temperature = slider("α", 0.05, 2, 0.05, 0.5);
      const update = function () {
        const values = entropyPolicy(Number(temperature.value));
        bars([words("动作 A · Q=0", "Action A · Q=0"), words("动作 B · Q=1", "Action B · Q=1")], values.probabilities, 1);
        explanation.textContent = "H = " + number(values.entropy) + " nats. " +
          words("条形显示动作概率。这是固定 Q 下的两动作解析解，不是连续 SAC 的训练曲线。",
            "Bars show action probabilities. This is an analytic distribution for fixed Q, not a continuous SAC learning curve.");
      };
      temperature.addEventListener("input", update);
      update();
    } else if (kind === "projection") {
      heading.textContent = words("回归误差降了，估值会更准吗？", "Does a better fit mean better values?");
      const discount = slider("γ", 0, 0.99, 0.01, 0.9);
      let iteration = 0;
      const advance = document.createElement("button");
      advance.type = "button";
      advance.textContent = words("再拟合一轮", "Fit one more round");
      const reset = document.createElement("button");
      reset.type = "button";
      reset.textContent = words("重置", "Reset");
      controls.append(advance, reset);
      const update = function () {
        const rows = projectionTrace(Number(discount.value), iteration);
        plot([{ name: "V(1)=θ", values: rows.map(function (row) { return row.parameter; }) },
          { name: "V(2)=2θ", values: rows.map(function (row) { return 2 * row.parameter; }) },
          { name: words("真实值 = 0", "True value = 0"), values: rows.map(function () { return 0; }) }],
        words("拟合轮数", "Fit iteration"), words("价值估计", "Estimated value"));
        dataTable(words("每轮只比较同一份旧标签", "Compare the same frozen labels within each round"),
          [words("轮", "Round"), "θ", words("拟合前 MSE", "MSE before"), words("拟合后 MSE", "MSE after")],
          rows.map(function (row) { return [row.step, row.parameter, row.lossBefore === null ? "—" : row.lossBefore, row.lossAfter === null ? "—" : row.lossAfter]; }));
        advance.disabled = iteration >= 20;
        explanation.textContent = "θₖ₊₁ = " + number(1.2 * Number(discount.value)) + " θₖ. " + words(
          "先点几轮，再把 γ 调到 0.5。换折扣会从 θ=1 重新计算。此图每轮精确拟合，不是神经网络训练曲线。",
          "Step a few rounds, then set γ to 0.5. Changing γ recomputes from θ=1. Each fit is exact; this is not a neural training curve.");
      };
      advance.addEventListener("click", function () { if (iteration < 20) iteration += 1; update(); });
      reset.addEventListener("click", function () { iteration = 0; update(); });
      discount.addEventListener("input", update);
      update();
    } else if (kind === "trust-step") {
      heading.textContent = words("参数改了一点，动作概率变了多少？", "How much does a parameter step change action probabilities?");
      const step = slider(words("Logit 步长 Δ", "Logit step Δ"), 0, 3, 0.05, 0.3);
      const budget = slider(words("KL 预算 δ", "KL budget δ"), 0.005, 0.2, 0.005, 0.01);
      const update = function () {
        const values = trustStep(Number(step.value), Number(budget.value));
        bars([words("旧 P(1)", "Old P(1)"), words("新 P(1)", "New P(1)")], [0.5, values.probability], 1);
        dataTable(words("真实 KL 与局部近似", "Exact KL versus its local approximation"),
          [words("量", "Quantity"), words("数值", "Value")],
          [[words("真实 KL", "Exact KL"), values.exact], ["Δ² / 8", values.quadratic], ["δ", Number(budget.value)]]);
        explanation.textContent = "KL=" + number(values.exact) + ", δ=" + budget.value + ". " +
          (values.accepted ? words("在预算内。", "Within budget. ") : words("超出预算。", "Over budget. ")) +
          words("这里只检查你提出的步长，不自动执行 TRPO 回溯。KL 预算不是 reward，也不是 PPO clip range。",
            "This checks your proposed step; it does not run TRPO backtracking. The KL budget is not a reward or PPO clip range.");
      };
      step.addEventListener("input", update);
      budget.addEventListener("input", update);
      update();
    } else if (kind === "mpc") {
      heading.textContent = words("预测错了以后，还要照着原计划走吗？", "When prediction is wrong, keep the original plan?");
      const gain = slider(words("模型增益 b̂", "Model gain b̂"), 0.5, 1.5, 0.1, 1.5);
      const horizon = slider(words("规划长度 H", "Planning horizon H"), 2, 6, 1, 5);
      const update = function () {
        const steps = Number(horizon.value);
        const values = planningComparison(steps, Number(gain.value));
        plot([{ name: words("每步重新规划", "Replan each step"), values: values.feedbackStates },
          { name: words("执行原计划", "Execute initial plan"), values: values.openStates },
          { name: words("原计划的预测", "Initial prediction"), values: values.predicted }],
        words("环境步数", "Environment step"), words("位置 x · 目标 0", "Position x · goal 0"));
        dataTable(words("真实轨迹与执行动作", "Real trajectories and executed actions"),
          ["t", words("原计划 x", "Open-loop x"), words("原计划 a", "Open-loop a"), "MPC x", "MPC a"],
          values.openStates.map(function (state, offset) {
            return [offset, state, offset < steps ? values.openActions[offset] : "—",
              values.feedbackStates[offset], offset < steps ? values.feedbackActions[offset] : "—"];
          }));
        explanation.textContent = words("真实累计代价：原计划 ", "Real cumulative cost: open-loop ") + number(values.openCost) +
          ", MPC " + number(values.feedbackCost) + ". " + words("每次枚举 ", "Each plan enumerates ") + values.candidatesPerPlan +
          words(" 条序列；原计划搜索 1 次，MPC 搜索 ", " sequences; open-loop plans once and MPC plans ") + steps +
          words(" 次。两者都执行 H 步；MPC 每步仍向前看 H 步。代价越低越好，但反馈并不能修正所有模型错误。",
            " times. Both execute H steps; MPC still looks H steps ahead at each decision. Lower cost is better, but feedback cannot correct every model error.");
      };
      gain.addEventListener("input", update);
      horizon.addEventListener("input", update);
      update();
    } else {
      return;
    }
    root.replaceChildren(panel);
  }

  document.querySelectorAll("[data-drl-lab]").forEach(mount);
}());
