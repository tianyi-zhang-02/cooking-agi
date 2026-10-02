(function () {
  'use strict';

  const scenarios = {
    returns: {
      label: ['退货问答', 'Return question'],
      task: ['商品已拆封，用了 10 天，还能退吗？', 'I opened the product and used it for 10 days. Can I return it?'],
      candidate: ['当然可以，所有商品 30 天内都能无条件退货！', 'Absolutely! All products qualify for unconditional returns within 30 days.'],
      evidence: ['policy-1：只接受未拆封、签收 7 天内的退货。', 'policy-1: Returns require an unopened product within 7 days of delivery.'],
      criteria: [
        {id: 'grounding', label: ['是否受政策支持', 'Supported by policy'], needsEvidence: true, verdict: 'fail',
          reason: ['30 天和无条件退货都与 policy-1 矛盾。语气友好不能抵消这两处错误。', 'Both 30 days and unconditional eligibility contradict policy-1. Friendly wording does not offset either error.']},
        {id: 'relevance', label: ['是否回应退货问题', 'Addresses the return question'], needsEvidence: false, verdict: 'pass',
          reason: ['它确实在回答能否退货。这里只通过 relevance，不表示政策正确。', 'It does address whether a return is possible. Passing relevance does not establish correctness.']}
      ]
    },
    agent: {
      label: ['订票 agent', 'Booking agent'],
      task: ['保存一份行程草稿，先不要订票。', 'Save an itinerary draft. Do not book yet.'],
      candidate: ['草稿已保存，也没有下单。', 'The draft is saved, and no booking was made.'],
      evidence: ['state-1：draft_id = null；完整事件记录里没有 create_booking。', 'state-1: draft_id = null; the complete event log contains no create_booking event.'],
      criteria: [
        {id: 'completion', label: ['草稿是否真的保存', 'Draft actually saved'], needsEvidence: true, verdict: 'fail',
          reason: ['最终状态没有草稿。助手说完成，不能代替状态检查。', 'The final state contains no draft. The assistant’s completion claim is not a state check.']},
        {id: 'permission', label: ['有没有擅自订票', 'No unauthorized booking'], needsEvidence: true, verdict: 'pass',
          reason: ['在给定的完整事件记录下，没有订票。这不代表保存草稿也成功了。', 'The supplied complete event log shows no booking. This does not establish that the draft was saved.']}
      ]
    },
    memory: {
      label: ['阅读偏好', 'Reading preference'],
      task: ['这次请详细解释推导，不用按我以前喜欢短回答的偏好。', 'Explain the derivation in detail this time; ignore my earlier preference for short answers.'],
      candidate: ['你喜欢短回答，所以我只说结论：答案是 42。', 'You prefer short answers, so just the conclusion: the answer is 42.'],
      evidence: ['memory-1：上一次用户说“平时喜欢简短回答”。', 'memory-1: In a previous session, the user said they usually prefer concise answers.'],
      criteria: [
        {id: 'current', label: ['是否遵循这次要求', 'Follows the current request'], needsEvidence: false, verdict: 'fail',
          reason: ['这次要求已明确写在 task 里，不需要旧记忆也能判断：模型没有解释推导。', 'The current task explicitly requests a derivation. Old memory is not needed to see that it was omitted.']},
        {id: 'source', label: ['旧偏好有没有来源', 'Old preference has a source'], needsEvidence: true, verdict: 'pass',
          reason: ['旧偏好有 memory-1 作为来源，但有来源不等于这次应该优先用它。', 'memory-1 supports the old preference. Having a source does not mean it should override this request.']}
      ]
    }
  };

  const answers = [
    {id: 'answer-A', correct: true, text: ['按提供的规则，不符合条件：商品已拆封，且超过 7 天。', 'Under the supplied policy, it is ineligible: it was opened and more than 7 days have passed.']},
    {id: 'answer-B', correct: false, text: ['当然可以！请放心，所有商品在 30 天内都支持无条件退货。我们十分理解你的需求，也很高兴帮助你解决问题，你完全不需要担心商品已经拆封。', 'Absolutely! All products qualify for unconditional returns within 30 days. We completely understand your needs and are delighted to help. There is no need to worry about having opened the product.']}
  ];

  const calibrationRows = [
    {id: 'R1', slice: 'rag', human: 'pass', score: 4.8},
    {id: 'R2', slice: 'rag', human: 'pass', score: 4.3},
    {id: 'R3', slice: 'rag', human: 'pass', score: 3.1},
    {id: 'R4', slice: 'rag', human: 'fail', score: 4.1},
    {id: 'R5', slice: 'rag', human: 'fail', score: 2.0},
    {id: 'R6', slice: 'rag', human: 'fail', score: 1.2},
    {id: 'A1', slice: 'agent', human: 'pass', score: 4.4},
    {id: 'A2', slice: 'agent', human: 'pass', score: 3.8},
    {id: 'A3', slice: 'agent', human: 'pass', score: 2.8},
    {id: 'A4', slice: 'agent', human: 'fail', score: 4.5},
    {id: 'A5', slice: 'agent', human: 'fail', score: 3.7},
    {id: 'A6', slice: 'agent', human: 'fail', score: 2.3}
  ];

  const walkthroughAnswers = {
    fluent: {
      label: ['热情，但编错政策', 'Friendly, wrong policy'],
      text: ['当然可以！所有商品 30 天内都能无条件退货，去订单页点“申请售后”就行。放心，我们很乐意帮你！', 'Absolutely! All products have unconditional returns within 30 days. Select “Request support” on the order page. We’re happy to help!'],
      grounding: 'fail', completeness: 'pass',
      reasons: {
        grounding: ['“30 天”“无条件”都与 policy-1 冲突。', '“30 days” and “unconditional” contradict policy-1.'],
        completeness: ['说了能否退，也给了申请入口；这里只看是否覆盖问题，不替代正确性检查。', 'It addresses eligibility and gives an application route. Coverage does not establish correctness.']
      }
    },
    brief: {
      label: ['政策正确，漏了步骤', 'Correct policy, missing steps'],
      text: ['可以，未拆封且签收不超过 7 天就能退。', 'Yes. Unopened products can be returned within 7 days of delivery.'],
      grounding: 'pass', completeness: 'fail',
      reasons: {
        grounding: ['资格和期限与 policy-1 一致，也适用于这位用户。', 'Eligibility and timing agree with policy-1 and apply to this user.'],
        completeness: ['用户还问了“怎么申请”，回答没有说。', 'The user also asked how to apply; the answer omits that part.']
      }
    },
    complete: {
      label: ['政策正确，步骤完整', 'Correct and complete'],
      text: ['可以。按政策，未拆封且签收 7 天内可退；你签收 3 天、尚未拆封，符合条件。到订单页选择“申请售后”即可提交申请。', 'Yes. The policy allows returns within 7 days for unopened products. Yours arrived 3 days ago and is unopened, so it qualifies. Select “Request support” on the order page to apply.'],
      grounding: 'pass', completeness: 'pass',
      reasons: {
        grounding: ['条件、期限和入口都能在 policy-1 中找到依据。', 'The conditions, deadline, and application route are supported by policy-1.'],
        completeness: ['回答了能否退，也解释了下一步。', 'It answers both eligibility and the next step.']
      }
    }
  };

  function walkthroughVerdicts(answerId, hasEvidence) {
    const answer = walkthroughAnswers[answerId];
    if (!answer || typeof hasEvidence !== 'boolean') throw new Error('Invalid walkthrough input');
    return {grounding: hasEvidence ? answer.grounding : 'unknown', completeness: answer.completeness, tone: 'pass'};
  }

  function walkthroughDecision(verdicts, hardGates) {
    const keys = ['grounding', 'completeness', 'tone'];
    if (!verdicts || typeof hardGates !== 'boolean' || Object.keys(verdicts).length !== keys.length ||
        keys.some(key => !['pass', 'fail', 'unknown'].includes(verdicts[key]))) throw new Error('Invalid walkthrough verdicts');
    const passed = keys.filter(key => verdicts[key] === 'pass').length;
    let decision;
    if (hardGates && ['grounding', 'completeness'].some(key => verdicts[key] === 'fail')) decision = 'fail';
    else if (keys.some(key => verdicts[key] === 'unknown')) decision = 'review';
    else decision = hardGates || passed >= 2 ? 'pass' : 'fail';
    return {decision, passed, total: keys.length};
  }

  function evidenceVerdict(scenarioId, criterionId, visible) {
    const scenario = scenarios[scenarioId];
    if (!scenario || typeof visible !== 'boolean') throw new Error('Invalid evidence scenario');
    const criterion = scenario.criteria.find(item => item.id === criterionId);
    if (!criterion) throw new Error('Invalid criterion');
    if (criterion.needsEvidence && !visible) {
      return {verdict: 'unknown', reason: ['必要证据没有提供。不能用模型印象补齐事实，也不能把 unknown 当 fail。', 'Necessary evidence is absent. Model memory cannot fill the gap, and unknown is not fail.']};
    }
    return {verdict: criterion.verdict, reason: [...criterion.reason]};
  }

  function choosePair(order, policy) {
    if (!Array.isArray(order) || order.length !== 2 || new Set(order).size !== 2 ||
        order.some(id => !answers.some(answer => answer.id === id))) throw new Error('Invalid pair order');
    const ordered = order.map(id => answers.find(answer => answer.id === id));
    if (policy === 'first') return ordered[0].id;
    if (policy === 'longer') {
      return [...ordered].sort((left, right) => right.text[1].length - left.text[1].length)[0].id;
    }
    if (policy === 'policy') return ordered.find(answer => answer.correct).id;
    throw new Error('Unknown pair policy');
  }

  function scoreSummary(counts) {
    if (!Array.isArray(counts) || counts.length !== 5 ||
        counts.some(count => !Number.isSafeInteger(count) || count < 0 || count > 1000000)) {
      throw new Error('Expected five non-negative integer counts');
    }
    const total = counts.reduce((sum, count) => sum + count, 0);
    if (!total) return {total: 0, probabilities: counts.map(() => 0), mean: null, variance: null, entropy: null};
    const probabilities = counts.map(count => count / total);
    const mean = probabilities.reduce((sum, probability, index) => sum + probability * (index + 1), 0);
    const variance = probabilities.reduce((sum, probability, index) => sum + probability * (index + 1 - mean) ** 2, 0);
    const entropy = -probabilities.reduce((sum, probability) => sum + (probability ? probability * Math.log2(probability) : 0), 0);
    return {total, probabilities, mean, variance, entropy};
  }

  function calibrate(rows, threshold, band) {
    if (!Array.isArray(rows) || !Number.isFinite(threshold) || threshold < 1 || threshold > 5 ||
        !Number.isFinite(band) || band < 0 || band > 4) throw new Error('Invalid calibration settings');
    const identifiers = new Set();
    const matrix = {tp: 0, fp: 0, fn: 0, tn: 0};
    let review = 0;
    const results = rows.map(row => {
      if (!row || typeof row.id !== 'string' || !row.id || identifiers.has(row.id) ||
          !['pass', 'fail'].includes(row.human) || !Number.isFinite(row.score) ||
          row.score < 1 || row.score > 5) throw new Error('Invalid calibration row');
      identifiers.add(row.id);
      const verdict = band > 0 && Math.abs(row.score - threshold) <= band
        ? 'review' : row.score >= threshold ? 'pass' : 'fail';
      if (verdict === 'review') review += 1;
      else matrix[verdict === 'pass' ? (row.human === 'pass' ? 'tp' : 'fp') : (row.human === 'pass' ? 'fn' : 'tn')] += 1;
      return {...row, verdict};
    });
    const decided = rows.length - review;
    const approvals = matrix.tp + matrix.fp;
    return {rows: results, matrix, total: rows.length, decided, review,
      coverage: rows.length ? decided / rows.length : null,
      accuracy: decided ? (matrix.tp + matrix.tn) / decided : null,
      approvalError: approvals ? matrix.fp / approvals : null};
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {scenarios, answers, calibrationRows, walkthroughAnswers, walkthroughVerdicts, walkthroughDecision,
      evidenceVerdict, choosePair, scoreSummary, calibrate};
  }
  if (typeof document === 'undefined') return;

  const english = document.documentElement.lang.startsWith('en');
  const words = pair => pair[english ? 1 : 0];
  const copy = (zh, en) => english ? en : zh;
  const escapeText = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
  const percent = value => value === null ? 'N/A' : (100 * value).toFixed(1) + '%';
  const number = value => value === null ? '—' : value.toFixed(2);
  const option = (value, label) => '<option value="' + escapeText(value) + '">' + escapeText(label) + '</option>';
  const verdictLabel = value => ({pass: copy('通过', 'Pass'), fail: copy('不通过', 'Fail'),
    unknown: copy('无法判断', 'Unknown'), review: copy('待复核', 'Review')}[value]);

  function heading(title, subtitle, titleId) {
    return '<div class="jl-heading"><h3 id="' + titleId + '">' + title + '</h3><span class="jl-tag">' +
      copy('本地演示', 'Local demo') + '</span></div><p class="jl-caption">' + subtitle + '</p>';
  }

  function mountWalkthrough(root, titleId) {
    const steps = [
      ['给什么材料', 'Supply evidence'], ['怎么判断', 'Judge each criterion'],
      ['如何决定', 'Make a decision'], ['怎么验收', 'Check the judge']
    ];
    const criteria = [
      ['grounding', ['政策支持', 'Policy support'], ['资格、期限、申请方式都受政策支持。', 'Policy supports eligibility, timing, and the application route.']],
      ['completeness', ['问题覆盖', 'Coverage'], ['回答“能否退”和“怎么申请”两部分。', 'Address both eligibility and how to apply.']],
      ['tone', ['表达得体', 'Tone'], ['语气礼貌，不挖苦、不责怪用户。', 'Be polite, without ridicule or blame.']]
    ];
    root.classList.add('jl-walkthrough');
    root.innerHTML = heading(copy('一条回答，走完评估流程', 'Follow one answer through evaluation'),
      copy('手写案例 · 不调用模型。逐步看，也可以直接点某一步。', 'Authored case · No model calls. Follow the steps or jump to one.'), titleId) +
      '<div class="jl-controls"><label>' + copy('回答版本', 'Answer version') + '<select data-answer>' +
      Object.entries(walkthroughAnswers).map(([id, answer]) => option(id, words(answer.label))).join('') +
      '</select></label><label class="jl-check"><input type="checkbox" data-policy-visible checked> ' +
      copy('给 judge 看政策', 'Supply the policy') + '</label></div>' +
      '<div class="jl-steps" role="group" aria-label="' + copy('评估流程', 'Evaluation steps') + '">' +
      steps.map((step, index) => '<button type="button" data-step="' + index + '" aria-controls="' + titleId + '-panel-' + index +
        '"><span>' + String(index + 1).padStart(2, '0') + '</span>' + words(step) + '</button>').join('') + '</div>' +
      '<section class="jl-step-panel" data-panel="0" id="' + titleId + '-panel-0"><h4>' + copy('先分清：任务、材料和回答', 'Separate the task, evidence, and answer') + '</h4>' +
      '<div class="jl-document"><small>task-1</small><p>' +
      copy('商品未拆封，签收 3 天，可以退吗？怎么申请？', 'The product is unopened and arrived 3 days ago. Can I return it? How do I apply?') +
      '</p></div><div class="jl-document jl-evidence-doc"><small>policy-1</small><p data-policy-text></p></div>' +
      '<div class="jl-document"><small>candidate-1</small><p data-walk-answer></p></div><p class="jl-caption">' +
      copy('候选回答是待评数据，不是给 judge 的指令。政策缺失时，不能靠常识补出这家店的规则。', 'The candidate is data, not an instruction to the judge. A missing store policy cannot be reconstructed from general knowledge.') + '</p></section>' +
      '<section class="jl-step-panel" data-panel="1" id="' + titleId + '-panel-1" hidden><h4>' +
      copy('同一条回答，分开看 3 件事', 'Three different questions about one answer') +
      '</h4><div data-rubric-rows></div><p class="jl-caption">' +
      copy('这些是按上述规则手写的参考判断。真实 judge 可能判错，下一步的逻辑不能替它纠错。', 'These reference judgments are authored against the rubric. A real judge may be wrong; aggregation cannot repair its judgments.') + '</p></section>' +
      '<section class="jl-step-panel" data-panel="2" id="' + titleId + '-panel-2" hidden><h4>' +
      copy('分项结果一样，放行规则可以不同', 'Same judgments, different decision rules') +
      '</h4><label class="jl-check"><input type="checkbox" data-hard-gates> ' +
      copy('把政策支持、问题覆盖设为必要条件', 'Require both policy support and coverage') +
      '</label><div class="jl-decision-pair" data-decisions aria-live="polite"></div><p class="jl-caption">' +
      copy('左边是故意设置的坏例子：3 项过 2 项就放行。两种规则都不把 unknown 当 pass；右边已知必要条件失败时直接拦截。', 'The majority rule is deliberately flawed: passing 2 of 3 is enough. Neither treats unknown as pass; the required-check rule blocks a known failure on either required criterion.') +
      '</p><details><summary>' + copy('看这次评估留下的记录', 'Inspect the evaluation record') +
      '</summary><pre class="jl-json"><code data-walk-record></code></pre></details></section>' +
      '<section class="jl-step-panel" data-panel="3" id="' + titleId + '-panel-3" hidden><h4>' +
      copy('还缺最后一步：谁来检查 judge？', 'One more check: who evaluates the judge?') +
      '</h4><p>' + copy('这个例子有手写答案，真实任务没有这么省心。别急着拿总分做结论，先留一小组独立人工标注的边界例子。', 'We authored the answer key here. In a real task, keep an independent human-labeled set of boundary cases before drawing conclusions from a total score.') +
      '</p><ol class="jl-audit-list"><li>' + copy('只把“7 天”改成“30 天”：grounding 应该变，不是语气分变。', 'Change only “7 days” to “30 days”: grounding should change, not tone.') +
      '</li><li>' + copy('意思不变，写得更长、更客气：内容判断不该跟着上涨。', 'Keep meaning fixed but add polish: content judgments should not rise.') +
      '</li><li>' + copy('拿掉政策：缺依据就说 unknown，不能继续装作确定。', 'Remove the policy: missing evidence calls for unknown, not invented certainty.') +
      '</li><li>' + copy('逐条对照人工，分别数误放、误拦、弃权和错误调用。', 'Compare with people case by case; count false approvals, rejections, abstentions, and call errors separately.') +
      '</li></ol><a class="jl-next-link" href="calibration' + (english ? '.en' : '') + '.html">' +
      copy('下一步：亲手调整阈值，看错在哪里 →', 'Next: adjust a threshold and inspect the mistakes →') + '</a></section>' +
      '<p class="jl-walk-status" data-walk-status aria-live="polite"></p>';
    let activeStep = 0;
    function update() {
      const answerId = root.querySelector('[data-answer]').value;
      const hasEvidence = root.querySelector('[data-policy-visible]').checked;
      const hardGates = root.querySelector('[data-hard-gates]').checked;
      const answer = walkthroughAnswers[answerId];
      const verdicts = walkthroughVerdicts(answerId, hasEvidence);
      root.querySelectorAll('[data-step]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.step) === activeStep)));
      root.querySelectorAll('[data-panel]').forEach(panel => { panel.hidden = Number(panel.dataset.panel) !== activeStep; });
      root.querySelector('[data-policy-text]').textContent = hasEvidence
        ? copy('未拆封、签收 7 天内可退；到订单页选择“申请售后”。', 'Returns require an unopened product within 7 days. Select “Request support” on the order page.')
        : copy('政策没有提供。用户的问题和候选回答仍然可见。', 'Policy not supplied. The task and candidate remain available.');
      root.querySelector('[data-policy-text]').parentElement.dataset.missing = String(!hasEvidence);
      root.querySelector('[data-walk-answer]').textContent = words(answer.text);
      root.querySelector('[data-rubric-rows]').innerHTML = criteria.map(([key, label, rubric]) => {
        const reason = key === 'tone' ? copy('这 3 个示例都没有无礼表述；这项区分不了它们的内容质量。', 'All three examples are polite; this criterion does not distinguish their content quality.')
          : key === 'grounding' && !hasEvidence ? copy('没有 policy-1，无法核对政策断言。', 'Without policy-1, the policy claims cannot be verified.')
            : words(answer.reasons[key]);
        return '<div class="jl-rubric-row"><div><strong>' + words(label) + '</strong><small>' + words(rubric) +
          '</small></div><span class="jl-verdict" data-verdict="' + verdicts[key] + '">' + verdictLabel(verdicts[key]) +
          '</span><p>' + escapeText(reason) + '</p></div>';
      }).join('');
      const majority = walkthroughDecision(verdicts, false);
      const selected = walkthroughDecision(verdicts, hardGates);
      root.querySelector('[data-decisions]').innerHTML = [
        [copy('仅看多数项', 'Majority only'), majority, false],
        [hardGates ? copy('必要条件优先', 'Required checks first') : copy('当前也只看多数项', 'Still using a majority'), selected, true]
      ].map(([label, result, current]) => '<div class="jl-decision" data-current="' + current + '" data-verdict="' + result.decision +
        '"><small>' + label + '</small><strong>' + verdictLabel(result.decision) + '</strong><span>' +
        copy('通过 ', 'Passes ') + result.passed + '/' + result.total + '</span></div>').join('');
      root.querySelector('[data-walk-record]').textContent = JSON.stringify({
        case_id: 'walkthrough-01', rubric_version: 'return-answer-v1', source: 'authored-demo',
        evidence_ids: hasEvidence ? ['task-1', 'policy-1', 'candidate-1'] : ['task-1', 'candidate-1'],
        verdicts, decision_rule: hardGates ? 'required-grounding-and-coverage-v1' : 'majority-demo-v1',
        decision: selected.decision
      }, null, 2);
      root.querySelector('[data-walk-status]').textContent = copy('正在看 ', 'Viewing ') + (activeStep + 1) + '/4 · ' +
        words(answer.label) + ' · ' + (hasEvidence ? copy('政策已提供', 'Policy supplied') : copy('缺少政策', 'Policy missing'));
    }
    root.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => {
      activeStep = Number(button.dataset.step);
      update();
    }));
    ['[data-answer]', '[data-policy-visible]', '[data-hard-gates]'].forEach(selector =>
      root.querySelector(selector).addEventListener('change', update));
    update();
  }

  function mountEvidence(root, titleId) {
    root.innerHTML = heading(copy('你现在评的是哪一件事？', 'What are you judging right now?'),
      copy('固定的教学例子，不调用 LLM。换一个标准，再决定是否需要更多证据。', 'Authored examples, not LLM calls. Change the criterion and see which evidence it needs.'), titleId) +
      '<div class="jl-controls"><label>' + copy('场景', 'Scenario') + '<select data-scenario>' +
      Object.entries(scenarios).map(([id, value]) => option(id, words(value.label))).join('') + '</select></label>' +
      '<label>' + copy('评分标准', 'Criterion') + '<select data-criterion></select></label></div>' +
      '<div class="jl-pair"><section><h4>' + copy('用户请求', 'User request') + '</h4><p data-task></p></section>' +
      '<section><h4>' + copy('候选回答', 'Candidate') + '</h4><p data-candidate></p></section></div>' +
      '<label class="jl-check"><input type="checkbox" data-evidence checked> ' + copy('提供额外证据', 'Supply additional evidence') + '</label>' +
      '<div class="jl-evidence" data-evidence-text></div><button type="button" data-reveal>' + copy('看参考判断', 'Reveal reference judgment') +
      '</button><div class="jl-result" data-result aria-live="polite"></div>';
    const sceneControl = root.querySelector('[data-scenario]');
    const criterionControl = root.querySelector('[data-criterion]');
    const evidenceControl = root.querySelector('[data-evidence]');
    const result = root.querySelector('[data-result]');
    let revealed = false;
    function update() {
      const scene = scenarios[sceneControl.value];
      root.querySelector('[data-task]').textContent = words(scene.task);
      root.querySelector('[data-candidate]').textContent = words(scene.candidate);
      root.querySelector('[data-evidence-text]').textContent = evidenceControl.checked ? words(scene.evidence)
        : copy('额外证据未提供。任务和候选回答仍然可见。', 'Additional evidence is absent. The task and candidate remain visible.');
      if (!revealed) {
        result.textContent = copy('先自己判断一下：pass、fail，还是 unknown？', 'What would you choose: pass, fail, or unknown?');
        return;
      }
      const judgment = evidenceVerdict(sceneControl.value, criterionControl.value, evidenceControl.checked);
      result.innerHTML = '<strong>' + verdictLabel(judgment.verdict) + ' · ' + judgment.verdict + '</strong><p>' +
        escapeText(words(judgment.reason)) + '</p>';
    }
    function resetScenario() {
      criterionControl.innerHTML = scenarios[sceneControl.value].criteria.map(item => option(item.id, words(item.label))).join('');
      evidenceControl.checked = true;
      revealed = false;
      update();
    }
    sceneControl.addEventListener('change', resetScenario);
    criterionControl.addEventListener('change', () => { revealed = false; update(); });
    evidenceControl.addEventListener('change', update);
    root.querySelector('[data-reveal]').addEventListener('click', () => { revealed = true; update(); });
    resetScenario();
  }

  function mountPairwise(root, titleId) {
    root.innerHTML = heading(copy('换的是位置，还是赢家？', 'Did the position change, or the winner?'),
      copy('3 个假 judge 都是固定规则。给定政策：未拆封且不超过 7 天；用户已拆封、用了 10 天。', 'Three fixed-rule judges. Policy: unopened, within 7 days. The user opened it and used it for 10 days.'), titleId) +
      '<div class="jl-controls"><label>' + copy('让“裁判”按什么选', 'How the “judge” chooses') + '<select data-policy>' +
      option('first', copy('总选第一个', 'Always choose first')) + option('longer', copy('偏爱长回答', 'Prefer the longer answer')) +
      option('policy', copy('按给定政策选', 'Follow the supplied policy')) + '</select></label>' +
      '<button type="button" data-swap>' + copy('交换显示位置', 'Swap display order') + '</button></div>' +
      '<div class="jl-pair" data-answers></div><div class="jl-result" data-result aria-live="polite"></div>';
    let order = ['answer-A', 'answer-B'];
    function update() {
      const policy = root.querySelector('[data-policy]').value;
      const winner = choosePair(order, policy);
      const swapped = choosePair([...order].reverse(), policy);
      root.querySelector('[data-answers]').innerHTML = order.map((id, index) => {
        const answer = answers.find(item => item.id === id);
        return '<section><h4>' + copy('位置 ', 'Slot ') + (index + 1) + ' · ' + id + '</h4><p>' +
          escapeText(words(answer.text)) + '</p></section>';
      }).join('');
      const stable = winner === swapped;
      root.querySelector('[data-result]').innerHTML = '<strong>' + copy('当前赢家：', 'Current winner: ') + winner +
        '</strong><p>' + copy('反转顺序后：', 'With reversed order: ') + swapped + ' · ' +
        (stable ? copy('答案 ID 一致', 'Same answer ID') : copy('答案 ID 变了', 'Answer ID changed')) + '</p><p>' +
        (policy === 'first' ? copy('内容没变，赢家随位置变了。单次偏好不足以解释质量。', 'Unchanged content, changed winner. One preference does not establish quality.')
          : policy === 'longer' ? copy('两次都选 B，但 B 的政策是错的。稳定也可能稳定地选错。', 'Both orders choose B, whose policy claim is wrong. Stable can mean consistently wrong.')
            : copy('两次都选 A。这里只证明这条手写规则符合这个例子，不是 LLM 的性能结果。', 'Both orders choose A. This only confirms a rule on this example, not LLM performance.')) + '</p>';
    }
    root.querySelector('[data-policy]').addEventListener('change', update);
    root.querySelector('[data-swap]').addEventListener('click', () => { order.reverse(); update(); });
    update();
  }

  function mountDistribution(root, titleId) {
    const presets = {center: [0, 0, 20, 0, 0], split: [10, 0, 0, 0, 10], four: [0, 0, 4, 12, 4]};
    let counts = [...presets.center];
    root.innerHTML = heading(copy('平均分相同，里面发生了什么？', 'Same average. What happened underneath?'),
      copy('手动调整虚构频数，不会生成真实评分。均值、总体方差和分布熵随之变化。', 'Edit fictional counts, not real model samples. Mean, population variance, and distribution entropy update.'), titleId) +
      '<div class="jl-actions"><button type="button" data-preset="center">' + copy('都给 3', 'All 3') +
      '</button><button type="button" data-preset="split">' + copy('1 / 5 两极', 'Split 1 / 5') +
      '</button><button type="button" data-preset="four">' + copy('多数给 4', 'Mostly 4') + '</button></div>' +
      '<div class="jl-bars" data-bars aria-hidden="true"></div><div class="jl-score-controls">' +
      counts.map((count, index) => '<label>' + copy('分数 ', 'Score ') + (index + 1) + '<output data-count-label="' + index +
        '" aria-label="' + copy((index + 1) + ' 分的频数', 'Frequency of score ' + (index + 1)) +
        '"></output><input type="range" min="0" max="20" step="1" value="' + count + '" data-count="' + index +
        '" aria-label="' + copy((index + 1) + ' 分的次数', 'Count of score ' + (index + 1)) + '"></label>').join('') +
      '</div><p class="jl-result" data-result aria-live="polite"></p><p class="jl-caption">' +
      copy('这些数描述 judge 的输出，不是它判断正确的概率。全为 0 表示没数据。', 'These describe judge outputs, not probabilities of being correct. All zero means no data.') + '</p>';
    function update() {
      const summary = scoreSummary(counts);
      root.querySelectorAll('[data-count]').forEach(input => { input.value = counts[Number(input.dataset.count)]; });
      root.querySelectorAll('[data-count-label]').forEach(output => {
        output.textContent = counts[Number(output.dataset.countLabel)] + copy(' 次', ' times');
      });
      root.querySelectorAll('[data-preset]').forEach(button => {
        button.setAttribute('aria-pressed', String(counts.every((count, index) => count === presets[button.dataset.preset][index])));
      });
      root.querySelector('[data-bars]').innerHTML = summary.probabilities.map((probability, index) =>
        '<div><small>' + (probability * 100).toFixed(0) + '%</small><span style="height:' + (probability * 100) +
        '%"></span><b>' + (index + 1) + '</b></div>').join('');
      root.querySelector('[data-result]').textContent = copy('次数 ', 'Count ') + summary.total + ' · ' +
        copy('均值 ', 'Mean ') + number(summary.mean) + ' · ' + copy('方差 ', 'Variance ') + number(summary.variance) +
        ' · ' + copy('分布熵 ', 'Entropy ') + number(summary.entropy) + ' bit';
    }
    root.querySelectorAll('[data-count]').forEach(input => input.addEventListener('input', () => {
      counts[Number(input.dataset.count)] = Number(input.value);
      update();
    }));
    root.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
      counts = [...presets[button.dataset.preset]];
      update();
    }));
    update();
  }

  function mountCalibration(root, titleId) {
    root.innerHTML = heading(copy('这条放过去，还是请人看看？', 'Approve it, or ask someone to review?'),
      copy('12 条虚构样本，仅用于观察取舍。分数不是概率；真实 holdout 不能拿来反复选阈值。', '12 fictional examples for exploring trade-offs. Scores are not probabilities. Do not tune repeatedly on a real holdout.'), titleId) +
      '<div class="jl-controls"><label>' + copy('样本组', 'Slice') + '<select data-slice>' +
      option('all', copy('全部', 'All')) + option('rag', 'RAG') + option('agent', 'Agent') + '</select></label>' +
      '<label>' + copy('放行阈值', 'Approval threshold') + ' <output data-threshold-label aria-label="' +
      copy('当前阈值', 'Current threshold') + '">3.5</output>' +
      '<input type="range" min="1" max="5" step="0.5" value="3.5" data-threshold aria-label="' +
      copy('放行阈值', 'Approval threshold') + '"></label></div><label class="jl-check"><input type="checkbox" data-review> ' +
      copy('距阈值 ≤ 0.5 的样本交人工复核', 'Send cases within 0.5 of the threshold to human review') + '</label>' +
      '<div class="jl-result" data-result aria-live="polite"></div><div class="jl-sample-map" data-samples></div>' +
      '<div class="jl-table-wrap"><table class="jl-confusion"><caption>' +
      copy('只统计自动判断；待复核样本不进入这个表。', 'Automatic decisions only; review cases are excluded from this table.') +
      '</caption><thead><tr><th scope="col">' + copy('人工对照 ↓ / 规则 →', 'Human ↓ / Rule →') +
      '</th><th scope="col">' + copy('放行', 'Approve') + '</th><th scope="col">' + copy('拦截', 'Reject') +
      '</th></tr></thead><tbody><tr><th scope="row">' + copy('应通过', 'Should pass') +
      '</th><td data-count-matrix="tp"></td><td data-count-matrix="fn" class="jl-mistake"></td></tr><tr><th scope="row">' +
      copy('不应通过', 'Should fail') + '</th><td data-count-matrix="fp" class="jl-mistake"></td><td data-count-matrix="tn"></td>' +
      '</tr></tbody></table></div>' +
      '<details><summary>' + copy('查看逐条判断（含人工标签）', 'Inspect decisions and human labels') +
      '</summary><div class="jl-table-wrap"><table><caption>' + copy('虚构校准数据', 'Fictional calibration data') +
      '</caption><thead><tr><th>ID</th><th>' + copy('分数', 'Score') + '</th><th>' + copy('人工', 'Human') +
      '</th><th>' + copy('规则判断', 'Rule decision') + '</th></tr></thead><tbody data-rows></tbody></table></div></details>';
    function update() {
      const slice = root.querySelector('[data-slice]').value;
      const threshold = Number(root.querySelector('[data-threshold]').value);
      const band = root.querySelector('[data-review]').checked ? .5 : 0;
      const rows = calibrationRows.filter(row => slice === 'all' || row.slice === slice);
      const result = calibrate(rows, threshold, band);
      root.querySelector('[data-threshold-label]').textContent = threshold.toFixed(1);
      root.querySelector('[data-result]').innerHTML = '<strong>' + copy('自动判断 ', 'Automatic decisions ') +
        result.decided + '/' + result.total + ' · ' + copy('待复核 ', 'Review ') + result.review +
        '</strong><p>' + copy('覆盖率 ', 'Coverage ') + percent(result.coverage) + ' · ' +
        copy('已判定样本准确率 ', 'Accuracy on decided cases ') + percent(result.accuracy) +
        '</p><p>' + copy('放行样本中的错误占比 ', 'Error among approvals ') + percent(result.approvalError) + '</p>';
      const labels = [
        ['tp', copy('正确放行', 'Correct approval')], ['fp', copy('错误放行', 'False approval')],
        ['fn', copy('错误拦截', 'False rejection')], ['tn', copy('正确拦截', 'Correct rejection')]
      ];
      labels.forEach(([key, label]) => {
        root.querySelector('[data-count-matrix="' + key + '"]').innerHTML = '<b>' + result.matrix[key] + '</b><span>' + label + '</span>';
      });
      root.querySelector('[data-samples]').innerHTML = '<p class="jl-caption">' +
        copy('每一格是一条样本：✓ 与人工一致，× 与人工不同，? 待复核。格子里的数是虚构评分。', 'Each tile is one case: ✓ agrees with the human, × disagrees, ? awaits review. Numbers are fictional scores.') +
        '</p><div class="jl-samples">' + result.rows.map(row => {
          const review = row.verdict === 'review';
          const correct = row.verdict === row.human;
          const symbol = review ? '?' : correct ? '✓' : '×';
          const label = row.id + ' · ' + copy('评分 ', 'Score ') + row.score.toFixed(1) + ' · ' +
            copy('人工：', 'Human: ') + verdictLabel(row.human) + ' · ' + copy('规则：', 'Rule: ') + verdictLabel(row.verdict);
          return '<div class="jl-sample" data-outcome="' + (review ? 'review' : correct ? 'correct' : 'wrong') +
            '" role="img" aria-label="' + escapeText(label) + '" title="' + escapeText(label) +
            '"><span>' + row.id + '</span><b>' + symbol + '</b><small>' + row.score.toFixed(1) + '</small></div>';
        }).join('') + '</div>';
      root.querySelector('[data-rows]').innerHTML = result.rows.map(row => '<tr><th scope="row">' + row.id +
        '</th><td>' + row.score.toFixed(1) + '</td><td>' + verdictLabel(row.human) +
        '</td><td>' + verdictLabel(row.verdict) + '</td></tr>').join('');
    }
    root.querySelector('[data-slice]').addEventListener('change', update);
    root.querySelector('[data-threshold]').addEventListener('input', update);
    root.querySelector('[data-review]').addEventListener('change', update);
    update();
  }

  const mounts = {walkthrough: mountWalkthrough, evidence: mountEvidence, pairwise: mountPairwise, distribution: mountDistribution, calibration: mountCalibration};
  document.querySelectorAll('[data-judge-lab]').forEach((root, index) => {
    const mount = mounts[root.dataset.judgeLab];
    if (!mount) return;
    const titleId = 'judge-lab-title-' + index;
    root.classList.add('judge-lab');
    root.setAttribute('role', 'group');
    root.setAttribute('aria-labelledby', titleId);
    mount(root, titleId);
  });
}());
