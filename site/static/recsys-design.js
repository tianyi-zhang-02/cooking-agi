(function () {
  'use strict';

  const stages = {
    request: {
      label: ['接到请求', 'Request'],
      input: ['当前会话、用户上下文、权限、剩余时间。', 'Session, user context, permissions, and remaining time.'],
      output: ['本次查询表示与统一的 deadline。', 'A query representation and a shared deadline.'],
      why: ['只有请求发生后才知道的信息，不能全靠昨天的缓存。', 'Yesterday’s cache cannot contain information that only arrives with this request.'],
      options: ['现算用户向量，或读兼容的缓存；缓存省计算，但会晚一步知道兴趣变化。', 'Encode now or reuse a compatible cached query. Caching saves work but delays interest updates.'],
      check: ['比较相同请求的实时与缓存表示，记录新鲜度和尾延迟。', 'Compare fresh and cached queries on the same requests; measure freshness and tail latency.'],
      lesson: '06-why-two-towers'
    },
    retrieve: {
      label: ['找候选', 'Retrieve'],
      input: ['查询表示、关系线索、可检索的内容集合。', 'Query vectors, relationship signals, and a searchable catalog.'],
      output: ['候选 ID、来源、来源内分数，以及成功或超时状态。', 'Candidate IDs, provenance, source-local scores, and completion status.'],
      why: ['先便宜地找一批；昂贵的模型没必要看整个内容库。', 'Find a manageable pool cheaply before spending more computation on each candidate.'],
      options: ['双塔、关系来源、关键词、新内容来源可以并用；每加一路都要付查询和合并成本。', 'Combine dual encoders, relationships, keywords, or fresh-content sources. Each adds query and merging work.'],
      check: ['固定总预算，看独有相关候选；把来源超时与正常空结果分开。', 'Fix the total budget and inspect unique relevant candidates. Separate timeouts from successful empty results.'],
      lesson: '02-candidate-retrieval'
    },
    features: {
      label: ['去重与补特征', 'Merge & hydrate'],
      input: ['各路候选、资格信息和所需特征。', 'Source results, eligibility information, and required features.'],
      output: ['去重后的候选、全部来源、特征值与缺失标记。', 'Deduplicated candidates with provenance, feature values, and missingness.'],
      why: ['同一条内容别算两次；默认值也别装成真实观测。', 'Do not count a post twice or confuse a default with an observed feature.'],
      options: ['预聚合特征较快，在线读取较新；两条路径必须保持字段含义一致。', 'Preaggregation is cheaper; request-time reads can be fresher. Both paths must agree on feature meaning.'],
      check: ['检查重复 ID、过滤后不足、特征缺失与时间戳。', 'Test duplicate IDs, post-filter shortfalls, missing features, and timestamps.'],
      lesson: '08-serving-lifecycle'
    },
    rank: {
      label: ['逐条打分', 'Rank'],
      input: ['较小候选集及用户—内容特征。', 'A smaller candidate pool with user–item features.'],
      output: ['每条候选的行为预测和排序分。', 'Per-candidate behavior predictions and ranking scores.'],
      why: ['这里才有预算细看用户和内容的交互。', 'The smaller pool makes richer user–item interactions affordable.'],
      options: ['线性或树模型适合基线；多任务模型和联合编码能表达更多，也更依赖标签与预算。', 'Linear or tree models make useful baselines. Multi-task and joint models add capacity, label demands, and cost.'],
      check: ['固定候选池比较；区分训练 loss 权重与服务时的目标权重。', 'Compare on a fixed pool. Distinguish training-loss weights from serving-objective weights.'],
      lesson: '03-ranking-and-diversity'
    },
    list: {
      label: ['组成一屏', 'Select a list'],
      input: ['单条分数、作者与主题信息、列表限制。', 'Item scores, author/topic information, and list constraints.'],
      output: ['有顺序的结果，以及没选中的原因。', 'An ordered result with reasons for removals.'],
      why: ['3 条各自不错的帖子，也可能讲的是同一件事。', 'Three individually useful posts may still repeat the same thing.'],
      options: ['作者上限、相似度惩罚或约束优化；不要把内容分散自动当成更满意。', 'Author caps, similarity penalties, or constrained optimization. More spread does not automatically mean more satisfaction.'],
      check: ['同时看相关性、重复和覆盖；返回前再次确认关键资格。', 'Check relevance, repetition, and coverage together; recheck critical eligibility before responding.'],
      lesson: '03-ranking-and-diversity'
    },
    feedback: {
      label: ['观察反馈', 'Observe feedback'],
      input: ['返回内容、实际曝光、位置与后续行为。', 'Returned items, actual exposure, position, and later behavior.'],
      output: ['带时间窗和可观测性说明的事件。', 'Events with an observation window and visibility context.'],
      why: ['返回过不代表看见过；没点击也不自动代表拒绝。', 'Returned is not necessarily seen; no click is not automatically rejection.'],
      options: ['只记录诊断和训练所需的数据，限制访问与保留期限。', 'Record only what diagnosis and training need, with access and retention limits.'],
      check: ['核对曝光埋点、延迟反馈和采样；先确认这把尺子在量什么。', 'Check exposure logging, delayed feedback, and sampling before interpreting a score.'],
      lesson: '05-evaluation-lab'
    },
    content: {
      label: ['内容变化', 'Content changes'],
      input: ['发布、编辑、删除事件。', 'Publication, edit, and deletion events.'],
      output: ['带内容版本的更新任务。', 'An update task tied to a content version.'],
      why: ['同一个 ID 的正文可能已经不是上次编码的那份。', 'The same ID may now contain different content.'],
      options: ['定期批处理简单；事件驱动更新较及时，但要处理积压、重复和乱序。', 'Periodic batches are simpler; event-driven updates can be fresher but need backlog and ordering controls.'],
      check: ['编辑后是否及时可搜？删除是否绕过慢速回填及时生效？', 'Can edits become searchable promptly? Can deletion take effect without waiting for a slow backfill?'],
      lesson: '08-serving-lifecycle'
    },
    encode: {
      label: ['编码内容', 'Encode items'],
      input: ['可用文本或图像、预处理配置、内容编码器。', 'Available text/images, preprocessing, and an item encoder.'],
      output: ['与模型和内容版本绑定的向量。', 'Vectors tied to model and content versions.'],
      why: ['提前算一次，才能让许多请求复用同一份表示。', 'Compute once so many requests can reuse the representation.'],
      options: ['先做文本基线；只有图片补上任务所需信息时，再考虑多模态及重算成本。', 'Start with text; add images when they supply task-relevant information worth the refresh cost.'],
      check: ['缺图、截断、预处理变化和批量推理的一致性。', 'Test missing images, truncation, preprocessing changes, and batch-inference parity.'],
      lesson: '07-component-choices'
    },
    index: {
      label: ['维护索引', 'Maintain an index'],
      input: ['向量、ID、打分方式及更新事件。', 'Vectors, IDs, a scoring metric, and updates.'],
      output: ['可查询、可追溯版本的搜索结构。', 'A searchable structure with traceable versions.'],
      why: ['模型负责定义分数，索引负责把搜索做得可承受。', 'The model defines scores; the index makes searching affordable.'],
      options: ['Flat 是精确基线；HNSW 和 IVF/PQ 用不同的内存、构建与近似取舍减少搜索工作。', 'Flat gives an exact reference. HNSW and IVF/PQ trade memory, build work, and approximation for search efficiency.'],
      check: ['用相同向量对照精确 top-k；另外测更新、过滤、内存和延迟。', 'Compare against exact top-k on identical vectors; separately measure updates, filtering, memory, and latency.'],
      lesson: '07-component-choices'
    },
    release: {
      label: ['配套发布', 'Release together'],
      input: ['用户塔、内容塔、索引、schema 和验证结果。', 'Query/item towers, an index, schemas, and validation results.'],
      output: ['一次请求可固定使用的兼容 release。', 'A compatible release that each request can pin.'],
      why: ['维度相同不代表同一个坐标系，单独换一半可能静悄悄地出错。', 'Equal dimensions do not imply compatible coordinates; swapping half a bundle can fail silently.'],
      options: ['影子流量后逐步切换，并保留旧版；安全感来自可回滚，不是一次性全量替换。', 'Shadow, then roll out gradually while retaining the old bundle for rollback.'],
      check: ['混用版本要被拒绝；回滚后检查缓存、索引和删除规则。', 'Reject incompatible versions; test caches, indexes, and deletion rules after rollback.'],
      lesson: '08-serving-lifecycle'
    },
    examples: {
      label: ['定义样本', 'Define examples'],
      input: ['带曝光信息的反馈和当时可得的特征。', 'Feedback with exposure context and features available at the time.'],
      output: ['训练样本、任务 mask、采样与标签说明。', 'Examples, task masks, and sampling/label definitions.'],
      why: ['模型学到什么，先取决于哪些行为被你写成了“对”和“错”。', 'What the model learns depends first on what the dataset calls right or wrong.'],
      options: ['显式行为较稀，弱信号覆盖较广；采样负例不是这个人的明确拒绝。', 'Explicit actions are sparse; weak signals cover more cases. Sampled negatives are not explicit rejections.'],
      check: ['按时间切分，检查假负例、缺失标签和曝光偏差。', 'Use temporal splits and inspect false negatives, missing labels, and exposure bias.'],
      lesson: '05-evaluation-lab'
    },
    train: {
      label: ['学习表示与目标', 'Learn representations'],
      input: ['样本、模型结构和清楚定义的 loss。', 'Examples, architecture, and explicitly defined losses.'],
      output: ['新模型参数与可复现配置。', 'New model parameters and reproducible configuration.'],
      why: ['只刷新内容向量不等于训练；训练会改变整个表示空间。', 'Refreshing item vectors is not training; training can change the representation space itself.'],
      options: ['召回可用对比目标，排序可用多任务目标；哪个任务缺信号比 head 数量更重要。', 'Retrieval can use contrastive objectives and ranking multiple tasks. Missing supervision matters more than head count.'],
      check: ['固定数据与预算做消融，同时看损失实现和 train–serve parity。', 'Ablate with fixed data and budget; check loss correctness and train–serve parity.'],
      lesson: '06-why-two-towers'
    },
    evaluate: {
      label: ['判断是否值得发布', 'Evaluate a release'],
      input: ['旧版、新版、独立评估集与资源预算。', 'Old/new releases, an independent evaluation set, and a resource budget.'],
      output: ['支持上线、继续实验或停止的证据。', 'Evidence to deploy, experiment further, or stop.'],
      why: ['整体分数上涨，仍可能掩盖某类用户或内容退步。', 'An aggregate gain can hide regressions for a user or content group.'],
      options: ['离线控制变量方便归因；受控在线实验观察真实行为，二者不能互相替代。', 'Offline controls help attribution; controlled deployment observes real behavior. Neither replaces the other.'],
      check: ['记录候选池、分母和切片；模拟结果不当成真实用户收益。', 'Record candidate pools, denominators, and slices. Simulation is not proof of real user benefit.'],
      lesson: '05-evaluation-lab'
    }
  };
  const lanes = {
    serve: {label: ['一次刷新 · online', 'One refresh · online'], nodes: ['request', 'retrieve', 'features', 'rank', 'list', 'feedback']},
    update: {label: ['内容变化 · batch / nearline', 'Content changes · batch / nearline'], nodes: ['content', 'encode', 'index', 'release']},
    learn: {label: ['模型学习 · offline', 'Model learning · offline'], nodes: ['feedback', 'examples', 'train', 'evaluate', 'release']}
  };
  const scenarios = {
    normal: ['正常请求', 'Normal request'],
    timeout: ['向量来源超时', 'Dense source times out'],
    mismatch: ['查询与索引不兼容', 'Query/index mismatch'],
    unseen: ['返回了，但未曝光', 'Returned but not seen']
  };

  function traceRequest(scenario) {
    if (!Object.hasOwn(scenarios, scenario)) throw new Error('Unknown scenario');
    const denseStatus = scenario === 'timeout' ? 'timeout' : scenario === 'mismatch' ? 'incompatible' : 'ok';
    const graph = ['A', 'B'];
    const dense = denseStatus === 'ok' ? ['B', 'C', 'D'] : [];
    const merged = [...new Set([...graph, ...dense])];
    const eligible = merged.filter(identifier => identifier !== 'B');
    const scores = {A: .7, C: .9, D: .8};
    const ranked = [...eligible].sort((left, right) => scores[right] - scores[left]);
    const returned = ranked.slice(0, 2);
    return {graph, dense, denseStatus, merged, eligible, ranked, returned,
      exposed: scenario === 'unseen' ? [] : [...returned],
      removed: merged.filter(identifier => identifier === 'B')};
  }

  const vectorItems = [0, 14, 32, 43, 48, 60, 77, 90].map((angle, index) => ({
    id: String.fromCharCode(65 + index), angle,
    vector: [Math.cos(angle * Math.PI / 180), Math.sin(angle * Math.PI / 180)]
  }));

  function vectorSearch(weight, budget, mode) {
    if (!Number.isFinite(weight) || weight < 0 || weight > 1 ||
        !Number.isInteger(budget) || budget < 2 || budget > 8 || !['mixed', 'split'].includes(mode)) {
      throw new Error('Invalid vector controls');
    }
    const secondIntent = [Math.cos(70 * Math.PI / 180), Math.sin(70 * Math.PI / 180)];
    const combined = [weight + (1 - weight) * secondIntent[0], (1 - weight) * secondIntent[1]];
    const length = Math.hypot(...combined);
    const mixed = combined.map(value => value / length);
    const firstBudget = Math.max(1, Math.min(budget - 1, Math.round(weight * budget)));
    const queries = mode === 'mixed' ? [{vector: mixed, limit: budget}] : [
      {vector: [1, 0], limit: firstBudget}, {vector: secondIntent, limit: budget - firstBudget}
    ];
    const results = queries.map(query => vectorItems.map(item => ({
      id: item.id, score: query.vector[0] * item.vector[0] + query.vector[1] * item.vector[1]
    })).sort((left, right) => right.score - left.score || left.id.localeCompare(right.id)).slice(0, query.limit));
    const unique = [...new Set(results.flat().map(item => item.id))];
    return {queries, results, unique, duplicates: budget - unique.length, comparisons: vectorItems.length * queries.length};
  }

  const evaluationCases = {
    small: {label: ['小候选池', 'Small pool'], ranked: ['P', 'N1', 'N2'], positives: ['P']},
    larger: {label: ['新增高分负例', 'Add high-scoring negatives'], ranked: ['N3', 'N4', 'P', 'N1', 'N2'], positives: ['P']},
    multi: {label: ['多个已知正例', 'Multiple known positives'], ranked: ['P1', 'P2', 'N1', 'P3', 'P4'], positives: ['P1', 'P2', 'P3', 'P4']},
    none: {label: ['没有已知正例', 'No known positives'], ranked: ['N1', 'N2', 'N3'], positives: []}
  };

  function evaluateRecall(caseId, cutoff, denominator) {
    if (!Object.hasOwn(evaluationCases, caseId) || !Number.isInteger(cutoff) || cutoff < 1 ||
        !['standard', 'capped'].includes(denominator)) throw new Error('Invalid evaluation controls');
    const example = evaluationCases[caseId];
    const hits = example.ranked.slice(0, cutoff).filter(identifier => example.positives.includes(identifier)).length;
    const divisor = denominator === 'standard' ? example.positives.length : Math.min(example.positives.length, cutoff);
    return {hits, divisor, score: divisor ? hits / divisor : null};
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {stages, lanes, scenarios, traceRequest, vectorItems, vectorSearch, evaluationCases, evaluateRecall};
  }
  if (typeof document === 'undefined') return;

  const translate = (pair, english) => pair[english ? 1 : 0];
  const text = (zh, en, english) => translate([zh, en], english);
  const shell = (title, subtitle, english) => `<div class="rd-heading"><strong>${title}</strong><button type="button" data-language>${text('English 对照', '中文对照', english)}</button></div><p class="rd-caption">${subtitle}</p>`;
  const joinIds = identifiers => identifiers.length ? identifiers.join(' · ') : '∅';
  const option = (value, label, selected) => `<option value="${value}"${selected ? ' selected' : ''}>${label}</option>`;

  document.querySelectorAll('[data-recsys-architecture]').forEach((root, instance) => {
    let english = document.documentElement.lang.startsWith('en');
    let lane = root.dataset.lane || 'serve';
    if (!Object.hasOwn(lanes, lane)) lane = 'serve';
    let stageIndex = 0;
    let scenario = 'normal';
    const panelId = `rd-panel-${instance}`;
    root.classList.add('recsys-design');

    function drawPanel() {
      const current = lanes[lane].nodes[stageIndex];
      const stage = stages[current];
      root.querySelectorAll('[data-stage]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.stage === current)));
      root.querySelector('[data-detail]').innerHTML = `<div class="rd-detail-head"><span>${String(stageIndex + 1).padStart(2, '0')} / ${lanes[lane].nodes.length}</span><h3>${translate(stage.label, english)}</h3></div><dl class="rd-details">${[
        ['input', ['拿到什么', 'Input']], ['output', ['交出什么', 'Output']], ['why', ['为什么放在这里', 'Why here']],
        ['options', ['还能怎么选 · trade-off', 'Alternatives · trade-off']], ['check', ['怎么验证', 'How to check']]
      ].map(([key, label]) => `<div><dt>${translate(label, english)}</dt><dd>${translate(stage[key], english)}</dd></div>`).join('')}</dl><a class="rd-read" href="${stage.lesson}${english ? '.en' : ''}.html">${text('继续读这个模块', 'Read the module', english)} ↗</a>`;
      root.querySelector('[data-previous]').disabled = stageIndex === 0;
      root.querySelector('[data-next]').disabled = stageIndex === lanes[lane].nodes.length - 1;
      root.querySelector('[data-status]').textContent = text(`当前：${translate(stage.label, english)}，第 ${stageIndex + 1} 步。`, `Now: ${translate(stage.label, english)}, step ${stageIndex + 1}.`, english);
    }

    function drawTrace() {
      const result = traceRequest(scenario);
      const label = result.denseStatus === 'ok' ? 'OK' : result.denseStatus === 'timeout' ? 'TIMEOUT' : 'INCOMPATIBLE';
      root.querySelector('[data-trace]').innerHTML = `<div class="rd-trace"><div><small>${text('来源返回', 'Source results', english)}</small><b>Graph: ${joinIds(result.graph)}</b><span>Dense: ${joinIds(result.dense)} · ${label}</span></div><div><small>${text('去重 / 删除 B', 'Deduplicate / remove B', english)}</small><b>${joinIds(result.eligible)}</b><span>${text('已合并', 'Merged', english)}: ${joinIds(result.merged)}</span></div><div><small>${text('打分后取 2 条', 'Rank, then take 2', english)}</small><b>${joinIds(result.returned)}</b><span>${text('曝光', 'Exposed', english)}: ${joinIds(result.exposed)}</span></div></div><p class="rd-caption">${text(
        '手写玩具请求：Graph 返回 A/B，Dense 返回 B/C/D；B 已删除。基础分 C=0.9、D=0.8、A=0.7；不做补满。',
        'Hand-written toy request: Graph returns A/B; Dense returns B/C/D. B is deleted. Scores: C=0.9, D=0.8, A=0.7. No backfill.', english)}</p><p class="rd-observation" role="status">${translate({
          normal: ['这次留下 C、D。重复候选只算一次，删除规则不因分数高而失效。', 'C and D survive. Duplicates count once; a high score never overrides deletion.'],
          timeout: ['只剩 A，不强行补满。空列表是超时造成的，不能据此说 Dense 找不到相关内容。', 'Only A remains; we do not force-fill. Dense timed out, so this is not evidence that it found nothing relevant.'],
          mismatch: ['拒绝用不兼容的向量空间查询，保留可用的 Graph 来源；不要默默计算错误分数。', 'Reject the incompatible vector-space query and keep the usable Graph source rather than silently scoring mismatched vectors.'],
          unseen: ['返回 C、D，但没有实际曝光。此时没有点击，不能写成 2 个已确认负例。', 'C and D were returned but not exposed. No click is not two confirmed negative labels.']
        }[scenario], english)}</p>`;
    }

    function render() {
      root.lang = english ? 'en' : 'zh-CN';
      root.innerHTML = shell(text('把架构拆成 3 条线', 'Read the architecture in 3 paths', english), text('点一个环节看输入、输出、替代方案和验证方法。示例不是任何公司的生产架构。', 'Select a stage for its inputs, outputs, alternatives, and checks. This is a teaching design, not a production blueprint.', english), english) +
        `<div class="rd-tabs" role="group" aria-label="${text('架构路径', 'Architecture paths', english)}">${Object.entries(lanes).map(([key, value]) => `<button type="button" data-lane="${key}" aria-pressed="${key === lane}">${translate(value.label, english)}</button>`).join('')}</div><div class="rd-path" role="group" aria-label="${text('链路环节', 'Pipeline stages', english)}">${lanes[lane].nodes.map((identifier, index) => `<button type="button" data-stage="${identifier}" aria-pressed="false" aria-controls="${panelId}"><small>${String(index + 1).padStart(2, '0')}</small>${translate(stages[identifier].label, english)}</button>`).join('')}</div><section id="${panelId}" data-detail></section><div class="rd-controls"><button type="button" data-previous>${text('← 上一步', '← Previous', english)}</button><span data-status role="status"></span><button type="button" data-next>${text('下一步 →', 'Next →', english)}</button></div>` +
        (lane === 'serve' ? `<div class="rd-experiment"><label>${text('改一个条件，再看结果', 'Change one condition and trace the result', english)}<select data-scenario>${Object.entries(scenarios).map(([key, value]) => option(key, translate(value, english), key === scenario)).join('')}</select></label><div data-trace></div></div>` : `<p class="rd-observation">${text(lane === 'update' ? '内容更新会重新计算输入的表示；模型训练会改变表示函数。两者都可能影响索引，但不是同一件事。' : '训练不会直接接到每一次刷新上。经过评估的模型要先与索引、特征等部件配套发布。', lane === 'update' ? 'Content updates recompute representations; training changes the function producing them. Both can affect the index, but they are not the same operation.' : 'Training does not sit inside every refresh. An evaluated model must be released with compatible indexes and features.', english)}</p>`);
      drawPanel();
      if (lane === 'serve') drawTrace();
    }
    root.addEventListener('click', event => {
      const button = event.target.closest('button');
      if (!button || !root.contains(button)) return;
      if (button.hasAttribute('data-language')) {
        english = !english; render(); root.querySelector('[data-language]').focus();
      } else if (button.dataset.lane) {
        lane = button.dataset.lane; stageIndex = 0; render(); root.querySelector(`[data-lane="${lane}"]`).focus();
      } else if (button.dataset.stage) {
        stageIndex = lanes[lane].nodes.indexOf(button.dataset.stage); drawPanel();
      } else if (button.hasAttribute('data-next') || button.hasAttribute('data-previous')) {
        stageIndex += button.hasAttribute('data-next') ? 1 : -1; drawPanel();
        if (button.disabled) root.querySelector(`[data-stage="${lanes[lane].nodes[stageIndex]}"]`).focus();
      }
    });
    root.addEventListener('change', event => {
      if (event.target.matches('[data-scenario]')) { scenario = event.target.value; drawTrace(); }
    });
    render();
  });

  document.querySelectorAll('[data-recsys-vectors]').forEach((root, instance) => {
    let english = document.documentElement.lang.startsWith('en');
    let weight = .5;
    let budget = 4;
    let mode = 'mixed';
    root.classList.add('recsys-design');
    function draw() {
      const result = vectorSearch(weight, budget, mode);
      root.querySelector('[data-weight-value]').textContent = weight.toFixed(2);
      root.querySelector('[data-budget-value]').textContent = budget;
      const chart = root.querySelector('[data-vector-chart]');
      const origin = [42, 262];
      const point = vector => [origin[0] + 210 * vector[0], origin[1] - 210 * vector[1]];
      chart.innerHTML = `<svg viewBox="0 0 310 300" role="img" aria-labelledby="rd-vector-title-${instance}"><title id="rd-vector-title-${instance}">${text('二维单位向量：实心点是选中候选，射线是查询方向', '2D unit vectors: filled dots are retrieved items; rays show queries', english)}</title><path class="rd-axis" d="M42 30V262H283 M42 52 A210 210 0 0 1 252 262" fill="none"/><text x="200" y="286">${text('维度 1', 'Dimension 1', english)}</text><text x="10" y="22">${text('维度 2', 'Dimension 2', english)}</text>${result.queries.map((query, index) => {
        const target = point(query.vector);
        return `<path class="rd-query rd-query-${index}" d="M42 262 L${target[0]} ${target[1]}"/><circle class="rd-tip" cx="${target[0]}" cy="${target[1]}" r="3"/>`;
      }).join('')}${vectorItems.map(item => {
        const location = point(item.vector);
        return `<g class="rd-vector${result.unique.includes(item.id) ? ' is-picked' : ''}"><circle cx="${location[0]}" cy="${location[1]}" r="6"/><text x="${location[0] + 9}" y="${location[1] - 6}">${item.id}</text></g>`;
      }).join('')}</svg>`;
      root.querySelector('[data-vector-result]').innerHTML = `<h3>${text('实际找到什么', 'What was retrieved', english)}</h3>${result.results.map((items, index) => `<p><b>Q${index + 1}</b> · ${items.map(item => `${item.id} (${item.score.toFixed(2)})`).join(' · ')}</p>`).join('')}<p><b>${text('去重后的集合', 'Deduplicated set', english)}</b><br>${joinIds(result.unique)}</p><p class="rd-caption">${text('合并集合的顺序不代表最终排序。', 'Set order is not the final ranking.', english)}</p>`;
      root.querySelector('[data-vector-status]').textContent = text(
        `${result.queries.length} 次查询 · 总返回槽位 ${budget} · 独立候选 ${result.unique.length} · 重复 ${result.duplicates} · ${result.comparisons} 次二维点积`,
        `${result.queries.length} ${result.queries.length === 1 ? 'query' : 'queries'} · ${budget} total slots · ${result.unique.length} unique candidates · ${result.duplicates} ${result.duplicates === 1 ? 'duplicate' : 'duplicates'} · ${result.comparisons} 2D dot products`, english);
    }
    function render() {
      root.lang = english ? 'en' : 'zh-CN';
      root.innerHTML = shell(text('同一个人，两种兴趣，怎么查？', 'One person, two interests: how do you search?', english), text('8 个手写单位向量，做精确 cosine 检索。没有训练，也没有真实相关性标签。', '8 hand-written unit vectors with exact cosine search. No training or real relevance labels.', english), english) + `<div class="rd-inputs"><label>${text('查询方式', 'Query mode', english)}<select data-vector-mode>${option('mixed', text('先混成一条向量', 'One mixed query', english), mode === 'mixed')}${option('split', text('分别查，再合并', 'Separate queries', english), mode === 'split')}</select></label><label>${text('兴趣 1 的权重', 'Weight of interest 1', english)} <output data-weight-value></output><input type="range" data-vector-weight aria-label="${text('兴趣 1 的权重', 'Weight of interest 1', english)}" min="0" max="1" step="0.05" value="${weight}"></label><label>${text('总返回槽位 K', 'Total result slots K', english)} <output data-budget-value></output><input type="range" data-vector-budget aria-label="${text('总返回槽位 K', 'Total result slots K', english)}" min="2" max="8" step="1" value="${budget}"></label></div><div class="rd-vector-layout"><div data-vector-chart></div><div data-vector-result></div></div><p class="rd-observation" data-vector-status role="status"></p><p class="rd-caption">${text('两条兴趣向量分别指向 0° 和 70°。单查询将加权和归一化；分开查时按权重分配 K，每路至少 1 个槽位，去重后不补满。固定槽位不等于固定计算，两路精确搜索会扫描两遍。', 'The two intent vectors point at 0° and 70°. One query normalizes their weighted sum. Split search allocates K by weight, with at least 1 slot per query and no backfill after deduplication. Equal slots do not mean equal computation: two exact queries scan twice.', english)}</p><button type="button" data-vector-reset>${text('重置实验', 'Reset experiment', english)}</button>`;
      draw();
    }
    root.addEventListener('input', event => {
      if (event.target.matches('[data-vector-weight]')) weight = Number(event.target.value);
      else if (event.target.matches('[data-vector-budget]')) budget = Number(event.target.value);
      else return;
      draw();
    });
    root.addEventListener('change', event => {
      if (event.target.matches('[data-vector-mode]')) { mode = event.target.value; draw(); }
    });
    root.addEventListener('click', event => {
      const button = event.target.closest('button');
      if (!button) return;
      if (button.hasAttribute('data-language')) { english = !english; render(); root.querySelector('[data-language]').focus(); }
      if (button.hasAttribute('data-vector-reset')) { weight = .5; budget = 4; mode = 'mixed'; render(); root.querySelector('[data-vector-reset]').focus(); }
    });
    render();
  });

  document.querySelectorAll('[data-recsys-evaluation]').forEach(root => {
    let english = document.documentElement.lang.startsWith('en');
    let caseId = 'small';
    let cutoff = 2;
    let denominator = 'standard';
    root.classList.add('recsys-design');
    function draw() {
      const example = evaluationCases[caseId];
      const result = evaluateRecall(caseId, cutoff, denominator);
      root.querySelector('[data-cutoff-value]').textContent = cutoff;
      root.querySelector('[data-eval-ranking]').innerHTML = example.ranked.map((identifier, index) => `<span class="rd-ranked${index < cutoff ? ' is-included' : ''}"><small>${index + 1}</small><b>${identifier}</b><span>${caseId === 'none' ? '?' : example.positives.includes(identifier) ? '+' : '−'}</span></span>`).join('');
      root.querySelector('[data-eval-score]').textContent = result.score === null ? text('未定义：没有已知正例，单独报告，不当作 0 或 1。', 'Undefined: no known positives. Report separately, not as 0 or 1.', english) : `${text(denominator === 'standard' ? '标准 Recall' : '封顶分母的归一化分数', denominator === 'standard' ? 'Standard Recall' : 'Capped-denominator score', english)}@${cutoff} = ${result.hits} / ${result.divisor} = ${result.score.toFixed(2)}`;
      root.querySelector('[data-eval-observation]').textContent = text(
        caseId === 'small' || caseId === 'larger' ? '在前两种候选池之间切换：原有内容的分数没变，但 P 的名次变了。别把候选池变化当成模型变化。' : caseId === 'multi' ? 'K=2 时，2 个命中除以 4 是 0.5，除以 min(4,2) 是 1。名字都叫 Recall，含义也未必一样。' : '没有正例不是“全部答对”，也不说明这些内容都不相关。真实未标注数据更不能直接当作负例。',
        caseId === 'small' || caseId === 'larger' ? 'Switch between the first two pools: existing scores are unchanged, but P changes rank. A pool change is not a model change.' : caseId === 'multi' ? 'At K=2, two hits divided by 4 gives 0.5; divided by min(4,2) gives 1. The same metric name can hide different definitions.' : 'No positives is not a perfect score, nor proof that all items are irrelevant. Unlabeled real data should not automatically be treated as negative.', english);
    }
    function render() {
      root.lang = english ? 'en' : 'zh-CN';
      root.innerHTML = shell(text('模型没改，为什么分数变了？', 'Same model—why did the score change?', english), text('前两组只改候选池；后两组专门检查分母。符号 + / − 是手写教学标签，不是行为推断。', 'The first two cases change only the pool; the other two test denominators. + / − are invented teaching labels, not inferred behavior.', english), english) + `<div class="rd-inputs"><label>${text('教学样本', 'Teaching case', english)}<select data-eval-case>${Object.entries(evaluationCases).map(([key, value]) => option(key, translate(value.label, english), key === caseId)).join('')}</select></label><label>${text('分母定义', 'Denominator', english)}<select data-eval-denominator>${option('standard', '|P| · Standard Recall', denominator === 'standard')}${option('capped', 'min(|P|, K)', denominator === 'capped')}</select></label><label>Top-K <output data-cutoff-value></output><input data-eval-cutoff type="range" aria-label="Top-K" min="1" max="5" value="${cutoff}" step="1"></label></div><div class="rd-ranking" data-eval-ranking></div><p class="rd-caption">${text('实线框内是 Top-K；位置从左到右。', 'Solid borders mark Top-K, ordered left to right.', english)}</p><p class="rd-observation" data-eval-score role="status"></p><p data-eval-observation></p>`;
      draw();
    }
    root.addEventListener('change', event => {
      if (event.target.matches('[data-eval-case]')) caseId = event.target.value;
      else if (event.target.matches('[data-eval-denominator]')) denominator = event.target.value;
      else return;
      draw();
    });
    root.addEventListener('input', event => {
      if (event.target.matches('[data-eval-cutoff]')) { cutoff = Number(event.target.value); draw(); }
    });
    root.addEventListener('click', event => {
      if (event.target.closest('[data-language]')) { english = !english; render(); root.querySelector('[data-language]').focus(); }
    });
    render();
  });
})();
