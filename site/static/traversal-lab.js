(function () {
  'use strict';

  const graphs = {
    tree: {A: ['B', 'C'], B: ['D', 'E'], C: ['F'], D: [], E: [], F: []},
    cycle: {A: ['B', 'C'], B: ['D'], C: ['D'], D: ['A', 'E'], E: []},
    chain: {A: ['B'], B: ['C'], C: ['D'], D: ['E'], E: ['F'], F: []},
    wide: {A: ['B', 'C', 'D', 'E', 'F'], B: [], C: [], D: [], E: [], F: []}
  };
  const layouts = {
    tree: {A: [150, 28], B: [80, 100], C: [230, 100], D: [35, 185], E: [125, 185], F: [230, 185]},
    cycle: {A: [45, 35], B: [175, 35], C: [45, 155], D: [175, 155], E: [265, 185]},
    chain: {A: [25, 30], B: [75, 60], C: [125, 90], D: [175, 120], E: [225, 150], F: [275, 180]},
    wide: {A: [150, 35], B: [25, 175], C: [85, 175], D: [150, 175], E: [215, 175], F: [275, 175]}
  };

  function traceGraph(graph, mode, start = 'A') {
    if (!['dfs', 'bfs'].includes(mode)) throw new Error('Unknown traversal');
    const seen = new Set([start]);
    const order = [start];
    const states = [];
    const distances = {[start]: 0};
    const queue = [start];
    let queueHead = 0;
    const frames = [{node: start, next: 0}];
    const save = (current, event, target = '') => states.push({
      current, event, target, order: [...order], seen: [...seen],
      frontier: mode === 'bfs' ? queue.slice(queueHead) : frames.map(frame => frame.node),
      distances: {...distances}
    });
    save('', 'start', start);
    if (mode === 'bfs') {
      while (queueHead < queue.length) {
        const node = queue[queueHead++];
        for (const neighbor of graph[node] || []) {
          if (!seen.has(neighbor)) {
            seen.add(neighbor);
            order.push(neighbor);
            distances[neighbor] = distances[node] + 1;
            queue.push(neighbor);
          }
        }
        save(node, 'expand');
      }
    } else {
      while (frames.length) {
        const frame = frames[frames.length - 1];
        const neighbors = graph[frame.node] || [];
        if (frame.next === neighbors.length) {
          frames.pop();
          save(frame.node, 'return');
        } else {
          const neighbor = neighbors[frame.next++];
          if (seen.has(neighbor)) {
            save(frame.node, 'skip', neighbor);
          } else {
            seen.add(neighbor);
            order.push(neighbor);
            frames.push({node: neighbor, next: 0});
            save(neighbor, 'enter');
          }
        }
      }
    }
    return states;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = {graphs, traceGraph};
  if (typeof document === 'undefined') return;

  document.querySelectorAll('[data-traversal-lab]').forEach((root, instance) => {
    let language = root.dataset.language === 'en' ? 'en' : 'zh';
    let selected = 'tree';
    let step = 0;
    const copy = (zh, en) => language === 'zh' ? zh : en;

    function diagram(graph, state, mode) {
      const nodes = Object.keys(graph);
      const positions = layouts[selected];
      const arrow = `traversal-arrow-${instance}-${mode}`;
      const edges = Object.entries(graph).flatMap(([source, neighbors]) => neighbors.map(target => {
        const [startX, startY] = positions[source];
        const [endX, endY] = positions[target];
        const length = Math.hypot(endX - startX, endY - startY);
        const unitX = (endX - startX) / length;
        const unitY = (endY - startY) / length;
        return `<line x1="${startX + unitX * 17}" y1="${startY + unitY * 17}" x2="${endX - unitX * 21}" y2="${endY - unitY * 21}" marker-end="url(#${arrow})"/>`;
      })).join('');
      const circles = nodes.map(node => {
        const [centerX, centerY] = positions[node];
        const status = state.current === node ? 'current' : state.seen.includes(node) ? 'seen' : 'unseen';
        return `<g class="traversal-node ${status}"><circle cx="${centerX}" cy="${centerY}" r="16"/><text x="${centerX}" y="${centerY + 5}">${node}</text></g>`;
      }).join('');
      return `<svg viewBox="0 0 300 220" role="img" aria-label="${mode.toUpperCase()} ${copy('图；当前节点', 'graph; current node')} ${state.current || '—'}"><defs><marker id="${arrow}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z"/></marker></defs><g class="traversal-edges">${edges}</g>${circles}</svg>`;
    }

    function draw() {
      const graph = graphs[selected];
      const traces = {dfs: traceGraph(graph, 'dfs'), bfs: traceGraph(graph, 'bfs')};
      const last = Math.max(traces.dfs.length, traces.bfs.length) - 1;
      step = Math.min(step, last);
      const title = `traversal-title-${instance}`;
      root.innerHTML = `<div class="traversal-heading"><strong id="${title}">${copy('同一张图，两种走法', 'Same graph, two ways through')}</strong><button type="button" data-action="language">${copy('EN', '中文')}</button></div>
        <label>${copy('图的形状', 'Graph shape')} <select data-action="graph"><option value="tree">${copy('分叉树', 'Branching tree')}</option><option value="cycle">${copy('共享路径与环', 'Shared paths and cycle')}</option><option value="chain">${copy('长链', 'Chain')}</option><option value="wide">${copy('宽树', 'Wide tree')}</option></select></label>
        <div class="traversal-columns">${['dfs', 'bfs'].map(mode => {
          const trace = traces[mode];
          const state = trace[Math.min(step, trace.length - 1)];
          const finished = step >= trace.length - 1;
          const events = {
            start: copy(`起点 ${state.target} 已登记。`, `Registered start ${state.target}.`),
            expand: copy(`处理 ${state.current}，未发现的邻居入队。`, `Process ${state.current}; queue undiscovered neighbors.`),
            enter: copy(`进入 ${state.current}，保留父节点的栈帧。`, `Enter ${state.current}; retain its parent's frame.`),
            return: copy(`${state.current} 的邻居已看完，返回。`, `Finished ${state.current}'s neighbors; return.`),
            skip: copy(`${state.target} 已登记，跳过这条边。`, `${state.target} already discovered; skip this edge.`)
          };
          return `<section class="traversal-lane"><h3>${mode.toUpperCase()} · ${mode === 'dfs' ? copy('递归 / 显式帧', 'Recursive / explicit frames') : copy('队列', 'Queue')}</h3>${diagram(graph, state, mode)}
            <p class="traversal-event">${events[state.event]}${finished ? copy(' 已完成。', ' Done.') : ''}</p>
            <dl><dt>${mode === 'dfs' ? copy('调用栈 · 右侧是栈顶', 'Call stack · top on right') : copy('等待队列 · 左侧先出', 'Pending queue · left exits first')}</dt><dd>${state.frontier.join(' → ') || '∅'}</dd>
            <dt>${copy('首次发现顺序', 'Discovery order')}</dt><dd>${state.order.join(' → ')}</dd>
            ${mode === 'bfs' ? `<dt>${copy('最少边数', 'Minimum edge counts')}</dt><dd>${Object.entries(state.distances).map(([node, distance]) => `${node}:${distance}`).join(' · ')}</dd>` : `<dt>${copy('栈帧在等什么', 'What a frame remembers')}</dt><dd>${copy('当前节点，以及下一个要看的邻居。', 'Its node and the next neighbor to examine.')}</dd>`}</dl></section>`;
        }).join('')}</div>
        <div class="traversal-controls"><button type="button" data-action="previous" ${step === 0 ? 'disabled' : ''}>${copy('上一步', 'Back')}</button><button type="button" data-action="next" ${step === last ? 'disabled' : ''}>${copy('下一步', 'Next')}</button><button type="button" data-action="reset">${copy('重来', 'Reset')}</button><span role="status">${copy('演示步骤', 'Trace step')} ${step} / ${last}</span></div>
        <p class="traversal-note">${copy('着色节点已发现，粗虚线圈标出本步处理对象。两边按各自的操作推进，不是速度比较；DFS 的返回也算一步。', 'Tinted nodes are discovered; the thick dashed ring marks this step’s node. Each side advances its own operations, not equal time. DFS returns also count as steps.')}</p>`;
      root.querySelector('select').value = selected;
      root.setAttribute('role', 'group');
      root.setAttribute('aria-labelledby', title);
      root.querySelector('[data-action="graph"]').addEventListener('change', event => {
        selected = event.target.value;
        step = 0;
        draw();
        root.querySelector('[data-action="graph"]').focus();
      });
      root.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
        const action = button.dataset.action;
        if (action === 'language') language = language === 'zh' ? 'en' : 'zh';
        if (action === 'next') step += 1;
        if (action === 'previous') step -= 1;
        if (action === 'reset') step = 0;
        draw();
        const replacement = root.querySelector(`[data-action="${action}"]`);
        (replacement.disabled ? root.querySelector('[data-action="reset"]') : replacement).focus();
      }));
    }
    draw();
  });
}());
