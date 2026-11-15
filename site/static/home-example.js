/* The first calculation on the home page (build.example_html, site/home.toml [example]).
   The page ships the whole calculation worked out, so it reads the same without this script.
   Here it becomes three steps (predict, take the loss, update the weight) with the loss curve beside
   it. The formulas are build.example_numbers' own: prediction = weight × input, mean squared error,
   and one gradient step per learning rate. Nothing moves on its own; every change follows a button. */
(function () {
  'use strict';
  var figure = document.querySelector('[data-home-example]');
  if (!figure) return;
  var data;
  try { data = JSON.parse(figure.getAttribute('data-home-example')); } catch (error) { return; }
  var work = figure.querySelector('.hx-work');
  var plot = figure.querySelector('.hx-plot');
  if (!work || !plot) return;

  var SVG = 'http://www.w3.org/2000/svg';
  var PARTS = ['predict', 'loss', 'update'];
  var state = { step: 0, rate: 0 };

  function num(value) { return String(Math.round(value * 1e6) / 1e6).replace('-', '−'); }
  function lossAt(weight) {
    var total = 0;
    data.inputs.forEach(function (x, i) { total += Math.pow(weight * x - data.targets[i], 2); });
    return total / data.inputs.length;
  }
  var start = { weight: data.weight, loss: lossAt(data.weight) };
  var gradient = data.inputs.reduce(function (sum, x, i) {
    return sum + 2 * (data.weight * x - data.targets[i]) * x;
  }, 0) / data.inputs.length;
  var moves = data.rates.map(function (rate) {
    var weight = data.weight - rate * gradient;
    return { rate: rate, weight: weight, loss: lossAt(weight) };
  });

  function el(tag, attributes, text) {
    var node = document.createElement(tag);
    Object.keys(attributes || {}).forEach(function (key) { node.setAttribute(key, attributes[key]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function svg(tag, attributes) {
    var node = document.createElementNS(SVG, tag);
    Object.keys(attributes || {}).forEach(function (key) { node.setAttribute(key, attributes[key]); });
    return node;
  }

  // the table cells of later steps keep their value and show a question mark until then
  var cells = Array.prototype.slice.call(figure.querySelectorAll('td[data-part]'));
  cells.forEach(function (cell) { cell.setAttribute('data-value', cell.textContent); });
  var parts = Array.prototype.slice.call(figure.querySelectorAll('[data-part]:not(td):not(th)'));
  var stepItems = Array.prototype.slice.call(figure.querySelectorAll('.hx-steps li'));

  var controls = el('div', { 'class': 'hx-controls' });
  var rates = el('div', { 'class': 'hx-rates', role: 'group', 'aria-label': data.rate });
  rates.appendChild(el('span', { 'aria-hidden': 'true' }, data.rate));
  var rateButtons = data.rates.map(function (rate, index) {
    var button = el('button', { type: 'button', 'aria-pressed': 'false' }, num(rate));
    button.addEventListener('click', function () { state.rate = index; render('rate'); });
    rates.appendChild(button);
    return button;
  });
  var next = el('button', { type: 'button', 'class': 'hx-next' });
  next.addEventListener('click', function () {
    if (state.step < PARTS.length - 1) state.step += 1;
    else { state.step = 0; state.rate = 0; }
    render('step');
  });
  var live = el('p', { 'class': 'hx-live', role: 'status', 'aria-live': 'polite' });
  controls.appendChild(rates);
  controls.appendChild(next);
  // the controls sit under the curve on phones (style.css grid areas), so a click shows its result
  figure.insertBefore(controls, figure.querySelector('figcaption'));
  figure.appendChild(live);

  var title = el('p', { 'class': 'hx-plot-title' }, data.curve);
  var drawing = svg('svg', { viewBox: '0 0 360 250', 'aria-hidden': 'true', focusable: 'false' });
  var legend = el('ul', { 'class': 'hx-legend' });
  plot.appendChild(title);
  plot.appendChild(drawing);
  plot.appendChild(legend);
  plot.hidden = false;
  figure.classList.add('is-live');

  function draw() {
    var move = moves[state.rate];
    var wide = state.step === 2 && (move.weight > 4 || move.loss > 12);   // the overshoot needs a wider view
    var wMax = wide ? 6.5 : 4, lMax = wide ? 45 : 12;
    var left = 40, right = 346, top = 16, bottom = 214;
    function X(w) { return left + w / wMax * (right - left); }
    function Y(l) { return bottom - l / lMax * (bottom - top); }
    while (drawing.firstChild) drawing.removeChild(drawing.firstChild);
    while (legend.firstChild) legend.removeChild(legend.firstChild);

    drawing.appendChild(svg('line', { 'class': 'hx-axis', x1: left, y1: bottom, x2: right, y2: bottom }));
    drawing.appendChild(svg('line', { 'class': 'hx-axis', x1: left, y1: bottom, x2: left, y2: top }));
    for (var w = 0; w <= wMax; w += 1) {
      var tick = svg('text', { 'class': 'hx-tick', x: X(w), y: bottom + 18, 'text-anchor': 'middle' });
      tick.textContent = num(w);
      drawing.appendChild(tick);
    }
    (wide ? [0, 20, 40] : [0, 5, 10]).forEach(function (l) {
      if (l > 0) drawing.appendChild(svg('line', { 'class': 'hx-grid', x1: left, y1: Y(l), x2: right, y2: Y(l) }));
      var label = svg('text', { 'class': 'hx-tick', x: left - 8, y: Y(l) + 4, 'text-anchor': 'end' });
      label.textContent = num(l);
      drawing.appendChild(label);
    });
    var xName = svg('text', { 'class': 'hx-axis-name', x: right, y: bottom + 34, 'text-anchor': 'end' });
    xName.textContent = data.axis[0];
    var yName = svg('text', { 'class': 'hx-axis-name', x: left, y: top - 4, 'text-anchor': 'middle' });
    yName.textContent = data.axis[1];
    drawing.appendChild(xName);
    drawing.appendChild(yName);

    var points = [];
    for (var i = 0; i <= 160; i += 1) {
      var weight = wMax * i / 160, loss = lossAt(weight);
      if (loss <= lMax) points.push(X(weight).toFixed(1) + ',' + Y(loss).toFixed(1));
    }
    drawing.appendChild(svg('polyline', { 'class': 'hx-curve', points: points.join(' ') }));

    function item(kind, text) {
      var row = el('li', { 'class': 'hx-key-' + kind });
      row.appendChild(el('span', { 'aria-hidden': 'true' }));
      row.appendChild(document.createTextNode(text));
      legend.appendChild(row);
    }
    if (state.step === 0) {
      drawing.appendChild(svg('line', { 'class': 'hx-marker', x1: X(start.weight), y1: bottom, x2: X(start.weight), y2: top }));
      item('marker', 'weight = ' + num(start.weight));
      return;
    }
    if (state.step === 2) {
      // the slope at the current weight, drawn as far as it stays inside the plot
      var a = Math.max(0, start.weight - 0.8), b = start.weight + 0.8;
      var la = start.loss + gradient * (a - start.weight), lb = start.loss + gradient * (b - start.weight);
      if (lb < 0) { b = start.weight - start.loss / gradient; lb = 0; }
      drawing.appendChild(svg('line', { 'class': 'hx-tangent', x1: X(a), y1: Y(la), x2: X(b), y2: Y(lb) }));
      // the update moves the weight: an arrow along the weight axis, then the new point on the curve
      var y = bottom - 7, x1 = X(start.weight), x2 = X(move.weight), dir = x2 > x1 ? 1 : -1;
      drawing.appendChild(svg('line', { 'class': 'hx-move', x1: x1, y1: y, x2: x2 - dir * 6, y2: y }));
      drawing.appendChild(svg('path', { 'class': 'hx-head', d: 'M' + x2 + ',' + y + 'l' + (-dir * 8) + ',-4.5v9z' }));
      drawing.appendChild(svg('line', { 'class': 'hx-drop', x1: x2, y1: y, x2: x2, y2: Y(move.loss) }));
    }
    drawing.appendChild(svg('circle', { 'class': 'hx-before', cx: X(start.weight), cy: Y(start.loss), r: 5.5 }));
    item('before', 'weight ' + num(start.weight) + ' · loss ' + num(start.loss));
    if (state.step === 2) {
      drawing.appendChild(svg('circle', { 'class': 'hx-after', cx: X(move.weight), cy: Y(move.loss), r: 5.5 }));
      item('tangent', data.slope + ' ' + num(gradient));
      item('move', 'weight ' + num(start.weight) + ' \u2192 ' + num(move.weight));
      item('after', 'weight ' + num(move.weight) + ' · loss ' + num(move.loss));
    }
  }

  // announce what the click revealed: the new step's lines, or only the result after a new learning rate
  function render(announce) {
    var step = state.step;
    var rate = num(data.rates[state.rate]);
    var said = [];
    parts.forEach(function (node) {
      var index = PARTS.indexOf(node.getAttribute('data-part'));
      var shown = index <= step;
      if (node.hasAttribute('data-rate')) shown = shown && node.getAttribute('data-rate') === rate;
      node.hidden = !shown;
      if (shown && index === step && (announce !== 'rate' || node.hasAttribute('data-rate'))) {
        said.push((node.innerText || node.textContent).replace(/\s+/g, ' ').trim());   // innerText keeps the label apart from the value
      }
    });
    cells.forEach(function (cell) {
      var shown = PARTS.indexOf(cell.getAttribute('data-part')) <= step;
      cell.textContent = shown ? cell.getAttribute('data-value') : '?';
      cell.classList.toggle('is-pending', !shown);
    });
    stepItems.forEach(function (item, index) {
      if (index === step) item.setAttribute('aria-current', 'step');
      else item.removeAttribute('aria-current');
      item.classList.toggle('is-done', index < step);
    });
    rates.hidden = step < PARTS.length - 1;
    rateButtons.forEach(function (button, index) { button.setAttribute('aria-pressed', String(index === state.rate)); });
    next.textContent = data.next[step];
    draw();
    if (announce) live.textContent = said.join(' ');
  }

  render(false);
})();
