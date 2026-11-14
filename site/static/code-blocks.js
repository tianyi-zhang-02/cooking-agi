/* Code on reading pages (build.code_blocks). The page already carries numbered lines, the folded
   ranges, the marks and the changed lines of each step; this script only adds the controls:
   - a set of implementations (.code-variants) becomes tabs, one per <details class="code-drop">;
   - a folded range (data-fold) hides behind a row that opens it;
   - a set of steps (.code-steps) can show only the lines that changed from the step before.
   Without it every implementation stays a <details> and every line shows. */
(function () {
  'use strict';
  var zh = document.documentElement.lang.indexOf('zh') === 0;
  var text = zh ? {
    group: '写法', only: '只看改动', show: '展开', hide: '收起',
    same: function (n) { return n + ' 行和上一步相同'; }
  } : {
    group: 'Implementations', only: 'Only changes', show: 'Show', hide: 'Hide',
    same: function (n) { return n + (n === 1 ? ' line' : ' lines') + ' as in the step before'; }
  };
  var each = function (list, fn) { Array.prototype.forEach.call(list, fn); };
  function el(tag, className, content) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = content;
    return node;
  }

  // folded ranges: the row above the first line opens and closes them
  each(document.querySelectorAll('.code-lines'), function (block) {
    each(block.querySelectorAll('.cl[data-fold-label]'), function (first) {
      var lines = block.querySelectorAll('.cl[data-fold="' + first.getAttribute('data-fold') + '"]');
      var row = el('button', 'cl-fold');
      row.type = 'button';
      var action = el('span', 'cl-fold-action');
      row.appendChild(el('span', 'cl-fold-lines', first.getAttribute('data-fold-lines')));
      row.appendChild(el('span', 'cl-fold-label', first.getAttribute('data-fold-label')));
      row.appendChild(action);
      first.parentNode.insertBefore(row, first);
      function set(open) {
        each(lines, function (line) { line.hidden = !open; });
        row.setAttribute('aria-expanded', String(open));
        action.textContent = open ? text.hide : text.show;
      }
      row.addEventListener('click', function () { set(row.getAttribute('aria-expanded') !== 'true'); });
      set(false);
    });
  });

  // in a set of steps, mark where unchanged lines would be skipped
  each(document.querySelectorAll('.code-steps .code-diff'), function (block) {
    if (block.querySelector('.cl-gap')) return;
    var run = [];
    function close() {
      if (run.length) run[0].parentNode.insertBefore(el('span', 'cl-gap', text.same(run.length)), run[0]);
      run = [];
    }
    each(block.querySelectorAll('.cl'), function (line) {
      if (line.hasAttribute('data-new')) close(); else run.push(line);
    });
    close();
  });

  // implementations as tabs
  var findable = 'onbeforematch' in document.body;
  each(document.querySelectorAll('.code-variants'), function (group, g) {
    var drops = Array.prototype.filter.call(group.children, function (node) { return node.matches('details.code-drop'); });
    if (drops.length < 2) return;
    var steps = group.classList.contains('code-steps');
    var bar = el('div', 'code-tabs-bar');
    var list = el('div', 'code-tabs');
    list.setAttribute('role', 'tablist');
    list.setAttribute('aria-label', text.group);
    bar.appendChild(list);
    var tabs = [], panels = [], only = null;
    drops.forEach(function (details, i) {
      var summary = details.querySelector('summary');
      var name = summary.querySelector('b') || summary;
      var note = summary.textContent.replace(name.textContent, '').replace(/^[\s·:：—-]+/, '').trim();
      var id = 'code-set-' + g + '-' + i;
      var tab = el('button', 'code-tab', name.textContent.trim());
      tab.type = 'button';
      tab.id = id + '-tab';
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-controls', id);
      var panel = el('div', 'code-panel');
      panel.id = id;
      panel.setAttribute('role', 'tabpanel');
      panel.setAttribute('aria-labelledby', tab.id);
      if (note) panel.appendChild(el('p', 'code-panel-note', note));
      while (summary.nextSibling) panel.appendChild(summary.nextSibling);
      group.insertBefore(panel, details);
      group.removeChild(details);
      tab.addEventListener('click', function () { select(i, false); });
      if (findable) panel.addEventListener('beforematch', function () { select(i, false); });
      list.appendChild(tab);
      tabs.push(tab);
      panels.push(panel);
    });
    if (steps) {
      var label = el('label', 'code-only');
      only = el('input');
      only.type = 'checkbox';
      label.appendChild(only);
      label.appendChild(document.createTextNode(text.only));
      only.addEventListener('change', function () { group.classList.toggle('only-changes', only.checked); });
      bar.appendChild(label);
    }
    group.insertBefore(bar, group.firstChild);
    group.classList.add('is-tabbed');

    function select(index, focus) {
      tabs.forEach(function (tab, i) {
        var on = i === index;
        tab.setAttribute('aria-selected', String(on));
        tab.tabIndex = on ? 0 : -1;
        if (on) panels[i].removeAttribute('hidden');
        else if (findable) panels[i].setAttribute('hidden', 'until-found');
        else panels[i].hidden = true;
      });
      if (only) {
        only.disabled = index === 0;            // the first step has nothing before it
        if (index === 0 && only.checked) { only.checked = false; group.classList.remove('only-changes'); }
      }
      if (focus) tabs[index].focus();
    }
    list.addEventListener('keydown', function (event) {
      var current = tabs.indexOf(document.activeElement);
      if (current < 0) return;
      var next = { ArrowRight: current + 1, ArrowLeft: current - 1, Home: 0, End: tabs.length - 1 }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      select((next + tabs.length) % tabs.length, true);
    });
    // a link into one implementation opens its tab
    var target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
    var start = 0;
    panels.forEach(function (panel, i) { if (target && panel.contains(target)) start = i; });
    select(start, false);
  });
})();
