(function () {
  'use strict';
  var chinese = document.documentElement.lang.indexOf('zh') === 0;
  var labels = chinese ? {
    scroll: '左右滑动查看完整内容', expand: '放大查看', figure: '图示', close: '关闭图示', fit: '适合屏幕', original: '原始尺寸'
  } : {
    scroll: 'Scroll sideways to see the rest', expand: 'Enlarge figure', figure: 'Figure', close: 'Close figure', fit: 'Fit to screen', original: 'Original size'
  };
  var body = document.querySelector('.article-body');
  if (!body) return;
  var regions = new Map();
  var images = new WeakSet();
  var viewer = null;
  var stageImage = null;
  var credit = null;
  var fitButton = null;
  var returnFocus = null;

  function updateRegion(region) {
    var record = regions.get(region);
    var overflow = region.clientWidth > 0 && region.scrollWidth > region.clientWidth + 2;
    record.hint.hidden = !overflow;
    if (overflow && record.tabindex === null) region.setAttribute('tabindex', '0');
    if (!overflow && record.tabindex === null) region.removeAttribute('tabindex');
    if (overflow && record.label === null) region.setAttribute('aria-label', labels.scroll);
    if (!overflow && record.label === null) region.removeAttribute('aria-label');
  }
  var resize = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(function (entries) {
    entries.forEach(function (entry) {
      var region = regions.has(entry.target) ? entry.target : entry.target.parentElement;
      if (regions.has(region)) updateRegion(region);
    });
  });

  function addRegion(region) {
    if (regions.has(region)) return;
    var hint = document.createElement('small');
    hint.className = 'reading-scroll-hint';
    hint.textContent = labels.scroll;
    hint.hidden = true;
    region.after(hint);
    region.classList.add('reading-scroll');
    regions.set(region, {hint: hint, tabindex: region.getAttribute('tabindex'), label: region.getAttribute('aria-label')});
    var table = region.querySelector('table');
    if (table && table.rows.length) {
      var columns = Array.from(table.rows).reduce(function (maximum, row) {
        return Math.max(maximum, Array.from(row.cells).reduce(function (total, cell) { return total + cell.colSpan; }, 0));
      }, 0);
      if (columns >= 3) region.style.setProperty('--reading-table-width', Math.min(columns * 144, 960) + 'px');
    }
    if (resize) {
      resize.observe(region);
      if (region.firstElementChild) resize.observe(region.firstElementChild);
    }
    updateRegion(region);
  }

  function createViewer() {
    viewer = document.createElement('dialog');
    if (typeof viewer.showModal !== 'function') return false;
    viewer.className = 'figure-viewer';
    viewer.setAttribute('aria-label', labels.figure);
    var header = document.createElement('header');
    var title = document.createElement('h2');
    title.textContent = labels.figure;
    fitButton = document.createElement('button');
    fitButton.type = 'button';
    fitButton.textContent = labels.fit;
    fitButton.setAttribute('aria-pressed', 'false');
    fitButton.addEventListener('click', function () {
      var fit = viewer.classList.toggle('is-fit');
      fitButton.setAttribute('aria-pressed', String(fit));
      fitButton.textContent = fit ? labels.original : labels.fit;
    });
    var close = document.createElement('button');
    close.type = 'button';
    close.textContent = '×';
    close.setAttribute('aria-label', labels.close);
    close.addEventListener('click', function () { viewer.close(); });
    header.append(title, fitButton, close);
    var stage = document.createElement('div');
    stage.className = 'figure-viewer-stage';
    stage.tabIndex = 0;
    stageImage = document.createElement('img');
    stage.append(stageImage);
    credit = document.createElement('div');
    credit.className = 'figure-viewer-credit';
    viewer.append(header, stage, credit);
    viewer.addEventListener('close', function () {
      document.documentElement.classList.remove('figure-is-open');
      if (returnFocus && returnFocus.isConnected) returnFocus.focus({preventScroll: true});
    });
    viewer.addEventListener('click', function (event) {
      if (event.target !== viewer) return;
      var bounds = viewer.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) viewer.close();
    });
    document.body.append(viewer);
    return true;
  }

  function addImage(image) {
    if (images.has(image) || image.closest('a, button, [data-widget], .bert-story, .tx-lab, .algo-lab')) return;
    images.add(image);
    function ready() {
      if (image.naturalWidth < 320) return;
      if (!viewer && !createViewer()) return;
      if (typeof viewer.showModal !== 'function') return;
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'figure-expand';
      button.textContent = labels.expand;
      button.setAttribute('aria-label', labels.expand + (image.alt ? ': ' + image.alt : ''));
      button.addEventListener('click', function () {
        returnFocus = button;
        stageImage.src = image.currentSrc || image.src;
        stageImage.alt = image.alt;
        stageImage.style.setProperty('--figure-width', image.naturalWidth + 'px');
        var caption = image.closest('figure');
        caption = caption && caption.querySelector('figcaption');
        credit.textContent = caption ? caption.textContent : image.alt;
        viewer.classList.remove('is-fit');
        fitButton.textContent = labels.fit;
        fitButton.setAttribute('aria-pressed', 'false');
        document.documentElement.classList.add('figure-is-open');
        viewer.showModal();
        stageImage.parentElement.scrollTo(0, 0);
      });
      image.after(button);
    }
    if (image.complete) ready();
    else image.addEventListener('load', ready, {once: true});
  }

  function enhance() {
    body.querySelectorAll('.table-wrap, .fig-scroll').forEach(addRegion);
    body.querySelectorAll('img').forEach(addImage);
  }
  enhance();
  if (typeof MutationObserver !== 'undefined') {
    new MutationObserver(function (records) {
      if (records.some(function (record) {
        return Array.from(record.addedNodes).some(function (node) {
          return node.nodeType === 1 && (node.matches('img,.table-wrap,.fig-scroll') || node.querySelector('img,.table-wrap,.fig-scroll'));
        });
      })) enhance();
    }).observe(body, {childList: true, subtree: true});
  }
  window.addEventListener('resize', function () { regions.forEach(function (_, region) { updateRegion(region); }); }, {passive: true});
})();
