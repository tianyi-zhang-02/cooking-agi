(function () {
  "use strict";

  function normalize(value) {
    return String(value || "").normalize("NFKC").toLowerCase().trim();
  }

  function safePath(value) {
    return value === "index.zh.html" || (typeof value === "string" && /^(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+(?:\.en)?\.html$/.test(value));
  }

  function noteKey(url) {
    return url === "index.zh.html" ? "index.html" : url.replace(/\.en\.html$/, ".html");
  }

  function prepareIndex(items) {
    if (!Array.isArray(items)) throw new Error("Invalid search index");
    var valid = items.filter(function (item) {
      return item && safePath(item.u) && ["zh", "en"].includes(item.l) &&
        [item.t, item.s, item.x].every(function (value) { return typeof value === "string"; });
    });
    var titles = new Map();
    valid.forEach(function (item) {
      var key = noteKey(item.u);
      titles.set(key, (titles.get(key) || "") + " " + normalize(item.t));
    });
    return valid.map(function (item) {
      return { u: item.u, t: item.t, s: item.s, l: item.l, x: item.x,
        preview: typeof item.p === "string" ? item.p : item.x,
        title: titles.get(noteKey(item.u)),
        section: normalize(item.s), body: normalize(item.x) };
    });
  }

  function rankNotes(items, query, language, allLanguages) {
    var phrase = normalize(query);
    if (!phrase) return [];
    var terms = [...new Set(phrase.split(/\s+/))];
    return items.filter(function (item) {
      return (allLanguages || item.l === language) && terms.every(function (term) {
        return item.title.includes(term) || item.section.includes(term) || item.body.includes(term);
      });
    }).map(function (item) {
      var score = item.title.includes(phrase) ? 80 : 0;
      terms.forEach(function (term) {
        score += item.title.includes(term) ? 24 : item.section.includes(term) ? 8 : 1;
        score += Math.min(item.body.split(term).length - 1, 12);
      });
      return { item: item, score: score };
    }).sort(function (left, right) {
      return right.score - left.score || (right.item.l === language) - (left.item.l === language) ||
        left.item.t.localeCompare(right.item.t, language);
    }).map(function (hit) { return hit.item; });
  }

  function excerpt(text, query) {
    var clean = text.replace(/\s+/g, " ").trim();
    var terms = normalize(query).split(/\s+/).filter(Boolean);
    var positions = terms.map(function (term) { return clean.toLowerCase().indexOf(term); })
      .filter(function (position) { return position >= 0; });
    var start = Math.max(0, (positions.length ? Math.min(...positions) : 0) - 32);
    return (start ? "…" : "") + clean.slice(start, start + 150) + (clean.length > start + 150 ? "…" : "");
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { normalize: normalize, safePath: safePath, prepareIndex: prepareIndex,
      rankNotes: rankNotes, excerpt: excerpt };
  }
  if (typeof document === "undefined") return;
  var dialog = document.querySelector(".note-search");
  if (!dialog || typeof dialog.showModal !== "function") return;
  var language = window.SITE.lang === "en" ? "en" : "zh";
  var copy = language === "zh" ? {
    start: "从一个问题开始", hint: "中文概念、英文术语，都可以试试。",
    loading: "正在查找…", empty: "还没找到这条笔记", emptyHint: "试试更短的关键词，或切换到「中英一起搜」。",
    failed: "搜索暂时没加载出来", failedHint: "笔记还在，可以重试一次，或先从目录浏览。",
    retry: "重新加载", count: " 篇笔记", limited: " · 显示前 12 篇"
  } : {
    start: "Start with a question", hint: "Look up a concept, a model, or something you want to practice.",
    loading: "Searching…", empty: "No matching notes yet", emptyHint: "Try a shorter term, or search both languages.",
    failed: "Search couldn't load", failedHint: "You can try again, or browse the notes from the navigation.",
    retry: "Try again", count: " notes", limited: " · Showing the first 12"
  };
  var input = dialog.querySelector("input");
  var results = dialog.querySelector("ol");
  var empty = dialog.querySelector(".note-search-empty");
  var status = dialog.querySelector("[role=status]");
  var scopeButtons = Array.from(dialog.querySelectorAll("[data-search-scope]"));
  var openers = Array.from(document.querySelectorAll("[data-search-open]"));
  var indexPromise = null, allLanguages = false, request = 0, timer = null, composing = false, opener = null;

  function loadIndex() {
    if (!indexPromise) {
      indexPromise = fetch((window.SITE.prefix || "") + "search-index.json?v=" + encodeURIComponent(window.SITE.built || ""))
        .then(function (response) {
          if (!response.ok) throw new Error("Search unavailable");
          return response.json();
        }).then(prepareIndex).catch(function (error) {
          indexPromise = null;
          throw error;
        });
    }
    return indexPromise;
  }

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function highlighted(tag, className, text, query) {
    var node = element(tag, className);
    var words = normalize(query).split(/\s+/).filter(Boolean).sort(function (left, right) { return right.length - left.length; });
    var pattern = new RegExp(words.map(function (word) { return word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }).join("|"), "gi");
    var offset = 0;
    for (var match of text.matchAll(pattern)) {
      node.append(document.createTextNode(text.slice(offset, match.index)), element("mark", "", match[0]));
      offset = match.index + match[0].length;
    }
    node.append(document.createTextNode(text.slice(offset)));
    return node;
  }

  function emptyState(title, hint, suggestions, retry) {
    results.hidden = true;
    empty.hidden = false;
    empty.replaceChildren(element("h3", "", title), element("p", "", hint));
    if (suggestions) {
      var topics = element("div", "note-search-topics");
      ["Attention", "CLIP", "RLHF", "LeetCode"].forEach(function (topic) {
        var button = element("button", "", topic);
        button.type = "button";
        button.addEventListener("click", function () { input.value = topic; input.focus(); render(); });
        topics.append(button);
      });
      empty.append(topics);
    }
    if (retry) {
      var button = element("button", "note-search-retry", copy.retry);
      button.type = "button";
      button.addEventListener("click", function () { input.focus(); render(); });
      empty.append(button);
    }
  }

  async function render() {
    clearTimeout(timer);
    var current = ++request;
    var query = input.value.trim();
    status.textContent = "";
    results.replaceChildren();
    if (!query) { emptyState(copy.start, copy.hint, true); return; }
    emptyState(copy.loading, "");
    status.textContent = copy.loading;
    try {
      var index = await loadIndex();
      if (current !== request || !dialog.open) return;
      var hits = rankNotes(index, query, language, allLanguages);
      status.textContent = hits.length + copy.count + (hits.length > 12 ? copy.limited : "");
      if (!hits.length) { emptyState(copy.empty, copy.emptyHint); return; }
      empty.hidden = true;
      results.hidden = false;
      hits.slice(0, 12).forEach(function (item) {
        var row = element("li");
        var link = element("a", "note-search-result");
        link.href = (window.SITE.prefix || "") + item.u;
        var context = element("span", "note-search-context");
        context.append(element("span", "", item.s), element("span", "note-search-language", item.l === "en" ? "EN" : "中文"));
        link.append(context, highlighted("strong", "note-search-title", item.t, query),
          highlighted("span", "note-search-excerpt", excerpt(item.preview, query), query));
        row.append(link);
        results.append(row);
      });
      dialog.querySelector(".note-search-body").scrollTop = 0;
    } catch (error) {
      if (current !== request || !dialog.open) return;
      status.textContent = copy.failed;
      emptyState(copy.failed, copy.failedHint, false, true);
    }
  }

  function openSearch(trigger) {
    if (dialog.open) { input.focus(); return; }
    opener = trigger;
    var menu = document.querySelector(".icon-btn.menu");
    if (menu && menu.getAttribute("aria-expanded") === "true") { menu.click(); opener = menu; }
    dialog.showModal();
    document.documentElement.classList.add("search-is-open");
    input.focus();
    input.select();
    render();
  }

  openers.forEach(function (button) {
    button.hidden = false;
    button.addEventListener("click", function () { openSearch(button); });
  });
  dialog.querySelector(".note-search-close").addEventListener("click", function () { dialog.close(); });
  dialog.addEventListener("close", function () {
    request++;
    clearTimeout(timer);
    document.documentElement.classList.remove("search-is-open");
    if (opener && opener.isConnected) opener.focus();
  });
  dialog.addEventListener("click", function (event) {
    if (event.target !== dialog) return;
    var bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  input.addEventListener("compositionstart", function () { composing = true; clearTimeout(timer); request++; });
  input.addEventListener("compositionend", function () { composing = false; render(); });
  input.addEventListener("input", function () {
    request++;
    clearTimeout(timer);
    results.replaceChildren();
    if (!composing) timer = setTimeout(render, 140);
  });
  scopeButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      allLanguages = button.dataset.searchScope === "all";
      scopeButtons.forEach(function (item) { item.setAttribute("aria-pressed", String(item === button)); });
      render();
    });
  });
  dialog.addEventListener("keydown", function (event) {
    if (event.isComposing || composing) return;
    var links = Array.from(results.querySelectorAll("a"));
    var focused = document.activeElement;
    if (!links.length || (focused !== input && !links.includes(focused))) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      var position = links.indexOf(focused);
      if (event.key === "ArrowDown") links[(position + 1) % links.length].focus();
      else if (position === 0) input.focus();
      else links[position < 0 ? links.length - 1 : position - 1].focus();
    } else if (event.key === "Enter" && focused === input) {
      event.preventDefault();
      links[0].click();
    }
  });
  document.addEventListener("keydown", function (event) {
    var target = event.target;
    if (event.isComposing || event.altKey || event.defaultPrevented) return;
    var command = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k";
    var slash = event.key === "/" && !event.metaKey && !event.ctrlKey && !event.shiftKey &&
      !target.closest("input, textarea, select, [contenteditable]:not([contenteditable=false])");
    if ((command || slash) && !document.querySelector("dialog[open]:not(.note-search)")) {
      event.preventDefault();
      openSearch(target);
    }
  });
}());
