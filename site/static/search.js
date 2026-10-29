(function () {
  "use strict";

  function normalize(value) {
    return String(value || "").normalize("NFKC").toLowerCase().trim();
  }

  function safePath(value) {
    return value === "index.zh.html" || (typeof value === "string" && /^(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+(?:\.en)?\.html$/.test(value));
  }

  // the form a query and the notes are compared in: hyphens, slashes and underscores read as
  // spaces ("kv-cache" finds "KV cache"), and a Latin word glued to Chinese stands apart ("kv缓存")
  function fold(value) {
    return normalize(value).replace(/[-_/·・]+/g, " ")
      .replace(/([a-z0-9])(?=[㐀-鿿])|([㐀-鿿])(?=[a-z0-9])/g, "$1$2 ")
      .replace(/\s+/g, " ").trim();
  }

  // the next place term occurs in text from position from; a Latin term only where a word
  // starts, so "lora" is not found inside "exploration" or "rope" inside "property"
  function find(text, term, from) {
    var latin = /^[a-z0-9]/.test(term), at = text.indexOf(term, from);
    while (latin && at > 0 && /[a-z0-9]/.test(text[at - 1])) at = text.indexOf(term, at + 1);
    return at;
  }

  function has(text, term) { return find(text, term, 0) >= 0; }

  function count(text, term, cap) {
    var found = 0, at = find(text, term, 0);
    while (at >= 0 && found < cap) { found++; at = find(text, term, at + term.length); }
    return found;
  }

  function isSection(section) {
    return Array.isArray(section) && section.length === 4 &&
      section.every(function (value) { return typeof value === "string"; });
  }

  // Each note arrives cut at its headings: [anchor, heading, prose, code].
  function prepareIndex(items) {
    if (!Array.isArray(items)) throw new Error("Invalid search index");
    return items.filter(function (item) {
      return item && safePath(item.u) && ["zh", "en"].includes(item.l) &&
        [item.t, item.s].every(function (value) { return typeof value === "string"; }) &&
        Array.isArray(item.h);
    }).map(function (item) {
      return { u: item.u, t: item.t, s: item.s, l: item.l,
        // the other language's title too, so an English term finds the Chinese note
        title: fold(item.t + " " + (typeof item.a === "string" ? item.a : "")),
        name: fold(item.t), section: fold(item.s),
        sections: item.h.filter(isSection).map(function (section) {
          return { anchor: section[0], heading: section[1], text: section[2],
            head: fold(section[1]), body: fold(section[2] + " " + section[3]) };
        }) };
    });
  }

  function rankNotes(items, query, language, allLanguages) {
    var phrase = fold(query);
    if (!phrase) return [];
    var terms = [...new Set(phrase.split(" "))];
    return items.filter(function (item) {
      return (allLanguages || item.l === language) && terms.every(function (term) {
        return has(item.title, term) || has(item.section, term) || item.sections.some(function (section) {
          return has(section.head, term) || has(section.body, term);
        });
      });
    }).map(function (item) {
      var aboutIt = terms.every(function (term) { return has(item.title, term); });
      var score = has(item.title, phrase) ? 80 : 0;
      // a note named for the query ("KV cache：…", "LoRA 与 QLoRA") comes before one that mentions it
      if (item.name.indexOf(phrase) === 0) score += 16;
      terms.forEach(function (term) {
        score += has(item.title, term) ? 24 : has(item.section, term) ? 8 : 1;
        score += item.sections.reduce(function (sum, section) {
          return Math.min(12, sum + count(section.body, term, 12));
        }, 0);
        if (item.sections.some(function (section) { return section.anchor && has(section.head, term); })) score += 6;
      });
      // the section that matched best: a heading counts most, then how often the words appear there
      var best = null, bestPoints = 0;
      item.sections.forEach(function (section) {
        var points = 0, every = true;
        terms.forEach(function (term) {
          var inHeading = has(section.head, term), inBody = count(section.body, term, 6);
          points += (inHeading ? 10 : 0) + inBody;
          every = every && (inHeading || inBody > 0);
        });
        if (every) points += 5;
        if (points > bestPoints) { best = section; bestPoints = points; }
      });
      // a note about the query opens at the top; otherwise at the section that matched
      var open = best && best.anchor && !aboutIt ? best : null;
      var preview = (best && best.text) || (item.sections.find(function (section) { return section.text; }) || { text: "" }).text;
      return { score: score, hit: { u: item.u, t: item.t, s: item.s, l: item.l,
        anchor: open ? open.anchor : "", heading: open ? open.heading : "", preview: preview } };
    }).sort(function (left, right) {
      return right.score - left.score || (right.hit.l === language) - (left.hit.l === language) ||
        left.hit.t.localeCompare(right.hit.t, language);
    }).map(function (ranked) { return ranked.hit; });
  }

  function excerpt(text, query) {
    var clean = text.replace(/\s+/g, " ").trim();
    var terms = fold(query).split(" ").filter(Boolean);
    var positions = terms.map(function (term) { return clean.toLowerCase().indexOf(term); })
      .filter(function (position) { return position >= 0; });
    var start = Math.max(0, (positions.length ? Math.min(...positions) : 0) - 32);
    return (start ? "…" : "") + clean.slice(start, start + 150) + (clean.length > start + 150 ? "…" : "");
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { normalize: normalize, fold: fold, safePath: safePath, prepareIndex: prepareIndex,
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
  var indexes = {}, allLanguages = false, request = 0, timer = null, composing = false, opener = null;

  // One file per language, versioned by its content: the other language loads only when asked
  // for, and a reader keeps the cached copy until the notes change.
  function loadIndex(lang) {
    if (!indexes[lang]) {
      var version = (window.SITE.search && window.SITE.search[lang]) || window.SITE.built || "";
      indexes[lang] = fetch((window.SITE.prefix || "") + "search-index." + lang + ".json?v=" + encodeURIComponent(version))
        .then(function (response) {
          if (!response.ok) throw new Error("Search unavailable");
          return response.json();
        }).then(prepareIndex).catch(function (error) {
          delete indexes[lang];
          throw error;
        });
    }
    return indexes[lang];
  }

  function loadIndexes() {
    var languages = allLanguages ? [language, language === "zh" ? "en" : "zh"] : [language];
    return Promise.all(languages.map(loadIndex)).then(function (lists) { return [].concat.apply([], lists); });
  }

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function highlighted(tag, className, text, query) {
    var node = element(tag, className);
    var words = fold(query).split(" ").filter(Boolean).sort(function (left, right) { return right.length - left.length; });
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
      var index = await loadIndexes();
      if (current !== request || !dialog.open) return;
      var hits = rankNotes(index, query, language, allLanguages);
      status.textContent = hits.length + copy.count + (hits.length > 12 ? copy.limited : "");
      if (!hits.length) { emptyState(copy.empty, copy.emptyHint); return; }
      empty.hidden = true;
      results.hidden = false;
      hits.slice(0, 12).forEach(function (item) {
        var row = element("li");
        var link = element("a", "note-search-result");
        var explicitLanguage = item.u === "index.html" && item.l === "en" ? "?lang=en" : "";
        link.href = (window.SITE.prefix || "") + item.u + explicitLanguage + (item.anchor ? "#" + encodeURIComponent(item.anchor) : "");
        // a section on the page already open only scrolls, so the dialog must step aside
        link.addEventListener("click", function () { dialog.close(); });
        var context = element("span", "note-search-context");
        context.append(element("span", "", item.s), element("span", "note-search-language", item.l === "en" ? "EN" : "中文"));
        link.append(context, highlighted("strong", "note-search-title", item.t, query));
        if (item.heading) link.append(highlighted("span", "note-search-section", item.heading, query));
        link.append(highlighted("span", "note-search-excerpt", excerpt(item.preview, query), query));
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

  // start fetching the moment a reader reaches for search, not when the dialog opens
  function warm() { loadIndex(language).catch(function () {}); }
  openers.forEach(function (button) {
    button.hidden = false;
    button.addEventListener("click", function () { openSearch(button); });
    ["pointerenter", "pointerdown", "focus"].forEach(function (name) { button.addEventListener(name, warm); });
  });
  dialog.querySelector(".note-search-close").addEventListener("click", function () { dialog.close(); });
  dialog.addEventListener("close", function () {
    request++;
    clearTimeout(timer);
    document.documentElement.classList.remove("search-is-open");
    if (opener && opener.isConnected) opener.focus({ preventScroll: true });
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
