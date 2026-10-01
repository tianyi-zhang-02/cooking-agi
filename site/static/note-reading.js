(function () {
  "use strict";

  function cleanPositions(value) {
    if (!Array.isArray(value)) return [];
    var seen = new Set();
    return value.filter(function (entry) {
      if (!entry || typeof entry.note !== "string" || !/^[a-zA-Z0-9_./-]{1,300}\.md$/.test(entry.note) ||
          !["zh", "en"].includes(entry.lang) || typeof entry.anchor !== "string" ||
          !/^[\p{L}\p{N}_:.-]{1,180}$/u.test(entry.anchor)) return false;
      var key = entry.note + ":" + entry.lang;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    }).slice(0, 100).map(function (entry) { return {note: entry.note, lang: entry.lang, anchor: entry.anchor}; });
  }

  function rememberPosition(value, entry) {
    return cleanPositions([entry].concat(cleanPositions(value).filter(function (saved) {
      return saved.note !== entry.note || saved.lang !== entry.lang;
    })));
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {cleanPositions: cleanPositions, rememberPosition: rememberPosition};
  }
  if (typeof document === "undefined") return;
  var tools = document.querySelector("[data-reading-note]");
  var article = document.querySelector(".article-body");
  if (!tools || !article) return;
  var headings = Array.from(article.querySelectorAll("h2[id]"));
  if (headings.length < 2) return;
  var resume = tools.querySelector("[data-note-resume]");
  var note = tools.dataset.readingNote;
  var lang = document.documentElement.lang.startsWith("zh") ? "zh" : "en";
  var storageKey = "cooking-agi:positions:v1";
  var timer = null;
  var moved = false;
  var lastAnchor = null;

  function load() {
    try { return cleanPositions(JSON.parse(localStorage.getItem(storageKey) || "[]")); }
    catch (error) { return []; }
  }

  function showResume() {
    var saved = load().find(function (entry) { return entry.note === note && entry.lang === lang; });
    var heading = saved && headings.find(function (item) { return item.id === saved.anchor; });
    resume.hidden = !heading;
    if (!heading) {
      resume.removeAttribute("href");
      return;
    }
    var label = heading.textContent.replace(/¶|#\s*$/g, "").trim();
    resume.href = "#" + encodeURIComponent(heading.id);
    resume.textContent = (lang === "zh" ? "继续上次位置：" : "Pick up at: ") + label;
  }

  function remember() {
    clearTimeout(timer);
    if (!moved) return;
    var passed = headings.filter(function (heading) {
      return heading.getClientRects().length && heading.getBoundingClientRect().top <= 160;
    });
    var heading = passed[passed.length - 1];
    if (!heading || heading.id === lastAnchor) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(rememberPosition(load(), {note: note, lang: lang, anchor: heading.id})));
      lastAnchor = heading.id;
    } catch (error) {}
  }

  showResume();
  resume.addEventListener("click", function () { resume.hidden = true; });
  window.addEventListener("scroll", function () {
    moved = true;
    clearTimeout(timer);
    timer = setTimeout(remember, 250);
  }, {passive: true});
  window.addEventListener("pagehide", remember);
  document.addEventListener("visibilitychange", function () { if (document.hidden) remember(); });
  window.addEventListener("notes:reading-cleared", function () {
    clearTimeout(timer);
    moved = false;
    lastAnchor = null;
    showResume();
  });
  window.addEventListener("storage", function (event) {
    if (event.key === storageKey || event.key === null) {
      clearTimeout(timer);
      moved = false;
      lastAnchor = null;
      showResume();
    }
  });
})();
