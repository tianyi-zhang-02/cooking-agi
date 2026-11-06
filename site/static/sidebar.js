(function () {
  "use strict";
  /* The sidebar directory (fold state, "current chapter") and the one save button
     under a note's title. The list of saved notes, reading history and backups live
     in the reading drawer (reader-hub.js); both read the same records through
     window.PreparationProgress.library. */

  function cleanList(value, known, limit) {
    if (!Array.isArray(value)) return [];
    return Array.from(new Set(value.filter(function (note) {
      return typeof note === "string" && known.has(note);
    }))).slice(0, limit);
  }

  // `recent` is no longer shown, but the stored list is passed through untouched
  function cleanReading(value, known) {
    value = value && typeof value === "object" ? value : {};
    return { saved: cleanList(value.saved, known, 100), recent: cleanList(value.recent, known, 8) };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { cleanReading: cleanReading };
  }
  if (typeof document === "undefined") return;
  var side = document.querySelector("#side");
  if (!side) return;
  var zh = document.documentElement.lang.startsWith("zh");
  var directory = side.querySelector("#side-directory");
  var scroll = side.querySelector(".side-scroll");
  if (!directory || !scroll) return;
  var saveButtons = Array.from(document.querySelectorAll("[data-side-save]"));
  var inlineStatus = document.querySelector(".note-tool-status");
  var links = Array.from(directory.querySelectorAll("a[data-note]"));
  var catalog = new Map(links.map(function (link) { return [link.dataset.note, true]; }));
  var active = directory.querySelector("a.active[data-note]");
  var current = active ? active.dataset.note : null;
  var readingKey = "cooking-agi:reading:v1";
  var treeKey = "cooking-agi:navigation:v2";
  var library = window.PreparationProgress?.library;
  var identifiers = Object.fromEntries(links.filter(function (link) { return link.dataset.progressId; }).map(function (link) { return [link.dataset.note, link.dataset.progressId]; }));
  function load(key) {
    if (library && key === readingKey) {
      try { return library.read(identifiers); } catch (error) { return null; }
    }
    try { return JSON.parse(localStorage.getItem(key) || "null"); }
    catch (error) { return null; }
  }
  function persist(key, value) {
    if (library && key === readingKey) {
      try { return library.write(value, identifiers); } catch (error) { return false; }
    }
    try { localStorage.setItem(key, JSON.stringify(value)); return true; }
    catch (error) { return false; }
  }
  function announce(message) {
    if (inlineStatus) inlineStatus.textContent = message;
  }
  var state = cleanReading(load(readingKey), catalog);
  var tree = load(treeKey);
  tree = tree && typeof tree === "object" && !Array.isArray(tree) ? tree : {};
  var details = Array.from(directory.querySelectorAll("details"));
  var treeState = Object.create(null);
  details.forEach(function (detail) {
    var key = detail.dataset.grp ? "group:" + detail.dataset.grp : "chapter:" + detail.dataset.chapter;
    if (!detail.querySelector("a.active") && typeof tree[key] === "boolean") detail.open = tree[key];
    treeState[key] = detail.open;
    detail.addEventListener("toggle", function () {
      treeState[key] = detail.open;
      persist(treeKey, treeState);
    });
  });

  function revealCurrent() {
    if (!active) return;
    var parent = active.parentElement;
    while (parent && parent !== directory) {
      if (parent.tagName === "DETAILS") parent.open = true;
      parent = parent.parentElement;
    }
    requestAnimationFrame(function () {
      var bounds = scroll.getBoundingClientRect();
      var item = active.getBoundingClientRect();
      var actionsHeight = directoryActions && !directoryActions.hidden ? directoryActions.offsetHeight || 0 : 0;
      if (item.top < bounds.top + actionsHeight || item.bottom > bounds.bottom) {
        scroll.scrollTop += item.top - bounds.top - bounds.height / 3;
      }
    });
  }

  var directoryActions = directory.querySelector('.side-directory-actions');
  if (directoryActions && active) {
    directoryActions.hidden = false;
    directoryActions.querySelector('[data-side-current]').addEventListener('click', function () {
      revealCurrent();
      active.focus({ preventScroll: true });
    });
    directoryActions.querySelector('[data-side-collapse]').addEventListener('click', function () {
      details.forEach(function (detail) {
        detail.open = Boolean(detail.querySelector('a.active'));
      });
      revealCurrent();
    });
  }

  function renderSave() {
    var saved = state.saved.includes(current);
    saveButtons.forEach(function (button) {
      button.hidden = false;
      button.disabled = !current;
      button.setAttribute("aria-pressed", String(saved));
      button.textContent = saved ? (zh ? "已收藏 · 取消" : "Saved · Unsave") : (zh ? "＋ 收藏本页" : "+ Save this page");
    });
  }

  saveButtons.forEach(function (button) { button.addEventListener("click", function () {
    if (!current) return;
    var saved = state.saved.includes(current);
    if (!saved && state.saved.length >= 100) {
      announce(zh ? "已收藏 100 篇，请先在右上角「我的阅读」里移除不再需要的笔记。" : "100 notes saved. Remove one under “My reading” at the top right first.");
      return;
    }
    state.saved = saved ? state.saved.filter(function (note) { return note !== current; }) : [current].concat(state.saved);
    var stored = persist(readingKey, state);
    if (stored && identifiers[current]) window.dispatchEvent(new CustomEvent('notes:bookmark-action', {detail: {noteId: identifiers[current], saved: !saved}}));
    window.dispatchEvent(new Event("notes:prep-storage"));
    renderSave();
    announce(!stored ? (zh ? "本页已更新，但浏览器无法保存。" : "Updated for this page, but browser storage is unavailable.")
      : saved ? (zh ? "已取消收藏。" : "Note unsaved.") : (zh ? "已加入收藏，右上角「我的阅读」里能找到。" : "Saved. Find it under “My reading” at the top right."));
  }); });

  function refresh() {
    state = cleanReading(load(readingKey), catalog);
    renderSave();
  }
  window.addEventListener("storage", function (event) {
    if (event.key === readingKey || event.key === window.PreparationProgress?.storageKey || event.key === null) refresh();
  });
  window.addEventListener("notes:prep-storage", refresh);
  window.addEventListener("notes:prep-restored", refresh);
  renderSave();
  var noteTools = document.querySelector(".note-tools");
  if (noteTools && current) noteTools.hidden = false;
  if (active) revealCurrent();
}());
