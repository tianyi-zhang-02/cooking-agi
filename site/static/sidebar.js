(function () {
  "use strict";

  function cleanList(value, known, limit) {
    if (!Array.isArray(value)) return [];
    return Array.from(new Set(value.filter(function (note) {
      return typeof note === "string" && known.has(note);
    }))).slice(0, limit);
  }

  function cleanReading(value, known) {
    value = value && typeof value === "object" ? value : {};
    return { saved: cleanList(value.saved, known, 100), recent: cleanList(value.recent, known, 8) };
  }

  function recordVisit(state, note) {
    return { saved: state.saved.slice(), recent: [note].concat(state.recent.filter(function (item) {
      return item !== note;
    })).slice(0, 8) };
  }

  function importBookmarks(text, state, known) {
    if (typeof text !== "string" || text.length > 131072) throw Error("invalid-backup");
    var backup = JSON.parse(text);
    if (!backup || backup.format !== "cooking-agi-bookmarks" || backup.version !== 1 ||
        !Array.isArray(backup.saved) || backup.saved.length > 1000 ||
        !backup.saved.every(function (note) { return typeof note === "string" && note.length <= 300; })) {
      throw Error("invalid-backup");
    }
    var existing = cleanReading(state, known);
    var requested = Array.from(new Set(backup.saved));
    var candidates = requested.filter(function (note) { return known.has(note) && !existing.saved.includes(note); });
    var added = candidates.slice(0, 100 - existing.saved.length);
    return {
      state: { saved: existing.saved.concat(added), recent: existing.recent },
      added: added.length,
      unavailable: requested.filter(function (note) { return !known.has(note); }).length,
      overflow: candidates.length - added.length
    };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { cleanReading: cleanReading, recordVisit: recordVisit, importBookmarks: importBookmarks };
  }
  if (typeof document === "undefined") return;
  var side = document.querySelector("#side");
  if (!side) return;
  var tools = side.querySelector(".side-tools");
  if (!tools) return;
  var zh = document.documentElement.lang.startsWith("zh");
  var directory = side.querySelector("#side-directory");
  var reading = side.querySelector("#side-reading");
  var scroll = side.querySelector(".side-scroll");
  var save = side.querySelector("[data-side-save]");
  var saveButtons = Array.from(document.querySelectorAll("[data-side-save]"));
  var status = side.querySelector(".side-status");
  var inlineStatus = document.querySelector(".note-tool-status");
  var exportButton = side.querySelector("[data-reading-export]");
  var importButton = side.querySelector("[data-reading-import]");
  var importFile = side.querySelector("[data-reading-file]");
  var links = Array.from(directory.querySelectorAll("a[data-note]"));
  var catalog = new Map(links.map(function (link) {
    var group = link.closest("[data-grp]");
    return [link.dataset.note, {
      href: link.getAttribute("href"), title: link.title || link.textContent,
      group: group.querySelector(".grp-name").textContent,
      category: link.closest("[data-category]").dataset.category
    }];
  }));
  var active = directory.querySelector("a.active[data-note]");
  var current = active ? active.dataset.note : null;
  var readingKey = "cooking-agi:reading:v1";
  var treeKey = "cooking-agi:navigation:v1";
  function load(key) {
    try { return JSON.parse(localStorage.getItem(key) || "null"); }
    catch (error) { return null; }
  }
  function persist(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; }
    catch (error) {
      side.querySelector(".side-local").textContent = zh
        ? "浏览器暂时无法保存记录，离开本页后这些改动可能会丢失。"
        : "Browser storage is unavailable. Changes last for this page only.";
      return false;
    }
  }
  function announce(message) {
    status.textContent = message;
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
      if (item.top < bounds.top || item.bottom > bounds.bottom) {
        scroll.scrollTop += item.top - bounds.top - bounds.height / 3;
      }
    });
  }

  function renderList(selector, notes, removable) {
    var list = side.querySelector(selector);
    list.replaceChildren();
    if (!notes.length) {
      var empty = document.createElement("li");
      empty.className = "side-empty";
      empty.textContent = removable
        ? (zh ? "把想回看的笔记收在这里。" : "Keep notes you want to revisit here.")
        : (zh ? "打开一篇笔记，就会出现在这里。" : "Open a note to start your reading history.");
      list.append(empty);
    }
    notes.forEach(function (note) {
      var entry = catalog.get(note);
      var item = document.createElement("li");
      var link = document.createElement("a");
      link.href = entry.href;
      link.textContent = entry.title;
      if (note === current) link.setAttribute("aria-current", "page");
      var group = document.createElement("small");
      group.textContent = entry.group;
      link.append(group);
      item.append(link);
      if (removable) {
        var remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.setAttribute("aria-label", (zh ? "取消收藏：" : "Unsave: ") + entry.title);
        remove.addEventListener("click", function () {
          state.saved = state.saved.filter(function (saved) { return saved !== note; });
          persist(readingKey, state);
          renderReading();
          save.focus();
          status.textContent = zh ? "已取消收藏。" : "Note removed from saved.";
        });
        item.append(remove);
      }
      list.append(item);
    });
  }

  function renderReading() {
    var saved = state.saved.includes(current);
    saveButtons.forEach(function (button) {
      button.disabled = !current;
      button.setAttribute("aria-pressed", String(saved));
      button.textContent = !current
        ? (zh ? "打开一篇笔记后即可收藏" : "Open a note to save it")
        : saved ? (zh ? "已收藏 · 取消" : "Saved · Unsave")
        : (zh ? "＋ 收藏本页" : "+ Save this page");
    });
    renderList("[data-side-saved]", state.saved, true);
    renderList("[data-side-recent]", state.recent, false);
    side.querySelector("[data-side-clear]").disabled = !state.recent.length;
    exportButton.disabled = !state.saved.length;
  }

  saveButtons.forEach(function (button) { button.addEventListener("click", function () {
    if (!current) return;
    var saved = state.saved.includes(current);
    if (!saved && state.saved.length >= 100) {
      announce(zh ? "已收藏 100 篇，请先移除不再需要的笔记。" : "100 notes saved. Remove a note before adding another.");
      return;
    }
    state.saved = saved ? state.saved.filter(function (note) { return note !== current; }) : [current].concat(state.saved);
    var stored = persist(readingKey, state);
    renderReading();
    announce(!stored ? (zh ? "本页已更新，但浏览器无法保存。离开前可以导出收藏。" : "Updated for this page, but browser storage is unavailable. Export before leaving.")
      : saved ? (zh ? "已取消收藏。" : "Note unsaved.") : (zh ? "已加入收藏。" : "Note saved."));
  }); });
  exportButton.addEventListener("click", function () {
    if (!state.saved.length) return;
    var backup = JSON.stringify({format: "cooking-agi-bookmarks", version: 1, saved: state.saved}, null, 2);
    var url = URL.createObjectURL(new Blob([backup], {type: "application/json"}));
    var link = document.createElement("a");
    link.href = url;
    link.download = "agi-notes-bookmarks.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    announce(zh ? "已生成收藏备份，不含阅读记录。" : "Bookmark backup created; reading history is not included.");
  });
  importButton.addEventListener("click", function () { importFile.click(); });
  importFile.addEventListener("change", async function () {
    var file = importFile.files && importFile.files[0];
    if (!file) return;
    try {
      if (file.size > 131072) throw Error("invalid-backup");
      var text = await file.text();
      var imported = importBookmarks(text, state, catalog);
      state = imported.state;
      var stored = persist(readingKey, state);
      renderReading();
      var message = zh ? "新增 " + imported.added + " 篇收藏，已有收藏保留。" : "Added " + imported.added + " saved notes; existing bookmarks kept.";
      if (imported.unavailable) message += zh ? " " + imported.unavailable + " 篇不在当前目录中，已跳过。" : " Skipped " + imported.unavailable + " notes not in this catalog.";
      if (imported.overflow) message += zh ? " 收藏已达 100 篇，另有 " + imported.overflow + " 篇未导入。" : " The 100-note limit left " + imported.overflow + " notes unimported.";
      if (!stored) message += zh ? " 浏览器无法持久保存，请离开前导出备份。" : " Browser storage is unavailable; export before leaving.";
      announce(message);
    } catch (error) {
      announce(zh ? "没能读取这份备份，请选择本站导出的收藏 JSON 文件（不超过 128 KB）。已有收藏没有变。" : "Could not read this backup. Choose a bookmark JSON exported by this site (up to 128 KB). Existing bookmarks are unchanged.");
    }
    importFile.value = "";
  });
  side.querySelector("[data-side-clear]").addEventListener("click", function () {
    state.recent = [];
    persist(readingKey, state);
    persist("cooking-agi:positions:v1", []);
    window.dispatchEvent(new Event("notes:reading-cleared"));
    renderReading();
    announce(zh ? "最近打开和阅读位置已清空，收藏仍然保留。" : "Recent notes and reading positions cleared. Bookmarks are unchanged.");
  });
  side.querySelectorAll("[data-side-view]").forEach(function (button) {
    button.addEventListener("click", function () {
      var showDirectory = button.dataset.sideView === "directory";
      directory.hidden = !showDirectory;
      reading.hidden = showDirectory;
      side.querySelectorAll("[data-side-view]").forEach(function (view) {
        view.setAttribute("aria-pressed", String(view === button));
      });
      scroll.scrollTop = 0;
      if (showDirectory) revealCurrent();
    });
  });
  window.addEventListener("storage", function (event) {
    if (event.key === readingKey || event.key === null) {
      state = cleanReading(load(readingKey), catalog);
      renderReading();
    }
  });
  if (current) {
    state = recordVisit(state, current);
    persist(readingKey, state);
  }
  renderReading();
  tools.hidden = false;
  var noteTools = document.querySelector(".note-tools");
  if (noteTools && current) noteTools.hidden = false;
  if (active) revealCurrent();
}());
