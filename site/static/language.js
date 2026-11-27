(function () {
  "use strict";

  function validLanguage(value) {
    return value === "en" || value === "zh";
  }

  function storageKeys(baseHref) {
    var prefix = "cooking-agi:" + new URL(baseHref).pathname;
    return {choice: prefix + ":language:v1"};
  }

  function entryDecision(href, baseHref, saved) {
    var current = new URL(href);
    var base = new URL(baseHref);
    var decision = {redirect: null, choice: null};
    if (current.origin !== base.origin ||
        ![base.pathname, base.pathname + "index.html"].includes(current.pathname)) return decision;
    var requested = current.searchParams.get("lang");
    if (validLanguage(requested)) decision.choice = requested;
    var language = decision.choice || (validLanguage(saved) ? saved : "en");
    if (language === "zh") {
      var target = new URL("index.zh.html", base);
      target.search = current.search;
      target.searchParams.delete("lang");
      target.hash = current.hash;
      decision.redirect = target.href;
    }
    return decision;
  }

  function linkLanguage(href, declared, baseHref) {
    var base = new URL(baseHref);
    var target = new URL(href, base);
    if (target.origin !== base.origin || !target.pathname.startsWith(base.pathname)) return null;
    if (validLanguage(declared)) return declared;
    if (target.pathname === base.pathname + "index.zh.html") return "zh";
    if ([base.pathname, base.pathname + "index.html", base.pathname + "index.en.html"].includes(target.pathname)) return "en";
    return null;
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {entryDecision: entryDecision, storageKeys: storageKeys, linkLanguage: linkLanguage};
  }
  if (typeof document === "undefined") return;

  var script = document.currentScript;
  var base = new URL(script.dataset.siteRoot || "./", window.location.href);
  var keys = storageKeys(base.href);
  function load(key) {
    try {
      var saved = window.localStorage.getItem(key);
      if (saved !== null) return saved;
    } catch (error) {}
    try { return window.sessionStorage.getItem(key); } catch (error) { return null; }
  }
  function save(key, value) {
    try { window.localStorage.setItem(key, value); } catch (error) {}
    try { window.sessionStorage.setItem(key, value); } catch (error) {}
  }

  var decision = entryDecision(window.location.href, base.href, load(keys.choice));
  if (decision.choice) save(keys.choice, decision.choice);
  if (decision.redirect) {
    window.location.replace(decision.redirect);
    return;
  }

  // A click on a link into the other language (the switch in the top bar, the line on the home
  // cover) remembers that language for the next visit to the home page.
  function ready() {
    document.addEventListener("click", function (event) {
      var link = event.target.closest("a[href]");
      if (!link || event.defaultPrevented || event.button !== 0 ||
          event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      var language = linkLanguage(link.href, link.dataset.languageSwitch, base.href);
      if (language) save(keys.choice, language);
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready, {once: true});
  else ready();
})();
