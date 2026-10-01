const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../static/discussions.js"), "utf8");

function setup(lang = "en", present = true) {
  const listeners = {};
  const button = {hidden: true, addEventListener(name, handler) { this[name] = handler; }};
  const status = {textContent: ""};
  const scripts = [];
  const messages = [];
  const frame = {contentWindow: {postMessage(value, target) { messages.push({value, target}); }}};
  const mount = {appendChild(script) { scripts.push(script); }, querySelector() { return scripts.length ? frame : null; }};
  const widget = {dataset: {lang, number: "46"},
    getAttribute(key) { return {"data-repo": "tianyi-zhang-02/cooking-agi", "data-repo-id": "repo", "data-category": "Announcements", "data-category-id": "category", "data-lang": lang}[key]; },
    querySelector(selector) { return selector === "[data-load-comments]" ? button : selector === ".giscus" ? mount : status; }};
  const root = {dataset: {theme: "dark"}};
  let timer;
  vm.runInNewContext(source, {
    document: {documentElement: root, querySelector() { return present ? widget : null; },
      createElement() { return {attributes: {}, setAttribute(key, value) { this.attributes[key] = value; }, addEventListener(name, handler) { this[name] = handler; }}; }},
    window: {addEventListener(name, handler) { listeners[name] = handler; }},
    setTimeout(handler) { timer = handler; return 1; }, clearTimeout() { timer = null; }
  });
  return {button, status, scripts, frame, root, messages, listeners, timeout() { timer(); }};
}

assert.equal(setup("en", false).scripts.length, 0);
const page = setup();
assert.equal(page.scripts.length, 0);
assert.equal(page.button.hidden, false);
page.button.click();
page.button.click();
assert.equal(page.scripts.length, 1);
assert.equal(page.scripts[0].src, "https://giscus.app/client.js");
assert.equal(page.scripts[0].attributes["data-mapping"], "number");
assert.equal(page.scripts[0].attributes["data-term"], "46");
assert.equal(page.scripts[0].attributes["data-lang"], "en");
page.listeners.message({origin: "https://bad.example", source: page.frame.contentWindow, data: {giscus: {discussion: {number: 46}}}});
assert.notEqual(page.status.textContent, "");
page.listeners.message({origin: "https://giscus.app", source: {}, data: {giscus: {discussion: {number: 46}}}});
assert.notEqual(page.status.textContent, "");
page.listeners.message({origin: "https://giscus.app", source: page.frame.contentWindow, data: {giscus: {discussion: {number: 46}}}});
assert.equal(page.status.textContent, "");
assert.equal(page.button.hidden, true);
page.root.dataset.theme = "light";
page.listeners.themechange();
assert.equal(page.messages[0].target, "https://giscus.app");
assert.equal(page.messages[0].value.giscus.setConfig.theme, "light");
const slow = setup("zh-CN");
slow.button.click();
slow.timeout();
assert.match(slow.status.textContent, /直接回复/);
assert.equal(slow.button.hidden, true);
const failed = setup();
failed.button.click();
failed.scripts[0].error();
assert.match(failed.status.textContent, /GitHub link/);
console.log("Discussion loader checks passed");
