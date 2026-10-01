(function () {
  "use strict";
  var widget = document.querySelector("[data-discussion-widget]");
  if (!widget) return;
  var button = widget.querySelector("[data-load-comments]");
  var status = widget.querySelector(".talk-load-status");
  var mount = widget.querySelector(".giscus");
  var chinese = widget.dataset.lang === "zh-CN";
  var started = false;
  var timeout;
  var origin = "https://giscus.app";

  function theme() {
    return document.documentElement.dataset.theme === "light" ? "light" : "dark";
  }

  function failed() {
    clearTimeout(timeout);
    button.hidden = true;
    status.textContent = chinese ? "留言暂时没加载出来，可以用下方链接直接回复。" : "Comments couldn't load. You can still reply using the GitHub link below.";
  }

  button.hidden = false;
  button.addEventListener("click", function () {
    if (started) return;
    started = true;
    button.disabled = true;
    status.textContent = chinese ? "正在连接留言区…" : "Connecting to the conversation…";
    var script = document.createElement("script");
    script.src = origin + "/client.js";
    script.async = true;
    script.crossOrigin = "anonymous";
    ["repo", "repo-id", "category", "category-id", "lang"].forEach(function (key) {
      script.setAttribute("data-" + key, widget.getAttribute("data-" + key));
    });
    var settings = {mapping: "number", term: widget.dataset.number, strict: "1",
      "reactions-enabled": "1", "emit-metadata": "1", "input-position": "top", theme: theme()};
    Object.keys(settings).forEach(function (key) { script.setAttribute("data-" + key, settings[key]); });
    script.addEventListener("error", failed);
    timeout = setTimeout(failed, 15000);
    mount.appendChild(script);
  });

  window.addEventListener("message", function (event) {
    var frame = mount.querySelector("iframe.giscus-frame");
    if (event.origin !== origin || !frame || event.source !== frame.contentWindow ||
        !event.data || typeof event.data !== "object" || !event.data.giscus) return;
    var message = event.data.giscus;
    if (message.error) { failed(); return; }
    if (message.discussion && message.discussion.number === Number(widget.dataset.number)) {
      clearTimeout(timeout);
      button.hidden = true;
      status.textContent = "";
    }
  });

  window.addEventListener("themechange", function () {
    var frame = mount.querySelector("iframe.giscus-frame");
    if (frame) frame.contentWindow.postMessage({giscus: {setConfig: {theme: theme()}}}, origin);
  });
})();
