(function () {
  "use strict";
  var root = document.documentElement;
  var storageKey = "agi-reading-english-terms";
  var enabled = true;
  var toggles = Array.from(document.querySelectorAll("[data-term-toggle]"));
  var menus = Array.from(document.querySelectorAll(".reading-settings"));

  try {
    enabled = localStorage.getItem(storageKey) !== "off";
  } catch (error) {}

  function update(value) {
    root.dataset.englishTerms = value ? "on" : "off";
    toggles.forEach(function (toggle) { toggle.checked = value; });
  }

  update(enabled);
  toggles.forEach(function (toggle) {
    toggle.closest("label").hidden = false;
    toggle.addEventListener("change", function () {
      update(toggle.checked);
      try {
        localStorage.setItem(storageKey, toggle.checked ? "on" : "off");
      } catch (error) {}
    });
  });

  document.addEventListener("click", function (event) {
    menus.forEach(function (menu) {
      if (!menu.contains(event.target)) menu.open = false;
    });
  });
  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") return;
    menus.forEach(function (menu) {
      if (!menu.open) return;
      var focused = menu.contains(document.activeElement);
      menu.open = false;
      if (focused) menu.querySelector("summary").focus();
    });
  });
})();
