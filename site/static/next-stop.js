(function () {
  "use strict";
  const root = document.querySelector("[data-next-stop-root]");
  if (!root) return;
  const chinese = root.dataset.lang === "zh";
  const records = Array.from(root.querySelectorAll("[data-stop-record]"));
  const filters = Array.from(root.querySelectorAll("[data-stop-filter]"));
  const views = Array.from(root.querySelectorAll("[data-stop-view]"));
  const companyList = root.querySelector("[data-stop-companies]");
  const companies = Array.from(companyList.children);
  const clear = root.querySelector("[data-stop-clear]");
  const status = root.querySelector("[data-stop-status]");
  let view = "timeline";

  const search = root.querySelector("[data-company-search]");
  if (search) {
    const presets = Array.from(root.querySelectorAll("[data-company-preset]"));
    const normalize = value => value.toLocaleLowerCase().replace(/[\s._-]+/g, "");
    const filterPresets = () => {
      const query = normalize(search.value);
      let count = 0;
      presets.forEach(preset => {
        preset.hidden = !normalize(preset.dataset.search).includes(query);
        if (!preset.hidden) count += 1;
      });
      root.querySelectorAll("[data-company-group]").forEach(group => {
        group.hidden = !Array.from(group.querySelectorAll("[data-company-preset]")).some(preset => !preset.hidden);
      });
      root.querySelector("[data-company-status]").textContent = chinese ? `${count} 个预设标识` : `${count} company presets`;
      root.querySelector("[data-company-no-match]").hidden = count > 0;
    };
    search.addEventListener("input", filterPresets);
    root.querySelector("[data-company-search-wrap]").hidden = false;
    filterPresets();
  }
  root.querySelectorAll(".company-mark img").forEach(image => {
    const fallback = () => { image.hidden = true; };
    image.addEventListener("error", fallback);
    if (image.complete && image.naturalWidth === 0) fallback();
  });

  function update() {
    const selected = Object.fromEntries(filters.map(filter => [filter.dataset.stopFilter, filter.value]));
    const counts = new Map();
    let total = 0;
    records.forEach(record => {
      record.hidden = Object.entries(selected).some(([field, value]) => value !== "all" && record.dataset[field] !== value);
      if (!record.hidden) {
        total += 1;
        counts.set(record.dataset.company, (counts.get(record.dataset.company) || 0) + 1);
      }
    });
    root.querySelectorAll("[data-stop-year]").forEach(year => {
      year.hidden = !Array.from(year.querySelectorAll("[data-stop-record]")).some(record => !record.hidden);
    });
    const peak = Math.max(1, ...counts.values());
    companies.sort((first, second) => {
      const difference = (counts.get(second.dataset.stopCompany) || 0) - (counts.get(first.dataset.stopCompany) || 0);
      if (difference) return difference;
      const firstName = first.dataset.name.toLowerCase();
      const secondName = second.dataset.name.toLowerCase();
      return firstName < secondName ? -1 : firstName > secondName ? 1 : 0;
    }).forEach(company => {
      const count = counts.get(company.dataset.stopCompany) || 0;
      company.hidden = !count;
      company.querySelector("[data-stop-count]").textContent = String(count);
      company.querySelector("[data-stop-unit]").textContent = chinese ? "条分享" : count === 1 ? "update" : "updates";
      company.querySelector(".stop-bar > span").style.width = (count / peak * 100) + "%";
      companyList.appendChild(company);
    });
    views.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.stopView === view)));
    root.querySelector("[data-stop-timeline]").hidden = view !== "timeline";
    companyList.hidden = view !== "companies";
    root.querySelector("[data-stop-no-match]").hidden = total > 0 || records.length === 0;
    clear.hidden = Object.values(selected).every(value => value === "all");
    const order = view === "timeline"
      ? (chinese ? "按开始时间由近到远" : "newest start dates first")
      : (chinese ? "按公司分享数量排序" : "companies by update count");
    status.textContent = chinese ? `${total} 条公开分享 · ${order}` : `${total} published ${total === 1 ? "update" : "updates"} · ${order}`;
  }

  filters.forEach(filter => filter.addEventListener("change", update));
  views.forEach(button => button.addEventListener("click", () => {
    view = button.dataset.stopView;
    update();
  }));
  clear.addEventListener("click", () => {
    filters.forEach(filter => { filter.value = "all"; });
    filters[0].focus();
    update();
  });
  root.querySelectorAll("[data-stop-open-record]").forEach(link => link.addEventListener("click", () => {
    view = "timeline";
    filters.forEach(filter => { filter.value = "all"; });
    update();
  }));
  root.querySelector("[data-stop-controls]").hidden = false;
  update();
})();
