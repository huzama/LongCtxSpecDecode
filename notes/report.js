/* Shared report controls. Measurement values come from saved audits. */
(() => {
  const $ = (s) => document.querySelector(s);
  const model = $("#model-filter"),
    context = $("#context-filter"),
    format = $("#format-filter");
  const sidebar = $(".sidebar"),
    menu = $("#menu-toggle"),
    scrim = $(".nav-scrim");
  const mobile = matchMedia("(max-width: 900px)");
  const storageKey = "longcontext-report-controls";
  let state = { model: "all", context: "all", format: "all", unit: "speedup" },
    printState;
  try {
    Object.assign(
      state,
      JSON.parse(sessionStorage.getItem(storageKey) || "{}"),
    );
  } catch {}
  const panels = [...document.querySelectorAll("[data-study]")];
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const controls = $("#shared-controls");
  const labels = {
    setup: "Setup & Method",
    main: "Main Results",
    all: "All Results",
    weights: "Weight Quantization",
    parallel: "Parallel Drafting",
  };
  const evidence = {
    setup: "validation",
    main: "main-campaign-details",
    all: "validation",
    weights: "ffn-validation-and-evidence",
    parallel: "jacobi-evidence",
  };
  let active = "main";
  const current = () => panels.find((p) => p.dataset.study === active);
  const hash = () => {
    try {
      return decodeURIComponent(location.hash.slice(1));
    } catch {
      return "";
    }
  };
  function remember() {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(state));
    } catch {}
  }
  for (const [name, control] of [
    ["model", model],
    ["context", context],
    ["format", format],
  ]) {
    if (!control) continue;
    if (![...control.options].some((o) => o.value === state[name]))
      state[name] = "all";
    control.value = state[name];
    control.addEventListener("change", () => {
      state[name] = control.value;
      filter();
      remember();
    });
  }
  function lengthGroup(value) {
    const clean = value.replaceAll(",", "");
    if (/129024|129073|129079|130560|131072|128K/.test(clean)) return "128k";
    if (/65536|64K/.test(clean)) return "64k";
    if (/32768|32K/.test(clean)) return "32k";
    if (/4096|4K/.test(clean)) return "4k";
    return "";
  }
  // Generated table replacements retain context labels, even without data-context.
  document.querySelectorAll("tbody[data-context-group]").forEach((g) => {
    g.dataset.context ||= lengthGroup(
      g.querySelector(".context-row")?.textContent || "",
    );
  });
  document.querySelectorAll("table").forEach((table) => {
    const headers = [...table.querySelectorAll("thead tr:first-child th")];
    const column = headers.findIndex((h) =>
      /^(Context|Input tokens|Haystack|Length)/.test(h.textContent.trim()),
    );
    if (column >= 0)
      table.querySelectorAll("tbody tr:not(.context-row)").forEach((row) => {
        row.dataset.context ||= lengthGroup(
          row.cells[column]?.textContent || "",
        );
      });
  });
  // Keep the curated view aligned with the canonical tables after publication.
  document.querySelectorAll("[data-result-source]").forEach((cell) => {
    const source = document.querySelector(cell.dataset.resultSource);
    if (!source) return;
    let value = source.textContent.trim().replace(/\s+/g, " ");
    if (cell.dataset.resultExtract === "percentage")
      value = value.match(/\(([\d.]+%)\)/)?.[1] || value;
    cell.textContent = !value || /^n\/a$/i.test(value) ? "N/A" : value;
    cell.classList.remove("pending", "empty-cell");
    cell.classList.toggle("na", cell.textContent === "N/A");
    for (const attribute of ["data-speedup", "data-absolute", "title"]) {
      if (source.hasAttribute(attribute))
        cell.setAttribute(attribute, source.getAttribute(attribute));
      else cell.removeAttribute(attribute);
    }
    if (cell.dataset.resultAbsoluteSource) {
      const absolute = document.querySelector(
        cell.dataset.resultAbsoluteSource,
      );
      if (absolute) {
        cell.dataset.speedup = cell.textContent;
        cell.dataset.absolute = absolute.textContent.trim();
      }
    }
  });
  const hasIndexerFilter = () => active === "all";
  function mismatches(data, all) {
    return (
      !all &&
      ["model", "context", "format"].some(
        (name) =>
          data[name] &&
          data[name] !== "shared" &&
          state[name] !== "all" &&
          (name !== "format" || hasIndexerFilter()) &&
          data[name] !== state[name],
      )
    );
  }
  function filter(panel = current(), all = false) {
    panel.querySelectorAll("[data-context-column]").forEach((cell) => {
      cell.hidden =
        !all &&
        state.context !== "all" &&
        cell.dataset.contextColumn !== state.context;
    });
    panel
      .querySelectorAll("tr[data-model],tr[data-format],tr[data-context]")
      .forEach((r) => (r.hidden = mismatches(r.dataset, all)));
    panel.querySelectorAll("tbody[data-context-group]").forEach((g) => {
      const rows = [...g.querySelectorAll("tr[data-result-row]")];
      g.hidden =
        mismatches(g.dataset, all) ||
        (rows.length > 0 && rows.every((r) => r.hidden));
    });
    panel
      .querySelectorAll("[data-model-block]")
      .forEach(
        (b) =>
          (b.hidden =
            !all &&
            state.model !== "all" &&
            b.dataset.modelBlock !== state.model),
      );
    panel.querySelectorAll(".table-wrap").forEach((wrap) => {
      const rows = [...wrap.querySelectorAll("tbody tr:not(.context-row)")];
      const empty =
        mismatches(wrap.dataset, all) ||
        (rows.length > 0 &&
          rows.every((r) => r.hidden || r.closest("tbody")?.hidden));
      wrap.hidden = empty;
      let message = wrap.nextElementSibling;
      if (!message?.classList.contains("empty-filter")) {
        message = document.createElement("p");
        message.className = "empty-filter";
        message.textContent = "No measurements match these filters.";
        wrap.after(message);
      }
      message.hidden = !empty;
    });
    const feedback = $("#filter-feedback");
    if (feedback) {
      const selection = [model, context, format]
        .filter(
          (c) => c && c.value !== "all" && (c !== format || hasIndexerFilter()),
        )
        .map((c) => c.selectedOptions[0].textContent);
      feedback.textContent = selection.length
        ? "Showing " + selection.join(" · ") + "."
        : "All configurations shown.";
    }
  }
  function displayUnit(value) {
    state.unit = value === "absolute" ? "absolute" : "speedup";
    document
      .querySelectorAll("[data-throughput-mode]")
      .forEach((b) =>
        b.setAttribute(
          "aria-pressed",
          String(b.dataset.throughputMode === state.unit),
        ),
      );
    document.querySelectorAll("[data-speedup][data-absolute]").forEach((c) => {
      c.textContent =
        state.unit === "speedup" ? c.dataset.speedup : c.dataset.absolute;
      const missing = /^n\/a$/i.test(c.textContent.trim());
      c.classList.toggle("na", missing);
      c.classList.toggle("speedup", state.unit === "speedup" && !missing);
      c.classList.toggle(
        "slower",
        state.unit === "speedup" && parseFloat(c.dataset.speedup) < 1,
      );
    });
    document
      .querySelectorAll("[data-throughput-unit]")
      .forEach(
        (l) =>
          (l.textContent = state.unit === "speedup" ? "× dense" : "tokens/s"),
      );
  }
  $("#reset-filters")?.addEventListener("click", () => {
    state = { model: "all", context: "all", format: "all", unit: "speedup" };
    [model, context, format].filter(Boolean).forEach((c) => (c.value = "all"));
    filter();
    displayUnit(state.unit);
    remember();
  });
  document.querySelectorAll("[data-throughput-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      displayUnit(b.dataset.throughputMode);
      remember();
    }),
  );
  function activate(key) {
    active = key;
    panels.forEach((p) => (p.hidden = p.dataset.study !== key));
    document
      .querySelectorAll("[data-study-nav]")
      .forEach((n) => (n.hidden = n.dataset.studyNav !== key));
    tabs.forEach((t) => {
      const selected = t.id === "tab-" + key;
      t.setAttribute("aria-selected", String(selected));
      t.tabIndex = selected ? 0 : -1;
    });
    document.body.dataset.activeStudy = key;
    $("#contents-label").textContent = labels[key];
    $("#study-evidence-link").href = "#" + evidence[key];
    document.title = labels[key] + " | Long-context decoding";
    current().querySelector(".view-controls-slot").append(controls);
    controls.hidden = key === "setup";
    format.closest(".filter-field").hidden = !hasIndexerFilter();
    controls.querySelector(".segmented").hidden = key === "parallel";
    filter(current(), key === "setup");
    displayUnit(state.unit);
  }
  function closeMenu(focus = false) {
    sidebar.classList.remove("open");
    scrim.hidden = true;
    sidebar.inert = mobile.matches;
    menu.setAttribute("aria-expanded", "false");
    if (focus) menu.focus();
  }
  menu.addEventListener("click", () => {
    if (sidebar.classList.contains("open")) return closeMenu(true);
    sidebar.inert = false;
    sidebar.classList.add("open");
    scrim.hidden = false;
    menu.setAttribute("aria-expanded", "true");
    sidebar.querySelector("a").focus();
  });
  scrim.addEventListener("click", () => closeMenu(true));
  mobile.addEventListener("change", () => closeMenu());
  document.addEventListener("keydown", (e) => {
    if (!mobile.matches || !sidebar.classList.contains("open")) return;
    if (e.key === "Escape") return closeMenu(true);
    if (e.key === "Tab") {
      const links = [...sidebar.querySelectorAll("a")].filter(
        (a) => !a.closest("[hidden]"),
      );
      if (e.shiftKey && document.activeElement === links[0]) {
        e.preventDefault();
        links.at(-1).focus();
      } else if (!e.shiftKey && document.activeElement === links.at(-1)) {
        e.preventDefault();
        links[0].focus();
      }
    }
  });
  function reveal(scroll = true) {
    const target = document.getElementById(hash());
    const panel = target?.closest("[data-study]");
    if (panel) activate(panel.dataset.study);
    else if (!location.hash) activate("main");
    if (!target) return;
    for (let n = target; n; n = n.parentElement)
      if (n.tagName === "DETAILS") n.open = true;
    if (target.closest("[hidden]")) $("#reset-filters").click();
    if (scroll)
      requestAnimationFrame(() => {
        if (target.matches("[data-study],.view-header"))
          window.scrollTo({ top: 0, behavior: "instant" });
        else target.scrollIntoView({ block: "start", behavior: "instant" });
      });
  }
  function navigate(id) {
    if (location.hash !== "#" + id) history.pushState(null, "", "#" + id);
    reveal();
    closeMenu();
  }
  window.addEventListener("hashchange", () => reveal());
  window.addEventListener("popstate", () => reveal());
  document.addEventListener("click", (e) => {
    const link = e.target.closest('a[href^="#"]');
    if (
      !link ||
      e.ctrlKey ||
      e.metaKey ||
      e.shiftKey ||
      e.altKey ||
      e.button !== 0
    )
      return;
    const id = link.getAttribute("href").slice(1);
    const target = document.getElementById(id);
    if (!target) return;
    e.preventDefault();
    navigate(id);
    if (
      link.getAttribute("role") !== "tab" &&
      (link.closest(".sidebar") ||
        e.detail === 0 ||
        link.classList.contains("skip-link"))
    ) {
      const focus = target.querySelector("h1,h2,h3") || target;
      focus.tabIndex = -1;
      requestAnimationFrame(() => focus.focus({ preventScroll: true }));
    }
  });
  tabs.forEach((tab, index) =>
    tab.addEventListener("keydown", (e) => {
      let next;
      if (e.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (e.key === "ArrowLeft")
        next = (index + tabs.length - 1) % tabs.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = tabs.length - 1;
      else if (e.key === " ") {
        e.preventDefault();
        navigate(tab.hash.slice(1));
        return;
      } else return;
      e.preventDefault();
      navigate(tabs[next].hash.slice(1));
      tabs[next].focus();
    }),
  );
  document.querySelectorAll(".section-heading h2,h3[id]").forEach((h) => {
    const a = document.createElement("a");
    a.className = "permalink";
    a.href = "#" + (h.id || h.closest("section").id);
    a.setAttribute("aria-label", "Link to " + h.textContent.trim());
    a.textContent = "#";
    h.append(a);
  });
  const observer = new IntersectionObserver(
    (entries) => {
      const entry = entries.find(
        (e) =>
          e.isIntersecting && e.target.closest("[data-study]") === current(),
      );
      if (!entry) return;
      sidebar.querySelectorAll("nav a").forEach((a) => {
        if (a.hash === "#" + entry.target.id)
          a.setAttribute("aria-current", "location");
        else a.removeAttribute("aria-current");
      });
    },
    { rootMargin: "-140px 0px -60% 0px", threshold: 0 },
  );
  document
    .querySelectorAll("article > section")
    .forEach((s) => observer.observe(s));
  $("#refresh-report").addEventListener("click", () => location.reload());
  $("#print-report").addEventListener("click", () => window.print());
  window.addEventListener("beforeprint", () => {
    if (printState) return;
    printState = {
      active,
      y: scrollY,
      details: [...document.querySelectorAll("details")].map((d) => [
        d,
        d.open,
      ]),
    };
    const all = $("#print-scope").value === "all";
    panels.forEach((p) => {
      p.hidden = !all && p.dataset.study !== active;
      if (!p.hidden) {
        filter(p, true);
        p.querySelectorAll("details").forEach((d) => (d.open = true));
      }
    });
    document.body.classList.add("printing");
  });
  window.addEventListener("afterprint", () => {
    if (!printState) return;
    const saved = printState;
    printState = null;
    saved.details.forEach(([d, open]) => (d.open = open));
    document.body.classList.remove("printing");
    activate(saved.active);
    window.scrollTo({ top: saved.y, behavior: "instant" });
  });
  activate("main");
  closeMenu();
  reveal(Boolean(location.hash));
  document.documentElement.classList.add("report-ready");
})();
