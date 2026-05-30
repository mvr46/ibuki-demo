import { state, logUi, onLog, addLocalLog } from "../state.ts";
import { el } from "../util.ts";
import type { LevelFilter, LogEntry, View } from "../types.ts";

const MAX_VISIBLE_LOGS = 300;
const SCROLL_BOTTOM_THRESHOLD = 40;
const USER_SCROLL_WINDOW_MS = 400;
const FILTER_LABELS: Record<LevelFilter, string> = { ALL: "All", INFO: "Info", WARNING: "Warn", ERROR: "Error" };

function entryLevel(entry: LogEntry): string {
  return (entry.level || "INFO").toUpperCase();
}
function entryCategory(entry: LogEntry): string {
  return (entry.category || "SYSTEM").toUpperCase();
}
function formatLine(entry: LogEntry): string {
  const time = entry.createdAt ? new Date(entry.createdAt).toISOString() : "";
  return `${time}\t${entryLevel(entry)}\t${entryCategory(entry)}\t${entry.message}`;
}

export function createLogsView(): View {
  let unsubLog: (() => void) | null = null;
  let lastUserScrollAt = 0;
  let categorySig = "";

  const logSearch = el("input", { id: "log-search", type: "search", placeholder: "Filter logs...", autocomplete: "off", spellcheck: "false" });

  const filterTabs = (Object.keys(FILTER_LABELS) as LevelFilter[]).map((level) => {
    const tab = el("button", { class: "filter-tab", type: "button", text: FILTER_LABELS[level] });
    tab.dataset.level = level;
    if (level === logUi.filter) tab.classList.add("is-active");
    tab.addEventListener("click", () => {
      logUi.filter = level;
      filterTabs.forEach((other) => other.classList.toggle("is-active", other === tab));
      renderLogs();
    });
    return tab;
  });
  const filters = el("div", { class: "log-filters", role: "tablist" }, filterTabs);

  const categorySelect = el("select", { class: "log-category", "aria-label": "Category" });
  categorySelect.addEventListener("change", () => { logUi.category = categorySelect.value; renderLogs(); });

  const countInfo = el("span", { class: "count count-info", text: "0" });
  const countWarn = el("span", { class: "count count-warn", text: "0" });
  const countErr = el("span", { class: "count count-err", text: "0" });
  const counts = el("div", { class: "log-counts" }, [countInfo, countWarn, countErr]);

  const copyBtn = el("button", { class: "link-btn", type: "button", text: "Copy" });
  copyBtn.addEventListener("click", () => void copyVisible());
  const exportBtn = el("button", { class: "link-btn", type: "button", text: "Export" });
  exportBtn.addEventListener("click", () => exportVisible());
  const clearBtn = el("button", { class: "link-btn", type: "button", text: "Clear" });
  clearBtn.addEventListener("click", () => { logUi.cleared = state.logs.length; renderLogs(); });
  const actions = el("div", { class: "log-actions" }, [copyBtn, exportBtn, clearBtn]);

  const toolbar = el("div", { class: "log-toolbar" }, [logSearch, filters, categorySelect, counts, actions]);

  const logsList = el("ol", { id: "logs" });
  const jumpCount = el("span", { text: "0" });
  const jumpLatest = el("button", { class: "jump-latest", type: "button", hidden: true }, [
    el("span", { "aria-hidden": "true", text: "↓" }),
    jumpCount,
    el("span", { text: "new" }),
  ]);
  const logBody = el("div", { class: "log-body" }, [logsList, jumpLatest]);

  logSearch.addEventListener("input", () => { logUi.search = logSearch.value; renderLogs(); });
  jumpLatest.addEventListener("click", () => {
    logUi.autoScroll = true;
    logUi.newSincePaused = 0;
    jumpLatest.hidden = true;
    scrollToBottom();
  });

  const intentEvents: Array<keyof HTMLElementEventMap> = ["wheel", "touchstart", "pointerdown", "keydown"];
  intentEvents.forEach((name) => logsList.addEventListener(name, () => { lastUserScrollAt = Date.now(); }, { passive: true }));
  logsList.addEventListener("scroll", () => {
    if (Date.now() - lastUserScrollAt > USER_SCROLL_WINDOW_MS) return;
    const atBottom = isAtBottom();
    if (atBottom && !logUi.autoScroll) { logUi.autoScroll = true; logUi.newSincePaused = 0; updateJumpLatest(); }
    else if (!atBottom && logUi.autoScroll) { logUi.autoScroll = false; updateJumpLatest(); }
  });

  function refreshCategories(): void {
    const seen = new Set<string>();
    for (let i = logUi.cleared; i < state.logs.length; i += 1) seen.add(entryCategory(state.logs[i]));
    const cats = [...seen].sort();
    const sig = cats.join(",");
    if (sig === categorySig) return;
    categorySig = sig;
    if (logUi.category !== "ALL" && !seen.has(logUi.category)) logUi.category = "ALL";
    categorySelect.replaceChildren(
      el("option", { value: "ALL", text: "All categories" }),
      ...cats.map((cat) => el("option", { value: cat, text: cat.charAt(0) + cat.slice(1).toLowerCase() })),
    );
    categorySelect.value = logUi.category;
  }
  function entryPasses(entry: LogEntry): boolean {
    const search = logUi.search.trim().toLowerCase();
    if (logUi.filter !== "ALL" && entryLevel(entry) !== logUi.filter) return false;
    if (logUi.category !== "ALL" && entryCategory(entry) !== logUi.category) return false;
    if (search && !entry.message.toLowerCase().includes(search)) return false;
    return true;
  }
  function visibleLogs(): LogEntry[] {
    const out: LogEntry[] = [];
    for (let i = logUi.cleared; i < state.logs.length; i += 1) {
      if (entryPasses(state.logs[i])) out.push(state.logs[i]);
    }
    return out;
  }
  function renderLog(entry: LogEntry): HTMLLIElement {
    const level = entryLevel(entry);
    const category = entryCategory(entry);
    const item = el("li", {}, [
      el("time", { text: entry.createdAt ? new Date(entry.createdAt).toLocaleTimeString() : "" }),
      el("span", { class: `cat cat--${category.toLowerCase()}`, text: category }),
      el("span", { class: "log-message", text: entry.message }),
    ]);
    item.dataset.level = level;
    item.dataset.category = category;
    return item;
  }
  function setCount(node: HTMLElement, count: number): void {
    node.textContent = String(count);
    node.classList.toggle("has-items", count > 0);
  }
  function updateCounts(): void {
    let info = 0;
    let warn = 0;
    let err = 0;
    for (let i = logUi.cleared; i < state.logs.length; i += 1) {
      const level = entryLevel(state.logs[i]);
      if (level === "ERROR") err += 1;
      else if (level === "WARNING") warn += 1;
      else info += 1;
    }
    setCount(countInfo, info);
    setCount(countWarn, warn);
    setCount(countErr, err);
  }
  function showEmpty(): void {
    logsList.replaceChildren(el("li", { class: "log-empty", text: "Waiting for events..." }));
  }
  function isAtBottom(): boolean {
    return logsList.scrollHeight - logsList.scrollTop - logsList.clientHeight < SCROLL_BOTTOM_THRESHOLD;
  }
  function scrollToBottom(): void {
    logsList.scrollTop = logsList.scrollHeight;
  }
  function updateJumpLatest(): void {
    if (logUi.autoScroll) {
      jumpLatest.hidden = true;
      logUi.newSincePaused = 0;
    } else if (logUi.newSincePaused > 0) {
      jumpCount.textContent = String(logUi.newSincePaused);
      jumpLatest.hidden = false;
    } else {
      jumpLatest.hidden = true;
    }
  }
  function renderLogs(): void {
    refreshCategories();
    updateCounts();
    const items = visibleLogs().slice(-MAX_VISIBLE_LOGS);
    if (!items.length) showEmpty();
    else logsList.replaceChildren(...items.map(renderLog));
    if (logUi.autoScroll) {
      scrollToBottom();
      logUi.newSincePaused = 0;
    }
    updateJumpLatest();
  }
  function appendToDom(entry: LogEntry): void {
    if (!entryPasses(entry)) return;
    const first = logsList.firstElementChild;
    if (first && first.classList.contains("log-empty")) logsList.replaceChildren();
    logsList.appendChild(renderLog(entry));
    while (logsList.childElementCount > MAX_VISIBLE_LOGS) logsList.removeChild(logsList.firstElementChild!);
  }
  function onNewLog(entry: LogEntry): void {
    refreshCategories();
    updateCounts();
    appendToDom(entry);
    if (logUi.autoScroll) scrollToBottom();
    else if (entryPasses(entry)) logUi.newSincePaused += 1;
    updateJumpLatest();
  }

  async function copyVisible(): Promise<void> {
    const text = visibleLogs().map(formatLine).join("\n");
    if (!text) return;
    try {
      await navigator.clipboard?.writeText(text);
      copyBtn.textContent = "Copied";
      window.setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
    } catch {
      addLocalLog("Clipboard unavailable; use Export instead.", "WARNING");
    }
  }
  function exportVisible(): void {
    const text = visibleLogs().map(formatLine).join("\n");
    if (!text) return;
    const blob = new Blob([`${text}\n`], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const anchor = el("a", { href: url, download: `reachy-logs-${stamp}.log` });
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  return {
    mount(container) {
      container.append(el("div", { class: "log-view" }, [toolbar, logBody]));
      renderLogs();
      unsubLog = onLog(onNewLog);
    },
    onEnter() {
      renderLogs();
    },
    update() {},
    destroy() {
      unsubLog?.();
      unsubLog = null;
    },
  };
}
