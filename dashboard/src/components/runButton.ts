import { state, subscribe, notify, addLocalLog } from "../state.ts";
import { api } from "../api.ts";
import { el } from "../util.ts";
import { createPopover } from "./popover.ts";
import { renderLaunchForm, currentCommand, portError } from "./launchConfig.ts";

export async function startProcess(): Promise<void> {
  const invalid = portError();
  if (invalid) {
    addLocalLog(invalid, "WARNING", "PROCESS");
    notify();
    return;
  }
  const command = currentCommand();
  if (!command.trim()) {
    addLocalLog("No command to run.", "WARNING", "PROCESS");
    return;
  }
  try {
    state.process = await api.processStart(command);
    state.processAvailable = true;
    addLocalLog(`Started ${command}`, "INFO", "PROCESS");
  } catch (error) {
    addLocalLog(error instanceof Error ? error.message : "Failed to start.", "ERROR", "PROCESS");
  }
  notify();
}

export async function stopProcess(): Promise<void> {
  try {
    state.process = await api.processStop();
    state.processAvailable = true;
    addLocalLog("Stopped process.", "INFO", "PROCESS");
  } catch (error) {
    addLocalLog(error instanceof Error ? error.message : "Failed to stop.", "ERROR", "PROCESS");
  }
  notify();
}

function buildPopoverBody(close: () => void): HTMLElement {
  const running = !!state.process?.running;
  const form = renderLaunchForm({ variant: "compact" });

  const allLink = el("button", { class: "link-btn", type: "button", text: "All launch settings →" });
  allLink.addEventListener("click", () => {
    close();
    window.location.hash = "#/settings";
  });

  const runNow = el("button", {
    class: `btn btn-sm ${running ? "btn-danger" : "btn-primary"}`,
    type: "button",
    text: running ? "Stop" : "Run",
  });
  runNow.addEventListener("click", () => {
    close();
    if (running) void stopProcess();
    else void startProcess();
  });

  return el("div", {}, [
    el("div", { class: "popover-title", text: "Launch" }),
    form,
    el("div", { class: "popover-foot" }, [allLink, el("span", { class: "spacer" }), runNow]),
  ]);
}

export function createRunButton(): HTMLElement {
  const label = el("span", { text: "Run" });
  const runBtn = el("button", { class: "btn btn-primary run-btn", type: "button" }, [label]);
  const caret = el("button", {
    class: "run-caret",
    type: "button",
    "aria-haspopup": "true",
    "aria-expanded": "false",
    "aria-label": "Launch settings",
    text: "▾",
  });
  const group = el("div", { class: "run-group" }, [runBtn, caret]);

  const popover = createPopover({ anchor: caret, align: "right", build: buildPopoverBody });

  runBtn.addEventListener("click", () => {
    if (state.process?.running) void stopProcess();
    else void startProcess();
  });
  caret.addEventListener("click", () => popover.toggle());

  function sync(): void {
    group.hidden = !state.processAvailable;
    const running = !!state.process?.running;
    label.textContent = running ? "Stop" : "Run";
    runBtn.classList.toggle("btn-primary", !running);
    runBtn.classList.toggle("btn-danger", running);
    caret.classList.toggle("is-running", running);
  }

  subscribe(sync);
  sync();
  return group;
}
