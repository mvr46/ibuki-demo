import { state, subscribe, notify, addLocalLog, appPhase } from "../state.ts";
import { api } from "../api.ts";
import { el, setDot } from "../util.ts";
import { currentCommand } from "./launchConfig.ts";
import type { AppPhase, DotState } from "../types.ts";

export async function startProcess(): Promise<void> {
  const command = currentCommand();
  if (!command.trim()) {
    addLocalLog("No command to run.", "WARNING", "PROCESS");
    return;
  }
  try {
    state.process = await api.processStart(command);
    state.processAvailable = true;
    addLocalLog("Starting local robot app.", "INFO", "PROCESS");
  } catch (error) {
    addLocalLog(error instanceof Error ? error.message : "Failed to start.", "ERROR", "PROCESS");
  }
  notify();
}

export async function stopProcess(): Promise<void> {
  try {
    state.process = await api.processStop();
    state.processAvailable = true;
    addLocalLog("Stopping local robot app.", "INFO", "PROCESS");
  } catch (error) {
    addLocalLog(error instanceof Error ? error.message : "Failed to stop.", "ERROR", "PROCESS");
  }
  notify();
}

type PhaseView = {
  label: string;
  tone: DotState;
  detail?: string;
  action: "start" | "stop";
  actionLabel: string;
  danger: boolean;
};

function shortFailure(): string {
  const process = state.process;
  if (process?.failureHint) {
    const head = process.failureHint.split(/[:.]/)[0].trim();
    return head || "Robot unavailable";
  }
  if (process?.exitCode != null && process.exitCode !== 0) return `Exited (code ${process.exitCode})`;
  return "Stopped with error";
}

function phaseView(phase: AppPhase): PhaseView {
  const pid = state.process?.pid;
  switch (phase) {
    case "starting":
      return { label: "Starting…", tone: "warn", detail: "waiting for app API", action: "stop", actionLabel: "Stop", danger: true };
    case "running":
      return { label: "Running", tone: "ok", detail: pid ? `pid ${pid}` : undefined, action: "stop", actionLabel: "Stop", danger: true };
    case "failed":
      return { label: shortFailure(), tone: "err", action: "start", actionLabel: "Start local robot app", danger: false };
    case "stopped":
      return { label: "Stopped", tone: "idle", action: "start", actionLabel: "Start local robot app", danger: false };
    default:
      return { label: "Idle", tone: "idle", action: "start", actionLabel: "Start local robot app", danger: false };
  }
}

export function createStartControl(): HTMLElement {
  const dot = el("span", { class: "status-dot" });
  const labelText = el("span", { class: "start-label" });
  const detailText = el("span", { class: "start-detail" });
  const stateBox = el("div", { class: "start-state" }, [dot, labelText, detailText]);

  const button = el("button", { class: "btn", type: "button" });
  const group = el("div", { class: "start-control" }, [stateBox, button]);

  let onClick: (() => void) | null = null;
  button.addEventListener("click", () => onClick?.());

  function sync(): void {
    const phase = appPhase();
    group.hidden = phase === "unavailable";
    if (phase === "unavailable") return;
    const view = phaseView(phase);

    setDot(dot, view.tone);
    stateBox.classList.toggle("is-busy", phase === "starting");
    labelText.textContent = view.label;
    labelText.title = state.process?.failureHint || "";
    detailText.textContent = view.detail || "";
    detailText.hidden = !view.detail;

    button.textContent = view.actionLabel;
    button.classList.toggle("btn-primary", view.action === "start");
    button.classList.toggle("btn-danger", view.danger);
    onClick = view.action === "start" ? () => void startProcess() : () => void stopProcess();
  }

  subscribe(sync);
  sync();
  return group;
}
