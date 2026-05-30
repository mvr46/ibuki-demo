import { state } from "../state.ts";
import { el } from "../util.ts";
import type { LaunchOptions } from "../types.ts";

const LS_KEY = "reachy-dashboard.launch-options";
const FALLBACK_COMMAND = "REACHY_DASHBOARD_SERVER=1 uv run reachy-mini-conversation-app";

export function persistLaunchOptions(): void {
  try {
    window.localStorage.setItem(LS_KEY, JSON.stringify(state.launchOptions));
  } catch {
    // storage unavailable; keep in-memory options
  }
}

export function loadPersistedLaunchOptions(): void {
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as Partial<LaunchOptions>;
    state.launchOptions = {
      camera: parsed.camera ?? true,
      localVision: parsed.localVision ?? false,
      debug: parsed.debug ?? false,
    };
  } catch {
    // ignore malformed storage; keep defaults
  }
}

function baseCommand(): string {
  return state.process?.defaultCommand?.trim() || FALLBACK_COMMAND;
}

export function currentCommand(): string {
  const parts = [baseCommand()];
  const opts = state.launchOptions;
  if (!opts.camera) parts.push("--no-camera");
  if (opts.localVision) parts.push("--local-vision");
  if (opts.debug) parts.push("--debug");
  return parts.join(" ");
}

function optionRow(labelText: string, description: string, checked: boolean, onChange: (value: boolean) => void): HTMLElement {
  const input = el("input", { type: "checkbox", checked });
  input.addEventListener("change", () => onChange(input.checked));
  return el("label", { class: "toggle" }, [
    el("span", { class: "toggle-text" }, [
      el("span", { class: "toggle-name", text: labelText }),
      el("span", { class: "toggle-desc", text: description }),
    ]),
    el("span", { class: "toggle-switch" }, [input, el("span", { class: "toggle-track" })]),
  ]);
}

export function renderAppOptions(): HTMLElement {
  const opts = state.launchOptions;

  const preview = el("pre");
  const copyBtn = el("button", { class: "copy-btn", type: "button", title: "Copy command", "aria-label": "Copy command", text: "Copy" });
  copyBtn.addEventListener("click", () => {
    void navigator.clipboard?.writeText(currentCommand()).then(() => {
      copyBtn.textContent = "Copied";
      window.setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
    });
  });
  const previewBlock = el("div", { class: "command-preview" }, [preview, copyBtn]);

  function changed(): void {
    persistLaunchOptions();
    preview.textContent = currentCommand();
  }

  const toggles = el("div", { class: "option-list" }, [
    optionRow("Camera", "Stream the robot camera for vision and face tools.", opts.camera, (value) => {
      opts.camera = value;
      changed();
    }),
    optionRow("Local vision", "Run the on-device VLM for scene questions (--local-vision).", opts.localVision, (value) => {
      opts.localVision = value;
      changed();
    }),
    optionRow("Debug logging", "Emit verbose logs from the robot app (--debug).", opts.debug, (value) => {
      opts.debug = value;
      changed();
    }),
  ]);

  const disclosure = el("details", { class: "disclosure" }, [
    el("summary", { text: "Command preview" }),
    previewBlock,
  ]);

  preview.textContent = currentCommand();
  return el("div", { class: "col-stack", style: "gap:14px;" }, [toggles, disclosure]);
}
