import { state, notify } from "../state.ts";
import { el, quoteCommandArg, splitCommand } from "../util.ts";
import type { LaunchConfig, ProcessStatus } from "../types.ts";

const LS_KEY = "reachy-dashboard.launch-config";
const LEGACY_HOST = "reachy-dashboard.robot-host";
const LEGACY_PORT = "reachy-dashboard.robot-port";

export function persistLaunchConfig(): void {
  try {
    window.localStorage.setItem(LS_KEY, JSON.stringify(state.launchConfig));
  } catch {
    // storage unavailable; keep in-memory config
  }
}

export function loadPersistedLaunchConfig(): void {
  const raw = window.localStorage.getItem(LS_KEY);
  if (raw) {
    try {
      const parsed = JSON.parse(raw) as Partial<LaunchConfig>;
      state.launchConfig = { ...state.launchConfig, ...parsed };
      state.defaultsSeeded = true;
      return;
    } catch {
      // fall through to legacy migration
    }
  }
  const host = window.localStorage.getItem(LEGACY_HOST);
  const port = window.localStorage.getItem(LEGACY_PORT);
  if (host) state.launchConfig.robotHost = host;
  if (port) state.launchConfig.robotPort = port;
  if (host || port) {
    persistLaunchConfig();
    window.localStorage.removeItem(LEGACY_HOST);
    window.localStorage.removeItem(LEGACY_PORT);
    state.defaultsSeeded = true;
  }
}

export function seedLaunchDefaults(process: ProcessStatus): void {
  if (state.defaultsSeeded) return;
  const cfg = state.launchConfig;
  if (!cfg.robotHost && process.defaultRobotHost) cfg.robotHost = process.defaultRobotHost;
  if (!cfg.robotPort && process.defaultRobotPort) cfg.robotPort = process.defaultRobotPort;
  if (!cfg.robotName && process.defaultRobotName) cfg.robotName = process.defaultRobotName;
  const tracker = (process.defaultHeadTracker || "").toLowerCase();
  if (cfg.headTracker === "off" && (tracker === "yolo" || tracker === "mediapipe")) {
    cfg.headTracker = tracker;
  }
  state.defaultsSeeded = true;
}

function baseTokens(defaultCommand: string): string[] {
  let tokens: string[] = [];
  try {
    tokens = splitCommand(defaultCommand);
  } catch {
    tokens = [];
  }
  const base: string[] = [];
  for (const token of tokens) {
    if (token.startsWith("--")) break;
    base.push(token);
  }
  if (!base.length) return ["uv", "run", "python", "-m", "reachy_mini_conversation_app.main"];
  return base;
}

export function buildLaunchCommand(cfg: LaunchConfig, defaultCommand: string): string {
  if (cfg.rawOverride.trim()) return cfg.rawOverride.trim();
  const args: string[] = ["--connection-mode", cfg.connectionMode];
  if (cfg.robotHost.trim()) args.push("--robot-host", cfg.robotHost.trim());
  if (cfg.robotPort.trim()) args.push("--robot-port", cfg.robotPort.trim());
  if (cfg.robotName.trim()) args.push("--robot-name", cfg.robotName.trim());
  if (cfg.headTracker !== "off") args.push("--head-tracker", cfg.headTracker);
  if (cfg.mediaBackend !== "auto") args.push("--media-backend", cfg.mediaBackend);
  if (cfg.hardwareProfile !== "auto") args.push("--hardware-profile", cfg.hardwareProfile);
  if (!cfg.camera) args.push("--no-camera");
  if (cfg.localVision) args.push("--local-vision");
  if (cfg.debug) args.push("--debug");
  return [...baseTokens(defaultCommand), ...args].map(quoteCommandArg).join(" ");
}

export function currentCommand(): string {
  return buildLaunchCommand(state.launchConfig, state.process?.defaultCommand || "");
}

export function portError(): string | null {
  const port = state.launchConfig.robotPort.trim();
  if (!port) return null;
  const parsed = Number.parseInt(port, 10);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 65535) return "Enter a valid robot port (1–65535).";
  return null;
}

function selectControl(value: string, options: Array<[string, string]>): HTMLSelectElement {
  const select = el("select");
  for (const [optionValue, label] of options) {
    select.append(el("option", { value: optionValue, text: label }));
  }
  select.value = value;
  return select;
}

function fieldBlock(labelText: string, control: HTMLElement, span = false): HTMLElement {
  return el("div", { class: span ? "field span-2" : "field" }, [
    el("label", { class: "field-label", text: labelText }),
    control,
  ]);
}

function toggleRow(labelText: string, checked: boolean, onChange: (value: boolean) => void): HTMLElement {
  const input = el("input", { type: "checkbox", checked });
  input.addEventListener("change", () => onChange(input.checked));
  return el("label", { class: "toggle" }, [
    el("span", { text: labelText }),
    el("span", {}, [input, el("span", { class: "toggle-track" })]),
  ]);
}

export function renderLaunchForm(opts: { variant: "compact" | "full"; onChange?: () => void }): HTMLElement {
  const cfg = state.launchConfig;
  const full = opts.variant === "full";

  const preview = el("pre");
  const copyBtn = el("button", { class: "copy-btn", type: "button", title: "Copy command", "aria-label": "Copy command", text: "⧉" });
  copyBtn.addEventListener("click", () => {
    void navigator.clipboard?.writeText(currentCommand()).then(() => {
      copyBtn.textContent = "✓";
      window.setTimeout(() => (copyBtn.textContent = "⧉"), 1200);
    });
  });
  const previewBlock = el("div", { class: "command-preview" }, [preview, copyBtn]);

  function changed(): void {
    persistLaunchConfig();
    preview.textContent = currentCommand();
    notify();
    opts.onChange?.();
  }

  const connection = selectControl(cfg.connectionMode, [
    ["network", "network"],
    ["auto", "auto"],
    ["localhost_only", "localhost only"],
  ]);
  connection.addEventListener("change", () => {
    cfg.connectionMode = connection.value as LaunchConfig["connectionMode"];
    changed();
  });

  const host = el("input", { type: "text", value: cfg.robotHost, placeholder: "192.168.1.42", autocomplete: "off", spellcheck: "false" });
  host.addEventListener("input", () => {
    cfg.robotHost = host.value;
    changed();
  });

  const port = el("input", { type: "number", value: cfg.robotPort, min: "1", max: "65535", placeholder: "8000" });
  port.addEventListener("input", () => {
    cfg.robotPort = port.value;
    changed();
  });

  const headTracker = selectControl(cfg.headTracker, [
    ["off", "Disabled"],
    ["yolo", "yolo"],
    ["mediapipe", "mediapipe"],
  ]);
  headTracker.addEventListener("change", () => {
    cfg.headTracker = headTracker.value as LaunchConfig["headTracker"];
    changed();
  });

  const grid = el("div", { class: full ? "form-grid" : "" }, [
    fieldBlock("Connection mode", connection, full),
    el("div", { class: "host-port span-2" }, [
      fieldBlock("Robot host", host),
      fieldBlock("Port", port),
    ]),
    fieldBlock("Head tracker", headTracker, full),
  ]);

  if (full) {
    const robotName = el("input", { type: "text", value: cfg.robotName, placeholder: "optional", autocomplete: "off", spellcheck: "false" });
    robotName.addEventListener("input", () => {
      cfg.robotName = robotName.value;
      changed();
    });

    const mediaBackend = selectControl(cfg.mediaBackend, [
      ["auto", "auto"],
      ["default", "default"],
      ["local", "local"],
      ["webrtc", "webrtc"],
      ["no_media", "no_media"],
    ]);
    mediaBackend.addEventListener("change", () => {
      cfg.mediaBackend = mediaBackend.value as LaunchConfig["mediaBackend"];
      changed();
    });

    const hardwareProfile = selectControl(cfg.hardwareProfile, [
      ["auto", "auto"],
      ["mac-mini-wired", "mac-mini-wired"],
      ["legacy", "legacy"],
    ]);
    hardwareProfile.addEventListener("change", () => {
      cfg.hardwareProfile = hardwareProfile.value as LaunchConfig["hardwareProfile"];
      changed();
    });

    grid.append(
      fieldBlock("Robot name", robotName),
      fieldBlock("Media backend", mediaBackend),
      fieldBlock("Hardware profile", hardwareProfile, true),
    );
  }

  const toggles = el("div", { class: "col-stack", style: "gap:0;margin-top:4px;" }, [
    toggleRow("Camera enabled", cfg.camera, (value) => {
      cfg.camera = value;
      changed();
    }),
  ]);
  if (full) {
    toggles.append(
      toggleRow("Local vision (--local-vision)", cfg.localVision, (value) => {
        cfg.localVision = value;
        changed();
      }),
      toggleRow("Debug logging (--debug)", cfg.debug, (value) => {
        cfg.debug = value;
        changed();
      }),
    );
  }

  const wrap = el("div", { class: "col-stack", style: "gap:12px;" }, [grid, toggles]);

  if (full) {
    const rawText = el("textarea", { placeholder: "Leave empty to use the generated command", rows: "2" });
    rawText.value = cfg.rawOverride;
    rawText.addEventListener("input", () => {
      cfg.rawOverride = rawText.value;
      changed();
    });
    wrap.append(
      el("details", { class: "disclosure" }, [
        el("summary", { text: "Advanced: raw command override" }),
        el("p", { class: "muted", style: "margin-bottom:8px;", text: "If set, this exact command runs instead of the generated one." }),
        rawText,
      ]),
    );
  }

  wrap.append(
    el("div", {}, [
      el("div", { class: "field-label", text: "Command preview" }),
      previewBlock,
    ]),
  );

  preview.textContent = currentCommand();
  return wrap;
}
