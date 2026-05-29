import "./styles.css";

type DotState = "ok" | "warn" | "err" | "idle";
type LevelFilter = "ALL" | "INFO" | "WARNING" | "ERROR";

type BackendStatus = {
  active_backend?: string;
  backend_provider?: string;
  has_hf_connection?: boolean;
  has_hf_session_url?: boolean;
  has_hf_ws_url?: boolean;
  hf_connection_mode?: string;
  hf_direct_host?: string;
  hf_direct_port?: number;
  can_proceed?: boolean;
  can_proceed_with_hf?: boolean;
  can_proceed_with_local?: boolean;
  ollama_base_url?: string;
  ollama_model?: string;
  requires_restart?: boolean;
  backend_unavailable?: boolean;
  backend_unavailable_reason?: string;
};

type PerformanceStatus = {
  daemon_rtt_ms?: number | null;
  daemon_state?: string | null;
  media_state?: Record<string, unknown>;
  transport?: Record<string, unknown>;
  health_checks?: Record<string, unknown>;
  local_model?: Record<string, unknown>;
  local_tts?: Record<string, unknown>;
  voice_activity?: Record<string, unknown>;
  camera_frame_age_ms?: number | null;
  camera_fps?: number | null;
  audio_input_frames?: number;
  audio_output_frames?: number;
  dropped_audio_frames?: number;
  audio_queue_depth_s?: number | null;
  stt_ms?: number | null;
  llm_first_token_ms?: number | null;
  llm_total_ms?: number | null;
  tts_ms?: number | null;
  first_audio_ms?: number | null;
};

type DashboardStatus = BackendStatus & {
  performance?: PerformanceStatus;
  camera: {
    available: boolean;
    frame_available: boolean;
    head_tracker: string | null;
  };
  face_recognition: {
    available: boolean;
    recognition_available?: boolean;
    db_path: string | null;
    visible_count: number;
    people: Array<{ name: string; exemplar_count: number }>;
  };
};

type FaceBox = {
  id: number | null;
  track_id: number | null;
  name: string | null;
  label: string;
  similarity: number;
  x_offset: number;
  y_offset: number;
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
  focused: boolean;
  observed?: boolean;
  held?: boolean;
  stability?: number;
  can_remember?: boolean;
  last_observed_at?: number | null;
};

type FaceState = {
  ok: boolean;
  available: boolean;
  recognition_available?: boolean;
  focus_name: string | null;
  faces: FaceBox[];
};

type LogEntry = {
  id?: number;
  type: string;
  createdAt: string;
  level: string;
  category: string;
  message: string;
};

type ProcessStatus = {
  available: boolean;
  running: boolean;
  pid: number | null;
  command: string;
  defaultCommand: string;
  defaultRobotHost: string;
  defaultRobotPort: string;
  defaultRobotName: string;
  defaultHeadTracker: string;
  startedAt: string | null;
  exitedAt: string | null;
  exitCode: number | null;
  signal: string | null;
  backendTarget: string;
  backendReady?: boolean;
  failureHint?: string | null;
};

type ProfileList = {
  choices: string[];
  current: string;
  startup: string;
  locked?: boolean;
};

type ProfilePayload = {
  instructions: string;
  tools_text: string;
  voice: string;
  available_tools: string[];
  enabled_tools: string[];
};

const HF_BACKEND = "huggingface";
const LOCAL_BACKEND = "local";
const DEFAULT_BACKEND = LOCAL_BACKEND;

const AUTO_WITH: Record<string, string[]> = {
  dance: ["stop_dance"],
  play_emotion: ["stop_emotion"],
};

const state = {
  status: null as DashboardStatus | null,
  process: null as ProcessStatus | null,
  processAvailable: false,
  faces: [] as FaceBox[],
  faceStateAvailable: false,
  faceRecognitionAvailable: false,
  selectedFaceId: null as number | null,
  logs: [] as LogEntry[],
};

const logUi = {
  filter: "ALL" as LevelFilter,
  search: "",
  autoScroll: true,
  newSincePaused: 0,
  cleared: 0,
};

const elements = {
  app: byId("app"),
  cameraDot: byId("camera-dot"),
  faceDot: byId("face-dot"),
  backendDot: byId("backend-dot"),
  transportDot: byId("transport-dot"),
  processDot: byId("process-dot"),
  cameraState: byId("camera-state"),
  faceState: byId("face-state"),
  backendState: byId("backend-state"),
  transportState: byId("transport-state"),
  processState: byId("process-state"),
  processPill: byId("process-pill"),
  processSection: byId("process-section"),
  commandInput: byId<HTMLInputElement>("command-input"),
  robotHost: byId<HTMLInputElement>("robot-host"),
  robotPort: byId<HTMLInputElement>("robot-port"),
  startCommand: byId<HTMLButtonElement>("start-command"),
  stopCommand: byId<HTMLButtonElement>("stop-command"),
  commandStatus: byId("command-status"),
  selectedFace: byId("selected-face"),
  faceName: byId<HTMLInputElement>("face-name"),
  saveFace: byId<HTMLButtonElement>("save-face"),
  faceSaveStatus: byId("face-save-status"),
  peopleList: byId<HTMLOListElement>("people-list"),
  cameraFeed: byId<HTMLImageElement>("camera-feed"),
  faceOverlay: byId("face-overlay"),
  cameraEmpty: byId("camera-empty"),
  cameraHelp: byId("camera-help"),
  refreshFaceState: byId<HTMLButtonElement>("refresh-face-state"),
  diagControlHost: byId("diag-control-host"),
  diagMediaHost: byId("diag-media-host"),
  diagDaemon: byId("diag-daemon"),
  diagMediaState: byId("diag-media-state"),
  diagHealth: byId("diag-health"),
  diagCamera: byId("diag-camera"),
  diagAudio: byId("diag-audio"),
  diagVad: byId("diag-vad"),
  diagNoise: byId("diag-noise"),
  diagRejects: byId("diag-rejects"),
  diagStt: byId("diag-stt"),
  diagLlm: byId("diag-llm"),
  diagRouter: byId("diag-router"),
  diagTts: byId("diag-tts"),
  backendGrid: byId("backend-grid"),
  hfFields: byId("hf-fields"),
  hfMode: byId<HTMLSelectElement>("hf-mode"),
  hfHost: byId<HTMLInputElement>("hf-host"),
  hfPort: byId<HTMLInputElement>("hf-port"),
  saveBackend: byId<HTMLButtonElement>("save-backend"),
  backendStatus: byId("backend-status"),
  profileSelect: byId<HTMLSelectElement>("profile-select"),
  applyProfile: byId<HTMLButtonElement>("apply-profile"),
  persistProfile: byId<HTMLButtonElement>("persist-profile"),
  profileName: byId<HTMLInputElement>("profile-name"),
  instructions: byId<HTMLTextAreaElement>("instructions-ta"),
  toolsText: byId<HTMLTextAreaElement>("tools-ta"),
  voiceSelect: byId<HTMLSelectElement>("voice-select"),
  applyVoice: byId<HTMLButtonElement>("apply-voice"),
  newProfile: byId<HTMLButtonElement>("new-profile"),
  toolsAvailable: byId("tools-available"),
  saveProfile: byId<HTMLButtonElement>("save-profile"),
  overwriteProfile: byId<HTMLButtonElement>("overwrite-profile"),
  profileStatus: byId("profile-status"),
  logs: byId<HTMLOListElement>("logs"),
  logSearch: byId<HTMLInputElement>("log-search"),
  clearLogs: byId<HTMLButtonElement>("clear-logs"),
  jumpLatest: byId<HTMLButtonElement>("jump-latest"),
  jumpLatestCount: byId("jump-latest-count"),
  countInfo: byId("count-info"),
  countWarn: byId("count-warn"),
  countErr: byId("count-err"),
};

function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing element #${id}`);
  return element as T;
}

function setDot(el: HTMLElement, dotState: DotState): void {
  el.classList.remove("dot--ok", "dot--warn", "dot--err");
  if (dotState !== "idle") el.classList.add(`dot--${dotState}`);
}

function setStatus(el: HTMLElement, text: string, tone = ""): void {
  el.textContent = text;
  el.className = tone ? `status-line ${tone}` : "status-line";
}

async function fetchJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(String(data.error || response.statusText || "request_failed"));
    throw error;
  }
  return data as T;
}

function activeBackend(): string {
  return state.status?.backend_provider || DEFAULT_BACKEND;
}

function backendReady(status: DashboardStatus | null): boolean {
  if (!status) return false;
  const backend = status.backend_provider || DEFAULT_BACKEND;
  if (backend === LOCAL_BACKEND) return !!status.can_proceed_with_local;
  return !!status.can_proceed_with_hf;
}

function shortValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "-";
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${value.toFixed(value < 10 ? 1 : 0)}ms`;
}

function formatFps(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return `${value.toFixed(1)} fps`;
}

function renderStatus(): void {
  const status = state.status;
  const cameraOk = !!status?.camera.available && !!status.camera.frame_available;
  const faceOk = !!status?.face_recognition.available;
  const backendUnavailable = !!status?.backend_unavailable;
  const backendOk = backendReady(status);
  const performance = status?.performance;
  const transport = performance?.transport || {};
  const mediaHost = shortValue(transport.media_host);
  const mediaSource = shortValue(transport.media_host_source);
  const rttMs = performance?.daemon_rtt_ms;
  const linkOk = mediaHost !== "-" && (rttMs === null || rttMs === undefined || rttMs <= 15);
  const linkWarn = mediaSource === "daemon_wlan_ip" || (rttMs !== null && rttMs !== undefined && rttMs > 15);

  elements.cameraState.textContent = !status
    ? "checking"
    : cameraOk
      ? status.camera.head_tracker || "streaming"
      : status.camera.available
        ? "no frame"
        : "unavailable";
  elements.faceState.textContent = !status
    ? "checking"
    : faceOk
      ? `${status.face_recognition.visible_count} visible`
      : "unavailable";
  elements.backendState.textContent = !status
    ? "checking"
    : backendUnavailable
      ? "app offline"
    : `${status.backend_provider || DEFAULT_BACKEND}${status.requires_restart ? " pending" : ""}`;
  elements.transportState.textContent = !status ? "checking" : `${mediaHost}${rttMs ? ` ${formatMs(rttMs)}` : ""}`;
  setDot(elements.cameraDot, cameraOk ? "ok" : status?.camera.available ? "warn" : "err");
  setDot(elements.faceDot, faceOk ? "ok" : "warn");
  setDot(elements.backendDot, backendUnavailable ? "warn" : backendOk ? "ok" : "warn");
  setDot(elements.transportDot, !status ? "idle" : linkWarn ? "warn" : linkOk ? "ok" : "warn");

  renderPeople();
  renderDiagnostics();
  renderBackendControls();
}

function renderDiagnostics(): void {
  const performance = state.status?.performance || {};
  const transport = performance.transport || {};
  const mediaState = performance.media_state || {};
  const health = performance.health_checks || {};
  const localModel = performance.local_model || {};
  const localTts = performance.local_tts || {};
  const voiceActivity = performance.voice_activity || {};
  const mediaSource = shortValue(transport.media_host_source);
  elements.diagControlHost.textContent = shortValue(transport.control_host);
  elements.diagMediaHost.textContent = `${shortValue(transport.media_host)} (${mediaSource})`;
  elements.diagDaemon.textContent = `${shortValue(performance.daemon_state)} / ${formatMs(performance.daemon_rtt_ms)}`;
  elements.diagMediaState.textContent = `available ${shortValue(mediaState.available)}, released ${shortValue(mediaState.released)}`;
  elements.diagHealth.textContent = `daemon ${shortValue(health.daemon_running)}, media ${shortValue(health.media_available)}, doa ${shortValue(health.doa_status || health.doa_available)}, wired ${shortValue(health.wired_link_present)}, model ${shortValue(localModel.configured_model)} installed ${shortValue(localModel.installed)}`;
  elements.diagCamera.textContent = `${formatFps(performance.camera_fps)} / age ${formatMs(performance.camera_frame_age_ms)}`;
  elements.diagAudio.textContent = `in ${shortValue(performance.audio_input_frames)}, out ${shortValue(performance.audio_output_frames)}, drop ${shortValue(performance.dropped_audio_frames)}, q ${shortValue(performance.audio_queue_depth_s)}s`;
  elements.diagVad.textContent = `${shortValue(voiceActivity.vad_state)}, active ${shortValue(voiceActivity.active_motion_playback)}`;
  elements.diagNoise.textContent = `floor ${shortValue(voiceActivity.noise_floor_rms)}, conf ${shortValue(voiceActivity.speech_confidence_ratio)}, window ${shortValue(voiceActivity.robot_noise_suppression_window_ms)}ms`;
  elements.diagRejects.textContent = `${shortValue(voiceActivity.rejected_segment_count)} / ${shortValue(voiceActivity.last_reject_reason)}`;
  elements.diagStt.textContent = `${formatMs(performance.stt_ms)} / reject ${shortValue(voiceActivity.last_stt_reject_reason)}`;
  elements.diagLlm.textContent = `${formatMs(performance.llm_first_token_ms)} first / ${formatMs(performance.llm_total_ms)} total`;
  elements.diagRouter.textContent = `${formatMs(Number(localModel.qwen_router_latency_ms))} / ${shortValue(localModel.qwen_router_status)}`;
  elements.diagTts.textContent = `${shortValue(localTts.ready)} ${shortValue(localTts.error)} / ${formatMs(performance.tts_ms)} / first audio ${formatMs(performance.first_audio_ms)}`;
}

function renderProcessControls(): void {
  const status = state.process;
  elements.app.classList.toggle("has-process-controls", state.processAvailable);
  elements.processPill.hidden = !state.processAvailable;
  elements.processSection.hidden = !state.processAvailable;
  if (!state.processAvailable || !status) return;

  if (!elements.commandInput.value.trim()) {
    elements.commandInput.value = status.command || status.defaultCommand;
  }
  if (!elements.robotHost.value.trim()) {
    elements.robotHost.value = storedRobotHost() || status.defaultRobotHost || "";
  }
  if (!elements.robotPort.value.trim()) {
    elements.robotPort.value = storedRobotPort() || status.defaultRobotPort || "";
  }

  elements.startCommand.disabled = status.running;
  elements.stopCommand.disabled = !status.running;
  if (status.running) {
    elements.processState.textContent = status.pid ? `running ${status.pid}` : "running";
    setDot(elements.processDot, status.backendReady === false ? "warn" : "ok");
  } else if (status.failureHint) {
    elements.processState.textContent = "robot unavailable";
    setDot(elements.processDot, "err");
  } else if (status.exitCode !== null || status.signal) {
    elements.processState.textContent = status.signal ? `stopped ${status.signal}` : `exit ${status.exitCode}`;
    setDot(elements.processDot, status.exitCode === 0 ? "idle" : "err");
  } else {
    elements.processState.textContent = "idle";
    setDot(elements.processDot, "idle");
  }

  const target = status.backendTarget.replace(/^https?:\/\//, "");
  setStatus(
    elements.commandStatus,
    status.running
      ? status.backendReady === false
        ? `Starting. Waiting for app API on ${target}.`
        : `Running. Dashboard APIs proxy to ${target}.`
      : status.failureHint
        ? status.failureHint
        : status.exitCode !== null || status.signal
          ? `Stopped. Dashboard APIs proxy to ${target}.`
          : `Ready. Dashboard APIs proxy to ${target}.`,
    status.running ? (status.backendReady === false ? "warn" : "ok") : status.failureHint ? "error" : "",
  );
}

function renderPeople(): void {
  const people = state.status?.face_recognition.people || [];
  if (!people.length) {
    const empty = document.createElement("li");
    empty.className = "empty-row";
    empty.textContent = "No saved people yet.";
    elements.peopleList.replaceChildren(empty);
    return;
  }
  elements.peopleList.replaceChildren(
    ...people.map((person) => {
      const item = document.createElement("li");
      const name = document.createElement("strong");
      name.textContent = person.name;
      const count = document.createElement("span");
      count.textContent = `${person.exemplar_count} exemplar${person.exemplar_count === 1 ? "" : "s"}`;
      item.append(name, count);
      return item;
    }),
  );
}

async function loadDashboardStatus(): Promise<void> {
  state.status = await fetchJson<DashboardStatus>("/api/dashboard/status");
  renderStatus();
}

async function loadProcessStatus(): Promise<boolean> {
  try {
    state.process = await fetchJson<ProcessStatus>("/__dashboard/process/status");
    state.processAvailable = !!state.process.available;
    renderProcessControls();
    return true;
  } catch {
    state.processAvailable = false;
    renderProcessControls();
    return false;
  }
}

async function loadFaceState(): Promise<void> {
  const data = await fetchJson<FaceState>("/api/face/state");
  state.faceStateAvailable = !!data.available;
  state.faceRecognitionAvailable = data.recognition_available !== false;
  state.faces = data.faces || [];
  if (state.selectedFaceId !== null && !state.faces.some((face) => face.id === state.selectedFaceId)) {
    state.selectedFaceId = null;
  }
  renderFaces();
}

function refreshFrame(): void {
  const src = `/api/face/frame.jpg?_=${Date.now()}`;
  elements.cameraFeed.src = src;
}

async function startProcess(): Promise<void> {
  const baseCommand = elements.commandInput.value.trim() || state.process?.defaultCommand || "";
  if (!baseCommand) {
    setStatus(elements.commandStatus, "Enter a command.", "warn");
    return;
  }
  const command = commandWithRobotTarget(baseCommand);
  if (!command) return;
  elements.startCommand.disabled = true;
  setStatus(elements.commandStatus, "Starting...");
  try {
    state.process = await fetchJson<ProcessStatus>("/__dashboard/process/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    });
    state.processAvailable = true;
    renderProcessControls();
    addLocalLog(`Started ${command}`, "INFO", "PROCESS");
  } catch (error) {
    setStatus(elements.commandStatus, error instanceof Error ? error.message : "Failed to start.", "error");
    elements.startCommand.disabled = false;
  }
}

const ROBOT_HOST_STORAGE_KEY = "reachy-dashboard.robot-host";
const ROBOT_PORT_STORAGE_KEY = "reachy-dashboard.robot-port";
const ROBOT_TARGET_OPTIONS = new Set(["--connection-mode", "--robot-host", "--robot-port"]);

function storedRobotHost(): string {
  return window.localStorage.getItem(ROBOT_HOST_STORAGE_KEY) || "";
}

function storedRobotPort(): string {
  return window.localStorage.getItem(ROBOT_PORT_STORAGE_KEY) || "";
}

function persistRobotTarget(): void {
  window.localStorage.setItem(ROBOT_HOST_STORAGE_KEY, elements.robotHost.value.trim());
  window.localStorage.setItem(ROBOT_PORT_STORAGE_KEY, elements.robotPort.value.trim());
}

function quoteCommandArg(value: string): string {
  if (/^[A-Za-z0-9_./:@%+=,-]+$/.test(value)) return value;
  return `"${value.replace(/(["\\$`])/g, "\\$1")}"`;
}

function splitCommand(commandText: string): string[] {
  const parts: string[] = [];
  let current = "";
  let quote: "'" | '"' | null = null;
  let escaped = false;

  for (const char of commandText.trim()) {
    if (escaped) {
      current += char;
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (quote) {
      if (char === quote) quote = null;
      else current += char;
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      continue;
    }
    if (/\s/.test(char)) {
      if (current) {
        parts.push(current);
        current = "";
      }
      continue;
    }
    if ("|&;<>".includes(char)) throw new Error("shell_operators_not_supported");
    current += char;
  }

  if (escaped) current += "\\";
  if (quote) throw new Error("unterminated_quote");
  if (current) parts.push(current);
  return parts;
}

function withoutRobotTargetArgs(parts: string[]): string[] {
  const cleaned: string[] = [];
  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index];
    if (ROBOT_TARGET_OPTIONS.has(part)) {
      index += 1;
      continue;
    }
    if (Array.from(ROBOT_TARGET_OPTIONS).some((option) => part.startsWith(`${option}=`))) continue;
    cleaned.push(part);
  }
  return cleaned;
}

function commandParseErrorMessage(error: unknown): string {
  if (!(error instanceof Error)) return "Invalid command.";
  if (error.message === "unterminated_quote") return "Fix command quoting before starting.";
  if (error.message === "shell_operators_not_supported") return "Shell operators are not supported in the command.";
  return "Invalid command.";
}

function commandWithRobotTarget(command: string): string {
  const robotHost = elements.robotHost.value.trim();
  const robotPort = elements.robotPort.value.trim();
  persistRobotTarget();

  if (!robotHost && !robotPort) return command;

  const additions = ["--connection-mode", "network"];
  if (robotHost) additions.push("--robot-host", robotHost);
  if (robotPort) {
    const parsedPort = Number.parseInt(robotPort, 10);
    if (!Number.isInteger(parsedPort) || parsedPort < 1 || parsedPort > 65535) {
      setStatus(elements.commandStatus, "Enter a valid robot port.", "warn");
      return "";
    }
    additions.push("--robot-port", String(parsedPort));
  }

  try {
    const baseParts = withoutRobotTargetArgs(splitCommand(command));
    if (!baseParts.length) return "";
    return [...baseParts, ...additions].map(quoteCommandArg).join(" ");
  } catch (error) {
    setStatus(elements.commandStatus, commandParseErrorMessage(error), "warn");
    return "";
  }
}

async function stopProcess(): Promise<void> {
  elements.stopCommand.disabled = true;
  setStatus(elements.commandStatus, "Stopping...");
  try {
    state.process = await fetchJson<ProcessStatus>("/__dashboard/process/stop", { method: "POST" });
    state.processAvailable = true;
    renderProcessControls();
  } catch (error) {
    setStatus(elements.commandStatus, error instanceof Error ? error.message : "Failed to stop.", "error");
    elements.stopCommand.disabled = false;
  }
}

function renderFaces(): void {
  const selected = selectedFace();
  elements.saveFace.disabled = !selected || selected.id === null || selected.can_remember === false;
  const faceWord = state.faces.length === 1 ? "face" : "faces";
  elements.cameraHelp.textContent = state.faces.length
    ? state.faceRecognitionAvailable
      ? `${state.faces.length} ${faceWord} visible. Click a box to name it.`
      : `${state.faces.length} ${faceWord} visible. Face naming unavailable.`
    : state.faceStateAvailable
      ? "No faces detected."
      : "Camera unavailable until the app API is ready.";
  elements.selectedFace.textContent = selected
    ? `${selected.label} - x ${selected.x_offset.toFixed(2)} - confidence ${selected.confidence.toFixed(2)}`
    : "Select a face in the camera feed.";

  elements.faceOverlay.replaceChildren(
    ...state.faces.map((face) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "face-box";
      if (face.id === state.selectedFaceId) button.classList.add("is-selected");
      if (face.name) button.classList.add("is-known");
      if (face.focused) button.classList.add("is-focused");
      if (face.held) button.classList.add("is-held");
      button.style.left = `${face.bbox.x * 100}%`;
      button.style.top = `${face.bbox.y * 100}%`;
      button.style.width = `${face.bbox.width * 100}%`;
      button.style.height = `${face.bbox.height * 100}%`;
      button.disabled = face.id === null;
      button.addEventListener("click", () => {
        state.selectedFaceId = face.id;
        elements.faceName.value = face.name || "";
        renderFaces();
      });

      const label = document.createElement("span");
      label.className = "face-label";
      label.textContent = face.name ? `${face.name} ${face.similarity.toFixed(2)}` : "unknown";
      button.append(label);
      return button;
    }),
  );
}

function selectedFace(): FaceBox | null {
  if (state.selectedFaceId === null) return null;
  return state.faces.find((face) => face.id === state.selectedFaceId) || null;
}

async function saveSelectedFace(): Promise<void> {
  const selected = selectedFace();
  const name = elements.faceName.value.trim();
  if (!selected || selected.id === null) {
    setStatus(elements.faceSaveStatus, "Select a visible face first.", "warn");
    return;
  }
  if (!name) {
    setStatus(elements.faceSaveStatus, "Enter a name.", "warn");
    return;
  }

  elements.saveFace.disabled = true;
  setStatus(elements.faceSaveStatus, "Saving...");
  try {
    const result = await fetchJson<{ name: string; exemplar_count: number }>("/api/face/remember", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ face_id: selected.id, name }),
    });
    setStatus(elements.faceSaveStatus, `Saved ${result.name} (${result.exemplar_count} exemplar).`, "ok");
    await Promise.all([loadDashboardStatus(), loadFaceState()]);
  } catch (error) {
    setStatus(elements.faceSaveStatus, error instanceof Error ? error.message : "Failed to save.", "error");
  } finally {
    elements.saveFace.disabled = false;
  }
}

function backendInputs(): HTMLInputElement[] {
  return Array.from(document.querySelectorAll<HTMLInputElement>('input[name="backend"]'));
}

function renderBackendControls(): void {
  const backend = activeBackend();
  backendInputs().forEach((input) => {
    input.checked = input.value === backend;
    input.parentElement?.classList.toggle("is-selected", input.checked);
  });
  elements.hfFields.hidden = backend !== HF_BACKEND;
  elements.hfMode.value = state.status?.hf_connection_mode || "deployed";
  elements.hfHost.value = state.status?.hf_direct_host || "localhost";
  elements.hfPort.value = String(state.status?.hf_direct_port || 8765);
}

async function saveBackend(): Promise<void> {
  const backend = backendInputs().find((input) => input.checked)?.value || DEFAULT_BACKEND;
  const body: Record<string, string | number> = { backend };
  if (backend === HF_BACKEND) {
    body.hf_mode = elements.hfMode.value;
    body.hf_host = elements.hfHost.value.trim();
    body.hf_port = Number.parseInt(elements.hfPort.value || "8765", 10);
  }
  setStatus(elements.backendStatus, "Saving...");
  try {
    await fetchJson<BackendStatus>("/backend_config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setStatus(elements.backendStatus, "Saved. Refreshing status...", "ok");
    await loadDashboardStatus();
  } catch (error) {
    setStatus(elements.backendStatus, error instanceof Error ? error.message : "Failed to save.", "error");
  }
}

async function loadProfiles(): Promise<void> {
  let list: ProfileList;
  try {
    list = await fetchJson<ProfileList>("/profiles");
  } catch {
    elements.profileSelect.innerHTML = '<option value="">Profile routes starting...</option>';
    setStatus(elements.profileStatus, "Profile controls become available after the conversation loop starts.", "warn");
    return;
  }

  elements.profileSelect.replaceChildren(
    ...list.choices.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    }),
  );
  elements.profileSelect.value = list.choices.includes(list.startup) ? list.startup : list.current;
  await loadSelectedProfile();
}

async function loadSelectedProfile(): Promise<void> {
  const name = elements.profileSelect.value;
  if (!name) return;
  const url = new URL("/profiles/load", window.location.origin);
  url.searchParams.set("name", name);
  const data = await fetchJson<ProfilePayload>(url.toString());
  elements.instructions.value = data.instructions || "";
  elements.toolsText.value = data.tools_text || "";
  elements.profileName.value = name;
  renderToolCheckboxes(data.available_tools || [], data.enabled_tools || []);
  await loadVoices(data.voice);
  setStatus(elements.profileStatus, `Loaded ${name}.`);
}

async function loadVoices(preferred: string): Promise<void> {
  let voices: string[] = [];
  try {
    voices = await fetchJson<string[]>("/voices");
  } catch {
    voices = [];
  }
  if (!voices.length) voices = [preferred || ""].filter(Boolean);
  elements.voiceSelect.replaceChildren(
    ...voices.map((voice) => {
      const option = document.createElement("option");
      option.value = voice;
      option.textContent = voice;
      return option;
    }),
  );
  if (voices.includes(preferred)) elements.voiceSelect.value = preferred;
}

function renderToolCheckboxes(available: string[], enabled: string[]): void {
  const enabledSet = new Set(enabled);
  elements.toolsAvailable.replaceChildren(
    ...available.map((tool) => {
      const label = document.createElement("label");
      label.className = "tool-check";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = tool;
      checkbox.checked = enabledSet.has(tool);
      checkbox.addEventListener("change", syncToolsFromCheckboxes);
      const text = document.createElement("span");
      text.textContent = tool;
      label.append(checkbox, text);
      return label;
    }),
  );
}

function syncToolsFromCheckboxes(): void {
  if (!elements.toolsAvailable.querySelector('input[type="checkbox"]')) return;
  const selected = new Set<string>();
  elements.toolsAvailable.querySelectorAll<HTMLInputElement>('input[type="checkbox"]').forEach((input) => {
    if (input.checked) selected.add(input.value);
  });
  for (const [tool, deps] of Object.entries(AUTO_WITH)) {
    if (selected.has(tool)) deps.forEach((dep) => selected.add(dep));
  }
  const comments = elements.toolsText.value.split("\n").filter((line) => line.trim().startsWith("#"));
  elements.toolsText.value = `${comments.length ? `${comments.join("\n")}\n` : ""}${Array.from(selected).sort().join("\n")}\n`;
}

async function applyProfile(persist: boolean): Promise<void> {
  const url = new URL("/profiles/apply", window.location.origin);
  url.searchParams.set("name", elements.profileSelect.value || "");
  if (persist) url.searchParams.set("persist", "1");
  setStatus(elements.profileStatus, persist ? "Saving startup profile..." : "Applying...");
  try {
    const result = await fetchJson<{ status?: string }>(url.toString(), { method: "POST" });
    setStatus(elements.profileStatus, result.status || "Applied.", "ok");
  } catch (error) {
    setStatus(elements.profileStatus, error instanceof Error ? error.message : "Failed.", "error");
  }
}

async function saveProfile(overwrite = false): Promise<void> {
  const name = elements.profileName.value.trim();
  if (!name) {
    setStatus(elements.profileStatus, "Enter a profile name.", "warn");
    return;
  }
  syncToolsFromCheckboxes();
  setStatus(elements.profileStatus, overwrite ? "Overwriting..." : "Saving...");
  try {
    const result = await fetchJson<{ profile: string; choices: string[] }>("/profiles/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        instructions: elements.instructions.value,
        tools_text: elements.toolsText.value,
        voice: elements.voiceSelect.value,
        overwrite,
      }),
    });
    elements.profileSelect.replaceChildren(
      ...result.choices.map((choice) => {
        const option = document.createElement("option");
        option.value = choice;
        option.textContent = choice;
        return option;
      }),
    );
    elements.profileSelect.value = result.profile;
    setStatus(elements.profileStatus, overwrite ? "Overwritten. Restart to apply tool changes." : "Saved. Restart to apply tool changes.", "ok");
  } catch (error) {
    setStatus(elements.profileStatus, error instanceof Error ? error.message : "Failed to save.", "error");
  }
}

async function applyVoice(): Promise<void> {
  const voice = elements.voiceSelect.value;
  if (!voice) return;
  const url = new URL("/voices/apply", window.location.origin);
  url.searchParams.set("voice", voice);
  setStatus(elements.profileStatus, "Applying voice...");
  try {
    const result = await fetchJson<{ status?: string }>(url.toString(), { method: "POST" });
    setStatus(elements.profileStatus, result.status || `Voice changed to ${voice}.`, "ok");
  } catch (error) {
    setStatus(elements.profileStatus, error instanceof Error ? error.message : "Failed to apply voice.", "error");
  }
}

function newProfile(): void {
  elements.profileName.value = "";
  elements.instructions.value = "# Write your instructions here\n";
  elements.toolsText.value = "# tools enabled for this profile\n";
  elements.toolsAvailable.querySelectorAll<HTMLInputElement>('input[type="checkbox"]').forEach((input) => {
    input.checked = false;
  });
  setStatus(elements.profileStatus, "Fill fields and save.");
}

function entryLevel(entry: LogEntry): string {
  return (entry.level || "INFO").toUpperCase();
}

function visibleLogs(): LogEntry[] {
  const search = logUi.search.trim().toLowerCase();
  const visible: LogEntry[] = [];
  for (let index = logUi.cleared; index < state.logs.length; index += 1) {
    const entry = state.logs[index];
    if (logUi.filter !== "ALL" && entryLevel(entry) !== logUi.filter) continue;
    if (search && !entry.message.toLowerCase().includes(search)) continue;
    visible.push(entry);
  }
  return visible;
}

const MAX_VISIBLE_LOGS = 300;
const SCROLL_BOTTOM_THRESHOLD = 40;
const USER_SCROLL_WINDOW_MS = 400;

let lastUserScrollAt = 0;

function entryPassesFilters(entry: LogEntry): boolean {
  const search = logUi.search.trim().toLowerCase();
  if (logUi.filter !== "ALL" && entryLevel(entry) !== logUi.filter) return false;
  if (search && !entry.message.toLowerCase().includes(search)) return false;
  return true;
}

function updateLogCounts(): void {
  let info = 0;
  let warn = 0;
  let err = 0;
  for (let index = logUi.cleared; index < state.logs.length; index += 1) {
    const level = entryLevel(state.logs[index]);
    if (level === "ERROR") err += 1;
    else if (level === "WARNING") warn += 1;
    else info += 1;
  }
  setCount(elements.countInfo, info);
  setCount(elements.countWarn, warn);
  setCount(elements.countErr, err);
}

function showEmptyLogPlaceholder(): void {
  const empty = document.createElement("li");
  empty.className = "log-empty";
  empty.textContent = "Waiting for events...";
  elements.logs.replaceChildren(empty);
}

function updateJumpLatest(): void {
  if (logUi.autoScroll) {
    elements.jumpLatest.hidden = true;
    logUi.newSincePaused = 0;
  } else if (logUi.newSincePaused > 0) {
    elements.jumpLatestCount.textContent = String(logUi.newSincePaused);
    elements.jumpLatest.hidden = false;
  } else {
    elements.jumpLatest.hidden = true;
  }
}

function scrollLogsToBottom(): void {
  elements.logs.scrollTop = elements.logs.scrollHeight;
}

function renderLogs(): void {
  updateLogCounts();

  const items = visibleLogs().slice(-MAX_VISIBLE_LOGS);
  if (!items.length) {
    showEmptyLogPlaceholder();
  } else {
    elements.logs.replaceChildren(...items.map(renderLog));
  }

  if (logUi.autoScroll) {
    scrollLogsToBottom();
    logUi.newSincePaused = 0;
  }
  updateJumpLatest();
}

function appendLogToDom(entry: LogEntry): void {
  if (!entryPassesFilters(entry)) return;
  const placeholder = elements.logs.firstElementChild;
  if (placeholder && placeholder.classList.contains("log-empty")) {
    elements.logs.replaceChildren();
  }
  elements.logs.appendChild(renderLog(entry));
  while (elements.logs.childElementCount > MAX_VISIBLE_LOGS) {
    elements.logs.removeChild(elements.logs.firstElementChild!);
  }
}

function renderLog(entry: LogEntry): HTMLLIElement {
  const item = document.createElement("li");
  const level = entryLevel(entry);
  const categoryText = (entry.category || "SYSTEM").toUpperCase();
  item.dataset.level = level;
  item.dataset.category = categoryText;

  const time = document.createElement("time");
  time.textContent = entry.createdAt ? new Date(entry.createdAt).toLocaleTimeString() : "";
  const category = document.createElement("span");
  category.className = `cat cat--${categoryText.toLowerCase()}`;
  category.textContent = categoryText;
  const message = document.createElement("span");
  message.className = "log-message";
  message.textContent = entry.message;
  item.append(time, category, message);
  return item;
}

function setCount(el: HTMLElement, count: number): void {
  el.textContent = String(count);
  el.classList.toggle("has-items", count > 0);
}

function isLogAtBottom(): boolean {
  return (
    elements.logs.scrollHeight - elements.logs.scrollTop - elements.logs.clientHeight <
    SCROLL_BOTTOM_THRESHOLD
  );
}

function noteUserScrollIntent(): void {
  lastUserScrollAt = Date.now();
}

function appendLog(entry: LogEntry): void {
  state.logs = [...state.logs, entry].slice(-500);
  if (state.logs.length < logUi.cleared) logUi.cleared = state.logs.length;

  updateLogCounts();
  appendLogToDom(entry);

  if (logUi.autoScroll) {
    scrollLogsToBottom();
  } else if (entryPassesFilters(entry)) {
    logUi.newSincePaused += 1;
  }
  updateJumpLatest();
}

function addLocalLog(message: string, level = "INFO", category = "SYSTEM"): void {
  appendLog({ type: "log", createdAt: new Date().toISOString(), level, category, message });
}

let dashboardLogDisconnectedAt = 0;
let processLogSource: EventSource | null = null;

function connectLogs(): void {
  const source = new EventSource("/api/dashboard/events");
  source.onopen = () => {
    dashboardLogDisconnectedAt = 0;
  };
  source.addEventListener("log", (event) => {
    const parsed = JSON.parse((event as MessageEvent).data) as LogEntry;
    appendLog(parsed);
  });
  source.onerror = () => {
    const now = Date.now();
    const waitingForAppApi = state.status?.backend_unavailable || state.process?.backendReady === false;
    if (!waitingForAppApi && (!dashboardLogDisconnectedAt || now - dashboardLogDisconnectedAt > 8000)) {
      dashboardLogDisconnectedAt = now;
      addLocalLog("Dashboard event stream disconnected; retrying", "WARNING");
    }
  };
}

function connectProcessLogs(): void {
  if (!state.processAvailable || processLogSource) return;
  processLogSource = new EventSource("/__dashboard/process/events");
  processLogSource.addEventListener("log", (event) => {
    const parsed = JSON.parse((event as MessageEvent).data) as LogEntry;
    appendLog(parsed);
  });
  processLogSource.onerror = () => {
    processLogSource?.close();
    processLogSource = null;
  };
}

function wireEvents(): void {
  elements.cameraFeed.addEventListener("load", () => {
    elements.cameraEmpty.hidden = true;
  });
  elements.cameraFeed.addEventListener("error", () => {
    elements.cameraEmpty.hidden = false;
  });
  elements.startCommand.addEventListener("click", () => void startProcess());
  elements.stopCommand.addEventListener("click", () => void stopProcess());
  elements.robotHost.addEventListener("input", persistRobotTarget);
  elements.robotPort.addEventListener("input", persistRobotTarget);
  elements.refreshFaceState.addEventListener("click", () => void loadFaceState());
  elements.saveFace.addEventListener("click", () => void saveSelectedFace());
  elements.saveBackend.addEventListener("click", () => void saveBackend());
  elements.profileSelect.addEventListener("change", () => void loadSelectedProfile());
  elements.applyProfile.addEventListener("click", () => void applyProfile(false));
  elements.persistProfile.addEventListener("click", () => void applyProfile(true));
  elements.applyVoice.addEventListener("click", () => void applyVoice());
  elements.newProfile.addEventListener("click", newProfile);
  elements.saveProfile.addEventListener("click", () => void saveProfile(false));
  elements.overwriteProfile.addEventListener("click", () => void saveProfile(true));
  backendInputs().forEach((input) => {
    input.addEventListener("change", renderBackendControls);
  });
  const userIntentEvents: Array<keyof HTMLElementEventMap> = [
    "wheel",
    "touchstart",
    "pointerdown",
    "keydown",
  ];
  userIntentEvents.forEach((eventName) => {
    elements.logs.addEventListener(eventName, noteUserScrollIntent, { passive: true });
  });

  elements.logs.addEventListener("scroll", () => {
    if (Date.now() - lastUserScrollAt > USER_SCROLL_WINDOW_MS) return;
    const atBottom = isLogAtBottom();
    if (atBottom && !logUi.autoScroll) {
      logUi.autoScroll = true;
      logUi.newSincePaused = 0;
      updateJumpLatest();
    } else if (!atBottom && logUi.autoScroll) {
      logUi.autoScroll = false;
      updateJumpLatest();
    }
  });

  elements.jumpLatest.addEventListener("click", () => {
    logUi.autoScroll = true;
    logUi.newSincePaused = 0;
    elements.jumpLatest.hidden = true;
    scrollLogsToBottom();
  });
  elements.clearLogs.addEventListener("click", () => {
    logUi.cleared = state.logs.length;
    renderLogs();
  });
  elements.logSearch.addEventListener("input", () => {
    logUi.search = elements.logSearch.value;
    renderLogs();
  });
  document.querySelectorAll<HTMLButtonElement>(".filter-tab").forEach((chip) => {
    chip.addEventListener("click", () => {
      logUi.filter = (chip.dataset.level as LevelFilter) || "ALL";
      document.querySelectorAll(".filter-tab").forEach((item) => {
        item.classList.toggle("is-active", item === chip);
      });
      renderLogs();
    });
  });
}

async function init(): Promise<void> {
  wireEvents();
  const processAvailable = await loadProcessStatus();
  if (processAvailable) connectProcessLogs();
  connectLogs();
  await loadDashboardStatus().catch(() => addLocalLog("Dashboard status unavailable", "WARNING"));
  await loadFaceState().catch(() => addLocalLog("Face state unavailable", "WARNING", "VISION"));
  await loadProfiles();
  renderLogs();
  refreshFrame();
  window.setInterval(() => void loadProcessStatus().then((available) => {
    if (available) connectProcessLogs();
  }), 2000);
  window.setInterval(() => void loadDashboardStatus().catch(() => undefined), 3000);
  window.setInterval(() => void loadFaceState().catch(() => undefined), 500);
  window.setInterval(refreshFrame, 350);
}

void init();
