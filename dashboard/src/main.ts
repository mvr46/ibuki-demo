import "./styles.css";

type DotState = "ok" | "warn" | "err" | "idle";
type LevelFilter = "ALL" | "INFO" | "WARNING" | "ERROR";

type BackendStatus = {
  active_backend?: string;
  backend_provider?: string;
  has_openai_key?: boolean;
  has_gemini_key?: boolean;
  has_hf_connection?: boolean;
  has_hf_session_url?: boolean;
  has_hf_ws_url?: boolean;
  hf_connection_mode?: string;
  hf_direct_host?: string;
  hf_direct_port?: number;
  can_proceed?: boolean;
  can_proceed_with_openai?: boolean;
  can_proceed_with_gemini?: boolean;
  can_proceed_with_hf?: boolean;
  requires_restart?: boolean;
};

type DashboardStatus = BackendStatus & {
  camera: {
    available: boolean;
    frame_available: boolean;
    head_tracker: string | null;
  };
  face_recognition: {
    available: boolean;
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
};

type FaceState = {
  ok: boolean;
  available: boolean;
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

type PersonalityList = {
  choices: string[];
  current: string;
  startup: string;
  locked?: boolean;
};

type PersonalityPayload = {
  instructions: string;
  tools_text: string;
  voice: string;
  available_tools: string[];
  enabled_tools: string[];
};

const OPENAI_BACKEND = "openai";
const GEMINI_BACKEND = "gemini";
const HF_BACKEND = "huggingface";
const DEFAULT_BACKEND = HF_BACKEND;

const AUTO_WITH: Record<string, string[]> = {
  dance: ["stop_dance"],
  play_emotion: ["stop_emotion"],
};

const state = {
  status: null as DashboardStatus | null,
  faces: [] as FaceBox[],
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
  cameraDot: byId("camera-dot"),
  faceDot: byId("face-dot"),
  backendDot: byId("backend-dot"),
  cameraState: byId("camera-state"),
  faceState: byId("face-state"),
  backendState: byId("backend-state"),
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
  backendGrid: byId("backend-grid"),
  apiKey: byId<HTMLInputElement>("api-key"),
  hfFields: byId("hf-fields"),
  hfMode: byId<HTMLSelectElement>("hf-mode"),
  hfHost: byId<HTMLInputElement>("hf-host"),
  hfPort: byId<HTMLInputElement>("hf-port"),
  saveBackend: byId<HTMLButtonElement>("save-backend"),
  backendStatus: byId("backend-status"),
  personalitySelect: byId<HTMLSelectElement>("personality-select"),
  applyPersonality: byId<HTMLButtonElement>("apply-personality"),
  persistPersonality: byId<HTMLButtonElement>("persist-personality"),
  personalityName: byId<HTMLInputElement>("personality-name"),
  instructions: byId<HTMLTextAreaElement>("instructions-ta"),
  toolsText: byId<HTMLTextAreaElement>("tools-ta"),
  voiceSelect: byId<HTMLSelectElement>("voice-select"),
  applyVoice: byId<HTMLButtonElement>("apply-voice"),
  newPersonality: byId<HTMLButtonElement>("new-personality"),
  toolsAvailable: byId("tools-available"),
  savePersonality: byId<HTMLButtonElement>("save-personality"),
  personalityStatus: byId("personality-status"),
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
  if (backend === OPENAI_BACKEND) return !!status.can_proceed_with_openai;
  if (backend === GEMINI_BACKEND) return !!status.can_proceed_with_gemini;
  return !!status.can_proceed_with_hf;
}

function renderStatus(): void {
  const status = state.status;
  const cameraOk = !!status?.camera.available && !!status.camera.frame_available;
  const faceOk = !!status?.face_recognition.available;
  const backendOk = backendReady(status);

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
    : `${status.backend_provider || DEFAULT_BACKEND}${status.requires_restart ? " pending" : ""}`;
  setDot(elements.cameraDot, cameraOk ? "ok" : status?.camera.available ? "warn" : "err");
  setDot(elements.faceDot, faceOk ? "ok" : "warn");
  setDot(elements.backendDot, backendOk ? "ok" : "warn");

  renderPeople();
  renderBackendControls();
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

async function loadFaceState(): Promise<void> {
  const data = await fetchJson<FaceState>("/api/face/state");
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

function renderFaces(): void {
  const selected = selectedFace();
  elements.saveFace.disabled = !selected || selected.id === null;
  elements.cameraHelp.textContent = state.faces.length
    ? `${state.faces.length} face${state.faces.length === 1 ? "" : "s"} visible. Click a box to name it.`
    : "No faces detected.";
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
  const needsApiKey = backend === OPENAI_BACKEND || backend === GEMINI_BACKEND;
  elements.apiKey.hidden = !needsApiKey;
  elements.apiKey.placeholder = backend === GEMINI_BACKEND ? "GEMINI_API_KEY" : "OPENAI_API_KEY";
  elements.hfFields.hidden = backend !== HF_BACKEND;
  elements.hfMode.value = state.status?.hf_connection_mode || "deployed";
  elements.hfHost.value = state.status?.hf_direct_host || "localhost";
  elements.hfPort.value = String(state.status?.hf_direct_port || 8765);
}

async function saveBackend(): Promise<void> {
  const backend = backendInputs().find((input) => input.checked)?.value || DEFAULT_BACKEND;
  const body: Record<string, string | number> = { backend };
  if (backend === OPENAI_BACKEND || backend === GEMINI_BACKEND) {
    if (elements.apiKey.value.trim()) body.api_key = elements.apiKey.value.trim();
  }
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
    elements.apiKey.value = "";
    await loadDashboardStatus();
  } catch (error) {
    setStatus(elements.backendStatus, error instanceof Error ? error.message : "Failed to save.", "error");
  }
}

async function loadPersonalities(): Promise<void> {
  let list: PersonalityList;
  try {
    list = await fetchJson<PersonalityList>("/personalities");
  } catch {
    elements.personalitySelect.innerHTML = '<option value="">Personality routes starting...</option>';
    setStatus(elements.personalityStatus, "Personality controls become available after the conversation loop starts.", "warn");
    return;
  }

  elements.personalitySelect.replaceChildren(
    ...list.choices.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    }),
  );
  elements.personalitySelect.value = list.choices.includes(list.startup) ? list.startup : list.current;
  await loadSelectedPersonality();
}

async function loadSelectedPersonality(): Promise<void> {
  const name = elements.personalitySelect.value;
  if (!name) return;
  const url = new URL("/personalities/load", window.location.origin);
  url.searchParams.set("name", name);
  const data = await fetchJson<PersonalityPayload>(url.toString());
  elements.instructions.value = data.instructions || "";
  elements.toolsText.value = data.tools_text || "";
  elements.personalityName.value = name.includes("/") ? name.split("/").pop() || "" : "";
  renderToolCheckboxes(data.available_tools || [], data.enabled_tools || []);
  await loadVoices(data.voice);
  setStatus(elements.personalityStatus, `Loaded ${name}.`);
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

async function applyPersonality(persist: boolean): Promise<void> {
  const url = new URL("/personalities/apply", window.location.origin);
  url.searchParams.set("name", elements.personalitySelect.value || "");
  if (persist) url.searchParams.set("persist", "1");
  setStatus(elements.personalityStatus, persist ? "Saving startup personality..." : "Applying...");
  try {
    const result = await fetchJson<{ status?: string }>(url.toString(), { method: "POST" });
    setStatus(elements.personalityStatus, result.status || "Applied.", "ok");
  } catch (error) {
    setStatus(elements.personalityStatus, error instanceof Error ? error.message : "Failed.", "error");
  }
}

async function savePersonality(): Promise<void> {
  const name = elements.personalityName.value.trim();
  if (!name) {
    setStatus(elements.personalityStatus, "Enter a profile name.", "warn");
    return;
  }
  syncToolsFromCheckboxes();
  setStatus(elements.personalityStatus, "Saving...");
  try {
    const result = await fetchJson<{ value: string; choices: string[] }>("/personalities/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        instructions: elements.instructions.value,
        tools_text: elements.toolsText.value,
        voice: elements.voiceSelect.value,
      }),
    });
    elements.personalitySelect.replaceChildren(
      ...result.choices.map((choice) => {
        const option = document.createElement("option");
        option.value = choice;
        option.textContent = choice;
        return option;
      }),
    );
    elements.personalitySelect.value = result.value;
    setStatus(elements.personalityStatus, "Saved.", "ok");
  } catch (error) {
    setStatus(elements.personalityStatus, error instanceof Error ? error.message : "Failed to save.", "error");
  }
}

async function applyVoice(): Promise<void> {
  const voice = elements.voiceSelect.value;
  if (!voice) return;
  const url = new URL("/voices/apply", window.location.origin);
  url.searchParams.set("voice", voice);
  setStatus(elements.personalityStatus, "Applying voice...");
  try {
    const result = await fetchJson<{ status?: string }>(url.toString(), { method: "POST" });
    setStatus(elements.personalityStatus, result.status || `Voice changed to ${voice}.`, "ok");
  } catch (error) {
    setStatus(elements.personalityStatus, error instanceof Error ? error.message : "Failed to apply voice.", "error");
  }
}

function newPersonality(): void {
  elements.personalityName.value = "";
  elements.instructions.value = "# Write your instructions here\n";
  elements.toolsText.value = "# tools enabled for this profile\n";
  elements.toolsAvailable.querySelectorAll<HTMLInputElement>('input[type="checkbox"]').forEach((input) => {
    input.checked = false;
  });
  setStatus(elements.personalityStatus, "Fill fields and save.");
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

function renderLogs(): void {
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

  const items = visibleLogs().slice(-300);
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "log-empty";
    empty.textContent = "Waiting for events...";
    elements.logs.replaceChildren(empty);
  } else {
    elements.logs.replaceChildren(...items.map(renderLog));
  }

  if (logUi.autoScroll) {
    elements.logs.scrollTop = elements.logs.scrollHeight;
    logUi.newSincePaused = 0;
    elements.jumpLatest.hidden = true;
  } else if (logUi.newSincePaused > 0) {
    elements.jumpLatestCount.textContent = String(logUi.newSincePaused);
    elements.jumpLatest.hidden = false;
  }
}

function renderLog(entry: LogEntry): HTMLLIElement {
  const item = document.createElement("li");
  const level = entryLevel(entry);
  item.dataset.level = level;
  item.dataset.category = entry.category;

  const time = document.createElement("time");
  time.textContent = entry.createdAt ? new Date(entry.createdAt).toLocaleTimeString() : "";
  const category = document.createElement("span");
  category.className = `cat cat--${entry.category.toLowerCase()}`;
  category.textContent = entry.category;
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
  return elements.logs.scrollHeight - elements.logs.scrollTop - elements.logs.clientHeight < 12;
}

function addLocalLog(message: string, level = "INFO", category = "SYSTEM"): void {
  state.logs = [
    ...state.logs,
    { type: "log", createdAt: new Date().toISOString(), level, category, message },
  ].slice(-500);
  if (!logUi.autoScroll) logUi.newSincePaused += 1;
  renderLogs();
}

function connectLogs(): void {
  const source = new EventSource("/api/dashboard/events");
  source.addEventListener("log", (event) => {
    const parsed = JSON.parse((event as MessageEvent).data) as LogEntry;
    state.logs = [...state.logs, parsed].slice(-500);
    if (state.logs.length < logUi.cleared) logUi.cleared = state.logs.length;
    if (!logUi.autoScroll) logUi.newSincePaused += 1;
    renderLogs();
  });
  source.onerror = () => addLocalLog("Dashboard event stream disconnected; retrying", "WARNING");
}

function wireEvents(): void {
  elements.cameraFeed.addEventListener("load", () => {
    elements.cameraEmpty.hidden = true;
  });
  elements.cameraFeed.addEventListener("error", () => {
    elements.cameraEmpty.hidden = false;
  });
  elements.refreshFaceState.addEventListener("click", () => void loadFaceState());
  elements.saveFace.addEventListener("click", () => void saveSelectedFace());
  elements.saveBackend.addEventListener("click", () => void saveBackend());
  elements.personalitySelect.addEventListener("change", () => void loadSelectedPersonality());
  elements.applyPersonality.addEventListener("click", () => void applyPersonality(false));
  elements.persistPersonality.addEventListener("click", () => void applyPersonality(true));
  elements.applyVoice.addEventListener("click", () => void applyVoice());
  elements.newPersonality.addEventListener("click", newPersonality);
  elements.savePersonality.addEventListener("click", () => void savePersonality());
  backendInputs().forEach((input) => {
    input.addEventListener("change", renderBackendControls);
  });
  elements.logs.addEventListener("scroll", () => {
    logUi.autoScroll = isLogAtBottom();
    if (logUi.autoScroll) {
      logUi.newSincePaused = 0;
      elements.jumpLatest.hidden = true;
    }
  });
  elements.jumpLatest.addEventListener("click", () => {
    logUi.autoScroll = true;
    logUi.newSincePaused = 0;
    elements.jumpLatest.hidden = true;
    elements.logs.scrollTop = elements.logs.scrollHeight;
  });
  elements.clearLogs.addEventListener("click", () => {
    logUi.cleared = state.logs.length;
    renderLogs();
  });
  elements.logSearch.addEventListener("input", () => {
    logUi.search = elements.logSearch.value;
    renderLogs();
  });
  document.querySelectorAll<HTMLButtonElement>(".filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      logUi.filter = (chip.dataset.level as LevelFilter) || "ALL";
      document.querySelectorAll(".filter-chip").forEach((item) => {
        item.classList.toggle("is-active", item === chip);
      });
      renderLogs();
    });
  });
}

async function init(): Promise<void> {
  wireEvents();
  connectLogs();
  await loadDashboardStatus().catch(() => addLocalLog("Dashboard status unavailable", "WARNING"));
  await loadFaceState().catch(() => addLocalLog("Face state unavailable", "WARNING", "VISION"));
  await loadPersonalities();
  renderLogs();
  refreshFrame();
  window.setInterval(() => void loadDashboardStatus().catch(() => undefined), 3000);
  window.setInterval(() => void loadFaceState().catch(() => undefined), 500);
  window.setInterval(refreshFrame, 350);
}

void init();
