import { api } from "./api.ts";
import type { DashboardStatus, FaceBox, LaunchConfig, LevelFilter, LogEntry, ProcessStatus, ViewId } from "./types.ts";

export function defaultLaunchConfig(): LaunchConfig {
  return {
    connectionMode: "network",
    robotHost: "",
    robotPort: "",
    robotName: "",
    headTracker: "off",
    mediaBackend: "webrtc",
    hardwareProfile: "mac-mini-wired",
    camera: true,
    localVision: false,
    debug: false,
    rawOverride: "",
  };
}

export const state = {
  status: null as DashboardStatus | null,
  process: null as ProcessStatus | null,
  processAvailable: false,
  defaultsSeeded: false,
  faces: [] as FaceBox[],
  faceStateAvailable: false,
  faceRecognitionAvailable: false,
  selectedFaceId: null as number | null,
  logs: [] as LogEntry[],
  launchConfig: defaultLaunchConfig(),
  activeView: "monitor" as ViewId,
};

export const logUi = {
  filter: "ALL" as LevelFilter,
  search: "",
  autoScroll: true,
  newSincePaused: 0,
  cleared: 0,
};

type Listener = () => void;
const listeners = new Set<Listener>();

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function notify(): void {
  for (const fn of [...listeners]) fn();
}

type LogListener = (entry: LogEntry) => void;
const logListeners = new Set<LogListener>();

export function onLog(fn: LogListener): () => void {
  logListeners.add(fn);
  return () => logListeners.delete(fn);
}

export function appendLog(entry: LogEntry): void {
  state.logs = [...state.logs, entry].slice(-500);
  if (state.logs.length < logUi.cleared) logUi.cleared = state.logs.length;
  for (const fn of [...logListeners]) fn(entry);
}

export function addLocalLog(message: string, level = "INFO", category = "SYSTEM"): void {
  appendLog({ type: "log", createdAt: new Date().toISOString(), level, category, message });
}

export async function loadDashboardStatus(): Promise<void> {
  state.status = await api.dashboardStatus();
  notify();
}

export async function loadProcessStatus(): Promise<boolean> {
  try {
    state.process = await api.processStatus();
    state.processAvailable = !!state.process.available;
    notify();
    return true;
  } catch {
    state.processAvailable = false;
    notify();
    return false;
  }
}

export async function loadFaceState(): Promise<void> {
  const data = await api.faceState();
  state.faceStateAvailable = !!data.available;
  state.faceRecognitionAvailable = data.recognition_available !== false;
  state.faces = data.faces || [];
  if (state.selectedFaceId !== null && !state.faces.some((face) => face.id === state.selectedFaceId)) {
    state.selectedFaceId = null;
  }
  notify();
}
