import "./styles/base.css";
import "./styles/shell.css";
import "./styles/components.css";
import "./styles/views.css";

import {
  state,
  appendLog,
  addLocalLog,
  loadDashboardStatus,
  loadProcessStatus,
  loadFaceState,
} from "./state.ts";
import { byId } from "./util.ts";
import { loadPersistedLaunchOptions } from "./components/launchConfig.ts";
import { createShell } from "./shell.ts";
import { createRouter } from "./router.ts";
import { createMonitorView } from "./views/monitor.ts";
import { createLogsView } from "./views/logs.ts";
import { createDiagnosticsView } from "./views/diagnostics.ts";
import { createSettingsView } from "./views/settings.ts";
import type { LogEntry } from "./types.ts";

let dashboardLogDisconnectedAt = 0;
let processLogSource: EventSource | null = null;

function connectLogs(): void {
  const source = new EventSource("/api/dashboard/events");
  source.onopen = () => {
    dashboardLogDisconnectedAt = 0;
  };
  source.addEventListener("log", (event) => {
    appendLog(JSON.parse((event as MessageEvent).data) as LogEntry);
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
    appendLog(JSON.parse((event as MessageEvent).data) as LogEntry);
  });
  processLogSource.onerror = () => {
    processLogSource?.close();
    processLogSource = null;
  };
}

function onProcessStatus(available: boolean): void {
  if (available && state.process) {
    connectProcessLogs();
  }
}

async function init(): Promise<void> {
  loadPersistedLaunchOptions();

  const shell = createShell(byId("app"));
  const router = createRouter({
    container: shell.viewContainer,
    factories: {
      monitor: createMonitorView,
      logs: createLogsView,
      diagnostics: createDiagnosticsView,
      settings: createSettingsView,
    },
    onNavigate: (id) => shell.setActive(id),
  });
  router.start();

  onProcessStatus(await loadProcessStatus());
  connectLogs();
  await loadDashboardStatus().catch(() => addLocalLog("Dashboard status unavailable", "WARNING"));
  await loadFaceState().catch(() => addLocalLog("Face state unavailable", "WARNING", "VISION"));

  window.setInterval(() => void loadProcessStatus().then(onProcessStatus), 2000);
  window.setInterval(() => void loadDashboardStatus().catch(() => undefined), 3000);
  window.setInterval(() => void loadFaceState().catch(() => undefined), 500);
}

void init();
