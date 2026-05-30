import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const dashboardDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(dashboardDir, "..");
const projectSrc = path.resolve(projectRoot, "src");
const pythonSitePackages = path.resolve(projectRoot, ".venv/lib/python3.12/site-packages");
const gstreamerPythonRoot = path.resolve(pythonSitePackages, "gstreamer_python");
const gstreamerLibsRoot = path.resolve(pythonSitePackages, "gstreamer_libs");
const gstreamerGtkRoot = path.resolve(pythonSitePackages, "gstreamer_gtk");
const gstreamerPluginsRoot = path.resolve(pythonSitePackages, "gstreamer_plugins");
const gstreamerPluginsLibsRoot = path.resolve(pythonSitePackages, "gstreamer_plugins_libs");
const gstreamerPythonSite = path.resolve(
  gstreamerPythonRoot,
  "lib/python3.12/site-packages",
);
const gstreamerTypelibPaths = [
  path.resolve(gstreamerPythonRoot, "lib/girepository-1.0"),
  path.resolve(gstreamerLibsRoot, "lib/girepository-1.0"),
  path.resolve(gstreamerGtkRoot, "lib/girepository-1.0"),
];
const gstreamerLibraryPaths = [
  path.resolve(gstreamerPythonRoot, "lib"),
  path.resolve(gstreamerLibsRoot, "lib"),
  path.resolve(gstreamerGtkRoot, "lib"),
  path.resolve(gstreamerPluginsRoot, "lib"),
  path.resolve(gstreamerPluginsLibsRoot, "lib"),
];
const gstreamerPluginPaths = [
  path.resolve(gstreamerLibsRoot, "lib/gstreamer-1.0"),
  path.resolve(gstreamerGtkRoot, "lib/gstreamer-1.0"),
  path.resolve(gstreamerPluginsRoot, "lib/gstreamer-1.0"),
  path.resolve(gstreamerPluginsLibsRoot, "lib/gstreamer-1.0"),
];
const gstreamerPluginScanner = path.resolve(
  gstreamerLibsRoot,
  "libexec/gstreamer-1.0/gst-plugin-scanner",
);
const backendTarget = process.env.DASHBOARD_BACKEND_URL || "http://127.0.0.1:7860";
const defaultRobotHost = process.env.DASHBOARD_ROBOT_HOST || "";
const defaultRobotPort = process.env.DASHBOARD_ROBOT_PORT || "";
const defaultRobotName = process.env.DASHBOARD_ROBOT_NAME || "";
const defaultHeadTracker = process.env.DASHBOARD_HEAD_TRACKER || "yolo";
const defaultMediaBackend = process.env.DASHBOARD_MEDIA_BACKEND || "webrtc";
const defaultHardwareProfile = process.env.DASHBOARD_HARDWARE_PROFILE || "mac-mini-wired";

function quoteCommandArg(value: string): string {
  if (/^[A-Za-z0-9_./:@%+=,-]+$/.test(value)) return value;
  return `"${value.replace(/(["\\$`])/g, "\\$1")}"`;
}

function buildDefaultCommand(): string {
  const parts = [
    "REACHY_DASHBOARD_SERVER=1",
    "uv",
    "run",
    "reachy-mini-conversation-app",
    "--connection-mode",
    "network",
  ];

  if (defaultHeadTracker && defaultHeadTracker.toLowerCase() !== "none") {
    parts.push("--head-tracker", defaultHeadTracker);
  }
  if (defaultMediaBackend) parts.push("--media-backend", defaultMediaBackend);
  if (defaultHardwareProfile) parts.push("--hardware-profile", defaultHardwareProfile);
  if (defaultRobotHost) parts.push("--robot-host", defaultRobotHost);
  if (defaultRobotPort) parts.push("--robot-port", defaultRobotPort);
  if (defaultRobotName) parts.push("--robot-name", defaultRobotName);

  return parts.map(quoteCommandArg).join(" ");
}

const defaultCommand = process.env.DASHBOARD_DEFAULT_COMMAND || buildDefaultCommand();
const backendProbeTimeoutMs = 250;
const backendReadyPollMs = 500;
const backendReadyPollAttempts = 120;
const backendRouteProbeCacheMs = 1000;

type ProcessLogEvent = {
  id: number;
  type: "log";
  createdAt: string;
  level: "INFO" | "WARNING" | "ERROR";
  category: string;
  message: string;
};

type CommandParts = {
  command: string;
  args: string[];
  env: Record<string, string>;
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function gstreamerSafeEnv(baseEnv: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env = { ...baseEnv };
  const pluginSuffix = path.join("gstreamer_python", "lib", "gstreamer-1.0");
  for (const name of [
    "GST_PLUGIN_PATH_1_0",
    "GST_PLUGIN_SYSTEM_PATH_1_0",
    "GST_PLUGIN_PATH",
    "GST_PLUGIN_SYSTEM_PATH",
  ]) {
    const value = env[name];
    if (!value) continue;
    env[name] = value
      .split(path.delimiter)
      .filter((entry) => !entry.endsWith(pluginSuffix))
      .join(path.delimiter);
  }
  return env;
}

function dashboardProcessPlugin(): Plugin {
  const logs: ProcessLogEvent[] = [];
  const emitter = new EventEmitter();
  let nextLogId = 1;
  let child: ChildProcessWithoutNullStreams | null = null;
  let runningCommand = defaultCommand;
  let startedAt: string | null = null;
  let exitedAt: string | null = null;
  let exitCode: number | null = null;
  let signal: string | null = null;
  let stdoutRemainder = "";
  let stderrRemainder = "";
  let backendReadyAnnouncedForStartedAt: string | null = null;
  let backendRoutesReady = false;
  let lastBackendRouteProbeAt = 0;
  let failureHint: string | null = null;

  function addLog(message: string, level: ProcessLogEvent["level"] = "INFO", category = "PROCESS"): void {
    const cleaned = message.trim();
    if (!cleaned) return;
    const event: ProcessLogEvent = {
      id: nextLogId++,
      type: "log",
      createdAt: new Date().toISOString(),
      level,
      category,
      message: cleaned,
    };
    logs.push(event);
    if (logs.length > 500) logs.shift();
    emitter.emit("log", event);
  }

  function classifyLine(line: string): string {
    const lowered = line.toLowerCase();
    if (lowered.includes("reachy") || lowered.includes("robot") || lowered.includes("daemon") || lowered.includes("connectionerror")) {
      return "ROBOT";
    }
    if (lowered.includes("face") || lowered.includes("vision") || lowered.includes("camera") || lowered.includes("yolo")) {
      return "VISION";
    }
    if (lowered.includes("tool")) return "TOOL";
    if (lowered.includes("llm") || lowered.includes("huggingface") || lowered.includes("realtime")) {
      return "LLM";
    }
    if (lowered.includes("audio") || lowered.includes("voice") || lowered.includes("speech")) return "VOICE";
    if (lowered.includes("movement") || lowered.includes("motion") || lowered.includes("head")) return "MOTION";
    return "PROCESS";
  }

  function levelForLine(line: string, fallback: ProcessLogEvent["level"]): ProcessLogEvent["level"] {
    if (/^\s*INFO:/.test(line)) return "INFO";
    if (/\b(ERROR|CRITICAL|FATAL)\b/.test(line)) return "ERROR";
    if (/^\s*(Traceback|ConnectionError|TimeoutError):?/.test(line)) return "ERROR";
    if (/\b(WARNING|WARN)\b/.test(line)) return "WARNING";
    return fallback;
  }

  function isReachyConversationCommand(): boolean {
    return (
      runningCommand.includes("reachy-mini-conversation-app") ||
      runningCommand.includes("reachy_mini_conversation_app.main")
    );
  }

  function isBackendProbeNoise(line: string): boolean {
    return /GET \/api\/dashboard\/status HTTP\/1\.1"\s+404 Not Found/.test(line);
  }

  function isTracebackNoise(line: string): boolean {
    const trimmed = line.trim();
    return (
      trimmed === "Traceback (most recent call last):" ||
      trimmed === "During handling of the above exception, another exception occurred:" ||
      trimmed.startsWith("File \"") ||
      trimmed.startsWith("with ReachyMini(") ||
      trimmed.startsWith("client = ") ||
      trimmed === "client.wait_for_connection(timeout=timeout)" ||
      trimmed === "app.wrapped_run()" ||
      trimmed.startsWith("self.client") ||
      trimmed.startsWith("raise ") ||
      /^[~^]+$/.test(trimmed)
    );
  }

  function friendlyFailureForLine(line: string): string | null {
    if (
      line.includes("Timeout while waiting for connection with the server") ||
      line.includes("Could not connect to daemon on localhost")
    ) {
      return "Robot unavailable: the local daemon is reachable, but robot telemetry is not streaming. Reconnect the robot to Wi-Fi and start again.";
    }
    if (
      line.includes("Network connection attempt failed") ||
      line.includes("Auto connection: both localhost and remote attempts failed") ||
      line.includes("No address associated with hostname") ||
      line.includes("nodename nor servname provided")
    ) {
      return "Robot unavailable: the daemon host is not reachable. Check Wi-Fi, hostname, and daemon status, then start again.";
    }
    return null;
  }

  function noteFailureHint(message: string): void {
    if (failureHint === message) return;
    failureHint = message;
    addLog(message, "ERROR", "ROBOT");
  }

  function appendOutput(chunk: Buffer, stream: "stdout" | "stderr"): void {
    const fallbackLevel = stream === "stderr" ? "WARNING" : "INFO";
    const current = (stream === "stdout" ? stdoutRemainder : stderrRemainder) + chunk.toString("utf8");
    const lines = current.split(/\r?\n/);
    const remainder = lines.pop() || "";
    if (stream === "stdout") stdoutRemainder = remainder;
    else stderrRemainder = remainder;
    for (const line of lines) {
      if (isBackendProbeNoise(line)) continue;
      const friendlyFailure = friendlyFailureForLine(line);
      if (friendlyFailure) {
        noteFailureHint(friendlyFailure);
        continue;
      }
      if (isReachyConversationCommand() && isTracebackNoise(line)) continue;
      addLog(line, levelForLine(line, fallbackLevel), classifyLine(line));
    }
  }

  function flushOutputRemainders(): void {
    if (stdoutRemainder.trim()) addLog(stdoutRemainder, levelForLine(stdoutRemainder, "INFO"), classifyLine(stdoutRemainder));
    if (stderrRemainder.trim()) addLog(stderrRemainder, levelForLine(stderrRemainder, "WARNING"), classifyLine(stderrRemainder));
    stdoutRemainder = "";
    stderrRemainder = "";
  }

  function isRunning(): boolean {
    return child !== null && child.exitCode === null && child.signalCode === null;
  }

  async function isBackendReady(): Promise<boolean> {
    const now = Date.now();
    if (backendRoutesReady) return true;
    if (now - lastBackendRouteProbeAt < backendRouteProbeCacheMs) return false;
    lastBackendRouteProbeAt = now;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), backendProbeTimeoutMs);
    try {
      const response = await fetch(new URL("/api/dashboard/status", backendTarget), {
        cache: "no-store",
        signal: controller.signal,
      });
      backendRoutesReady = response.ok;
      return backendRoutesReady;
    } catch {
      backendRoutesReady = false;
      return false;
    } finally {
      clearTimeout(timeout);
    }
  }

  async function statusPayload(): Promise<Record<string, unknown>> {
    return {
      available: true,
      running: isRunning(),
      pid: isRunning() ? child?.pid || null : null,
      command: runningCommand,
      defaultCommand,
      defaultRobotHost,
      defaultRobotPort,
      defaultRobotName,
      defaultHeadTracker,
      defaultMediaBackend,
      defaultHardwareProfile,
      startedAt,
      exitedAt,
      exitCode,
      signal,
      backendTarget,
      backendReady: isRunning() ? await isBackendReady() : false,
      failureHint,
    };
  }

  async function waitForBackendReady(runStartedAt: string): Promise<void> {
    for (let attempt = 0; attempt < backendReadyPollAttempts; attempt += 1) {
      if (!isRunning() || startedAt !== runStartedAt) return;
      if (await isBackendReady()) {
        if (backendReadyAnnouncedForStartedAt !== runStartedAt) {
          backendReadyAnnouncedForStartedAt = runStartedAt;
          addLog(`Dashboard API is ready at ${backendTarget}`, "INFO", "PROCESS");
        }
        return;
      }
      await sleep(backendReadyPollMs);
    }

    if (isRunning() && startedAt === runStartedAt) {
      addLog(`Still waiting for dashboard API at ${backendTarget}`, "WARNING", "PROCESS");
    }
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
      if ("|&;<>".includes(char)) {
        throw new Error("shell_operators_not_supported");
      }
      current += char;
    }

    if (escaped) current += "\\";
    if (quote) throw new Error("unterminated_quote");
    if (current) parts.push(current);
    return parts;
  }

  function parseCommand(commandText: string): CommandParts {
    const parts = splitCommand(commandText);
    const env: Record<string, string> = {};
    while (parts[0] && /^[A-Za-z_][A-Za-z0-9_]*=/.test(parts[0])) {
      const [name, ...valueParts] = parts.shift()!.split("=");
      env[name] = valueParts.join("=");
    }
    const command = parts.shift();
    if (!command) throw new Error("empty_command");
    return { command, args: parts, env };
  }

  function startProcess(commandText: string): void {
    if (isRunning()) throw new Error("process_already_running");
    const parsed = parseCommand(commandText);
    runningCommand = commandText.trim();
    startedAt = new Date().toISOString();
    exitedAt = null;
    exitCode = null;
    signal = null;
    backendReadyAnnouncedForStartedAt = null;
    backendRoutesReady = false;
    lastBackendRouteProbeAt = 0;
    failureHint = null;
    stdoutRemainder = "";
    stderrRemainder = "";
    addLog(`$ ${runningCommand}`, "INFO", "PROCESS");
    addLog(`Waiting for dashboard API at ${backendTarget}`, "INFO", "PROCESS");

    child = spawn(parsed.command, parsed.args, {
      cwd: projectRoot,
      detached: true,
      env: {
        ...gstreamerSafeEnv(process.env),
        GST_REGISTRY_FORK: "no",
        GST_REGISTRY_UPDATE: "no",
        GST_PLUGIN_PATH_1_0: [gstreamerPluginPaths.join(path.delimiter), process.env.GST_PLUGIN_PATH_1_0]
          .filter(Boolean)
          .join(path.delimiter),
        GST_PLUGIN_SCANNER_1_0: process.env.GST_PLUGIN_SCANNER_1_0 || gstreamerPluginScanner,
        GI_TYPELIB_PATH: [gstreamerTypelibPaths.join(path.delimiter), process.env.GI_TYPELIB_PATH]
          .filter(Boolean)
          .join(path.delimiter),
        DYLD_LIBRARY_PATH: [gstreamerLibraryPaths.join(path.delimiter), process.env.DYLD_LIBRARY_PATH]
          .filter(Boolean)
          .join(path.delimiter),
        PYTHONUNBUFFERED: "1",
        PYTHONPATH: [projectSrc, gstreamerPythonSite, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
        ...parsed.env,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    child.stdout.on("data", (chunk: Buffer) => appendOutput(chunk, "stdout"));
    child.stderr.on("data", (chunk: Buffer) => appendOutput(chunk, "stderr"));
    child.on("error", (error) => {
      flushOutputRemainders();
      addLog(`Failed to start command: ${error.message}`, "ERROR", "PROCESS");
      child = null;
    });
    child.on("exit", (code, exitSignal) => {
      flushOutputRemainders();
      exitedAt = new Date().toISOString();
      exitCode = code;
      signal = exitSignal;
      if (code && code !== 0 && !failureHint) {
        failureHint = `Command exited with code ${code}. Check live logs for details.`;
      }
      const suffix = exitSignal ? `signal ${exitSignal}` : `code ${code ?? "unknown"}`;
      addLog(`Command exited with ${suffix}`, code === 0 ? "INFO" : "WARNING", "PROCESS");
      child = null;
    });
    void waitForBackendReady(startedAt);
  }

  function stopProcess(): void {
    if (!isRunning() || child === null) return;
    addLog("Stopping command with SIGINT", "INFO", "PROCESS");
    signalChild("SIGINT");
    const stoppedChild = child;
    setTimeout(() => {
      if (stoppedChild.exitCode === null && stoppedChild.signalCode === null) {
        addLog("Command did not stop after SIGINT; sending SIGTERM", "WARNING", "PROCESS");
        signalChild("SIGTERM", stoppedChild);
      }
    }, 5000);
  }

  function signalChild(signal: NodeJS.Signals, target: ChildProcessWithoutNullStreams | null = child): void {
    if (target === null || target.pid === undefined) return;
    try {
      process.kill(-target.pid, signal);
    } catch {
      try {
        target.kill(signal);
      } catch {
        // Process may already be gone.
      }
    }
  }

  function sendJson(res: { statusCode: number; setHeader: (name: string, value: string) => void; end: (body: string) => void }, status: number, payload: object): void {
    res.statusCode = status;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(payload));
  }

  function readRequestBody(req: NodeJS.ReadableStream): Promise<string> {
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
      req.on("error", reject);
    });
  }

  function formatSse(event: ProcessLogEvent): string {
    return `id: ${event.id}\nevent: log\ndata: ${JSON.stringify(event)}\n\n`;
  }

  function isBackendProxyPath(pathname: string): boolean {
    return (
      pathname.startsWith("/api") ||
      pathname === "/backend_config" ||
      pathname.startsWith("/profiles") ||
      pathname.startsWith("/voices") ||
      pathname === "/ready" ||
      pathname === "/status"
    );
  }

  function fallbackBackendStatus(): Record<string, unknown> {
    return {
      active_backend: "local",
      backend_provider: "local",
      has_local_backend: false,
      has_hf_session_url: false,
      has_hf_ws_url: false,
      has_hf_connection: false,
      hf_connection_mode: "deployed",
      hf_direct_host: "localhost",
      hf_direct_port: 8765,
      can_proceed: false,
      can_proceed_with_hf: false,
      can_proceed_with_local: false,
      requires_restart: false,
      backend_unavailable: true,
      backend_unavailable_reason: isRunning()
        ? "App process is starting; dashboard API is not ready yet."
        : failureHint || "App process is not running.",
    };
  }

  function fallbackDashboardStatus(): Record<string, unknown> {
    return {
      ...fallbackBackendStatus(),
      camera: {
        available: false,
        frame_available: false,
        head_tracker: null,
      },
      face_recognition: {
        available: false,
        db_path: null,
        visible_count: 0,
        people: [],
      },
    };
  }

  function sendEmptyImage(res: {
    statusCode: number;
    setHeader: (name: string, value: string) => void;
    end: (body?: string) => void;
  }): void {
    res.statusCode = 503;
    res.setHeader("Content-Type", "text/plain");
    res.setHeader("Cache-Control", "no-store");
    res.end("camera_unavailable");
  }

  function sendBackendWaitingEvent(res: {
    writeHead: (statusCode: number, headers: Record<string, string>) => void;
    write: (chunk: string) => void;
    end: () => void;
  }): void {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });
    res.write(
      formatSse({
        id: nextLogId++,
        type: "log",
        createdAt: new Date().toISOString(),
        level: "INFO",
        category: "SYSTEM",
        message: `Waiting for app API at ${backendTarget}`,
      }),
    );
    setTimeout(() => res.end(), 2000);
  }

  async function handleBackendFallback(
    req: NodeJS.ReadableStream & { method?: string; url?: string },
    res: {
      statusCode: number;
      setHeader: (name: string, value: string) => void;
      writeHead: (statusCode: number, headers: Record<string, string>) => void;
      write: (chunk: string) => void;
      end: (body?: string) => void;
    },
    next: () => void,
  ): Promise<void> {
    const url = new URL(req.url || "/", "http://localhost");
    if (!isBackendProxyPath(url.pathname)) {
      next();
      return;
    }

    if (await isBackendReady()) {
      next();
      return;
    }

    if (req.method === "GET" && url.pathname === "/api/dashboard/status") {
      sendJson(res, 200, fallbackDashboardStatus());
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/face/state") {
      sendJson(res, 200, { ok: true, available: false, faces: [] });
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/face/frame.jpg") {
      sendEmptyImage(res);
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/dashboard/events") {
      sendBackendWaitingEvent(res);
      return;
    }
    if (req.method === "GET" && url.pathname === "/status") {
      sendJson(res, 200, fallbackBackendStatus());
      return;
    }
    if (req.method === "GET" && url.pathname === "/ready") {
      sendJson(res, 200, { ready: false });
      return;
    }
    if (req.method === "GET" && url.pathname === "/voices") {
      sendJson(res, 200, []);
      return;
    }

    sendJson(res, 503, { ok: false, error: "app_backend_not_ready" });
  }

  return {
    name: "reachy-dashboard-process",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url || "/", "http://localhost");
        if (!url.pathname.startsWith("/__dashboard/process")) {
          next();
          return;
        }

        try {
          if (req.method === "GET" && url.pathname === "/__dashboard/process/status") {
            sendJson(res, 200, await statusPayload());
            return;
          }

          if (req.method === "POST" && url.pathname === "/__dashboard/process/start") {
            const rawBody = await readRequestBody(req);
            const body = rawBody ? JSON.parse(rawBody) : {};
            const commandText = String(body.command || defaultCommand).trim();
            startProcess(commandText);
            sendJson(res, 200, await statusPayload());
            return;
          }

          if (req.method === "POST" && url.pathname === "/__dashboard/process/stop") {
            stopProcess();
            sendJson(res, 200, await statusPayload());
            return;
          }

          if (req.method === "GET" && url.pathname === "/__dashboard/process/events") {
            res.writeHead(200, {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-cache, no-transform",
              Connection: "keep-alive",
              "X-Accel-Buffering": "no",
            });
            for (const event of logs.slice(-100)) res.write(formatSse(event));
            const listener = (event: ProcessLogEvent) => res.write(formatSse(event));
            emitter.on("log", listener);
            req.on("close", () => emitter.off("log", listener));
            return;
          }

          sendJson(res, 404, { ok: false, error: "not_found" });
        } catch (error) {
          const message = error instanceof Error ? error.message : "request_failed";
          sendJson(res, 400, { ok: false, error: message });
        }
      });

      server.middlewares.use((req, res, next) => {
        void handleBackendFallback(req, res, next);
      });

      server.httpServer?.once("close", () => {
        if (isRunning()) stopProcess();
      });
    },
  };
}

export default defineConfig({
  base: "/static/",
  plugins: [dashboardProcessPlugin()],
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/backend_config": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/profiles": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/voices": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/ready": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/status": {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "../src/reachy_mini_conversation_app/static",
    emptyOutDir: true,
    assetsDir: "assets",
  },
});
