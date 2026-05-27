import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const dashboardDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(dashboardDir, "..");
const backendTarget = process.env.DASHBOARD_BACKEND_URL || "http://127.0.0.1:7860";
const defaultCommand =
  process.env.DASHBOARD_DEFAULT_COMMAND ||
  "uv run python -m reachy_mini_conversation_app.main --head-tracker yolo";

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
    if (lowered.includes("face") || lowered.includes("vision") || lowered.includes("camera") || lowered.includes("yolo")) {
      return "VISION";
    }
    if (lowered.includes("tool")) return "TOOL";
    if (lowered.includes("openai") || lowered.includes("gemini") || lowered.includes("huggingface") || lowered.includes("realtime")) {
      return "LLM";
    }
    if (lowered.includes("audio") || lowered.includes("voice") || lowered.includes("speech")) return "VOICE";
    if (lowered.includes("movement") || lowered.includes("motion") || lowered.includes("head")) return "MOTION";
    return "PROCESS";
  }

  function levelForLine(line: string, fallback: ProcessLogEvent["level"]): ProcessLogEvent["level"] {
    if (/\b(ERROR|CRITICAL|FATAL)\b/.test(line)) return "ERROR";
    if (/\b(WARNING|WARN)\b/.test(line)) return "WARNING";
    return fallback;
  }

  function appendOutput(chunk: Buffer, stream: "stdout" | "stderr"): void {
    const fallbackLevel = stream === "stderr" ? "WARNING" : "INFO";
    const current = (stream === "stdout" ? stdoutRemainder : stderrRemainder) + chunk.toString("utf8");
    const lines = current.split(/\r?\n/);
    const remainder = lines.pop() || "";
    if (stream === "stdout") stdoutRemainder = remainder;
    else stderrRemainder = remainder;
    for (const line of lines) {
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

  function statusPayload(): Record<string, unknown> {
    return {
      available: true,
      running: isRunning(),
      pid: isRunning() ? child?.pid || null : null,
      command: runningCommand,
      defaultCommand,
      startedAt,
      exitedAt,
      exitCode,
      signal,
      backendTarget,
    };
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
    stdoutRemainder = "";
    stderrRemainder = "";
    addLog(`$ ${runningCommand}`, "INFO", "PROCESS");

    child = spawn(parsed.command, parsed.args, {
      cwd: projectRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1", ...parsed.env },
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
      const suffix = exitSignal ? `signal ${exitSignal}` : `code ${code ?? "unknown"}`;
      addLog(`Command exited with ${suffix}`, code === 0 ? "INFO" : "WARNING", "PROCESS");
      child = null;
    });
  }

  function stopProcess(): void {
    if (!isRunning() || child === null) return;
    addLog("Stopping command with SIGINT", "INFO", "PROCESS");
    child.kill("SIGINT");
    const stoppedChild = child;
    setTimeout(() => {
      if (stoppedChild.exitCode === null && stoppedChild.signalCode === null) {
        addLog("Command did not stop after SIGINT; sending SIGTERM", "WARNING", "PROCESS");
        stoppedChild.kill("SIGTERM");
      }
    }, 5000);
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
            sendJson(res, 200, statusPayload());
            return;
          }

          if (req.method === "POST" && url.pathname === "/__dashboard/process/start") {
            const rawBody = await readRequestBody(req);
            const body = rawBody ? JSON.parse(rawBody) : {};
            const commandText = String(body.command || defaultCommand).trim();
            startProcess(commandText);
            sendJson(res, 200, statusPayload());
            return;
          }

          if (req.method === "POST" && url.pathname === "/__dashboard/process/stop") {
            stopProcess();
            sendJson(res, 200, statusPayload());
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
      "/personalities": {
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
