import { state, subscribe } from "../state.ts";
import { el, shortValue, formatMs, formatFps } from "../util.ts";
import type { PerformanceStatus, Tone, View } from "../types.ts";

type MetricValue = { text: string; tone?: Tone };
type Metric = { label: string; get: () => MetricValue };
type Panel = { title: string; metrics: Metric[] };
type Stage = { label: string; get: () => MetricValue; sub?: () => string };

function perf(): PerformanceStatus {
  return state.status?.performance || {};
}
function rec(value: unknown): Record<string, unknown> {
  return (value as Record<string, unknown>) || {};
}
function num(value: unknown): number | null {
  const n = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(n) ? n : null;
}
function ms(value: unknown): MetricValue {
  return { text: formatMs(num(value)) };
}
function basename(value: unknown): string {
  const text = shortValue(value);
  if (text === "-") return text;
  return text.split(/[\\/]/).pop() || text;
}
function flag(value: unknown, goodWhenTrue = true): MetricValue {
  if (value === null || value === undefined || value === "") return { text: "-" };
  const truthy = value === true || value === "yes";
  return { text: truthy ? "yes" : "no", tone: truthy === goodWhenTrue ? "ok" : "warn" };
}

function vadValue(): MetricValue {
  const stateText = shortValue(rec(perf().voice_activity).vad_state);
  if (stateText === "-") return { text: "-" };
  return { text: stateText, tone: stateText === "speech" ? "ok" : "muted" };
}
const ROUTER_LABELS: Record<string, string> = {
  router_ok: "tool ok",
  router_no_tool: "no tool",
  router_parse_error: "parse error",
};
function routerStatusText(): string {
  const status = rec(perf().local_model).qwen_router_status;
  return status ? ROUTER_LABELS[String(status)] || String(status) : "—";
}
function routerTone(): Tone | undefined {
  const status = rec(perf().local_model).qwen_router_status;
  if (status === "router_ok") return "ok";
  if (status === "router_parse_error") return "err";
  if (status === "router_no_tool") return "muted";
  return undefined;
}
function queueText(): MetricValue {
  const depth = num(perf().audio_queue_depth_s);
  if (depth === null) return { text: "-" };
  const text = depth < 1 ? `${(depth * 1000).toFixed(0)}ms` : `${depth.toFixed(2)}s`;
  return { text, tone: depth > 1.5 ? "warn" : undefined };
}
function frameAgeValue(): MetricValue {
  const age = num(perf().camera_frame_age_ms);
  if (age === null) return { text: "-" };
  return { text: formatMs(age), tone: age > 1500 ? "warn" : undefined };
}
function visionStatusValue(): MetricValue {
  const status = rec(perf().local_model).last_vision_status;
  if (!status) return { text: "—" };
  return { text: String(status), tone: status === "ok" ? "ok" : status === "error" ? "err" : undefined };
}
function pair(a: unknown, b: unknown): MetricValue {
  return { text: `${shortValue(a)} · ${shortValue(b)}` };
}

const PIPELINE: Stage[] = [
  { label: "VAD", get: vadValue },
  { label: "STT", get: () => ms(perf().stt_ms) },
  { label: "Router", get: () => ({ text: formatMs(num(rec(perf().local_model).qwen_router_latency_ms)), tone: routerTone() }), sub: routerStatusText },
  { label: "First token", get: () => ms(perf().llm_first_token_ms) },
  { label: "LLM total", get: () => ms(perf().llm_total_ms) },
  { label: "TTS", get: () => ms(perf().tts_ms) },
  { label: "First audio", get: () => ms(perf().first_audio_ms) },
];

const PANELS: Panel[] = [
  {
    title: "Voice & VAD",
    metrics: [
      { label: "State", get: vadValue },
      { label: "Speech confidence", get: () => ({ text: shortValue(rec(perf().voice_activity).speech_confidence_ratio) }) },
      { label: "Noise floor", get: () => ({ text: shortValue(rec(perf().voice_activity).noise_floor_rms) }) },
      { label: "Frame SNR", get: () => { const v = rec(perf().voice_activity).last_frame_snr_db; return { text: v == null ? "-" : `${shortValue(v)} dB` }; } },
      { label: "Noise class", get: () => ({ text: shortValue(rec(perf().voice_activity).last_frame_noise_class) }) },
      { label: "Suppress window", get: () => { const v = rec(perf().voice_activity).robot_noise_suppression_window_ms; return { text: v == null ? "-" : `${shortValue(v)} ms` }; } },
      { label: "Motion playback", get: () => flag(rec(perf().voice_activity).active_motion_playback, false) },
      { label: "Rejected", get: () => ({ text: `${shortValue(rec(perf().voice_activity).rejected_segment_count ?? 0)} · ${shortValue(rec(perf().voice_activity).last_reject_reason)}` }) },
    ],
  },
  {
    title: "Audio I/O",
    metrics: [
      { label: "Input frames", get: () => ({ text: shortValue(perf().audio_input_frames) }) },
      { label: "Output frames", get: () => ({ text: shortValue(perf().audio_output_frames) }) },
      { label: "Dropped", get: () => { const v = num(perf().dropped_audio_frames) ?? 0; return { text: String(v), tone: v > 0 ? "warn" : undefined }; } },
      { label: "Queue depth", get: queueText },
    ],
  },
  {
    title: "Camera & Vision",
    metrics: [
      { label: "Frame rate", get: () => ({ text: formatFps(num(perf().camera_fps)) }) },
      { label: "Frame age", get: frameAgeValue },
      { label: "Vision model", get: () => ({ text: shortValue(rec(perf().local_model).vision_model) }) },
      { label: "Vision status", get: visionStatusValue },
    ],
  },
  {
    title: "Link & daemon",
    metrics: [
      { label: "Daemon", get: () => { const s = shortValue(perf().daemon_state); return { text: `${s} · ${formatMs(num(perf().daemon_rtt_ms))}`, tone: s === "running" ? "ok" : s === "-" ? undefined : "warn" }; } },
      { label: "Media host", get: () => { const t = rec(perf().transport); const wlan = t.media_host_source === "daemon_wlan_ip"; return { text: `${shortValue(t.media_host)} (${shortValue(t.media_host_source)})`, tone: shortValue(t.media_host) === "-" ? undefined : wlan ? "warn" : "ok" }; } },
      { label: "Control host", get: () => ({ text: shortValue(rec(perf().transport).control_host) }) },
      { label: "Hardware profile", get: () => ({ text: shortValue(rec(perf().transport).hardware_profile) }) },
      { label: "Media available", get: () => flag(rec(perf().health_checks).media_available) },
      { label: "Wired link", get: () => flag(rec(perf().health_checks).wired_link_present) },
    ],
  },
  {
    title: "Models & TTS",
    metrics: [
      { label: "Chat model", get: () => pair(rec(perf().local_model).chat_provider, rec(perf().local_model).configured_model) },
      { label: "Last tool", get: () => { const s = rec(perf().local_model).last_tool_status; return { text: shortValue(s), tone: s === "error" ? "err" : undefined }; } },
      { label: "Router model", get: () => pair(rec(perf().local_model).router_provider, rec(perf().local_model).router_model) },
      { label: "TTS voice", get: () => pair(rec(perf().local_tts).provider, basename(rec(perf().local_tts).voice_model)) },
      { label: "TTS ready", get: () => { const tts = rec(perf().local_tts); const v = flag(tts.ready); return tts.error ? { text: `no · ${shortValue(tts.error)}`, tone: "warn" } : v; } },
      { label: "Last model error", get: () => { const e = rec(perf().local_model).last_local_model_error; return { text: e ? String(e).slice(0, 60) : "—", tone: e ? "err" : undefined }; } },
    ],
  },
];

function setTone(node: HTMLElement, tone: Tone | undefined): void {
  node.classList.remove("val--ok", "val--warn", "val--err", "val--muted");
  if (tone) node.classList.add(`val--${tone}`);
}

export function createDiagnosticsView(): View {
  let unsub: (() => void) | null = null;
  const cells: Array<() => void> = [];

  // Turn latency pipeline
  const stageNodes = PIPELINE.map((stage, index) => {
    const value = el("span", { class: "stage-value" });
    const sub = stage.sub ? el("span", { class: "stage-sub" }) : null;
    const box = el("div", { class: "stage" }, [el("span", { class: "stage-label", text: stage.label }), value, sub]);
    cells.push(() => {
      const result = stage.get();
      value.textContent = result.text;
      setTone(value, result.tone);
      if (sub && stage.sub) sub.textContent = stage.sub();
    });
    const out: HTMLElement[] = [box];
    if (index < PIPELINE.length - 1) out.push(el("span", { class: "stage-sep", "aria-hidden": "true", text: "›" }));
    return out;
  });
  const pipeline = el("div", { class: "card diag-pipeline" }, [
    el("div", { class: "card-head" }, [
      el("div", { class: "card-title", text: "Turn latency" }),
      el("span", { class: "section-desc", text: "most recent voice turn" }),
    ]),
    el("div", { class: "pipeline" }, stageNodes.flat()),
  ]);

  // Dense metric panels
  const panelCards = PANELS.map((panel) => {
    const rows = panel.metrics.map((metric) => {
      const dd = el("dd", { class: "metric-val" });
      cells.push(() => {
        const result = metric.get();
        dd.textContent = result.text;
        setTone(dd, result.tone);
      });
      return el("div", { class: "metric-row" }, [el("dt", { text: metric.label }), dd]);
    });
    return el("div", { class: "card diag-panel" }, [
      el("div", { class: "card-head" }, [el("div", { class: "card-title", text: panel.title })]),
      el("dl", { class: "metric-list" }, rows),
    ]);
  });

  function update(): void {
    for (const apply of cells) apply();
  }

  return {
    mount(container) {
      container.append(el("div", { class: "view-scroll" }, [
        el("div", { class: "diag-stack" }, [pipeline, el("div", { class: "diag-grid" }, panelCards)]),
      ]));
      unsub = subscribe(update);
      update();
    },
    update,
    destroy() {
      unsub?.();
      unsub = null;
    },
  };
}
