import { state, subscribe } from "../state.ts";
import { el, shortValue, formatMs, formatFps } from "../util.ts";
import type { PerformanceStatus, View } from "../types.ts";

type Metric = { label: string; get: () => string };
type Group = { title: string; metrics: Metric[] };

function perf(): PerformanceStatus {
  return state.status?.performance || {};
}
function rec(value: unknown): Record<string, unknown> {
  return (value as Record<string, unknown>) || {};
}

const GROUPS: Group[] = [
  {
    title: "Transport & link",
    metrics: [
      { label: "Control host", get: () => shortValue(rec(perf().transport).control_host) },
      { label: "Media host", get: () => `${shortValue(rec(perf().transport).media_host)} (${shortValue(rec(perf().transport).media_host_source)})` },
      { label: "Daemon", get: () => `${shortValue(perf().daemon_state)} / ${formatMs(perf().daemon_rtt_ms)}` },
    ],
  },
  {
    title: "Media",
    metrics: [
      { label: "State", get: () => `available ${shortValue(rec(perf().media_state).available)}, released ${shortValue(rec(perf().media_state).released)}` },
    ],
  },
  {
    title: "Camera",
    metrics: [
      { label: "Frames", get: () => `${formatFps(perf().camera_fps)} / age ${formatMs(perf().camera_frame_age_ms)}` },
    ],
  },
  {
    title: "Audio & VAD",
    metrics: [
      { label: "Audio", get: () => `in ${shortValue(perf().audio_input_frames)}, out ${shortValue(perf().audio_output_frames)}, drop ${shortValue(perf().dropped_audio_frames)}, q ${shortValue(perf().audio_queue_depth_s)}s` },
      { label: "VAD", get: () => `${shortValue(rec(perf().voice_activity).vad_state)}, active ${shortValue(rec(perf().voice_activity).active_motion_playback)}` },
      { label: "Noise", get: () => `floor ${shortValue(rec(perf().voice_activity).noise_floor_rms)}, conf ${shortValue(rec(perf().voice_activity).speech_confidence_ratio)}, window ${shortValue(rec(perf().voice_activity).robot_noise_suppression_window_ms)}ms` },
      { label: "Rejects", get: () => `${shortValue(rec(perf().voice_activity).rejected_segment_count)} / ${shortValue(rec(perf().voice_activity).last_reject_reason)}` },
    ],
  },
  {
    title: "STT · LLM · Router · TTS",
    metrics: [
      { label: "STT", get: () => `${formatMs(perf().stt_ms)} / reject ${shortValue(rec(perf().voice_activity).last_stt_reject_reason)}` },
      { label: "LLM", get: () => `${formatMs(perf().llm_first_token_ms)} first / ${formatMs(perf().llm_total_ms)} total` },
      { label: "Router", get: () => `${formatMs(Number(rec(perf().local_model).qwen_router_latency_ms))} / ${shortValue(rec(perf().local_model).qwen_router_status)}` },
      { label: "TTS", get: () => `${shortValue(rec(perf().local_tts).ready)} ${shortValue(rec(perf().local_tts).error)} / ${formatMs(perf().tts_ms)} / first audio ${formatMs(perf().first_audio_ms)}` },
    ],
  },
  {
    title: "Health",
    metrics: [
      {
        label: "Checks",
        get: () => {
          const health = rec(perf().health_checks);
          const model = rec(perf().local_model);
          return `daemon ${shortValue(health.daemon_running)}, media ${shortValue(health.media_available)}, doa ${shortValue(health.doa_status || health.doa_available)}, wired ${shortValue(health.wired_link_present)}, model ${shortValue(model.configured_model)} installed ${shortValue(model.installed)}`;
        },
      },
    ],
  },
];

export function createDiagnosticsView(): View {
  let unsub: (() => void) | null = null;
  const cells: Array<{ dd: HTMLElement; get: () => string }> = [];

  const cards = GROUPS.map((group) => {
    const rows = group.metrics.map((metric) => {
      const dd = el("dd", { text: metric.get() });
      cells.push({ dd, get: metric.get });
      return el("div", { class: "metric-row" }, [el("dt", { text: metric.label }), dd]);
    });
    return el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [el("div", { class: "card-title", text: group.title })]),
      el("dl", { class: "metric-list" }, rows),
    ]);
  });

  function update(): void {
    for (const cell of cells) cell.dd.textContent = cell.get();
  }

  return {
    mount(container) {
      container.append(el("div", { class: "view-scroll" }, [el("div", { class: "diag-groups" }, cards)]));
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
