export type DotState = "ok" | "warn" | "err" | "idle";
export type Tone = "ok" | "warn" | "err" | "muted";
export type LevelFilter = "ALL" | "INFO" | "WARNING" | "ERROR";

export type BackendStatus = {
  backend_unavailable?: boolean;
  backend_unavailable_reason?: string;
};

export type PerformanceStatus = {
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

export type DashboardStatus = BackendStatus & {
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

export type FaceBox = {
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

export type FaceState = {
  ok: boolean;
  available: boolean;
  recognition_available?: boolean;
  focus_name: string | null;
  faces: FaceBox[];
};

export type LogEntry = {
  id?: number;
  type: string;
  createdAt: string;
  level: string;
  category: string;
  message: string;
};

export type ProcessStatus = {
  available: boolean;
  running: boolean;
  pid: number | null;
  command: string;
  defaultCommand: string;
  startedAt: string | null;
  exitedAt: string | null;
  exitCode: number | null;
  signal: string | null;
  backendTarget: string;
  backendReady?: boolean;
  failureHint?: string | null;
};

export type AppPhase = "unavailable" | "idle" | "starting" | "running" | "stopped" | "failed";

export type ProfileSummary = { name: string; is_default: boolean };

export type ProfileList = {
  profiles?: ProfileSummary[];
  choices: string[];
  current: string;
  startup: string;
  locked?: boolean;
  locked_to?: string | null;
};

export type ProfilePayload = {
  name?: string;
  instructions: string;
  tools_text: string;
  voice: string;
  uses_default_voice?: boolean;
  available_tools: string[];
  enabled_tools: string[];
};

export type LaunchOptions = {
  camera: boolean;
  localVision: boolean;
  debug: boolean;
};

export type ViewId = "monitor" | "logs" | "diagnostics" | "settings";

export interface View {
  mount(container: HTMLElement): void;
  update(): void;
  destroy(): void;
  onEnter?(): void;
  onLeave?(): void;
}
