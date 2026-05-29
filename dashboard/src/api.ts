import type {
  BackendStatus,
  DashboardStatus,
  FaceState,
  ProcessStatus,
  ProfileList,
  ProfilePayload,
} from "./types.ts";

export async function fetchJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(String((data as { error?: string }).error || response.statusText || "request_failed"));
  }
  return data as T;
}

function jsonBody(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export const api = {
  dashboardStatus: () => fetchJson<DashboardStatus>("/api/dashboard/status"),
  faceState: () => fetchJson<FaceState>("/api/face/state"),
  rememberFace: (faceId: number, name: string) =>
    fetchJson<{ name: string; exemplar_count: number }>("/api/face/remember", jsonBody({ face_id: faceId, name })),

  processStatus: () => fetchJson<ProcessStatus>("/__dashboard/process/status"),
  processStart: (command: string) =>
    fetchJson<ProcessStatus>("/__dashboard/process/start", jsonBody({ command })),
  processStop: () => fetchJson<ProcessStatus>("/__dashboard/process/stop", { method: "POST" }),

  backendConfig: (body: Record<string, string | number>) =>
    fetchJson<BackendStatus>("/backend_config", jsonBody(body)),

  profiles: () => fetchJson<ProfileList>("/profiles"),
  profileLoad: (name: string) => {
    const url = new URL("/profiles/load", window.location.origin);
    url.searchParams.set("name", name);
    return fetchJson<ProfilePayload>(url.toString());
  },
  profileSave: (body: { name: string; instructions: string; tools_text: string; voice: string; overwrite: boolean }) =>
    fetchJson<ProfileList & { profile: string }>("/profiles/save", jsonBody(body)),
  profileApply: (name: string, persist: boolean) => {
    const url = new URL("/profiles/apply", window.location.origin);
    url.searchParams.set("name", name);
    if (persist) url.searchParams.set("persist", "1");
    return fetchJson<{ status?: string }>(url.toString(), { method: "POST" });
  },

  voices: () => fetchJson<string[]>("/voices"),
  voiceApply: (voice: string) => {
    const url = new URL("/voices/apply", window.location.origin);
    url.searchParams.set("voice", voice);
    return fetchJson<{ status?: string }>(url.toString(), { method: "POST" });
  },
};

export function frameUrl(): string {
  return `/api/face/frame.jpg?_=${Date.now()}`;
}
