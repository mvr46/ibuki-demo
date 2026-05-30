import { state, subscribe, appPhase } from "./state.ts";
import { el, setDot, shortValue, formatMs } from "./util.ts";
import { createStartControl } from "./components/runButton.ts";
import type { DotState, ViewId } from "./types.ts";

const ICONS: Record<ViewId, string> = {
  monitor:
    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
  logs:
    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>',
  diagnostics:
    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
  settings:
    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round"><path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3M14 2v4M8 10v4M16 18v4"/></svg>',
};

const TITLES: Record<ViewId, string> = {
  monitor: "Monitor",
  logs: "Logs",
  diagnostics: "Diagnostics",
  settings: "Settings",
};

const NAV: ViewId[] = ["monitor", "logs", "diagnostics", "settings"];

export interface ShellHandle {
  viewContainer: HTMLElement;
  setActive(id: ViewId): void;
}

function pill(key: string): { root: HTMLElement; dot: HTMLElement; val: HTMLElement } {
  const dot = el("span", { class: "status-dot" });
  const val = el("span", { class: "status-val", text: "checking" });
  const root = el("span", { class: "status-item" }, [dot, el("span", { class: "status-key", text: key }), val]);
  return { root, dot, val };
}

export function createShell(root: HTMLElement): ShellHandle {
  const navLinks = new Map<ViewId, HTMLAnchorElement>();
  const nav = el("nav", { class: "sidebar-nav", "aria-label": "Views" });
  for (const id of NAV) {
    const link = el("a", { class: "nav-item", href: `#/${id}` }, [
      el("span", { class: "nav-icon", html: ICONS[id] }),
      el("span", { text: TITLES[id] }),
    ]);
    navLinks.set(id, link);
    nav.append(link);
  }

  const footerDot = el("span", { class: "status-dot" });
  const footerText = el("span", { text: "Connecting..." });
  const sidebar = el("aside", { class: "sidebar" }, [
    el("div", { class: "sidebar-brand" }, [el("span", { class: "brand-mark" }), el("span", { text: "Reachy Mini" })]),
    nav,
    el("div", { class: "sidebar-footer" }, [footerDot, footerText]),
  ]);

  const viewTitle = el("div", { class: "view-title", text: TITLES.monitor });
  const pills = {
    camera: pill("Camera"),
    faces: pill("Faces"),
    link: pill("Link"),
  };
  const headerStatus = el("div", { class: "header-status" }, [
    pills.camera.root,
    pills.faces.root,
    pills.link.root,
  ]);

  const headerActions = el("div", { class: "header-actions" }, [createStartControl()]);
  const header = el("header", { class: "content-header" }, [viewTitle, headerStatus, headerActions]);

  const viewContainer = el("div", { class: "view" });
  root.append(sidebar, el("div", { class: "content" }, [header, viewContainer]));

  function renderPills(): void {
    const status = state.status;
    const transport = (status?.performance?.transport || {}) as Record<string, unknown>;
    const mediaHost = shortValue(transport.media_host);
    const mediaSource = shortValue(transport.media_host_source);
    const rttMs = status?.performance?.daemon_rtt_ms;
    const cameraOk = !!status?.camera.available && !!status.camera.frame_available;
    const faceOk = !!status?.face_recognition.available;
    const linkUp = mediaHost !== "-";
    const linkWarn = linkUp && (mediaSource === "daemon_wlan_ip" || (rttMs !== null && rttMs !== undefined && rttMs > 15));

    pills.camera.val.textContent = !status
      ? "checking"
      : cameraOk
        ? status.camera.head_tracker || "streaming"
        : status.camera.available
          ? "no frame"
          : "off";
    pills.faces.val.textContent = !status ? "checking" : faceOk ? `${status.face_recognition.visible_count} visible` : "off";
    pills.link.val.textContent = !status || !linkUp ? "—" : `${mediaHost}${rttMs ? ` ${formatMs(rttMs)}` : ""}`;

    setDot(pills.camera.dot, cameraOk ? "ok" : status?.camera.available ? "warn" : "idle");
    setDot(pills.faces.dot, faceOk ? "ok" : "idle");
    setDot(pills.link.dot, !status || !linkUp ? "idle" : linkWarn ? "warn" : "ok");

    const phase = appPhase();
    const footerTone: DotState = phase === "running" ? "ok" : phase === "starting" ? "warn" : phase === "failed" ? "err" : "idle";
    setDot(footerDot, footerTone);
    footerText.textContent =
      phase === "running"
        ? "local · ready"
        : phase === "starting"
          ? "local · starting"
          : phase === "failed"
            ? "local · error"
            : "local · idle";
  }

  subscribe(renderPills);
  renderPills();

  return {
    viewContainer,
    setActive(id: ViewId): void {
      viewTitle.textContent = TITLES[id];
      for (const [key, link] of navLinks) {
        if (key === id) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
      }
    },
  };
}
