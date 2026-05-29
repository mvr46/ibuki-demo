import { state } from "./state.ts";
import type { View, ViewId } from "./types.ts";

const ROUTES: ViewId[] = ["monitor", "logs", "diagnostics", "settings"];
const DEFAULT_VIEW: ViewId = "monitor";

export interface Router {
  start(): void;
}

export function createRouter(opts: {
  container: HTMLElement;
  factories: Record<ViewId, () => View>;
  onNavigate: (id: ViewId) => void;
}): Router {
  let current: View | null = null;

  function parseHash(): ViewId {
    const raw = window.location.hash.replace(/^#\/?/, "").trim();
    return (ROUTES as string[]).includes(raw) ? (raw as ViewId) : DEFAULT_VIEW;
  }

  function navigate(id: ViewId): void {
    if (current) {
      current.onLeave?.();
      current.destroy();
      current = null;
    }
    opts.container.replaceChildren();
    state.activeView = id;
    const view = opts.factories[id]();
    view.mount(opts.container);
    view.onEnter?.();
    current = view;
    opts.onNavigate(id);
  }

  return {
    start(): void {
      window.addEventListener("hashchange", () => navigate(parseHash()));
      const initial = parseHash();
      const canonical = `#/${initial}`;
      if (window.location.hash === canonical) navigate(initial);
      else window.location.hash = canonical;
    },
  };
}
