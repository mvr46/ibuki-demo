import { el } from "../util.ts";

export interface ModalHandle {
  root: HTMLElement;
  body: HTMLElement;
  footer: HTMLElement;
  close(): void;
}

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function openModal(opts: {
  title: string;
  body: HTMLElement;
  footer?: HTMLElement;
  onClose?: () => void;
}): ModalHandle {
  const previousFocus = document.activeElement as HTMLElement | null;
  const closeBtn = el("button", { class: "modal-close", type: "button", "aria-label": "Close", text: "×" });
  const footer = opts.footer ?? el("div", { class: "modal-foot" });
  const modal = el("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": opts.title }, [
    el("div", { class: "modal-head" }, [
      el("div", { class: "modal-title", text: opts.title }),
      closeBtn,
    ]),
    el("div", { class: "modal-body" }, [opts.body]),
    footer,
  ]);
  const backdrop = el("div", { class: "modal-backdrop" }, [modal]);

  function onKey(event: KeyboardEvent): void {
    if (event.key === "Escape") {
      event.stopPropagation();
      close();
    } else if (event.key === "Tab") {
      const focusable = Array.from(modal.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((node) => node.offsetParent !== null);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  }

  function close(): void {
    backdrop.remove();
    document.removeEventListener("keydown", onKey, true);
    document.documentElement.style.overflow = "";
    opts.onClose?.();
    previousFocus?.focus?.();
  }

  backdrop.addEventListener("mousedown", (event) => {
    if (event.target === backdrop) close();
  });
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", onKey, true);
  document.documentElement.style.overflow = "hidden";
  document.body.append(backdrop);

  const firstField = modal.querySelector<HTMLElement>(FOCUSABLE);
  firstField?.focus();

  return { root: modal, body: modal.querySelector(".modal-body") as HTMLElement, footer, close };
}
