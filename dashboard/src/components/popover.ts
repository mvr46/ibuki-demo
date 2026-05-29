export interface PopoverHandle {
  isOpen(): boolean;
  open(): void;
  close(): void;
  toggle(): void;
}

export function createPopover(opts: {
  anchor: HTMLElement;
  build: (close: () => void) => HTMLElement;
  align?: "left" | "right";
  onOpen?: () => void;
}): PopoverHandle {
  let panel: HTMLElement | null = null;
  const align = opts.align ?? "right";

  function position(): void {
    if (!panel) return;
    const rect = opts.anchor.getBoundingClientRect();
    const top = rect.bottom + 8;
    panel.style.top = `${top}px`;
    panel.style.maxHeight = `${Math.max(160, window.innerHeight - top - 16)}px`;
    panel.style.overflowY = "auto";
    if (align === "right") {
      panel.style.right = `${Math.max(12, window.innerWidth - rect.right)}px`;
      panel.style.left = "auto";
    } else {
      panel.style.left = `${Math.max(12, rect.left)}px`;
      panel.style.right = "auto";
    }
  }

  function onDocPointer(event: MouseEvent): void {
    const target = event.target as Node;
    if (panel?.contains(target) || opts.anchor.contains(target)) return;
    close();
  }

  function onKey(event: KeyboardEvent): void {
    if (event.key === "Escape") close();
  }

  function open(): void {
    if (panel) return;
    panel = document.createElement("div");
    panel.className = "popover";
    panel.append(opts.build(close));
    document.body.append(panel);
    position();
    opts.anchor.setAttribute("aria-expanded", "true");
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    document.addEventListener("mousedown", onDocPointer);
    document.addEventListener("keydown", onKey);
    opts.onOpen?.();
  }

  function close(): void {
    if (!panel) return;
    panel.remove();
    panel = null;
    opts.anchor.setAttribute("aria-expanded", "false");
    window.removeEventListener("resize", position);
    window.removeEventListener("scroll", position, true);
    document.removeEventListener("mousedown", onDocPointer);
    document.removeEventListener("keydown", onKey);
  }

  return {
    isOpen: () => panel !== null,
    open,
    close,
    toggle: () => (panel ? close() : open()),
  };
}
