import type { DotState } from "./types.ts";

export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing element #${id}`);
  return element as T;
}

type ElAttrs = Record<string, string | number | boolean | null | undefined>;
type ElChild = Node | string | null | undefined | false;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: ElAttrs = {},
  children: ElChild[] | ElChild = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = String(value);
    else if (key === "text") node.textContent = String(value);
    else if (key === "html") node.innerHTML = String(value);
    else if (key in node && key !== "list") (node as Record<string, unknown>)[key] = value;
    else node.setAttribute(key, String(value));
  }
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function setDot(target: HTMLElement, dotState: DotState): void {
  target.classList.remove("dot--ok", "dot--warn", "dot--err");
  if (dotState !== "idle") target.classList.add(`dot--${dotState}`);
}

export function setStatus(target: HTMLElement, text: string, tone = ""): void {
  target.textContent = text;
  target.className = tone ? `status-line ${tone}` : "status-line";
}

export function shortValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "-";
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${value.toFixed(value < 10 ? 1 : 0)}ms`;
}

export function formatFps(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return `${value.toFixed(1)} fps`;
}
