import { api } from "../api.ts";
import { el, setStatus } from "../util.ts";
import { openModal } from "./modal.ts";
import type { ProfileList } from "../types.ts";

const AUTO_WITH: Record<string, string[]> = {
  dance: ["stop_dance"],
  play_emotion: ["stop_emotion"],
};

function labeled(labelText: string, control: HTMLElement): HTMLElement {
  return el("div", {}, [el("div", { class: "field-label", text: labelText }), control]);
}

export function openProfileModal(opts: {
  list: ProfileList;
  initialName?: string;
  onChanged?: (list: ProfileList) => void;
}): void {
  let list = opts.list;
  const locked = !!list.locked;

  const nameInput = el("input", { type: "text", value: opts.initialName || "", placeholder: "profile-name", autocomplete: "off", spellcheck: "false" });
  const instructions = el("textarea", { rows: "8", placeholder: "System instructions for this profile" });
  const toolsText = el("textarea", { rows: "4", placeholder: "# tools enabled for this profile" });
  const toolsGrid = el("div", { class: "checkbox-grid" });
  const voiceSelect = el("select");
  const applyVoiceBtn = el("button", { class: "btn btn-sm", type: "button", text: "Apply voice" });
  const status = el("p", { class: "status-line" });

  function syncToolsFromCheckboxes(): void {
    const checks = Array.from(toolsGrid.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'));
    if (!checks.length) return;
    const selected = new Set<string>();
    checks.forEach((input) => { if (input.checked) selected.add(input.value); });
    for (const [tool, deps] of Object.entries(AUTO_WITH)) {
      if (selected.has(tool)) deps.forEach((dep) => selected.add(dep));
    }
    checks.forEach((input) => { input.checked = selected.has(input.value); });
    const comments = toolsText.value.split("\n").filter((line) => line.trim().startsWith("#"));
    toolsText.value = `${comments.length ? `${comments.join("\n")}\n` : ""}${[...selected].sort().join("\n")}\n`;
  }

  function renderToolCheckboxes(available: string[], enabled: string[]): void {
    const enabledSet = new Set(enabled);
    toolsGrid.replaceChildren(
      ...available.map((tool) => {
        const checkbox = el("input", { type: "checkbox", value: tool, checked: enabledSet.has(tool) });
        checkbox.addEventListener("change", syncToolsFromCheckboxes);
        return el("label", { class: "tool-check" }, [checkbox, el("span", { text: tool })]);
      }),
    );
    if (!available.length) toolsGrid.append(el("p", { class: "muted", text: "Tool list loads with the profile." }));
  }

  async function loadVoices(preferred: string): Promise<void> {
    let voices: string[] = [];
    try { voices = await api.voices(); } catch { voices = []; }
    if (!voices.length) voices = [preferred].filter(Boolean);
    voiceSelect.replaceChildren(...voices.map((voice) => el("option", { value: voice, text: voice })));
    if (voices.includes(preferred)) voiceSelect.value = preferred;
  }

  async function loadProfile(name: string): Promise<void> {
    try {
      const data = await api.profileLoad(name);
      nameInput.value = data.name || name;
      instructions.value = data.instructions || "";
      toolsText.value = data.tools_text || "";
      renderToolCheckboxes(data.available_tools || [], data.enabled_tools || []);
      await loadVoices(data.voice || "");
      setStatus(status, `Loaded ${name}.`);
    } catch (error) {
      setStatus(status, error instanceof Error ? error.message : "Failed to load profile.", "error");
    }
  }

  function startNew(): void {
    instructions.value = "# Instructions for this profile\n";
    toolsText.value = "# tools enabled for this profile\n";
    renderToolCheckboxes([], []);
    void loadVoices("");
    setStatus(status, "Fill in the fields and save.");
  }

  async function save(forceNew: boolean): Promise<void> {
    const name = nameInput.value.trim();
    if (!name) { setStatus(status, "Enter a profile name.", "warn"); return; }
    syncToolsFromCheckboxes();
    const overwrite = forceNew ? false : list.choices.includes(name);
    setStatus(status, overwrite ? "Saving..." : "Creating...");
    try {
      const result = await api.profileSave({
        name,
        instructions: instructions.value,
        tools_text: toolsText.value,
        voice: voiceSelect.value,
        overwrite,
      });
      list = result;
      setStatus(status, `Saved ${result.profile}. Restart to apply tool changes.`, "ok");
      opts.onChanged?.(list);
    } catch (error) {
      setStatus(status, error instanceof Error ? error.message : "Failed to save.", "error");
    }
  }

  async function apply(persist: boolean): Promise<void> {
    const name = nameInput.value.trim();
    if (!name) { setStatus(status, "Enter a profile name.", "warn"); return; }
    setStatus(status, persist ? "Setting startup profile..." : "Applying...");
    try {
      const result = await api.profileApply(name, persist);
      setStatus(status, result.status || (persist ? "Set as startup profile." : "Applied."), "ok");
    } catch (error) {
      setStatus(status, error instanceof Error ? error.message : "Failed to apply.", "error");
    }
  }

  async function applyVoice(): Promise<void> {
    const voice = voiceSelect.value;
    if (!voice) { setStatus(status, "Select a voice first.", "warn"); return; }
    setStatus(status, "Applying voice...");
    try {
      const result = await api.voiceApply(voice);
      setStatus(status, result.status || `Voice set to ${voice}.`, "ok");
    } catch (error) {
      setStatus(status, error instanceof Error ? error.message : "Failed to apply voice.", "error");
    }
  }

  applyVoiceBtn.addEventListener("click", () => void applyVoice());

  const bodyChildren: Array<HTMLElement | null> = [
    locked
      ? el("div", { class: "notice warn", text: `Locked to “${list.locked_to || "a profile"}”. Applying and startup changes are disabled on this deployment.` })
      : null,
    labeled("Name", nameInput),
    labeled("Instructions", instructions),
    el("div", {}, [
      el("div", { class: "field-label", text: "Tools" }),
      toolsGrid,
      el("details", { class: "disclosure" }, [
        el("summary", { text: "Advanced: tools.txt" }),
        el("p", { class: "muted", style: "margin-bottom:8px;", text: "Edited automatically from the checkboxes; comment lines are preserved." }),
        toolsText,
      ]),
    ]),
    el("div", { class: "voice-row" }, [labeled("Voice", voiceSelect), applyVoiceBtn]),
  ];
  const body = el("div", { class: "col-stack", style: "gap:14px;" }, bodyChildren);

  const saveBtn = el("button", { class: "btn btn-primary btn-sm", type: "button", text: "Save" });
  const saveNewBtn = el("button", { class: "btn btn-sm", type: "button", text: "Save as new" });
  const applyBtn = el("button", { class: "btn btn-sm", type: "button", text: "Apply now" });
  const startupBtn = el("button", { class: "btn btn-sm", type: "button", text: "Use on start" });
  saveBtn.addEventListener("click", () => void save(false));
  saveNewBtn.addEventListener("click", () => void save(true));
  applyBtn.addEventListener("click", () => void apply(false));
  startupBtn.addEventListener("click", () => void apply(true));
  if (locked) {
    applyBtn.disabled = true;
    startupBtn.disabled = true;
  }

  const footer = el("div", { class: "modal-foot" }, [
    status,
    el("span", { class: "spacer" }),
    startupBtn,
    applyBtn,
    saveNewBtn,
    saveBtn,
  ]);

  openModal({ title: opts.initialName ? `Edit profile · ${opts.initialName}` : "New profile", body, footer });

  if (opts.initialName) void loadProfile(opts.initialName);
  else startNew();
}
