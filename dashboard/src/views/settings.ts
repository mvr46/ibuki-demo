import { state, subscribe, loadDashboardStatus } from "../state.ts";
import { api } from "../api.ts";
import { el, setStatus } from "../util.ts";
import { renderLaunchForm } from "../components/launchConfig.ts";
import { openProfileModal } from "../components/profileModal.ts";
import type { ProfileList, View } from "../types.ts";

const HF_BACKEND = "huggingface";
const LOCAL_BACKEND = "local";

function activeBackend(): string {
  return state.status?.backend_provider || LOCAL_BACKEND;
}

function sectionCard(title: string, desc: string, children: Array<HTMLElement | null>, action?: HTMLElement | null): HTMLElement {
  const head = el("div", { class: "card-head" }, [
    el("div", {}, [el("div", { class: "card-title", text: title }), el("p", { class: "section-desc", text: desc })]),
    action ?? null,
  ]);
  return el("div", { class: "card settings-section" }, [head, ...(children.filter(Boolean) as HTMLElement[])]);
}

export function createSettingsView(): View {
  let unsub: (() => void) | null = null;
  let profiles: ProfileList | null = null;

  // ---- Backend ----
  const localRadio = el("input", { type: "radio", name: "backend", value: LOCAL_BACKEND });
  const hfRadio = el("input", { type: "radio", name: "backend", value: HF_BACKEND });
  const localChoice = el("label", { class: "choice" }, [localRadio, el("span", { text: "Local Mac" })]);
  const hfChoice = el("label", { class: "choice" }, [hfRadio, el("span", { text: "Hugging Face" })]);
  const choiceGrid = el("div", { class: "choice-grid" }, [localChoice, hfChoice]);

  const hfMode = el("select", {}, [
    el("option", { value: "deployed", text: "Built-in server" }),
    el("option", { value: "local", text: "Local websocket" }),
  ]);
  const hfHost = el("input", { type: "text", placeholder: "localhost" });
  const hfPort = el("input", { type: "number", min: "1", max: "65535", value: "8765" });
  const hfFields = el("div", { class: "hf-grid" }, [hfMode, hfHost, hfPort]);
  const saveBackendBtn = el("button", { class: "btn btn-block", type: "button", text: "Save backend" });
  const backendStatus = el("p", { class: "status-line" });

  function syncBackendUi(): void {
    const usingHf = hfRadio.checked;
    localChoice.classList.toggle("is-selected", localRadio.checked);
    hfChoice.classList.toggle("is-selected", usingHf);
    hfFields.hidden = !usingHf;
  }
  function fillBackendFromStatus(force: boolean): void {
    if (force) {
      const backend = activeBackend();
      localRadio.checked = backend === LOCAL_BACKEND;
      hfRadio.checked = backend === HF_BACKEND;
    }
    if (document.activeElement !== hfMode) hfMode.value = state.status?.hf_connection_mode || "deployed";
    if (document.activeElement !== hfHost) hfHost.value = state.status?.hf_direct_host || "localhost";
    if (document.activeElement !== hfPort) hfPort.value = String(state.status?.hf_direct_port || 8765);
    syncBackendUi();
  }
  localRadio.addEventListener("change", syncBackendUi);
  hfRadio.addEventListener("change", syncBackendUi);
  saveBackendBtn.addEventListener("click", () => void saveBackend());

  async function saveBackend(): Promise<void> {
    const backend = hfRadio.checked ? HF_BACKEND : LOCAL_BACKEND;
    const body: Record<string, string | number> = { backend };
    if (backend === HF_BACKEND) {
      body.hf_mode = hfMode.value;
      body.hf_host = hfHost.value.trim();
      body.hf_port = Number.parseInt(hfPort.value || "8765", 10);
    }
    setStatus(backendStatus, "Saving...");
    try {
      await api.backendConfig(body);
      setStatus(backendStatus, "Saved. Refreshing status...", "ok");
      await loadDashboardStatus();
      fillBackendFromStatus(true);
    } catch (error) {
      setStatus(backendStatus, error instanceof Error ? error.message : "Failed to save.", "error");
    }
  }

  // ---- Profiles ----
  const lockNotice = el("div", { class: "notice warn", hidden: true });
  const profileList = el("ul", { class: "profile-list" });
  const newProfileBtn = el("button", { class: "btn btn-sm", type: "button", text: "New profile" });
  const profileStatus = el("p", { class: "status-line" });

  function openModalFor(initialName?: string): void {
    if (!profiles) return;
    openProfileModal({
      list: profiles,
      initialName,
      onChanged: (updated) => {
        profiles = updated;
        renderProfiles();
      },
    });
  }
  newProfileBtn.addEventListener("click", () => openModalFor());

  function renderProfiles(): void {
    if (!profiles) return;
    const { choices, current, startup, locked, locked_to } = profiles;
    if (locked) {
      lockNotice.hidden = false;
      lockNotice.textContent = `Locked to “${locked_to || "a profile"}”. Applying and startup changes are disabled on this deployment.`;
    } else {
      lockNotice.hidden = true;
    }
    if (!choices.length) {
      profileList.replaceChildren(el("li", { class: "empty-row", text: "No profiles yet." }));
      return;
    }
    profileList.replaceChildren(
      ...choices.map((name) => {
        const tags: HTMLElement[] = [];
        if (name === current) tags.push(el("span", { class: "tag", text: "current" }));
        if (name === startup) tags.push(el("span", { class: "tag", text: "startup" }));
        const editBtn = el("button", { class: "btn btn-sm", type: "button", text: "Edit" });
        editBtn.addEventListener("click", () => openModalFor(name));
        return el("li", { class: "profile-row" }, [
          el("span", { class: "profile-name", text: name }),
          ...tags,
          el("span", { class: "spacer" }),
          editBtn,
        ]);
      }),
    );
  }

  async function loadProfiles(): Promise<void> {
    try {
      profiles = await api.profiles();
    } catch {
      newProfileBtn.disabled = true;
      profileList.replaceChildren(el("li", { class: "empty-row", text: "Profile controls become available after the app starts." }));
      setStatus(profileStatus, "");
      return;
    }
    newProfileBtn.disabled = false;
    renderProfiles();
  }

  function update(): void {
    fillBackendFromStatus(false);
  }

  return {
    mount(container) {
      const launchForm = renderLaunchForm({ variant: "full" });
      const launchSection = sectionCard(
        "Launch",
        "Configure the conversation app flags. The Run button in the header uses these settings.",
        [launchForm],
      );
      const backendSection = sectionCard(
        "Backend",
        "Choose where inference runs.",
        [choiceGrid, hfFields, saveBackendBtn, backendStatus],
      );
      const profilesSection = sectionCard(
        "Profiles",
        "Manage conversation personalities, tools, and voice.",
        [lockNotice, profileList, profileStatus],
        newProfileBtn,
      );

      container.append(el("div", { class: "view-scroll narrow" }, [
        el("div", { class: "col-stack" }, [launchSection, backendSection, profilesSection]),
      ]));

      fillBackendFromStatus(true);
      void loadProfiles();
      unsub = subscribe(update);
    },
    update,
    destroy() {
      unsub?.();
      unsub = null;
    },
  };
}
