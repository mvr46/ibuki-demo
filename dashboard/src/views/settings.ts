import { subscribe } from "../state.ts";
import { api } from "../api.ts";
import { el, setStatus } from "../util.ts";
import { renderAppOptions } from "../components/launchConfig.ts";
import { openProfileModal } from "../components/profileModal.ts";
import type { ProfileList, View } from "../types.ts";

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

  return {
    mount(container) {
      const appOptionsSection = sectionCard(
        "App options",
        "Toggles applied when you start the robot app from the header.",
        [renderAppOptions()],
      );
      const profilesSection = sectionCard(
        "Profiles",
        "Manage conversation personalities, tools, and voice.",
        [lockNotice, profileList, profileStatus],
        newProfileBtn,
      );

      container.append(el("div", { class: "view-scroll narrow" }, [
        el("div", { class: "col-stack" }, [appOptionsSection, profilesSection]),
      ]));

      void loadProfiles();
      unsub = subscribe(renderProfiles);
    },
    update: renderProfiles,
    destroy() {
      unsub?.();
      unsub = null;
    },
  };
}
