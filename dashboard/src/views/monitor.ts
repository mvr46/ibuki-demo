import { state, subscribe, loadDashboardStatus, loadFaceState } from "../state.ts";
import { api, frameUrl } from "../api.ts";
import { el, setStatus } from "../util.ts";
import type { FaceBox, View } from "../types.ts";

export function createMonitorView(): View {
  let unsub: (() => void) | null = null;
  let frameTimer: number | null = null;

  const cameraFeed = el("img", { id: "camera-feed", alt: "Reachy camera feed" });
  const cameraEmpty = el("div", { class: "camera-empty", text: "Waiting for a camera frame." });
  const faceOverlay = el("div", { class: "face-overlay" });
  const videoShell = el("div", { class: "video-shell" }, [cameraFeed, faceOverlay, cameraEmpty]);
  const cameraHelp = el("p", { id: "camera-help", class: "muted", text: "Camera unavailable until the app API is ready." });

  const selectedFaceLine = el("div", { class: "selected-face", text: "Select a face in the camera feed." });
  const faceNameInput = el("input", { type: "text", placeholder: "Name this face", autocomplete: "off", spellcheck: "false" });
  const saveFaceBtn = el("button", { class: "btn btn-primary btn-block", type: "button", text: "Save face", disabled: true });
  const faceSaveStatus = el("p", { class: "status-line" });
  const refreshBtn = el("button", { class: "link-btn", type: "button", text: "Refresh" });
  const peopleList = el("ol", { class: "people-list" });

  cameraFeed.addEventListener("load", () => { cameraEmpty.hidden = true; });
  cameraFeed.addEventListener("error", () => { cameraEmpty.hidden = false; });
  saveFaceBtn.addEventListener("click", () => void saveSelectedFace());
  refreshBtn.addEventListener("click", () => void loadFaceState());

  function selectedFace(): FaceBox | null {
    if (state.selectedFaceId === null) return null;
    return state.faces.find((face) => face.id === state.selectedFaceId) || null;
  }

  function renderFaces(): void {
    const selected = selectedFace();
    saveFaceBtn.disabled = !selected || selected.id === null || selected.can_remember === false;
    const faceWord = state.faces.length === 1 ? "face" : "faces";
    cameraHelp.textContent = state.faces.length
      ? state.faceRecognitionAvailable
        ? `${state.faces.length} ${faceWord} visible. Click a box to name it.`
        : `${state.faces.length} ${faceWord} visible. Face naming unavailable.`
      : state.faceStateAvailable
        ? "No faces detected."
        : "Camera unavailable until the app API is ready.";
    selectedFaceLine.textContent = selected
      ? `${selected.label} · x ${selected.x_offset.toFixed(2)} · confidence ${selected.confidence.toFixed(2)}`
      : "Select a face in the camera feed.";

    faceOverlay.replaceChildren(
      ...state.faces.map((face) => {
        const label = el("span", { class: "face-label", text: face.name ? `${face.name} ${face.similarity.toFixed(2)}` : "unknown" });
        const box = el("button", { type: "button", class: "face-box" }, [label]);
        if (face.id === state.selectedFaceId) box.classList.add("is-selected");
        if (face.name) box.classList.add("is-known");
        if (face.focused) box.classList.add("is-focused");
        if (face.held) box.classList.add("is-held");
        box.style.left = `${face.bbox.x * 100}%`;
        box.style.top = `${face.bbox.y * 100}%`;
        box.style.width = `${face.bbox.width * 100}%`;
        box.style.height = `${face.bbox.height * 100}%`;
        box.disabled = face.id === null;
        box.addEventListener("click", () => {
          state.selectedFaceId = face.id;
          faceNameInput.value = face.name || "";
          renderFaces();
        });
        return box;
      }),
    );
  }

  function renderPeople(): void {
    const people = state.status?.face_recognition.people || [];
    if (!people.length) {
      peopleList.replaceChildren(el("li", { class: "empty-row", text: "No saved people yet." }));
      return;
    }
    peopleList.replaceChildren(
      ...people.map((person) =>
        el("li", {}, [
          el("strong", { text: person.name }),
          el("span", { text: `${person.exemplar_count} exemplar${person.exemplar_count === 1 ? "" : "s"}` }),
        ]),
      ),
    );
  }

  async function saveSelectedFace(): Promise<void> {
    const selected = selectedFace();
    const name = faceNameInput.value.trim();
    if (!selected || selected.id === null) { setStatus(faceSaveStatus, "Select a visible face first.", "warn"); return; }
    if (!name) { setStatus(faceSaveStatus, "Enter a name.", "warn"); return; }
    saveFaceBtn.disabled = true;
    setStatus(faceSaveStatus, "Saving...");
    try {
      const result = await api.rememberFace(selected.id, name);
      setStatus(faceSaveStatus, `Saved ${result.name} (${result.exemplar_count} exemplar).`, "ok");
      await Promise.all([loadDashboardStatus(), loadFaceState()]);
    } catch (error) {
      setStatus(faceSaveStatus, error instanceof Error ? error.message : "Failed to save.", "error");
    } finally {
      saveFaceBtn.disabled = false;
    }
  }

  function update(): void {
    renderFaces();
    renderPeople();
  }

  return {
    mount(container) {
      const layout = el("div", { class: "monitor-grid" }, [
        el("div", { class: "col-stack" }, [videoShell, cameraHelp]),
        el("div", { class: "col-stack" }, [
          el("div", { class: "card" }, [
            el("div", { class: "card-head" }, [el("div", { class: "card-title", text: "Name a face" }), refreshBtn]),
            selectedFaceLine,
            el("div", { class: "field" }, [faceNameInput]),
            saveFaceBtn,
            faceSaveStatus,
          ]),
          el("div", { class: "card" }, [
            el("div", { class: "card-head" }, [el("div", { class: "card-title", text: "Saved people" })]),
            peopleList,
          ]),
        ]),
      ]);
      container.append(el("div", { class: "view-scroll" }, [layout]));
      unsub = subscribe(update);
      update();
    },
    onEnter() {
      cameraFeed.src = frameUrl();
      frameTimer = window.setInterval(() => { cameraFeed.src = frameUrl(); }, 350);
    },
    onLeave() {
      if (frameTimer !== null) { window.clearInterval(frameTimer); frameTimer = null; }
    },
    update,
    destroy() {
      unsub?.();
      unsub = null;
    },
  };
}
