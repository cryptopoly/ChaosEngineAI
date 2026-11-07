/**
 * Adding sound to a lane: choose a file, pick from the library, or type a path. The backend copies the
 * file into the project and places the clip at the playhead. Voice clips can be marked as a cloned voice,
 * which locks the AI tag on at export (D-34). The Music Library does not exist yet, so only the Voice lane
 * has library items.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getLibrary, getLibraryItem } from "../../../api";
import { InputAudioPicker, type AudioLibraryChoice, type InputAudio } from "../../../components/InputAudioPicker";
import type { AudioLaneId } from "./timelineGeometry";
import { resolveLibraryPath, voiceLibraryChoices } from "./mixerLogic";

export interface AddAudioPanelProps {
  lane: AudioLaneId;
  laneLabel: string;
  disabled: boolean;
  onAdd: (picked: { path: string; name?: string }, clonedVoice: boolean) => void;
  onClose: () => void;
}

const NO_LIBRARY: AudioLibraryChoice[] = [];

export function AddAudioPanel({ lane, laneLabel, disabled, onAdd, onClose }: AddAudioPanelProps) {
  const { t } = useTranslation("studio");
  const [library, setLibrary] = useState<AudioLibraryChoice[]>(NO_LIBRARY);
  const [typed, setTyped] = useState("");
  const [cloned, setCloned] = useState(false);

  useEffect(() => {
    if (lane !== "voice") {
      setLibrary(NO_LIBRARY);
      return;
    }
    let cancelled = false;
    getLibrary("", "speech")
      .then((items) => {
        if (!cancelled) setLibrary(voiceLibraryChoices(items));
      })
      .catch(() => {
        if (!cancelled) setLibrary(NO_LIBRARY);
      });
    return () => {
      cancelled = true;
    };
  }, [lane]);

  async function picked(next: InputAudio[]) {
    const last = next[next.length - 1];
    if (!last) return;
    // A Voice Library pick is added by the file's real location, which moves with the data folder.
    const choice = library.find((item) => item.path === last.path);
    let path = last.path;
    if (choice) {
      try {
        path = resolveLibraryPath((await getLibraryItem(choice.id)).audioPath, last.path);
      } catch {
        // keep the default folder path
      }
    }
    onAdd({ path, name: last.name }, cloned);
  }

  function submitTyped() {
    const path = typed.trim();
    if (!path) return;
    onAdd({ path }, cloned);
    setTyped("");
  }

  return (
    <section className="add-audio" aria-label={t("mixer.addAudioTo", { defaultValue: "Add audio to {lane}", lane: laneLabel })}>
      <div className="add-audio__head">
        <h4>{t("mixer.addAudioTo", { defaultValue: "Add audio to {lane}", lane: laneLabel })}</h4>
        <button type="button" className="secondary-button" onClick={onClose}>
          {t("storyboard.close", { defaultValue: "Close" })}
        </button>
      </div>
      <InputAudioPicker
        label={t("mixer.addFromLibrary", { defaultValue: "Add from library" })}
        hint={lane === "voice" ? undefined : t("mixer.noLibraryYet", { defaultValue: "The Music Library is not available yet. Choose a file or type its path." })}
        audio={[]}
        max={1}
        onChange={(next) => void picked(next)}
        library={library}
        disabled={disabled}
      />
      <div className="path-field">
        <span className="eyebrow">{t("mixer.importAudio", { defaultValue: "Import audio" })}</span>
        <div className="path-field__row">
          <input
            type="text"
            value={typed}
            placeholder={t("storyboard.pathPlaceholder", { defaultValue: "/path/to/file" })}
            aria-label={t("mixer.importAudioPath", { defaultValue: "Path of the audio file" })}
            disabled={disabled}
            onChange={(event) => setTyped(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                submitTyped();
              }
            }}
          />
          <button type="button" className="primary-button" disabled={disabled || !typed.trim()} onClick={submitTyped}>
            {t("mixer.addButton", { defaultValue: "Add" })}
          </button>
        </div>
      </div>
      {lane === "voice" ? (
        <label className="check-row">
          <input type="checkbox" checked={cloned} onChange={(event) => setCloned(event.target.checked)} />
          {t("mixer.add.cloned", { defaultValue: "This clip uses a cloned voice" })}
        </label>
      ) : null}
    </section>
  );
}
