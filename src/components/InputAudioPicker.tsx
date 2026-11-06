/**
 * Reference audio picker for the Music Studio. Same pattern as
 * `InputImagePicker`: a file dialog (desktop app) or the music library.
 * The backend takes absolute paths on this machine, so a pick is a path,
 * never an upload. Reuses the `.input-image-*` layout classes.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { fileNameOf } from "./InputImagePicker";

export interface InputAudio {
  path: string;
  name: string;
}

/** One library entry the picker can offer; the parent maps its own summaries to this. */
export interface AudioLibraryChoice {
  id: string;
  title: string;
  path: string;
}

export interface InputAudioPickerProps {
  label: string;
  hint?: string;
  audio: InputAudio[];
  max: number;
  onChange: (next: InputAudio[]) => void;
  library: AudioLibraryChoice[];
  disabled?: boolean;
}

const LIBRARY_LIMIT = 12;
export const AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "m4a"];

export function addInputAudio(audio: InputAudio[], next: InputAudio, max: number): InputAudio[] {
  if (audio.some((item) => item.path === next.path)) return audio;
  return [...audio, next].slice(-max);
}

export function InputAudioPicker({ label, hint, audio, max, onChange, library, disabled }: InputAudioPickerProps) {
  const { t } = useTranslation("studio");
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [pickError, setPickError] = useState<string | null>(null);
  const full = audio.length >= max;
  const choices = library.filter((item) => item.path).slice(0, LIBRARY_LIMIT);

  async function chooseFile() {
    setPickError(null);
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false,
        title: label,
        filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
      });
      if (typeof selected === "string" && selected) {
        onChange(addInputAudio(audio, { path: selected, name: fileNameOf(selected) }, max));
      }
    } catch {
      setPickError(t("inputAudio.pickFailed", { defaultValue: "The file picker is not available here. Pick from the library instead." }));
    }
  }

  return (
    <div className="control-stack input-image-picker input-audio-picker">
      <span className="eyebrow">{label}</span>
      {hint ? <p className="muted-text">{hint}</p> : null}
      {audio.length > 0 ? (
        <ul className="input-image-list">
          {audio.map((item) => (
            <li key={item.path} className="input-image-item">
              <span className="input-image-name" title={item.path}>{item.name}</span>
              <button
                type="button"
                className="secondary-button"
                disabled={disabled}
                onClick={() => onChange(audio.filter((other) => other.path !== item.path))}
              >
                {t("inputAudio.remove", { defaultValue: "Remove" })}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <div className="image-pill-row">
        <button type="button" className="secondary-button" disabled={disabled || full} onClick={() => void chooseFile()}>
          {t("inputAudio.chooseFile", { defaultValue: "Choose file" })}
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={disabled || full || choices.length === 0}
          aria-expanded={libraryOpen}
          onClick={() => setLibraryOpen((open) => !open)}
        >
          {t("inputAudio.fromLibrary", { defaultValue: "From library" })}
        </button>
      </div>
      {libraryOpen && !full ? (
        <ul className="input-image-list input-audio-library">
          {choices.map((item) => (
            <li key={item.id} className="input-image-item">
              <button
                type="button"
                className="secondary-button input-audio-choice"
                title={item.path}
                onClick={() => {
                  onChange(addInputAudio(audio, { path: item.path, name: item.title || fileNameOf(item.path) }, max));
                  setLibraryOpen(false);
                }}
              >
                {item.title}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      {pickError ? <p className="error-text">{pickError}</p> : null}
    </div>
  );
}
