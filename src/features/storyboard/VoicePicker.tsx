/**
 * The project voice (D-65): the real voice list from `GET /api/voice/voices` in four groups with a search,
 * a Play sample button per voice (spoken through the speak stream, the same as the Voice Models preview), and
 * a plain note when a voice needs a model first. "No voice (automatic)" is the first choice. Replaces the
 * blank text box.
 */
import { useEffect, useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { getVoices } from "../../api";
import { PlayIcon, StopIcon } from "../../components/icons/ActionIcons";
import type { VoiceEntry, VoiceEntryKind } from "../../types";
import { speechPlayer, useSpeechPlayer } from "../voice/speechPlayer";
import { canPlaySample, groupVoices, voiceLabel } from "./voicePickerLogic";

export interface VoicePickerProps {
  value: string | null;
  disabled?: boolean;
  onChange: (voice: string | null) => void;
}

let cached: VoiceEntry[] | null = null;

/** The name of a stored voice id when the voice list has been read this session, else the id itself. */
export function voiceDisplayName(id: string): string {
  return cached?.find((voice) => voice.id === id)?.name ?? id;
}

export function VoicePicker({ value, disabled, onChange }: VoicePickerProps) {
  const { t } = useTranslation("studio");
  const [open, setOpen] = useState(false);
  const [voices, setVoices] = useState<VoiceEntry[] | null>(cached);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [typing, setTyping] = useState(false);
  const [typed, setTyped] = useState("");
  const player = useSpeechPlayer();
  const listId = useId();

  useEffect(() => {
    if (!open) return;
    let live = true;
    getVoices()
      .then((rows) => {
        cached = rows;
        if (live) setVoices(rows);
      })
      .catch((err) => live && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      live = false;
    };
  }, [open]);

  useEffect(() => () => {
    if (speechPlayer.getState().ownerId?.startsWith("story-voice-")) speechPlayer.stop();
  }, []);

  const groupName: Record<VoiceEntryKind, string> = {
    builtin: t("voicePicker.group.builtin", { defaultValue: "Built-in" }),
    preset: t("voicePicker.group.preset", { defaultValue: "Preset" }),
    design: t("voicePicker.group.design", { defaultValue: "Designed" }),
    clone: t("voicePicker.group.clone", { defaultValue: "Cloned" }),
  };
  const current = voiceLabel(value, voices ?? []) ?? t("voicePicker.automatic", { defaultValue: "No voice (automatic)" });
  const groups = groupVoices(voices ?? [], query);
  const sample = t("voicePicker.sampleText", { defaultValue: "Hello. This is how I sound in your film." });

  function pick(id: string | null) {
    onChange(id);
    setOpen(false);
  }

  return (
    <div className="voice-picker">
      <span className="eyebrow">{t("storyProjects.dialog.voice", { defaultValue: "Voice" })}</span>
      <div className="voice-picker__current">
        <span className="voice-picker__value">{current}</span>
        <button
          type="button"
          className="secondary-button"
          aria-expanded={open}
          aria-controls={listId}
          disabled={disabled}
          onClick={() => setOpen((on) => !on)}
        >
          {open ? t("voicePicker.close", { defaultValue: "Close" }) : t("voicePicker.change", { defaultValue: "Change voice" })}
        </button>
      </div>
      {open ? (
        <div className="voice-picker__typed-row">
          {typing ? (
            <form
              className="voice-picker__typed"
              onSubmit={(event) => {
                event.preventDefault();
                if (typed.trim()) pick(typed.trim());
              }}
            >
              <input
                type="text"
                className="text-input"
                value={typed}
                maxLength={200}
                aria-label={t("voicePicker.typeLabel", { defaultValue: "Voice id" })}
                placeholder={t("voicePicker.typePlaceholder", { defaultValue: "A voice id, as an agent or another app names it" })}
                onChange={(event) => setTyped(event.target.value)}
              />
              <button type="submit" className="secondary-button" disabled={!typed.trim()}>
                {t("voicePicker.useTyped", { defaultValue: "Use this voice" })}
              </button>
            </form>
          ) : (
            <button type="button" className="mixer__link" onClick={() => { setTyped(value ?? ""); setTyping(true); }}>
              {t("voicePicker.typeLink", { defaultValue: "Type a voice id instead" })}
            </button>
          )}
        </div>
      ) : null}
      {open ? (
        <div className="voice-picker__panel" id={listId}>
          <input
            type="search"
            className="text-input"
            value={query}
            placeholder={t("voicePicker.search", { defaultValue: "Search voices" })}
            aria-label={t("voicePicker.search", { defaultValue: "Search voices" })}
            onChange={(event) => setQuery(event.target.value)}
          />
          <div role="radiogroup" aria-label={t("storyProjects.dialog.voice", { defaultValue: "Voice" })} className="voice-picker__list">
            <div className="voice-picker__row">
              <label className="voice-picker__choice">
                <input type="radio" name={`${listId}-voice`} checked={!value} onChange={() => pick(null)} />
                <span className="voice-picker__name">{t("voicePicker.automatic", { defaultValue: "No voice (automatic)" })}</span>
              </label>
            </div>
            {error ? <p className="error-text" role="alert">{error}</p> : null}
            {!voices && !error ? <p className="muted-text" role="status">{t("voicePicker.loading", { defaultValue: "Loading voices…" })}</p> : null}
            {voices && groups.length === 0 ? <p className="muted-text">{t("voicePicker.none", { defaultValue: "No voice matches." })}</p> : null}
            {groups.map((group) => (
              <fieldset key={group.kind} className="voice-picker__group">
                <legend className="eyebrow">{groupName[group.kind]}</legend>
                {group.voices.map((voice) => {
                  const owner = `story-voice-${voice.id}`;
                  const playing = player.ownerId === owner && player.status !== "idle";
                  return (
                    <div key={voice.id} className="voice-picker__row">
                      <label className="voice-picker__choice">
                        <input
                          type="radio"
                          name={`${listId}-voice`}
                          checked={value === voice.id}
                          disabled={!voice.available}
                          onChange={() => pick(voice.id)}
                        />
                        <span className="voice-picker__name">{voice.name}</span>
                        <span className="muted-text">{voice.languageLabel ?? voice.language}</span>
                        {voice.kind === "clone" ? <span className="badge warning">{groupName.clone}</span> : null}
                      </label>
                      {!voice.available ? (
                        <span className="muted-text voice-picker__needs">
                          {voice.needs || t("voicePicker.needsModel", { defaultValue: "Needs the model" })}
                        </span>
                      ) : canPlaySample(voice) ? (
                        <button
                          type="button"
                          className="secondary-button icon-button"
                          aria-label={
                            playing
                              ? t("voicePicker.stop", { defaultValue: "Stop sample of {voice}", voice: voice.name })
                              : t("voicePicker.play", { defaultValue: "Play sample of {voice}", voice: voice.name })
                          }
                          onClick={() =>
                            playing
                              ? speechPlayer.stop()
                              : void speechPlayer.play(owner, sample, voice.id, 1)
                          }
                        >
                          {playing ? <StopIcon aria-hidden="true" /> : <PlayIcon aria-hidden="true" />}
                        </button>
                      ) : null}
                    </div>
                  );
                })}
              </fieldset>
            ))}
            {value && voices?.find((voice) => voice.id === value)?.kind === "clone" ? (
              <p className="muted-text">{t("voicePicker.cloneNote", { defaultValue: "A cloned voice always carries the AI tag on its clips." })}</p>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
