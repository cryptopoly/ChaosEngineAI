/**
 * The inspector for the selected clip (gain, fades, trim, duck, loop, anchor, remove) or the selected
 * shot (its own sound level). Every edit goes through the project's ops. Numbers keep what the user is
 * typing until they leave the field, so a half-typed value is never rewritten under the cursor, and a
 * slider commits once when it is let go. These fields are also the keyboard path for every drag.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { CloseIcon, DuckIcon, LoopIcon, TrashIcon } from "../../../components/icons/ActionIcons";
import type { StoryboardAudioClip, StoryboardShot, StoryboardShotSpan } from "../../../types";
import type { RunOp } from "../ReferencesPanel";
import {
  anchorPatchFor,
  clampNumber,
  clampTrimIn,
  clampTrimOut,
  outPoint,
  type ClipSpan,
} from "./timelineGeometry";

export const GAIN_MIN_DB = -60;
export const GAIN_MAX_DB = 12;

interface NumberFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  disabled?: boolean;
  onCommit: (value: number) => void;
}

function NumberField({ label, value, min, max, step, suffix, disabled, onCommit }: NumberFieldProps) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const next = clampNumber(text, min, max, value);
    setText(String(next));
    if (next !== value) onCommit(next);
  };
  return (
    <label className="clip-inspector__field">
      <span className="eyebrow">{label}</span>
      <span className="clip-inspector__number">
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={text}
          disabled={disabled}
          onChange={(event) => setText(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commit();
            }
          }}
        />
        {suffix ? <span aria-hidden="true">{suffix}</span> : null}
      </span>
    </label>
  );
}

interface GainSliderProps {
  label: string;
  value: number;
  disabled?: boolean;
  onCommit: (db: number) => void;
}

function GainSlider({ label, value, disabled, onCommit }: GainSliderProps) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const commit = () => {
    if (draft !== value) onCommit(draft);
  };
  return (
    <div className="clip-inspector__field clip-inspector__field--wide">
      <span className="eyebrow">{label}</span>
      <div className="slider-number-row">
        <input
          type="range"
          min={GAIN_MIN_DB}
          max={GAIN_MAX_DB}
          step={0.5}
          value={draft}
          disabled={disabled}
          aria-label={label}
          onChange={(event) => setDraft(event.target.valueAsNumber)}
          onPointerUp={commit}
          onKeyUp={commit}
          onBlur={commit}
        />
        <output className="clip-inspector__db">{draft} dB</output>
      </div>
    </div>
  );
}

export interface ClipInspectorProps {
  clip: StoryboardAudioClip;
  span: ClipSpan;
  shots: StoryboardShot[];
  shotSpans: StoryboardShotSpan[];
  disabled: boolean;
  op: RunOp;
  onRemove: () => void;
  /** Shown at narrow widths, where the inspector is a drawer. */
  onClose: () => void;
}

export function ClipInspector({ clip, span, shots, shotSpans, disabled, op, onRemove, onClose }: ClipInspectorProps) {
  const { t } = useTranslation("studio");
  const edit = (fields: Record<string, unknown>) => void op("edit_audio", { clipId: clip.id, ...fields });
  const anchoredIndex = clip.anchor ? shots.findIndex((s) => s.id === clip.anchor?.shotId) : -1;
  const untitled = (index: number) => t("board.shot.untitled", { defaultValue: "Shot {n}", n: index + 1 });
  const sourceLength = clip.source.measuredSeconds;

  return (
    <section className="clip-inspector" role="region" aria-label={t("mixer.inspector.title", { defaultValue: "Selected clip" })}>
      <div className="clip-inspector__head">
        <h3>{t("mixer.inspector.title", { defaultValue: "Selected clip" })}</h3>
        <button
          type="button"
          className="secondary-button icon-button clip-inspector__close"
          aria-label={t("storyboard.close", { defaultValue: "Close" })}
          title={t("storyboard.close", { defaultValue: "Close" })}
          onClick={onClose}
        >
          <CloseIcon />
        </button>
      </div>
      <label className="clip-inspector__field clip-inspector__field--wide">
        <span className="eyebrow">{t("mixer.inspector.name", { defaultValue: "Name" })}</span>
        <input
          key={clip.id + clip.source.label}
          type="text"
          maxLength={200}
          defaultValue={clip.source.label}
          placeholder={clip.source.path.split("/").pop() ?? ""}
          disabled={disabled}
          onBlur={(event) => {
            const next = event.target.value.trim();
            if (next && next !== clip.source.label) edit({ label: next });
          }}
        />
      </label>
      <GainSlider
        label={t("mixer.inspector.gain", { defaultValue: "Gain" })}
        value={clip.gainDb}
        disabled={disabled}
        onCommit={(gainDb) => edit({ gainDb })}
      />
      <div className="clip-inspector__row">
        <NumberField
          label={t("mixer.inspector.fadeIn", { defaultValue: "Fade in" })}
          value={clip.fadeIn}
          min={0}
          max={60}
          step={0.1}
          suffix="s"
          disabled={disabled}
          onCommit={(fadeIn) => edit({ fadeIn })}
        />
        <NumberField
          label={t("mixer.inspector.fadeOut", { defaultValue: "Fade out" })}
          value={clip.fadeOut}
          min={0}
          max={60}
          step={0.1}
          suffix="s"
          disabled={disabled}
          onCommit={(fadeOut) => edit({ fadeOut })}
        />
      </div>
      <div className="clip-inspector__row">
        <NumberField
          label={`${t("mixer.inspector.trim", { defaultValue: "Trim" })} · ${t("mixer.inspector.trimIn", { defaultValue: "in" })}`}
          value={clip.trimIn}
          min={0}
          max={sourceLength ?? 3600}
          step={0.1}
          suffix="s"
          disabled={disabled}
          onCommit={(value) => edit({ trimIn: clampTrimIn(clip, value) })}
        />
        <NumberField
          label={`${t("mixer.inspector.trim", { defaultValue: "Trim" })} · ${t("mixer.inspector.trimOut", { defaultValue: "out" })}`}
          value={Math.round(outPoint(clip) * 1000) / 1000}
          min={0}
          max={sourceLength ?? 3600}
          step={0.1}
          suffix="s"
          disabled={disabled || clip.loop}
          onCommit={(value) => edit({ trimOut: clampTrimOut(clip, value) })}
        />
      </div>
      <div className="clip-inspector__row clip-inspector__row--toggles">
        <label className="check-row" title={t("mixer.inspector.duckTitle", { defaultValue: "Lowers this clip while someone speaks (sidechain compression)." })}>
          <input type="checkbox" checked={clip.duck} disabled={disabled || clip.kind === "voice"} onChange={(event) => edit({ duck: event.target.checked })} />
          <DuckIcon /> {t("mixer.inspector.duck", { defaultValue: "Duck under voice" })}
        </label>
        <label className="check-row">
          <input type="checkbox" checked={clip.loop} disabled={disabled} onChange={(event) => edit({ loop: event.target.checked })} />
          <LoopIcon /> {t("mixer.inspector.loop", { defaultValue: "Loop to fill" })}
        </label>
      </div>
      <fieldset className="clip-inspector__anchor">
        <legend className="eyebrow">{t("mixer.inspector.anchor", { defaultValue: "Anchor" })}</legend>
        <div className="clip-inspector__row">
          <label className="clip-inspector__field">
            <span className="sr-only">{t("mixer.inspector.anchorTo", { defaultValue: "Anchored to" })}</span>
            <select
              value={clip.anchor?.shotId ?? ""}
              disabled={disabled}
              onChange={(event) => edit(anchorPatchFor(event.target.value || null, span.start, shotSpans))}
            >
              <option value="">{t("mixer.inspector.absolute", { defaultValue: "Film time (not anchored)" })}</option>
              {shots.map((shot, index) => (
                <option key={shot.id} value={shot.id}>
                  {t("mixer.inspector.anchorShot", {
                    defaultValue: "Shot {n}: {name}",
                    n: index + 1,
                    name: shot.title.trim() || untitled(index),
                  })}
                </option>
              ))}
            </select>
          </label>
          {clip.anchor ? (
            <NumberField
              label={t("mixer.inspector.offset", { defaultValue: "Offset" })}
              value={clip.anchor.offset}
              min={-60}
              max={3600}
              step={0.1}
              suffix="s"
              disabled={disabled}
              onCommit={(offset) => edit({ anchor: { shotId: clip.anchor?.shotId ?? "", offset } })}
            />
          ) : (
            <NumberField
              label={t("mixer.inspector.start", { defaultValue: "Starts at" })}
              value={clip.start ?? 0}
              min={0}
              max={3600}
              step={0.1}
              suffix="s"
              disabled={disabled}
              onCommit={(start) => edit({ start })}
            />
          )}
        </div>
        {anchoredIndex >= 0 && clip.anchor ? (
          <p className="muted-text clip-inspector__hint">
            {t("mixer.inspector.anchorSummary", {
              defaultValue: "Moves with shot {n}. Offset {offset} s.",
              n: anchoredIndex + 1,
              offset: clip.anchor.offset,
            })}
          </p>
        ) : null}
      </fieldset>
      <button type="button" className="secondary-button danger-button clip-inspector__remove" disabled={disabled} onClick={onRemove}>
        <TrashIcon /> {t("mixer.inspector.remove", { defaultValue: "Remove from lane" })}
      </button>
    </section>
  );
}

export interface ShotInspectorProps {
  shot: StoryboardShot;
  index: number;
  span: StoryboardShotSpan | undefined;
  disabled: boolean;
  op: RunOp;
  onOpenBoard: () => void;
  onClose: () => void;
}

export function ShotInspector({ shot, index, span, disabled, op, onOpenBoard, onClose }: ShotInspectorProps) {
  const { t } = useTranslation("studio");
  const name = shot.title.trim() || t("board.shot.untitled", { defaultValue: "Shot {n}", n: index + 1 });
  return (
    <section className="clip-inspector" role="region" aria-label={t("mixer.inspector.shotTitle", { defaultValue: "Selected shot" })}>
      <div className="clip-inspector__head">
        <h3>{name}</h3>
        <button
          type="button"
          className="secondary-button icon-button clip-inspector__close"
          aria-label={t("storyboard.close", { defaultValue: "Close" })}
          title={t("storyboard.close", { defaultValue: "Close" })}
          onClick={onClose}
        >
          <CloseIcon />
        </button>
      </div>
      {span ? (
        <p className="muted-text">
          {t("mixer.inspector.shotSpan", {
            defaultValue: "From {start} s, {length} s long",
            start: Math.round(span.start * 10) / 10,
            length: Math.round(span.duration * 10) / 10,
          })}
        </p>
      ) : null}
      <GainSlider
        label={t("mixer.inspector.nativeLevel", { defaultValue: "Shot sound level for shot {n}", n: index + 1 })}
        value={shot.nativeSoundDb}
        disabled={disabled}
        onCommit={(nativeSoundDb) => void op("edit_shot", { shotId: shot.id, nativeSoundDb })}
      />
      <button type="button" className="secondary-button" onClick={onOpenBoard}>
        {t("mixer.openOnBoard", { defaultValue: "Open on Board" })}
      </button>
    </section>
  );
}
