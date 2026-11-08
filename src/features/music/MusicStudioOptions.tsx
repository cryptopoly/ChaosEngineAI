/**
 * The bodies of the Music Studio's two disclosures. Options: tempo and key, variations, seed. Expert: the
 * model, the output format, the AI tag switch (D-34, on by default) and the idle-unload minutes. Only
 * controls the backend really has are here: no steps, guidance or reference audio.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { formatDuration } from "../../components/RunEstimateStrip";
import { summaryLine } from "../../components/studioSummary";
import type { MusicModelVariant, MusicRuntimeStatus } from "../../types";
import {
  BPM_MAX,
  BPM_MIN,
  MUSIC_KEYS,
  SEED_MAX,
  VARIATION_CHOICES,
  type MusicForm,
  type Variations,
} from "./musicLogic";

type Translate = (key: string, options?: Record<string, unknown>) => string;

export function optionsSummary(t: Translate, form: MusicForm): string {
  return (
    summaryLine([
      t("musicStudio.options.summaryBpm", { defaultValue: "BPM {value}", value: form.bpm ?? t("musicStudio.options.auto", { defaultValue: "Auto" }).toLowerCase() }),
      t("musicStudio.options.summaryKey", { defaultValue: "Key {value}", value: form.key ?? t("musicStudio.options.auto", { defaultValue: "Auto" }).toLowerCase() }),
      t("musicStudio.options.summaryVariations", { defaultValue: "{n, plural, one {# variation} other {# variations}}", n: form.variations }),
      form.seed === null
        ? t("musicStudio.options.randomSeed", { defaultValue: "Random seed" })
        : t("musicStudio.options.summarySeed", { defaultValue: "Seed {n}", n: form.seed }),
    ]) ?? ""
  );
}

export function expertSummary(t: Translate, runtime: MusicRuntimeStatus | null, variant: MusicModelVariant | null, installedCount: number): string {
  return (
    summaryLine(
      [
        installedCount > 1 && variant ? variant.name : false,
        runtime && !runtime.embedTag ? t("musicStudio.expert.summaryTagOff", { defaultValue: "AI tag off" }) : false,
        runtime && runtime.idleUnloadMinutes !== 10
          ? t("musicStudio.expert.summaryIdle", { defaultValue: "Unload after {n} min", n: runtime.idleUnloadMinutes })
          : false,
      ],
      t("disclosure.defaults", { defaultValue: "Model defaults" }),
    ) ?? ""
  );
}

export interface OptionsBodyProps {
  form: MusicForm;
  onChange: (patch: Partial<MusicForm>) => void;
  /** The estimate's seconds for the chosen variations, for the "about X for N" note. */
  estimateSeconds: number | null;
  disabled?: boolean;
}

function numberOrNull(text: string): number | null {
  if (text.trim() === "") return null;
  const value = Number(text);
  return Number.isFinite(value) ? Math.round(value) : null;
}

export function MusicOptionsBody({ form, onChange, estimateSeconds, disabled }: OptionsBodyProps) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as Translate;
  const timeFor = estimateSeconds != null && form.variations > 1 ? formatDuration(t as unknown as Parameters<typeof formatDuration>[0], estimateSeconds) : null;
  return (
    <div className="music-options">
      <div className="music-options__group">
        <span className="preset-row-label">{t("musicStudio.options.tempoKey", { defaultValue: "Tempo and key" })}</span>
        <div className="music-options__row">
          <label className="music-options__field">
            <span>{t("musicStudio.options.bpm", { defaultValue: "BPM" })}</span>
            <input
              className="text-input"
              type="number"
              inputMode="numeric"
              min={BPM_MIN}
              max={BPM_MAX}
              disabled={disabled}
              placeholder={t("musicStudio.options.auto", { defaultValue: "Auto" })}
              value={form.bpm ?? ""}
              onChange={(event) => onChange({ bpm: numberOrNull(event.target.value) })}
            />
          </label>
          <label className="music-options__field">
            <span>{t("musicStudio.options.key", { defaultValue: "Key" })}</span>
            <select
              className="text-input"
              disabled={disabled}
              value={form.key ?? ""}
              onChange={(event) => onChange({ key: event.target.value || null })}
            >
              <option value="">{t("musicStudio.options.auto", { defaultValue: "Auto" })}</option>
              {MUSIC_KEYS.map((key) => (
                <option key={key} value={key}>
                  {key}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="music-options__group">
        <span className="preset-row-label">{t("musicStudio.options.variations", { defaultValue: "Variations" })}</span>
        <div className="preset-row">
          {VARIATION_CHOICES.map((choice) => (
            <button
              key={choice}
              type="button"
              className={`preset-pill${form.variations === choice ? " active" : ""}`}
              aria-pressed={form.variations === choice}
              disabled={disabled}
              onClick={() => onChange({ variations: choice as Variations })}
            >
              <span className="preset-pill-label">{choice}</span>
            </button>
          ))}
        </div>
        <p className="muted-text music-options__note">
          {t("musicStudio.options.variationsCost", { defaultValue: "Each variation adds the same time again." })}
          {timeFor
            ? ` ${t("musicStudio.options.variationsTotal", { defaultValue: "About {time} for {n}.", time: timeFor, n: form.variations })}`
            : ""}
        </p>
      </div>

      <div className="music-options__group">
        <span className="preset-row-label">{t("musicStudio.options.seed", { defaultValue: "Seed" })}</span>
        <div className="music-options__row">
          <label className="check-row">
            <input
              type="checkbox"
              checked={form.seed === null}
              disabled={disabled}
              onChange={(event) => onChange({ seed: event.target.checked ? null : 0 })}
            />
            {t("musicStudio.options.randomSeed", { defaultValue: "Random seed" })}
          </label>
          <input
            className="text-input music-options__seed"
            type="number"
            inputMode="numeric"
            min={0}
            max={SEED_MAX}
            aria-label={t("musicStudio.options.seed", { defaultValue: "Seed" })}
            disabled={disabled || form.seed === null}
            value={form.seed ?? ""}
            onChange={(event) => onChange({ seed: numberOrNull(event.target.value) ?? 0 })}
          />
        </div>
      </div>
    </div>
  );
}

export interface ExpertBodyProps {
  runtime: MusicRuntimeStatus | null;
  variants: MusicModelVariant[];
  selected: MusicModelVariant | null;
  onSelectModel: (id: string) => void;
  onEmbedTag: (on: boolean) => void;
  onIdleMinutes: (minutes: number) => void;
  disabled?: boolean;
}

export function MusicExpertBody({ runtime, variants, selected, onSelectModel, onEmbedTag, onIdleMinutes, disabled }: ExpertBodyProps) {
  const { t } = useTranslation("studio");
  const stored = runtime?.idleUnloadMinutes ?? 10;
  const [idle, setIdle] = useState(String(stored));
  useEffect(() => setIdle(String(stored)), [stored]);

  function commitIdle() {
    const minutes = Math.min(240, Math.max(0, Math.round(Number(idle))));
    if (!Number.isFinite(minutes)) {
      setIdle(String(stored));
      return;
    }
    setIdle(String(minutes));
    if (minutes !== stored) onIdleMinutes(minutes);
  }

  return (
    <div className="music-options">
      <label className="music-options__field">
        <span>{t("musicStudio.expert.model", { defaultValue: "Model" })}</span>
        <select
          className="text-input"
          disabled={disabled || variants.length === 0}
          value={selected?.id ?? ""}
          onChange={(event) => onSelectModel(event.target.value)}
        >
          {variants.map((variant) => (
            <option key={variant.id} value={variant.id}>
              {variant.name}
            </option>
          ))}
        </select>
      </label>
      <p className="muted-text music-options__note">
        {t("musicStudio.expert.output", { defaultValue: "Output: FLAC, 24-bit, 48 kHz, stereo." })}
      </p>
      <label className="check-row">
        <input
          type="checkbox"
          checked={runtime?.embedTag ?? true}
          disabled={disabled || !runtime}
          onChange={(event) => onEmbedTag(event.target.checked)}
        />
        {t("aiLabel.metadata", { defaultValue: "Mark the file as AI-generated (metadata)" })}
      </label>
      <p className="muted-text music-options__note">
        {t("musicStudio.expert.aiTagNote", {
          defaultValue: "A machine-readable tag in the audio file. It is on by default; switch it off for private use.",
        })}
      </p>
      <label className="music-options__field">
        <span>{t("musicStudio.expert.idleUnload", { defaultValue: "Unload the model after this many idle minutes (0 keeps it loaded)" })}</span>
        <input
          className="text-input music-options__seed"
          type="number"
          inputMode="numeric"
          min={0}
          max={240}
          disabled={disabled || !runtime}
          value={idle}
          onChange={(event) => setIdle(event.target.value)}
          onBlur={commitIdle}
          onKeyDown={(event) => {
            if (event.key === "Enter") commitIdle();
          }}
        />
      </label>
    </div>
  );
}
