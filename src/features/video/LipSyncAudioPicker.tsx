/**
 * The vocal for a lip-sync run in the Video Studio (FU-372), shown on rows that can lip-sync: the file (WAV or FLAC),
 * where in it the clip starts, whether it is sung, how many takes (D-110) and what that costs in time. The clip
 * starts from the start image and lasts 3 to 8 s; the backend checks both again.
 */
import { useTranslation } from "react-i18next";
import { PathField } from "../storyboard/PathField";
import {
  LIP_SYNC_MAX_SECONDS,
  LIP_SYNC_MIN_SECONDS,
  LIP_SYNC_TAKES_BEST,
  formatLipSyncMinutes,
  lipSyncClip,
  lipSyncEstimateSeconds,
  lipSyncTakeCount,
  type LipSyncBlock,
  type LipSyncInput,
} from "./lipSyncLogic";

export interface LipSyncAudioPickerProps {
  block: LipSyncBlock | null | undefined;
  value: LipSyncInput;
  onChange: (next: LipSyncInput) => void;
  numFrames: number;
  fps: number;
  width: number;
  height: number;
  hasStartImage: boolean;
  warm?: boolean;
  disabled?: boolean;
}

export function LipSyncAudioPicker({
  block, value, onChange, numFrames, fps, width, height, hasStartImage, warm = false, disabled,
}: LipSyncAudioPickerProps) {
  const { t } = useTranslation("studio");
  const clip = lipSyncClip(numFrames, fps);
  const takes = lipSyncTakeCount(value.takes, value.singing);
  const estimate = lipSyncEstimateSeconds(block, width, height, clip.frames, takes, warm);
  const shown = estimate != null ? formatLipSyncMinutes(estimate) : null;
  const set = (patch: Partial<LipSyncInput>) => onChange({ ...value, ...patch });

  return (
    <fieldset className="lip-sync-picker">
      <legend className="eyebrow">{t("videoStudio.lipSync.title", { defaultValue: "Vocal for lip sync" })}</legend>
      <p className="muted-text">
        {t("videoStudio.lipSync.hint", {
          defaultValue:
            "Optional: a WAV or FLAC the mouth follows. The clip starts from the start image, lasts {min} to {max} s and keeps this sound. A vocal stem follows best.",
          min: LIP_SYNC_MIN_SECONDS,
          max: LIP_SYNC_MAX_SECONDS,
        })}
      </p>
      {value.audioPath ? (
        <div className="lip-sync-picker__file">
          <code title={value.audioPath}>{value.audioPath}</code>
          <button type="button" className="secondary-button" disabled={disabled} onClick={() => set({ audioPath: null })}>
            {t("videoStudio.lipSync.remove", { defaultValue: "Remove vocal" })}
          </button>
        </div>
      ) : (
        <PathField
          label={t("videoStudio.lipSync.file", { defaultValue: "Vocal file (WAV or FLAC)" })}
          buttonLabel={t("videoStudio.lipSync.attach", { defaultValue: "Use this vocal" })}
          extensions={["wav", "flac"]}
          disabled={disabled}
          onSubmit={(path) => set({ audioPath: path })}
        />
      )}
      {value.audioPath ? (
        <>
          <div className="lip-sync-picker__row">
            <label>
              <span>{t("videoStudio.lipSync.start", { defaultValue: "Start in the vocal (s)" })}</span>
              <input
                type="number"
                min={0}
                max={3600}
                step={0.1}
                value={value.start}
                disabled={disabled}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  set({ start: Number.isFinite(next) ? Math.max(0, Math.min(3600, next)) : 0 });
                }}
              />
            </label>
            <label className="lip-sync-picker__check">
              <input
                type="checkbox"
                checked={value.singing}
                disabled={disabled}
                onChange={(event) => set({ singing: event.target.checked })}
              />
              <span>{t("videoStudio.lipSync.singing", { defaultValue: "Sung vocal" })}</span>
            </label>
            <label>
              <span>{t("videoStudio.lipSync.takes", { defaultValue: "Takes" })}</span>
              <select
                value={value.takes == null ? "" : String(value.takes)}
                disabled={disabled}
                onChange={(event) => set({ takes: event.target.value ? Number(event.target.value) : null })}
              >
                <option value="">
                  {t("videoStudio.lipSync.takesDefault", {
                    defaultValue: "Default ({count})",
                    count: lipSyncTakeCount(null, value.singing),
                  })}
                </option>
                {[1, 2, 3].map((n) => (
                  <option key={n} value={String(n)}>{n}</option>
                ))}
                <option value={String(LIP_SYNC_TAKES_BEST)}>
                  {t("videoStudio.lipSync.takesBest", { defaultValue: "4 (best quality)" })}
                </option>
              </select>
            </label>
          </div>
          <p className="muted-text">
            {t("videoStudio.lipSync.takesHint", {
              defaultValue:
                "Each take uses its own seed; with more than one, the sync score keeps the best and the others stay in the Video Gallery. Sung vocals default to 3 takes, speech to 1.",
            })}
          </p>
          {!clip.ok ? (
            <p className="caution-text" role="note">
              {t("videoStudio.lipSync.length", {
                defaultValue: "This clip is {seconds} s; lip sync needs {min} to {max} s. Change the clip length.",
                seconds: clip.seconds.toFixed(1),
                min: LIP_SYNC_MIN_SECONDS,
                max: LIP_SYNC_MAX_SECONDS,
              })}
            </p>
          ) : null}
          {!hasStartImage ? (
            <p className="caution-text" role="note">
              {t("videoStudio.lipSync.needsStill", { defaultValue: "Lip sync starts from a start image: add one above." })}
            </p>
          ) : null}
          {shown ? (
            <p className="muted-text" role="note">
              {t("videoStudio.lipSync.estimate", {
                defaultValue:
                  "About {value} {unit} for {takes, plural, one {# take} other {# takes}} of a {seconds} s clip (measured on an RTX 4090).",
                value: shown.value,
                unit: shown.unit,
                takes,
                seconds: clip.seconds.toFixed(1),
              })}
            </p>
          ) : null}
        </>
      ) : null}
    </fieldset>
  );
}
