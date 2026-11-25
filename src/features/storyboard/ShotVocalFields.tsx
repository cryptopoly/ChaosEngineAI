/**
 * The shot editor's "Vocal for lip sync" part (D-103): the vocal file, where in it the clip starts (FU-373), whether
 * it is sung and how many takes (D-110). Each change is its own op, like attaching the vocal.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardProject, StoryboardShot } from "../../types";
import { LIP_SYNC_TAKES_BEST, lipSyncTakeCount } from "../video/lipSyncLogic";
import { PathField } from "./PathField";
import { ProjectAudio } from "./ProjectMedia";
import { LIP_SYNC_RANGE } from "./song/coverageLogic";
import type { RunOp } from "./useStoryboard";

export interface ShotVocalFieldsProps {
  project: StoryboardProject;
  shot: StoryboardShot;
  disabled: boolean;
  op: RunOp;
}

export function ShotVocalFields({ project, shot, disabled, op }: ShotVocalFieldsProps) {
  const { t } = useTranslation("studio");
  const [start, setStart] = useState(String(shot.vocalStart ?? 0));
  useEffect(() => setStart(String(shot.vocalStart ?? 0)), [shot.vocalStart]);
  const singing = Boolean(shot.vocalSinging);

  function commitStart() {
    const value = Number(start);
    const next = Number.isFinite(value) ? Math.max(0, Math.min(3600, value)) : 0;
    if (next !== (shot.vocalStart ?? 0)) void op("edit_shot", { shotId: shot.id, vocalStart: next });
    else setStart(String(next));
  }

  return (
    <fieldset className="shot-editor__refs shot-editor__vocal">
      <legend className="eyebrow">{t("board.editor.vocal", { defaultValue: "Vocal for lip sync" })}</legend>
      <p className="muted-text">
        {t("board.editor.vocalHintShot", {
          defaultValue:
            "A WAV or FLAC the mouth follows: the clip starts from this shot's still and keeps this sound. Needs LTX-2.3 as the video model (the 4-bit NVIDIA model wants a 24 GB card).",
        })}{" "}
        {t(LIP_SYNC_RANGE.key, { defaultValue: LIP_SYNC_RANGE.defaultValue, ...LIP_SYNC_RANGE.values })}{" "}
        {t("board.editor.vocalOptional", {
          defaultValue: "A shot does not have to sing: remove its vocal and it plays as an ordinary clip.",
        })}
      </p>
      {shot.vocalClip ? (
        <ProjectAudio projectId={project.id} rel={shot.vocalClip} version={shot.vocalClip} className="shot-editor__vocal-player" />
      ) : null}
      <PathField
        label={t("board.editor.vocalFile", { defaultValue: "Vocal file (WAV or FLAC)" })}
        buttonLabel={
          shot.vocalClip
            ? t("board.editor.vocalChange", { defaultValue: "Change vocal" })
            : t("board.editor.vocalAttach", { defaultValue: "Attach vocal" })
        }
        extensions={["wav", "flac"]}
        disabled={disabled}
        onSubmit={(path) => void op("set_vocal_clip", { shotId: shot.id, audio: path })}
      />
      {shot.vocalClip ? (
        <>
          <div className="lip-sync-picker__row">
            <label>
              <span>{t("board.editor.vocalStart", { defaultValue: "Start in the vocal (s)" })}</span>
              <input
                type="number"
                min={0}
                max={3600}
                step={0.1}
                value={start}
                disabled={disabled}
                onChange={(event) => setStart(event.target.value)}
                onBlur={commitStart}
                onKeyDown={(event) => {
                  if (event.key === "Enter") commitStart();
                }}
              />
            </label>
            <label className="lip-sync-picker__check">
              <input
                type="checkbox"
                checked={singing}
                disabled={disabled}
                onChange={(event) => void op("edit_shot", { shotId: shot.id, vocalSinging: event.target.checked })}
              />
              <span>{t("board.editor.vocalSinging", { defaultValue: "Sung vocal" })}</span>
            </label>
            <label>
              <span>{t("board.editor.vocalTakes", { defaultValue: "Takes" })}</span>
              <select
                value={shot.vocalTakes == null ? "" : String(shot.vocalTakes)}
                disabled={disabled}
                onChange={(event) =>
                  void op("edit_shot", { shotId: shot.id, vocalTakes: event.target.value ? Number(event.target.value) : null })
                }
              >
                <option value="">
                  {t("board.editor.vocalTakesDefault", { defaultValue: "Default ({count})", count: lipSyncTakeCount(null, singing) })}
                </option>
                {[1, 2, 3].map((n) => (
                  <option key={n} value={String(n)}>{n}</option>
                ))}
                <option value={String(LIP_SYNC_TAKES_BEST)}>
                  {t("board.editor.vocalTakesBest", { defaultValue: "4 (best quality)" })}
                </option>
              </select>
            </label>
          </div>
          <p className="muted-text">
            {t("board.editor.vocalTakesHint", {
              defaultValue:
                "Sung vocals: use the vocal stem, a close-up frontal still with the mouth closed, and start about 0.4 s before a phrase. Each take uses its own seed; the sync score keeps the best and the others stay in the Video Gallery.",
            })}
          </p>
          <div className="shot-editor__actions">
            <button
              type="button"
              className="secondary-button"
              disabled={disabled}
              onClick={() => void op("set_vocal_clip", { shotId: shot.id, audio: "" })}
            >
              {t("board.editor.vocalRemove", { defaultValue: "Remove vocal" })}
            </button>
          </div>
        </>
      ) : null}
    </fieldset>
  );
}
