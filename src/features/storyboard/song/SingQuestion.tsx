/**
 * The one question after the song is analysed (audit change 8): "Does someone sing on screen?". Yes cuts the song
 * into singing shots of 3 to 5 s and then gives each shot its part of the vocal (a vocal stem follows best); No cuts
 * to the beat with no lip sync. Either way the cut is shown for review first and nothing is made. A shot without
 * singing plays as an ordinary clip, and any shot can drop its vocal in its editor.
 */
import { useTranslation } from "react-i18next";
import { PathField } from "../PathField";
import { LIP_SYNC_RANGE, SUNG_SHOT_MAX_SECONDS, SUNG_SHOT_MIN_SECONDS, type SingAnswer } from "./coverageLogic";

export interface SingQuestionProps {
  answer: SingAnswer | null;
  onAnswer: (answer: SingAnswer) => void;
  stem: string | null;
  onStem: (path: string | null) => void;
  disabled: boolean;
  /** "cut" while the cut is worked out, "vocals" while the vocal is cut and attached. */
  working: "cut" | "vocals" | null;
  /** The board already has shots: offer giving them their vocal without cutting again. */
  hasShots: boolean;
  onCut: () => void;
  onVocalsOnly: () => void;
}

export function SingQuestion({ answer, onAnswer, stem, onStem, disabled, working, hasShots, onCut, onVocalsOnly }: SingQuestionProps) {
  const { t } = useTranslation("studio");
  const busy = disabled || working !== null;
  return (
    <div className="song-sing">
      <fieldset className="sing-question" disabled={busy}>
        <legend>{t("board.song.sing.question", { defaultValue: "Does someone sing on screen?" })}</legend>
        <label className="sing-question__choice">
          <input type="radio" name="sing-on-screen" value="yes" checked={answer === "yes"} onChange={() => onAnswer("yes")} />
          <span>
            {t("board.song.sing.yes", { defaultValue: "Yes, lip-sync the singing" })}
            <span className="muted-text">
              {" · "}
              {t("board.song.sing.yesHint", {
                defaultValue: "Cuts the song into shots of {min} to {max} s, then gives each shot its part of the vocal.",
                min: SUNG_SHOT_MIN_SECONDS,
                max: SUNG_SHOT_MAX_SECONDS,
              })}
            </span>
          </span>
        </label>
        {answer === "yes" ? (
          <div className="sing-question__more">
            {stem ? (
              <div className="lip-sync-picker__file">
                <code title={stem}>{stem}</code>
                <button type="button" className="secondary-button" onClick={() => onStem(null)}>
                  {t("board.song.vocals.stemRemove", { defaultValue: "Use the song instead" })}
                </button>
              </div>
            ) : (
              <PathField
                label={t("board.song.vocals.stem", { defaultValue: "Vocal stem (optional, WAV or FLAC)" })}
                buttonLabel={t("board.song.vocals.stemUse", { defaultValue: "Use this stem" })}
                extensions={["wav", "flac"]}
                disabled={busy}
                onSubmit={onStem}
              />
            )}
            <p className="muted-text">
              {t("board.song.sing.stemHint", {
                defaultValue: "A vocal stem (the voice alone) follows best; without one the song itself is used.",
              })}{" "}
              {t(LIP_SYNC_RANGE.key, { defaultValue: LIP_SYNC_RANGE.defaultValue, ...LIP_SYNC_RANGE.values })}{" "}
              {t("board.song.sing.perShot", {
                defaultValue: "Not every shot has to sing: a shot without singing plays as an ordinary clip, and any shot can drop its vocal in its editor.",
              })}
            </p>
          </div>
        ) : null}
        <label className="sing-question__choice">
          <input type="radio" name="sing-on-screen" value="no" checked={answer === "no"} onChange={() => onAnswer("no")} />
          <span>
            {t("board.song.sing.no", { defaultValue: "No, pictures only" })}
            <span className="muted-text">
              {" · "}
              {t("board.song.sing.noHint", { defaultValue: "Cuts to the beat, with no lip sync." })}
            </span>
          </span>
        </label>
      </fieldset>
      <div className="song-panel__actions">
        <button type="button" className="primary-button" disabled={busy || answer === null} onClick={onCut}>
          {working === "cut"
            ? t("board.song.cutting", { defaultValue: "Cutting to the music…" })
            : t("board.song.sing.cut", { defaultValue: "Cut the song into shots" })}
        </button>
        {answer === "yes" && hasShots ? (
          <button type="button" className="secondary-button" disabled={busy} onClick={onVocalsOnly}>
            {working === "vocals"
              ? t("board.song.vocals.working", { defaultValue: "Cutting the vocal…" })
              : t("board.song.sing.keepShots", { defaultValue: "Keep my shots and give each its vocal" })}
          </button>
        ) : null}
        <span className="muted-text">
          {answer === null
            ? t("board.song.sing.choose", { defaultValue: "Choose an answer first." })
            : t("board.song.sing.review", { defaultValue: "You review the cut first; nothing is made yet." })}
        </span>
      </div>
    </div>
  );
}
