import { useTranslation } from "react-i18next";
import type { VoiceRuntime } from "../../../types";
import { useVoice } from "../VoiceContext";
import { InstallProgress, useVoicePackJob } from "../VoicePackPanel";
import { ActionCard } from "./VoiceCard";

/** The Mandarin add-on as a card that installs it (Speak's Quick group and the Voices tab). */
export function MandarinCard({ onInstalled }: { onInstalled?: () => void }) {
  const { runtime, refresh } = useVoice();
  if (!runtime) return null;
  return (
    <MandarinCardBody
      runtime={runtime}
      onChanged={() => {
        refresh();
        onInstalled?.();
      }}
    />
  );
}

function MandarinCardBody({ runtime, onChanged }: { runtime: VoiceRuntime; onChanged: () => void }) {
  const { t } = useTranslation("studio");
  const pack = useVoicePackJob(runtime, onChanged);
  return (
    <ActionCard
      title={t("voice.picker.addMandarin", { defaultValue: "Add Mandarin voices" })}
      detail={t("voice.picker.addMandarinDetail", { defaultValue: "Eight quick voices, a small add-on to the voice pack." })}
    >
      {pack.job.running ? (
        <InstallProgress pack={pack} />
      ) : (
        <button type="button" className="secondary-button" onClick={() => void pack.start(["chinese"])}>
          {t("voice.voices.addChinese", { defaultValue: "Add Mandarin Chinese voices" })}
        </button>
      )}
      {pack.error ? (
        <p className="voice-error" role="alert">
          {pack.error}
        </p>
      ) : null}
    </ActionCard>
  );
}
