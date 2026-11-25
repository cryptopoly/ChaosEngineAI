import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import { useVoice } from "../VoiceContext";
import type { ClonedVoice } from "./cloneApi";
import { cloneActionBlock, clonedVoiceId } from "./cloneLogic";
import { NewVoiceForm } from "./NewVoiceForm";

export interface CloneVoiceScreenProps {
  /** What the screen needs of the voice catalogue (``VoiceCatalog`` fits). */
  catalog: { modelsAvailable: boolean; cloneAllowed: boolean; refresh: () => Promise<void> };
  /** "Speak with it": open Speak with the new voice selected. The id is the voice id Speak uses (``clone:<id>``). */
  onSpeakWith: (voiceId: string) => void;
  /** Fix for a greyed screen or a missing part: Voice setup (Voice Models until the setup card exists). */
  onOpenSetup?: () => void;
}

/** After Save: "Voice saved. [Speak with it]", and a way to make another. */
export function CloneSaved({ voice, onSpeakWith, onAnother }: { voice: ClonedVoice; onSpeakWith: (voiceId: string) => void; onAnother: () => void }) {
  const { t } = useTranslation("studio");
  return (
    <div className="voice-studio voice-clone__saved">
      <p role="status">
        {t("voice.clone.saved", { defaultValue: "Voice saved." })} <strong>{voice.name}</strong>
      </p>
      <div className="button-row">
        <button type="button" className="primary-button" onClick={() => onSpeakWith(clonedVoiceId(voice))}>
          {t("voice.clone.speakWithIt", { defaultValue: "Speak with it" })}
        </button>
        <button type="button" className="secondary-button" onClick={onAnother}>
          {t("voice.clone.another", { defaultValue: "Clone another voice" })}
        </button>
      </div>
    </div>
  );
}

/**
 * The "Clone a voice" task of the Voice Studio: the four-step form, or the reason it cannot be used here with the
 * button that fixes it, and after Save "Voice saved. [Speak with it]". A component the Studio mounts as a task.
 */
export function CloneVoiceScreen({ catalog, onSpeakWith, onOpenSetup }: CloneVoiceScreenProps) {
  const { t } = useTranslation("studio");
  const { runtime, openVoiceModels } = useVoice();
  const [saved, setSaved] = useState<ClonedVoice | null>(null);
  const [formKey, setFormKey] = useState(0);
  const openSetup = onOpenSetup ?? openVoiceModels;
  const block = cloneActionBlock(
    "clone",
    { modelsAvailable: catalog.modelsAvailable, cloneAllowed: catalog.cloneAllowed, runtime },
    t as unknown as (key: string, options?: Record<string, unknown>) => string,
  );

  return (
    <Panel
      title={t("voice.clone.screenTitle", { defaultValue: "Clone a voice" })}
      subtitle={t("voice.clone.screenSubtitle", { defaultValue: "Make a voice from a short recording, then speak with it." })}
      className="span-2"
    >
      {saved ? (
        <CloneSaved
          voice={saved}
          onSpeakWith={onSpeakWith}
          onAnother={() => {
            setSaved(null);
            setFormKey((key) => key + 1);
          }}
        />
      ) : block ? (
        <div className="voice-studio voice-clone__blocked">
          <p className="muted-text">{block.reason}</p>
          {block.fix === "setup" ? (
            <div className="button-row">
              <button type="button" className="secondary-button" onClick={openSetup}>
                {t("voice.clone.openSetup", { defaultValue: "Open Voice setup" })}
              </button>
            </div>
          ) : null}
        </div>
      ) : (
        <NewVoiceForm
          key={formKey}
          onOpenSetup={openSetup}
          onCreated={(voice) => {
            setSaved(voice);
            void catalog.refresh();
          }}
        />
      )}
    </Panel>
  );
}
