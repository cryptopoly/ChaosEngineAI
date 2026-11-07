import { VoiceStudioTab } from "../features/voice/VoiceStudioTab";
import { VoiceModelsTab } from "../features/voice/VoiceModelsTab";
import { VoiceLibraryTab } from "../features/voice/VoiceLibraryTab";
import type { AppTabContext } from "./context";

export function VoiceStudioView({ ctx }: { ctx: AppTabContext }) {
  const {
    voiceRuntime,
    backendOnline,
    voiceOpenRequest,
    chat,
    setActiveTab,
  } = ctx;
  const {
    setDraftMessage,
  } = chat;
  return (
    <VoiceStudioTab
      voiceRuntime={voiceRuntime}
      backendOnline={backendOnline}
      openRequest={voiceOpenRequest}
      onSendToChat={(text) => {
        // Append rather than overwrite so a half-typed draft survives.
        setDraftMessage((current) => (current.trim() ? `${current}\n${text}` : text));
        void setActiveTab("chat");
      }}
      onTabChange={setActiveTab}
    />
  );
}

export function VoiceModelsView({ ctx }: { ctx: AppTabContext }) {
  const {
    voiceRuntime,
    backendOnline,
    refreshVoiceRuntime,
  } = ctx;
  return (
    <VoiceModelsTab
      voiceRuntime={voiceRuntime}
      backendOnline={backendOnline}
      onRefreshVoiceRuntime={refreshVoiceRuntime}
    />
  );
}

export function VoiceLibraryView({ ctx }: { ctx: AppTabContext }) {
  const {
    backendOnline,
    setVoiceOpenRequest,
    setActiveTab,
  } = ctx;
  return (
    <VoiceLibraryTab
      backendOnline={backendOnline}
      onOpenTranscript={(id) => {
        setVoiceOpenRequest({ id, nonce: Date.now() });
        void setActiveTab("voice-studio");
      }}
      onTabChange={setActiveTab}
    />
  );
}
