/**
 * Plain text for voice errors (voice studio UX audit, change 6).
 *
 * The backend gives every failed voice request a stable ``code`` and an ``action`` (``VoiceErrorInfo``); this maps
 * them to the one sentence and the one button label the screen shows, with the raw text kept for a Details
 * disclosure. English comes from ``defaultValue``; the other locales fall back to it until they are filled.
 * Pure functions: no React, no network.
 */

import i18next from "i18next";
import { VoiceApiError } from "../../api/voice";
import type { VoiceErrorAction, VoiceErrorCode, VoiceErrorInfo } from "../../types";

/** The sentence for a code (what happened, in plain words; no screen paths, the button does that). */
export function voiceErrorSentence(code: VoiceErrorCode): string {
  switch (code) {
    case "runtime_missing":
      return i18next.t("studio:voice.errors.runtime_missing", {
        defaultValue: "The voice runtime is not installed yet.",
      });
    case "runtime_outdated":
      return i18next.t("studio:voice.errors.runtime_outdated", {
        defaultValue: "The voice runtime needs an update before it can do this.",
      });
    case "addon_missing":
      return i18next.t("studio:voice.errors.addon_missing", {
        defaultValue: "A voice add-on that this needs is not installed.",
      });
    case "model_missing":
      return i18next.t("studio:voice.errors.model_missing", {
        defaultValue: "The model for this is not downloaded yet.",
      });
    case "watermark_missing":
      return i18next.t("studio:voice.errors.watermark_missing", {
        defaultValue: "The watermark part that cloned voices need is missing.",
      });
    case "gpu_memory":
      return i18next.t("studio:voice.errors.gpu_memory", {
        defaultValue: "This model needs more GPU memory than is free.",
      });
    case "gpu_unsupported":
      return i18next.t("studio:voice.errors.gpu_unsupported", {
        defaultValue: "This model needs a newer NVIDIA GPU than this computer has.",
      });
    case "unsupported_here":
      return i18next.t("studio:voice.errors.unsupported_here", {
        defaultValue: "This model cannot run on this computer.",
      });
    case "ffmpeg_missing":
      return i18next.t("studio:voice.errors.ffmpeg_missing", {
        defaultValue: "Changing the speed or pitch needs ffmpeg, which is not installed.",
      });
    case "reference_too_short":
      return i18next.t("studio:voice.errors.reference_too_short", {
        defaultValue: "The voice sample is too short.",
      });
    case "reference_too_long":
      return i18next.t("studio:voice.errors.reference_too_long", {
        defaultValue: "The voice sample is too long.",
      });
    case "reference_unreadable":
      return i18next.t("studio:voice.errors.reference_unreadable", {
        defaultValue: "The voice sample could not be used.",
      });
    case "audio_unreadable":
      return i18next.t("studio:voice.errors.audio_unreadable", {
        defaultValue: "This file could not be read as audio.",
      });
    case "language_unsupported":
      return i18next.t("studio:voice.errors.language_unsupported", {
        defaultValue: "This model does not speak the chosen language.",
      });
    case "permission_required":
      return i18next.t("studio:voice.errors.permission_required", {
        defaultValue: "Permission to clone this voice has not been confirmed.",
      });
    case "cloning_not_allowed":
      return i18next.t("studio:voice.errors.cloning_not_allowed", {
        defaultValue: "Cloned voices can only be used from the app window unless agent access is on.",
      });
    case "voice_not_found":
      return i18next.t("studio:voice.errors.voice_not_found", {
        defaultValue: "That voice no longer exists.",
      });
    case "download_blocked":
      return i18next.t("studio:voice.errors.download_blocked", {
        defaultValue: "The download was refused; the model may need a Hugging Face sign-in.",
      });
    case "download_failed":
      return i18next.t("studio:voice.errors.download_failed", {
        defaultValue: "The download did not finish.",
      });
    case "worker_busy":
      return i18next.t("studio:voice.errors.worker_busy", {
        defaultValue: "Another voice job is running.",
      });
    case "worker_timeout":
      return i18next.t("studio:voice.errors.worker_timeout", {
        defaultValue: "The voice worker took too long to answer.",
      });
    case "worker_stopped":
      return i18next.t("studio:voice.errors.worker_stopped", {
        defaultValue: "The voice worker stopped unexpectedly.",
      });
    default:
      return i18next.t("studio:voice.errors.unknown", {
        defaultValue: "Voice failed for a reason the app does not recognise.",
      });
  }
}

/** The label of the one button for an action; null for ``none`` (the sentence stands alone). */
export function voiceErrorActionLabel(action: VoiceErrorAction): string | null {
  switch (action) {
    case "install-voice-runtime":
      return i18next.t("studio:voice.errorActions.install-voice-runtime", { defaultValue: "Install the voice runtime" });
    case "install-addon":
      return i18next.t("studio:voice.errorActions.install-addon", { defaultValue: "Install the add-on" });
    case "download-model":
      return i18next.t("studio:voice.errorActions.download-model", { defaultValue: "Download the model" });
    case "smaller-model":
      return i18next.t("studio:voice.errorActions.smaller-model", { defaultValue: "Choose a smaller model" });
    case "pick-another-model":
      return i18next.t("studio:voice.errorActions.pick-another-model", { defaultValue: "Choose another model" });
    case "install-ffmpeg":
      return i18next.t("studio:voice.errorActions.install-ffmpeg", { defaultValue: "Install ffmpeg" });
    case "pick-another-language":
      return i18next.t("studio:voice.errorActions.pick-another-language", { defaultValue: "Choose another language" });
    case "change-sample":
      return i18next.t("studio:voice.errorActions.change-sample", { defaultValue: "Choose another sample" });
    case "confirm-permission":
      return i18next.t("studio:voice.errorActions.confirm-permission", { defaultValue: "Confirm permission" });
    case "add-hf-token":
      return i18next.t("studio:voice.errorActions.add-hf-token", { defaultValue: "Add a Hugging Face token" });
    case "retry":
      return i18next.t("studio:voice.errorActions.retry", { defaultValue: "Try again" });
    default:
      return null;
  }
}

/** What a screen shows for a failed voice request. */
export interface VoiceErrorView {
  /** ``unknown`` when the response carried no code (an older backend, a validation error, a network failure). */
  code: VoiceErrorCode;
  /** The one plain sentence. */
  sentence: string;
  action: VoiceErrorAction;
  /** The one button's label; null when there is no action to offer. */
  actionLabel: string | null;
  /** The voice add-on to tick, for ``addon_missing``. */
  extra: VoiceErrorInfo["extra"];
  /** The raw text for the Details disclosure; empty when it only repeats the sentence. */
  details: string;
}

/** The sentence, button and Details for a ``VoiceErrorInfo`` and the raw text. */
export function describeVoiceErrorInfo(info: VoiceErrorInfo | null, raw: string): VoiceErrorView {
  if (!info) {
    // No code: show the raw text as the sentence (it is what the user saw before), and no button.
    return { code: "unknown", sentence: raw || voiceErrorSentence("unknown"), action: "none", actionLabel: null, extra: null, details: "" };
  }
  const sentence = voiceErrorSentence(info.code);
  return {
    code: info.code,
    sentence,
    action: info.action,
    actionLabel: voiceErrorActionLabel(info.action),
    extra: info.extra,
    details: raw && raw !== sentence ? raw : "",
  };
}

/** The same for anything a voice call threw: a ``VoiceApiError``, another ``Error`` or a string. */
export function describeVoiceError(error: unknown): VoiceErrorView {
  if (error instanceof VoiceApiError) return describeVoiceErrorInfo(error.info, error.message);
  const raw = error instanceof Error ? error.message : typeof error === "string" ? error : "";
  return describeVoiceErrorInfo(null, raw);
}
