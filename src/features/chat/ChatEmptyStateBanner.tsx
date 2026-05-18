import { useTranslation } from "react-i18next";

/**
 * Empty-state CTA for the Chat tab on a fresh install (FU-056 follow-up).
 *
 * Renders inside the empty thread when the user opens Chat before
 * downloading their first chat model. Instead of just "Send a message
 * to start the conversation." (which silently auto-loads the largest
 * MLX direct variant — broken on Windows + slow on Macs), this card
 * points the user at the Discover tab where they can pick their own
 * starter.
 *
 * Two states:
 *   - No chat models in the library → "Browse Discover" CTA
 *   - Models present but none loaded → "Load a model from Models →" hint
 *
 * Both states are non-blocking — the composer is still usable above,
 * but sending without a model does nothing useful, so the card sits
 * inside the thread where the empty conversation would otherwise be.
 */

export interface ChatEmptyStateBannerProps {
  /** True when the library has zero chat-capable models. Drives the
   * primary CTA (Discover for new users, Models for users who have
   * downloaded but not loaded). */
  noChatModelsInstalled: boolean;
  /** Fired when the user clicks the primary CTA. The parent maps this
   * to the appropriate tab change. */
  onBrowseDiscover: () => void;
  /** Fired when the user clicks "go to Models" (only shown when they
   * already have at least one chat model). */
  onOpenModels: () => void;
}

export function ChatEmptyStateBanner({
  noChatModelsInstalled,
  onBrowseDiscover,
  onOpenModels,
}: ChatEmptyStateBannerProps) {
  const { t } = useTranslation("chat");

  if (noChatModelsInstalled) {
    return (
      <div className="chat-empty-banner" role="region" aria-label="Get started with chat">
        <h3 className="chat-empty-banner-title">
          {t("emptyBanner.welcomeTitle", {
            defaultValue: "👋 Welcome to ChaosEngineAI Chat",
          })}
        </h3>
        <p className="chat-empty-banner-body">
          {t("emptyBanner.welcomeBody", {
            defaultValue:
              "Pick a chat model from Discover to get started. We recommend a small Qwen3 or Llama 3 variant for your first run — they download in a minute or two and run on any laptop.",
          })}
        </p>
        <div className="chat-empty-banner-actions">
          <button
            type="button"
            className="primary-button"
            onClick={onBrowseDiscover}
          >
            {t("emptyBanner.browseDiscover", { defaultValue: "Browse Discover" })}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-empty-banner" role="region" aria-label="Load a model to chat">
      <p className="chat-empty-banner-body">
        {t("emptyBanner.noModelLoaded", {
          defaultValue:
            "A model needs to be loaded before you can chat. Pick one from your library to bring it into memory.",
        })}
      </p>
      <div className="chat-empty-banner-actions">
        <button type="button" className="secondary-button" onClick={onOpenModels}>
          {t("emptyBanner.loadModel", { defaultValue: "Load Model" })}
        </button>
      </div>
    </div>
  );
}
