import type { SidebarGroupId, TabId } from "../types";

/**
 * Tab registry — single source of truth for sidebar / subtab / workspace
 * header navigation.
 *
 * FU-042: each entry carries both an English literal *and* an i18n key.
 * Consumers (Sidebar, SubtabBar, App.tsx workspace header) resolve the
 * key via ``t(tab.labelKey, { defaultValue: tab.label })`` so the
 * English literal is the source of truth when no translation has been
 * authored yet, and locale catalogs override on a per-locale basis.
 */
export interface TabConfig {
  id: TabId;
  /** English fallback used when the i18n key isn't resolved. */
  label: string;
  /** i18n key under ``common.tabs.<key>``. */
  labelKey: string;
  /** English fallback for the subtitle / hover hint. */
  caption: string;
  /** i18n key under ``common.tabCaptions.<key>``. */
  captionKey: string;
  group?: SidebarGroupId;
  /** Compact label shown in collapsed-tabs mode. */
  shortLabel?: string;
  /** i18n key under ``common.tabShortLabels.<key>``. */
  shortLabelKey?: string;
}

export const tabs: TabConfig[] = [
  { id: "dashboard", label: "Dashboard", labelKey: "tabs.dashboard", caption: "System overview", captionKey: "tabCaptions.dashboard" },
  { id: "chat", label: "Chat", labelKey: "tabs.chat", caption: "Local AI chat", captionKey: "tabCaptions.chat", group: "chat", shortLabel: "Chat", shortLabelKey: "tabShortLabels.chat" },
  { id: "chat-compare", label: "Compare", labelKey: "tabs.chatCompare", caption: "Compare language models side by side", captionKey: "tabCaptions.chatCompare", group: "chat", shortLabel: "Compare", shortLabelKey: "tabShortLabels.chatCompare" },
  { id: "html-challenge", label: "HTML Challenge", labelKey: "tabs.htmlChallenge", caption: "Generate shareable webpage comparisons", captionKey: "tabCaptions.htmlChallenge", group: "chat", shortLabel: "HTML Challenge", shortLabelKey: "tabShortLabels.htmlChallenge" },
  { id: "my-models", label: "My Models", labelKey: "tabs.myModels", caption: "Models on this machine", captionKey: "tabCaptions.myModels", group: "chat", shortLabel: "My Models", shortLabelKey: "tabShortLabels.myModels" },
  { id: "online-models", label: "Discover", labelKey: "tabs.onlineModels", caption: "Browse and download AI models", captionKey: "tabCaptions.onlineModels", group: "chat", shortLabel: "Discover", shortLabelKey: "tabShortLabels.onlineModels" },
  { id: "image-models", label: "Image Models", labelKey: "tabs.imageModels", caption: "Installed image generators", captionKey: "tabCaptions.imageModels", group: "images", shortLabel: "My Models", shortLabelKey: "tabShortLabels.imageModels" },
  { id: "image-discover", label: "Image Discover", labelKey: "tabs.imageDiscover", caption: "Browse image models", captionKey: "tabCaptions.imageDiscover", group: "images", shortLabel: "Discover", shortLabelKey: "tabShortLabels.imageDiscover" },
  { id: "image-studio", label: "Image Studio", labelKey: "tabs.imageStudio", caption: "Prompt, generate, and iterate", captionKey: "tabCaptions.imageStudio", group: "images", shortLabel: "Studio", shortLabelKey: "tabShortLabels.imageStudio" },
  { id: "image-gallery", label: "Image Gallery", labelKey: "tabs.imageGallery", caption: "Saved outputs and filters", captionKey: "tabCaptions.imageGallery", group: "images", shortLabel: "Gallery", shortLabelKey: "tabShortLabels.imageGallery" },
  { id: "video-models", label: "Video Models", labelKey: "tabs.videoModels", caption: "Installed video generators", captionKey: "tabCaptions.videoModels", group: "video", shortLabel: "My Models", shortLabelKey: "tabShortLabels.videoModels" },
  { id: "video-discover", label: "Video Discover", labelKey: "tabs.videoDiscover", caption: "Browse video models", captionKey: "tabCaptions.videoDiscover", group: "video", shortLabel: "Discover", shortLabelKey: "tabShortLabels.videoDiscover" },
  { id: "video-studio", label: "Video Studio", labelKey: "tabs.videoStudio", caption: "Prompt, generate, and iterate", captionKey: "tabCaptions.videoStudio", group: "video", shortLabel: "Studio", shortLabelKey: "tabShortLabels.videoStudio" },
  { id: "video-gallery", label: "Video Gallery", labelKey: "tabs.videoGallery", caption: "Saved outputs and filters", captionKey: "tabCaptions.videoGallery", group: "video", shortLabel: "Gallery", shortLabelKey: "tabShortLabels.videoGallery" },
  { id: "server", label: "Server", labelKey: "tabs.server", caption: "OpenAI-compatible local API", captionKey: "tabCaptions.server" },
  { id: "benchmarks", label: "Benchmarks", labelKey: "tabs.benchmarks", caption: "Run a new benchmark", captionKey: "tabCaptions.benchmarks", group: "benchmarks", shortLabel: "Run", shortLabelKey: "tabShortLabels.benchmarks" },
  { id: "benchmark-history", label: "History", labelKey: "tabs.benchmarkHistory", caption: "Compare saved runs", captionKey: "tabCaptions.benchmarkHistory", group: "benchmarks", shortLabel: "History", shortLabelKey: "tabShortLabels.benchmarkHistory" },
  { id: "conversion", label: "Conversion", labelKey: "tabs.conversion", caption: "Convert models to MLX format", captionKey: "tabCaptions.conversion", group: "tools", shortLabel: "Conversion", shortLabelKey: "tabShortLabels.conversion" },
  { id: "finetuning", label: "Fine-Tuning", labelKey: "tabs.fineTuning", caption: "LoRA adapters and training", captionKey: "tabCaptions.fineTuning", group: "tools", shortLabel: "Fine-Tuning", shortLabelKey: "tabShortLabels.fineTuning" },
  { id: "prompt-library", label: "Prompts", labelKey: "tabs.promptLibrary", caption: "Reusable prompt templates", captionKey: "tabCaptions.promptLibrary", group: "tools", shortLabel: "Prompts", shortLabelKey: "tabShortLabels.promptLibrary" },
  { id: "plugins", label: "Plugins", labelKey: "tabs.plugins", caption: "Extensions and plugin system", captionKey: "tabCaptions.plugins", group: "tools", shortLabel: "Plugins", shortLabelKey: "tabShortLabels.plugins" },
  { id: "logs", label: "Logs", labelKey: "tabs.logs", caption: "Runtime events", captionKey: "tabCaptions.logs" },
  { id: "settings", label: "Settings", labelKey: "tabs.settings", caption: "Directories and defaults", captionKey: "tabCaptions.settings" },
];
