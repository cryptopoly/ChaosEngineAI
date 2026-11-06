import type { ComponentType } from "react";
import type { TabId } from "../types";
import type { AppTabContext } from "./context";
import { ChatModelsView, ConversionView } from "./ModelsTabs";
import { ImageModelsView, ImageStudioView, ImageGalleryView } from "./ImageTabs";
import { VideoModelsView, VideoStudioView, VideoGalleryView } from "./VideoTabs";
import { VoiceStudioView, VoiceModelsView, VoiceLibraryView } from "./VoiceTabs";
import { MusicModelsView, MusicStudioView, MusicLibraryView } from "./MusicTabs";
import { StoryProjectsView, StoryBoardView, StoryMixerView } from "./StoryboardTabs";
import { isHomeStudioTab, HOME_STUDIO_ENABLED } from "../constants/features";
import { ChatView, CompareChatView, HtmlChallengeView } from "./ChatTabs";
import { BenchmarkRunView, BenchmarkHistoryView } from "./BenchmarkTabs";
import {
  DashboardView,
  ServerView,
  LogsView,
  SettingsView,
  FineTuningView,
  PromptLibraryView,
  PluginsView,
} from "./SystemTabs";

type TabView = ComponentType<{ ctx: AppTabContext }>;

/** The view each tab id renders. Any id not listed falls back to the dashboard. */
const TAB_VIEWS: Partial<Record<TabId, TabView>> = {
  "chat-models": ChatModelsView,
  "image-models": ImageModelsView,
  "image-studio": ImageStudioView,
  "image-gallery": ImageGalleryView,
  "video-models": VideoModelsView,
  "video-studio": VideoStudioView,
  "video-gallery": VideoGalleryView,
  "voice-studio": VoiceStudioView,
  "voice-models": VoiceModelsView,
  "voice-gallery": VoiceLibraryView,
  "music-models": MusicModelsView,
  "music-studio": MusicStudioView,
  "music-library": MusicLibraryView,
  "story-projects": StoryProjectsView,
  "story-board": StoryBoardView,
  "story-mixer": StoryMixerView,
  conversion: ConversionView,
  chat: ChatView,
  "chat-compare": CompareChatView,
  "html-challenge": HtmlChallengeView,
  server: ServerView,
  benchmarks: BenchmarkRunView,
  "benchmark-history": BenchmarkHistoryView,
  finetuning: FineTuningView,
  "prompt-library": PromptLibraryView,
  plugins: PluginsView,
  logs: LogsView,
  settings: SettingsView,
};

/** Music and Storyboard ids render the dashboard while the feature flag is off. */
export function tabViewFor(tab: TabId, homeStudioEnabled: boolean = HOME_STUDIO_ENABLED): TabView {
  if (!homeStudioEnabled && isHomeStudioTab(tab)) return DashboardView;
  return TAB_VIEWS[tab] ?? DashboardView;
}
