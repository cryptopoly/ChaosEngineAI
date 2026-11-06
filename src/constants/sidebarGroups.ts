import type { SidebarGroupId, TabId } from "../types";

export interface SidebarGroup {
  id: SidebarGroupId;
  label: string;
  caption: string;
  defaultChild: TabId;
  /** Where the group opens once it is "ready" (a model installed, a project open). */
  readyChild?: TabId;
  /** Sidebar section the group is listed under. */
  section?: "make" | "run";
}

export const sidebarGroups: SidebarGroup[] = [
  { id: "chat", label: "Chat", caption: "Chat, compare, and language models", defaultChild: "chat", section: "run" },
  { id: "images", label: "Images", caption: "Image generation", defaultChild: "image-models", section: "make" },
  { id: "video", label: "Video", caption: "Video generation", defaultChild: "video-models", section: "make" },
  { id: "voice", label: "Voice", caption: "STT, TTS, and voice chat", defaultChild: "voice-studio", section: "make" },
  { id: "music", label: "Music", caption: "Music and sound effects", defaultChild: "music-models", readyChild: "music-studio", section: "make" },
  { id: "storyboard", label: "Storyboard", caption: "Shots, takes and mixing", defaultChild: "story-projects", readyChild: "story-board", section: "make" },
  { id: "benchmarks", label: "Benchmarks", caption: "Performance tests", defaultChild: "benchmarks", section: "run" },
  { id: "tools", label: "Tools", caption: "Conversion, fine-tuning, prompts, plugins", defaultChild: "conversion", section: "run" },
];
