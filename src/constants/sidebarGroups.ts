import type { SidebarGroupId, TabId } from "../types";

export interface SidebarGroup {
  id: SidebarGroupId;
  label: string;
  caption: string;
  defaultChild: TabId;
  /** Where the group opens once it is "ready" (a model installed). */
  readyChild?: TabId;
  /** Sidebar section the group is listed under. */
  section?: "make" | "run";
}

export const sidebarGroups: SidebarGroup[] = [
  { id: "chat", label: "Chat", caption: "Chat, compare, and language models", defaultChild: "chat" },
  { id: "images", label: "Images", caption: "Image generation", defaultChild: "image-models", section: "make" },
  { id: "video", label: "Video", caption: "Video generation", defaultChild: "video-models", section: "make" },
  { id: "voice", label: "Voice", caption: "Dictation, speech and voice chat", defaultChild: "voice-models", readyChild: "voice-studio", section: "make" },
  { id: "music", label: "Music and Audio", caption: "Music and sound effects", defaultChild: "music-models", readyChild: "music-studio", section: "make" },
  { id: "benchmarks", label: "Benchmarks", caption: "Performance tests", defaultChild: "benchmarks", section: "run" },
  { id: "tools", label: "Tools", caption: "Conversion, fine-tuning, prompts, plugins", defaultChild: "conversion", section: "run" },
];
