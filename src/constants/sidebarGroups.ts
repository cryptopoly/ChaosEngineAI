import type { SidebarGroupId, TabId } from "../types";

export interface SidebarGroup {
  id: SidebarGroupId;
  label: string;
  caption: string;
  defaultChild: TabId;
}

export const sidebarGroups: SidebarGroup[] = [
  { id: "models", label: "Models", caption: "Language models", defaultChild: "my-models" },
  { id: "images", label: "Images", caption: "Image generation", defaultChild: "image-models" },
  { id: "benchmarks", label: "Benchmarks", caption: "Performance tests", defaultChild: "benchmarks" },
  { id: "tools", label: "Tools", caption: "Conversion, fine-tuning, prompts, plugins", defaultChild: "conversion" },
];
