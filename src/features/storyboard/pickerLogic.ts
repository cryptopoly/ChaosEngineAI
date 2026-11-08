/**
 * Pure rules for the Cast picture picker (D-65): what each source offers (the Image gallery, this project's
 * shot stills), which dropped or pasted files are pictures, the four-picture limit and the small-picture
 * note. No React here so the rules are unit-tested directly.
 */
import type { ImageOutputArtifact, StoryboardProject, StoryboardReference } from "../../types";

/** Pictures per character or place (front, side, detail...). Mirrors MAX_REF_IMAGES in models.py. */
export const MAX_PICTURES = 4;
/** A picture with a shorter side than this gets the "results may drift" note. */
export const SMALL_PICTURE_PX = 512;
/** Gallery and still grids show this many at a time; "Show more" adds as many again. */
export const PICKER_GRID_LIMIT = 24;

export type PictureSource = "file" | "gallery" | "shot" | "path";

export const PICTURE_MIME = ["image/png", "image/jpeg", "image/webp"] as const;
const PICTURE_EXT = /\.(png|jpe?g|webp)$/i;

export function canAddPicture(reference: Pick<StoryboardReference, "images">): boolean {
  return reference.images.length < MAX_PICTURES;
}

/** True for a file the picker takes: a PNG, JPEG or WebP by type, or by name when the type is empty. */
export function isPictureFile(file: Pick<File, "type" | "name">): boolean {
  if (file.type) return (PICTURE_MIME as readonly string[]).includes(file.type);
  return PICTURE_EXT.test(file.name);
}

/** The pictures in a drop or a paste, in order, at most `room` of them. */
export function picturesFrom(files: ArrayLike<File> | null | undefined, room: number): File[] {
  if (!files || room <= 0) return [];
  return Array.from(files).filter(isPictureFile).slice(0, room);
}

export function isSmallPicture(width: number, height: number): boolean {
  return width > 0 && height > 0 && Math.min(width, height) < SMALL_PICTURE_PX;
}

export interface GalleryChoice {
  id: string;
  path: string;
  previewUrl: string;
  label: string;
}

/**
 * Saved images the picker can use, newest first as the gallery keeps them. Images made for this project come
 * first; the query matches the prompt and the model name.
 */
export function galleryChoices(
  gallery: readonly ImageOutputArtifact[],
  query: string,
  projectId: string | null,
  limit = PICKER_GRID_LIMIT,
): GalleryChoice[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const usable = gallery.filter((item) => item.imagePath && item.previewUrl);
  const matching = usable.filter((item) => {
    const text = `${item.prompt ?? ""} ${item.modelName ?? ""}`.toLowerCase();
    return words.every((word) => text.includes(word));
  });
  const mine = projectId ? matching.filter((item) => item.projectId === projectId) : [];
  const rest = matching.filter((item) => !mine.includes(item));
  return [...mine, ...rest].slice(0, limit).map((item) => ({
    id: item.artifactId,
    path: item.imagePath as string,
    previewUrl: item.previewUrl,
    label: item.prompt || item.modelName || item.artifactId,
  }));
}

export interface ShotStillChoice {
  key: string;
  /** The project-relative still; the backend copies it into the cast folder. */
  path: string;
  shotNumber: number;
  takeNumber: number;
  active: boolean;
  takeId: string;
}

/** Every still of this project's takes, shot by shot, the active take of each shot first. */
export function shotStillChoices(project: Pick<StoryboardProject, "shots">, limit = PICKER_GRID_LIMIT): ShotStillChoice[] {
  const choices: ShotStillChoice[] = [];
  project.shots.forEach((shot, index) => {
    const rows = shot.takes
      .map((take, takeIndex) => ({ take, takeIndex }))
      .filter(({ take }) => Boolean(take.still));
    rows.sort((a, b) => Number(b.take.id === shot.activeTake) - Number(a.take.id === shot.activeTake));
    for (const { take, takeIndex } of rows) {
      choices.push({
        key: `${shot.id}:${take.id}`,
        path: take.still as string,
        shotNumber: index + 1,
        takeNumber: takeIndex + 1,
        active: take.id === shot.activeTake,
        takeId: take.id,
      });
    }
  });
  return choices.slice(0, limit);
}

/** A name for a character made from a dropped file: the file name without its extension, tidied. */
export function nameFromFile(name: string): string {
  return name
    .replace(/\.[^.]+$/, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 200);
}

/** How many shots use a reference. */
export function usedByCount(project: Pick<StoryboardProject, "shots">, refId: string): number {
  return project.shots.filter((shot) => shot.referenceImages.includes(refId)).length;
}
