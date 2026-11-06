/**
 * Reference / start / end image picker for the Image and Video Studios.
 *
 * Two ways in: a file dialog (desktop app) and the Image Studio gallery. The backend takes absolute
 * paths on this machine, so a pick is a path, never an upload. The parent shows it only for models
 * that take an input image.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { ImageOutputArtifact } from "../types";

export interface InputImage {
  path: string;
  name: string;
  previewUrl?: string;
}

export interface InputImagePickerProps {
  label: string;
  hint?: string;
  images: InputImage[];
  max: number;
  onChange: (next: InputImage[]) => void;
  gallery: ImageOutputArtifact[];
  disabled?: boolean;
}

const GALLERY_LIMIT = 12;

export function fileNameOf(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

export function addInputImage(images: InputImage[], next: InputImage, max: number): InputImage[] {
  if (images.some((image) => image.path === next.path)) return images;
  return [...images, next].slice(-max);
}

export function InputImagePicker({ label, hint, images, max, onChange, gallery, disabled }: InputImagePickerProps) {
  const { t } = useTranslation("studio");
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [pickError, setPickError] = useState<string | null>(null);
  const full = images.length >= max;
  const choices = gallery.filter((item) => item.imagePath).slice(0, GALLERY_LIMIT);

  async function chooseFile() {
    setPickError(null);
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false,
        title: label,
        filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }],
      });
      if (typeof selected === "string" && selected) {
        onChange(addInputImage(images, { path: selected, name: fileNameOf(selected) }, max));
      }
    } catch {
      setPickError(t("inputImage.pickFailed", { defaultValue: "The file picker is not available here. Pick from the gallery instead." }));
    }
  }

  return (
    <div className="control-stack input-image-picker">
      <span className="eyebrow">{label}</span>
      {hint ? <p className="muted-text">{hint}</p> : null}
      {images.length > 0 ? (
        <ul className="input-image-list">
          {images.map((image) => (
            <li key={image.path} className="input-image-item">
              {image.previewUrl ? <img src={image.previewUrl} alt="" className="input-image-thumb" /> : null}
              <span className="input-image-name" title={image.path}>{image.name}</span>
              <button
                type="button"
                className="secondary-button"
                disabled={disabled}
                onClick={() => onChange(images.filter((item) => item.path !== image.path))}
              >
                {t("inputImage.remove", { defaultValue: "Remove" })}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <div className="image-pill-row">
        <button type="button" className="secondary-button" disabled={disabled || full} onClick={() => void chooseFile()}>
          {t("inputImage.chooseFile", { defaultValue: "Choose file" })}
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={disabled || full || choices.length === 0}
          onClick={() => setGalleryOpen((open) => !open)}
        >
          {t("inputImage.fromGallery", { defaultValue: "From gallery" })}
        </button>
      </div>
      {galleryOpen && !full ? (
        <div className="input-image-gallery">
          {choices.map((item) => (
            <button
              key={item.artifactId}
              type="button"
              className="input-image-choice"
              title={item.prompt}
              onClick={() => {
                onChange(addInputImage(images, { path: item.imagePath as string, name: fileNameOf(item.imagePath as string), previewUrl: item.previewUrl }, max));
                setGalleryOpen(false);
              }}
            >
              <img src={item.previewUrl} alt={item.prompt} className="input-image-thumb" />
            </button>
          ))}
        </div>
      ) : null}
      {pickError ? <p className="error-text">{pickError}</p> : null}
    </div>
  );
}
