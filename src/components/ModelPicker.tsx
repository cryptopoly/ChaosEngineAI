import { useEffect, useState } from "react";
import type { LibraryItem } from "../types";
import { fetchJson } from "../api";

export interface WeightFile {
  name: string;
  path: string;
  sizeGb: number;
  role: "main" | "mmproj" | string;
}

export interface ListWeightsResponse {
  path: string;
  format: "GGUF" | "MLX" | "Transformers" | "unknown" | string;
  files: WeightFile[];
  broken: boolean;
  brokenReason: string | null;
}

interface ModelPickerProps {
  open: boolean;
  title: string;
  library: LibraryItem[];
  filter?: (item: LibraryItem) => boolean;
  selectedPath: string | null;
  onSelect: (item: LibraryItem, resolvedFilePath?: string) => void;
  onClose: () => void;
}

export async function fetchWeightList(path: string): Promise<ListWeightsResponse> {
  return await fetchJson<ListWeightsResponse>(
    `/api/models/list-weights?path=${encodeURIComponent(path)}`,
  );
}

function formatSize(gb: number): string {
  if (!gb) return "";
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(gb * 1024).toFixed(0)} MB`;
}

export function ModelPicker({ open, title, library, filter, selectedPath, onSelect, onClose }: ModelPickerProps) {
  const [search, setSearch] = useState("");
  const [subPickItem, setSubPickItem] = useState<LibraryItem | null>(null);
  const [subPickLoading, setSubPickLoading] = useState(false);
  const [subPickError, setSubPickError] = useState<string | null>(null);
  const [subPickFiles, setSubPickFiles] = useState<WeightFile[]>([]);

  useEffect(() => {
    if (!open) {
      setSearch("");
      setSubPickItem(null);
      setSubPickFiles([]);
      setSubPickError(null);
    }
  }, [open]);

  if (!open) return null;

  const filtered = library
    .filter((item) => !filter || filter(item))
    .filter((item) => {
      if (!search) return true;
      const q = search.toLowerCase();
      return item.name.toLowerCase().includes(q) || item.format.toLowerCase().includes(q);
    });

  async function handleItemClick(item: LibraryItem) {
    if (item.broken) return;
    // For GGUF directory-style paths, check for sub-files.
    if (item.format?.toUpperCase() === "GGUF" || item.format?.toLowerCase().includes("gguf")) {
      setSubPickItem(item);
      setSubPickLoading(true);
      setSubPickError(null);
      setSubPickFiles([]);
      try {
        const resp = await fetchWeightList(item.path);
        if (resp.broken) {
          setSubPickError(resp.brokenReason ?? "Incomplete or broken model directory.");
          setSubPickLoading(false);
          return;
        }
        const mainFiles = resp.files.filter((f) => f.role !== "mmproj");
        if (mainFiles.length === 0) {
          setSubPickError("No .gguf weights present in this directory.");
          setSubPickLoading(false);
          return;
        }
        if (mainFiles.length === 1) {
          onSelect(item, mainFiles[0].path);
          onClose();
          return;
        }
        setSubPickFiles(resp.files);
        setSubPickLoading(false);
      } catch (err) {
        setSubPickError(err instanceof Error ? err.message : "Failed to list weight files.");
        setSubPickLoading(false);
      }
      return;
    }
    onSelect(item);
    onClose();
  }

  function handleSubPickSelect(file: WeightFile) {
    if (!subPickItem) return;
    onSelect(subPickItem, file.path);
    onClose();
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{subPickItem ? `Select weight file: ${subPickItem.name}` : title}</h3>
        </div>
        <div className="modal-body">
          {subPickItem ? (
            <>
              {subPickLoading ? <p className="muted-text">Scanning...</p> : null}
              {subPickError ? <p className="muted-text">{subPickError}</p> : null}
              {!subPickLoading && !subPickError ? (
                <div className="model-select-list">
                  {subPickFiles.map((file) => (
                    <button
                      key={file.path}
                      className="model-select-item"
                      type="button"
                      onClick={() => handleSubPickSelect(file)}
                    >
                      <div className="model-select-item-info">
                        <strong>{file.name}</strong>
                        <div className="model-select-item-meta">
                          <span>{formatSize(file.sizeGb)}</span>
                          {file.role === "mmproj" ? <span className="badge muted">vision projector</span> : null}
                          {file.role === "main" ? <span className="badge muted">main</span> : null}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : null}
              <div style={{ marginTop: 12 }}>
                <button className="secondary-button" type="button" onClick={() => { setSubPickItem(null); setSubPickFiles([]); setSubPickError(null); }}>
                  Back
                </button>
              </div>
            </>
          ) : (
            <>
              <input
                className="text-input"
                type="search"
                placeholder="Search local models..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                autoFocus
              />
              <div className="model-select-list">
                {filtered.map((item) => {
                  const isBroken = !!item.broken;
                  return (
                    <button
                      key={item.path}
                      className={`model-select-item${selectedPath === item.path ? " active" : ""}${isBroken ? " disabled" : ""}`}
                      type="button"
                      disabled={isBroken}
                      onClick={() => void handleItemClick(item)}
                      title={isBroken ? item.brokenReason ?? "Incomplete / broken" : undefined}
                    >
                      <div className="model-select-item-info">
                        <strong>{item.name}</strong>
                        <div className="model-select-item-meta">
                          <span>{item.format}</span>
                          {item.sizeGb ? <span>{item.sizeGb.toFixed(1)} GB</span> : null}
                          {item.maxContext ? (
                            <span>
                              {item.maxContext >= 1000 ? `${Math.round(item.maxContext / 1024)}K ctx` : `${item.maxContext} ctx`}
                            </span>
                          ) : null}
                          {isBroken ? <span className="badge warning">BROKEN</span> : null}
                        </div>
                        {isBroken && item.brokenReason ? (
                          <small className="muted-text">{item.brokenReason}</small>
                        ) : null}
                      </div>
                    </button>
                  );
                })}
                {filtered.length === 0 ? <p className="model-select-empty">No models match.</p> : null}
              </div>
            </>
          )}
        </div>
        <div className="modal-footer">
          <button className="secondary-button" type="button" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
