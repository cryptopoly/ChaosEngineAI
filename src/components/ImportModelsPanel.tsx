import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { importModel, scanImportableModels, type ImportableModel, type ImportScanResult } from "../api";

/**
 * "Import existing models" (#4): surfaces models already on disk in
 * another local-AI app's store and registers them by reference
 * (symlink, no re-download). Imported models then appear in My Models
 * and load like any other. Each group is labelled with the folder it
 * was found in, so the panel never names the other app.
 *
 * Self-contained: scans on mount (read-only) and hides itself entirely
 * when neither store is present, so it adds no clutter for users who
 * have no other local-AI app installed.
 */
export function ImportModelsPanel() {
  const { t } = useTranslation("common");
  const [scan, setScan] = useState<ImportScanResult | null>(null);
  const [scanning, setScanning] = useState(true);
  const [importingPath, setImportingPath] = useState<string | null>(null);
  const [importedPaths, setImportedPaths] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    scanImportableModels()
      .then((result) => {
        if (!cancelled) setScan(result);
      })
      .catch(() => {
        /* read-only discovery; a failure just hides the panel */
      })
      .finally(() => {
        if (!cancelled) setScanning(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleImport(model: ImportableModel) {
    setImportingPath(model.path);
    setError(null);
    try {
      await importModel({ source: model.source, path: model.path, name: model.name, repo: model.repo });
      setImportedPaths((prev) => new Set(prev).add(model.path));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("importModels.failed", { defaultValue: "Import failed." }));
    } finally {
      setImportingPath(null);
    }
  }

  if (scanning || !scan) return null;

  const hasAny = scan.ollama.models.length > 0 || scan.lmstudio.models.length > 0;
  if (!scan.ollama.available && !scan.lmstudio.available) return null;
  if (!hasAny) return null;

  function storeLabel(path: string | null): string {
    return path
      ? t("importModels.foundIn", { defaultValue: "Found in {path}", path })
      : t("importModels.foundElsewhere", { defaultValue: "Found on this computer" });
  }

  function renderGroup(label: string, models: ImportableModel[]) {
    if (models.length === 0) return null;
    return (
      <div className="import-models-group">
        <small className="muted-text">{label}</small>
        {models.map((m) => {
          const done = importedPaths.has(m.path);
          return (
            <div className="import-models-row" key={`${m.source}:${m.path}`}>
              <div className="import-models-meta">
                <span className="mono-text">{m.name}</span>
                {m.sizeGb > 0 ? <small className="muted-text">{m.sizeGb} GB</small> : null}
              </div>
              {done ? (
                <span className="badge success">{t("importModels.imported", { defaultValue: "Imported" })}</span>
              ) : (
                <button
                  type="button"
                  className="secondary-button"
                  disabled={importingPath !== null}
                  onClick={() => void handleImport(m)}
                >
                  {importingPath === m.path
                    ? t("importModels.importing", { defaultValue: "Importing…" })
                    : t("importModels.import", { defaultValue: "Import" })}
                </button>
              )}
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <section className="import-models-panel" aria-label={t("importModels.title", { defaultValue: "Import existing models" })}>
      <div className="import-models-head">
        <strong>{t("importModels.title", { defaultValue: "Import existing models" })}</strong>
        <small className="muted-text">{t("importModels.subtitle", { defaultValue: "Found models in another local app's store. Import links them in place — no re-download." })}</small>
      </div>
      {error ? (
        <div className="callout error import-models-msg">
          <p>{error}</p>
        </div>
      ) : null}
      {renderGroup(storeLabel(scan.ollama.dir), scan.ollama.models)}
      {renderGroup(storeLabel(scan.lmstudio.dirs[0] ?? null), scan.lmstudio.models)}
    </section>
  );
}
