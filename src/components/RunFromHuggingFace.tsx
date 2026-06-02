import { useState } from "react";
import { useTranslation } from "react-i18next";
import { downloadModel, loadModel, resolveHfModel, type ResolvedHfModel } from "../api";

/**
 * "Run from Hugging Face" (#5): paste any GGUF / MLX repo and run it
 * without a curated catalog row. Resolves the repo's own metadata
 * (backend, GGUF file, context, capabilities) and loads with
 * ``canonicalRepo=<repo>`` so it never fuzzy-matches the wrong catalog
 * entry (the FU-041 failure mode).
 *
 * Self-contained: talks to the API directly so it can be dropped into
 * the Discover tab without prop threading. Download is fire-and-forget
 * (the existing My Models download UI tracks progress); Load surfaces a
 * "download first" hint when weights aren't present yet.
 */
export function RunFromHuggingFace() {
  const { t } = useTranslation("common");
  const [repo, setRepo] = useState("");
  const [resolving, setResolving] = useState(false);
  const [resolved, setResolved] = useState<ResolvedHfModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"download" | "load" | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const runnable = resolved != null && (resolved.backend === "llama.cpp" || resolved.backend === "mlx");

  async function handleResolve() {
    const trimmed = repo.trim();
    if (!trimmed) return;
    setResolving(true);
    setError(null);
    setResolved(null);
    setNotice(null);
    try {
      setResolved(await resolveHfModel(trimmed));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("runHf.resolveFailed", { defaultValue: "Could not resolve that repo." }));
    } finally {
      setResolving(false);
    }
  }

  async function handleDownload() {
    if (!resolved) return;
    setBusy("download");
    setNotice(null);
    setError(null);
    try {
      await downloadModel(resolved.repo);
      setNotice(t("runHf.downloadStarted", { defaultValue: "Download started — track progress in My Models, then Load." }));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("runHf.downloadFailed", { defaultValue: "Download failed." }));
    } finally {
      setBusy(null);
    }
  }

  async function handleLoad() {
    if (!resolved) return;
    setBusy("load");
    setNotice(null);
    setError(null);
    try {
      await loadModel({
        modelRef: resolved.repo,
        modelName: resolved.label,
        canonicalRepo: resolved.repo, // bypasses catalog fuzzy-match (FU-041)
        backend: resolved.backend,
        contextTokens: resolved.contextTokens,
        source: "custom",
      });
      setNotice(t("runHf.loaded", { label: resolved.label, defaultValue: "Loaded {label}." }));
    } catch (err) {
      const msg = err instanceof Error ? err.message : t("runHf.loadFailed", { defaultValue: "Load failed." });
      setError(msg);
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="run-from-hf" aria-label={t("runHf.title", { defaultValue: "Run from Hugging Face" })}>
      <div className="run-from-hf-head">
        <strong>{t("runHf.title", { defaultValue: "Run from Hugging Face" })}</strong>
        <small className="muted-text">{t("runHf.subtitle", { defaultValue: "Paste any GGUF or MLX repo (owner/name or URL) to run it without a catalog entry." })}</small>
      </div>
      <div className="run-from-hf-input-row">
        <input
          className="text-input"
          type="text"
          placeholder="bartowski/Some-Model-GGUF"
          value={repo}
          disabled={resolving}
          onChange={(e) => setRepo(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing) void handleResolve();
          }}
        />
        <button className="secondary-button" type="button" disabled={resolving || !repo.trim()} onClick={() => void handleResolve()}>
          {resolving ? t("runHf.resolving", { defaultValue: "Resolving…" }) : t("runHf.resolve", { defaultValue: "Resolve" })}
        </button>
      </div>

      {error ? (
        <div className="callout error run-from-hf-msg">
          <p>{error}</p>
        </div>
      ) : null}

      {resolved ? (
        <div className="run-from-hf-card">
          <div className="run-from-hf-card-head">
            <span className="badge muted">{resolved.backend}</span>
            <strong>{resolved.label}</strong>
            {resolved.capabilities.vision ? <span className="badge muted">vision</span> : null}
          </div>
          <p className="mono-text muted-text">
            {resolved.ggufFile ? `${resolved.ggufFile} · ` : ""}
            {t("runHf.ctx", { ctx: resolved.contextTokens, defaultValue: "{ctx} ctx" })}
            {resolved.totalSizeGb > 0 ? ` · ${resolved.totalSizeGb} GB` : ""}
          </p>
          {resolved.warnings.length > 0 ? (
            <div className="callout warning run-from-hf-msg">
              {resolved.warnings.map((w, i) => (
                <p key={i}>{w}</p>
              ))}
            </div>
          ) : null}
          <div className="button-row">
            <button className="secondary-button" type="button" disabled={busy !== null} onClick={() => void handleDownload()}>
              {busy === "download" ? t("runHf.downloading", { defaultValue: "Starting…" }) : t("runHf.download", { defaultValue: "Download" })}
            </button>
            <button
              className="primary-button"
              type="button"
              disabled={busy !== null || !runnable}
              title={!runnable ? t("runHf.notRunnable", { defaultValue: "This repo can't run directly on this platform — see the note above." }) : undefined}
              onClick={() => void handleLoad()}
            >
              {busy === "load" ? t("runHf.loading", { defaultValue: "Loading…" }) : t("runHf.load", { defaultValue: "Load" })}
            </button>
          </div>
          {notice ? <p className="muted-text run-from-hf-msg">{notice}</p> : null}
        </div>
      ) : null}
    </section>
  );
}
