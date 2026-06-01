import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  getEmbeddingModelInstallStatus,
  getRagStatus,
  startEmbeddingModelInstall,
  type EmbeddingInstallJobState,
  type RagStatus,
} from "../api";

/**
 * Shows whether RAG retrieval is running in semantic ("vector") or
 * keyword ("lexical") mode, and offers a one-click download of the
 * recommended embedding model when only the lexical fallback is wired.
 *
 * Self-contained: fetches its own status on mount so it can be dropped
 * next to the session-documents chips without threading props through
 * ChatHeader. A render is cheap and only happens when documents exist.
 */
export function RagStatusBadge() {
  const { t } = useTranslation("common");
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [installing, setInstalling] = useState(false);
  const [job, setJob] = useState<EmbeddingInstallJobState | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    getRagStatus()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {
        /* status is best-effort; a failure just hides the badge */
      });
    return () => {
      cancelled = true;
      if (pollRef.current != null) window.clearInterval(pollRef.current);
    };
  }, []);

  async function handleEnable() {
    setInstalling(true);
    try {
      const initial = await startEmbeddingModelInstall();
      setJob(initial);
      pollRef.current = window.setInterval(async () => {
        try {
          const next = await getEmbeddingModelInstallStatus();
          setJob(next);
          if (next.done) {
            if (pollRef.current != null) window.clearInterval(pollRef.current);
            pollRef.current = null;
            setInstalling(false);
            // Re-read readiness — flips the badge to "vector" on success.
            const refreshed = await getRagStatus();
            setStatus(refreshed);
          }
        } catch {
          if (pollRef.current != null) window.clearInterval(pollRef.current);
          pollRef.current = null;
          setInstalling(false);
        }
      }, 1500);
    } catch {
      setInstalling(false);
    }
  }

  if (!status) return null;

  if (status.mode === "vector") {
    return (
      <span className="rag-status-badge" title={t("rag.vectorTooltip", { defaultValue: "Documents are searched by meaning using a local embedding model." })}>
        <span className="badge success">{t("rag.semantic", { defaultValue: "Semantic search" })}</span>
      </span>
    );
  }

  // Lexical mode. Offer the upgrade only when the binary is present —
  // installing the model alone can't enable vectors without it.
  const canInstall = status.binaryAvailable && !status.modelAvailable;

  return (
    <span className="rag-status-badge">
      <span
        className="badge muted"
        title={
          status.binaryAvailable
            ? t("rag.lexicalTooltip", { defaultValue: "Documents are searched by keyword (TF-IDF + BM25). Enable semantic search for better recall." })
            : t("rag.binaryMissingTooltip", { defaultValue: "Keyword search only — the llama-embedding binary was not found, so semantic search is unavailable on this install." })
        }
      >
        {t("rag.keyword", { defaultValue: "Keyword search" })}
      </span>
      {canInstall ? (
        installing ? (
          <small className="muted-text">
            {job?.phase === "verifying"
              ? t("rag.verifying", { defaultValue: "Verifying…" })
              : t("rag.downloading", { sizeLabel: status.recommended.sizeLabel, defaultValue: "Downloading {sizeLabel}…" })}
          </small>
        ) : (
          <button type="button" className="secondary-button rag-enable-button" onClick={() => void handleEnable()}>
            {t("rag.enable", { defaultValue: "Enable semantic search" })}
          </button>
        )
      ) : null}
      {job?.phase === "error" ? (
        <small className="muted-text" title={job.error ?? undefined}>
          {t("rag.installFailed", { defaultValue: "Install failed" })}
        </small>
      ) : null}
    </span>
  );
}
