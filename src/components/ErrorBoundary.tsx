import { Component, type ErrorInfo, type ReactNode } from "react";
import { i18n } from "../i18n";

/**
 * FU-042: i18n helper for the class component below.  We can't use the
 * ``useTranslation`` hook in a React class, so we go through the
 * i18next instance directly.  Calls are safe even before
 * ``initI18n`` resolves — i18next returns the ``defaultValue`` when
 * the catalog isn't loaded yet, which keeps the boundary functional
 * during early startup crashes.
 */
function bt(key: string, defaultValue: string, vars?: Record<string, unknown>): string {
  return i18n.t(key, { ns: "errors", defaultValue, ...(vars ?? {}) });
}

/**
 * FU-037 (2026-05-10): per-tab React error boundary.
 *
 * Before this landed, any uncaught render or effect exception inside a
 * tab tore down the entire ``<main>`` content frame and left the user
 * staring at a blank screen with no way back except a full webview
 * reload (which dumps them to the Dashboard and crashes again the
 * moment they navigate back). The blank-screen path was reported
 * after a tool-call in the Chat tab; the actual stack trace lived
 * only in the webview console, which is unreachable in our release
 * builds.
 *
 * The boundary captures errors per tab so:
 *
 *  - A crash in Chat no longer blanks Dashboard, HTML Challenge, etc.
 *  - The user sees the error message, the component stack, and a
 *    "Copy details" button that loads enough into the clipboard to
 *    file a useful bug report without devtools.
 *  - A "Try again" button resets the boundary's local state so a
 *    transient render error (e.g. stale streaming state from an
 *    aborted tool call) can recover without quitting the app.
 *  - Switching to another tab unmounts the boundary entirely
 *    (we ``key`` it by tab id at the call site), so navigation
 *    is its own recovery path.
 *
 * Surfacing the log paths inline matches the "give the user enough
 * data to act" rule in CLAUDE.md — frontend errors go to the webview
 * console (now reachable via the right-click → Inspect Element entry
 * we enable in the Cargo ``devtools`` feature in the same FU), and
 * backend errors land in the rolling buffer the diagnostics tab
 * exposes.
 */
export interface ErrorBoundaryProps {
  /** Short noun-phrase used in the headline, e.g. ``"Chat"``. */
  scope: string;
  children: ReactNode;
  /**
   * Optional callback invoked on every caught error. Useful for
   * forwarding to a remote logger or toast surface. The boundary
   * still owns the fallback UI either way.
   */
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  error: Error | null;
  componentStack: string | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null, componentStack: null };

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ componentStack: info.componentStack ?? null });
    // eslint-disable-next-line no-console -- intentional: console is
    // the only frontend log sink in release builds, and we explicitly
    // want this to land there for the devtools "Console" tab.
    console.error(`[ErrorBoundary:${this.props.scope}]`, error, info.componentStack);
    this.props.onError?.(error, info);
  }

  reset = (): void => {
    this.setState({ error: null, componentStack: null });
  };

  copyDetails = (): void => {
    const { error, componentStack } = this.state;
    if (!error) return;
    const payload = [
      `ChaosEngineAI ErrorBoundary report — scope: ${this.props.scope}`,
      `When: ${new Date().toISOString()}`,
      `User agent: ${typeof navigator !== "undefined" ? navigator.userAgent : "n/a"}`,
      "",
      `Error: ${error.name}: ${error.message}`,
      "",
      "JS stack:",
      error.stack ?? "(no stack)",
      "",
      "Component stack:",
      componentStack ?? "(no component stack)",
    ].join("\n");
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(payload).catch(() => {
        // Clipboard can reject in non-secure contexts; ignore and let
        // the user copy from the on-screen <pre> fallback below.
      });
    }
  };

  render(): ReactNode {
    const { error, componentStack } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="error-boundary" role="alert">
        <div className="error-boundary__head">
          <strong>
            {bt("boundary.scopeCrashed", "{{scope}} crashed", { scope: this.props.scope })}
          </strong>
          <span className="error-boundary__sub">{error.name}: {error.message}</span>
        </div>
        <div className="error-boundary__actions">
          <button className="primary-button" type="button" onClick={this.reset}>
            {bt("boundary.tryAgain", "Try again")}
          </button>
          <button className="secondary-button" type="button" onClick={this.copyDetails}>
            {bt("boundary.copyDetails", "Copy details")}
          </button>
        </div>
        <details className="error-boundary__details">
          <summary>{bt("boundary.stackTrace", "Stack trace")}</summary>
          <pre className="error-boundary__stack">
            {error.stack ?? bt("boundary.noJsStack", "(no JS stack captured)")}
          </pre>
          {componentStack ? (
            <>
              <strong>{bt("boundary.componentStack", "Component stack")}</strong>
              <pre className="error-boundary__stack">{componentStack}</pre>
            </>
          ) : null}
        </details>
        <p className="error-boundary__hint">
          {bt(
            "boundary.hint",
            "Frontend errors also appear in the webview console (right-click → Inspect Element in release builds, or run the app with npm run dev). Backend logs are visible in the Diagnostics tab.",
          )}
        </p>
      </div>
    );
  }
}
