import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { initI18n } from "./i18n";
import "katex/dist/katex.min.css";
import "./styles.css";

// FU-042: bootstrap i18n *before* the first React render so that the
// initial paint already uses the negotiated locale.  The persisted
// `settings.locale` is hydrated lazily inside `App` via the existing
// settings fetch — when it differs from the OS / browser-detected
// default, the `App` effect calls `changeLocale(...)` and i18next
// re-renders.  This boot-time `initI18n` only seeds the navigator/OS
// detection so we never flash English on a non-en machine.
void initI18n({
  debug: import.meta.env.DEV,
}).finally(() => {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
});
