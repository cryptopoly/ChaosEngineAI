import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import type { DiagnosticsSnapshot } from "../../api";

// The read-only part of Settings > Diagnostics: the header, the key/value
// sections and the backend log tail. Every label goes through t() (keys
// under diagnostics.json "panel"); names of files, packages and
// environment variables stay verbatim. The copied markdown is built
// separately in DiagnosticsPanel and stays in English for support.

export type DiagnosticsRow = [string, string];

export interface DiagnosticsSectionData {
  id: string;
  title: string;
  rows: DiagnosticsRow[];
}

const DASH = "—";

function fmtGb(value: number | null | undefined): string {
  return value != null ? `${value} GB` : DASH;
}

function fmtBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return DASH;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

/** The sections the panel shows, with labels in the language of ``t``. */
export function diagnosticsSections(snapshot: DiagnosticsSnapshot, t: TFunction): DiagnosticsSectionData[] {
  const present = (flag: boolean) => (flag
    ? t("panel.values.installed", { defaultValue: "installed" })
    : t("panel.values.missing", { defaultValue: "missing" }));
  const notSet = t("panel.values.notSet", { defaultValue: "not set" });
  const version = t("panel.rows.version", { defaultValue: "Version" });
  const { os, hardware, python, runtime, gpu, extras } = snapshot;

  const sections: DiagnosticsSectionData[] = [
    {
      id: "os",
      title: t("panel.sections.os", { defaultValue: "OS" }),
      rows: [
        [t("panel.rows.system", { defaultValue: "System" }), os.system as string],
        [t("panel.rows.release", { defaultValue: "Release" }), os.release as string],
        [version, os.version as string],
        [t("panel.rows.machine", { defaultValue: "Machine" }), os.machine as string],
        [t("panel.rows.platform", { defaultValue: "Platform" }), os.platform as string],
      ],
    },
    {
      id: "hardware",
      title: t("panel.sections.hardware", { defaultValue: "Hardware" }),
      rows: [
        [t("panel.rows.cpuLogical", { defaultValue: "CPU logical cores" }), String(hardware.cpu?.logicalCount ?? DASH)],
        [t("panel.rows.cpuPhysical", { defaultValue: "CPU physical cores" }), String(hardware.cpu?.physicalCount ?? DASH)],
        [t("panel.rows.memoryTotal", { defaultValue: "Memory total" }), fmtGb(hardware.memory.totalGb)],
        [t("panel.rows.memoryAvailable", { defaultValue: "Memory available" }), fmtGb(hardware.memory.availableGb)],
        [t("panel.rows.swap", { defaultValue: "Swap used / total" }), `${fmtGb(hardware.swap.usedGb)} / ${fmtGb(hardware.swap.totalGb)}`],
        [t("panel.rows.gpu", { defaultValue: "GPU" }), (hardware.gpu.gpuName as string) || DASH],
        [t("panel.rows.vramTotal", { defaultValue: "VRAM total" }), fmtGb(hardware.gpu.vramTotalGb as number | null)],
        [t("panel.rows.driver", { defaultValue: "Driver" }), (hardware.gpu.driverVersion as string) || DASH],
        [t("panel.rows.systemCuda", { defaultValue: "System CUDA (driver)" }), (hardware.gpu.systemCudaVersion as string) || DASH],
      ],
    },
    {
      id: "python",
      title: t("panel.sections.python", { defaultValue: "Python" }),
      rows: [
        [t("panel.rows.executable", { defaultValue: "Executable" }), python.executable],
        [version, python.version ?? DASH],
        [t("panel.rows.implementation", { defaultValue: "Implementation" }), python.implementation],
        [t("panel.rows.prefix", { defaultValue: "Prefix" }), python.prefix],
        [t("panel.rows.basePrefix", { defaultValue: "Base prefix" }), python.basePrefix],
        ["cwd", python.cwd ?? DASH],
      ],
    },
    {
      id: "runtime",
      title: t("panel.sections.runtime", { defaultValue: "Runtime" }),
      rows: [
        [t("panel.rows.engine", { defaultValue: "Engine" }), runtime.engineLabel ?? DASH],
        [
          t("panel.rows.loadedModel", { defaultValue: "Loaded model" }),
          (runtime.loadedModel?.name as string) ?? t("panel.values.none", { defaultValue: "none" }),
        ],
        [t("panel.rows.warmPool", { defaultValue: "Warm pool size" }), String(runtime.warmPoolCount ?? 0)],
        ["llama-server", runtime.llamaServerPath ?? notSet],
        ["llama-server-turbo", runtime.llamaServerTurboPath ?? notSet],
        ["llama-cli", runtime.llamaCliPath ?? notSet],
      ],
    },
    {
      id: "gpuPackages",
      title: t("panel.sections.gpuPackages", { defaultValue: "GPU packages" }),
      rows: [
        ["torch", present(gpu.torchFindSpec)],
        ["diffusers", present(gpu.diffusersFindSpec)],
        ["accelerate", present(gpu.accelerateFindSpec)],
        ["transformers", present(gpu.transformersFindSpec)],
        ["imageio", present(gpu.imageioFindSpec)],
        ["imageio-ffmpeg", present(gpu.ffmpegFindSpec)],
        ["sentencepiece", present(gpu.sentencepieceFindSpec)],
        ["tiktoken", present(gpu.tiktokenFindSpec)],
        ["protobuf", present(gpu.protobufFindSpec)],
        ["ftfy", present(gpu.ftfyFindSpec)],
      ],
    },
  ];
  if (gpu.torchSubprocess) {
    sections.push({
      id: "torchRuntime",
      title: t("panel.sections.torchRuntime", { defaultValue: "torch runtime" }),
      rows: Object.entries(gpu.torchSubprocess).map(([k, v]) => [k, String(v)]),
    });
  }
  sections.push(
    {
      id: "extras",
      title: t("panel.sections.extras", { defaultValue: "Extras (GPU install target)" }),
      rows: [
        [t("panel.rows.path", { defaultValue: "Path" }), extras.path],
        [
          t("panel.rows.exists", { defaultValue: "Exists" }),
          extras.exists ? t("panel.values.yes", { defaultValue: "yes" }) : t("panel.values.no", { defaultValue: "no" }),
        ],
        [t("panel.rows.size", { defaultValue: "Size" }), extras.sizeBytes != null ? fmtBytes(extras.sizeBytes) : DASH],
        [t("panel.rows.freeOnVolume", { defaultValue: "Free on volume" }), extras.freeBytes != null ? fmtBytes(extras.freeBytes) : DASH],
        [
          t("panel.rows.topLevelEntries", { defaultValue: "Top-level entries" }),
          extras.topLevelEntries.length
            ? extras.topLevelEntries.slice(0, 20).join(", ")
            : t("panel.values.empty", { defaultValue: "(empty)" }),
        ],
      ],
    },
    {
      id: "environment",
      title: t("panel.sections.environment", { defaultValue: "Environment" }),
      rows: Object.entries(snapshot.environment).map(
        ([k, v]) => [k, v ?? t("panel.values.unset", { defaultValue: "(unset)" })],
      ),
    },
  );
  return sections;
}

export function DiagnosticsSnapshotView({
  snapshot,
  lastFetchedAt,
}: {
  snapshot: DiagnosticsSnapshot;
  lastFetchedAt: number | null;
}) {
  const { t, i18n } = useTranslation("diagnostics");
  const timestamp = lastFetchedAt ? new Date(lastFetchedAt).toLocaleString(i18n.language) : DASH;
  return (
    <>
      <div className="diagnostics-header">
        <div>
          <strong>ChaosEngineAI v{snapshot.app.appVersion}</strong>
          <span className="muted-text" style={{ marginLeft: 8 }}>
            {t("panel.captured", { defaultValue: "captured {timestamp}", timestamp })}
          </span>
        </div>
      </div>
      {diagnosticsSections(snapshot, t).map((section) => (
        <details className="diagnostics-section" open key={section.id} data-section={section.id}>
          <summary className="diagnostics-section-title">{section.title}</summary>
          <table className="diagnostics-table">
            <tbody>
              {section.rows.map(([key, value], idx) => (
                <tr key={`${key}-${idx}`}>
                  <td className="diagnostics-key">{key}</td>
                  <td className="diagnostics-value">{value || DASH}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      ))}

      <details className="install-log-panel" style={{ marginTop: 16 }}>
        <summary className="install-log-summary">
          {t("panel.logTail", {
            defaultValue: "Backend log tail ({count, plural, one {# line} other {# lines}})",
            count: snapshot.logs.tailLines.length,
          })}
          {snapshot.logs.path ? <> · <code style={{ fontSize: "0.7rem" }}>{snapshot.logs.path}</code></> : null}
        </summary>
        <div className="install-log-body">
          <pre className="install-log-output" style={{ maxHeight: 420 }}>
            {snapshot.logs.tailLines.join("\n") || t("panel.values.noLog", { defaultValue: "(no log content)" })}
          </pre>
        </div>
      </details>
    </>
  );
}
