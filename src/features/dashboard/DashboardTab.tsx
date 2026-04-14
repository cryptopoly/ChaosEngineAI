import { Panel } from "../../components/Panel";
import { ProgressRow } from "../../components/ProgressRow";
import { StatCard } from "../../components/StatCard";
import { GPUCard } from "../../components/GPUCard";
import { number } from "../../utils/format";
import type { ActivityItem, Recommendation, RuntimeStatus, SystemStats } from "../../types";

export interface DashboardTabProps {
  system: SystemStats;
  recommendation: Recommendation;
  runtime: RuntimeStatus;
  activity: ActivityItem[];
  backendOnline: boolean;
}

export function DashboardTab({ system, recommendation, runtime, activity, backendOnline }: DashboardTabProps) {
  const warmModels = runtime.warmModels ?? [];
  const activeReq = runtime.activeRequests ?? 0;
  const servedReq = runtime.requestsServed ?? 0;
  const memPressure = system.memoryPressurePercent ?? 0;
  const compressedGb = system.compressedMemoryGb ?? 0;
  const swapGb = system.swapUsedGb ?? 0;
  const diskFree = system.diskFreeGb;
  const diskTotal = system.diskTotalGb;
  const battery = system.battery;

  return (
    <div className="content-grid">
      <Panel
        title="Live System Stats"
        subtitle="Refreshed from the Python sidecar so the desktop shell can make fit recommendations."
        className="span-2"
      >
        <div className="stat-grid">
          <StatCard
            label="Runtime engine"
            value={runtime.engineLabel}
            hint={runtime.loadedModel ? runtime.loadedModel.name : "No model loaded"}
          />
          <StatCard
            label="Inference activity"
            value={`${activeReq} active`}
            hint={`${servedReq} total served`}
          />
          <StatCard
            label="Warm pool"
            value={`${warmModels.length} model${warmModels.length === 1 ? "" : "s"}`}
            hint={warmModels.length > 0 ? warmModels.map((w) => w.name).join(" · ") : "No warm models"}
          />
          {diskFree !== undefined && diskTotal ? (
            <StatCard
              label="Model disk"
              value={`${number(diskFree, 2)} GB free`}
              hint={`${number(diskTotal, 2)} GB total`}
            />
          ) : (
            <StatCard
              label="Spare headroom"
              value={`${number(system.spareHeadroomGb, 2)} GB`}
              hint={`${number(recommendation.headroomPercent, 0)}% working headroom`}
            />
          )}
        </div>
        <div className="panel-grid">
          <div className="stack">
            <ProgressRow
              label="Memory in use"
              value={system.usedMemoryGb}
              max={system.totalMemoryGb}
              valueLabel={`${number(system.usedMemoryGb, 2)} GB / ${number(system.totalMemoryGb, 2)} GB`}
            />
            <ProgressRow
              label="Memory pressure"
              value={memPressure}
              valueLabel={`${number(memPressure, 0)}%${compressedGb > 0 ? ` · ${number(compressedGb, 2)} GB compressed` : ""}`}
            />
            {swapGb > 0.01 ? (
              <ProgressRow
                label="Swap usage"
                value={swapGb}
                max={Math.max(system.swapTotalGb ?? swapGb, swapGb, 0.01)}
                valueLabel={`${number(swapGb, 2)} GB${system.swapTotalGb ? ` / ${number(system.swapTotalGb, 2)} GB` : ""}`}
              />
            ) : null}
            <ProgressRow
              label="CPU load"
              value={system.cpuUtilizationPercent}
              valueLabel={`${number(system.cpuUtilizationPercent, 0)}%`}
            />
            <ProgressRow
              label={`Headroom for ${recommendation.targetModel}`}
              value={recommendation.headroomPercent}
              valueLabel={`${recommendation.headroomPercent}%`}
            />
            {battery ? (
              <div className={`battery-card${battery.powerSource === "Battery" && battery.percent < 20 ? " battery-card--low" : ""}`}>
                <div className="battery-card-header">
                  <span className="eyebrow">Power</span>
                  <span className={`badge ${battery.powerSource === "AC" ? "success" : battery.percent < 20 ? "warning" : "muted"}`}>
                    {battery.powerSource === "AC" ? (battery.charging ? "Charging" : "AC Power") : "On Battery"}
                  </span>
                </div>
                <div className="battery-card-bar">
                  <div
                    className="battery-card-fill"
                    style={{ width: `${battery.percent}%` }}
                  />
                </div>
                <div className="battery-card-footer">
                  <strong>{battery.percent}%</strong>
                  {battery.powerSource === "Battery" ? (
                    <small>Unplugged — inference may throttle on thermal pressure</small>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>
          <div className="data-table compact-table">
            <div className="table-row table-head">
              <span>Process</span>
              <span>Owner</span>
              <span>Memory</span>
              <span>CPU</span>
            </div>
            <div className="data-table-body">
              {system.runningLlmProcesses.length ? (
                system.runningLlmProcesses.map((process) => (
                  <div className="table-row" key={process.pid}>
                    <div className="process-name-cell">
                      <div className="process-name-line">
                        <strong>{process.name}</strong>
                        {process.modelStatus ? (
                          <span className={`badge ${process.modelStatus === "active" ? "success" : "muted"} process-status-badge`}>
                            {process.modelStatus === "active" ? "ACTIVE" : "WARM"}
                          </span>
                        ) : null}
                      </div>
                      {process.modelName ? <small className="process-model-name">{process.modelName}</small> : null}
                    </div>
                    <span><span className={`badge ${process.owner === "ChaosEngineAI" ? "accent" : "muted"}`}>{process.owner ?? "System"}</span></span>
                    <span>{number(process.memoryGb, 2)} GB</span>
                    <span>{number(process.cpuPercent, 0)}%</span>
                  </div>
                ))
              ) : (
                <div className="empty-state small-empty">
                  <p>No active local LLM processes were detected.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </Panel>

      <GPUCard backendOnline={backendOnline} visible={true} />

      <Panel title="Hardware Fit" subtitle="Guidance relative to the recommended target profile for this machine.">
        <div className="callout">
          <span className="badge accent">Recommended target</span>
          <h3>{recommendation.title}</h3>
          <p>{recommendation.detail}</p>
        </div>
        <ProgressRow
          label={`Headroom for ${recommendation.targetModel}`}
          value={recommendation.headroomPercent}
          valueLabel={`${recommendation.headroomPercent}%`}
        />
        <div className="callout quiet">
          <h3>Current runtime</h3>
          <p>
            {runtime.loadedModel
              ? `${runtime.loadedModel.name} loaded via ${runtime.engineLabel}.`
              : "No model is loaded yet. Pick a thread model in Chat or browse a newer family in Online Models."}
          </p>
        </div>
      </Panel>

      <Panel title="Activity Feed" subtitle="Operational events that should stay visible without digging into logs.">
        <div className="list scrollable-list">
          {activity.map((item, idx) => (
            <div className="list-row" key={`${idx}-${item.title}`}>
              <div>
                <strong>{item.title}</strong>
                <p>{item.detail}</p>
              </div>
              <span className="badge muted">{item.time}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
