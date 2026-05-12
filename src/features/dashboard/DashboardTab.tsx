import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation("dashboard");
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
        title={t("liveStats.title")}
        subtitle={t("liveStats.subtitle")}
        className="span-2"
      >
        <div className="stat-grid">
          <StatCard
            label={t("liveStats.runtimeEngine")}
            value={runtime.engineLabel}
            hint={runtime.loadedModel ? runtime.loadedModel.name : t("liveStats.noModelLoaded")}
          />
          <StatCard
            label={t("liveStats.inferenceActivity")}
            value={t("liveStats.activeCount", { count: activeReq })}
            hint={t("liveStats.totalServed", { count: servedReq })}
          />
          <StatCard
            label={t("liveStats.warmPool")}
            value={t("liveStats.warmModelCount", { count: warmModels.length })}
            hint={warmModels.length > 0 ? warmModels.map((w) => w.name).join(" · ") : t("liveStats.noWarmModels")}
          />
          {diskFree !== undefined && diskTotal ? (
            <StatCard
              label={t("liveStats.modelDisk")}
              value={t("liveStats.diskFree", { value: number(diskFree, 2) })}
              hint={t("liveStats.diskTotal", { value: number(diskTotal, 2) })}
            />
          ) : (
            <StatCard
              label={t("liveStats.spareHeadroom")}
              value={t("liveStats.spareHeadroomValue", { value: number(system.spareHeadroomGb, 2) })}
              hint={t("liveStats.workingHeadroom", { value: number(recommendation.headroomPercent, 0) })}
            />
          )}
        </div>
        <div className="panel-grid">
          <div className="stack">
            <ProgressRow
              label={t("liveStats.memoryInUse")}
              value={system.usedMemoryGb}
              max={system.totalMemoryGb}
              valueLabel={t("liveStats.memoryUsedLabel", {
                used: number(system.usedMemoryGb, 2),
                total: number(system.totalMemoryGb, 2),
              })}
            />
            <ProgressRow
              label={t("liveStats.memoryPressure")}
              value={memPressure}
              valueLabel={
                compressedGb > 0
                  ? t("liveStats.memoryPressureCompressed", {
                      value: number(memPressure, 0),
                      compressed: number(compressedGb, 2),
                    })
                  : t("liveStats.memoryPressureLabel", { value: number(memPressure, 0) })
              }
            />
            {swapGb > 0.01 ? (
              <ProgressRow
                label={t("liveStats.swapUsage")}
                value={swapGb}
                max={Math.max(system.swapTotalGb ?? swapGb, swapGb, 0.01)}
                valueLabel={
                  system.swapTotalGb
                    ? t("liveStats.swapLabelWithTotal", {
                        used: number(swapGb, 2),
                        total: number(system.swapTotalGb, 2),
                      })
                    : t("liveStats.swapLabel", { used: number(swapGb, 2) })
                }
              />
            ) : null}
            <ProgressRow
              label={t("liveStats.cpuLoad")}
              value={system.cpuUtilizationPercent}
              valueLabel={t("liveStats.cpuLabel", { value: number(system.cpuUtilizationPercent, 0) })}
            />
            <ProgressRow
              label={t("liveStats.headroomFor", { target: recommendation.targetModel })}
              value={recommendation.headroomPercent}
              valueLabel={t("liveStats.headroomLabel", { value: recommendation.headroomPercent })}
            />
            {battery ? (
              <div className={`battery-card${battery.powerSource === "Battery" && battery.percent < 20 ? " battery-card--low" : ""}`}>
                <div className="battery-card-header">
                  <span className="eyebrow">{t("liveStats.battery.eyebrow")}</span>
                  <span className={`badge ${battery.powerSource === "AC" ? "success" : battery.percent < 20 ? "warning" : "muted"}`}>
                    {battery.powerSource === "AC"
                      ? battery.charging
                        ? t("liveStats.battery.charging")
                        : t("liveStats.battery.acPower")
                      : t("liveStats.battery.onBattery")}
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
                    <small>{t("liveStats.battery.lowNotice")}</small>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>
          <div className="data-table compact-table">
            <div className="table-row table-head">
              <span>{t("liveStats.processes.name")}</span>
              <span>{t("liveStats.processes.owner")}</span>
              <span>{t("liveStats.processes.memory")}</span>
              <span>{t("liveStats.processes.cpu")}</span>
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
                            {process.modelStatus === "active"
                              ? t("liveStats.processes.active")
                              : t("liveStats.processes.warm")}
                          </span>
                        ) : null}
                      </div>
                      {process.modelName ? <small className="process-model-name">{process.modelName}</small> : null}
                    </div>
                    <span>
                      <span className={`badge ${process.owner === "ChaosEngineAI" ? "accent" : "muted"}`}>
                        {process.owner ?? t("liveStats.processes.system")}
                      </span>
                    </span>
                    <span>{t("liveStats.processes.memoryGb", { value: number(process.memoryGb, 2) })}</span>
                    <span>{t("liveStats.processes.cpuPercent", { value: number(process.cpuPercent, 0) })}</span>
                  </div>
                ))
              ) : (
                <div className="empty-state small-empty">
                  <p>{t("liveStats.processes.empty")}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </Panel>

      <GPUCard backendOnline={backendOnline} visible={true} />

      <Panel title={t("hardwareFit.title")} subtitle={t("hardwareFit.subtitle")}>
        <div className="callout">
          <span className="badge accent">{t("hardwareFit.recommendedTarget")}</span>
          <h3>
            {recommendation.titleKey
              ? t(recommendation.titleKey, { ...(recommendation.payload ?? {}), defaultValue: recommendation.title })
              : recommendation.title}
          </h3>
          <p>
            {recommendation.detailKey
              ? t(recommendation.detailKey, { ...(recommendation.payload ?? {}), defaultValue: recommendation.detail })
              : recommendation.detail}
          </p>
        </div>
        <ProgressRow
          label={t("liveStats.headroomFor", { target: recommendation.targetModel })}
          value={recommendation.headroomPercent}
          valueLabel={t("liveStats.headroomLabel", { value: recommendation.headroomPercent })}
        />
        <div className="callout quiet">
          <h3>{t("hardwareFit.currentRuntime")}</h3>
          <p>
            {runtime.loadedModel
              ? t("hardwareFit.modelLoadedVia", {
                  model: runtime.loadedModel.name,
                  engine: runtime.engineLabel,
                })
              : t("hardwareFit.noModelHint")}
          </p>
        </div>
      </Panel>

      <Panel title={t("activityFeed.title")} subtitle={t("activityFeed.subtitle")}>
        <div className="list scrollable-list">
          {activity.map((item, idx) => {
            const localTitle = item.titleKey
              ? t(item.titleKey, { ...(item.payload ?? {}), defaultValue: item.title })
              : item.title;
            const localDetail = item.detailKey
              ? t(item.detailKey, { ...(item.payload ?? {}), defaultValue: item.detail })
              : item.detail;
            const localTime = item.time === "Now"
              ? t("activityFeed.now", { defaultValue: "Now" })
              : item.time;
            return (
              <div className="list-row" key={`${idx}-${item.title}`}>
                <div>
                  <strong>{localTitle}</strong>
                  <p>{localDetail}</p>
                </div>
                <span className="badge muted">{localTime}</span>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
