import { Panel } from "../../components/Panel";
import { ProgressRow } from "../../components/ProgressRow";
import { StatCard } from "../../components/StatCard";
import { GPUCard } from "../../components/GPUCard";
import { useI18n } from "../../hooks/useI18n";
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
  const { t, ti } = useI18n();
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
        title={t("dashboard.liveSystemStats", "实时系统状态")}
        subtitle={t("dashboard.liveSystemStats.subtitle", "从 Python sidecar 刷新，帮助桌面外壳给出硬件适配建议。")}
        className="span-2"
      >
        <div className="stat-grid">
          <StatCard
            label={t("dashboard.runtimeEngine", "运行时引擎")}
            value={runtime.engineLabel}
            hint={runtime.loadedModel ? runtime.loadedModel.name : t("dashboard.noModelLoaded", "未加载模型")}
          />
          <StatCard
            label={t("dashboard.inferenceActivity", "推理活动")}
            value={ti("dashboard.activeRequests", "{{count}} 个活跃", { count: activeReq })}
            hint={ti("dashboard.totalServedCount", "累计服务 {{count}} 次", { count: servedReq })}
          />
          <StatCard
            label={t("dashboard.warmPool", "预热池")}
            value={ti("dashboard.modelCount", "{{count}} 个模型", { count: warmModels.length })}
            hint={warmModels.length > 0 ? warmModels.map((w) => w.name).join(" · ") : t("dashboard.noWarmModels", "无预热模型")}
          />
          {diskFree !== undefined && diskTotal ? (
            <StatCard
              label={t("dashboard.modelDisk", "模型磁盘")}
              value={ti("dashboard.gbFreeValue", "{{gb}} GB 可用", { gb: number(diskFree, 2) })}
              hint={ti("dashboard.gbTotalValue", "{{gb}} GB 总计", { gb: number(diskTotal, 2) })}
            />
          ) : (
            <StatCard
              label={t("dashboard.spareHeadroom", "剩余空间")}
              value={`${number(system.spareHeadroomGb, 2)} GB`}
              hint={ti("dashboard.workingHeadroomValue", "{{percent}}% 工作余量", { percent: number(recommendation.headroomPercent, 0) })}
            />
          )}
        </div>
        <div className="panel-grid">
          <div className="stack">
            <ProgressRow
              label={t("dashboard.memoryInUse", "已用内存")}
              value={system.usedMemoryGb}
              max={system.totalMemoryGb}
              valueLabel={`${number(system.usedMemoryGb, 2)} GB / ${number(system.totalMemoryGb, 2)} GB`}
            />
            <ProgressRow
              label={t("dashboard.memoryPressure", "内存压力")}
              value={memPressure}
              valueLabel={`${number(memPressure, 0)}%${compressedGb > 0 ? ` · ${ti("dashboard.gbCompressedValue", "{{gb}} GB 已压缩", { gb: number(compressedGb, 2) })}` : ""}`}
            />
            {swapGb > 0.01 ? (
              <ProgressRow
                label={t("dashboard.swapUsage", "交换空间使用")}
                value={swapGb}
                max={Math.max(system.swapTotalGb ?? swapGb, swapGb, 0.01)}
                valueLabel={`${number(swapGb, 2)} GB${system.swapTotalGb ? ` / ${number(system.swapTotalGb, 2)} GB` : ""}`}
              />
            ) : null}
            <ProgressRow
              label={t("dashboard.cpuLoad", "CPU 负载")}
              value={system.cpuUtilizationPercent}
              valueLabel={`${number(system.cpuUtilizationPercent, 0)}%`}
            />
            <ProgressRow
              label={ti("dashboard.headroomFor", "{{model}} 的余量", { model: recommendation.targetModel })}
              value={recommendation.headroomPercent}
              valueLabel={`${recommendation.headroomPercent}%`}
            />
            {battery ? (
              <div className={`battery-card${battery.powerSource === "Battery" && battery.percent < 20 ? " battery-card--low" : ""}`}>
                <div className="battery-card-header">
                  <span className="eyebrow">{t("dashboard.power", "电源")}</span>
                  <span className={`badge ${battery.powerSource === "AC" ? "success" : battery.percent < 20 ? "warning" : "muted"}`}>
                    {battery.powerSource === "AC" ? (battery.charging ? t("dashboard.charging", "充电中") : t("dashboard.acPower", "交流电源")) : t("dashboard.onBattery", "使用电池")}
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
                    <small>{t("dashboard.batteryThrottleHint", "未接电源——推理可能因温度压力降速")}</small>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>
          <div className="data-table compact-table">
            <div className="table-row table-head">
              <span>{t("dashboard.process", "进程")}</span>
              <span>{t("dashboard.owner", "所有者")}</span>
              <span>{t("dashboard.memory", "内存")}</span>
              <span>{t("dashboard.cpu", "CPU")}</span>
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
                            {process.modelStatus === "active" ? t("dashboard.activeBadge", "活跃") : t("dashboard.warmBadge", "预热")}
                          </span>
                        ) : null}
                      </div>
                      {process.modelName ? <small className="process-model-name">{process.modelName}</small> : null}
                    </div>
                    <span><span className={`badge ${process.owner === "ChaosEngineAI" ? "accent" : "muted"}`}>{process.owner ?? t("dashboard.systemOwner", "系统")}</span></span>
                    <span>{number(process.memoryGb, 2)} GB</span>
                    <span>{number(process.cpuPercent, 0)}%</span>
                  </div>
                ))
              ) : (
                <div className="empty-state small-empty">
                  <p>{t("dashboard.noLocalProcesses", "未检测到活跃的本地 LLM 进程。")}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </Panel>

      <GPUCard backendOnline={backendOnline} visible={true} />

      <Panel title={t("dashboard.hardwareFit", "硬件适配")} subtitle={t("dashboard.hardwareFit.subtitle", "相对于此设备推荐目标配置的指导。")}>
        <div className="callout">
          <span className="badge accent">{t("dashboard.recommendedTarget", "推荐目标")}</span>
          <h3>{recommendation.title}</h3>
          <p>{recommendation.detail}</p>
        </div>
        <ProgressRow
          label={ti("dashboard.headroomFor", "{{model}} 的余量", { model: recommendation.targetModel })}
          value={recommendation.headroomPercent}
          valueLabel={`${recommendation.headroomPercent}%`}
        />
        <div className="callout quiet">
          <h3>{t("dashboard.currentRuntime", "当前运行时")}</h3>
          <p>
            {runtime.loadedModel
              ? ti("dashboard.currentRuntime.loaded", "{{model}} 已通过 {{engine}} 加载。", {
                  model: runtime.loadedModel.name,
                  engine: runtime.engineLabel,
                })
              : t("dashboard.currentRuntime.empty", "尚未加载模型。请在聊天中选择线程模型，或在在线模型中浏览更新的模型系列。")}
          </p>
        </div>
      </Panel>

      <Panel title={t("dashboard.activityFeed", "活动流")} subtitle={t("dashboard.activityFeed.subtitle", "无需翻日志即可查看的重要运行事件。")}>
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
