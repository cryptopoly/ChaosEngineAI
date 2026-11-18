export interface PluginInfo {
  id: string;
  name: string;
  type: string;
  version: string;
  author: string;
  description: string;
  builtin: boolean;
  enabled: boolean;
}

/**
 * `GET /api/plugins` answers with the type-to-list map itself (`{"tool": [...], "cache_strategy": [...]}`).
 * The page used to read a `plugins` key that the route never sent, so it always showed "No plugins found".
 * Both shapes are accepted; anything that is not a list of plugin rows is dropped.
 */
export function groupedPlugins(data: unknown): Record<string, PluginInfo[]> {
  if (!data || typeof data !== "object" || Array.isArray(data)) return {};
  const record = data as Record<string, unknown>;
  const source =
    record.plugins && typeof record.plugins === "object" && !Array.isArray(record.plugins)
      ? (record.plugins as Record<string, unknown>)
      : record;
  const grouped: Record<string, PluginInfo[]> = {};
  for (const [type, list] of Object.entries(source)) {
    if (!Array.isArray(list)) continue;
    const rows = list.filter(
      (row): row is PluginInfo => Boolean(row) && typeof row === "object" && typeof (row as PluginInfo).id === "string",
    );
    if (rows.length) grouped[type] = rows;
  }
  return grouped;
}
