import { describe, expect, it } from "vitest";
import { groupedPlugins } from "../pluginsLogic";

const row = (id: string, type: string) => ({
  id, name: id, type, version: "0.1.0", author: "", description: "", builtin: true, enabled: true,
});

describe("groupedPlugins", () => {
  it("reads the map the route sends (no wrapper key)", () => {
    const grouped = groupedPlugins({ cache_strategy: [row("cache.native", "cache_strategy")], tool: [row("tool.calculator", "tool")] });
    expect(Object.keys(grouped)).toEqual(["cache_strategy", "tool"]);
    expect(grouped.tool[0].id).toBe("tool.calculator");
  });

  it("still reads a wrapped { plugins } answer", () => {
    const grouped = groupedPlugins({ plugins: { tool: [row("tool.calculator", "tool")] } });
    expect(grouped.tool).toHaveLength(1);
  });

  it("drops what is not a list of plugin rows", () => {
    expect(groupedPlugins(null)).toEqual({});
    expect(groupedPlugins([row("a", "tool")])).toEqual({});
    expect(groupedPlugins({ detail: "Not Found", tool: [null, { name: "no id" }, row("tool.ok", "tool")] })).toEqual({
      tool: [row("tool.ok", "tool")],
    });
  });
});
