import { describe, expect, it } from "vitest";
import { agentConfigJson, agentConfigLine } from "./AgentAccessPanel";

const access = { enabled: true, token: "tok_abc-123", mcpUrl: "http://127.0.0.1:8876/mcp" };

describe("agent access config", () => {
  it("builds the Claude Code line with the token header", () => {
    expect(agentConfigLine(access)).toBe(
      'claude mcp add --transport http chaosengine http://127.0.0.1:8876/mcp --header "Authorization: Bearer tok_abc-123"',
    );
  });

  it("builds JSON for other clients", () => {
    const parsed = JSON.parse(agentConfigJson(access));
    expect(parsed.mcpServers.chaosengine).toEqual({
      type: "http",
      url: access.mcpUrl,
      headers: { Authorization: "Bearer tok_abc-123" },
    });
  });
});
