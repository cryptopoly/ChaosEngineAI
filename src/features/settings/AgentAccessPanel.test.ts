import { describe, expect, it } from "vitest";
import { agentConfigJson, agentConfigLine } from "./AgentAccessPanel";

const access = { enabled: true, token: "tok_abc-123", mcpUrl: "http://127.0.0.1:8876/mcp", stdioCommand: null as string[] | null };

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

describe("stdio line", () => {
  it("is null without the CLI and quotes each argument with it", async () => {
    const { agentStdioLine } = await import("./AgentAccessPanel");
    expect(agentStdioLine({ ...access, stdioCommand: null })).toBeNull();
    expect(agentStdioLine({ ...access, stdioCommand: ["/opt/py", "/app/chaosengine-cli", "mcp"] })).toBe(
      'claude mcp add chaosengine -- "/opt/py" "/app/chaosengine-cli" "mcp"',
    );
  });
});
