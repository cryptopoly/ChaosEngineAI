"""Web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from backend_service.tools import BaseTool, StructuredToolOutput


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for current information. Returns a list of search results with titles, URLs, and snippets."

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        # Legacy text path — kept for callers / tests that don't go
        # through `execute_structured`. The model-facing return is the
        # same human-readable summary structured produces below.
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "Error: no search query provided."
        max_results = min(max(int(kwargs.get("max_results", 5)), 1), 10)
        try:
            return self._search_ddg(query, max_results)
        except Exception as exc:
            return f"Search failed: {exc}"

    def execute_structured(self, **kwargs: Any) -> StructuredToolOutput | None:
        """Phase 2.8: surface a `table` of {title, url, snippet} rows.

        The model still sees the human-readable summary text in
        `text` so its next reasoning step has all the data; the UI
        renders the rows as a clickable table via ToolCallCard.
        """
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return StructuredToolOutput(
                text="Error: no search query provided.",
                render_as="markdown",
            )
        max_results = min(max(int(kwargs.get("max_results", 5)), 1), 10)
        try:
            results = self._search_results(query, max_results)
        except Exception as exc:
            return StructuredToolOutput(
                text=f"Search failed: {exc}",
                render_as="markdown",
            )
        if not results:
            return StructuredToolOutput(
                text=f"No results found for: {query}",
                render_as="markdown",
            )
        return StructuredToolOutput(
            text=_format_results_text(query, results),
            render_as="table",
            data={
                "columns": ["#", "Title", "URL", "Snippet"],
                "rows": [
                    [str(i + 1), r["title"], r["url"], r["snippet"]]
                    for i, r in enumerate(results)
                ],
                "title": f"Web search results for \"{query}\"",
            },
        )

    def _search_results(self, query: str, max_results: int) -> list[dict[str, str]]:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })

        req = urllib.request.Request(url, headers={
            "User-Agent": "ChaosEngineAI/0.5 (desktop AI tool-use agent)",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results: list[dict[str, str]] = []
        abstract = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        if abstract:
            results.append({
                "title": data.get("Heading", "Answer"),
                "url": abstract_url,
                "snippet": abstract,
            })
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if isinstance(topic, dict):
                text = topic.get("Text", "").strip()
                first_url = topic.get("FirstURL", "").strip()
                if text and first_url:
                    results.append({
                        "title": text[:80],
                        "url": first_url,
                        "snippet": text,
                    })
        return results

    def _search_ddg(self, query: str, max_results: int) -> str:
        results = self._search_results(query, max_results)
        if not results:
            return f"No results found for: {query}"
        return _format_results_text(query, results)


def _format_results_text(query: str, results: list[dict[str, str]]) -> str:
    """Plain-text summary of the result list — fed to the language
    model on the next agent turn. Kept identical across the legacy
    `execute` and Phase 2.8 `execute_structured` paths so the model's
    reasoning is unchanged regardless of which entry point fired."""
    lines = [f"Web search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r.get("url"):
            lines.append(f"   URL: {r['url']}")
        lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines)
