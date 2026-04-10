"""Web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from backend_service.tools import BaseTool


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
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "Error: no search query provided."

        max_results = min(max(int(kwargs.get("max_results", 5)), 1), 10)

        try:
            return self._search_ddg(query, max_results)
        except Exception as exc:
            return f"Search failed: {exc}"

    def _search_ddg(self, query: str, max_results: int) -> str:
        """Use DuckDuckGo HTML search as a lightweight fallback.

        This avoids any external SDK dependency while still providing
        real web search results via the DDG instant answer API.
        """
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

        # Abstract (instant answer)
        abstract = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        if abstract:
            results.append({
                "title": data.get("Heading", "Answer"),
                "url": abstract_url,
                "snippet": abstract,
            })

        # Related topics
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

        if not results:
            return f"No results found for: {query}"

        lines = [f"Web search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            if r.get("url"):
                lines.append(f"   URL: {r['url']}")
            lines.append(f"   {r['snippet']}")
            lines.append("")

        return "\n".join(lines)
