"""LFM2.5 text-style tool calls (FU-192): ``<|tool_call_start|>[name(key="x", n=3), other()]<|tool_call_end|>``.

The block body is a Python list of calls with keyword arguments. It is read with ``ast.parse`` and each argument
with ``ast.literal_eval``, never ``eval``, so only literals pass (strings, numbers, booleans, None, lists, tuples,
dicts). JSON's ``true``, ``false`` and ``null`` are read too: the LFM2.5 chat template writes list and dict values
back as JSON, so the model copies that. A block that does not read cleanly, or has no end marker, is not a call
and stays in the reply as text, like a malformed ``<tool_call>`` body does.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any

OPEN = "<|tool_call_start|>"
CLOSE = "<|tool_call_end|>"
_JSON_WORDS = {"true": True, "false": False, "null": None}


class _JsonWords(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _JSON_WORDS:
            return ast.copy_location(ast.Constant(_JSON_WORDS[node.id]), node)
        return node


def _read_block(body: str) -> list[tuple[str, dict[str, Any]]] | None:
    """The calls in one block body, or None when any part of it is not a plain keyword call with literal values."""
    try:
        node = ast.parse(body.strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    calls: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.args:
            return None
        arguments: dict[str, Any] = {}
        for keyword in item.keywords:
            if keyword.arg is None:  # ``**kwargs``
                return None
            try:
                arguments[keyword.arg] = ast.literal_eval(_JsonWords().visit(keyword.value))
            except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
                return None
        try:
            json.dumps(arguments)  # a set or a complex number has no JSON form
        except (TypeError, ValueError):
            return None
        calls.append((item.func.id, arguments))
    return calls or None


def _blocks(text: str) -> list[tuple[int, int, list[tuple[str, dict[str, Any]]]]]:
    """Every closed block whose body reads cleanly, as ``(start, end, calls)``."""
    found: list[tuple[int, int, list[tuple[str, dict[str, Any]]]]] = []
    cursor = 0
    while True:
        start = text.find(OPEN, cursor)
        if start < 0:
            return found
        body_start = start + len(OPEN)
        end = text.find(CLOSE, body_start)
        if end < 0:
            return found  # unclosed: not a call
        reopened = text.find(OPEN, body_start)
        if 0 <= reopened < end:  # this opener was never closed; try the next one
            cursor = reopened
            continue
        calls = _read_block(text[body_start:end])
        if calls is not None:
            found.append((start, end + len(CLOSE), calls))
        cursor = end + len(CLOSE)


def call_spans(text: str) -> list[tuple[dict[str, Any], int, int]]:
    """Each call in ``text`` in the OpenAI ``tool_calls`` shape with the span of the block it came from."""
    if OPEN not in text:
        return []
    return [
        (
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            },
            start,
            end,
        )
        for start, end, block in _blocks(text)
        for name, arguments in block
    ]


def parse_calls(text: str) -> list[dict[str, Any]] | None:
    """The calls in ``text`` in the OpenAI ``tool_calls`` shape, or None when it holds none."""
    return [call for call, _start, _end in call_spans(text)] or None


def strip_calls(text: str) -> str:
    """``text`` without the blocks ``parse_calls`` reads; anything else, a broken block included, stays."""
    if OPEN not in text:
        return text
    blocks = _blocks(text)
    if not blocks:
        return text
    out: list[str] = []
    cursor = 0
    for start, end, _calls in blocks:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()
