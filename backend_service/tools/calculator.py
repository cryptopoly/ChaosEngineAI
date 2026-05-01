"""Safe math expression evaluator tool."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from backend_service.tools import BaseTool

# Allowed operators and functions for safe evaluation
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MATH_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "ceil": math.ceil,
    "floor": math.floor,
    "pow": pow,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node using only allowed operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNCTIONS:
            args = [_safe_eval(arg) for arg in node.args]
            return _MATH_FUNCTIONS[node.func.id](*args)
        raise ValueError(f"Unsupported function call")
    if isinstance(node, ast.Name):
        if node.id in _MATH_FUNCTIONS:
            val = _MATH_FUNCTIONS[node.id]
            if isinstance(val, (int, float)):
                return val
        raise ValueError(f"Unknown variable: {node.id}")
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_safe_eval(elt) for elt in node.elts]
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a mathematical expression safely. Supports arithmetic, common math functions (sqrt, log, sin, cos, etc.), and constants (pi, e)."

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. 'sqrt(144) + 2**3' or '(5 * 12.5) / 3'.",
                },
            },
            "required": ["expression"],
        }

    def execute(self, **kwargs: Any) -> str:
        expression = str(kwargs.get("expression", "")).strip()
        if not expression:
            return "Error: no expression provided."

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            return f"{expression} = {result}"
        except (ValueError, TypeError, ZeroDivisionError, SyntaxError, OverflowError) as exc:
            return f"Error evaluating '{expression}': {exc}"

    def execute_structured(self, **kwargs: Any) -> Any:
        """Phase 2.8: render the calculation as a one-line code block
        so the result reads like ``2 + 2 = 4`` in monospace rather
        than getting collapsed into a JSON dump."""
        from backend_service.tools import StructuredToolOutput

        text = self.execute(**kwargs)
        if text.startswith("Error"):
            return StructuredToolOutput(text=text, render_as="markdown")
        return StructuredToolOutput(
            text=text,
            render_as="code",
            data={"code": text, "language": "text"},
        )
