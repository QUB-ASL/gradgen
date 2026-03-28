"""Expression simplification utilities."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .sx import SX, SXNode, SXVector

if TYPE_CHECKING:
    from .function import Function


SimplifyValue = Any
Effort = int | str

_EFFORT_PRESETS = {
    "none": 0,
    "basic": 1,
    "medium": 3,
    "high": 6,
    "max": 10,
}


def simplify(value: SimplifyValue, max_effort: Effort = "basic") -> SimplifyValue:
    """Simplify a symbolic expression or function.

    Args:
        value: Expression-like value or ``Function`` to simplify.
        max_effort: Maximum rewrite effort. This can be an integer pass
            count or one of ``"none"``, ``"basic"``, ``"medium"``,
            ``"high"``, or ``"max"``.

    Returns:
        A simplified value of the same high-level shape.
    """
    from .function import Function

    effort = _resolve_effort(max_effort)
    if isinstance(value, Function):
        return value.simplify(max_effort=effort)
    return _simplify_value(value, effort)


def _resolve_effort(max_effort: Effort) -> int:
    """Resolve an effort preset or explicit pass count."""
    if isinstance(max_effort, int):
        if max_effort < 0:
            raise ValueError("max_effort must be non-negative")
        return max_effort
    try:
        return _EFFORT_PRESETS[max_effort]
    except KeyError as exc:
        raise ValueError(f"unknown simplification effort {max_effort!r}") from exc


def _simplify_value(value: SX | SXVector | tuple[SXVector, ...], effort: int):
    """Simplify a supported expression-like value."""
    if effort == 0:
        return value
    if isinstance(value, SX):
        return _simplify_scalar(value, effort)
    if isinstance(value, SXVector):
        return SXVector(tuple(_simplify_scalar(element, effort) for element in value))
    return tuple(
        SXVector(tuple(_simplify_scalar(element, effort) for element in row))
        for row in value
    )


def _simplify_scalar(expr: SX, effort: int) -> SX:
    """Run bounded simplification passes over a scalar expression."""
    current = expr
    for _ in range(effort):
        next_expr = _simplify_scalar_once(current, {})
        if next_expr.node is current.node:
            break
        current = next_expr
    return current


def _simplify_scalar_once(expr: SX, cache: dict[SXNode, SX]) -> SX:
    """Simplify an expression with a single bottom-up pass."""
    cached = cache.get(expr.node)
    if cached is not None:
        return cached

    if expr.op in {"symbol", "const"}:
        cache[expr.node] = expr
        return expr

    args = tuple(_simplify_scalar_once(SX(arg), cache) for arg in expr.node.args)
    simplified = _apply_rules(expr.op, args)
    cache[expr.node] = simplified
    return simplified


def _apply_rules(op: str, args: tuple[SX, ...]) -> SX:
    """Apply local algebraic simplification rules."""
    if op == "neg":
        arg = args[0]
        if _is_const(arg):
            return SX.const(-_const_value(arg))
        if arg.op == "neg":
            return arg.args[0]
        return SX(SXNode.make(op, (arg.node,)))

    if len(args) == 2:
        left, right = args

        if _is_const(left) and _is_const(right):
            return SX.const(_evaluate_const_op(op, _const_value(left), _const_value(right)))

        if op == "add":
            if _is_zero(left):
                return right
            if _is_zero(right):
                return left
            if left.node is right.node:
                return SX.const(2.0) * left

        if op == "sub":
            if _is_zero(right):
                return left
            if left.node is right.node:
                return SX.const(0.0)
            if _is_zero(left):
                return -right

        if op == "mul":
            if _is_zero(left) or _is_zero(right):
                return SX.const(0.0)
            if _is_one(left):
                return right
            if _is_one(right):
                return left

        if op == "div":
            if _is_zero(left):
                return SX.const(0.0)
            if _is_one(right):
                return left
            if left.node is right.node:
                return SX.const(1.0)

        if op == "pow":
            if _is_zero(right):
                return SX.const(1.0)
            if _is_one(right):
                return left
            if _is_zero(left):
                return SX.const(0.0)
            if _is_one(left):
                return SX.const(1.0)

        return SX(SXNode.make(op, (left.node, right.node)))

    arg = args[0]
    if _is_const(arg):
        return SX.const(_evaluate_const_unary(op, _const_value(arg)))
    return SX(SXNode.make(op, (arg.node,)))


def _evaluate_const_op(op: str, left: float, right: float) -> float:
    """Evaluate a binary operator on constant arguments."""
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        return left / right
    if op == "pow":
        return left**right
    raise ValueError(f"cannot constant-fold operation {op!r}")


def _evaluate_const_unary(op: str, value: float) -> float:
    """Evaluate a unary operator on a constant argument."""
    if op == "sin":
        return math.sin(value)
    if op == "cos":
        return math.cos(value)
    if op == "exp":
        return math.exp(value)
    if op == "log":
        return math.log(value)
    if op == "sqrt":
        return math.sqrt(value)
    raise ValueError(f"cannot constant-fold operation {op!r}")


def _is_const(expr: SX) -> bool:
    """Return ``True`` if the expression is a constant."""
    return expr.op == "const" and expr.value is not None


def _const_value(expr: SX) -> float:
    """Return the numeric value of a constant expression."""
    if expr.value is None:
        raise ValueError("expected a constant expression")
    return expr.value


def _is_zero(expr: SX) -> bool:
    """Return ``True`` if the expression is the constant zero."""
    return _is_const(expr) and _const_value(expr) == 0.0


def _is_one(expr: SX) -> bool:
    """Return ``True`` if the expression is the constant one."""
    return _is_const(expr) and _const_value(expr) == 1.0
