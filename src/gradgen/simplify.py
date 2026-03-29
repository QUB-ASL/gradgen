"""Expression simplification utilities."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .sx import SX, SXNode, SXVector, parse_bilinear_form_args, parse_matvec_component_args, parse_quadform_args

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
    if op == "matvec_component":
        rows, cols, row, matrix_values, x_values = parse_matvec_component_args(args)
        _ = rows
        if all(_is_const(arg) for arg in x_values):
            start = row * cols
            total = sum(matrix_values[start + index] * _const_value(x_values[index]) for index in range(cols))
            return SX.const(total)
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "quadform":
        size, matrix_values, x_values = parse_quadform_args(args)
        if all(_is_const(arg) for arg in x_values):
            total = 0.0
            for row in range(size):
                for col in range(size):
                    total += matrix_values[row * size + col] * _const_value(x_values[row]) * _const_value(x_values[col])
            return SX.const(total)
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "bilinear_form":
        rows, cols, matrix_values, x_values, y_values = parse_bilinear_form_args(args)
        if all(_is_const(arg) for arg in (*x_values, *y_values)):
            total = 0.0
            for row in range(rows):
                for col in range(cols):
                    total += matrix_values[row * cols + col] * _const_value(x_values[row]) * _const_value(y_values[col])
            return SX.const(total)
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "sum":
        if not args:
            return SX.const(0.0)
        if all(_is_const(arg) for arg in args):
            return SX.const(sum(_const_value(arg) for arg in args))
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "prod":
        if not args:
            return SX.const(1.0)
        if all(_is_const(arg) for arg in args):
            total = 1.0
            for arg in args:
                total *= _const_value(arg)
            return SX.const(total)
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "reduce_max":
        if not args:
            raise ValueError("vector max is undefined for empty vectors")
        if all(_is_const(arg) for arg in args):
            return SX.const(max(_const_value(arg) for arg in args))
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "reduce_min":
        if not args:
            raise ValueError("vector min is undefined for empty vectors")
        if all(_is_const(arg) for arg in args):
            return SX.const(min(_const_value(arg) for arg in args))
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "mean":
        if not args:
            raise ValueError("vector mean is undefined for empty vectors")
        if all(_is_const(arg) for arg in args):
            return SX.const(sum(_const_value(arg) for arg in args) / len(args))
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "norm2":
        if not args:
            return SX.const(0.0)
        if all(_is_const(arg) for arg in args):
            total = sum(_const_value(arg) * _const_value(arg) for arg in args)
            return SX.const(math.sqrt(total))
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "norm2sq":
        if not args:
            return SX.const(0.0)
        if all(_is_const(arg) for arg in args):
            total = sum(_const_value(arg) * _const_value(arg) for arg in args)
            return SX.const(total)
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "norm1":
        if not args:
            return SX.const(0.0)
        if all(_is_const(arg) for arg in args):
            return SX.const(sum(math.fabs(_const_value(arg)) for arg in args))
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "norm_inf":
        if not args:
            return SX.const(0.0)
        if all(_is_const(arg) for arg in args):
            return SX.const(max(math.fabs(_const_value(arg)) for arg in args))
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "norm_p_to_p":
        if len(args) < 2:
            return SX.const(0.0)
        if all(_is_const(arg) for arg in args):
            p = _const_value(args[-1])
            total = sum(math.fabs(_const_value(arg)) ** p for arg in args[:-1])
            return SX.const(total)
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

    if op == "norm_p":
        if len(args) < 2:
            return SX.const(0.0)
        if all(_is_const(arg) for arg in args):
            p = _const_value(args[-1])
            total = sum(math.fabs(_const_value(arg)) ** p for arg in args[:-1])
            return SX.const(total ** (1.0 / p))
        return SX(SXNode.make(op, tuple(arg.node for arg in args)))

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
            return _simplify_add(left, right)

        if op == "sub":
            if left.node is right.node:
                return SX.const(0.0)
            return _simplify_add(left, -right)

        if op == "mul":
            return _simplify_mul(left, right)

        if op == "div":
            if _is_zero(left):
                return SX.const(0.0)
            if _is_one(right):
                return left
            if left.node is right.node:
                return SX.const(1.0)
            left_sign, left_base = _strip_negation(left)
            right_sign, right_base = _strip_negation(right)
            sign = left_sign * right_sign
            quotient = SX(SXNode.make("div", (left_base.node, right_base.node)))
            if sign == -1.0:
                return -quotient
            return quotient

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
    if op == "sin":
        sign, base = _strip_negation(arg)
        if sign == -1.0:
            return -base.sin()
    if op == "cos":
        sign, base = _strip_negation(arg)
        if sign == -1.0:
            return base.cos()
    return SX(SXNode.make(op, (arg.node,)))


def _simplify_add(left: SX, right: SX) -> SX:
    """Simplify an addition-like expression after child simplification."""
    terms = _flatten_add_terms((left, right))
    if not terms:
        return SX.const(0.0)

    simplified_terms = _apply_trig_identity_terms(terms)
    if simplified_terms != terms:
        terms = simplified_terms

    constant_sum = 0.0
    coefficients: dict[SXNode, float] = {}
    ordered_bases: list[SX] = []
    residual_terms: list[SX] = []

    for term in terms:
        if _is_zero(term):
            continue
        if _is_const(term):
            constant_sum += _const_value(term)
            continue

        coefficient, base = _split_scalar_factor(term)
        if base is None:
            residual_terms.append(term)
            continue
        if base.node not in coefficients:
            ordered_bases.append(base)
            coefficients[base.node] = 0.0
        coefficients[base.node] += coefficient

    combined_terms: list[SX] = []
    for base in ordered_bases:
        coefficient = coefficients[base.node]
        if coefficient == 0.0:
            continue
        if coefficient == 1.0:
            combined_terms.append(base)
        elif coefficient == -1.0:
            combined_terms.append(-base)
        else:
            combined_terms.append(SX.const(coefficient) * base)

    combined_terms.extend(residual_terms)
    if constant_sum != 0.0:
        combined_terms.append(SX.const(constant_sum))

    if not combined_terms:
        return SX.const(0.0)
    if all(term.op == "neg" for term in combined_terms):
        positive_terms = [term.args[0] for term in combined_terms]
        return -_rebuild_add(positive_terms)
    return _rebuild_add(combined_terms)


def _simplify_mul(left: SX, right: SX) -> SX:
    """Simplify a multiplication-like expression after child simplification."""
    factors = _flatten_mul_factors((left, right))
    if not factors:
        return SX.const(1.0)

    constant_product = 1.0
    exponent_sums: dict[SXNode, float] = {}
    ordered_bases: list[SX] = []
    for factor in factors:
        if _is_zero(factor):
            return SX.const(0.0)
        if _is_const(factor):
            constant_product *= _const_value(factor)
            continue

        sign, normalized_factor = _strip_negation(factor)
        constant_product *= sign
        exponent, base = _split_power_factor(normalized_factor)
        if base.node not in exponent_sums:
            ordered_bases.append(base)
            exponent_sums[base.node] = 0.0
        exponent_sums[base.node] += exponent

    if constant_product == 0.0:
        return SX.const(0.0)

    leading_negation = constant_product == -1.0 and bool(ordered_bases)
    rebuilt_factors: list[SX] = []
    if not leading_negation and (constant_product != 1.0 or not ordered_bases):
        rebuilt_factors.append(SX.const(constant_product))

    for base in ordered_bases:
        exponent = exponent_sums[base.node]
        if exponent == 0.0:
            continue
        if exponent == 1.0:
            rebuilt_factors.append(base)
        else:
            rebuilt_factors.append(SX(SXNode.make("pow", (base.node, SX.const(exponent).node))))

    if not rebuilt_factors:
        return SX.const(1.0)
    product = _rebuild_mul(rebuilt_factors)
    if leading_negation:
        return -product
    return product


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
    if op == "atan2":
        return math.atan2(left, right)
    if op == "hypot":
        return math.hypot(left, right)
    if op == "min":
        return min(left, right)
    if op == "max":
        return max(left, right)
    raise ValueError(f"cannot constant-fold operation {op!r}")


def _evaluate_const_unary(op: str, value: float) -> float:
    """Evaluate a unary operator on a constant argument."""
    if op == "sin":
        return math.sin(value)
    if op == "cos":
        return math.cos(value)
    if op == "tan":
        return math.tan(value)
    if op == "asin":
        return math.asin(value)
    if op == "acos":
        return math.acos(value)
    if op == "atan":
        return math.atan(value)
    if op == "asinh":
        return math.asinh(value)
    if op == "acosh":
        return math.acosh(value)
    if op == "atanh":
        return math.atanh(value)
    if op == "sinh":
        return math.sinh(value)
    if op == "cosh":
        return math.cosh(value)
    if op == "tanh":
        return math.tanh(value)
    if op == "exp":
        return math.exp(value)
    if op == "expm1":
        return math.expm1(value)
    if op == "log":
        return math.log(value)
    if op == "log1p":
        return math.log1p(value)
    if op == "sqrt":
        return math.sqrt(value)
    if op == "cbrt":
        return math.copysign(abs(value) ** (1.0 / 3.0), value)
    if op == "erf":
        return math.erf(value)
    if op == "erfc":
        return math.erfc(value)
    if op == "floor":
        return math.floor(value)
    if op == "ceil":
        return math.ceil(value)
    if op == "round":
        if value >= 0.0:
            return math.floor(value + 0.5)
        return math.ceil(value - 0.5)
    if op == "trunc":
        return math.trunc(value)
    if op == "fract":
        return value - math.trunc(value)
    if op == "signum":
        if value > 0.0:
            return 1.0
        if value < 0.0:
            return -1.0
        return 0.0
    if op == "abs":
        return math.fabs(value)
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


def _flatten_add_terms(args: tuple[SX, ...]) -> list[SX]:
    """Return flattened additive terms from nested ``add`` nodes."""
    terms: list[SX] = []
    for arg in args:
        if arg.op == "add":
            terms.extend(_flatten_add_terms(arg.args))
        else:
            terms.append(arg)
    return terms


def _flatten_mul_factors(args: tuple[SX, ...]) -> list[SX]:
    """Return flattened multiplicative factors from nested ``mul`` nodes."""
    factors: list[SX] = []
    for arg in args:
        if arg.op == "mul":
            factors.extend(_flatten_mul_factors(arg.args))
        else:
            factors.append(arg)
    return factors


def _rebuild_add(terms: list[SX]) -> SX:
    """Rebuild an addition from a normalized term list."""
    result = terms[0]
    for term in terms[1:]:
        result = SX(SXNode.make("add", (result.node, term.node)))
    return result


def _rebuild_mul(factors: list[SX]) -> SX:
    """Rebuild a multiplication from a normalized factor list."""
    result = factors[0]
    for factor in factors[1:]:
        result = SX(SXNode.make("mul", (result.node, factor.node)))
    return result


def _split_scalar_factor(term: SX) -> tuple[float, SX | None]:
    """Split a term into a numeric coefficient and symbolic base.

    Returns ``(coefficient, base)`` when the term can be interpreted as a
    scalar multiple of a symbolic base. Constant terms return
    ``(value, SX.const(1.0))`` only through the caller's constant path and
    are therefore not expected here.
    """
    if _is_const(term):
        return _const_value(term), SX.const(1.0)
    if term.op == "neg":
        return -1.0, term.args[0]
    if term.op == "mul":
        factors = _flatten_mul_factors((term,))
        coefficient = 1.0
        symbolic_factors: list[SX] = []
        for factor in factors:
            if _is_const(factor):
                coefficient *= _const_value(factor)
            else:
                symbolic_factors.append(factor)
        if not symbolic_factors:
            return coefficient, SX.const(1.0)
        return coefficient, _rebuild_mul(symbolic_factors)
    return 1.0, term


def _strip_negation(expr: SX) -> tuple[float, SX]:
    """Return ``(sign, expr_without_outer_negation)``."""
    sign = 1.0
    current = expr
    while current.op == "neg":
        sign *= -1.0
        current = current.args[0]
    return sign, current


def _split_power_factor(term: SX) -> tuple[float, SX]:
    """Split a factor into a numeric exponent and symbolic base."""
    if term.op == "pow":
        base, exponent = term.args
        if _is_const(exponent):
            return _const_value(exponent), base
    return 1.0, term


def _apply_trig_identity_terms(terms: list[SX]) -> list[SX]:
    """Apply safe trigonometric identities to a flattened add term list."""
    remaining = list(terms)
    index = 0
    while index < len(remaining):
        lhs = remaining[index]
        lhs_sin_base = _match_square_of_unary(lhs, "sin")
        lhs_cos_base = _match_square_of_unary(lhs, "cos")
        if lhs_sin_base is not None:
            partner_index = _find_matching_square(remaining, lhs_sin_base, "cos", index + 1)
        elif lhs_cos_base is not None:
            partner_index = _find_matching_square(remaining, lhs_cos_base, "sin", index + 1)
        else:
            index += 1
            continue

        if partner_index is None:
            index += 1
            continue

        replacement_terms = remaining[:index] + [SX.const(1.0)] + remaining[index + 1 : partner_index] + remaining[partner_index + 1 :]
        return replacement_terms
    return terms


def _find_matching_square(
    terms: list[SX],
    base: SX,
    op: str,
    start: int,
) -> int | None:
    """Find ``op(base)^2`` in a flattened term list."""
    for index in range(start, len(terms)):
        candidate_base = _match_square_of_unary(terms[index], op)
        if candidate_base is not None and candidate_base.node is base.node:
            return index
    return None


def _match_square_of_unary(expr: SX, op: str) -> SX | None:
    """Match expressions of the form ``op(x)^2``."""
    if expr.op != "pow":
        return None
    base, exponent = expr.args
    if not (_is_const(exponent) and _const_value(exponent) == 2.0):
        return None
    if base.op != op:
        return None
    return base.args[0]
