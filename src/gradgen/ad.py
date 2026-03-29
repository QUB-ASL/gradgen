"""Automatic differentiation for ``SX`` expressions."""

from __future__ import annotations

from .sx import SX, SXNode, SXVector, vector


ADExpr = SX | SXVector
JacobianExpr = SX | SXVector | tuple[SXVector, ...]
HessianExpr = SX | tuple[SXVector, ...]
ADSeed = SX | SXVector | float | int | list[object] | tuple[object, ...]


def jvp(expr: ADExpr, wrt: ADExpr, tangent: ADSeed | None = None) -> ADExpr:
    """Compute a symbolic Jacobian-vector product in forward mode.

    Args:
        expr: Expression to differentiate.
        wrt: Scalar or vector variables with respect to which ``expr`` is
            differentiated.
        tangent: Tangent seed matching the shape of ``wrt``. For scalar
            differentiation this defaults to ``1.0``.

    Returns:
        A symbolic expression with the same shape as ``expr`` containing
        the forward-mode derivative in the supplied direction.
    """
    tangent_map = _build_tangent_map(wrt, tangent)

    if isinstance(expr, SX):
        return _forward_scalar(expr, tangent_map, {})
    return SXVector(tuple(_forward_scalar(element, tangent_map, {}) for element in expr))


def derivative(expr: SX, wrt: SX, seed: SX | float | int = 1.0) -> SX:
    """Compute the forward derivative of a scalar expression."""
    tangent = seed if isinstance(seed, SX) else SX.const(seed)
    return _forward_scalar(expr, {wrt.node: tangent}, {})


def vjp(expr: ADExpr, wrt: ADExpr, cotangent: ADSeed | None = None) -> ADExpr:
    """Compute a symbolic vector-Jacobian product in reverse mode.

    Args:
        expr: Output expression being differentiated.
        wrt: Scalar or vector variables with respect to which ``expr`` is
            differentiated.
        cotangent: Cotangent seed matching the shape of ``expr``. For
            scalar outputs this defaults to ``1.0``.

    Returns:
        A symbolic expression with the same shape as ``wrt`` containing the
        reverse-mode derivative in the supplied cotangent direction.
    """
    outputs = _flatten_expr(expr)
    output_nodes = tuple(output.node for output in outputs)
    seed_map = _build_cotangent_map(expr, cotangent)
    nodes = _topological_nodes(outputs)
    expr_by_node = {node: SX(node) for node in nodes}
    adjoints = {node: SX.const(0.0) for node in nodes}

    for node, seed in seed_map.items():
        adjoints[node] = adjoints[node] + seed

    for node in reversed(nodes):
        adjoint = adjoints[node]
        if _is_zero_const(adjoint):
            continue
        _propagate_reverse(node, expr_by_node, adjoint, adjoints)

    if isinstance(wrt, SX):
        return adjoints.get(wrt.node, SX.const(0.0))
    return SXVector(tuple(adjoints.get(variable.node, SX.const(0.0)) for variable in wrt))


def gradient(expr: SX, wrt: ADExpr, seed: SX | float | int = 1.0) -> ADExpr:
    """Compute a reverse-mode gradient of a scalar expression."""
    cotangent = seed if isinstance(seed, SX) else SX.const(seed)
    return vjp(expr, wrt, cotangent)


def jacobian(expr: ADExpr, wrt: ADExpr) -> JacobianExpr:
    """Compute a symbolic Jacobian with vector-first output structure.

    The return shape follows the current no-matrix design:

    - scalar wrt scalar -> ``SX``
    - scalar wrt vector -> ``SXVector``
    - vector wrt scalar -> ``SXVector``
    - vector wrt vector -> ``tuple[SXVector, ...]``, one row per output
    """
    if isinstance(expr, SX):
        if isinstance(wrt, SX):
            return derivative(expr, wrt)
        return gradient(expr, wrt)

    if isinstance(wrt, SX):
        return SXVector(tuple(derivative(element, wrt) for element in expr))

    return tuple(gradient(element, wrt) for element in expr)


def hessian(expr: SX, wrt: ADExpr) -> HessianExpr:
    """Compute a symbolic Hessian for a scalar expression.

    The current return shape follows the vector-first representation:

    - scalar wrt scalar -> ``SX``
    - scalar wrt vector -> ``tuple[SXVector, ...]``, one row per variable
    """
    if isinstance(wrt, SX):
        return derivative(derivative(expr, wrt), wrt)

    return tuple(gradient(component, wrt) for component in gradient(expr, wrt))


def _build_tangent_map(wrt: ADExpr, tangent: ADSeed | None) -> dict[SXNode, SX]:
    """Map formal differentiation variables to tangent expressions."""
    if isinstance(wrt, SX):
        if tangent is None:
            tangent_value = SX.const(1.0)
        else:
            tangent_value = _coerce_scalar_seed(tangent)
        return {wrt.node: tangent_value}

    if tangent is None:
        raise ValueError("vector differentiation requires an explicit tangent seed")

    tangent_vector = _coerce_vector_seed(tangent)
    if len(wrt) != len(tangent_vector):
        raise ValueError("tangent seed length must match the differentiation variable")

    return {
        variable.node: tangent_value
        for variable, tangent_value in zip(wrt, tangent_vector)
    }


def _build_cotangent_map(expr: ADExpr, cotangent: ADSeed | None) -> dict[SXNode, SX]:
    """Map output nodes to reverse-mode cotangent seed expressions."""
    if isinstance(expr, SX):
        if cotangent is None:
            cotangent_value = SX.const(1.0)
        else:
            cotangent_value = _coerce_scalar_seed(cotangent)
        return {expr.node: cotangent_value}

    if cotangent is None:
        raise ValueError("vector reverse-mode differentiation requires an explicit cotangent seed")

    cotangent_vector = _coerce_vector_seed(cotangent)
    if len(expr) != len(cotangent_vector):
        raise ValueError("cotangent seed length must match the differentiated expression")

    return {
        output.node: seed_value
        for output, seed_value in zip(expr, cotangent_vector)
    }


def _forward_scalar(
    expr: SX,
    tangent_map: dict[SXNode, SX],
    cache: dict[SXNode, SX],
) -> SX:
    """Recursively propagate forward-mode tangents through an expression."""
    cached = cache.get(expr.node)
    if cached is not None:
        return cached

    if expr.op == "const":
        result = SX.const(0.0)
    elif expr.op == "symbol":
        result = tangent_map.get(expr.node, SX.const(0.0))
    else:
        args = tuple(SX(arg) for arg in expr.node.args)
        tangents = tuple(_forward_scalar(arg, tangent_map, cache) for arg in args)
        result = _differentiate_op(expr, args, tangents)

    cache[expr.node] = result
    return result


def _differentiate_op(expr: SX, args: tuple[SX, ...], tangents: tuple[SX, ...]) -> SX:
    """Apply forward-mode differentiation rules for a single operation."""
    if expr.op == "norm2sq":
        total = SX.const(0.0)
        for arg, tangent in zip(args, tangents):
            total = total + (SX.const(2.0) * arg * tangent)
        return total
    if expr.op == "norm2":
        numerator = SX.const(0.0)
        for arg, tangent in zip(args, tangents):
            numerator = numerator + (arg * tangent)
        return numerator / expr
    if expr.op == "add":
        return tangents[0] + tangents[1]
    if expr.op == "sub":
        return tangents[0] - tangents[1]
    if expr.op == "mul":
        return (tangents[0] * args[1]) + (args[0] * tangents[1])
    if expr.op == "div":
        numerator = (tangents[0] * args[1]) - (args[0] * tangents[1])
        return numerator / (args[1] * args[1])
    if expr.op == "pow":
        base = args[0]
        exponent = args[1]
        base_tangent = tangents[0]
        exponent_tangent = tangents[1]
        return expr * ((exponent_tangent * base.log()) + (exponent * (base_tangent / base)))
    if expr.op == "neg":
        return -tangents[0]
    if expr.op == "sin":
        return args[0].cos() * tangents[0]
    if expr.op == "cos":
        return -(args[0].sin() * tangents[0])
    if expr.op == "tan":
        return tangents[0] / (args[0].cos() * args[0].cos())
    if expr.op == "asin":
        return tangents[0] / (SX.const(1.0) - (args[0] * args[0])).sqrt()
    if expr.op == "acos":
        return -(tangents[0] / (SX.const(1.0) - (args[0] * args[0])).sqrt())
    if expr.op == "atan":
        return tangents[0] / (SX.const(1.0) + (args[0] * args[0]))
    if expr.op == "sinh":
        return args[0].cosh() * tangents[0]
    if expr.op == "cosh":
        return args[0].sinh() * tangents[0]
    if expr.op == "tanh":
        return (SX.const(1.0) - (expr * expr)) * tangents[0]
    if expr.op == "exp":
        return expr * tangents[0]
    if expr.op == "expm1":
        return (expr + SX.const(1.0)) * tangents[0]
    if expr.op == "log":
        return tangents[0] / args[0]
    if expr.op == "log1p":
        return tangents[0] / (SX.const(1.0) + args[0])
    if expr.op == "sqrt":
        return tangents[0] / (SX.const(2.0) * expr)
    if expr.op == "norm_p_to_p":
        p = _require_constant_norm_p(args, expr.op)
        total = SX.const(0.0)
        scale = SX.const(p)
        exponent = SX.const(p - 2.0)
        for arg, tangent in zip(args[:-1], tangents[:-1]):
            total = total + (scale * (arg.abs() ** exponent) * arg * tangent)
        return total
    if expr.op == "norm_p":
        p = _require_constant_norm_p(args, expr.op)
        norm_p_to_p_expr = SX(SXNode.make("norm_p_to_p", tuple(arg.node for arg in args)))
        norm_p_to_p_tangent = _differentiate_op(norm_p_to_p_expr, args, tangents)
        return (SX.const(1.0 / p) * (norm_p_to_p_expr ** SX.const((1.0 / p) - 1.0))) * norm_p_to_p_tangent
    if expr.op in {"norm1", "norm_inf"}:
        raise ValueError(f"cannot differentiate operation {expr.op!r}")
    if expr.op == "abs":
        return (args[0] / expr) * tangents[0]

    raise ValueError(f"cannot differentiate operation {expr.op!r}")


def _coerce_scalar_seed(value: ADSeed) -> SX:
    """Convert a scalar tangent seed into ``SX``."""
    if isinstance(value, SX):
        return value
    if isinstance(value, (int, float)):
        return SX.const(value)
    raise TypeError("scalar tangent seeds require an SX, int, or float")


def _coerce_vector_seed(value: ADSeed) -> SXVector:
    """Convert a vector tangent seed into ``SXVector``."""
    if isinstance(value, SXVector):
        return value
    if isinstance(value, (list, tuple)):
        return vector(value)
    raise TypeError("vector tangent seeds require an SXVector or a sequence of scalar-like values")


def _flatten_expr(expr: ADExpr) -> tuple[SX, ...]:
    """Flatten a scalar or vector expression into scalar outputs."""
    if isinstance(expr, SX):
        return (expr,)
    return expr.elements


def _topological_nodes(outputs: tuple[SX, ...]) -> tuple[SXNode, ...]:
    """Return output dependency nodes in topological order."""
    ordered: list[SXNode] = []
    seen: set[SXNode] = set()

    for output in outputs:
        _visit_node(output.node, seen, ordered)

    return tuple(ordered)


def _visit_node(node: SXNode, seen: set[SXNode], ordered: list[SXNode]) -> None:
    """Depth-first topological traversal of expression nodes."""
    if node in seen:
        return

    for arg in node.args:
        _visit_node(arg, seen, ordered)

    seen.add(node)
    ordered.append(node)


def _propagate_reverse(
    node: SXNode,
    expr_by_node: dict[SXNode, SX],
    adjoint: SX,
    adjoints: dict[SXNode, SX],
) -> None:
    """Propagate a reverse-mode adjoint contribution to child nodes."""
    expr = expr_by_node[node]
    args = tuple(expr_by_node[arg] for arg in node.args)

    if node.op in {"const", "symbol"}:
        return
    if node.op == "add":
        _accumulate_adjoint(adjoints, node.args[0], adjoint)
        _accumulate_adjoint(adjoints, node.args[1], adjoint)
        return
    if node.op == "sub":
        _accumulate_adjoint(adjoints, node.args[0], adjoint)
        _accumulate_adjoint(adjoints, node.args[1], -adjoint)
        return
    if node.op == "mul":
        _accumulate_adjoint(adjoints, node.args[0], adjoint * args[1])
        _accumulate_adjoint(adjoints, node.args[1], adjoint * args[0])
        return
    if node.op == "div":
        _accumulate_adjoint(adjoints, node.args[0], adjoint / args[1])
        _accumulate_adjoint(
            adjoints,
            node.args[1],
            -(adjoint * args[0]) / (args[1] * args[1]),
        )
        return
    if node.op == "pow":
        _accumulate_adjoint(adjoints, node.args[0], adjoint * expr * (args[1] / args[0]))
        _accumulate_adjoint(adjoints, node.args[1], adjoint * expr * args[0].log())
        return
    if node.op == "neg":
        _accumulate_adjoint(adjoints, node.args[0], -adjoint)
        return
    if node.op == "sin":
        _accumulate_adjoint(adjoints, node.args[0], adjoint * args[0].cos())
        return
    if node.op == "cos":
        _accumulate_adjoint(adjoints, node.args[0], -(adjoint * args[0].sin()))
        return
    if node.op == "tan":
        _accumulate_adjoint(
            adjoints,
            node.args[0],
            adjoint / (args[0].cos() * args[0].cos()),
        )
        return
    if node.op == "asin":
        _accumulate_adjoint(
            adjoints,
            node.args[0],
            adjoint / (SX.const(1.0) - (args[0] * args[0])).sqrt(),
        )
        return
    if node.op == "acos":
        _accumulate_adjoint(
            adjoints,
            node.args[0],
            -(adjoint / (SX.const(1.0) - (args[0] * args[0])).sqrt()),
        )
        return
    if node.op == "atan":
        _accumulate_adjoint(
            adjoints,
            node.args[0],
            adjoint / (SX.const(1.0) + (args[0] * args[0])),
        )
        return
    if node.op == "sinh":
        _accumulate_adjoint(adjoints, node.args[0], adjoint * args[0].cosh())
        return
    if node.op == "cosh":
        _accumulate_adjoint(adjoints, node.args[0], adjoint * args[0].sinh())
        return
    if node.op == "tanh":
        _accumulate_adjoint(
            adjoints,
            node.args[0],
            adjoint * (SX.const(1.0) - (expr * expr)),
        )
        return
    if node.op == "exp":
        _accumulate_adjoint(adjoints, node.args[0], adjoint * expr)
        return
    if node.op == "expm1":
        _accumulate_adjoint(adjoints, node.args[0], adjoint * (expr + SX.const(1.0)))
        return
    if node.op == "log":
        _accumulate_adjoint(adjoints, node.args[0], adjoint / args[0])
        return
    if node.op == "log1p":
        _accumulate_adjoint(adjoints, node.args[0], adjoint / (SX.const(1.0) + args[0]))
        return
    if node.op == "sqrt":
        _accumulate_adjoint(adjoints, node.args[0], adjoint / (SX.const(2.0) * expr))
        return
    if node.op == "norm2":
        for child_node, arg in zip(node.args, args):
            _accumulate_adjoint(adjoints, child_node, adjoint * (arg / expr))
        return
    if node.op == "norm2sq":
        for child_node, arg in zip(node.args, args):
            _accumulate_adjoint(adjoints, child_node, adjoint * (SX.const(2.0) * arg))
        return
    if node.op == "norm_p_to_p":
        p = _require_constant_norm_p(args, node.op)
        scale = SX.const(p)
        exponent = SX.const(p - 2.0)
        for child_node, arg in zip(node.args[:-1], args[:-1]):
            _accumulate_adjoint(
                adjoints,
                child_node,
                adjoint * scale * (arg.abs() ** exponent) * arg,
            )
        return
    if node.op == "norm_p":
        p = _require_constant_norm_p(args, node.op)
        norm_p_to_p_expr = SX(SXNode.make("norm_p_to_p", tuple(node.args)))
        prefactor = adjoint * SX.const(1.0 / p) * (norm_p_to_p_expr ** SX.const((1.0 / p) - 1.0))
        for child_node, arg in zip(node.args[:-1], args[:-1]):
            _accumulate_adjoint(
                adjoints,
                child_node,
                prefactor * SX.const(p) * (arg.abs() ** SX.const(p - 2.0)) * arg,
            )
        return
    if node.op in {"norm1", "norm_inf"}:
        raise ValueError(f"cannot differentiate operation {node.op!r}")
    if node.op == "abs":
        _accumulate_adjoint(adjoints, node.args[0], adjoint * (args[0] / expr))
        return

    raise ValueError(f"cannot differentiate operation {node.op!r}")


def _require_constant_norm_p(args: tuple[SX, ...], op: str) -> float:
    """Return a validated constant ``p`` for p-norm differentiation rules."""
    p = args[-1]
    if p.op != "const" or p.value is None:
        raise ValueError(f"cannot differentiate operation {op!r} with non-constant p")
    if p.value == 1.0:
        raise ValueError(f"cannot differentiate operation {op!r} when p == 1")
    if p.value <= 1.0:
        raise ValueError(f"cannot differentiate operation {op!r} when p <= 1")
    return p.value


def _accumulate_adjoint(adjoints: dict[SXNode, SX], node: SXNode, contribution: SX) -> None:
    """Accumulate a reverse-mode contribution for a node."""
    adjoints[node] = adjoints.get(node, SX.const(0.0)) + contribution


def _is_zero_const(expr: SX) -> bool:
    """Return ``True`` when an expression is exactly the constant zero."""
    return expr.op == "const" and expr.value == 0.0
