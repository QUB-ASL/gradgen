"""Expression rendering and scalar helper emission for Rust code generation."""

from __future__ import annotations

import re

from ..._custom_elementary import (
    parse_custom_scalar_args,
    parse_custom_scalar_hvp_args,
    parse_custom_vector_args,
    parse_custom_vector_hessian_entry_args,
    parse_custom_vector_hvp_component_args,
    parse_custom_vector_jacobian_component_args,
)
from ...sx import (
    SX,
    SXNode,
    SXVector,
    parse_bilinear_form_args,
    parse_matvec_component_args,
    parse_quadform_args,
)
from ..config import RustBackendMode, RustScalarType
from .util import _format_float


def _emit_expr_ref(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a Rust expression reference for an already-available value."""
    if expr.op == "const":
        return _format_float(expr.value, scalar_type)
    if expr.op == "symbol":
        return scalar_bindings[expr.node]
    if expr.node in workspace_map:
        return f"work[{workspace_map[expr.node]}]"
    return _emit_node_expr(
        expr,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_math_call(
    op: str,
    args: tuple[str, ...],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a Rust math call for the selected backend mode."""
    if backend_mode == "std":
        if op == "pow":
            return f"{args[0]}.powf({args[1]})"
        if op == "atan2":
            return f"{args[0]}.atan2({args[1]})"
        if op == "hypot":
            return f"{args[0]}.hypot({args[1]})"
        if op == "min":
            return f"{args[0]}.min({args[1]})"
        if op == "max":
            return f"{args[0]}.max({args[1]})"
        if op == "log":
            return f"{args[0]}.ln()"
        if op == "log1p":
            return f"{args[0]}.ln_1p()"
        if op == "expm1":
            return f"{args[0]}.exp_m1()"
        return f"{args[0]}.{op}()"

    if op == "fract":
        trunc_expr = _emit_math_call(
            "trunc", args, backend_mode, scalar_type, math_library
        )
        return f"{args[0]} - {trunc_expr}"
    if op == "signum":
        return (
            f"if {args[0]} > 0.0_{scalar_type} {{ 1.0_{scalar_type} }} "
            f"else if {args[0]} < 0.0_{scalar_type} {{ -1.0_{scalar_type} }} "
            f"else {{ 0.0_{scalar_type} }}"
        )

    if math_library is None:
        raise ValueError("no_std math calls require a resolved math library")
    if op == "pow":
        return (
            f"{math_library}::{_math_function_name(op, scalar_type)}("
            f"{args[0]}, {args[1]})"
        )
    if op in {"atan2", "hypot", "max", "min"}:
        return (
            f"{math_library}::{_math_function_name(op, scalar_type)}("
            f"{args[0]}, {args[1]})"
        )
    return f"{math_library}::{_math_function_name(op, scalar_type)}({args[0]})"


def _match_contiguous_slice(args: tuple[str, ...]) -> str | None:
    """Return the slice name when args are exactly ``name[0], ...``."""
    if not args:
        return None

    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", args[0])
    if match is None:
        return None
    name = match.group(1)

    for index, arg in enumerate(args):
        candidate = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", arg)
        if (
            candidate is None
            or candidate.group(1) != name
            or int(candidate.group(2)) != index
        ):
            return None
    return name


def _emit_norm_slice_argument(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Return the Rust slice expression passed to a local norm helper."""
    args = tuple(
        _emit_expr_ref(
            arg,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        )
        for arg in expr.args
    )
    slice_name = _match_contiguous_slice(args)
    if slice_name is not None:
        return slice_name
    return "&[" + ", ".join(args) + "]"


def _emit_norm_slice_and_p_arguments(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> tuple[str, str]:
    """Return the Rust slice plus ``p`` expression for a p-norm helper."""
    value_expr = SX(SXNode.make("norm2", expr.node.args[:-1]))
    vector_ref = _emit_norm_slice_argument(
        value_expr,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    p_ref = _emit_expr_ref(
        SX(expr.node.args[-1]),
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    return vector_ref, p_ref


def _emit_matrix_literal(
    values: tuple[float, ...], scalar_type: RustScalarType
) -> str:
    """Return an inline Rust slice literal for a constant matrix."""
    return (
        "&["
        + ", ".join(_format_float(value, scalar_type) for value in values)
        + "]"
    )


def _emit_matrix_vector_argument(
    values: tuple[SX, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Return the Rust slice expression passed to matrix helpers."""
    refs = tuple(
        _emit_expr_ref(
            value,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        )
        for value in values
    )
    slice_name = _match_contiguous_slice(refs)
    if slice_name is not None:
        return slice_name
    return "&[" + ", ".join(refs) + "]"


def _emit_custom_scalar_call(
    name: str,
    value: SX,
    params: tuple[SX, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    w_ref = _emit_matrix_vector_argument(
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    refs = [
        _emit_expr_ref(
            value,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        ),
        w_ref,
    ]
    return f"{name}(" + ", ".join(refs) + ")"


def _emit_custom_scalar_derivative_call(
    name: str,
    derivative_kind: str,
    value: SX,
    params: tuple[SX, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    w_ref = _emit_matrix_vector_argument(
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    refs = [
        _emit_expr_ref(
            value,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        ),
        w_ref,
    ]
    return f"{name}_{derivative_kind}(" + ", ".join(refs) + ")"


def _emit_custom_scalar_hvp_call(
    name: str,
    value: SX,
    tangent: SX,
    params: tuple[SX, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    w_ref = _emit_matrix_vector_argument(
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    refs = [
        _emit_expr_ref(
            value,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        ),
        _emit_expr_ref(
            tangent,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        ),
        w_ref,
    ]
    return f"{name}_hvp(" + ", ".join(refs) + ")"


def _emit_custom_vector_call(
    name: str,
    value: SXVector,
    params: tuple[SX, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    x_ref = _emit_matrix_vector_argument(
        value.elements,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    w_ref = _emit_matrix_vector_argument(
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    refs = [x_ref, w_ref]
    return f"{name}(" + ", ".join(refs) + ")"


def _emit_custom_vector_component_call(
    name: str,
    derivative_kind: str,
    index: int,
    value: SXVector,
    tangent: SXVector | None,
    params: tuple[SX, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    x_ref = _emit_matrix_vector_argument(
        value.elements,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    refs: list[str] = [str(index), x_ref]
    if tangent is not None:
        refs.append(
            _emit_matrix_vector_argument(
                tangent.elements,
                scalar_bindings,
                workspace_map,
                backend_mode,
                scalar_type,
                math_library,
            )
        )
    refs.append(
        _emit_matrix_vector_argument(
            params,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        )
    )
    suffix = (
        "component"
        if derivative_kind in {"jacobian", "hvp"}
        else derivative_kind
    )
    return f"{name}_{derivative_kind}_{suffix}(" + ", ".join(refs) + ")"


def _emit_custom_vector_hessian_entry_call(
    name: str,
    row: int,
    col: int,
    value: SXVector,
    params: tuple[SX, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    x_ref = _emit_matrix_vector_argument(
        value.elements,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    refs = [
        str(row),
        str(col),
        x_ref,
        _emit_matrix_vector_argument(
            params,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        ),
    ]
    return f"{name}_hessian_entry(" + ", ".join(refs) + ")"


def _emit_norm_abs_expr(
    value_ref: str,
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Return a valid Rust absolute-value expression for iterator items."""
    if backend_mode == "std":
        return f"{value_ref}.abs()"
    if math_library is None:
        raise ValueError("no_std math calls require a resolved math library")
    return (
        f"{math_library}::{_math_function_name('abs', scalar_type)}"
        f"(*{value_ref})"
    )


_BINARY_INFIX_OPERATORS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}

_BINARY_MATH_OPERATORS = {
    "atan2": "atan2",
    "hypot": "hypot",
    "min": "min",
    "max": "max",
}

_UNARY_MATH_OPERATORS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "expm1": "expm1",
    "log": "log",
    "log1p": "log1p",
    "sqrt": "sqrt",
    "cbrt": "cbrt",
    "erf": "erf",
    "erfc": "erfc",
    "floor": "floor",
    "ceil": "ceil",
    "round": "round",
    "trunc": "trunc",
    "fract": "fract",
    "signum": "signum",
    "abs": "abs",
}

_REDUCTION_HELPERS = {
    "sum": "vec_sum",
    "prod": "vec_prod",
    "reduce_max": "vec_max",
    "reduce_min": "vec_min",
    "mean": "vec_mean",
    "norm2": "norm2",
    "norm2sq": "norm2sq",
    "norm1": "norm1",
    "norm_inf": "norm_inf",
    "norm_p_to_p": "norm_p_to_p",
    "norm_p": "norm_p",
}


def _emit_binary_infix_node(
    expr: SX,
    args: tuple[str, ...],
    *_unused,
) -> str:
    """Emit a simple binary infix Rust expression."""
    operator = _BINARY_INFIX_OPERATORS[expr.op]
    return f"{args[0]} {operator} {args[1]}"


def _emit_binary_math_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a binary Rust math call."""
    del scalar_bindings, workspace_map
    return _emit_math_call(
        _BINARY_MATH_OPERATORS[expr.op],
        args,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_pow_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a Rust power expression with a square fast path."""
    del scalar_bindings, workspace_map
    if expr.args[1].op == "const" and expr.args[1].value == 2.0:
        return f"{args[0]} * {args[0]}"
    return _emit_math_call(
        "pow",
        args,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_unary_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a unary Rust expression or math call."""
    del scalar_bindings, workspace_map
    if expr.op == "neg":
        return f"-{args[0]}"
    if expr.op in {"erf", "erfc"}:
        return f"{expr.op}({args[0]})"
    return _emit_math_call(
        _UNARY_MATH_OPERATORS[expr.op],
        args,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_custom_scalar_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a custom scalar primitive call."""
    del args
    spec, value, params = parse_custom_scalar_args(expr.name, expr.args)
    return _emit_custom_scalar_call(
        spec.name,
        value,
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_custom_scalar_derivative_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
    derivative_kind: str,
) -> str:
    """Emit a custom scalar derivative call."""
    del args
    spec, value, params = parse_custom_scalar_args(expr.name, expr.args)
    return _emit_custom_scalar_derivative_call(
        spec.name,
        derivative_kind,
        value,
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_custom_scalar_jacobian_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a custom scalar Jacobian call."""
    return _emit_custom_scalar_derivative_node(
        expr,
        args,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
        "jacobian",
    )


def _emit_custom_scalar_hessian_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a custom scalar Hessian call."""
    return _emit_custom_scalar_derivative_node(
        expr,
        args,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
        "hessian",
    )


def _emit_custom_scalar_hvp_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a custom scalar HVP call."""
    del args
    spec, value, tangent, params = parse_custom_scalar_hvp_args(
        expr.name,
        expr.args,
    )
    return _emit_custom_scalar_hvp_call(
        spec.name,
        value,
        tangent,
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_custom_vector_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a custom vector primitive call."""
    del args
    spec, value, params = parse_custom_vector_args(expr.name, expr.args)
    return _emit_custom_vector_call(
        spec.name,
        value,
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_custom_vector_jacobian_component_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a custom vector Jacobian component call."""
    del args
    spec, index, value, params = parse_custom_vector_jacobian_component_args(
        expr.name,
        expr.args,
    )
    return _emit_custom_vector_component_call(
        spec.name,
        "jacobian",
        index,
        value,
        None,
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_custom_vector_hvp_component_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a custom vector HVP component call."""
    del args
    spec, index, value, tangent, params = (
        parse_custom_vector_hvp_component_args(
            expr.name,
            expr.args,
        )
    )
    return _emit_custom_vector_component_call(
        spec.name,
        "hvp",
        index,
        value,
        tangent,
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_custom_vector_hessian_entry_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a custom vector Hessian entry call."""
    del args
    spec, row, col, value, params = parse_custom_vector_hessian_entry_args(
        expr.name,
        expr.args,
    )
    return _emit_custom_vector_hessian_entry_call(
        spec.name,
        row,
        col,
        value,
        params,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _emit_matvec_component_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a matrix-vector product component call."""
    rows, cols, row, matrix_values, x_values = parse_matvec_component_args(
        expr.args
    )
    matrix_ref = _emit_matrix_literal(matrix_values, scalar_type)
    x_ref = _emit_matrix_vector_argument(
        x_values,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    del args
    return f"matvec_component({matrix_ref}, {rows}, {cols}, {row}, {x_ref})"


def _emit_quadform_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a quadratic form call."""
    size, matrix_values, x_values = parse_quadform_args(expr.args)
    matrix_ref = _emit_matrix_literal(matrix_values, scalar_type)
    x_ref = _emit_matrix_vector_argument(
        x_values,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    del args
    return f"quadform({matrix_ref}, {size}, {x_ref})"


def _emit_bilinear_form_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a bilinear form call."""
    rows, cols, matrix_values, x_values, y_values = parse_bilinear_form_args(
        expr.args
    )
    matrix_ref = _emit_matrix_literal(matrix_values, scalar_type)
    x_ref = _emit_matrix_vector_argument(
        x_values,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    y_ref = _emit_matrix_vector_argument(
        y_values,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    del args
    return f"bilinear_form({x_ref}, {matrix_ref}, {rows}, {cols}, {y_ref})"


def _emit_vector_reduction_node(
    expr: SX,
    args: tuple[str, ...],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit a vector reduction or norm call."""
    if expr.op in {"norm_p_to_p", "norm_p"}:
        vector_ref, p_ref = _emit_norm_slice_and_p_arguments(
            expr,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        )
        return f"{_REDUCTION_HELPERS[expr.op]}({vector_ref}, {p_ref})"

    del args
    vector_ref = _emit_norm_slice_argument(
        expr,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    return f"{_REDUCTION_HELPERS[expr.op]}({vector_ref})"


_NODE_EXPR_DISPATCH = {
    "add": _emit_binary_infix_node,
    "sub": _emit_binary_infix_node,
    "mul": _emit_binary_infix_node,
    "div": _emit_binary_infix_node,
    "pow": _emit_pow_node,
    "atan2": _emit_binary_math_node,
    "hypot": _emit_binary_math_node,
    "min": _emit_binary_math_node,
    "max": _emit_binary_math_node,
    "neg": _emit_unary_node,
    "sin": _emit_unary_node,
    "cos": _emit_unary_node,
    "tan": _emit_unary_node,
    "asin": _emit_unary_node,
    "acos": _emit_unary_node,
    "atan": _emit_unary_node,
    "asinh": _emit_unary_node,
    "acosh": _emit_unary_node,
    "atanh": _emit_unary_node,
    "sinh": _emit_unary_node,
    "cosh": _emit_unary_node,
    "tanh": _emit_unary_node,
    "exp": _emit_unary_node,
    "expm1": _emit_unary_node,
    "log": _emit_unary_node,
    "log1p": _emit_unary_node,
    "sqrt": _emit_unary_node,
    "cbrt": _emit_unary_node,
    "erf": _emit_unary_node,
    "erfc": _emit_unary_node,
    "floor": _emit_unary_node,
    "ceil": _emit_unary_node,
    "round": _emit_unary_node,
    "trunc": _emit_unary_node,
    "fract": _emit_unary_node,
    "signum": _emit_unary_node,
    "abs": _emit_unary_node,
    "custom_scalar": _emit_custom_scalar_node,
    "custom_scalar_jacobian": _emit_custom_scalar_jacobian_node,
    "custom_scalar_hessian": _emit_custom_scalar_hessian_node,
    "custom_scalar_hvp": _emit_custom_scalar_hvp_node,
    "custom_vector": _emit_custom_vector_node,
    "custom_vector_jacobian_component": (
        _emit_custom_vector_jacobian_component_node
    ),
    "custom_vector_hvp_component": _emit_custom_vector_hvp_component_node,
    "custom_vector_hessian_entry": _emit_custom_vector_hessian_entry_node,
    "matvec_component": _emit_matvec_component_node,
    "quadform": _emit_quadform_node,
    "bilinear_form": _emit_bilinear_form_node,
    "sum": _emit_vector_reduction_node,
    "prod": _emit_vector_reduction_node,
    "reduce_max": _emit_vector_reduction_node,
    "reduce_min": _emit_vector_reduction_node,
    "mean": _emit_vector_reduction_node,
    "norm2": _emit_vector_reduction_node,
    "norm2sq": _emit_vector_reduction_node,
    "norm1": _emit_vector_reduction_node,
    "norm_inf": _emit_vector_reduction_node,
    "norm_p_to_p": _emit_vector_reduction_node,
    "norm_p": _emit_vector_reduction_node,
}


def _emit_node_expr(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Emit the Rust expression used to compute a workspace node."""
    if expr.op == "const":
        return _format_float(expr.value, scalar_type)
    if expr.op == "symbol":
        return scalar_bindings[expr.node]

    args = tuple(
        _emit_expr_ref(
            arg,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        )
        for arg in expr.args
    )

    handler = _NODE_EXPR_DISPATCH.get(expr.op)
    if handler is None:
        raise ValueError(f"unsupported Rust codegen operation {expr.op!r}")
    return handler(
        expr,
        args,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


_LIBM_FUNCTIONS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "exp": "exp",
    "expm1": "expm1",
    "log": "log",
    "log1p": "log1p",
    "pow": "pow",
    "sqrt": "sqrt",
    "cbrt": "cbrt",
    "floor": "floor",
    "ceil": "ceil",
    "round": "round",
    "trunc": "trunc",
    "hypot": "hypot",
    "abs": "fabs",
    "max": "fmax",
    "min": "fmin",
}


def _math_function_name(op: str, scalar_type: RustScalarType) -> str:
    """Return the backend math function name for a scalar type."""
    base_name = _LIBM_FUNCTIONS[op]
    if scalar_type == "f32":
        return f"{base_name}f"
    return base_name
