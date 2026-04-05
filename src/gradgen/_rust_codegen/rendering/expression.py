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
from ...sx import SX, SXNode, SXVector, parse_bilinear_form_args, parse_matvec_component_args, parse_quadform_args
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
        trunc_expr = _emit_math_call("trunc", args, backend_mode, scalar_type, math_library)
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
        return f"{math_library}::{_math_function_name(op, scalar_type)}({args[0]}, {args[1]})"
    if op in {"atan2", "hypot", "max", "min"}:
        return f"{math_library}::{_math_function_name(op, scalar_type)}({args[0]}, {args[1]})"
    return f"{math_library}::{_math_function_name(op, scalar_type)}({args[0]})"


def _match_contiguous_slice(args: tuple[str, ...]) -> str | None:
    """Return the slice name when args are exactly ``name[0], name[1], ...``."""
    if not args:
        return None

    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", args[0])
    if match is None:
        return None
    name = match.group(1)

    for index, arg in enumerate(args):
        candidate = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", arg)
        if candidate is None or candidate.group(1) != name or int(candidate.group(2)) != index:
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
        _emit_expr_ref(arg, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library)
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


def _emit_matrix_literal(values: tuple[float, ...], scalar_type: RustScalarType) -> str:
    """Return an inline Rust slice literal for a constant matrix."""
    return "&[" + ", ".join(_format_float(value, scalar_type) for value in values) + "]"


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
        _emit_expr_ref(value, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library)
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
        params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    refs = [
        _emit_expr_ref(value, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library),
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
        params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    refs = [
        _emit_expr_ref(value, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library),
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
        params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    refs = [
        _emit_expr_ref(value, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library),
        _emit_expr_ref(tangent, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library),
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
        value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    w_ref = _emit_matrix_vector_argument(
        params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
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
        value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    refs: list[str] = [str(index), x_ref]
    if tangent is not None:
        refs.append(
            _emit_matrix_vector_argument(
                tangent.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
            )
        )
    refs.append(
        _emit_matrix_vector_argument(
            params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
    )
    suffix = "component" if derivative_kind in {"jacobian", "hvp"} else derivative_kind
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
        value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    refs = [
        str(row),
        str(col),
        x_ref,
        _emit_matrix_vector_argument(
            params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
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
    return f"{math_library}::{_math_function_name('abs', scalar_type)}(*{value_ref})"


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
        _emit_expr_ref(arg, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library)
        for arg in expr.args
    )

    if expr.op == "add":
        return f"{args[0]} + {args[1]}"
    if expr.op == "sub":
        return f"{args[0]} - {args[1]}"
    if expr.op == "mul":
        return f"{args[0]} * {args[1]}"
    if expr.op == "div":
        return f"{args[0]} / {args[1]}"
    if expr.op == "pow":
        if expr.args[1].op == "const" and expr.args[1].value == 2.0:
            return f"{args[0]} * {args[0]}"
        return _emit_math_call("pow", args, backend_mode, scalar_type, math_library)
    if expr.op == "custom_scalar":
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
    if expr.op == "custom_scalar_jacobian":
        spec, value, params = parse_custom_scalar_args(expr.name, expr.args)
        return _emit_custom_scalar_derivative_call(
            spec.name,
            "jacobian",
            value,
            params,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        )
    if expr.op == "custom_scalar_hessian":
        spec, value, params = parse_custom_scalar_args(expr.name, expr.args)
        return _emit_custom_scalar_derivative_call(
            spec.name,
            "hessian",
            value,
            params,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        )
    if expr.op == "custom_scalar_hvp":
        spec, value, tangent, params = parse_custom_scalar_hvp_args(expr.name, expr.args)
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
    if expr.op == "custom_vector":
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
    if expr.op == "custom_vector_jacobian_component":
        spec, index, value, params = parse_custom_vector_jacobian_component_args(expr.name, expr.args)
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
    if expr.op == "custom_vector_hvp_component":
        spec, index, value, tangent, params = parse_custom_vector_hvp_component_args(expr.name, expr.args)
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
    if expr.op == "custom_vector_hessian_entry":
        spec, row, col, value, params = parse_custom_vector_hessian_entry_args(expr.name, expr.args)
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
    if expr.op == "atan2":
        return _emit_math_call("atan2", args, backend_mode, scalar_type, math_library)
    if expr.op == "hypot":
        return _emit_math_call("hypot", args, backend_mode, scalar_type, math_library)
    if expr.op == "neg":
        return f"-{args[0]}"
    if expr.op == "sin":
        return _emit_math_call("sin", args, backend_mode, scalar_type, math_library)
    if expr.op == "cos":
        return _emit_math_call("cos", args, backend_mode, scalar_type, math_library)
    if expr.op == "tan":
        return _emit_math_call("tan", args, backend_mode, scalar_type, math_library)
    if expr.op == "asin":
        return _emit_math_call("asin", args, backend_mode, scalar_type, math_library)
    if expr.op == "acos":
        return _emit_math_call("acos", args, backend_mode, scalar_type, math_library)
    if expr.op == "atan":
        return _emit_math_call("atan", args, backend_mode, scalar_type, math_library)
    if expr.op == "asinh":
        return _emit_math_call("asinh", args, backend_mode, scalar_type, math_library)
    if expr.op == "acosh":
        return _emit_math_call("acosh", args, backend_mode, scalar_type, math_library)
    if expr.op == "atanh":
        return _emit_math_call("atanh", args, backend_mode, scalar_type, math_library)
    if expr.op == "sinh":
        return _emit_math_call("sinh", args, backend_mode, scalar_type, math_library)
    if expr.op == "cosh":
        return _emit_math_call("cosh", args, backend_mode, scalar_type, math_library)
    if expr.op == "tanh":
        return _emit_math_call("tanh", args, backend_mode, scalar_type, math_library)
    if expr.op == "exp":
        return _emit_math_call("exp", args, backend_mode, scalar_type, math_library)
    if expr.op == "expm1":
        return _emit_math_call("expm1", args, backend_mode, scalar_type, math_library)
    if expr.op == "log":
        return _emit_math_call("log", args, backend_mode, scalar_type, math_library)
    if expr.op == "log1p":
        return _emit_math_call("log1p", args, backend_mode, scalar_type, math_library)
    if expr.op == "sqrt":
        return _emit_math_call("sqrt", args, backend_mode, scalar_type, math_library)
    if expr.op == "cbrt":
        return _emit_math_call("cbrt", args, backend_mode, scalar_type, math_library)
    if expr.op == "erf":
        return "erf(" + args[0] + ")"
    if expr.op == "erfc":
        return "erfc(" + args[0] + ")"
    if expr.op == "floor":
        return _emit_math_call("floor", args, backend_mode, scalar_type, math_library)
    if expr.op == "ceil":
        return _emit_math_call("ceil", args, backend_mode, scalar_type, math_library)
    if expr.op == "round":
        return _emit_math_call("round", args, backend_mode, scalar_type, math_library)
    if expr.op == "trunc":
        return _emit_math_call("trunc", args, backend_mode, scalar_type, math_library)
    if expr.op == "fract":
        return _emit_math_call("fract", args, backend_mode, scalar_type, math_library)
    if expr.op == "signum":
        return _emit_math_call("signum", args, backend_mode, scalar_type, math_library)
    if expr.op == "matvec_component":
        rows, cols, row, matrix_values, x_values = parse_matvec_component_args(expr.args)
        matrix_ref = _emit_matrix_literal(matrix_values, scalar_type)
        x_ref = _emit_matrix_vector_argument(
            x_values, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"matvec_component({matrix_ref}, {rows}, {cols}, {row}, {x_ref})"
    if expr.op == "quadform":
        size, matrix_values, x_values = parse_quadform_args(expr.args)
        matrix_ref = _emit_matrix_literal(matrix_values, scalar_type)
        x_ref = _emit_matrix_vector_argument(
            x_values, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"quadform({matrix_ref}, {size}, {x_ref})"
    if expr.op == "bilinear_form":
        rows, cols, matrix_values, x_values, y_values = parse_bilinear_form_args(expr.args)
        matrix_ref = _emit_matrix_literal(matrix_values, scalar_type)
        x_ref = _emit_matrix_vector_argument(
            x_values, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        y_ref = _emit_matrix_vector_argument(
            y_values, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"bilinear_form({x_ref}, {matrix_ref}, {rows}, {cols}, {y_ref})"
    if expr.op == "sum":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"vec_sum({vector_ref})"
    if expr.op == "prod":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"vec_prod({vector_ref})"
    if expr.op == "reduce_max":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"vec_max({vector_ref})"
    if expr.op == "reduce_min":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"vec_min({vector_ref})"
    if expr.op == "mean":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"vec_mean({vector_ref})"
    if expr.op == "norm2":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"norm2({vector_ref})"
    if expr.op == "norm2sq":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"norm2sq({vector_ref})"
    if expr.op == "norm1":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"norm1({vector_ref})"
    if expr.op == "norm_inf":
        vector_ref = _emit_norm_slice_argument(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"norm_inf({vector_ref})"
    if expr.op == "norm_p_to_p":
        vector_ref, p_ref = _emit_norm_slice_and_p_arguments(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"norm_p_to_p({vector_ref}, {p_ref})"
    if expr.op == "norm_p":
        vector_ref, p_ref = _emit_norm_slice_and_p_arguments(
            expr, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return f"norm_p({vector_ref}, {p_ref})"
    if expr.op == "abs":
        return _emit_math_call("abs", args, backend_mode, scalar_type, math_library)
    if expr.op == "max":
        return _emit_math_call("max", args, backend_mode, scalar_type, math_library)
    if expr.op == "min":
        return _emit_math_call("min", args, backend_mode, scalar_type, math_library)

    raise ValueError(f"unsupported Rust codegen operation {expr.op!r}")


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
