"""Expression rendering and helper emission for Rust code generation."""

from __future__ import annotations

import re

from ..custom_elementary import (
    get_registered_elementary_function,
    parse_custom_vector_hessian_entry_args,
    parse_custom_vector_hvp_component_args,
    parse_custom_vector_jacobian_component_args,
    render_custom_rust_snippet,
)
from ..sx import SX, SXNode, SXVector, parse_matvec_component_args
from .config import RustBackendMode, RustScalarType
from .validation import validate_scalar_type


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


def _emit_custom_vector_hessian_output_helper_call(
    output_arg: SXVector,
    output_name: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str | None:
    """Return a direct flat-output helper call for custom vector Hessians."""
    if not output_arg.elements:
        return None

    matched = _match_custom_vector_hessian_output(
        output_arg,
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if matched is None:
        return None

    name, x_ref, w_ref = matched
    args = [x_ref, w_ref, output_name]
    return f"{name}_hessian(" + ", ".join(args) + ");"


def _emit_matvec_output_helper_call(
    output_arg: SX | SXVector,
    output_name: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str | None:
    """Return a single helper call when ``output_arg`` is exactly a matvec."""
    if not isinstance(output_arg, SXVector) or not output_arg.elements:
        return None
    parsed_exprs: list[SX] = []
    for element in output_arg:
        matched = _match_passthrough_matvec_component(element)
        if matched is None:
            return None
        parsed_exprs.append(matched)

    parsed = [parse_matvec_component_args(element.args) for element in parsed_exprs]
    rows, cols, _row, matrix_values, x_values = parsed[0]
    if rows != len(output_arg):
        return None
    for index, candidate in enumerate(parsed):
        cand_rows, cand_cols, cand_row, cand_matrix_values, cand_x_values = candidate
        if (
            cand_rows != rows
            or cand_cols != cols
            or cand_row != index
            or cand_matrix_values != matrix_values
            or cand_x_values != x_values
        ):
            return None

    matrix_ref = _emit_matrix_literal(matrix_values, scalar_type)
    x_ref = _emit_matrix_vector_argument(
        x_values, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    return f"matvec({matrix_ref}, {rows}, {cols}, {x_ref}, {output_name});"


def _emit_custom_vector_output_helper_call(
    output_arg: SX | SXVector,
    output_name: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str | None:
    """Return a direct output helper call for custom vector Jacobian/HVP results."""
    if not isinstance(output_arg, SXVector) or not output_arg.elements:
        return None

    jacobian_call = _match_custom_vector_derivative_output(
        output_arg,
        derivative_kind="jacobian",
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if jacobian_call is not None:
        name, x_ref, tangent_ref, w_ref = jacobian_call
        _ = tangent_ref
        args = [x_ref, w_ref, output_name]
        return f"{name}_jacobian(" + ", ".join(args) + ");"

    hvp_call = _match_custom_vector_derivative_output(
        output_arg,
        derivative_kind="hvp",
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if hvp_call is not None:
        name, x_ref, tangent_ref, w_ref = hvp_call
        if tangent_ref is None:
            return None
        args = [x_ref, tangent_ref, w_ref, output_name]
        return f"{name}_hvp(" + ", ".join(args) + ");"

    hessian_call = _emit_custom_vector_hessian_output_helper_call(
        output_arg,
        output_name,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    if hessian_call is not None:
        return hessian_call

    return None


def _identify_direct_custom_output_marker(
    output_arg: SX | SXVector,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> tuple[str, str] | None:
    """Return the custom helper marker used by a direct output helper call."""
    if not isinstance(output_arg, SXVector) or not output_arg.elements:
        return None

    jacobian_call = _match_custom_vector_derivative_output(
        output_arg,
        derivative_kind="jacobian",
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if jacobian_call is not None:
        return jacobian_call[0], "jacobian"

    hvp_call = _match_custom_vector_derivative_output(
        output_arg,
        derivative_kind="hvp",
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if hvp_call is not None:
        return hvp_call[0], "hvp"

    hessian_call = _match_custom_vector_hessian_output(
        output_arg,
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if hessian_call is not None:
        return hessian_call[0], "hessian"

    return None


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


def _build_shared_helper_lines(
    nodes: tuple[SXNode, ...],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
    *,
    suppressed_custom_wrappers: set[tuple[str, str]] | None = None,
) -> tuple[str, ...]:
    """Return module-scope helper definitions needed by generated kernels."""
    used_ops = {node.op for node in nodes}
    lines: list[str] = []

    if {"matvec_component", "quadform", "bilinear_form"} & used_ops:
        lines.extend(
            [
                f"fn matvec_component(matrix: &[{scalar_type}], rows: usize, cols: usize, row: usize, x: &[{scalar_type}]) -> {scalar_type} {{",
                "    let start = row * cols;",
                "    matrix[start..start + cols]",
                "        .iter()",
                "        .zip(x.iter())",
                "        .map(|(entry, value)| *entry * *value)",
                "        .sum()",
                "}",
                f"fn matvec(matrix: &[{scalar_type}], rows: usize, cols: usize, x: &[{scalar_type}], y: &mut [{scalar_type}]) {{",
                "    for row in 0..rows {",
                "        y[row] = matvec_component(matrix, rows, cols, row, x);",
                "    }",
                "}",
                f"fn bilinear_form(x: &[{scalar_type}], matrix: &[{scalar_type}], rows: usize, cols: usize, y: &[{scalar_type}]) -> {scalar_type} {{",
                "    x.iter()",
                "        .enumerate()",
                "        .map(|(row, x_value)| {",
                "            let start = row * cols;",
                "            let row_sum: "
                + scalar_type
                + " = matrix[start..start + cols]",
                "                .iter()",
                "                .zip(y.iter())",
                "                .map(|(entry, y_value)| *entry * *y_value)",
                "                .sum();",
                "            *x_value * row_sum",
                "        })",
                "        .sum()",
                "}",
                f"fn quadform(matrix: &[{scalar_type}], size: usize, x: &[{scalar_type}]) -> {scalar_type} {{",
                "    bilinear_form(x, matrix, size, size, x)",
                "}",
            ]
        )

    if "sum" in used_ops or "mean" in used_ops:
        lines.extend(
            [
                f"fn vec_sum(values: &[{scalar_type}]) -> {scalar_type} {{",
                "    values.iter().copied().sum()",
                "}",
            ]
        )

    if "prod" in used_ops:
        lines.extend(
            [
                f"fn vec_prod(values: &[{scalar_type}]) -> {scalar_type} {{",
                f"    values.iter().copied().product::<{scalar_type}>()",
                "}",
            ]
        )

    if "reduce_max" in used_ops:
        lines.extend(
            [
                f"fn vec_max(values: &[{scalar_type}]) -> {scalar_type} {{",
                f"    values.iter().copied().fold({scalar_type}::NEG_INFINITY, |acc, value| acc.max(value))",
                "}",
            ]
        )

    if "reduce_min" in used_ops:
        lines.extend(
            [
                f"fn vec_min(values: &[{scalar_type}]) -> {scalar_type} {{",
                f"    values.iter().copied().fold({scalar_type}::INFINITY, |acc, value| acc.min(value))",
                "}",
            ]
        )

    if "mean" in used_ops:
        lines.extend(
            [
                f"fn vec_mean(values: &[{scalar_type}]) -> {scalar_type} {{",
                f"    vec_sum(values) / (values.len() as {scalar_type})",
                "}",
            ]
        )

    custom_helper_lines = _build_custom_helper_lines(
        nodes,
        scalar_type,
        math_library,
        suppressed_wrappers=suppressed_custom_wrappers or set(),
    )
    if custom_helper_lines:
        lines.extend(custom_helper_lines)

    if "norm2" in used_ops:
        sqrt_expr = _emit_math_call("sqrt", ("sum",), backend_mode, scalar_type, math_library)
        lines.extend(
            [
                f"fn norm2(values: &[{scalar_type}]) -> {scalar_type} {{",
                f"    let sum: {scalar_type} = values.iter().map(|value| *value * *value).sum();",
                f"    {sqrt_expr}",
                "}",
            ]
        )

    if "norm2sq" in used_ops:
        lines.extend(
            [
                f"fn norm2sq(values: &[{scalar_type}]) -> {scalar_type} {{",
                "    values.iter().map(|value| *value * *value).sum()",
                "}",
            ]
        )

    if "norm1" in used_ops:
        abs_term = _emit_norm_abs_expr("value", backend_mode, scalar_type, math_library)
        lines.extend(
            [
                f"fn norm1(values: &[{scalar_type}]) -> {scalar_type} {{",
                f"    values.iter().map(|value| {abs_term}).sum()",
                "}",
            ]
        )

    if "norm_inf" in used_ops:
        abs_term = _emit_norm_abs_expr("value", backend_mode, scalar_type, math_library)
        lines.extend(
            [
                f"fn norm_inf(values: &[{scalar_type}]) -> {scalar_type} {{",
                f"    values.iter().fold(0.0_{scalar_type}, |acc, value| acc.max({abs_term}))",
                "}",
            ]
        )

    if "norm_p_to_p" in used_ops:
        abs_term = _emit_norm_abs_expr("value", backend_mode, scalar_type, math_library)
        pow_term = _emit_math_call("pow", (abs_term, "p"), backend_mode, scalar_type, math_library)
        lines.extend(
            [
                f"fn norm_p_to_p(values: &[{scalar_type}], p: {scalar_type}) -> {scalar_type} {{",
                f"    values.iter().map(|value| {pow_term}).sum()",
                "}",
            ]
        )

    if "norm_p" in used_ops:
        pow_expr = _emit_math_call(
            "pow",
            (f"norm_p_to_p(values, p)", f"1.0_{scalar_type} / p"),
            backend_mode,
            scalar_type,
            math_library,
        )
        lines.extend(
            [
                f"fn norm_p(values: &[{scalar_type}], p: {scalar_type}) -> {scalar_type} {{",
                f"    {pow_expr}",
                "}",
            ]
        )

    if "erf" in used_ops or "erfc" in used_ops:
        abs_expr = _emit_math_call("abs", ("x",), backend_mode, scalar_type, math_library)
        exp_expr = _emit_math_call("exp", ("poly",), backend_mode, scalar_type, math_library)
        lines.extend(
            [
                f"fn erf(x: {scalar_type}) -> {scalar_type} {{",
                f"    let one = 1.0_{scalar_type};",
                f"    let half = 0.5_{scalar_type};",
                f"    let sign = if x < 0.0_{scalar_type} {{ -one }} else {{ one }};",
                f"    let x_abs = {abs_expr};",
                f"    let t = one / (one + half * x_abs);",
                f"    let poly = -x_abs * x_abs - 1.26551223_{scalar_type}",
                f"        + t * (1.00002368_{scalar_type}",
                f"        + t * (0.37409196_{scalar_type}",
                f"        + t * (0.09678418_{scalar_type}",
                f"        + t * (-0.18628806_{scalar_type}",
                f"        + t * (0.27886807_{scalar_type}",
                f"        + t * (-1.13520398_{scalar_type}",
                f"        + t * (1.48851587_{scalar_type}",
                f"        + t * (-0.82215223_{scalar_type}",
                f"        + t * 0.17087277_{scalar_type})))))))));",
                f"    let tau = t * {exp_expr};",
                f"    sign * (one - tau)",
                "}",
            ]
        )
    if "erfc" in used_ops:
        lines.extend(
            [
                f"fn erfc(x: {scalar_type}) -> {scalar_type} {{",
                f"    1.0_{scalar_type} - erf(x)",
                "}",
            ]
        )

    return tuple(lines)


def _build_custom_helper_lines(
    nodes: tuple[SXNode, ...],
    scalar_type: RustScalarType,
    math_library: str | None,
    *,
    suppressed_wrappers: set[tuple[str, str]],
) -> tuple[str, ...]:
    """Return shared helper lines for registered custom elementary functions."""
    emitted: list[str] = []
    seen: set[tuple[str, str]] = set()

    for node in nodes:
        if not node.op.startswith("custom_"):
            continue
        if node.name is None:
            raise ValueError("custom elementary nodes must carry a registered name")
        spec = get_registered_elementary_function(node.name)
        helper_kind: str
        snippet: str | None
        if node.op in {"custom_scalar", "custom_vector"}:
            helper_kind = "primal"
            snippet = spec.rust_primal
        elif node.op in {"custom_scalar_jacobian", "custom_vector_jacobian_component"}:
            helper_kind = "jacobian"
            snippet = spec.rust_jacobian
        elif node.op in {"custom_scalar_hvp", "custom_vector_hvp_component"}:
            helper_kind = "hvp"
            snippet = spec.rust_hvp
        elif node.op in {"custom_scalar_hessian", "custom_vector_hessian_entry"}:
            helper_kind = "hessian"
            snippet = spec.rust_hessian
        else:
            continue

        marker = (spec.name, helper_kind)
        if marker in seen:
            continue
        if snippet is None:
            raise ValueError(
                f"custom function {spec.name!r} requires rust_{helper_kind} for Rust code generation"
            )
        seen.add(marker)
        emitted.extend(render_custom_rust_snippet(snippet, scalar_type=scalar_type, math_library=math_library))
        if not spec.is_scalar and helper_kind == "jacobian" and (spec.name, "jacobian") not in suppressed_wrappers:
            emitted.extend(_build_custom_vector_jacobian_wrapper_lines(spec, scalar_type))
        if not spec.is_scalar and helper_kind == "hvp" and (spec.name, "hvp") not in suppressed_wrappers:
            emitted.extend(_build_custom_vector_hvp_wrapper_lines(spec, scalar_type))
        if not spec.is_scalar and helper_kind == "hessian" and (spec.name, "hessian") not in suppressed_wrappers:
            emitted.extend(_build_custom_vector_hessian_wrapper_lines(spec, scalar_type))

    return tuple(emitted)


def _build_custom_vector_jacobian_wrapper_lines(
    spec: object,
    scalar_type: RustScalarType,
) -> tuple[str, ...]:
    spec = spec
    vector_dim = getattr(spec, "vector_dim")
    name = getattr(spec, "name")
    return (
        f"fn {name}_jacobian_component(index: usize, x: &[{scalar_type}], w: &[{scalar_type}]) -> {scalar_type} {{",
        f"    let mut out = [0.0_{scalar_type}; {vector_dim}];",
        f"    {name}_jacobian(x, w, &mut out);",
        "    out[index]",
        "}",
    )


def _build_custom_vector_hvp_wrapper_lines(
    spec: object,
    scalar_type: RustScalarType,
) -> tuple[str, ...]:
    spec = spec
    vector_dim = getattr(spec, "vector_dim")
    name = getattr(spec, "name")
    return (
        f"fn {name}_hvp_component(index: usize, x: &[{scalar_type}], v_x: &[{scalar_type}], w: &[{scalar_type}]) -> {scalar_type} {{",
        f"    let mut out = [0.0_{scalar_type}; {vector_dim}];",
        f"    {name}_hvp(x, v_x, w, &mut out);",
        "    out[index]",
        "}",
    )


def _build_custom_vector_hessian_wrapper_lines(
    spec: object,
    scalar_type: RustScalarType,
) -> tuple[str, ...]:
    spec = spec
    vector_dim = getattr(spec, "vector_dim")
    name = getattr(spec, "name")
    flat_size = vector_dim * vector_dim
    return (
        f"fn {name}_hessian_entry(row: usize, col: usize, x: &[{scalar_type}], w: &[{scalar_type}]) -> {scalar_type} {{",
        f"    let mut out = [0.0_{scalar_type}; {flat_size}];",
        f"    {name}_hessian(x, w, &mut out);",
        f"    out[(row * {vector_dim}) + col]",
        "}",
    )


def _match_custom_vector_derivative_output(
    output_arg: SXVector,
    *,
    derivative_kind: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> tuple[str, str, str | None, str] | None:
    """Match a full custom vector derivative output emitted as components."""
    expected_op = (
        "custom_vector_jacobian_component"
        if derivative_kind == "jacobian"
        else "custom_vector_hvp_component"
    )
    if any(element.op != expected_op for element in output_arg):
        return None

    if derivative_kind == "jacobian":
        parsed = [parse_custom_vector_jacobian_component_args(element.name, element.args) for element in output_arg]
        spec, first_index, value, params = parsed[0]
        if first_index != 0:
            return None
        for index, candidate in enumerate(parsed):
            cand_spec, cand_index, cand_value, cand_params = candidate
            if cand_spec.name != spec.name or cand_index != index or cand_value != value or cand_params != params:
                return None
        x_ref = _emit_matrix_vector_argument(
            value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        w_ref = _emit_matrix_vector_argument(
            params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return spec.name, x_ref, None, w_ref

    parsed_hvp = [parse_custom_vector_hvp_component_args(element.name, element.args) for element in output_arg]
    spec, first_index, value, tangent, params = parsed_hvp[0]
    if first_index != 0:
        return None
    for index, candidate in enumerate(parsed_hvp):
        cand_spec, cand_index, cand_value, cand_tangent, cand_params = candidate
        if (
            cand_spec.name != spec.name
            or cand_index != index
            or cand_value != value
            or cand_tangent != tangent
            or cand_params != params
        ):
            return None
    x_ref = _emit_matrix_vector_argument(
        value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    tangent_ref = _emit_matrix_vector_argument(
        tangent.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    w_ref = _emit_matrix_vector_argument(
        params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    return spec.name, x_ref, tangent_ref, w_ref


def _match_custom_vector_hessian_output(
    output_arg: SXVector,
    *,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> tuple[str, str, str] | None:
    """Match a full flat custom vector Hessian output emitted row-major."""
    parsed: list[tuple[object, int, int, SXVector, tuple[SX, ...]]] = []
    for element in output_arg:
        matched = _match_passthrough_custom_vector_hessian_entry(element)
        if matched is None:
            return None
        parsed.append(parse_custom_vector_hessian_entry_args(matched.name, matched.args))

    spec = parsed[0][0]
    vector_dim = getattr(spec, "vector_dim")
    if vector_dim is None or len(output_arg) != vector_dim * vector_dim:
        return None

    _, _, _, x_value, params = parsed[0]
    for flat_index, candidate in enumerate(parsed):
        cand_spec, row, col, cand_x_value, cand_params = candidate
        if cand_spec != spec:
            return None
        expected_row, expected_col = divmod(flat_index, vector_dim)
        if row != expected_row or col != expected_col:
            return None
        if cand_x_value != x_value or cand_params != params:
            return None

    x_ref = _emit_matrix_vector_argument(
        x_value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    w_ref = _emit_matrix_vector_argument(
        params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    return getattr(spec, "name"), x_ref, w_ref


def _match_passthrough_custom_vector_hessian_entry(expr: SX) -> SX | None:
    """Return the underlying custom Hessian entry through trivial wrappers."""
    if expr.op == "custom_vector_hessian_entry":
        return expr
    if expr.op == "mul":
        left, right = expr.args
        if _is_passthrough_one(left):
            return _match_passthrough_custom_vector_hessian_entry(right)
        if _is_passthrough_one(right):
            return _match_passthrough_custom_vector_hessian_entry(left)
        return None
    if expr.op == "add":
        left, right = expr.args
        if _is_passthrough_zero(left):
            return _match_passthrough_custom_vector_hessian_entry(right)
        if _is_passthrough_zero(right):
            return _match_passthrough_custom_vector_hessian_entry(left)
        return None
    return None


def _match_passthrough_matvec_component(expr: SX) -> SX | None:
    """Return the underlying matvec component through trivial wrappers."""
    if expr.op == "matvec_component":
        return expr
    if expr.op == "mul":
        left, right = expr.args
        if _is_passthrough_one(left):
            return _match_passthrough_matvec_component(right)
        if _is_passthrough_one(right):
            return _match_passthrough_matvec_component(left)
        return None
    if expr.op == "add":
        left, right = expr.args
        if _is_passthrough_zero(left):
            return _match_passthrough_matvec_component(right)
        if _is_passthrough_zero(right):
            return _match_passthrough_matvec_component(left)
        return None
    return None


def _is_passthrough_zero(expr: SX) -> bool:
    """Return whether ``expr`` is structurally equivalent to zero."""
    if expr.op == "const" and expr.value == 0.0:
        return True
    if expr.op == "add":
        left, right = expr.args
        return _is_passthrough_zero(left) and _is_passthrough_zero(right)
    if expr.op == "mul":
        left, right = expr.args
        return _is_passthrough_zero(left) or _is_passthrough_zero(right)
    return False


def _is_passthrough_one(expr: SX) -> bool:
    """Return whether ``expr`` is structurally equivalent to one."""
    if expr.op == "const" and expr.value == 1.0:
        return True
    if expr.op == "add":
        left, right = expr.args
        return (_is_passthrough_zero(left) and _is_passthrough_one(right)) or (
            _is_passthrough_one(left) and _is_passthrough_zero(right)
        )
    if expr.op == "mul":
        left, right = expr.args
        return _is_passthrough_one(left) and _is_passthrough_one(right)
    return False


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


def _format_float(value: float | None, scalar_type: RustScalarType) -> str:
    """Format a Python float as a Rust floating-point literal."""
    if value is None:
        raise ValueError("expected a concrete floating-point value")
    validate_scalar_type(scalar_type)
    return f"{repr(float(value))}_{scalar_type}"


def _emit_node_expr(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    from ..rust_codegen import _emit_node_expr as _emit_node_expr_impl

    return _emit_node_expr_impl(
        expr,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )


def _math_function_name(op: str, scalar_type: RustScalarType) -> str:
    """Return the backend math function name for a scalar type."""
    validate_scalar_type(scalar_type)
    base_name = _LIBM_FUNCTIONS[op]
    if scalar_type == "f32":
        return f"{base_name}f"
    return base_name


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
