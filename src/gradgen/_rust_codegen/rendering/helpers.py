"""Shared helper emission for generated Rust modules."""

from __future__ import annotations

from ...sx import SXNode
from ..config import RustBackendMode, RustScalarType
from .custom import _build_custom_helper_lines
from .expression import _emit_math_call, _emit_norm_abs_expr


def _build_shared_helper_lines(
    nodes: tuple[SXNode, ...],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
    *,
    required_matrix_helpers: set[str] | None = None,
    suppressed_custom_wrappers: set[tuple[str, str]] | None = None,
) -> tuple[str, ...]:
    """Return module-scope helper definitions needed by generated kernels."""
    used_ops = {node.op for node in nodes}
    legacy_matrix_helper_mode = required_matrix_helpers is None
    required_matrix_helpers = required_matrix_helpers or set()
    lines: list[str] = []

    if legacy_matrix_helper_mode and {
        "matvec_component",
        "transpose_matvec_component",
    } & used_ops:
        lines.extend(
            [
                "/// Return a single component of a dense matrix-vector product.",
                "///",
                "/// This helper evaluates one row of a row-major matrix against",
                "/// the input vector and returns the corresponding output entry.",
                f"fn matvec(matrix: &[{scalar_type}], rows: usize, cols: usize, x: &[{scalar_type}], y: &mut [{scalar_type}]) {{",
                "    for row in 0..rows {",
                f"        let mut total = 0.0_{scalar_type};",
                "        let start = row * cols;",
                "        for col in 0..cols {",
                "            total += matrix[start + col] * x[col];",
                "        }",
                "        y[row] = total;",
                "    }",
                "}",
                "/// Return a single component of a dense transpose matrix-vector product.",
                "///",
                "/// This helper evaluates one column of a row-major matrix against",
                "/// the input vector and returns the corresponding output entry.",
                f"fn transpose_matvec(matrix: &[{scalar_type}], rows: usize, cols: usize, x: &[{scalar_type}], y: &mut [{scalar_type}]) {{",
                "    for col in 0..cols {",
                f"        let mut total = 0.0_{scalar_type};",
                "        for row in 0..rows {",
                "            total += matrix[row * cols + col] * x[row];",
                "        }",
                "        y[col] = total;",
                "    }",
                "}",
            ]
        )

    if "matvec" in required_matrix_helpers:
        lines.extend(
            [
                "/// Return a single component of a dense matrix-vector product.",
                "///",
                "/// This helper evaluates one row of a row-major matrix against",
                "/// the input vector and returns the corresponding output entry.",
                f"fn matvec(matrix: &[{scalar_type}], rows: usize, cols: usize, x: &[{scalar_type}], y: &mut [{scalar_type}]) {{",
                "    for row in 0..rows {",
                f"        let mut total = 0.0_{scalar_type};",
                "        let start = row * cols;",
                "        for col in 0..cols {",
                "            total += matrix[start + col] * x[col];",
                "        }",
                "        y[row] = total;",
                "    }",
                "}",
            ]
        )

    if "matvec_component" in used_ops:
        lines.extend(
            [
                "#[inline(always)]",
                "/// Return one component of a dense matrix-vector product.",
                "///",
                "/// This helper evaluates the requested row of a row-major matrix",
                "/// against the input vector and returns the scalar result.",
                f"fn matvec_component(matrix: &[{scalar_type}], _rows: usize, cols: usize, row: usize, x: &[{scalar_type}]) -> {scalar_type} {{",
                "    let start = row * cols;",
                f"    let row_slice = &matrix[start..start + cols];",
                f"    let mut total = 0.0_{scalar_type};",
                "    for (entry, value) in row_slice.iter().zip(x.iter()) {",
                "        total += *entry * *value;",
                "    }",
                "    total",
                "}",
            ]
        )

    if legacy_matrix_helper_mode and {
        "matvec_component",
        "transpose_matvec_component",
    } & used_ops:
        lines.extend(
            [
                "/// Return a single component of a dense transpose matrix-vector product.",
                "///",
                "/// This helper evaluates one column of a row-major matrix against",
                "/// the input vector and returns the corresponding output entry.",
                f"fn transpose_matvec(matrix: &[{scalar_type}], rows: usize, cols: usize, x: &[{scalar_type}], y: &mut [{scalar_type}]) {{",
                "    for col in 0..cols {",
                f"        let mut total = 0.0_{scalar_type};",
                "        for row in 0..rows {",
                "            total += matrix[row * cols + col] * x[row];",
                "        }",
                "        y[col] = total;",
                "    }",
                "}",
            ]
        )

    if "transpose_matvec" in required_matrix_helpers:
        lines.extend(
            [
                "/// Return a single component of a dense transpose matrix-vector product.",
                "///",
                "/// This helper evaluates one column of a row-major matrix against",
                "/// the input vector and returns the corresponding output entry.",
                f"fn transpose_matvec(matrix: &[{scalar_type}], rows: usize, cols: usize, x: &[{scalar_type}], y: &mut [{scalar_type}]) {{",
                "    for col in 0..cols {",
                f"        let mut total = 0.0_{scalar_type};",
                "        for row in 0..rows {",
                "            total += matrix[row * cols + col] * x[row];",
                "        }",
                "        y[col] = total;",
                "    }",
                "}",
            ]
        )

    if "transpose_matvec_component" in used_ops:
        lines.extend(
            [
                "#[inline(always)]",
                "/// Return one component of a dense transpose matrix-vector product.",
                "///",
                "/// This helper evaluates the requested column of a row-major matrix",
                "/// against the input vector and returns the scalar result.",
                f"fn transpose_matvec_component(matrix: &[{scalar_type}], rows: usize, cols: usize, col: usize, x: &[{scalar_type}]) -> {scalar_type} {{",
                f"    let mut out = 0.0_{scalar_type};",
                "    for row in 0..rows {",
                "        out += matrix[row * cols + col] * x[row];",
                "    }",
                "    out",
                "}",
            ]
        )

    if "bilinear_form" in used_ops or "quadform" in used_ops:
        lines.extend(
            [
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
            ]
        )

    if "quadform" in used_ops:
        lines.extend(
            [
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
