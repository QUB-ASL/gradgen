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
