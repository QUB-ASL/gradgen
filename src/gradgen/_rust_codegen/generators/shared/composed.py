"""Composed-function helper utilities for Rust code generation."""

from __future__ import annotations

import re

from ...naming import sanitize_ident
from ...models import _ArgSpec, _ComposedRepeatPlan, _ComposedSinglePlan
from ....function import Function
from ....sx import SXNode
from .. import shared as _shared


def _build_composed_input_specs(
    input_name: str,
    input_size: int,
    parameter_name: str,
    parameter_size: int,
) -> tuple[_ArgSpec, ...]:
    """Build input metadata for a composed driver kernel."""
    specs = [
        _ArgSpec(
            raw_name=input_name,
            rust_name=sanitize_ident(input_name),
            rust_label=f'"{input_name}"',
            doc_description=_shared._describe_input_arg(input_name),
            size=input_size,
        )
    ]
    if parameter_size > 0:
        specs.append(
            _ArgSpec(
                raw_name=parameter_name,
                rust_name=sanitize_ident(parameter_name),
                rust_label=f'"{parameter_name}"',
                doc_description=(
                    "packed stage-parameter slice for the composed kernel; "
                    "symbolic parameter blocks are laid out in forward stage order "
                    "and the terminal block is stored last"
                ),
                size=parameter_size,
            )
        )
    return tuple(specs)


def _emit_composed_fixed_repeat_constants(
    const_name: str,
    values: tuple[tuple[float, ...], ...],
    scalar_type,
) -> list[str]:
    """Emit a compile-time fixed-parameter table for a repeat block."""
    row_size = len(values[0]) if values else 0
    rows = ", ".join(
        "[" + ", ".join(_shared._format_float(value, scalar_type) for value in row) + "]"
        for row in values
    )
    return [f"const {const_name}: [[{scalar_type}; {row_size}]; {len(values)}] = [{rows}];"]


def _emit_composed_parameter_ref(
    parameter_kind: str,
    parameter_size: int,
    parameter_offset: int,
    fixed_values: tuple[float, ...],
    *,
    parameters_name: str | None,
    scalar_type,
    const_name: str,
    index_var: str | None = None,
) -> str:
    """Return the Rust expression used to pass one composed parameter slice."""
    if parameter_kind == "fixed":
        if index_var is not None and const_name:
            if parameter_size == 0:
                return "&[]"
            return f"&{const_name}[{index_var}]"
        if parameter_size == 0:
            return "&[]"
        return "&[" + ", ".join(_shared._format_float(value, scalar_type) for value in fixed_values) + "]"

    if parameters_name is None:
        raise ValueError("symbolic composed parameters require a packed parameter input")
    if parameter_size == 0:
        return "&[]"
    if index_var is None:
        end = parameter_offset + parameter_size
        return f"&{parameters_name}[{parameter_offset}..{end}]"
    if parameter_size == 1:
        start_expr = _compose_offset_expr(parameter_offset, index_var)
        end_expr = _compose_offset_expr(parameter_offset + 1, index_var)
        return f"&{parameters_name}[{start_expr}..{end_expr}]"
    start_expr = _compose_offset_expr(parameter_offset, f"({index_var} * {parameter_size})")
    end_expr = _compose_offset_expr(parameter_offset, f"(({index_var} + 1) * {parameter_size})")
    return f"&{parameters_name}[{start_expr}..{end_expr}]"


def _compose_offset_expr(offset: int, expr: str) -> str:
    """Combine a constant offset with a Rust index expression, omitting identity additions."""
    if offset == 0:
        return expr
    return f"{offset} + {expr}"


def _composed_helper_base_label(name: str) -> str:
    """Normalize a composed source name into the shared helper base used across artifacts."""
    if name.endswith("_f"):
        return name[:-2]
    match = re.fullmatch(r"(.+)_grad_[A-Za-z0-9_]+", name)
    if match is not None:
        return match.group(1)
    match = re.fullmatch(r"(.+)_gradient_[A-Za-z0-9_]+", name)
    if match is not None:
        return match.group(1)
    return name


def _compose_composed_helper_base_name(crate_name: str | None, source_name: str) -> str:
    """Build the shared helper base name for a composed source within one crate."""
    base_label = sanitize_ident(_composed_helper_base_label(source_name))
    if crate_name is None:
        return base_label
    crate_label = sanitize_ident(crate_name)
    if base_label == crate_label or base_label.startswith(f"{crate_label}_"):
        return base_label
    return sanitize_ident(f"{crate_label}_{base_label}")


def _emit_composed_primal_single_block(
    plan: _ComposedSinglePlan,
    *,
    parameters_name: str | None,
    scalar_type,
) -> list[str]:
    """Emit one explicit primal stage call."""
    parameter_ref = _emit_composed_parameter_ref(
        plan.parameter_kind,
        plan.parameter_size,
        plan.parameter_offset,
        plan.fixed_values,
        parameters_name=parameters_name,
        scalar_type=scalar_type,
        const_name="",
    )
    return [
        f"{plan.helper_name}(current_state, {parameter_ref}, next_state, stage_work);",
        "current_state.copy_from_slice(next_state);",
    ]


def _emit_composed_primal_repeat_block(
    plan: _ComposedRepeatPlan,
    *,
    parameters_name: str | None,
    scalar_type,
) -> list[str]:
    """Emit a loop-based primal repeat block."""
    parameter_ref = _emit_composed_parameter_ref(
        plan.parameter_kind,
        plan.parameter_size,
        plan.parameter_offset,
        (),
        parameters_name=parameters_name,
        scalar_type=scalar_type,
        const_name=plan.const_name,
        index_var="repeat_index",
    )
    return [
        f"for repeat_index in 0..{plan.repeat_count} {{",
        f"    {plan.helper_name}(current_state, {parameter_ref}, next_state, stage_work);",
        "    current_state.copy_from_slice(next_state);",
        "}",
    ]


def _emit_composed_gradient_forward_single_block(
    plan: _ComposedSinglePlan,
    *,
    parameters_name: str | None,
    scalar_type,
    state_size: int,
) -> list[str]:
    """Emit one forward-pass stage evaluation for a composed gradient."""
    parameter_ref = _emit_composed_parameter_ref(
        plan.parameter_kind,
        plan.parameter_size,
        plan.parameter_offset,
        plan.fixed_values,
        parameters_name=parameters_name,
        scalar_type=scalar_type,
        const_name="",
    )
    start = plan.stage_index * state_size
    end = start + state_size
    return [
        "{",
        f"    let next_state = &mut state_history[{start}..{end}];",
        f"    {plan.helper_name}(current_state, {parameter_ref}, next_state, stage_work);",
        "    current_state.copy_from_slice(next_state);",
        "}",
    ]


def _emit_composed_gradient_forward_repeat_block(
    plan: _ComposedRepeatPlan,
    *,
    parameters_name: str | None,
    scalar_type,
    state_size: int,
) -> list[str]:
    """Emit one forward-pass repeat loop for a composed gradient."""
    parameter_ref = _emit_composed_parameter_ref(
        plan.parameter_kind,
        plan.parameter_size,
        plan.parameter_offset,
        (),
        parameters_name=parameters_name,
        scalar_type=scalar_type,
        const_name=plan.const_name,
        index_var="repeat_index",
    )
    return [
        f"for repeat_index in 0..{plan.repeat_count} {{",
        f"    let stage_index = {_compose_offset_expr(plan.stage_start_index, 'repeat_index')};",
        f"    let stage_start = stage_index * {state_size};",
        f"    let stage_end = stage_start + {state_size};",
        "    {",
        "        let next_state = &mut state_history[stage_start..stage_end];",
        f"        {plan.helper_name}(current_state, {parameter_ref}, next_state, stage_work);",
        "        current_state.copy_from_slice(next_state);",
        "    }",
        "}",
    ]


def _emit_composed_gradient_reverse_single_block(
    plan: _ComposedSinglePlan,
    *,
    parameters_name: str | None,
    scalar_type,
    state_size: int,
    input_name: str,
) -> list[str]:
    """Emit one reverse-pass VJP stage for a composed gradient."""
    parameter_ref = _emit_composed_parameter_ref(
        plan.parameter_kind,
        plan.parameter_size,
        plan.parameter_offset,
        plan.fixed_values,
        parameters_name=parameters_name,
        scalar_type=scalar_type,
        const_name="",
    )
    if plan.stage_index == 0:
        state_input_ref = input_name
    else:
        prev_start = (plan.stage_index - 1) * state_size
        prev_end = prev_start + state_size
        state_input_ref = f"&state_history[{prev_start}..{prev_end}]"
    return [
        "{",
        "    if current_lambda_is_a {",
        f"        {plan.vjp_helper_name}({state_input_ref}, {parameter_ref}, &lambda_a[..], lambda_b, stage_work);",
        "    } else {",
        f"        {plan.vjp_helper_name}({state_input_ref}, {parameter_ref}, &lambda_b[..], lambda_a, stage_work);",
        "    }",
        "    current_lambda_is_a = !current_lambda_is_a;",
        "}",
    ]


def _emit_composed_gradient_reverse_repeat_block(
    plan: _ComposedRepeatPlan,
    *,
    parameters_name: str | None,
    scalar_type,
    state_size: int,
    input_name: str,
) -> list[str]:
    """Emit one reverse-pass repeat loop for a composed gradient."""
    parameter_ref = _emit_composed_parameter_ref(
        plan.parameter_kind,
        plan.parameter_size,
        plan.parameter_offset,
        (),
        parameters_name=parameters_name,
        scalar_type=scalar_type,
        const_name=plan.const_name,
        index_var="repeat_index",
    )
    return [
        f"for repeat_index in (0..{plan.repeat_count}).rev() {{",
        f"    let stage_index = {_compose_offset_expr(plan.stage_start_index, 'repeat_index')};",
        "    if stage_index == 0 {",
        "        if current_lambda_is_a {",
        f"            {plan.vjp_helper_name}({input_name}, {parameter_ref}, &lambda_a[..], lambda_b, stage_work);",
        "        } else {",
        f"            {plan.vjp_helper_name}({input_name}, {parameter_ref}, &lambda_b[..], lambda_a, stage_work);",
        "        }",
        "    } else {",
        f"        let prev_start = (stage_index - 1) * {state_size};",
        f"        let prev_end = prev_start + {state_size};",
        "        if current_lambda_is_a {",
        f"            {plan.vjp_helper_name}(&state_history[prev_start..prev_end], {parameter_ref}, &lambda_a[..], lambda_b, stage_work);",
        "        } else {",
        f"            {plan.vjp_helper_name}(&state_history[prev_start..prev_end], {parameter_ref}, &lambda_b[..], lambda_a, stage_work);",
        "        }",
        "    }",
        "    current_lambda_is_a = !current_lambda_is_a;",
        "}",
    ]
