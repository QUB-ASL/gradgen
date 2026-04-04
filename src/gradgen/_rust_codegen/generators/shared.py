"""Shared helpers for Rust code generation families."""

from __future__ import annotations

import re

from ... import rust_codegen as _rust_codegen
from ..config import RustBackendConfig, RustBackendMode, RustScalarType
from ..models import _ArgSpec, _ComposedRepeatPlan, _ComposedSinglePlan, _SingleShootingHelperBundle
from ..naming import sanitize_ident, validate_unique_rust_names
from ..render import _build_shared_helper_lines
from ..templates import _get_template
from ...ad import jvp
from ...function import Function, _add_like, _make_symbolic_input_like, _zero_like
from ...sx import SX, SXNode, SXVector, parse_bilinear_form_args, parse_matvec_component_args, parse_quadform_args
from ...custom_elementary import (
    get_registered_elementary_function,
    parse_custom_scalar_args,
    parse_custom_scalar_hvp_args,
    parse_custom_vector_args,
    parse_custom_vector_hessian_entry_args,
    parse_custom_vector_hvp_component_args,
    parse_custom_vector_jacobian_component_args,
    render_custom_rust_snippet,
)

generate_rust = _rust_codegen.generate_rust
_resolve_backend_config = _rust_codegen._resolve_backend_config
_validate_backend_mode = _rust_codegen._validate_backend_mode
_validate_scalar_type = _rust_codegen._validate_scalar_type
_validate_generated_argument_names = _rust_codegen._validate_generated_argument_names
_maybe_simplify_derivative_function = _rust_codegen._maybe_simplify_derivative_function
_derive_python_function_name = _rust_codegen._derive_python_function_name
_flatten_arg = _rust_codegen._flatten_arg
_arg_size = _rust_codegen._arg_size
_scaled_index_expr = _rust_codegen._scaled_index_expr
_format_float = _rust_codegen._format_float
_format_rust_string_literal = _rust_codegen._format_rust_string_literal
_describe_input_arg = _rust_codegen._describe_input_arg
_describe_output_arg = _rust_codegen._describe_output_arg
_emit_exact_length_assert = _rust_codegen._emit_exact_length_assert
_emit_min_length_assert = _rust_codegen._emit_min_length_assert
_allocate_workspace_slots = _rust_codegen._allocate_workspace_slots
_collect_required_workspace_nodes = _rust_codegen._collect_required_workspace_nodes
_collect_reachable_nodes = _rust_codegen._collect_reachable_nodes
_reemit_direct_output_helper_call = _rust_codegen._reemit_direct_output_helper_call
_collect_suppressed_custom_wrappers = _rust_codegen._collect_suppressed_custom_wrappers

def _build_directional_derivative_function(
    function: Function,
    *,
    active_indices: tuple[int, ...],
    tangent_names: tuple[str, ...],
    name: str,
) -> Function:
    """Build a function that returns the JVP of ``function`` for selected inputs."""
    if len(active_indices) != len(tangent_names):
        raise ValueError("active_indices and tangent_names must have the same length")

    tangent_inputs = tuple(
        _make_symbolic_input_like(function.inputs[index], tangent_name)
        for index, tangent_name in zip(active_indices, tangent_names)
    )
    tangent_mapping = dict(zip(active_indices, tangent_inputs))

    differentiated_outputs: list[SX | SXVector] = []
    for output in function.outputs:
        total = _zero_like(output)
        for index, tangent_input in tangent_mapping.items():
            total = _add_like(total, jvp(output, function.inputs[index], tangent_input))
        differentiated_outputs.append(total)

    return Function(
        name,
        (*function.inputs, *tangent_inputs),
        differentiated_outputs,
        input_names=(*function.input_names, *tangent_names),
        output_names=function.output_names,
    )


def _build_single_shooting_helpers(
    problem: SingleShootingProblem,
    *,
    helper_base_name: str,
    config: RustBackendConfig,
    simplification: int | str | None,
    include_cost: bool,
    include_gradient: bool,
    include_hvp: bool,
) -> _SingleShootingHelperBundle:
    """Generate stage helper kernels for a single-shooting problem."""
    helper_config = config.with_emit_metadata_helpers(False)
    helper_sources: list[str] = []
    helper_nodes: list[SXNode] = []
    max_workspace = 0

    dynamics_name = sanitize_ident(f"{helper_base_name}_dynamics")
    dynamics_jvp_name: str | None = None
    dynamics_function = _maybe_simplify_derivative_function(problem.dynamics, simplification)
    dynamics_codegen = generate_rust(
        dynamics_function,
        config=helper_config,
        function_name=dynamics_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )
    helper_sources.append(dynamics_codegen.source.rstrip())
    helper_nodes.extend(dynamics_function.nodes)
    max_workspace = max(max_workspace, dynamics_codegen.workspace_size)

    if include_hvp:
        dynamics_jvp_name = sanitize_ident(f"{helper_base_name}_dynamics_jvp")
        dynamics_jvp_function = _maybe_simplify_derivative_function(
            _build_directional_derivative_function(
                problem.dynamics,
                active_indices=(0, 1),
                tangent_names=("tangent_x", "tangent_u"),
                name=dynamics_jvp_name,
            ),
            simplification,
        )
        dynamics_jvp_codegen = generate_rust(
            dynamics_jvp_function,
            config=helper_config,
            function_name=dynamics_jvp_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.append(dynamics_jvp_codegen.source.rstrip())
        helper_nodes.extend(dynamics_jvp_function.nodes)
        max_workspace = max(max_workspace, dynamics_jvp_codegen.workspace_size)

    stage_cost_name = sanitize_ident(f"{helper_base_name}_stage_cost")
    terminal_cost_name = sanitize_ident(f"{helper_base_name}_terminal_cost")
    if include_cost:
        stage_cost_function = _maybe_simplify_derivative_function(problem.stage_cost, simplification)
        stage_cost_codegen = generate_rust(
            stage_cost_function,
            config=helper_config,
            function_name=stage_cost_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.append(stage_cost_codegen.source.rstrip())
        helper_nodes.extend(stage_cost_function.nodes)
        max_workspace = max(max_workspace, stage_cost_codegen.workspace_size)

        terminal_cost_function = _maybe_simplify_derivative_function(problem.terminal_cost, simplification)
        terminal_cost_codegen = generate_rust(
            terminal_cost_function,
            config=helper_config,
            function_name=terminal_cost_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.append(terminal_cost_codegen.source.rstrip())
        helper_nodes.extend(terminal_cost_function.nodes)
        max_workspace = max(max_workspace, terminal_cost_codegen.workspace_size)

    dynamics_vjp_x_name: str | None = None
    dynamics_vjp_u_name: str | None = None
    dynamics_vjp_x_jvp_name: str | None = None
    dynamics_vjp_u_jvp_name: str | None = None
    stage_cost_grad_x_name: str | None = None
    stage_cost_grad_u_name: str | None = None
    stage_cost_grad_x_jvp_name: str | None = None
    stage_cost_grad_u_jvp_name: str | None = None
    terminal_cost_grad_x_name: str | None = None
    terminal_cost_grad_x_jvp_name: str | None = None
    if include_gradient or include_hvp:
        dynamics_vjp_x_name = sanitize_ident(f"{helper_base_name}_dynamics_vjp_x")
        dynamics_vjp_u_name = sanitize_ident(f"{helper_base_name}_dynamics_vjp_u")
        stage_cost_grad_x_name = sanitize_ident(f"{helper_base_name}_stage_cost_grad_x")
        stage_cost_grad_u_name = sanitize_ident(f"{helper_base_name}_stage_cost_grad_u")
        terminal_cost_grad_x_name = sanitize_ident(f"{helper_base_name}_terminal_cost_grad_x")

        helper_functions = (
            _maybe_simplify_derivative_function(
                problem.dynamics.vjp(wrt_index=0, name=dynamics_vjp_x_name),
                simplification,
            ),
            _maybe_simplify_derivative_function(
                problem.dynamics.vjp(wrt_index=1, name=dynamics_vjp_u_name),
                simplification,
            ),
            _maybe_simplify_derivative_function(
                problem.stage_cost.gradient(0, name=stage_cost_grad_x_name),
                simplification,
            ),
            _maybe_simplify_derivative_function(
                problem.stage_cost.gradient(1, name=stage_cost_grad_u_name),
                simplification,
            ),
            _maybe_simplify_derivative_function(
                problem.terminal_cost.gradient(0, name=terminal_cost_grad_x_name),
                simplification,
            ),
        )
        helper_names = (
            dynamics_vjp_x_name,
            dynamics_vjp_u_name,
            stage_cost_grad_x_name,
            stage_cost_grad_u_name,
            terminal_cost_grad_x_name,
        )
        for helper_function, helper_name in zip(helper_functions, helper_names):
            helper_codegen = generate_rust(
                helper_function,
                config=helper_config,
                function_name=helper_name,
                function_index=1,
                shared_helper_nodes=(),
                emit_crate_header=False,
                emit_docs=False,
                function_keyword="fn",
            )
            helper_sources.append(helper_codegen.source.rstrip())
            helper_nodes.extend(helper_function.nodes)
            max_workspace = max(max_workspace, helper_codegen.workspace_size)

    if include_hvp:
        dynamics_vjp_x_jvp_name = sanitize_ident(f"{helper_base_name}_dynamics_vjp_x_jvp")
        dynamics_vjp_u_jvp_name = sanitize_ident(f"{helper_base_name}_dynamics_vjp_u_jvp")
        stage_cost_grad_x_jvp_name = sanitize_ident(f"{helper_base_name}_stage_cost_grad_x_jvp")
        stage_cost_grad_u_jvp_name = sanitize_ident(f"{helper_base_name}_stage_cost_grad_u_jvp")
        terminal_cost_grad_x_jvp_name = sanitize_ident(f"{helper_base_name}_terminal_cost_grad_x_jvp")

        dynamics_vjp_x_function = _maybe_simplify_derivative_function(
            problem.dynamics.vjp(wrt_index=0, name=dynamics_vjp_x_name),
            simplification,
        )
        dynamics_vjp_u_function = _maybe_simplify_derivative_function(
            problem.dynamics.vjp(wrt_index=1, name=dynamics_vjp_u_name),
            simplification,
        )
        stage_cost_grad_x_function = _maybe_simplify_derivative_function(
            problem.stage_cost.gradient(0, name=stage_cost_grad_x_name),
            simplification,
        )
        stage_cost_grad_u_function = _maybe_simplify_derivative_function(
            problem.stage_cost.gradient(1, name=stage_cost_grad_u_name),
            simplification,
        )
        terminal_cost_grad_x_function = _maybe_simplify_derivative_function(
            problem.terminal_cost.gradient(0, name=terminal_cost_grad_x_name),
            simplification,
        )

        hvp_helper_functions = (
            _maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    dynamics_vjp_x_function,
                    active_indices=(0, 1, 3),
                    tangent_names=("tangent_x", "tangent_u", "tangent_cotangent_x_next"),
                    name=dynamics_vjp_x_jvp_name,
                ),
                simplification,
            ),
            _maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    dynamics_vjp_u_function,
                    active_indices=(0, 1, 3),
                    tangent_names=("tangent_x", "tangent_u", "tangent_cotangent_x_next"),
                    name=dynamics_vjp_u_jvp_name,
                ),
                simplification,
            ),
            _maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    stage_cost_grad_x_function,
                    active_indices=(0, 1),
                    tangent_names=("tangent_x", "tangent_u"),
                    name=stage_cost_grad_x_jvp_name,
                ),
                simplification,
            ),
            _maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    stage_cost_grad_u_function,
                    active_indices=(0, 1),
                    tangent_names=("tangent_x", "tangent_u"),
                    name=stage_cost_grad_u_jvp_name,
                ),
                simplification,
            ),
            _maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    terminal_cost_grad_x_function,
                    active_indices=(0,),
                    tangent_names=("tangent_x",),
                    name=terminal_cost_grad_x_jvp_name,
                ),
                simplification,
            ),
        )
        hvp_helper_names = (
            dynamics_vjp_x_jvp_name,
            dynamics_vjp_u_jvp_name,
            stage_cost_grad_x_jvp_name,
            stage_cost_grad_u_jvp_name,
            terminal_cost_grad_x_jvp_name,
        )
        for helper_function, helper_name in zip(hvp_helper_functions, hvp_helper_names):
            helper_codegen = generate_rust(
                helper_function,
                config=helper_config,
                function_name=helper_name,
                function_index=1,
                shared_helper_nodes=(),
                emit_crate_header=False,
                emit_docs=False,
                function_keyword="fn",
            )
            helper_sources.append(helper_codegen.source.rstrip())
            helper_nodes.extend(helper_function.nodes)
            max_workspace = max(max_workspace, helper_codegen.workspace_size)

    return _SingleShootingHelperBundle(
        dynamics_name=dynamics_name,
        dynamics_jvp_name=dynamics_jvp_name,
        stage_cost_name=stage_cost_name,
        terminal_cost_name=terminal_cost_name,
        dynamics_vjp_x_name=dynamics_vjp_x_name,
        dynamics_vjp_u_name=dynamics_vjp_u_name,
        dynamics_vjp_x_jvp_name=dynamics_vjp_x_jvp_name,
        dynamics_vjp_u_jvp_name=dynamics_vjp_u_jvp_name,
        stage_cost_grad_x_name=stage_cost_grad_x_name,
        stage_cost_grad_u_name=stage_cost_grad_u_name,
        stage_cost_grad_x_jvp_name=stage_cost_grad_x_jvp_name,
        stage_cost_grad_u_jvp_name=stage_cost_grad_u_jvp_name,
        terminal_cost_grad_x_name=terminal_cost_grad_x_name,
        terminal_cost_grad_x_jvp_name=terminal_cost_grad_x_jvp_name,
        sources=tuple(helper_sources),
        helper_nodes=tuple(helper_nodes),
        max_workspace_size=max_workspace,
    )


def _build_single_shooting_input_specs(
    problem: SingleShootingProblem,
    *,
    include_hvp: bool,
) -> tuple[_ArgSpec, ...]:
    """Build input metadata for a single-shooting driver kernel."""
    specs = [
        _ArgSpec(
            raw_name=problem.initial_state_name,
            rust_name=sanitize_ident(problem.initial_state_name),
            rust_label=_format_rust_string_literal(problem.initial_state_name),
            doc_description="initial state slice",
            size=problem.state_size,
        ),
        _ArgSpec(
            raw_name=problem.control_sequence_name,
            rust_name=sanitize_ident(problem.control_sequence_name),
            rust_label=_format_rust_string_literal(problem.control_sequence_name),
            doc_description="packed control-sequence slice laid out stage-major over the horizon",
            size=problem.horizon * problem.control_size,
        ),
        _ArgSpec(
            raw_name=problem.parameter_name,
            rust_name=sanitize_ident(problem.parameter_name),
            rust_label=_format_rust_string_literal(problem.parameter_name),
            doc_description="shared parameter slice used at every stage and terminal evaluation",
            size=problem.parameter_size,
        ),
    ]
    if include_hvp:
        direction_name = f"v_{problem.control_sequence_name}"
        specs.append(
            _ArgSpec(
                raw_name=direction_name,
                rust_name=sanitize_ident(direction_name),
                rust_label=_format_rust_string_literal(direction_name),
                doc_description="packed control-sequence direction slice laid out stage-major over the horizon",
                size=problem.horizon * problem.control_size,
            )
        )
    return tuple(specs)


def _build_single_shooting_output_specs(
    problem: SingleShootingProblem,
    *,
    include_cost: bool,
    include_gradient: bool,
    include_hvp: bool,
    include_states: bool,
) -> tuple[_ArgSpec, ...]:
    """Build output metadata for a single-shooting driver kernel."""
    specs: list[_ArgSpec] = []
    if include_cost:
        specs.append(
            _ArgSpec(
                raw_name="cost",
                rust_name="cost",
                rust_label=_format_rust_string_literal("cost"),
                doc_description="total cost output",
                size=1,
            )
        )
    if include_gradient:
        gradient_name = f"gradient_{problem.control_sequence_name}"
        specs.append(
            _ArgSpec(
                raw_name=gradient_name,
                rust_name=sanitize_ident(gradient_name),
                rust_label=_format_rust_string_literal(gradient_name),
                doc_description="gradient with respect to the packed control sequence",
                size=problem.horizon * problem.control_size,
            )
        )
    if include_hvp:
        hvp_name = f"hvp_{problem.control_sequence_name}"
        specs.append(
            _ArgSpec(
                raw_name=hvp_name,
                rust_name=sanitize_ident(hvp_name),
                rust_label=_format_rust_string_literal(hvp_name),
                doc_description="Hessian-vector product with respect to the packed control sequence",
                size=problem.horizon * problem.control_size,
            )
        )
    if include_states:
        specs.append(
            _ArgSpec(
                raw_name="x_traj",
                rust_name="x_traj",
                rust_label=_format_rust_string_literal("x_traj"),
                doc_description="packed rollout state trajectory including x0 through xN",
                size=(problem.horizon + 1) * problem.state_size,
            )
        )
    return tuple(specs)


def _compose_single_shooting_helper_base_name(crate_name: str | None, problem_name: str) -> str:
    """Build the shared helper base name for a single-shooting problem."""
    base_label = sanitize_ident(problem_name)
    if crate_name is None:
        return base_label
    crate_label = sanitize_ident(crate_name)
    if base_label == crate_label or base_label.startswith(f"{crate_label}_"):
        return base_label
    return sanitize_ident(f"{crate_label}_{base_label}")


def _emit_single_shooting_control_slice(
    sequence_name: str,
    index_expr: str,
    control_size: int,
) -> str:
    """Return the Rust slice expression for one stage control block."""
    return f"&{sequence_name}[{_emit_single_shooting_stage_range(index_expr, control_size)}]"


def _emit_single_shooting_stage_range(
    index_expr: str,
    block_size: int,
) -> str:
    """Return the Rust range expression for one packed stage block."""
    if block_size == 1:
        start = "0" if index_expr == "0" else index_expr
        end = "1" if index_expr == "0" else f"({index_expr} + 1)"
        return f"{start}..{end}"
    start = "0" if index_expr == "0" else f"({index_expr} * {block_size})"
    end = str(block_size) if index_expr == "0" else f"(({index_expr} + 1) * {block_size})"
    return f"{start}..{end}"


def _emit_small_accumulate(
    target_name: str,
    source_name: str,
    size: int,
    *,
    indent: str = "",
) -> list[str]:
    """Emit direct accumulation for tiny fixed-size vectors, else a loop."""
    if size <= 0:
        return []
    if size <= 2:
        return [f"{indent}{target_name}[{index}] += {source_name}[{index}];" for index in range(size)]
    return [
        f"{indent}for index in 0..{size} {{",
        f"{indent}    {target_name}[index] += {source_name}[index];",
        f"{indent}}}",
    ]


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
            rust_label=_format_rust_string_literal(input_name),
            doc_description=_describe_input_arg(input_name),
            size=input_size,
        )
    ]
    if parameter_size > 0:
        specs.append(
            _ArgSpec(
                raw_name=parameter_name,
                rust_name=sanitize_ident(parameter_name),
                rust_label=_format_rust_string_literal(parameter_name),
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
    scalar_type: RustScalarType,
) -> list[str]:
    """Emit a compile-time fixed-parameter table for a repeat block."""
    row_size = len(values[0]) if values else 0
    rows = ", ".join(
        "[" + ", ".join(_format_float(value, scalar_type) for value in row) + "]"
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
    scalar_type: RustScalarType,
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
        return "&[" + ", ".join(_format_float(value, scalar_type) for value in fixed_values) + "]"

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
    scalar_type: RustScalarType,
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
    scalar_type: RustScalarType,
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
    scalar_type: RustScalarType,
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
    scalar_type: RustScalarType,
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
    scalar_type: RustScalarType,
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
    scalar_type: RustScalarType,
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
