"""Family-specific Rust generation helpers."""

from __future__ import annotations

import re

from .. import rust_codegen as _rust_codegen
from .config import RustBackendConfig, RustBackendMode, RustScalarType
from .models import (
    RustCodegenResult,
    RustMultiFunctionProjectResult,
    RustProjectResult,
    RustPythonInterfaceProjectResult,
    _ArgSpec,
    _ComposedRepeatPlan,
    _ComposedSinglePlan,
    _SingleShootingHelperBundle,
)
from .naming import sanitize_ident, validate_unique_rust_names
from .render import _build_shared_helper_lines
from .templates import _get_template
from ..ad import jvp
from ..function import Function, _add_like, _make_symbolic_input_like, _zero_like
from ..map_zip import ReducedFunction, ZippedFunction, ZippedJacobianFunction
from ..single_shooting import (
    SingleShootingGradientFunction,
    SingleShootingHvpFunction,
    SingleShootingJointFunction,
    SingleShootingPrimalFunction,
    SingleShootingProblem,
    _single_shooting_bundle_output_names,
)
from ..sx import SX, SXNode, SXVector, parse_bilinear_form_args, parse_matvec_component_args, parse_quadform_args
from ..custom_elementary import (
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

def _generate_composed_primal_rust(
    composed,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate compact Rust for a staged composed primal kernel."""
    from ..composed_function import _RepeatStage, _SingleStage

    terminal = composed._require_terminal()
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    _validate_backend_mode(resolved_config.backend_mode)
    _validate_scalar_type(resolved_config.scalar_type)
    if resolved_config.backend_mode == "std":
        if math_library is not None:
            raise ValueError("math_library is only supported for no_std backend mode")
        resolved_math_library = None
    else:
        resolved_math_library = math_library or "libm"
    helper_simplification = composed.simplification

    name = sanitize_ident(resolved_config.function_name or composed.name)
    helper_base_name = _compose_composed_helper_base_name(
        resolved_config.crate_name,
        name,
    )
    helper_config = resolved_config.with_emit_metadata_helpers(False)
    helper_sources: list[str] = []
    helper_nodes: list[SXNode] = []
    constant_lines: list[str] = []
    plans: list[_ComposedSinglePlan | _ComposedRepeatPlan] = []

    stage_index = 0
    parameter_offset = 0
    max_helper_workspace = 0
    for block_index, step in enumerate(composed.steps):
        if isinstance(step, _SingleStage):
            helper_name = sanitize_ident(f"{helper_base_name}_stage_{block_index}_{step.function.name}")
            helper_function = _maybe_simplify_derivative_function(step.function, helper_simplification)
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
            max_helper_workspace = max(max_helper_workspace, helper_codegen.workspace_size)
            plans.append(
                _ComposedSinglePlan(
                    helper_name=helper_name,
                    vjp_helper_name="",
                    parameter_kind=step.parameter.kind,
                    parameter_size=step.parameter.size,
                    parameter_offset=parameter_offset,
                    fixed_values=step.parameter.values,
                    stage_index=stage_index,
                )
            )
            parameter_offset += step.parameter.symbolic_size
            stage_index += 1
            continue

        helper_name = sanitize_ident(f"{helper_base_name}_repeat_{block_index}_{step.function.name}")
        helper_function = _maybe_simplify_derivative_function(step.function, helper_simplification)
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
        max_helper_workspace = max(max_helper_workspace, helper_codegen.workspace_size)

        const_name = sanitize_ident(f"{helper_base_name}_repeat_{block_index}_params").upper()
        parameter_kind = step.parameters[0].kind
        if parameter_kind == "fixed" and step.parameters[0].size > 0:
            constant_lines.extend(
                _emit_composed_fixed_repeat_constants(
                    const_name,
                    tuple(parameter.values for parameter in step.parameters),
                    resolved_config.scalar_type,
                )
            )

        plans.append(
            _ComposedRepeatPlan(
                helper_name=helper_name,
                vjp_helper_name="",
                parameter_kind=parameter_kind,
                parameter_size=step.parameters[0].size,
                parameter_offset=parameter_offset,
                fixed_values=tuple(parameter.values for parameter in step.parameters),
                repeat_count=len(step.parameters),
                stage_start_index=stage_index,
                const_name=const_name,
            )
        )
        parameter_offset += sum(parameter.symbolic_size for parameter in step.parameters)
        stage_index += len(step.parameters)

    terminal_helper_name = sanitize_ident(f"{helper_base_name}_terminal_{terminal.function.name}")
    terminal_function = _maybe_simplify_derivative_function(terminal.function, helper_simplification)
    terminal_codegen = generate_rust(
        terminal_function,
        config=helper_config,
        function_name=terminal_helper_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )
    helper_sources.append(terminal_codegen.source.rstrip())
    helper_nodes.extend(terminal_function.nodes)
    max_helper_workspace = max(max_helper_workspace, terminal_codegen.workspace_size)
    terminal_parameter_offset = parameter_offset

    state_size = _arg_size(composed.state_input)
    output_name = terminal.function.output_names[0]
    input_specs = _build_composed_input_specs(
        composed.input_name,
        state_size,
        composed.parameter_name,
        composed.parameter_size,
    )
    output_specs = (
        _ArgSpec(
            raw_name=output_name,
            rust_name=sanitize_ident(output_name),
            rust_label=_format_rust_string_literal(output_name),
            doc_description=_describe_output_arg(output_name),
            size=1,
        ),
    )
    _validate_generated_argument_names(input_specs, output_specs)
    input_assert_lines = []
    input_return_lines = []
    output_assert_lines = []
    output_return_lines = []
    _assert, _in_return, _out_return = _emit_exact_length_assert(
        input_specs[0].rust_name,
        input_specs[0].raw_name,
        state_size,
    )
    input_assert_lines.append(_assert)
    input_return_lines.append(_in_return)
    if composed.parameter_size > 0:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            input_specs[1].rust_name,
            input_specs[1].raw_name,
            composed.parameter_size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    _assert, _in_return, _out_return = _emit_exact_length_assert(
        output_specs[0].rust_name,
        output_specs[0].raw_name,
        1,
    )
    output_assert_lines.append(_assert)
    output_return_lines.append(_out_return)

    state_work_size = 2 * state_size
    workspace_size = state_work_size + max_helper_workspace
    computation_lines = [
        f"let (state_buffers, stage_work) = work.split_at_mut({state_work_size});",
        f"let (current_state, next_state) = state_buffers.split_at_mut({state_size});",
        f"current_state.copy_from_slice({input_specs[0].rust_name});",
    ]
    for plan in plans:
        if isinstance(plan, _ComposedSinglePlan):
            computation_lines.extend(
                _emit_composed_primal_single_block(
                    plan,
                    parameters_name=input_specs[1].rust_name if composed.parameter_size > 0 else None,
                    scalar_type=resolved_config.scalar_type,
                )
            )
            continue
        computation_lines.extend(
            _emit_composed_primal_repeat_block(
                plan,
                parameters_name=input_specs[1].rust_name if composed.parameter_size > 0 else None,
                scalar_type=resolved_config.scalar_type,
            )
        )

    terminal_parameter_ref = _emit_composed_parameter_ref(
        terminal.parameter.kind,
        terminal.parameter.size,
        terminal_parameter_offset,
        terminal.parameter.values,
        parameters_name=input_specs[1].rust_name if composed.parameter_size > 0 else None,
        scalar_type=resolved_config.scalar_type,
        const_name="",
    )
    computation_lines.append(
        f"{terminal_helper_name}(current_state, {terminal_parameter_ref}, {output_specs[0].rust_name}, stage_work);"
    )

    shared_helper_lines = list(_build_shared_helper_lines(
        tuple(helper_nodes),
        resolved_config.backend_mode,
        resolved_config.scalar_type,
        resolved_math_library,
    ))
    if constant_lines:
        if shared_helper_lines:
            shared_helper_lines.append("")
        shared_helper_lines.extend(constant_lines)

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            f"{output_specs[0].rust_name}: &mut [{resolved_config.scalar_type}]",
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )
    if workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        upper_name=name.upper(),
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=workspace_size,
        workspace_assert_line=_ws_assert,
        workspace_return_line=_ws_return,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        function_parameter_count=len(input_specs) + len(output_specs) + 1,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        input_return_lines=input_return_lines,
        output_assert_lines=output_assert_lines,
        output_return_lines=output_return_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=shared_helper_lines,
    ).rstrip()
    source_sections = [driver_source, *helper_sources]
    source = "\n\n".join(section for section in source_sections if section)

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=(output_name,),
        output_sizes=(1,),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


def _generate_composed_gradient_rust(
    gradient,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate compact Rust for a staged composed gradient kernel."""
    from ..composed_function import _RepeatStage, _SingleStage

    composed = gradient.composed
    terminal = composed._require_terminal()
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    _validate_backend_mode(resolved_config.backend_mode)
    _validate_scalar_type(resolved_config.scalar_type)
    resolved_math_library = "libm" if resolved_config.backend_mode == "no_std" else None
    helper_simplification = gradient.simplification

    name = sanitize_ident(resolved_config.function_name or gradient.name)
    helper_base_name = _compose_composed_helper_base_name(
        resolved_config.crate_name,
        name,
    )
    helper_config = resolved_config.with_emit_metadata_helpers(False)
    helper_sources: list[str] = []
    helper_nodes: list[SXNode] = []
    constant_lines: list[str] = []
    plans: list[_ComposedSinglePlan | _ComposedRepeatPlan] = []

    stage_index = 0
    parameter_offset = 0
    max_helper_workspace = 0
    for block_index, step in enumerate(composed.steps):
        if isinstance(step, _SingleStage):
            helper_name = sanitize_ident(f"{helper_base_name}_stage_{block_index}_{step.function.name}")
            vjp_helper_name = sanitize_ident(f"{helper_base_name}_stage_{block_index}_{step.function.name}_vjp")
            helper_function = _maybe_simplify_derivative_function(step.function, helper_simplification)
            vjp_function = _maybe_simplify_derivative_function(
                step.function.vjp(wrt_index=0, name=vjp_helper_name),
                helper_simplification,
            )
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
            vjp_codegen = generate_rust(
                vjp_function,
                config=helper_config,
                function_name=vjp_helper_name,
                function_index=1,
                shared_helper_nodes=(),
                emit_crate_header=False,
                emit_docs=False,
                function_keyword="fn",
            )
            helper_sources.extend((helper_codegen.source.rstrip(), vjp_codegen.source.rstrip()))
            helper_nodes.extend((*helper_function.nodes, *vjp_function.nodes))
            max_helper_workspace = max(
                max_helper_workspace,
                helper_codegen.workspace_size,
                vjp_codegen.workspace_size,
            )
            plans.append(
                _ComposedSinglePlan(
                    helper_name=helper_name,
                    vjp_helper_name=vjp_helper_name,
                    parameter_kind=step.parameter.kind,
                    parameter_size=step.parameter.size,
                    parameter_offset=parameter_offset,
                    fixed_values=step.parameter.values,
                    stage_index=stage_index,
                )
            )
            parameter_offset += step.parameter.symbolic_size
            stage_index += 1
            continue

        helper_name = sanitize_ident(f"{helper_base_name}_repeat_{block_index}_{step.function.name}")
        vjp_helper_name = sanitize_ident(f"{helper_base_name}_repeat_{block_index}_{step.function.name}_vjp")
        helper_function = _maybe_simplify_derivative_function(step.function, helper_simplification)
        vjp_function = _maybe_simplify_derivative_function(
            step.function.vjp(wrt_index=0, name=vjp_helper_name),
            helper_simplification,
        )
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
        vjp_codegen = generate_rust(
            vjp_function,
            config=helper_config,
            function_name=vjp_helper_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.extend((helper_codegen.source.rstrip(), vjp_codegen.source.rstrip()))
        helper_nodes.extend((*helper_function.nodes, *vjp_function.nodes))
        max_helper_workspace = max(
            max_helper_workspace,
            helper_codegen.workspace_size,
            vjp_codegen.workspace_size,
        )

        const_name = sanitize_ident(f"{helper_base_name}_repeat_{block_index}_params").upper()
        parameter_kind = step.parameters[0].kind
        if parameter_kind == "fixed" and step.parameters[0].size > 0:
            constant_lines.extend(
                _emit_composed_fixed_repeat_constants(
                    const_name,
                    tuple(parameter.values for parameter in step.parameters),
                    resolved_config.scalar_type,
                )
            )

        plans.append(
            _ComposedRepeatPlan(
                helper_name=helper_name,
                vjp_helper_name=vjp_helper_name,
                parameter_kind=parameter_kind,
                parameter_size=step.parameters[0].size,
                parameter_offset=parameter_offset,
                fixed_values=tuple(parameter.values for parameter in step.parameters),
                repeat_count=len(step.parameters),
                stage_start_index=stage_index,
                const_name=const_name,
            )
        )
        parameter_offset += sum(parameter.symbolic_size for parameter in step.parameters)
        stage_index += len(step.parameters)

    terminal_gradient_name = sanitize_ident(f"{helper_base_name}_terminal_{terminal.function.name}_grad")
    terminal_gradient_function = _maybe_simplify_derivative_function(
        terminal.function.gradient(0, name=terminal_gradient_name),
        helper_simplification,
    )
    terminal_gradient_codegen = generate_rust(
        terminal_gradient_function,
        config=helper_config,
        function_name=terminal_gradient_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )
    helper_sources.append(terminal_gradient_codegen.source.rstrip())
    helper_nodes.extend(terminal_gradient_function.nodes)
    max_helper_workspace = max(max_helper_workspace, terminal_gradient_codegen.workspace_size)
    terminal_parameter_offset = parameter_offset

    state_size = _arg_size(composed.state_input)
    output_name = terminal.function.output_names[0]
    input_specs = _build_composed_input_specs(
        composed.input_name,
        state_size,
        composed.parameter_name,
        composed.parameter_size,
    )
    output_specs = (
        _ArgSpec(
            raw_name=output_name,
            rust_name=sanitize_ident(output_name),
            rust_label=_format_rust_string_literal(output_name),
            doc_description=_describe_output_arg(output_name),
            size=state_size,
        ),
    )
    _validate_generated_argument_names(input_specs, output_specs)
    input_assert_lines = []
    input_return_lines = []
    output_assert_lines = []
    output_return_lines = []
    _assert, _in_return, _out_return = _emit_exact_length_assert(
        input_specs[0].rust_name,
        input_specs[0].raw_name,
        state_size,
    )
    input_assert_lines.append(_assert)
    input_return_lines.append(_in_return)
    if composed.parameter_size > 0:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            input_specs[1].rust_name,
            input_specs[1].raw_name,
            composed.parameter_size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    _assert, _in_return, _out_return = _emit_exact_length_assert(
        output_specs[0].rust_name,
        output_specs[0].raw_name,
        state_size,
    )
    output_assert_lines.append(_assert)
    output_return_lines.append(_out_return)

    states_size = composed.stage_count * state_size
    workspace_size = states_size + state_size + (2 * state_size) + max_helper_workspace
    computation_lines = [
        f"let (state_history, rest) = work.split_at_mut({states_size});",
        f"let (current_state, rest) = rest.split_at_mut({state_size});",
        f"let (lambda_buffers, stage_work) = rest.split_at_mut({2 * state_size});",
        f"let (lambda_a, lambda_b) = lambda_buffers.split_at_mut({state_size});",
        f"current_state.copy_from_slice({input_specs[0].rust_name});",
    ]
    for plan in plans:
        if isinstance(plan, _ComposedSinglePlan):
            computation_lines.extend(
                _emit_composed_gradient_forward_single_block(
                    plan,
                    parameters_name=input_specs[1].rust_name if composed.parameter_size > 0 else None,
                    scalar_type=resolved_config.scalar_type,
                    state_size=state_size,
                )
            )
            continue
        computation_lines.extend(
            _emit_composed_gradient_forward_repeat_block(
                plan,
                parameters_name=input_specs[1].rust_name if composed.parameter_size > 0 else None,
                scalar_type=resolved_config.scalar_type,
                state_size=state_size,
            )
        )

    terminal_parameter_ref = _emit_composed_parameter_ref(
        terminal.parameter.kind,
        terminal.parameter.size,
        terminal_parameter_offset,
        terminal.parameter.values,
        parameters_name=input_specs[1].rust_name if composed.parameter_size > 0 else None,
        scalar_type=resolved_config.scalar_type,
        const_name="",
    )
    computation_lines.append(
        f"{terminal_gradient_name}(current_state, {terminal_parameter_ref}, lambda_a, stage_work);"
    )
    computation_lines.append("let mut current_lambda_is_a = true;")

    for plan in reversed(plans):
        if isinstance(plan, _ComposedSinglePlan):
            computation_lines.extend(
                _emit_composed_gradient_reverse_single_block(
                    plan,
                    parameters_name=input_specs[1].rust_name if composed.parameter_size > 0 else None,
                    scalar_type=resolved_config.scalar_type,
                    state_size=state_size,
                    input_name=input_specs[0].rust_name,
                )
            )
            continue
        computation_lines.extend(
            _emit_composed_gradient_reverse_repeat_block(
                plan,
                parameters_name=input_specs[1].rust_name if composed.parameter_size > 0 else None,
                scalar_type=resolved_config.scalar_type,
                state_size=state_size,
                input_name=input_specs[0].rust_name,
            )
        )

    computation_lines.extend(
        [
            "let gradient = if current_lambda_is_a { &lambda_a[..] } else { &lambda_b[..] };",
            f"{output_specs[0].rust_name}.copy_from_slice(gradient);",
        ]
    )

    shared_helper_lines = list(_build_shared_helper_lines(
        tuple(helper_nodes),
        resolved_config.backend_mode,
        resolved_config.scalar_type,
        resolved_math_library,
    ))
    if constant_lines:
        if shared_helper_lines:
            shared_helper_lines.append("")
        shared_helper_lines.extend(constant_lines)

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            f"{output_specs[0].rust_name}: &mut [{resolved_config.scalar_type}]",
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )
    if workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        upper_name=name.upper(),
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=workspace_size,
        workspace_assert_line=_ws_assert,
        workspace_return_line=_ws_return,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        function_parameter_count=len(input_specs) + len(output_specs) + 1,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        input_return_lines=input_return_lines,
        output_assert_lines=output_assert_lines,
        output_return_lines=output_return_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=shared_helper_lines,
    ).rstrip()
    source_sections = [driver_source, *helper_sources]
    source = "\n\n".join(section for section in source_sections if section)

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=(output_name,),
        output_sizes=(state_size,),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


def _generate_zipped_primal_rust(
    zipped: ZippedFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate loop-based Rust for a staged map/zip primal kernel."""
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    resolved_math_library = "libm" if resolved_config.backend_mode == "no_std" else None
    name = sanitize_ident(resolved_config.function_name or zipped.name)
    helper_name = sanitize_ident(f"{name}_helper")

    helper_function = zipped.function
    if zipped.simplification is not None:
        helper_function = helper_function.simplify(
            max_effort=zipped.simplification,
            name=helper_function.name,
        )

    helper_config = resolved_config.with_emit_metadata_helpers(False)
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

    packed_input_sizes = tuple(zipped.count * _arg_size(arg) for arg in zipped.function.inputs)
    packed_output_sizes = tuple(zipped.count * _arg_size(arg) for arg in zipped.function.outputs)
    input_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_input_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(zipped.input_names, packed_input_sizes)
    )
    output_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_output_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(zipped.output_names, packed_output_sizes)
    )
    _validate_generated_argument_names(input_specs, output_specs)

    input_assert_lines: list[str] = []
    input_return_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_return_lines: list[str] = []

    for spec in input_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    for spec in output_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        output_assert_lines.append(_assert)
        output_return_lines.append(_out_return)

    helper_workspace_size = helper_codegen.workspace_size
    computation_lines: list[str] = []
    if helper_workspace_size > 0:
        computation_lines.append(f"let helper_work = &mut work[..{helper_workspace_size}];")
    else:
        computation_lines.append("let helper_work = &mut work[..0];")
    computation_lines.append(f"for stage_index in 0..{zipped.count} {{")
    for input_spec, formal in zip(input_specs, zipped.function.inputs):
        block_size = _arg_size(formal)
        start_expr = _scaled_index_expr("stage_index", block_size)
        end_expr = _scaled_index_expr("stage_index + 1", block_size)
        computation_lines.append(
            f"    let {input_spec.rust_name}_stage = "
            f"&{input_spec.rust_name}[{start_expr}..{end_expr}];"
        )
    for output_spec, formal in zip(output_specs, zipped.function.outputs):
        block_size = _arg_size(formal)
        start_expr = _scaled_index_expr("stage_index", block_size)
        end_expr = _scaled_index_expr("stage_index + 1", block_size)
        computation_lines.append(
            f"    let {output_spec.rust_name}_stage = "
            f"&mut {output_spec.rust_name}[{start_expr}..{end_expr}];"
        )
    helper_args = ", ".join(
        [
            *[f"{spec.rust_name}_stage" for spec in input_specs],
            *[f"{spec.rust_name}_stage" for spec in output_specs],
            "helper_work",
        ]
    )
    computation_lines.append(f"    {helper_name}({helper_args});")
    computation_lines.append("}")

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs],
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )
    if helper_workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", helper_workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=helper_workspace_size,
        workspace_assert_line=_ws_assert,
        workspace_return_line=_ws_return,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        function_parameter_count=len(input_specs) + len(output_specs) + 1,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        input_return_lines=input_return_lines,
        output_assert_lines=output_assert_lines,
        output_return_lines=output_return_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=[],
    ).rstrip()
    source = "\n\n".join([driver_source, helper_codegen.source.rstrip()])

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=helper_workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=tuple(spec.raw_name for spec in output_specs),
        output_sizes=tuple(spec.size for spec in output_specs),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


def _generate_zipped_jacobian_rust(
    zipped_jacobian: ZippedJacobianFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate loop-based Rust for a staged map/zip Jacobian kernel."""
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    resolved_math_library = "libm" if resolved_config.backend_mode == "no_std" else None
    name = sanitize_ident(resolved_config.function_name or zipped_jacobian.name)
    helper_name = sanitize_ident(f"{name}_helper")

    zipped = zipped_jacobian.zipped
    local_jacobian = zipped.function.jacobian(
        zipped_jacobian.wrt_index,
        name=helper_name,
    )
    if zipped_jacobian.simplification is not None:
        local_jacobian = local_jacobian.simplify(
            max_effort=zipped_jacobian.simplification,
            name=local_jacobian.name,
        )

    helper_config = resolved_config.with_emit_metadata_helpers(False)
    helper_codegen = generate_rust(
        local_jacobian,
        config=helper_config,
        function_name=helper_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )

    packed_input_sizes = tuple(zipped.count * _arg_size(arg) for arg in zipped.function.inputs)
    input_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_input_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(zipped.input_names, packed_input_sizes)
    )
    wrt_size = _arg_size(zipped.function.inputs[zipped_jacobian.wrt_index])
    packed_wrt_size = zipped.count * wrt_size
    output_specs: list[_ArgSpec] = []
    local_output_sizes: list[int] = []
    row_sizes: list[int] = []
    for raw_name, output in zip(zipped.function.output_names, zipped.function.outputs):
        row_size = _arg_size(output)
        row_sizes.append(row_size)
        local_block_size = row_size * wrt_size
        local_output_sizes.append(local_block_size)
        output_specs.append(
            _ArgSpec(
                raw_name=f"jacobian_{raw_name}",
                rust_name=sanitize_ident(f"jacobian_{raw_name}"),
                rust_label=_format_rust_string_literal(f"jacobian_{raw_name}"),
                doc_description=_describe_output_arg(f"jacobian_{raw_name}"),
                size=(zipped.count * row_size) * packed_wrt_size,
            )
        )
    output_specs_tuple = tuple(output_specs)
    _validate_generated_argument_names(input_specs, output_specs_tuple)

    temp_workspace_size = sum(local_output_sizes)
    total_workspace_size = temp_workspace_size + helper_codegen.workspace_size

    input_assert_lines: list[str] = []
    input_return_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_return_lines: list[str] = []
    for spec in input_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    for spec in output_specs_tuple:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        output_assert_lines.append(_assert)
        output_return_lines.append(_out_return)

    zero_literal = _format_float(0.0, resolved_config.scalar_type)
    computation_lines: list[str] = [f"{spec.rust_name}.fill({zero_literal});" for spec in output_specs_tuple]
    remaining_work_name = "work"
    for index, (spec, local_size) in enumerate(zip(output_specs_tuple, local_output_sizes)):
        next_remaining = "helper_work" if index == len(output_specs_tuple) - 1 else f"rest_work_{index}"
        computation_lines.append(
            f"let (temp_{spec.rust_name}, {next_remaining}) = {remaining_work_name}.split_at_mut({local_size});"
        )
        remaining_work_name = next_remaining
    computation_lines.append(f"for stage_index in 0..{zipped.count} {{")
    for input_spec, formal in zip(input_specs, zipped.function.inputs):
        block_size = _arg_size(formal)
        start_expr = _scaled_index_expr("stage_index", block_size)
        end_expr = _scaled_index_expr("stage_index + 1", block_size)
        computation_lines.append(
            f"    let {input_spec.rust_name}_stage = "
            f"&{input_spec.rust_name}[{start_expr}..{end_expr}];"
        )
    helper_args = ", ".join(
        [
            *[f"{spec.rust_name}_stage" for spec in input_specs],
            *[f"temp_{spec.rust_name}" for spec in output_specs_tuple],
            "helper_work",
        ]
    )
    computation_lines.append(f"    {helper_name}({helper_args});")
    for spec, row_size, local_size in zip(output_specs_tuple, row_sizes, local_output_sizes):
        computation_lines.append(f"    for local_row in 0..{row_size} {{")
        dest_row_base = _scaled_index_expr("stage_index", row_size)
        dest_stage_offset = _scaled_index_expr("stage_index", wrt_size)
        src_row_base = _scaled_index_expr("local_row", wrt_size)
        if wrt_size > 1:
            src_row_base = src_row_base.replace("(", "", 1).rsplit(")", 1)[0]
        computation_lines.append(
            f"        let dest_row = {dest_row_base} + local_row;"
        )
        computation_lines.append(
            f"        let dest_start = (dest_row * {packed_wrt_size}) + {dest_stage_offset};"
        )
        computation_lines.append(f"        let src_start = {src_row_base};")
        computation_lines.append(
            f"        {spec.rust_name}[dest_start..(dest_start + {wrt_size})]"
            f".copy_from_slice(&temp_{spec.rust_name}[src_start..(src_start + {wrt_size})]);"
        )
        computation_lines.append("    }")
    computation_lines.append("}")

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs_tuple],
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )
    if total_workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", total_workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=total_workspace_size,
        workspace_assert_line=_ws_assert,
        workspace_return_line=_ws_return,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs_tuple,
        function_parameter_count=len(input_specs) + len(output_specs_tuple) + 1,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        input_return_lines=input_return_lines,
        output_assert_lines=output_assert_lines,
        output_return_lines=output_return_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=[],
    ).rstrip()
    source = "\n\n".join([driver_source, helper_codegen.source.rstrip()])

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=total_workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=tuple(spec.raw_name for spec in output_specs_tuple),
        output_sizes=tuple(spec.size for spec in output_specs_tuple),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


def _generate_reduced_primal_rust(
    reduced: ReducedFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate loop-based Rust for a staged reduce primal kernel."""
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    resolved_math_library = "libm" if resolved_config.backend_mode == "no_std" else None
    name = sanitize_ident(resolved_config.function_name or reduced.name)
    helper_name = sanitize_ident(f"{name}_helper")

    helper_function = reduced.function
    if reduced.simplification is not None:
        helper_function = helper_function.simplify(
            max_effort=reduced.simplification,
            name=helper_function.name,
        )

    helper_config = resolved_config.with_emit_metadata_helpers(False)
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

    accumulator_formal = reduced.function.inputs[0]
    sequence_formal = reduced.function.inputs[1]

    accumulator_size = _arg_size(accumulator_formal)
    sequence_size = reduced.count * _arg_size(sequence_formal)
    output_size = accumulator_size

    input_specs = (
        _ArgSpec(
            raw_name=reduced.accumulator_input_name,
            rust_name=sanitize_ident(reduced.accumulator_input_name),
            rust_label=_format_rust_string_literal(reduced.accumulator_input_name),
            doc_description=_describe_input_arg(reduced.accumulator_input_name),
            size=accumulator_size,
        ),
        _ArgSpec(
            raw_name=reduced.input_name,
            rust_name=sanitize_ident(reduced.input_name),
            rust_label=_format_rust_string_literal(reduced.input_name),
            doc_description=_describe_input_arg(reduced.input_name),
            size=sequence_size,
        ),
    )
    output_specs = (
        _ArgSpec(
            raw_name=reduced.output_name,
            rust_name=sanitize_ident(reduced.output_name),
            rust_label=_format_rust_string_literal(reduced.output_name),
            doc_description=_describe_output_arg(reduced.output_name),
            size=output_size,
        ),
    )
    _validate_generated_argument_names(input_specs, output_specs)

    input_assert_lines: list[str] = []
    input_return_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_return_lines: list[str] = []

    for spec in input_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    for spec in output_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        output_assert_lines.append(_assert)
        output_return_lines.append(_out_return)

    helper_workspace_size = helper_codegen.workspace_size
    temp_acc_size = 2 * accumulator_size
    total_workspace_size = helper_workspace_size + temp_acc_size

    zero_literal = _format_float(0.0, resolved_config.scalar_type)
    acc0_name = input_specs[0].rust_name
    seq_name = input_specs[1].rust_name
    out_name = output_specs[0].rust_name

    computation_lines: list[str] = []
    if helper_workspace_size > 0:
        computation_lines.append(
            f"let (acc_work, helper_work) = work.split_at_mut({temp_acc_size});"
        )
    else:
        computation_lines.append("let acc_work = work;")
        computation_lines.append("let helper_work = &mut work[..0];")
    computation_lines.append(f"let (acc_curr_buf, acc_next_buf) = acc_work.split_at_mut({accumulator_size});")
    computation_lines.append(f"acc_curr_buf.copy_from_slice({acc0_name});")
    computation_lines.append(f"for stage_index in 0..{reduced.count} {{")
    block_size = _arg_size(sequence_formal)
    start_expr = _scaled_index_expr("stage_index", block_size)
    end_expr = _scaled_index_expr("stage_index + 1", block_size)
    computation_lines.append(f"    let x_stage = &{seq_name}[{start_expr}..{end_expr}];")
    computation_lines.append("    acc_next_buf.fill(" + zero_literal + ");")
    computation_lines.append(f"    {helper_name}(acc_curr_buf, x_stage, acc_next_buf, helper_work);")
    computation_lines.append("    acc_curr_buf.copy_from_slice(acc_next_buf);")
    computation_lines.append("}")
    computation_lines.append(f"{out_name}.copy_from_slice(acc_curr_buf);")

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs],
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )

    if total_workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", total_workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=total_workspace_size,
        workspace_assert_line=_ws_assert,
        workspace_return_line=_ws_return,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        function_parameter_count=len(input_specs) + len(output_specs) + 1,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        input_return_lines=input_return_lines,
        output_assert_lines=output_assert_lines,
        output_return_lines=output_return_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=[],
    ).rstrip()
    source = "\n\n".join([driver_source, helper_codegen.source.rstrip()])

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=total_workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=tuple(spec.raw_name for spec in output_specs),
        output_sizes=tuple(spec.size for spec in output_specs),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


def _generate_single_shooting_primal_rust(
    problem: SingleShootingProblem,
    *,
    include_states: bool,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate compact Rust for a single-shooting primal kernel."""
    return _generate_single_shooting_driver_rust(
        problem,
        include_cost=True,
        include_gradient=False,
        include_hvp=False,
        include_states=include_states,
        config=config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
        function_index=function_index,
    )


def _generate_single_shooting_gradient_rust(
    gradient: SingleShootingGradientFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate compact Rust for a single-shooting gradient kernel."""
    return _generate_single_shooting_driver_rust(
        gradient.problem,
        include_cost=False,
        include_gradient=True,
        include_hvp=False,
        include_states=gradient.include_states,
        config=config,
        function_name=function_name or gradient.name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
        function_index=function_index,
    )


def _generate_single_shooting_hvp_rust(
    hvp: SingleShootingHvpFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate compact Rust for a single-shooting HVP kernel."""
    return _generate_single_shooting_driver_rust(
        hvp.problem,
        include_cost=False,
        include_gradient=False,
        include_hvp=True,
        include_states=hvp.include_states,
        config=config,
        function_name=function_name or hvp.name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
        function_index=function_index,
    )


def _generate_single_shooting_joint_rust(
    joint: SingleShootingJointFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate compact Rust for a joint single-shooting kernel."""
    return _generate_single_shooting_driver_rust(
        joint.problem,
        include_cost=joint.bundle.include_cost,
        include_gradient=joint.bundle.include_gradient,
        include_hvp=joint.bundle.include_hvp,
        include_states=joint.bundle.include_states,
        config=config,
        function_name=function_name or joint.name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
        function_index=function_index,
    )


def _generate_single_shooting_driver_rust(
    problem: SingleShootingProblem,
    *,
    include_cost: bool,
    include_gradient: bool,
    include_hvp: bool,
    include_states: bool,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate a loop-based single-shooting Rust kernel."""
    if not (include_cost or include_gradient or include_hvp):
        raise ValueError("single-shooting kernels must compute cost, gradient, or hvp")

    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    _validate_backend_mode(resolved_config.backend_mode)
    _validate_scalar_type(resolved_config.scalar_type)
    if resolved_config.backend_mode == "std":
        if math_library is not None:
            raise ValueError("math_library is only supported for no_std backend mode")
        resolved_math_library = None
    else:
        resolved_math_library = math_library or "libm"

    name = sanitize_ident(resolved_config.function_name or problem.name)
    helper_simplification = problem.simplification
    helper_base_name = _compose_single_shooting_helper_base_name(
        resolved_config.crate_name,
        problem.name,
    )
    helpers = _build_single_shooting_helpers(
        problem,
        helper_base_name=helper_base_name,
        config=resolved_config,
        simplification=helper_simplification,
        include_cost=include_cost,
        include_gradient=include_gradient,
        include_hvp=include_hvp,
    )

    input_specs = _build_single_shooting_input_specs(problem, include_hvp=include_hvp)
    output_specs = _build_single_shooting_output_specs(
        problem,
        include_cost=include_cost,
        include_gradient=include_gradient,
        include_hvp=include_hvp,
        include_states=include_states,
    )
    _validate_generated_argument_names(input_specs, output_specs)

    state_size = problem.state_size
    control_size = problem.control_size
    horizon = problem.horizon
    need_history = include_gradient or include_hvp
    need_adjoint = include_gradient or include_hvp

    input_assert_lines = []
    input_return_lines = []
    for input_spec in input_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            input_spec.rust_name, input_spec.raw_name, input_spec.size
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    output_assert_lines = []
    output_return_lines = []
    for output_spec in output_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            output_spec.rust_name, output_spec.raw_name, output_spec.size
        )
        output_assert_lines.append(_assert)
        output_return_lines.append(_out_return)

    output_names = {spec.raw_name: spec.rust_name for spec in output_specs}
    x0_name = input_specs[0].rust_name
    U_name = input_specs[1].rust_name
    p_name = input_specs[2].rust_name
    v_U_name = input_specs[3].rust_name if include_hvp else None
    cost_output_name = output_names.get("cost")
    gradient_output_name = output_names.get(f"gradient_{problem.control_sequence_name}")
    hvp_output_name = output_names.get(f"hvp_{problem.control_sequence_name}")
    states_output_name = output_names.get("x_traj")
    use_rollout_states_as_history = (
        need_history and include_states and states_output_name is not None
    )

    state_history_size = (
        0 if use_rollout_states_as_history else ((horizon * state_size) if need_history else 0)
    )
    tangent_history_size = (horizon * state_size) if include_hvp else 0
    state_buffer_size = 2 * state_size
    tangent_buffer_size = (2 * state_size) if include_hvp else 0
    lambda_buffer_size = (2 * state_size) if need_adjoint else 0
    mu_buffer_size = (2 * state_size) if include_hvp else 0
    temp_state_size = state_size if need_adjoint else 0
    temp_control_size = control_size if need_adjoint else 0
    scalar_buffer_size = 1 if include_cost else 0
    driver_workspace_size = (
        state_history_size
        + tangent_history_size
        + state_buffer_size
        + tangent_buffer_size
        + lambda_buffer_size
        + mu_buffer_size
        + temp_state_size
        + temp_control_size
        + scalar_buffer_size
    )
    workspace_size = driver_workspace_size + helpers.max_workspace_size

    computation_lines: list[str] = []
    if need_history:
        if use_rollout_states_as_history:
            computation_lines.append("let rest = work;")
        else:
            computation_lines.append(
                f"let (state_history, rest) = work.split_at_mut({state_history_size});"
            )
    else:
        computation_lines.append("let rest = work;")
    if include_hvp:
        computation_lines.append(
            f"let (tangent_history, rest) = rest.split_at_mut({tangent_history_size});"
        )
    computation_lines.append(
        f"let (state_buffers, rest) = rest.split_at_mut({state_buffer_size});"
    )
    computation_lines.append(
        f"let (current_state_buf, next_state_buf) = state_buffers.split_at_mut({state_size});"
    )
    computation_lines.append("let mut current_state = current_state_buf;")
    computation_lines.append("let mut next_state = next_state_buf;")
    if include_hvp:
        computation_lines.append(
            f"let (tangent_buffers, rest) = rest.split_at_mut({tangent_buffer_size});"
        )
        computation_lines.append(
            f"let (current_tangent, next_tangent) = tangent_buffers.split_at_mut({state_size});"
        )
    if need_adjoint:
        computation_lines.append(
            f"let (lambda_buffers, rest) = rest.split_at_mut({lambda_buffer_size});"
        )
        computation_lines.append(
            f"let (lambda_current, lambda_next) = lambda_buffers.split_at_mut({state_size});"
        )
    if include_hvp:
        computation_lines.append(
            f"let (mu_buffers, rest) = rest.split_at_mut({mu_buffer_size});"
        )
        computation_lines.append(
            f"let (mu_current, mu_next) = mu_buffers.split_at_mut({state_size});"
        )
    if need_adjoint:
        computation_lines.append(
            f"let (temp_state, rest) = rest.split_at_mut({temp_state_size});"
        )
        computation_lines.append(
            f"let (temp_control, rest) = rest.split_at_mut({temp_control_size});"
        )
    if include_cost:
        computation_lines.append("let (scalar_buffer, stage_work) = rest.split_at_mut(1);")
    else:
        computation_lines.append("let stage_work = rest;")
    computation_lines.append(f"current_state.copy_from_slice({x0_name});")
    if include_hvp:
        computation_lines.append(f"current_tangent.fill({_format_float(0.0, resolved_config.scalar_type)});")
    if include_states and states_output_name is not None:
        computation_lines.append(
            f"{states_output_name}[0..{state_size}].copy_from_slice({x0_name});"
        )
    if use_rollout_states_as_history and states_output_name is not None:
        computation_lines.append(
            f"let state_history = &mut {states_output_name}[{state_size}..];"
        )
    if include_cost:
        computation_lines.append(f"let mut total_cost = { _format_float(0.0, resolved_config.scalar_type) };")

    for_lines: list[str] = [
        f"for stage_index in 0..{horizon} {{",
        f"    let u_t = {_emit_single_shooting_control_slice(U_name, 'stage_index', control_size)};",
    ]
    if include_hvp and v_U_name is not None:
        for_lines.append(
            f"    let v_u_t = {_emit_single_shooting_control_slice(v_U_name, 'stage_index', control_size)};"
        )
    if include_cost:
        for_lines.extend(
            [
                f"    {helpers.stage_cost_name}(current_state, u_t, {p_name}, scalar_buffer, stage_work);",
                "    total_cost += scalar_buffer[0];",
            ]
        )
    for_lines.extend(
        [
            f"    {helpers.dynamics_name}(current_state, u_t, {p_name}, next_state, stage_work);",
        ]
    )
    if include_hvp:
        for_lines.append(
            f"    {helpers.dynamics_jvp_name}(current_state, u_t, {p_name}, current_tangent, v_u_t, next_tangent, stage_work);"
        )
    if need_history:
        for_lines.append(
            f"    state_history[(stage_index * {state_size})..((stage_index + 1) * {state_size})].copy_from_slice(next_state);"
        )
    if include_hvp:
        for_lines.append(
            f"    tangent_history[(stage_index * {state_size})..((stage_index + 1) * {state_size})].copy_from_slice(next_tangent);"
        )
    if include_states and states_output_name is not None and not use_rollout_states_as_history:
        for_lines.append(
            f"    {states_output_name}[((stage_index + 1) * {state_size})..((stage_index + 2) * {state_size})].copy_from_slice(next_state);"
        )
    for_lines.append("    core::mem::swap(&mut current_state, &mut next_state);")
    if include_hvp:
        for_lines.append("    current_tangent.copy_from_slice(next_tangent);")
    for_lines.append("}")
    computation_lines.extend(for_lines)

    if include_cost:
        computation_lines.append(
            f"{helpers.terminal_cost_name}(current_state, {p_name}, scalar_buffer, stage_work);"
        )
        computation_lines.append("total_cost += scalar_buffer[0];")
        if cost_output_name is not None:
            computation_lines.append(f"{cost_output_name}[0] = total_cost;")

    if need_adjoint:
        computation_lines.append(
            f"{helpers.terminal_cost_grad_x_name}(current_state, {p_name}, lambda_current, stage_work);"
        )
    if include_hvp:
        computation_lines.append(
            f"{helpers.terminal_cost_grad_x_jvp_name}(current_state, {p_name}, current_tangent, mu_current, stage_work);"
        )
    if need_adjoint:
        computation_lines.extend(
            [
                f"for stage_index in (1..{horizon}).rev() {{",
                f"    let x_t = &state_history[((stage_index - 1) * {state_size})..(stage_index * {state_size})];",
                f"    let u_t = {_emit_single_shooting_control_slice(U_name, 'stage_index', control_size)};",
            ]
        )
        if include_hvp and v_U_name is not None:
            computation_lines.extend(
                [
                    f"    let tangent_x_t = &tangent_history[((stage_index - 1) * {state_size})..(stage_index * {state_size})];",
                    f"    let v_u_t = {_emit_single_shooting_control_slice(v_U_name, 'stage_index', control_size)};",
                ]
            )
        if include_gradient and gradient_output_name is not None:
            computation_lines.extend(
                [
                    f"    let grad_u_t = &mut {gradient_output_name}[{_emit_single_shooting_stage_range('stage_index', control_size)}];",
                    f"    {helpers.stage_cost_grad_u_name}(x_t, u_t, {p_name}, grad_u_t, stage_work);",
                    f"    {helpers.dynamics_vjp_u_name}(x_t, u_t, {p_name}, &lambda_current[..], temp_control, stage_work);",
                ]
            )
            computation_lines.extend(
                _emit_small_accumulate("grad_u_t", "temp_control", control_size, indent="    ")
            )
        if include_hvp and hvp_output_name is not None:
            computation_lines.extend(
                [
                    f"    let hvp_u_t = &mut {hvp_output_name}[{_emit_single_shooting_stage_range('stage_index', control_size)}];",
                    f"    {helpers.stage_cost_grad_u_jvp_name}(x_t, u_t, {p_name}, tangent_x_t, v_u_t, hvp_u_t, stage_work);",
                    f"    {helpers.dynamics_vjp_u_jvp_name}(x_t, u_t, {p_name}, &lambda_current[..], tangent_x_t, v_u_t, &mu_current[..], temp_control, stage_work);",
                ]
            )
            computation_lines.extend(
                _emit_small_accumulate("hvp_u_t", "temp_control", control_size, indent="    ")
            )
        computation_lines.extend(
            [
                f"    {helpers.stage_cost_grad_x_name}(x_t, u_t, {p_name}, lambda_next, stage_work);",
                f"    {helpers.dynamics_vjp_x_name}(x_t, u_t, {p_name}, &lambda_current[..], temp_state, stage_work);",
            ]
        )
        computation_lines.extend(
            _emit_small_accumulate("lambda_next", "temp_state", state_size, indent="    ")
        )
        if include_hvp:
            computation_lines.extend(
                [
                    f"    {helpers.stage_cost_grad_x_jvp_name}(x_t, u_t, {p_name}, tangent_x_t, v_u_t, mu_next, stage_work);",
                    f"    {helpers.dynamics_vjp_x_jvp_name}(x_t, u_t, {p_name}, &lambda_current[..], tangent_x_t, v_u_t, &mu_current[..], temp_state, stage_work);",
                ]
            )
            computation_lines.extend(
                _emit_small_accumulate("mu_next", "temp_state", state_size, indent="    ")
            )
        computation_lines.append("    lambda_current.copy_from_slice(lambda_next);")
        if include_hvp:
            computation_lines.append("    mu_current.copy_from_slice(mu_next);")
        computation_lines.append("}")
        computation_lines.append(
            f"let u_t = {_emit_single_shooting_control_slice(U_name, '0', control_size)};"
        )
        if include_gradient and gradient_output_name is not None:
            computation_lines.extend(
                [
                    f"let grad_u_t = &mut {gradient_output_name}[{_emit_single_shooting_stage_range('0', control_size)}];",
                    f"{helpers.stage_cost_grad_u_name}({x0_name}, u_t, {p_name}, grad_u_t, stage_work);",
                    f"{helpers.dynamics_vjp_u_name}({x0_name}, u_t, {p_name}, &lambda_current[..], temp_control, stage_work);",
                ]
            )
            computation_lines.extend(
                _emit_small_accumulate("grad_u_t", "temp_control", control_size)
            )
        if include_hvp and hvp_output_name is not None and v_U_name is not None:
            computation_lines.extend(
                [
                    f"next_tangent.fill({_format_float(0.0, resolved_config.scalar_type)});",
                    f"let v_u_t = {_emit_single_shooting_control_slice(v_U_name, '0', control_size)};",
                    f"let hvp_u_t = &mut {hvp_output_name}[{_emit_single_shooting_stage_range('0', control_size)}];",
                    f"{helpers.stage_cost_grad_u_jvp_name}({x0_name}, u_t, {p_name}, next_tangent, v_u_t, hvp_u_t, stage_work);",
                    f"{helpers.dynamics_vjp_u_jvp_name}({x0_name}, u_t, {p_name}, &lambda_current[..], next_tangent, v_u_t, &mu_current[..], temp_control, stage_work);",
                ]
            )
            computation_lines.extend(
                _emit_small_accumulate("hvp_u_t", "temp_control", control_size)
            )

    shared_helper_lines = list(
        _build_shared_helper_lines(
            helpers.helper_nodes,
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        )
    )
    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs],
            "work: &mut [{scalar_type}]".format(scalar_type=resolved_config.scalar_type),
        ]
    )
    if workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        upper_name=name.upper(),
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=workspace_size,
        workspace_assert_line=_ws_assert,
        workspace_return_line=_ws_return,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        function_parameter_count=len(input_specs) + len(output_specs) + 1,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        input_return_lines=input_return_lines,
        output_assert_lines=output_assert_lines,
        output_return_lines=output_return_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=shared_helper_lines,
    ).rstrip()
    source_sections = [driver_source, *helpers.sources]
    source = "\n\n".join(section for section in source_sections if section)

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=tuple(spec.raw_name for spec in output_specs),
        output_sizes=tuple(spec.size for spec in output_specs),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


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
