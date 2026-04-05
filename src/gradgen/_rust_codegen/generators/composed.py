"""Family-specific Rust generation helpers."""

from __future__ import annotations

from . import shared as _shared
from .rendering import KernelRenderContext, render_kernel_source
from ..config import RustBackendConfig, RustBackendMode, RustScalarType
from ..models import (
    RustCodegenResult,
    _ArgSpec,
    _ComposedRepeatPlan,
    _ComposedSinglePlan
)
from ..naming import sanitize_ident

generate_rust = _shared.generate_rust
_resolve_backend_config = _shared._resolve_backend_config
_validate_backend_mode = _shared._validate_backend_mode
_validate_scalar_type = _shared._validate_scalar_type
_validate_generated_argument_names = _shared._validate_generated_argument_names
_maybe_simplify_derivative_function = _shared._maybe_simplify_derivative_function
_derive_python_function_name = _shared._derive_python_function_name
_arg_size = _shared._arg_size
_format_rust_string_literal = _shared._format_rust_string_literal
_describe_output_arg = _shared._describe_output_arg
_emit_exact_length_assert = _shared._emit_exact_length_assert
_emit_min_length_assert = _shared._emit_min_length_assert
_build_shared_helper_lines = _shared._build_shared_helper_lines
_build_composed_input_specs = _shared._build_composed_input_specs
_emit_composed_fixed_repeat_constants = _shared._emit_composed_fixed_repeat_constants
_emit_composed_parameter_ref = _shared._emit_composed_parameter_ref
_compose_composed_helper_base_name = _shared._compose_composed_helper_base_name
_emit_composed_primal_single_block = _shared._emit_composed_primal_single_block
_emit_composed_primal_repeat_block = _shared._emit_composed_primal_repeat_block
_emit_composed_gradient_forward_single_block = _shared._emit_composed_gradient_forward_single_block
_emit_composed_gradient_forward_repeat_block = _shared._emit_composed_gradient_forward_repeat_block
_emit_composed_gradient_reverse_single_block = _shared._emit_composed_gradient_reverse_single_block
_emit_composed_gradient_reverse_repeat_block = _shared._emit_composed_gradient_reverse_repeat_block

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
    from ...composed_function import _RepeatStage, _SingleStage

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
    render_context = KernelRenderContext(
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
    )
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

    driver_source = render_kernel_source(render_context,
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
    from ...composed_function import _RepeatStage, _SingleStage

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
    render_context = KernelRenderContext(
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
    )
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

    driver_source = render_kernel_source(render_context,
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
