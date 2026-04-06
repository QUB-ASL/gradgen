"""Family-specific Rust generation helpers."""

from __future__ import annotations

from . import shared as _shared
from .rendering import KernelRenderContext, render_kernel_source
from ..config import RustBackendConfig, RustBackendMode, RustScalarType
from ..models import (
    RustCodegenResult,
    _SingleShootingHelperBundle,
)
from ..naming import sanitize_ident
from ...single_shooting import (
    SingleShootingGradientFunction,
    SingleShootingHvpFunction,
    SingleShootingJointFunction,
    SingleShootingProblem,
)

generate_rust = _shared.generate_rust
_resolve_backend_config = _shared._resolve_backend_config
_validate_backend_mode = _shared._validate_backend_mode
_validate_scalar_type = _shared._validate_scalar_type
_validate_generated_argument_names = _shared._validate_generated_argument_names
_derive_python_function_name = _shared._derive_python_function_name
_format_float = _shared._format_float
_format_rust_string_literal = _shared._format_rust_string_literal
_emit_exact_length_assert = _shared._emit_exact_length_assert
_emit_min_length_assert = _shared._emit_min_length_assert
_build_shared_helper_lines = _shared._build_shared_helper_lines
_build_single_shooting_helpers = _shared._build_single_shooting_helpers
_build_single_shooting_input_specs = _shared._build_single_shooting_input_specs
_build_single_shooting_output_specs = (
    _shared._build_single_shooting_output_specs
)
_compose_single_shooting_helper_base_name = (
    _shared._compose_single_shooting_helper_base_name
)
_emit_single_shooting_control_slice = (
    _shared._emit_single_shooting_control_slice
)
_emit_single_shooting_stage_range = _shared._emit_single_shooting_stage_range
_emit_small_accumulate = _shared._emit_small_accumulate


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
        raise ValueError(
            "single-shooting kernels must compute cost, gradient, or hvp"
        )

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
            raise ValueError(
                "math_library is only supported for no_std backend mode"
            )
        resolved_math_library = None
    else:
        resolved_math_library = math_library or "libm"
    render_context = KernelRenderContext(
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
    )

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

    return _build_single_shooting_driver_result(
        problem,
        include_cost,
        include_gradient,
        include_hvp,
        include_states,
        resolved_config,
        render_context,
        name,
        function_index,
        helpers,
        resolved_math_library,
    )


def _build_single_shooting_driver_result(
    problem: SingleShootingProblem,
    include_cost: bool,
    include_gradient: bool,
    include_hvp: bool,
    include_states: bool,
    resolved_config: RustBackendConfig,
    render_context: KernelRenderContext,
    name: str,
    function_index: int,
    helpers: _SingleShootingHelperBundle,
    resolved_math_library: str | None,
) -> RustCodegenResult:
    """Build a Rust codegen result for a single-shooting driver."""
    input_specs = _build_single_shooting_input_specs(
        problem, include_hvp=include_hvp
    )
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
    gradient_output_name = output_names.get(
        f"gradient_{problem.control_sequence_name}"
    )
    hvp_output_name = output_names.get(f"hvp_{problem.control_sequence_name}")
    states_output_name = output_names.get("x_traj")
    use_rollout_states_as_history = (
        need_history and include_states and states_output_name is not None
    )

    state_history_size = (
        0
        if use_rollout_states_as_history
        else ((horizon * state_size) if need_history else 0)
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
        computation_lines.append(
            "let (scalar_buffer, stage_work) = rest.split_at_mut(1);"
        )
    else:
        computation_lines.append("let stage_work = rest;")
    computation_lines.append(f"current_state.copy_from_slice({x0_name});")
    if include_hvp:
        computation_lines.append(
            f"current_tangent.fill({_format_float(0.0, resolved_config.scalar_type)});"
        )
    if include_states and states_output_name is not None:
        computation_lines.append(
            f"{states_output_name}[0..{state_size}].copy_from_slice({x0_name});"
        )
    if use_rollout_states_as_history and states_output_name is not None:
        computation_lines.append(
            f"let state_history = &mut {states_output_name}[{state_size}..];"
        )
    if include_cost:
        computation_lines.append(
            f"let mut total_cost = { _format_float(0.0, resolved_config.scalar_type) };"
        )

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
    if (
        include_states
        and states_output_name is not None
        and not use_rollout_states_as_history
    ):
        for_lines.append(
            f"    {states_output_name}[((stage_index + 1) * {state_size})..((stage_index + 2) * {state_size})].copy_from_slice(next_state);"
        )
    for_lines.append(
        "    core::mem::swap(&mut current_state, &mut next_state);"
    )
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
                _emit_small_accumulate(
                    "grad_u_t", "temp_control", control_size, indent="    "
                )
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
                _emit_small_accumulate(
                    "hvp_u_t", "temp_control", control_size, indent="    "
                )
            )
        computation_lines.extend(
            [
                f"    {helpers.stage_cost_grad_x_name}(x_t, u_t, {p_name}, lambda_next, stage_work);",
                f"    {helpers.dynamics_vjp_x_name}(x_t, u_t, {p_name}, &lambda_current[..], temp_state, stage_work);",
            ]
        )
        computation_lines.extend(
            _emit_small_accumulate(
                "lambda_next", "temp_state", state_size, indent="    "
            )
        )
        if include_hvp:
            computation_lines.extend(
                [
                    f"    {helpers.stage_cost_grad_x_jvp_name}(x_t, u_t, {p_name}, tangent_x_t, v_u_t, mu_next, stage_work);",
                    f"    {helpers.dynamics_vjp_x_jvp_name}(x_t, u_t, {p_name}, &lambda_current[..], tangent_x_t, v_u_t, &mu_current[..], temp_state, stage_work);",
                ]
            )
            computation_lines.extend(
                _emit_small_accumulate(
                    "mu_next", "temp_state", state_size, indent="    "
                )
            )
        computation_lines.append(
            "    lambda_current.copy_from_slice(lambda_next);"
        )
        if include_hvp:
            computation_lines.append(
                "    mu_current.copy_from_slice(mu_next);"
            )
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
                _emit_small_accumulate(
                    "grad_u_t", "temp_control", control_size
                )
            )
        if (
            include_hvp
            and hvp_output_name is not None
            and v_U_name is not None
        ):
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
            *[
                f"{spec.rust_name}: &[{resolved_config.scalar_type}]"
                for spec in input_specs
            ],
            *[
                f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]"
                for spec in output_specs
            ],
            "work: &mut [{scalar_type}]".format(
                scalar_type=resolved_config.scalar_type
            ),
        ]
    )
    if workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert(
            "work", "work", workspace_size
        )
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = render_kernel_source(
        render_context,
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
        python_name=_derive_python_function_name(
            name, resolved_config.crate_name
        ),
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
