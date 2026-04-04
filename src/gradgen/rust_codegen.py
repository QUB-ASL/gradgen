"""Rust primal code generation for symbolic functions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
import os
import json
import logging
import tomllib
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ._rust_codegen import (
    CodeGenerationBuilder,
    FunctionBundle,
    sanitize_ident,
    validate_rust_ident,
    validate_unique_rust_names,
)
from ._rust_codegen.config import RustBackendConfig, RustBackendMode, RustScalarType
from ._rust_codegen.models import (
    RustCodegenResult,
    RustDerivativeBundleResult,
    RustMultiFunctionProjectResult,
    RustProjectResult,
    RustPythonInterfaceProjectResult,
    _ArgSpec,
    _ComposedRepeatPlan,
    _ComposedSinglePlan,
    _SingleShootingHelperBundle,
)
from ._rust_codegen.project import (
    create_rust_derivative_bundle,
    create_rust_project,
    _create_python_interface_project,
    _gradgen_version,
    _metadata_created_at,
    _next_python_interface_version,
    _render_metadata_json,
    _render_python_interface_source,
    _run_cargo_build,
    _run_python_interface_build,
    _try_run_cargo_fmt,
)
from ._rust_codegen.render import (
    _build_custom_helper_lines,
    _build_custom_vector_hessian_wrapper_lines,
    _build_custom_vector_hvp_wrapper_lines,
    _build_custom_vector_jacobian_wrapper_lines,
    _build_shared_helper_lines,
    _emit_custom_scalar_call,
    _emit_custom_scalar_derivative_call,
    _emit_custom_scalar_hvp_call,
    _emit_custom_vector_call,
    _emit_custom_vector_component_call,
    _emit_custom_vector_hessian_entry_call,
    _emit_custom_vector_hessian_output_helper_call,
    _emit_custom_vector_output_helper_call,
    _emit_expr_ref,
    _emit_math_call,
    _emit_matrix_literal,
    _emit_matrix_vector_argument,
    _emit_norm_abs_expr,
    _emit_norm_slice_and_p_arguments,
    _emit_norm_slice_argument,
    _emit_matvec_output_helper_call,
    _identify_direct_custom_output_marker,
    _match_contiguous_slice,
)
from ._rust_codegen.templates import _get_template
from ._rust_codegen.validation import (
    resolve_backend_config as _resolve_backend_config,
    validate_backend_mode as _validate_backend_mode,
    validate_crate_name as _validate_crate_name,
    validate_generated_argument_names as _validate_generated_argument_names,
    validate_scalar_type as _validate_scalar_type,
)
from .ad import jvp
from .function import Function, _add_like, _make_symbolic_input_like, _zero_like
from .map_zip import ZippedFunction, ZippedJacobianFunction, ReducedFunction
from .single_shooting import (
    SingleShootingGradientFunction,
    SingleShootingHvpFunction,
    SingleShootingJointFunction,
    SingleShootingPrimalFunction,
    SingleShootingProblem,
    _single_shooting_bundle_output_names,
)
from .sx import SX, SXNode, SXVector, parse_bilinear_form_args, parse_matvec_component_args, parse_quadform_args
from .custom_elementary import (
    get_registered_elementary_function,
    parse_custom_scalar_args,
    parse_custom_scalar_hvp_args,
    parse_custom_vector_args,
    parse_custom_vector_hessian_entry_args,
    parse_custom_vector_hvp_component_args,
    parse_custom_vector_jacobian_component_args,
    render_custom_rust_snippet,
)
_LOGGER = logging.getLogger(__name__)




def generate_rust(
    function: object,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
    shared_helper_nodes: tuple[SXNode, ...] | None = None,
    shared_helper_suppressed_custom_wrappers: set[tuple[str, str]] | None = None,
    emit_crate_header: bool = True,
    emit_docs: bool = True,
    function_keyword: str = "pub fn",
) -> RustCodegenResult:
    """Generate Rust source code for primal function evaluation."""
    from .composed_function import ComposedFunction, ComposedGradientFunction

    if isinstance(function, ComposedFunction):
        return _generate_composed_primal_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, ComposedGradientFunction):
        return _generate_composed_gradient_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, ZippedFunction):
        return _generate_zipped_primal_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, ZippedJacobianFunction):
        return _generate_zipped_jacobian_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, ReducedFunction):
        return _generate_reduced_primal_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingProblem):
        return _generate_single_shooting_primal_rust(
            function,
            include_states=False,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingPrimalFunction):
        return _generate_single_shooting_primal_rust(
            function.problem,
            include_states=function.include_states,
            config=config,
            function_name=function_name or function.name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingGradientFunction):
        return _generate_single_shooting_gradient_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingHvpFunction):
        return _generate_single_shooting_hvp_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingJointFunction):
        return _generate_single_shooting_joint_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
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
            raise ValueError("math_library is only supported for no_std backend mode")
        resolved_math_library = None
    else:
        resolved_math_library = math_library or "libm"

    name = sanitize_ident(resolved_config.function_name or function.name)
    input_sizes = tuple(_arg_size(arg) for arg in function.inputs)
    output_sizes = tuple(_arg_size(arg) for arg in function.outputs)
    input_specs: list[_ArgSpec] = []
    reachable_nodes = _collect_reachable_nodes(tuple(scalar for arg in function.outputs for scalar in _flatten_arg(arg)))
    public_function = function_keyword.startswith("pub")
    for raw_name, size, input_arg in zip(function.input_names, input_sizes, function.inputs):
        rust_name = sanitize_ident(raw_name)
        if not public_function and all(scalar.node not in reachable_nodes for scalar in _flatten_arg(input_arg)):
            rust_name = sanitize_ident(f"_{raw_name}")
        input_specs.append(
            _ArgSpec(
                raw_name=raw_name,
                rust_name=rust_name,
                rust_label=_format_rust_string_literal(raw_name),
                doc_description=_describe_input_arg(raw_name),
                size=size,
            )
        )
    input_specs = tuple(input_specs)
    output_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_output_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(function.output_names, output_sizes)
    )
    _validate_generated_argument_names(input_specs, output_specs)

    scalar_bindings: dict[SXNode, str] = {}
    input_assert_lines: list[str] = []
    input_return_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_return_lines: list[str] = []
    output_write_lines: list[str] = []
    computation_lines: list[str] = []
    workspace_return_line: str | None = None
    suppressed_custom_wrappers: set[tuple[str, str]] = set()

    for input_spec, input_arg in zip(input_specs, function.inputs):
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            input_spec.rust_name,
            input_spec.raw_name,
            input_spec.size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
        for scalar_index, scalar in enumerate(_flatten_arg(input_arg)):
            scalar_bindings[scalar.node] = f"{input_spec.rust_name}[{scalar_index}]"

    direct_output_helpers: list[str | None] = []
    materialized_output_refs: list[SX] = []
    for output_spec, output_arg in zip(output_specs, function.outputs):
        custom_helper_call = _emit_custom_vector_output_helper_call(
            output_arg,
            output_spec.rust_name,
            scalar_bindings,
            {},
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        )
        if custom_helper_call is not None:
            direct_output_helpers.append(custom_helper_call)
            marker = _identify_direct_custom_output_marker(
                output_arg,
                scalar_bindings,
                {},
                resolved_config.backend_mode,
                resolved_config.scalar_type,
                resolved_math_library,
            )
            if marker is not None:
                suppressed_custom_wrappers.add(marker)
            continue

        matvec_helper_call = _emit_matvec_output_helper_call(
            output_arg,
            output_spec.rust_name,
            scalar_bindings,
            {},
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        )
        if matvec_helper_call is not None:
            direct_output_helpers.append(matvec_helper_call)
            continue

        direct_output_helpers.append(None)
        materialized_output_refs.extend(_flatten_arg(output_arg))

    workspace_map, workspace_size = _allocate_workspace_slots(
        function,
        output_refs=tuple(materialized_output_refs),
    )

    for node, work_index in workspace_map.items():
        rhs = _emit_node_expr(
            SX(node),
            scalar_bindings,
            workspace_map,
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        )
        computation_lines.append(
            _emit_workspace_assignment(
                node,
                work_index,
                rhs,
                scalar_bindings,
                workspace_map,
                resolved_config.backend_mode,
                resolved_config.scalar_type,
                resolved_math_library,
            )
        )

    for output_spec, output_arg, direct_helper_call in zip(output_specs, function.outputs, direct_output_helpers):
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            output_spec.rust_name,
            output_spec.raw_name,
            output_spec.size,
        )
        output_assert_lines.append(_assert)
        output_return_lines.append(_out_return)
        if direct_helper_call is not None:
            output_write_lines.append(
                _reemit_direct_output_helper_call(
                    direct_helper_call,
                    output_arg,
                    output_spec.rust_name,
                    scalar_bindings,
                    workspace_map,
                    resolved_config.backend_mode,
                    resolved_config.scalar_type,
                    resolved_math_library,
                )
            )
            continue
        for scalar_index, scalar in enumerate(_flatten_arg(output_arg)):
            output_ref = _emit_expr_ref(
                scalar,
                scalar_bindings,
                workspace_map,
                resolved_config.backend_mode,
                resolved_config.scalar_type,
                resolved_math_library,
            )
            output_write_lines.append(
                f"{output_spec.rust_name}[{scalar_index}] = {output_ref};"
            )

    workspace_name = "work" if workspace_map else "_work"

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs],
            f"{workspace_name}: &mut [{resolved_config.scalar_type}]",
        ]
    )

    # workspace assertion emits both legacy assert and Result-returning form
    if workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        upper_name=name.upper(),
        emit_crate_header=emit_crate_header,
        emit_docs=emit_docs,
        function_keyword=function_keyword,
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
        output_write_lines=output_write_lines,
        shared_helper_lines=_build_shared_helper_lines(
            function.nodes if shared_helper_nodes is None else shared_helper_nodes,
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
            suppressed_custom_wrappers=(
                suppressed_custom_wrappers
                if shared_helper_nodes is None
                else (shared_helper_suppressed_custom_wrappers or set())
            ),
        ),
    ).rstrip()

    codegen = RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(
            name,
            resolved_config.crate_name,
        ),
        function_name=name,
        workspace_size=workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=input_sizes,
        output_names=tuple(spec.raw_name for spec in output_specs),
        output_sizes=output_sizes,
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )
    return codegen


def create_multi_function_rust_project(
    functions: tuple[object, ...],
    path: str | Path,
    *,
    config: RustBackendConfig | None = None,
) -> RustMultiFunctionProjectResult:
    """Create a single Rust crate containing multiple generated kernels."""
    if not functions:
        raise ValueError("at least one function must be generated")

    resolved_config = config or RustBackendConfig()
    _validate_backend_mode(resolved_config.backend_mode)
    _validate_scalar_type(resolved_config.scalar_type)
    resolved_math_library = "libm" if resolved_config.backend_mode == "no_std" else None
    resolved_math_library_version = "0.2" if resolved_math_library == "libm" else None

    project_dir = Path(path).expanduser().resolve()
    crate = sanitize_ident(resolved_config.crate_name or functions[0].name)

    src_dir = project_dir / "src"
    cargo_toml = project_dir / "Cargo.toml"
    readme = project_dir / "README.md"
    metadata_json = project_dir / "metadata.json"
    lib_rs = src_dir / "lib.rs"

    src_dir.mkdir(parents=True, exist_ok=True)
    generation_config = (
        replace(resolved_config, enable_python_interface=False)
        if resolved_config.enable_python_interface
        else resolved_config
    )
    shared_helper_nodes = tuple(
        node
        for generated_function in functions
        if isinstance(generated_function, Function)
        for node in generated_function.nodes
    )

    codegens = tuple(
        generate_rust(
            function,
            config=generation_config,
            function_name=function.name,
            function_index=index,
            shared_helper_nodes=shared_helper_nodes if index == 0 else (),
            shared_helper_suppressed_custom_wrappers=(
                {
                    marker
                    for generated_function in functions
                    if isinstance(generated_function, Function)
                    for marker in _collect_suppressed_custom_wrappers(
                        generated_function,
                        resolved_config.backend_mode,
                        resolved_config.scalar_type,
                        "libm" if resolved_config.backend_mode == "no_std" else None,
                    )
                }
                if index == 0
                else set()
            ),
        )
        for index, function in enumerate(functions)
    )

    cargo_toml.write_text(
        _get_template("Cargo.toml.j2").render(
            crate_name=crate,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=resolved_math_library,
            math_library_version=resolved_math_library_version,
        ),
        encoding="utf-8",
    )
    readme.write_text(
        _get_template("rust_multi_project_README.md.j2").render(
            crate_name=crate,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=resolved_math_library,
            codegens=codegens,
            enable_python_interface=resolved_config.enable_python_interface,
            python_interface_project_name=(
                f"{crate}_python" if resolved_config.enable_python_interface else None
            ),
        ),
        encoding="utf-8",
    )
    metadata_json.write_text(
        _render_metadata_json(crate, codegens),
        encoding="utf-8",
    )
    lib_source = _render_multi_function_lib(codegens, resolved_config)
    lib_rs.write_text(lib_source, encoding="utf-8")
    stale_pyproject = project_dir / "pyproject.toml"
    if stale_pyproject.exists():
        stale_pyproject.unlink()
    if resolved_config.build_crate:
        _run_cargo_build(project_dir)
    python_interface = None
    if resolved_config.enable_python_interface:
        python_interface = _create_python_interface_project(
            low_level_project_dir=project_dir,
            low_level_crate_name=crate,
            codegens=codegens,
            build_python_interface=resolved_config.build_python_interface,
        )
    _try_run_cargo_fmt(project_dir)

    return RustMultiFunctionProjectResult(
        project_dir=project_dir,
        cargo_toml=cargo_toml,
        readme=readme,
        metadata_json=metadata_json,
        lib_rs=lib_rs,
        codegens=codegens,
        python_interface=python_interface,
    )


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
    from .composed_function import _RepeatStage, _SingleStage

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
    from .composed_function import _RepeatStage, _SingleStage

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
        return _emit_custom_scalar_call(spec.name, value, params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library)
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


def _emit_workspace_assignment(
    node: SXNode,
    work_index: int,
    rhs: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str,
) -> str:
    """Emit one workspace assignment, using compound operators when safe."""
    target = f"work[{work_index}]"
    expr = SX(node)

    if expr.op in {"add", "sub", "mul", "div"}:
        left, right = expr.args
        left_is_target = _workspace_ref_for_node(left.node, workspace_map) == target
        right_is_target = _workspace_ref_for_node(right.node, workspace_map) == target
        if left_is_target:
            other_ref = _emit_expr_ref(
                right,
                scalar_bindings,
                workspace_map,
                backend_mode,
                scalar_type,
                math_library,
            )
            operator = {
                "add": "+=",
                "sub": "-=",
                "mul": "*=",
                "div": "/=",
            }[expr.op]
            return f"{target} {operator} {other_ref};"
        if expr.op in {"add", "mul"} and right_is_target:
            other_ref = _emit_expr_ref(
                left,
                scalar_bindings,
                workspace_map,
                backend_mode,
                scalar_type,
                math_library,
            )
            operator = {
                "add": "+=",
                "mul": "*=",
            }[expr.op]
            return f"{target} {operator} {other_ref};"

    return f"{target} = {rhs};"


def _workspace_ref_for_node(node: SXNode, workspace_map: dict[SXNode, int]) -> str | None:
    """Return the workspace reference for ``node`` when it already has one."""
    work_index = workspace_map.get(node)
    if work_index is None:
        return None
    return f"work[{work_index}]"


def _emit_exact_length_assert(rust_name: str, display_name: str, expected_size: int) -> tuple[str, str, str]:
    """Emit exact-length checks for generated Rust entrypoints.

    Returns a triple of strings: (assert_line, input_return_line, output_return_line).
    Private helpers intentionally emit no assert line so all runtime shape
    validation stays in public ``Result``-returning functions.
    """
    assert_line = ""
    input_return = (
        f'if {rust_name}.len() != {expected_size} {{ '
        f'return Err(GradgenError::InputTooSmall("{display_name} expected length {expected_size}")); '
        f'}};'
    )
    output_return = (
        f'if {rust_name}.len() != {expected_size} {{ '
        f'return Err(GradgenError::OutputTooSmall("{display_name} expected length {expected_size}")); '
        f'}};'
    )
    return (assert_line, input_return, output_return)


def _emit_min_length_assert(rust_name: str, display_name: str, minimum_size: int) -> tuple[str, str]:
    """Emit minimum-length checks for generated Rust entrypoints.

    Returns a pair: (assert_line, return_line). Private helpers intentionally
    emit no assert line so all runtime workspace validation stays in public
    ``Result``-returning functions.
    """
    if minimum_size == 1:
        assert_line = ""
        return_line = (
            f'if {rust_name}.is_empty() {{ '
            f'return Err(GradgenError::WorkspaceTooSmall("{display_name} expected at least 1")); '
            f'}};'
        )
        return (assert_line, return_line)
    assert_line = ""
    return_line = (
        f'if {rust_name}.len() < {minimum_size} {{ '
        f'return Err(GradgenError::WorkspaceTooSmall("{display_name} expected at least {minimum_size}")); '
        f'}};'
    )
    return (assert_line, return_line)


def _allocate_workspace_slots(
    function: Function,
    *,
    output_refs: tuple[SX, ...] | None = None,
) -> tuple[dict[SXNode, int], int]:
    """Assign reusable workspace slots based on each node's last use."""
    if output_refs is None:
        output_refs = tuple(scalar for output in function.outputs for scalar in _flatten_arg(output))
    required_nodes = _collect_required_workspace_nodes(output_refs)
    workspace_nodes = [
        node
        for node in function.nodes
        if node.op not in {"symbol", "const"} and node in required_nodes
    ]
    if not workspace_nodes:
        return {}, 0

    node_index = {node: index for index, node in enumerate(workspace_nodes)}
    last_use = {node: index for index, node in enumerate(workspace_nodes)}

    for index, node in enumerate(workspace_nodes):
        for child in node.args:
            if child in node_index:
                last_use[child] = max(last_use[child], index)

    output_base = len(workspace_nodes)
    for offset, scalar in enumerate(output_refs):
        if scalar.node in node_index:
            last_use[scalar.node] = max(last_use[scalar.node], output_base + offset)

    available_slots: list[int] = []
    expiring_by_index: dict[int, list[int]] = {}
    workspace_map: dict[SXNode, int] = {}
    next_slot = 0

    for index, node in enumerate(workspace_nodes):
        for slot in expiring_by_index.pop(index, []):
            available_slots.append(slot)

        if available_slots:
            slot = min(available_slots)
            available_slots.remove(slot)
        else:
            slot = next_slot
            next_slot += 1

        workspace_map[node] = slot
        expiring_by_index.setdefault(last_use[node], []).append(slot)

    return workspace_map, next_slot


def _collect_required_workspace_nodes(output_refs: tuple[SX, ...]) -> set[SXNode]:
    """Return non-trivial nodes reachable from the materialized outputs."""
    required: set[SXNode] = set()
    stack = [expr.node for expr in output_refs]
    while stack:
        node = stack.pop()
        if node in required or node.op in {"symbol", "const"}:
            continue
        required.add(node)
        stack.extend(node.args)
    return required


def _collect_reachable_nodes(output_refs: tuple[SX, ...]) -> set[SXNode]:
    """Return all reachable nodes (including symbols/constants) from outputs."""
    reachable: set[SXNode] = set()
    stack = [expr.node for expr in output_refs]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(node.args)
    return reachable


def _reemit_direct_output_helper_call(
    direct_helper_call: str,
    output_arg: SX | SXVector,
    output_name: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Rebuild a direct output helper call using the final workspace map."""
    custom_helper_call = _emit_custom_vector_output_helper_call(
        output_arg,
        output_name,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    if custom_helper_call is not None:
        return custom_helper_call
    matvec_helper_call = _emit_matvec_output_helper_call(
        output_arg,
        output_name,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    if matvec_helper_call is not None:
        return matvec_helper_call
    return direct_helper_call


def _collect_suppressed_custom_wrappers(
    function: Function,
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> set[tuple[str, str]]:
    """Return custom wrapper kinds that are superseded by direct output helpers."""
    scalar_bindings: dict[SXNode, str] = {}
    for input_index, input_arg in enumerate(function.inputs):
        for scalar_index, scalar in enumerate(_flatten_arg(input_arg)):
            scalar_bindings[scalar.node] = f"arg_{input_index}[{scalar_index}]"

    suppressed: set[tuple[str, str]] = set()
    for output_arg in function.outputs:
        marker = _identify_direct_custom_output_marker(
            output_arg,
            scalar_bindings,
            {},
            backend_mode,
            scalar_type,
            math_library,
        )
        if marker is not None:
            suppressed.add(marker)
    return suppressed


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
    refs = [
        x_ref,
        w_ref,
    ]
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
    spec = spec  # satisfy type checkers without extra protocols
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


def _flatten_arg(arg: SX | SXVector) -> tuple[SX, ...]:
    """Flatten a scalar or vector argument into scalar expressions."""
    if isinstance(arg, SX):
        return (arg,)
    return arg.elements


def _arg_size(arg: SX | SXVector) -> int:
    """Return the number of scalar elements in an argument."""
    return len(_flatten_arg(arg))


def _scaled_index_expr(base_expr: str, scale: int) -> str:
    """Return ``base_expr * scale`` while removing identity multipliers."""
    if scale == 1:
        return base_expr
    if base_expr.isidentifier():
        return f"{base_expr} * {scale}"
    return f"(({base_expr}) * {scale})"


def _format_float(value: float | None, scalar_type: RustScalarType) -> str:
    """Format a Python float as a Rust floating-point literal."""
    if value is None:
        raise ValueError("expected a concrete floating-point value")
    _validate_scalar_type(scalar_type)
    return f"{repr(float(value))}_{scalar_type}"


def _format_rust_string_literal(value: str) -> str:
    """Format a Python string as a Rust string literal."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _describe_input_arg(raw_name: str) -> str:
    """Describe the semantic role of a generated Rust input slice."""
    if raw_name.startswith("cotangent_") and len(raw_name) > len("cotangent_"):
        base_name = raw_name[len("cotangent_") :]
        return (
            f"cotangent seed associated with declared result `{base_name}`; "
            "use this slice when forming Jacobian-transpose-vector or reverse-mode sensitivity terms"
        )
    if raw_name.startswith("v_") and len(raw_name) > 2:
        base_name = raw_name[2:]
        return (
            f"tangent or direction input associated with declared argument `{base_name}`; "
            "use this slice when forming Hessian-vector-product or directional-derivative terms"
        )
    return f"input slice for the declared argument `{raw_name}`"


def _describe_output_arg(raw_name: str) -> str:
    """Describe the semantic role of a generated Rust output slice."""
    if raw_name.startswith("vjp_") and len(raw_name) > len("vjp_"):
        base_name = raw_name[len("vjp_") :]
        return f"output slice receiving the vector-Jacobian product for declared input `{base_name}`"
    if raw_name.startswith("jacobian_") and len(raw_name) > len("jacobian_"):
        base_name = raw_name[len("jacobian_") :]
        return f"output slice receiving the Jacobian block for declared result `{base_name}`"
    if raw_name.startswith("gradient_") and len(raw_name) > len("gradient_"):
        base_name = raw_name[len("gradient_") :]
        return f"output slice receiving the gradient block for declared result `{base_name}`"
    if raw_name.startswith("hessian_") and len(raw_name) > len("hessian_"):
        base_name = raw_name[len("hessian_") :]
        return f"output slice receiving the Hessian block for declared result `{base_name}`"
    if raw_name.startswith("hvp_") and len(raw_name) > len("hvp_"):
        base_name = raw_name[len("hvp_") :]
        return f"output slice receiving the Hessian-vector product for declared result `{base_name}`"
    return f"primal output slice for the declared result `{raw_name}`"

def _validate_backend_mode(backend_mode: RustBackendMode) -> None:
    """Validate a Rust backend mode string."""
    if backend_mode not in {"std", "no_std"}:
        raise ValueError(f"unsupported Rust backend mode {backend_mode!r}")


def _validate_scalar_type(scalar_type: RustScalarType) -> None:
    """Validate a generated Rust scalar type."""
    if scalar_type not in {"f64", "f32"}:
        raise ValueError(f"unsupported Rust scalar type {scalar_type!r}")


def _validate_crate_name(crate_name: str | None) -> None:
    """Validate an explicitly configured crate name.

    The backend currently requires simple identifier-like crate names so the
    generated Cargo package name and emitted Rust module conventions stay
    aligned and predictable.
    """
    validate_rust_ident(crate_name, label="crate_name")


def _validate_generated_argument_names(
    input_specs: tuple[_ArgSpec, ...],
    output_specs: tuple[_ArgSpec, ...],
) -> None:
    """Validate Rust-facing argument names after sanitization."""
    validate_unique_rust_names(
        [(spec.raw_name, spec.rust_name) for spec in (*input_specs, *output_specs)],
        label="generated argument",
    )
    validate_unique_rust_names(
        [("work", "work"), *[(spec.raw_name, spec.rust_name) for spec in (*input_specs, *output_specs)]],
        label="generated argument",
    )


def _resolve_backend_config(
    config: RustBackendConfig | None,
    *,
    crate_name: str | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
) -> RustBackendConfig:
    """Merge explicit keyword arguments with an optional backend config.

    Explicit keyword arguments override the values carried in ``config``.
    This preserves backward compatibility while allowing callers to adopt
    the structured configuration object gradually.
    """
    resolved = config or RustBackendConfig()
    if crate_name is not None:
        resolved = resolved.with_crate_name(crate_name)
    if function_name is not None:
        resolved = resolved.with_function_name(function_name)
    if backend_mode != "std":
        resolved = resolved.with_backend_mode(backend_mode)
    if scalar_type != "f64":
        resolved = resolved.with_scalar_type(scalar_type)
    return resolved


def _maybe_simplify_derivative_function(
    function: Function,
    simplify_derivatives: int | str | None,
) -> Function:
    """Optionally simplify a derivative function before code generation."""
    if simplify_derivatives is None:
        return function
    return function.simplify(max_effort=simplify_derivatives, name=function.name)


def _render_multi_function_lib(
    codegens: tuple[RustCodegenResult, ...],
    config: RustBackendConfig,
) -> str:
    """Render a crate source file containing many generated functions."""
    sections: list[str] = []
    seen_private_helpers: set[str] = set()
    if config.backend_mode == "no_std":
        sections.append("#![no_std]")

    for codegen in codegens:
        source = codegen.source
        if source.startswith("#![no_std]\n\n"):
            source = source[len("#![no_std]\n\n") :]
        elif source.startswith("#![no_std]\n"):
            source = source[len("#![no_std]\n") :]
        for section in (part.rstrip() for part in source.split("\n\n") if part.strip()):
            helper_key = _private_helper_section_key(section)
            if helper_key is not None:
                if helper_key in seen_private_helpers:
                    continue
                seen_private_helpers.add(helper_key)
            sections.append(section)

    rendered = "\n\n".join(section for section in sections if section)
    return rendered if rendered.endswith("\n") else f"{rendered}\n"


def _derive_python_function_name(function_name: str, crate_name: str | None) -> str:
    """Derive the public Python name from a generated Rust symbol name."""
    candidate = function_name
    if crate_name:
        crate_prefix = sanitize_ident(crate_name)
        prefix = f"{crate_prefix}_"
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
    if candidate.endswith("_f"):
        candidate = candidate[:-2]
    return candidate or function_name


def _private_helper_section_key(section: str) -> str | None:
    """Return a stable deduplication key for a private helper section."""
    match = re.search(r"(?m)^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", section)
    if match is None:
        return None
    function_name = match.group(1)
    if function_name.endswith("_meta"):
        return None
    if re.search(r"(?m)^pub\s+fn\s+", section) is not None:
        return None
    return section


def _math_function_name(op: str, scalar_type: RustScalarType) -> str:
    """Return the backend math function name for a scalar type."""
    _validate_scalar_type(scalar_type)
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

from ._rust_codegen.render import (  # noqa: E402
    _build_custom_helper_lines,
    _build_custom_vector_hessian_wrapper_lines,
    _build_custom_vector_hvp_wrapper_lines,
    _build_custom_vector_jacobian_wrapper_lines,
    _build_shared_helper_lines,
    _emit_custom_scalar_call,
    _emit_custom_scalar_derivative_call,
    _emit_custom_scalar_hvp_call,
    _emit_custom_vector_call,
    _emit_custom_vector_component_call,
    _emit_custom_vector_hessian_entry_call,
    _emit_custom_vector_hessian_output_helper_call,
    _emit_custom_vector_output_helper_call,
    _emit_expr_ref,
    _emit_math_call,
    _emit_matrix_literal,
    _emit_matrix_vector_argument,
    _emit_norm_abs_expr,
    _emit_norm_slice_and_p_arguments,
    _emit_norm_slice_argument,
    _emit_matvec_output_helper_call,
    _identify_direct_custom_output_marker,
    _match_contiguous_slice,
    _match_custom_vector_derivative_output,
    _match_custom_vector_hessian_output,
    _match_passthrough_custom_vector_hessian_entry,
    _match_passthrough_matvec_component,
    _is_passthrough_zero,
    _is_passthrough_one,
    _math_function_name,
)
