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
    from ._rust_codegen.generators import (
        _generate_composed_gradient_rust,
        _generate_composed_primal_rust,
        _generate_reduced_primal_rust,
        _generate_single_shooting_gradient_rust,
        _generate_single_shooting_hvp_rust,
        _generate_single_shooting_joint_rust,
        _generate_single_shooting_primal_rust,
        _generate_zipped_jacobian_rust,
        _generate_zipped_primal_rust,
    )

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
