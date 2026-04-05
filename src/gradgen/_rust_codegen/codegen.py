"""Rust code generation orchestration for symbolic functions."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

from . import (
    CodeGenerationBuilder,
    FunctionBundle,
    sanitize_ident,
    validate_rust_ident,
    validate_unique_rust_names,
)
from .config import RustBackendConfig, RustBackendMode, RustScalarType
from .models import (
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
from .project import (
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
from .project_support import (
    _derive_python_function_name,
    _render_multi_function_lib,
)
from .rendering import (
    _allocate_workspace_slots,
    _arg_size,
    _build_custom_helper_lines,
    _build_custom_vector_hessian_wrapper_lines,
    _build_custom_vector_hvp_wrapper_lines,
    _build_custom_vector_jacobian_wrapper_lines,
    _build_shared_helper_lines,
    _collect_reachable_nodes,
    _collect_required_workspace_nodes,
    _collect_suppressed_custom_wrappers,
    _describe_input_arg,
    _describe_output_arg,
    _emit_custom_scalar_call,
    _emit_custom_scalar_derivative_call,
    _emit_custom_scalar_hvp_call,
    _emit_custom_vector_call,
    _emit_custom_vector_component_call,
    _emit_custom_vector_hessian_entry_call,
    _emit_custom_vector_hessian_output_helper_call,
    _emit_custom_vector_output_helper_call,
    _emit_exact_length_assert,
    _emit_expr_ref,
    _emit_math_call,
    _emit_matvec_output_helper_call,
    _emit_matrix_literal,
    _emit_matrix_vector_argument,
    _emit_min_length_assert,
    _emit_node_expr,
    _emit_norm_abs_expr,
    _emit_norm_slice_and_p_arguments,
    _emit_norm_slice_argument,
    _emit_workspace_assignment,
    _flatten_arg,
    _format_float,
    _format_rust_string_literal,
    _identify_direct_custom_output_marker,
    _is_passthrough_one,
    _is_passthrough_zero,
    _match_contiguous_slice,
    _match_custom_vector_derivative_output,
    _match_custom_vector_hessian_output,
    _match_passthrough_custom_vector_hessian_entry,
    _match_passthrough_matvec_component,
    _math_function_name,
    _reemit_direct_output_helper_call,
    _scaled_index_expr,
    _workspace_ref_for_node,
)
from .templates import _get_template
from .validation import (
    resolve_backend_config as _resolve_backend_config,
    validate_backend_mode as _validate_backend_mode,
    validate_crate_name as _validate_crate_name,
    validate_generated_argument_names as _validate_generated_argument_names,
    validate_scalar_type as _validate_scalar_type,
)
from ..ad import jvp
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
    from ..composed_function import ComposedFunction, ComposedGradientFunction
    from .generators import (
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


__all__ = [name for name in globals() if not name.startswith("__")]
