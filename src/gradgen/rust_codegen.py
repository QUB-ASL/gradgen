"""Rust primal code generation for symbolic functions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re

from jinja2 import Environment, FileSystemLoader

from ._rust_codegen import CodeGenerationBuilder, FunctionBundle, sanitize_ident
from .function import Function
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


RustBackendMode = str
RustScalarType = str


@dataclass(frozen=True, slots=True)
class RustBackendConfig:
    """Configuration for generated Rust source and project layout.

    The config object groups Rust backend options into a single immutable
    value. Users can build it incrementally with ``with_...`` methods:

    ``RustBackendConfig().with_backend_mode("no_std").with_math_lib("libm")``

    The same config can be reused for both source generation and project
    creation, which keeps backend behavior consistent as options grow.
    """

    backend_mode: RustBackendMode = "std"
    scalar_type: RustScalarType = "f64"
    math_library: str | None = None
    crate_name: str | None = None
    function_name: str | None = None
    emit_metadata_helpers: bool = True

    def with_backend_mode(self, backend_mode: RustBackendMode) -> RustBackendConfig:
        """Return a copy with a different Rust backend mode.

        The backend mode must be one of the supported code-generation
        targets, currently ``"std"`` or ``"no_std"``.
        """
        _validate_backend_mode(backend_mode)
        return replace(self, backend_mode=backend_mode)

    def with_math_lib(self, math_library: str | None) -> RustBackendConfig:
        """Return a copy with a different ``no_std`` math namespace."""
        return replace(self, math_library=math_library)

    def with_scalar_type(self, scalar_type: RustScalarType) -> RustBackendConfig:
        """Return a copy with a different generated Rust scalar type.

        Supported scalar types are currently ``"f64"`` and ``"f32"``.
        The selected type affects slice signatures, floating-point
        literals, and the emitted math calls for ``no_std`` backends.
        """
        _validate_scalar_type(scalar_type)
        return replace(self, scalar_type=scalar_type)

    def with_crate_name(self, crate_name: str | None) -> RustBackendConfig:
        """Return a copy with a different generated crate name.

        Crate names must already be valid simple Rust/Cargo identifiers.
        This method intentionally rejects names that would need implicit
        sanitization so configuration errors are caught early.
        """
        _validate_crate_name(crate_name)
        return replace(self, crate_name=crate_name)

    def with_function_name(self, function_name: str | None) -> RustBackendConfig:
        """Return a copy with a different generated Rust function name."""
        return replace(self, function_name=function_name)

    def with_emit_metadata_helpers(self, emit_metadata_helpers: bool) -> RustBackendConfig:
        """Return a copy with metadata helper emission enabled or disabled.

        Metadata helpers are the generated constants and convenience
        functions that describe the kernel shape, such as workspace size,
        input dimensions, and output dimensions. Turning them off keeps the
        emitted Rust smaller, but users then need to rely on Python-side
        metadata or their own bookkeeping when allocating buffers.
        """
        return replace(self, emit_metadata_helpers=emit_metadata_helpers)


@dataclass(frozen=True, slots=True)
class RustCodegenResult:
    """Generated Rust source and metadata for a symbolic function."""

    source: str
    function_name: str
    workspace_size: int
    input_sizes: tuple[int, ...]
    output_sizes: tuple[int, ...]
    backend_mode: RustBackendMode
    scalar_type: RustScalarType
    math_library: str | None


@dataclass(frozen=True, slots=True)
class RustProjectResult:
    """Information about a generated Rust project on disk."""

    project_dir: Path
    cargo_toml: Path
    readme: Path
    lib_rs: Path
    codegen: RustCodegenResult


@dataclass(frozen=True, slots=True)
class RustDerivativeBundleResult:
    """Information about a generated directory of derivative Rust crates."""

    bundle_dir: Path
    primal: RustProjectResult | None
    jacobians: tuple[RustProjectResult, ...]
    hessians: tuple[RustProjectResult, ...]


@dataclass(frozen=True, slots=True)
class RustMultiFunctionProjectResult:
    """Information about a generated single-crate multi-function project."""

    project_dir: Path
    cargo_toml: Path
    readme: Path
    lib_rs: Path
    codegens: tuple[RustCodegenResult, ...]


@dataclass(frozen=True, slots=True)
class _ArgSpec:
    """Rendered metadata for a generated Rust input or output."""

    raw_name: str
    rust_name: str
    rust_label: str
    doc_description: str
    size: int


@dataclass(frozen=True, slots=True)
class _ComposedSinglePlan:
    """Codegen metadata for one explicit composed stage."""

    helper_name: str
    vjp_helper_name: str
    parameter_kind: str
    parameter_size: int
    parameter_offset: int
    fixed_values: tuple[float, ...]
    stage_index: int


@dataclass(frozen=True, slots=True)
class _ComposedRepeatPlan:
    """Codegen metadata for one repeated composed stage block."""

    helper_name: str
    vjp_helper_name: str
    parameter_kind: str
    parameter_size: int
    parameter_offset: int
    fixed_values: tuple[tuple[float, ...], ...]
    repeat_count: int
    stage_start_index: int
    const_name: str




def generate_rust(
    function: Function,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
    shared_helper_nodes: tuple[SXNode, ...] | None = None,
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

    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    _validate_backend_mode(resolved_config.backend_mode)
    _validate_scalar_type(resolved_config.scalar_type)
    resolved_math_library = _resolve_math_library(
        resolved_config.backend_mode,
        resolved_config.math_library,
    )

    name = sanitize_ident(resolved_config.function_name or function.name)
    input_sizes = tuple(_arg_size(arg) for arg in function.inputs)
    output_sizes = tuple(_arg_size(arg) for arg in function.outputs)
    input_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_input_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(function.input_names, input_sizes)
    )
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

    scalar_bindings: dict[SXNode, str] = {}
    input_assert_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_write_lines: list[str] = []
    computation_lines: list[str] = []

    for input_spec, input_arg in zip(input_specs, function.inputs):
        input_assert_lines.append(
            f"assert_eq!({input_spec.rust_name}.len(), {input_spec.size});"
        )
        for scalar_index, scalar in enumerate(_flatten_arg(input_arg)):
            scalar_bindings[scalar.node] = f"{input_spec.rust_name}[{scalar_index}]"

    workspace_map, workspace_size = _allocate_workspace_slots(function)

    for node, work_index in workspace_map.items():
        rhs = _emit_node_expr(
            SX(node),
            scalar_bindings,
            workspace_map,
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        )
        computation_lines.append(f"work[{work_index}] = {rhs};")

    for output_spec, output_arg in zip(output_specs, function.outputs):
        output_assert_lines.append(
            f"assert_eq!({output_spec.rust_name}.len(), {output_spec.size});"
        )
        custom_helper_call = _emit_custom_vector_output_helper_call(
            output_arg,
            output_spec.rust_name,
            scalar_bindings,
            workspace_map,
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        )
        if custom_helper_call is not None:
            output_write_lines.append(custom_helper_call)
            continue
        matvec_helper_call = _emit_matvec_output_helper_call(
            output_arg,
            output_spec.rust_name,
            scalar_bindings,
            workspace_map,
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        )
        if matvec_helper_call is not None:
            output_write_lines.append(matvec_helper_call)
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
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        output_assert_lines=output_assert_lines,
        computation_lines=computation_lines,
        output_write_lines=output_write_lines,
        shared_helper_lines=_build_shared_helper_lines(
            shared_helper_nodes or function.nodes,
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        ),
    )

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        function_name=name,
        workspace_size=workspace_size,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


def create_rust_project(
    function: Function,
    path: str | Path,
    *,
    config: RustBackendConfig | None = None,
    crate_name: str | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
) -> RustProjectResult:
    """Create a minimal Rust library project containing generated code."""
    resolved_config = _resolve_backend_config(
        config,
        crate_name=crate_name,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    _validate_backend_mode(resolved_config.backend_mode)
    _validate_scalar_type(resolved_config.scalar_type)

    project_dir = Path(path).expanduser().resolve()
    crate = sanitize_ident(resolved_config.crate_name or function.name)
    codegen = generate_rust(
        function,
        config=resolved_config,
    )

    src_dir = project_dir / "src"
    cargo_toml = project_dir / "Cargo.toml"
    readme = project_dir / "README.md"
    lib_rs = src_dir / "lib.rs"

    src_dir.mkdir(parents=True, exist_ok=True)
    cargo_toml.write_text(
        _get_template("Cargo.toml.j2").render(
            crate_name=crate,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=codegen.math_library,
        ),
        encoding="utf-8",
    )
    readme.write_text(
        _get_template("rust_project_README.md.j2").render(
            crate_name=crate,
            codegen=codegen,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=codegen.math_library,
        ),
        encoding="utf-8",
    )
    lib_rs.write_text(codegen.source, encoding="utf-8")

    return RustProjectResult(
        project_dir=project_dir,
        cargo_toml=cargo_toml,
        readme=readme,
        lib_rs=lib_rs,
        codegen=codegen,
    )


def create_rust_derivative_bundle(
    function: Function,
    path: str | Path,
    *,
    config: RustBackendConfig | None = None,
    include_primal: bool = True,
    include_jacobians: bool = True,
    include_hessians: bool = True,
    simplify_derivatives: int | str | None = None,
) -> RustDerivativeBundleResult:
    """Create a directory containing Rust crates for primal and derivatives."""
    bundle_dir = Path(path).expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    primal_project: RustProjectResult | None = None
    if include_primal:
        primal_project = create_rust_project(
            function,
            bundle_dir / "primal",
            config=config,
        )

    jacobian_projects: list[RustProjectResult] = []
    if include_jacobians:
        for block in function.jacobian_blocks():
            derivative_function = _maybe_simplify_derivative_function(block, simplify_derivatives)
            jacobian_projects.append(
                create_rust_project(
                    derivative_function,
                    bundle_dir / derivative_function.name,
                    config=config,
                )
            )

    hessian_projects: list[RustProjectResult] = []
    if include_hessians and len(function.outputs) == 1 and isinstance(function.outputs[0], SX):
        for block in function.hessian_blocks():
            derivative_function = _maybe_simplify_derivative_function(block, simplify_derivatives)
            hessian_projects.append(
                create_rust_project(
                    derivative_function,
                    bundle_dir / derivative_function.name,
                    config=config,
                )
            )

    return RustDerivativeBundleResult(
        bundle_dir=bundle_dir,
        primal=primal_project,
        jacobians=tuple(jacobian_projects),
        hessians=tuple(hessian_projects),
    )


def create_multi_function_rust_project(
    functions: tuple[Function, ...],
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

    project_dir = Path(path).expanduser().resolve()
    crate = sanitize_ident(resolved_config.crate_name or functions[0].name)

    src_dir = project_dir / "src"
    cargo_toml = project_dir / "Cargo.toml"
    readme = project_dir / "README.md"
    lib_rs = src_dir / "lib.rs"

    src_dir.mkdir(parents=True, exist_ok=True)

    codegens = tuple(
        generate_rust(
            function,
            config=resolved_config,
            function_name=function.name,
            function_index=index,
            shared_helper_nodes=tuple(
                node
                for generated_function in functions
                for node in generated_function.nodes
            )
            if index == 0
            else (),
        )
        for index, function in enumerate(functions)
    )

    cargo_toml.write_text(
        _get_template("Cargo.toml.j2").render(
            crate_name=crate,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=_resolve_math_library(
                resolved_config.backend_mode,
                resolved_config.math_library,
            ),
        ),
        encoding="utf-8",
    )
    readme.write_text(
        _get_template("rust_multi_project_README.md.j2").render(
            crate_name=crate,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=_resolve_math_library(
                resolved_config.backend_mode,
                resolved_config.math_library,
            ),
            codegens=codegens,
        ),
        encoding="utf-8",
    )
    lib_rs.write_text(_render_multi_function_lib(codegens, resolved_config), encoding="utf-8")

    return RustMultiFunctionProjectResult(
        project_dir=project_dir,
        cargo_toml=cargo_toml,
        readme=readme,
        lib_rs=lib_rs,
        codegens=codegens,
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
    resolved_math_library = _resolve_math_library(
        resolved_config.backend_mode,
        resolved_config.math_library,
    )

    name = sanitize_ident(resolved_config.function_name or composed.name)
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
            helper_name = sanitize_ident(f"{name}_stage_{block_index}_{step.function.name}")
            helper_codegen = generate_rust(
                step.function,
                config=helper_config,
                function_name=helper_name,
                function_index=1,
                shared_helper_nodes=(),
                emit_crate_header=False,
                emit_docs=False,
                function_keyword="fn",
            )
            helper_sources.append(helper_codegen.source.rstrip())
            helper_nodes.extend(step.function.nodes)
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

        helper_name = sanitize_ident(f"{name}_repeat_{block_index}_{step.function.name}")
        helper_codegen = generate_rust(
            step.function,
            config=helper_config,
            function_name=helper_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.append(helper_codegen.source.rstrip())
        helper_nodes.extend(step.function.nodes)
        max_helper_workspace = max(max_helper_workspace, helper_codegen.workspace_size)

        const_name = sanitize_ident(f"{name}_repeat_{block_index}_params").upper()
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

    terminal_helper_name = sanitize_ident(f"{name}_terminal_{terminal.function.name}")
    terminal_codegen = generate_rust(
        terminal.function,
        config=helper_config,
        function_name=terminal_helper_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )
    helper_sources.append(terminal_codegen.source.rstrip())
    helper_nodes.extend(terminal.function.nodes)
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
    input_assert_lines = [f"assert_eq!({input_specs[0].rust_name}.len(), {state_size});"]
    if composed.parameter_size > 0:
        input_assert_lines.append(
            f"assert_eq!({input_specs[1].rust_name}.len(), {composed.parameter_size});"
        )
    output_assert_lines = [f"assert_eq!({output_specs[0].rust_name}.len(), 1);"]

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
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        output_assert_lines=output_assert_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=shared_helper_lines,
    ).rstrip()
    source_sections = [driver_source, *helper_sources]
    source = "\n\n".join(section for section in source_sections if section)

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        function_name=name,
        workspace_size=workspace_size,
        input_sizes=tuple(spec.size for spec in input_specs),
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
    resolved_math_library = _resolve_math_library(
        resolved_config.backend_mode,
        resolved_config.math_library,
    )

    name = sanitize_ident(resolved_config.function_name or gradient.name)
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
            helper_name = sanitize_ident(f"{name}_stage_{block_index}_{step.function.name}")
            vjp_helper_name = sanitize_ident(f"{name}_stage_{block_index}_{step.function.name}_vjp")
            helper_codegen = generate_rust(
                step.function,
                config=helper_config,
                function_name=helper_name,
                function_index=1,
                shared_helper_nodes=(),
                emit_crate_header=False,
                emit_docs=False,
                function_keyword="fn",
            )
            vjp_codegen = generate_rust(
                step.function.vjp(wrt_index=0, name=vjp_helper_name),
                config=helper_config,
                function_name=vjp_helper_name,
                function_index=1,
                shared_helper_nodes=(),
                emit_crate_header=False,
                emit_docs=False,
                function_keyword="fn",
            )
            helper_sources.extend((helper_codegen.source.rstrip(), vjp_codegen.source.rstrip()))
            helper_nodes.extend((*step.function.nodes, *step.function.vjp(wrt_index=0).nodes))
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

        helper_name = sanitize_ident(f"{name}_repeat_{block_index}_{step.function.name}")
        vjp_helper_name = sanitize_ident(f"{name}_repeat_{block_index}_{step.function.name}_vjp")
        helper_codegen = generate_rust(
            step.function,
            config=helper_config,
            function_name=helper_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        vjp_codegen = generate_rust(
            step.function.vjp(wrt_index=0, name=vjp_helper_name),
            config=helper_config,
            function_name=vjp_helper_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.extend((helper_codegen.source.rstrip(), vjp_codegen.source.rstrip()))
        helper_nodes.extend((*step.function.nodes, *step.function.vjp(wrt_index=0).nodes))
        max_helper_workspace = max(
            max_helper_workspace,
            helper_codegen.workspace_size,
            vjp_codegen.workspace_size,
        )

        const_name = sanitize_ident(f"{name}_repeat_{block_index}_params").upper()
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

    terminal_gradient_name = sanitize_ident(f"{name}_terminal_{terminal.function.name}_grad")
    terminal_gradient_function = terminal.function.gradient(0, name=terminal_gradient_name)
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
    input_assert_lines = [f"assert_eq!({input_specs[0].rust_name}.len(), {state_size});"]
    if composed.parameter_size > 0:
        input_assert_lines.append(
            f"assert_eq!({input_specs[1].rust_name}.len(), {composed.parameter_size});"
        )
    output_assert_lines = [f"assert_eq!({output_specs[0].rust_name}.len(), {state_size});"]

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
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        output_assert_lines=output_assert_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=shared_helper_lines,
    ).rstrip()
    source_sections = [driver_source, *helper_sources]
    source = "\n\n".join(section for section in source_sections if section)

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        function_name=name,
        workspace_size=workspace_size,
        input_sizes=tuple(spec.size for spec in input_specs),
        output_sizes=(state_size,),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


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
        return f"&{parameters_name}[{parameter_offset} + {index_var}..{parameter_offset} + {index_var} + 1]"
    return (
        f"&{parameters_name}[{parameter_offset} + ({index_var} * {parameter_size})"
        f"..{parameter_offset} + (({index_var} + 1) * {parameter_size})]"
    )


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
        f"    let stage_index = {plan.stage_start_index} + repeat_index;",
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
        f"    let stage_index = {plan.stage_start_index} + repeat_index;",
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


def _allocate_workspace_slots(function: Function) -> tuple[dict[SXNode, int], int]:
    """Assign reusable workspace slots based on each node's last use."""
    workspace_nodes = [node for node in function.nodes if node.op not in {"symbol", "const"}]
    if not workspace_nodes:
        return {}, 0

    node_index = {node: index for index, node in enumerate(workspace_nodes)}
    output_refs: list[SX] = [scalar for output in function.outputs for scalar in _flatten_arg(output)]
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
) -> tuple[str, ...]:
    """Return module-scope helper definitions needed by generated kernels."""
    used_ops = {node.op for node in nodes}
    lines: list[str] = []

    if {"matvec_component", "quadform", "bilinear_form"} & used_ops:
        lines.extend(
            [
                f"fn matvec_component(matrix: &[{scalar_type}], rows: usize, cols: usize, row: usize, x: &[{scalar_type}]) -> {scalar_type} {{",
                "    debug_assert_eq!(matrix.len(), rows * cols);",
                "    debug_assert_eq!(x.len(), cols);",
                "    let start = row * cols;",
                "    matrix[start..start + cols]",
                "        .iter()",
                "        .zip(x.iter())",
                "        .map(|(entry, value)| *entry * *value)",
                "        .sum()",
                "}",
                f"fn matvec(matrix: &[{scalar_type}], rows: usize, cols: usize, x: &[{scalar_type}], y: &mut [{scalar_type}]) {{",
                "    debug_assert_eq!(y.len(), rows);",
                "    for row in 0..rows {",
                "        y[row] = matvec_component(matrix, rows, cols, row, x);",
                "    }",
                "}",
                f"fn bilinear_form(x: &[{scalar_type}], matrix: &[{scalar_type}], rows: usize, cols: usize, y: &[{scalar_type}]) -> {scalar_type} {{",
                "    debug_assert_eq!(x.len(), rows);",
                "    debug_assert_eq!(y.len(), cols);",
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

    custom_helper_lines = _build_custom_helper_lines(nodes, scalar_type, math_library)
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
        if not spec.is_scalar and helper_kind == "jacobian":
            emitted.extend(_build_custom_vector_jacobian_wrapper_lines(spec, scalar_type))
        if not spec.is_scalar and helper_kind == "hvp":
            emitted.extend(_build_custom_vector_hvp_wrapper_lines(spec, scalar_type))
        if not spec.is_scalar and helper_kind == "hessian":
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
    if crate_name is None:
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", crate_name):
        raise ValueError(
            "crate_name must match the pattern [A-Za-z_][A-Za-z0-9_]*"
        )


def _resolve_math_library(
    backend_mode: RustBackendMode,
    math_library: str | None,
) -> str | None:
    """Resolve the math library namespace for the selected backend mode."""
    if backend_mode == "std":
        return math_library
    return math_library or "libm"


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
    if math_library is not None:
        resolved = resolved.with_math_lib(math_library)
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
    if config.backend_mode == "no_std":
        sections.append("#![no_std]")

    for codegen in codegens:
        source = codegen.source
        if source.startswith("#![no_std]\n\n"):
            source = source[len("#![no_std]\n\n") :]
        elif source.startswith("#![no_std]\n"):
            source = source[len("#![no_std]\n") :]
        sections.append(source.rstrip())

    rendered = "\n\n".join(section for section in sections if section)
    return rendered if rendered.endswith("\n") else f"{rendered}\n"


def _math_function_name(op: str, scalar_type: RustScalarType) -> str:
    """Return the backend math function name for a scalar type."""
    _validate_scalar_type(scalar_type)
    base_name = _LIBM_FUNCTIONS[op]
    if scalar_type == "f32":
        return f"{base_name}f"
    return base_name


def _get_template(name: str):
    """Return a Jinja2 template from the package template directory."""
    return _template_environment().get_template(name)


def _template_environment() -> Environment:
    """Build the Jinja2 environment used for Rust project rendering."""
    templates_dir = Path(__file__).with_name("templates")
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


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
