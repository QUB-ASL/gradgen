"""Rust primal code generation for symbolic functions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re

from jinja2 import Environment, FileSystemLoader

from .function import Function
from .sx import SX, SXNode, SXVector


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
class _BuilderRequest:
    """A requested generated kernel kind."""

    kind: str
    components: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ArgSpec:
    """Rendered metadata for a generated Rust input or output."""

    raw_name: str
    rust_name: str
    rust_label: str
    size: int


@dataclass(frozen=True, slots=True)
class CodeGenerationBuilder:
    """Fluent builder for a single generated Rust crate with many kernels."""

    function: Function
    config: RustBackendConfig = RustBackendConfig()
    requests: tuple[_BuilderRequest, ...] = ()
    simplification: int | str | None = None

    def with_backend_config(self, config: RustBackendConfig) -> CodeGenerationBuilder:
        """Return a copy using ``config`` for generated Rust code."""
        return replace(self, config=config)

    def with_simplification(self, max_effort: int | str | None) -> CodeGenerationBuilder:
        """Return a copy applying ``max_effort`` simplification to all generated kernels."""
        return replace(self, simplification=max_effort)

    def add_primal(self) -> CodeGenerationBuilder:
        """Include the primal function in the generated crate."""
        return self._add_request("primal")

    def add_gradient(self) -> CodeGenerationBuilder:
        """Include gradient kernels for scalar-output functions."""
        return self._add_request("gradient")

    def add_jacobian(self) -> CodeGenerationBuilder:
        """Include Jacobian kernels for all input blocks."""
        return self._add_request("jacobian")

    def add_joint(self, components: tuple[str, ...]) -> CodeGenerationBuilder:
        """Include kernels that compute several requested artifacts together."""
        return self._add_request("joint", components=components)

    def add_hessian(self) -> CodeGenerationBuilder:
        """Include Hessian kernels for scalar-output functions."""
        return self._add_request("hessian")

    def add_hvp(self) -> CodeGenerationBuilder:
        """Include Hessian-vector product kernels for scalar-output functions."""
        return self._add_request("hvp")

    def build(self, path: str | Path) -> RustMultiFunctionProjectResult:
        """Generate a single Rust crate containing all requested kernels."""
        resolved_config = self.config
        if resolved_config.crate_name is None:
            resolved_config = resolved_config.with_crate_name(
                _sanitize_ident(Path(path).expanduser().resolve().name)
            )
        functions = _resolve_builder_functions(
            self.function,
            resolved_config,
            self.requests,
            self.simplification,
        )
        return create_multi_function_rust_project(
            functions,
            path,
            config=resolved_config,
        )

    def _add_request(self, kind: str, *, components: tuple[str, ...] = ()) -> CodeGenerationBuilder:
        candidate = _BuilderRequest(kind, components)
        if any(request == candidate for request in self.requests):
            return self
        return replace(self, requests=(*self.requests, candidate))


def generate_rust(
    function: Function,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
) -> RustCodegenResult:
    """Generate Rust source code for primal function evaluation."""
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

    name = _sanitize_ident(resolved_config.function_name or function.name)
    input_sizes = tuple(_arg_size(arg) for arg in function.inputs)
    output_sizes = tuple(_arg_size(arg) for arg in function.outputs)
    input_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=_sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            size=size,
        )
        for raw_name, size in zip(function.input_names, input_sizes)
    )
    output_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=_sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
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

    workspace_nodes = [node for node in function.nodes if node.op not in {"symbol", "const"}]
    workspace_map = {node: index for index, node in enumerate(workspace_nodes)}

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

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs],
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )

    source = _get_template("lib.rs.j2").render(
        function_name=name,
        upper_name=name.upper(),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=len(workspace_map),
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        output_assert_lines=output_assert_lines,
        computation_lines=computation_lines,
        output_write_lines=output_write_lines,
    )

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        function_name=name,
        workspace_size=len(workspace_map),
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
    crate = _sanitize_ident(resolved_config.crate_name or function.name)
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
    crate = _sanitize_ident(resolved_config.crate_name or functions[0].name)

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
        )
        for function in functions
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
    if expr.op == "neg":
        return f"-{args[0]}"
    if expr.op == "sin":
        return _emit_math_call("sin", args, backend_mode, scalar_type, math_library)
    if expr.op == "cos":
        return _emit_math_call("cos", args, backend_mode, scalar_type, math_library)
    if expr.op == "exp":
        return _emit_math_call("exp", args, backend_mode, scalar_type, math_library)
    if expr.op == "log":
        return _emit_math_call("log", args, backend_mode, scalar_type, math_library)
    if expr.op == "sqrt":
        return _emit_math_call("sqrt", args, backend_mode, scalar_type, math_library)

    raise ValueError(f"unsupported Rust codegen operation {expr.op!r}")


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
        if op == "log":
            return f"{args[0]}.ln()"
        return f"{args[0]}.{op}()"

    if math_library is None:
        raise ValueError("no_std math calls require a resolved math library")
    if op == "pow":
        return f"{math_library}::{_math_function_name(op, scalar_type)}({args[0]}, {args[1]})"
    return f"{math_library}::{_math_function_name(op, scalar_type)}({args[0]})"


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


def _sanitize_ident(name: str) -> str:
    """Convert a user-facing name into a simple Rust identifier."""
    chars = [character if character.isalnum() or character == "_" else "_" for character in name]
    ident = "".join(chars)
    if not ident:
        ident = "value"
    if ident[0].isdigit():
        ident = f"_{ident}"
    return ident


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


def _resolve_builder_functions(
    function: Function,
    config: RustBackendConfig,
    requests: tuple[_BuilderRequest, ...],
    simplification: int | str | None,
) -> tuple[Function, ...]:
    """Expand builder requests into concrete symbolic functions."""
    if not requests:
        raise ValueError("no kernels were requested; call add_primal() or another add_* method first")

    base_function = _apply_builder_base_name(function, config.function_name)
    crate_prefix = _sanitize_ident(config.crate_name or base_function.name)
    resolved: list[Function] = []

    for request in requests:
        if request.kind == "primal":
            resolved.append(
                _rename_generated_function(
                    _maybe_simplify_generated_function(base_function, simplification),
                    _builder_function_name(crate_prefix, "f"),
                )
            )
            continue
        if request.kind == "gradient":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(base_function.gradient(index), simplification),
                    _builder_function_name(
                        crate_prefix,
                        "grad",
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index in range(len(base_function.inputs))
            )
            continue
        if request.kind == "jacobian":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(block, simplification),
                    _builder_function_name(
                        crate_prefix,
                        "jf",
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index, block in enumerate(base_function.jacobian_blocks())
            )
            continue
        if request.kind == "joint":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(
                        base_function.joint(
                            request.components,
                            index,
                        ),
                        simplification,
                    ),
                    _builder_function_name(
                        crate_prefix,
                        *_builder_joint_labels(request.components),
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index in range(len(base_function.inputs))
            )
            continue
        if request.kind == "hessian":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(block, simplification),
                    _builder_function_name(
                        crate_prefix,
                        "hessian",
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index, block in enumerate(base_function.hessian_blocks())
            )
            continue
        if request.kind == "hvp":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(block, simplification),
                    _builder_function_name(
                        crate_prefix,
                        "hvp",
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index, block in enumerate(base_function.hvp_blocks())
            )
            continue
        raise ValueError(f"unsupported builder request kind {request.kind!r}")

    return tuple(resolved)


def _apply_builder_base_name(function: Function, function_name: str | None) -> Function:
    """Optionally rename the base function used by the builder."""
    if function_name is None or function_name == function.name:
        return function
    return Function(
        function_name,
        function.inputs,
        function.outputs,
        input_names=function.input_names,
        output_names=function.output_names,
    )


def _rename_generated_function(function: Function, name: str) -> Function:
    """Return ``function`` with a different name while preserving its interface."""
    if function.name == name:
        return function
    return Function(
        name,
        function.inputs,
        function.outputs,
        input_names=function.input_names,
        output_names=function.output_names,
    )


def _builder_joint_labels(components: tuple[str, ...]) -> tuple[str, ...]:
    """Map joint component shorthands to builder function-name labels."""
    mapping = {
        "f": "f",
        "jf": "jf",
        "hvp": "hvp",
    }
    return tuple(mapping[component] for component in components)


def _builder_function_name(
    crate_prefix: str,
    *labels: str,
    input_name: str | None = None,
    include_input_name: bool = False,
) -> str:
    """Build a crate-prefixed Rust function name for builder-generated kernels."""
    parts = [crate_prefix, *labels]
    if include_input_name and input_name is not None:
        parts.append(_sanitize_ident(input_name))
    return "_".join(parts)


def _maybe_simplify_generated_function(
    function: Function,
    simplification: int | str | None,
) -> Function:
    """Optionally simplify a generated function while preserving its name."""
    if simplification is None:
        return function
    return function.simplify(max_effort=simplification, name=function.name)


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
    "exp": "exp",
    "log": "log",
    "pow": "pow",
    "sqrt": "sqrt",
}
