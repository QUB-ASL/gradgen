"""Rust primal code generation for symbolic functions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re

from jinja2 import Environment, FileSystemLoader

from .function import Function
from .sx import SX, SXNode, SXVector


RustBackendMode = str


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
class _ArgSpec:
    """Rendered metadata for a generated Rust input or output."""

    raw_name: str
    rust_name: str
    size: int


def generate_rust(
    function: Function,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    math_library: str | None = None,
) -> RustCodegenResult:
    """Generate Rust source code for primal function evaluation."""
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        math_library=math_library,
    )
    _validate_backend_mode(resolved_config.backend_mode)
    resolved_math_library = _resolve_math_library(
        resolved_config.backend_mode,
        resolved_config.math_library,
    )

    name = _sanitize_ident(resolved_config.function_name or function.name)
    input_sizes = tuple(_arg_size(arg) for arg in function.inputs)
    output_sizes = tuple(_arg_size(arg) for arg in function.outputs)
    input_specs = tuple(
        _ArgSpec(raw_name=raw_name, rust_name=_sanitize_ident(raw_name), size=size)
        for raw_name, size in zip(function.input_names, input_sizes)
    )
    output_specs = tuple(
        _ArgSpec(raw_name=raw_name, rust_name=_sanitize_ident(raw_name), size=size)
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
                resolved_math_library,
            )
            output_write_lines.append(
                f"{output_spec.rust_name}[{scalar_index}] = {output_ref};"
            )

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[f64]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [f64]" for spec in output_specs],
            "work: &mut [f64]",
        ]
    )

    source = _get_template("lib.rs.j2").render(
        function_name=name,
        upper_name=name.upper(),
        backend_mode=resolved_config.backend_mode,
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
    math_library: str | None = None,
) -> RustProjectResult:
    """Create a minimal Rust library project containing generated code."""
    resolved_config = _resolve_backend_config(
        config,
        crate_name=crate_name,
        function_name=function_name,
        backend_mode=backend_mode,
        math_library=math_library,
    )
    _validate_backend_mode(resolved_config.backend_mode)

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
            math_library=codegen.math_library,
        ),
        encoding="utf-8",
    )
    readme.write_text(
        _get_template("rust_project_README.md.j2").render(
            crate_name=crate,
            codegen=codegen,
            backend_mode=resolved_config.backend_mode,
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


def _emit_node_expr(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    math_library: str | None,
) -> str:
    """Emit the Rust expression used to compute a workspace node."""
    if expr.op == "const":
        return _format_float(expr.value)
    if expr.op == "symbol":
        return scalar_bindings[expr.node]

    args = tuple(
        _emit_expr_ref(arg, scalar_bindings, workspace_map, backend_mode, math_library)
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
        return _emit_math_call("pow", args, backend_mode, math_library)
    if expr.op == "neg":
        return f"-{args[0]}"
    if expr.op == "sin":
        return _emit_math_call("sin", args, backend_mode, math_library)
    if expr.op == "cos":
        return _emit_math_call("cos", args, backend_mode, math_library)
    if expr.op == "exp":
        return _emit_math_call("exp", args, backend_mode, math_library)
    if expr.op == "log":
        return _emit_math_call("log", args, backend_mode, math_library)
    if expr.op == "sqrt":
        return _emit_math_call("sqrt", args, backend_mode, math_library)

    raise ValueError(f"unsupported Rust codegen operation {expr.op!r}")


def _emit_expr_ref(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    math_library: str | None,
) -> str:
    """Emit a Rust expression reference for an already-available value."""
    if expr.op == "const":
        return _format_float(expr.value)
    if expr.op == "symbol":
        return scalar_bindings[expr.node]
    if expr.node in workspace_map:
        return f"work[{workspace_map[expr.node]}]"
    return _emit_node_expr(
        expr,
        scalar_bindings,
        workspace_map,
        backend_mode,
        math_library,
    )


def _emit_math_call(
    op: str,
    args: tuple[str, ...],
    backend_mode: RustBackendMode,
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
        return f"{math_library}::pow({args[0]}, {args[1]})"
    return f"{math_library}::{_LIBM_FUNCTIONS[op]}({args[0]})"


def _flatten_arg(arg: SX | SXVector) -> tuple[SX, ...]:
    """Flatten a scalar or vector argument into scalar expressions."""
    if isinstance(arg, SX):
        return (arg,)
    return arg.elements


def _arg_size(arg: SX | SXVector) -> int:
    """Return the number of scalar elements in an argument."""
    return len(_flatten_arg(arg))


def _format_float(value: float | None) -> str:
    """Format a Python float as a Rust ``f64`` literal."""
    if value is None:
        raise ValueError("expected a concrete floating-point value")
    return repr(float(value))


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
    if math_library is not None:
        resolved = resolved.with_math_lib(math_library)
    return resolved


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
    "sqrt": "sqrt",
}
