"""Rust primal code generation for symbolic functions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .function import Function
from .sx import SX, SXNode, SXVector


@dataclass(frozen=True, slots=True)
class RustCodegenResult:
    """Generated Rust source and metadata for a symbolic function."""

    source: str
    function_name: str
    workspace_size: int
    input_sizes: tuple[int, ...]
    output_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RustProjectResult:
    """Information about a generated Rust project on disk."""

    project_dir: Path
    cargo_toml: Path
    readme: Path
    lib_rs: Path
    codegen: RustCodegenResult


def generate_rust(function: Function, *, function_name: str | None = None) -> RustCodegenResult:
    """Generate Rust source code for primal function evaluation.

    The generated function uses a slice-based ABI:

    - each scalar or vector input is passed as ``&[f64]``
    - each scalar or vector output is passed as ``&mut [f64]``
    - intermediate values are stored in ``work: &mut [f64]``
    """
    name = _sanitize_ident(function_name or function.name)
    input_sizes = tuple(_arg_size(arg) for arg in function.inputs)
    output_sizes = tuple(_arg_size(arg) for arg in function.outputs)

    scalar_bindings: dict[SXNode, str] = {}
    input_access_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_write_lines: list[str] = []
    computation_lines: list[str] = []

    for index, (input_name, input_arg, input_size) in enumerate(
        zip(function.input_names, function.inputs, input_sizes)
    ):
        rust_name = _sanitize_ident(input_name)
        input_access_lines.append(f"    assert_eq!({rust_name}.len(), {input_size});")
        for scalar_index, scalar in enumerate(_flatten_arg(input_arg)):
            scalar_bindings[scalar.node] = f"{rust_name}[{scalar_index}]"

    workspace_nodes = [
        node for node in function.nodes if node.op not in {"symbol", "const"}
    ]
    workspace_map = {node: index for index, node in enumerate(workspace_nodes)}

    for node, work_index in workspace_map.items():
        rhs = _emit_node_expr(SX(node), scalar_bindings, workspace_map)
        computation_lines.append(f"    work[{work_index}] = {rhs};")

    for output_name, output_arg, output_size in zip(
        function.output_names, function.outputs, output_sizes
    ):
        rust_name = _sanitize_ident(output_name)
        output_assert_lines.append(f"    assert_eq!({rust_name}.len(), {output_size});")
        for scalar_index, scalar in enumerate(_flatten_arg(output_arg)):
            output_ref = _emit_expr_ref(scalar, scalar_bindings, workspace_map)
            output_write_lines.append(f"    {rust_name}[{scalar_index}] = {output_ref};")

    parameters = ", ".join(
        [
            *[
                f"{_sanitize_ident(name)}: &[f64]"
                for name in function.input_names
            ],
            *[
                f"{_sanitize_ident(name)}: &mut [f64]"
                for name in function.output_names
            ],
            "work: &mut [f64]",
        ]
    )

    upper_name = name.upper()
    input_size_fn_lines = _render_size_functions(
        base_name=name,
        role="input",
        names=function.input_names,
        sizes=input_sizes,
        prefix=upper_name,
    )
    output_size_fn_lines = _render_size_functions(
        base_name=name,
        role="output",
        names=function.output_names,
        sizes=output_sizes,
        prefix=upper_name,
    )

    lines = [
        f"/// Workspace length required by [`{name}`].",
        f"pub const {upper_name}_WORK_SIZE: usize = {len(workspace_map)};",
        "",
        f"/// Return the workspace length required by [`{name}`].",
        f"pub fn {name}_work_size() -> usize {{",
        f"    {upper_name}_WORK_SIZE",
        "}",
        "",
        *input_size_fn_lines,
        *output_size_fn_lines,
        f"/// Evaluate the generated symbolic function `{name}`.",
        "///",
        "/// Inputs are passed as immutable slices, outputs are written into mutable slices,",
        "/// and intermediate values are stored in `work`.",
        f"pub fn {name}({parameters}) {{",
        f"    assert!(work.len() >= {len(workspace_map)});",
        *input_access_lines,
        *output_assert_lines,
        *computation_lines,
        *output_write_lines,
        "}",
    ]

    return RustCodegenResult(
        source="\n".join(lines) + "\n",
        function_name=name,
        workspace_size=len(workspace_map),
        input_sizes=input_sizes,
        output_sizes=output_sizes,
    )


def create_rust_project(
    function: Function,
    path: str | Path,
    *,
    crate_name: str | None = None,
    function_name: str | None = None,
) -> RustProjectResult:
    """Create a minimal Rust library project containing generated code."""
    project_dir = Path(path).expanduser().resolve()
    crate = _sanitize_ident(crate_name or function.name)
    codegen = generate_rust(function, function_name=function_name)

    src_dir = project_dir / "src"
    cargo_toml = project_dir / "Cargo.toml"
    readme = project_dir / "README.md"
    lib_rs = src_dir / "lib.rs"

    src_dir.mkdir(parents=True, exist_ok=True)
    cargo_toml.write_text(_render_cargo_toml(crate), encoding="utf-8")
    readme.write_text(
        _render_project_readme(crate, codegen),
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
) -> str:
    """Emit the Rust expression used to compute a workspace node."""
    if expr.op == "const":
        return _format_float(expr.value)
    if expr.op == "symbol":
        return scalar_bindings[expr.node]

    args = tuple(_emit_expr_ref(arg, scalar_bindings, workspace_map) for arg in expr.args)

    if expr.op == "add":
        return f"{args[0]} + {args[1]}"
    if expr.op == "sub":
        return f"{args[0]} - {args[1]}"
    if expr.op == "mul":
        return f"{args[0]} * {args[1]}"
    if expr.op == "div":
        return f"{args[0]} / {args[1]}"
    if expr.op == "pow":
        return f"{args[0]}.powf({args[1]})"
    if expr.op == "neg":
        return f"-{args[0]}"
    if expr.op == "sin":
        return f"{args[0]}.sin()"
    if expr.op == "cos":
        return f"{args[0]}.cos()"
    if expr.op == "exp":
        return f"{args[0]}.exp()"
    if expr.op == "log":
        return f"{args[0]}.ln()"
    if expr.op == "sqrt":
        return f"{args[0]}.sqrt()"

    raise ValueError(f"unsupported Rust codegen operation {expr.op!r}")


def _emit_expr_ref(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
) -> str:
    """Emit a Rust expression reference for an already-available value."""
    if expr.op == "const":
        return _format_float(expr.value)
    if expr.op == "symbol":
        return scalar_bindings[expr.node]
    if expr.node in workspace_map:
        return f"work[{workspace_map[expr.node]}]"
    return _emit_node_expr(expr, scalar_bindings, workspace_map)


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


def _render_cargo_toml(crate_name: str) -> str:
    """Render a minimal Cargo manifest for generated code."""
    return "\n".join(
        [
            "[package]",
            f'name = "{crate_name}"',
            'version = "0.1.0"',
            'edition = "2021"',
            "",
            "[lib]",
            'path = "src/lib.rs"',
            "",
        ]
    )


def _render_project_readme(crate_name: str, codegen: RustCodegenResult) -> str:
    """Render a simple README for a generated Rust project."""
    return "\n".join(
        [
            f"# {crate_name}",
            "",
            "This Rust crate was generated by `gradgen`.",
            "",
            "## Build",
            "",
            "```bash",
            "cargo build",
            "```",
            "",
            "## Test Integration",
            "",
            "Add this crate to your Rust project or call the generated function directly from `src/lib.rs`.",
            "",
            "## Generated Function",
            "",
            f"- Function name: `{codegen.function_name}`",
            f"- Workspace size: `{codegen.workspace_size}`",
            f"- Input sizes: `{codegen.input_sizes}`",
            f"- Output sizes: `{codegen.output_sizes}`",
            "",
            "The generated ABI uses input slices, output slices, and a mutable workspace slice.",
            "The generated Rust file also includes helper functions that return the required",
            "workspace, input, and output dimensions.",
            "",
        ]
    )


def _render_size_functions(
    *,
    base_name: str,
    role: str,
    names: tuple[str, ...],
    sizes: tuple[int, ...],
    prefix: str,
) -> list[str]:
    """Render Rust constants and helper functions for slice dimensions."""
    lines: list[str] = []
    plural = "inputs" if role == "input" else "outputs"
    lines.append(f"/// Return the number of declared {plural} of [`{base_name}`].")
    lines.append(f"pub fn {base_name}_num_{plural}() -> usize {{")
    lines.append(f"    {len(sizes)}")
    lines.append("}")
    lines.append("")

    for index, (raw_name, size) in enumerate(zip(names, sizes)):
        ident = _sanitize_ident(raw_name)
        const_name = f"{prefix}_{role.upper()}_{index}_SIZE"
        lines.append(f"/// Length of the `{ident}` {role} slice for [`{base_name}`].")
        lines.append(f"pub const {const_name}: usize = {size};")
        lines.append("")
        lines.append(
            f"/// Return the length of the `{ident}` {role} slice for [`{base_name}`]."
        )
        lines.append(f"pub fn {base_name}_{role}_{index}_size() -> usize {{")
        lines.append(f"    {const_name}")
        lines.append("}")
        lines.append("")

    return lines
