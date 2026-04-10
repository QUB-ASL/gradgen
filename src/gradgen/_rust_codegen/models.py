"""Dataclasses describing generated Rust artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..sx import SXNode
from .config import RustBackendMode, RustScalarType


@dataclass(frozen=True, slots=True)
class RustCodegenResult:
    """Generated Rust source and metadata for a symbolic function."""

    source: str
    python_name: str
    function_name: str
    workspace_size: int
    input_names: tuple[str, ...]
    input_sizes: tuple[int, ...]
    output_names: tuple[str, ...]
    output_sizes: tuple[int, ...]
    backend_mode: RustBackendMode
    scalar_type: RustScalarType
    math_library: str | None


@dataclass(frozen=True, slots=True)
class RustPythonInterfaceProjectResult:
    """Information about a generated Python wrapper crate."""

    project_dir: Path
    cargo_toml: Path
    pyproject: Path
    readme: Path
    lib_rs: Path
    module_name: str
    low_level_crate_name: str


@dataclass(frozen=True, slots=True)
class RustProjectResult:
    """Information about a generated Rust project on disk."""

    project_dir: Path
    cargo_toml: Path
    readme: Path
    metadata_json: Path
    lib_rs: Path
    codegen: RustCodegenResult
    python_interface: RustPythonInterfaceProjectResult | None = None


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
    metadata_json: Path
    lib_rs: Path
    codegens: tuple[RustCodegenResult, ...]
    python_interface: RustPythonInterfaceProjectResult | None = None


@dataclass(frozen=True,
           slots=True)
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
    parameter_offsets: tuple[int, ...]
    fixed_values: tuple[tuple[float, ...], ...]
    repeat_count: int
    stage_start_index: int
    const_name: str


@dataclass(frozen=True, slots=True)
class _SingleShootingHelperBundle:
    """Generated helper kernels for a single-shooting problem."""

    dynamics_name: str
    dynamics_jvp_name: str | None
    stage_cost_name: str
    terminal_cost_name: str
    dynamics_vjp_x_name: str | None
    dynamics_vjp_u_name: str | None
    dynamics_vjp_x_jvp_name: str | None
    dynamics_vjp_u_jvp_name: str | None
    stage_cost_grad_x_name: str | None
    stage_cost_grad_u_name: str | None
    stage_cost_grad_x_jvp_name: str | None
    stage_cost_grad_u_jvp_name: str | None
    terminal_cost_grad_x_name: str | None
    terminal_cost_grad_x_jvp_name: str | None
    sources: tuple[str, ...]
    helper_nodes: tuple[SXNode, ...]
    max_workspace_size: int
