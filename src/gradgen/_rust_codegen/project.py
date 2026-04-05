"""Filesystem and build helpers for generated Rust projects."""

from __future__ import annotations

import logging
from pathlib import Path

from ..function import Function
from ..sx import SX
from .config import RustBackendConfig
from .models import (
    RustDerivativeBundleResult,
    RustProjectResult,
)
from .naming import sanitize_ident
from .templates import _get_template
from .validation import (
    resolve_backend_config,
    validate_backend_mode,
    validate_scalar_type,
)
from .project_support import (
    _create_python_interface_project,
    _maybe_simplify_derivative_function,
    _render_metadata_json,
    _run_cargo_build,
    _try_run_cargo_fmt,
)


_LOGGER = logging.getLogger(__name__)


def create_rust_project(
    function: Function,
    path: str | Path,
    *,
    config: RustBackendConfig | None = None,
    crate_name: str | None = None,
    function_name: str | None = None,
    backend_mode: str = "std",
    scalar_type: str = "f64",
    math_library: str | None = None,
) -> RustProjectResult:
    """Create a minimal Rust library project containing generated code."""
    from .codegen import generate_rust

    resolved_config = resolve_backend_config(
        config,
        crate_name=crate_name,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    validate_backend_mode(resolved_config.backend_mode)
    validate_scalar_type(resolved_config.scalar_type)

    project_dir = Path(path).expanduser().resolve()
    crate = sanitize_ident(resolved_config.crate_name or function.name)
    resolved_math_library = "libm" \
        if resolved_config.backend_mode == "no_std" \
        else None
    resolved_math_library_version = "0.2" \
        if resolved_math_library == "libm" \
        else None
    codegen = generate_rust(
        function,
        config=resolved_config,
    )

    src_dir = project_dir / "src"
    cargo_toml = project_dir / "Cargo.toml"
    readme = project_dir / "README.md"
    metadata_json = project_dir / "metadata.json"
    lib_rs = src_dir / "lib.rs"

    src_dir.mkdir(parents=True, exist_ok=True)
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
        _get_template("rust_project_README.md.j2").render(
            crate_name=crate,
            codegen=codegen,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=resolved_math_library,
            enable_python_interface=resolved_config.enable_python_interface,
            python_interface_project_name=(
                f"{crate}_python"
                if resolved_config.enable_python_interface
                else None
            ),
        ),
        encoding="utf-8",
    )
    metadata_json.write_text(
        _render_metadata_json(crate, (codegen,)),
        encoding="utf-8",
    )
    lib_rs.write_text(codegen.source, encoding="utf-8")
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
            codegens=(codegen,),
            build_python_interface=resolved_config.build_python_interface,
        )
    _try_run_cargo_fmt(project_dir)

    return RustProjectResult(
        project_dir=project_dir,
        cargo_toml=cargo_toml,
        readme=readme,
        metadata_json=metadata_json,
        lib_rs=lib_rs,
        codegen=codegen,
        python_interface=python_interface,
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
            derivative_function = _maybe_simplify_derivative_function(
                block, simplify_derivatives)
            jacobian_projects.append(
                create_rust_project(
                    derivative_function,
                    bundle_dir / derivative_function.name,
                    config=config,
                )
            )

    hessian_projects: list[RustProjectResult] = []
    if include_hessians \
            and len(function.outputs) == 1 \
            and isinstance(function.outputs[0], SX):
        for block in function.hessian_blocks():
            derivative_function = _maybe_simplify_derivative_function(
                block, simplify_derivatives)
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
