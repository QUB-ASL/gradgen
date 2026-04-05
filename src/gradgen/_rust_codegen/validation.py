"""Validation helpers for Rust code generation."""

from __future__ import annotations

from .naming import validate_rust_ident, validate_unique_rust_names


def validate_backend_mode(backend_mode: str) -> None:
    """Validate a Rust backend mode string."""
    if backend_mode not in {"std", "no_std"}:
        raise ValueError(f"unsupported Rust backend mode {backend_mode!r}")


def validate_scalar_type(scalar_type: str) -> None:
    """Validate a generated Rust scalar type."""
    if scalar_type not in {"f64", "f32"}:
        raise ValueError(f"unsupported Rust scalar type {scalar_type!r}")


def validate_crate_name(crate_name: str | None) -> None:
    """Validate an explicitly configured crate name."""
    validate_rust_ident(crate_name, label="crate_name")


def validate_generated_argument_names(
    input_specs,
    output_specs,
) -> None:
    """Validate Rust-facing argument names after sanitization."""
    validate_unique_rust_names(
        [(spec.raw_name, spec.rust_name)
         for spec in (*input_specs, *output_specs)],
        label="generated argument",
    )
    validate_unique_rust_names(
        [("work", "work"), *[(spec.raw_name, spec.rust_name)
                             for spec in (*input_specs, *output_specs)]],
        label="generated argument",
    )


def resolve_backend_config(
    config,
    *,
    crate_name: str | None = None,
    function_name: str | None = None,
    backend_mode: str = "std",
    scalar_type: str = "f64",
    math_library: str | None = None,
):
    """Merge explicit keyword arguments with an optional backend config."""
    from .config import RustBackendConfig

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
