"""Shared helpers for staged symbolic wrapper objects."""

from __future__ import annotations

from typing import Any


def _simplify_function(function: Any, simplification: int | str | None):
    """Return a simplified function when an effort is configured."""
    if simplification is None:
        return function
    return function.simplify(max_effort=simplification, name=function.name)


def _generate_rust(
    target: Any,
    *,
    config=None,
    function_name: str | None = None,
    backend_mode: str = "std",
    scalar_type: str = "f64",
):
    """Generate Rust for a staged wrapper via the codegen dispatcher."""
    from ._rust_codegen.codegen import generate_rust

    return generate_rust(
        target,
        config=config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
    )


def _create_rust_project(
    target: Any,
    path: str,
    *,
    config=None,
    crate_name: str | None = None,
    function_name: str | None = None,
    backend_mode: str = "std",
    scalar_type: str = "f64",
):
    """Create a Rust project for a staged wrapper via the project helper."""
    from ._rust_codegen.project import create_rust_project

    return create_rust_project(
        target,
        path,
        config=config,
        crate_name=crate_name,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
    )
