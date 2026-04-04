"""Template rendering helpers for generated Rust kernels."""

from __future__ import annotations

from dataclasses import dataclass

from ..templates import _get_template


@dataclass(frozen=True, slots=True)
class KernelRenderContext:
    """Common render-time metadata for a generated Rust kernel."""

    backend_mode: str
    scalar_type: str
    math_library: str | None
    emit_metadata_helpers: bool


def render_kernel_source(context: KernelRenderContext, /, **render_args) -> str:
    """Render a Rust kernel source file from the shared template."""
    render_args = dict(render_args)
    render_args.pop("backend_mode", None)
    render_args.pop("scalar_type", None)
    render_args.pop("math_library", None)
    render_args.pop("emit_metadata_helpers", None)
    return _get_template("lib.rs.j2").render(
        backend_mode=context.backend_mode,
        scalar_type=context.scalar_type,
        math_library=context.math_library,
        emit_metadata_helpers=context.emit_metadata_helpers,
        **render_args,
    )
