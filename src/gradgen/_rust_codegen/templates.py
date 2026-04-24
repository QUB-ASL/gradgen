"""Jinja2 template access helpers for Rust code generation."""

from __future__ import annotations

from pathlib import Path

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    select_autoescape,
)


def _template_environment() -> Environment:
    """Build the Jinja2 environment used for Rust project rendering."""
    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        # These templates generate Rust, TOML, and Markdown, not HTML.
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=False,
            default=False,
        ),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render_inline_template(template_source: str, /, **context: object) -> str:
    """Render a Jinja2 template string used inside generated artifacts."""
    environment = _template_environment()
    environment.undefined = StrictUndefined
    try:
        return environment.from_string(template_source).render(**context)
    except TemplateError as exc:
        raise ValueError(
            "failed to render the custom Rust header template"
        ) from exc


def _render_custom_rust_header(
    header: str | None,
    *,
    backend_mode: str,
    scalar_type: str,
    math_library: str | None,
    emit_metadata_helpers: bool,
) -> str | None:
    """Render a custom Rust header template using kernel render context."""
    if header is None:
        return None
    return _render_inline_template(
        header,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
        emit_metadata_helpers=emit_metadata_helpers,
    )


def _get_template(name: str):
    """Return a Jinja2 template from the package template directory."""
    return _template_environment().get_template(name)
