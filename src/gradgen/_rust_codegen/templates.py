"""Jinja2 template access helpers for Rust code generation."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


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


def _get_template(name: str):
    """Return a Jinja2 template from the package template directory."""
    return _template_environment().get_template(name)
