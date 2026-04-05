"""Shared naming helpers for Rust code generation."""

from __future__ import annotations

import re


RUST_KEYWORDS = frozenset(
    {
        "as",
        "break",
        "const",
        "continue",
        "crate",
        "else",
        "enum",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "match",
        "mod",
        "move",
        "mut",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "type",
        "unsafe",
        "use",
        "where",
        "while",
        "async",
        "await",
        "dyn",
        "abstract",
        "become",
        "box",
        "do",
        "final",
        "macro",
        "override",
        "priv",
        "try",
        "typeof",
        "unsized",
        "virtual",
        "yield",
    }
)


def is_rust_ident(name: str) -> bool:
    """Return whether ``name`` is a plain Rust identifier."""
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)) and name not in RUST_KEYWORDS


def validate_rust_ident(name: str | None, *, label: str) -> None:
    """Validate an explicit user-specified Rust identifier."""
    if name is None:
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"{label} must match the pattern [A-Za-z_][A-Za-z0-9_]*")
    if name in RUST_KEYWORDS:
        raise ValueError(f"{label} must not be a Rust keyword")


def validate_unique_rust_names(
    names: list[tuple[str, str]],
    *,
    label: str,
) -> None:
    """Validate that generated Rust names are unique after sanitization."""
    seen: dict[str, str] = {}
    for raw_name, rust_name in names:
        previous = seen.get(rust_name)
        if previous is not None:
            raise ValueError(
                f"{label} names {previous!r} and {raw_name!r} "
                "both map to the Rust identifier "
                f"{rust_name!r}"
            )
        seen[rust_name] = raw_name


def sanitize_ident(name: str) -> str:
    """Convert a user-facing name into a simple Rust identifier."""
    chars = [character if character.isalnum()
             or character == "_" else "_" for character in name]
    ident = "".join(chars)
    if not ident:
        ident = "value"
    if ident[0].isdigit():
        ident = f"_{ident}"
    if ident in RUST_KEYWORDS:
        ident = f"{ident}_"
    return ident
