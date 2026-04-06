"""Small formatting and argument-shape helpers for Rust rendering."""

from __future__ import annotations

from ..config import RustScalarType
from ...sx import SX, SXVector
from ..validation import validate_scalar_type


def _flatten_arg(arg: SX | SXVector) -> tuple[SX, ...]:
    """Flatten a scalar or vector argument into scalar expressions."""
    if isinstance(arg, SX):
        return (arg,)
    return arg.elements


def _arg_size(arg: SX | SXVector) -> int:
    """Return the number of scalar elements in an argument."""
    return len(_flatten_arg(arg))


def _scaled_index_expr(base_expr: str, scale: int) -> str:
    """Return ``base_expr * scale`` while removing identity multipliers."""
    if scale == 1:
        return base_expr
    if base_expr.isidentifier():
        return f"{base_expr} * {scale}"
    return f"(({base_expr}) * {scale})"


def _format_float(value: float | None, scalar_type: RustScalarType) -> str:
    """Format a Python float as a Rust floating-point literal."""
    if value is None:
        raise ValueError("expected a concrete floating-point value")
    validate_scalar_type(scalar_type)
    return f"{repr(float(value))}_{scalar_type}"


def _format_rust_string_literal(value: str) -> str:
    """Format a Python string as a Rust string literal."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _describe_input_arg(raw_name: str) -> str:
    """Describe the semantic role of a generated Rust input slice."""
    if raw_name.startswith("cotangent_") and len(raw_name) > len("cotangent_"):
        base_name = raw_name[len("cotangent_") :]
        return (
            f"cotangent seed associated with declared result `{base_name}`; "
            "use this slice when forming Jacobian-transpose-vector or reverse-mode sensitivity terms"
        )
    if raw_name.startswith("v_") and len(raw_name) > 2:
        base_name = raw_name[2:]
        return (
            f"tangent or direction input associated with declared argument `{base_name}`; "
            "use this slice when forming Hessian-vector-product or directional-derivative terms"
        )
    return f"input slice for the declared argument `{raw_name}`"


def _describe_output_arg(raw_name: str) -> str:
    """Describe the semantic role of a generated Rust output slice."""
    if raw_name.startswith("vjp_") and len(raw_name) > len("vjp_"):
        base_name = raw_name[len("vjp_") :]
        return f"output slice receiving the vector-Jacobian product for declared input `{base_name}`"
    if raw_name.startswith("jacobian_") and len(raw_name) > len("jacobian_"):
        base_name = raw_name[len("jacobian_") :]
        return f"output slice receiving the Jacobian block for declared result `{base_name}`"
    if raw_name.startswith("gradient_") and len(raw_name) > len("gradient_"):
        base_name = raw_name[len("gradient_") :]
        return f"output slice receiving the gradient block for declared result `{base_name}`"
    if raw_name.startswith("hessian_") and len(raw_name) > len("hessian_"):
        base_name = raw_name[len("hessian_") :]
        return f"output slice receiving the Hessian block for declared result `{base_name}`"
    if raw_name.startswith("hvp_") and len(raw_name) > len("hvp_"):
        base_name = raw_name[len("hvp_") :]
        return f"output slice receiving the Hessian-vector product for declared result `{base_name}`"
    return f"primal output slice for the declared result `{raw_name}`"
