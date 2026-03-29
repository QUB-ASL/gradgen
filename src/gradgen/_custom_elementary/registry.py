"""Registry and parsing helpers for user-defined elementary functions."""

from __future__ import annotations

from typing import Sequence

from ..sx import SX, SXVector
from .callbacks import validate_registered_function
from .model import (
    PythonEvalBuilder,
    RegisteredElementaryFunction,
    ScalarHessianBuilder,
    ScalarHvpBuilder,
    ScalarJacobianBuilder,
    VectorHessianBuilder,
    VectorHvpBuilder,
    VectorJacobianBuilder,
)


_REGISTRY: dict[str, RegisteredElementaryFunction] = {}


def register_elementary_function(
    *,
    name: str,
    input_dimension: int,
    parameter_dimension: int = 0,
    parameter_defaults: Sequence[float | int] | None = None,
    eval_python: PythonEvalBuilder | None = None,
    jacobian: ScalarJacobianBuilder | VectorJacobianBuilder,
    hessian: ScalarHessianBuilder | VectorHessianBuilder,
    hvp: ScalarHvpBuilder | VectorHvpBuilder | None = None,
    rust_primal: str | None = None,
    rust_jacobian: str | None = None,
    rust_hvp: str | None = None,
    rust_hessian: str | None = None,
) -> RegisteredElementaryFunction:
    """Register a user-defined elementary function."""
    if not name or not name.isidentifier():
        raise ValueError("custom elementary function names must be valid identifiers")
    if name in _REGISTRY:
        raise ValueError(f"custom elementary function {name!r} is already registered")

    normalized_parameter_dimension = _normalize_parameter_dimension(parameter_dimension)
    spec = RegisteredElementaryFunction(
        name=name,
        input_dimension=_normalize_input_dimension(input_dimension),
        parameter_dimension=normalized_parameter_dimension,
        parameter_defaults=_normalize_parameter_defaults(
            parameter_dimension=normalized_parameter_dimension,
            parameter_defaults=parameter_defaults,
        ),
        eval_python=eval_python,
        jacobian=jacobian,
        hessian=hessian,
        hvp=hvp,
        rust_primal=rust_primal,
        rust_jacobian=rust_jacobian,
        rust_hvp=rust_hvp,
        rust_hessian=rust_hessian,
    )
    validate_registered_function(spec)
    _REGISTRY[name] = spec
    return spec


def get_registered_elementary_function(name: str) -> RegisteredElementaryFunction:
    """Return a registered elementary function by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown custom elementary function {name!r}") from exc


def clear_registered_elementary_functions() -> None:
    """Clear the custom elementary-function registry."""
    _REGISTRY.clear()


def render_custom_rust_snippet(
    snippet: str,
    *,
    scalar_type: str,
    math_library: str | None,
) -> tuple[str, ...]:
    """Render a user-provided Rust snippet with simple placeholders."""
    rendered = snippet.replace("{{ scalar_type }}", scalar_type)
    rendered = rendered.replace("{{ math_library }}", math_library or "")
    return tuple(line.rstrip() for line in rendered.strip().splitlines())


def parse_custom_scalar_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, SX, tuple[SX, ...]]:
    """Parse symbolic arguments for a scalar-input custom primitive."""
    spec = get_registered_elementary_function(_require_name(name))
    if not spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not scalar-input")
    return spec, args[0], args[1:]


def parse_custom_scalar_hvp_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, SX, SX, tuple[SX, ...]]:
    """Parse symbolic arguments for a scalar-input custom HVP node."""
    spec = get_registered_elementary_function(_require_name(name))
    if not spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not scalar-input")
    return spec, args[0], args[1], args[2:]


def parse_custom_vector_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, SXVector, tuple[SX, ...]]:
    """Parse symbolic arguments for a vector-input custom primitive."""
    spec = get_registered_elementary_function(_require_name(name))
    if spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not vector-input")
    dim = spec.vector_dim or 0
    return spec, SXVector(args[:dim]), args[dim:]


def parse_custom_vector_jacobian_component_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, int, SXVector, tuple[SX, ...]]:
    """Parse symbolic arguments for a vector custom Jacobian component."""
    spec = get_registered_elementary_function(_require_name(name))
    if spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not vector-input")
    dim = spec.vector_dim or 0
    index = _require_integral_const(args[0], "index")
    return spec, index, SXVector(args[1 : 1 + dim]), args[1 + dim :]


def parse_custom_vector_hvp_component_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, int, SXVector, SXVector, tuple[SX, ...]]:
    """Parse symbolic arguments for a vector custom HVP component."""
    spec = get_registered_elementary_function(_require_name(name))
    if spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not vector-input")
    dim = spec.vector_dim or 0
    index = _require_integral_const(args[0], "index")
    return (
        spec,
        index,
        SXVector(args[1 : 1 + dim]),
        SXVector(args[1 + dim : 1 + (2 * dim)]),
        args[1 + (2 * dim) :],
    )


def parse_custom_vector_hessian_entry_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, int, int, SXVector, tuple[SX, ...]]:
    """Parse symbolic arguments for a vector custom Hessian entry."""
    spec = get_registered_elementary_function(_require_name(name))
    if spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not vector-input")
    dim = spec.vector_dim or 0
    row = _require_integral_const(args[0], "row")
    col = _require_integral_const(args[1], "col")
    return spec, row, col, SXVector(args[2 : 2 + dim]), args[2 + dim :]


def _normalize_input_dimension(input_dimension: int) -> int:
    if not isinstance(input_dimension, int) or input_dimension <= 0:
        raise ValueError("input_dimension must be a positive integer")
    return input_dimension


def _normalize_parameter_dimension(parameter_dimension: int) -> int:
    if not isinstance(parameter_dimension, int) or parameter_dimension < 0:
        raise ValueError("parameter_dimension must be a non-negative integer")
    return parameter_dimension


def _normalize_parameter_defaults(
    *,
    parameter_dimension: int,
    parameter_defaults: Sequence[float | int] | None,
) -> tuple[float, ...]:
    if parameter_defaults is None:
        return tuple(0.0 for _ in range(parameter_dimension))
    if len(parameter_defaults) != parameter_dimension:
        raise ValueError("parameter_defaults must match parameter_dimension")
    return tuple(float(value) for value in parameter_defaults)


def _require_name(name: str | None) -> str:
    if name is None:
        raise ValueError("custom elementary nodes must carry a registered name")
    return name


def _require_integral_const(expr: SX, label: str) -> int:
    if expr.op != "const" or expr.value is None or expr.value != int(expr.value):
        raise ValueError(f"{label} must be an integer constant")
    return int(expr.value)
