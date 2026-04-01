"""Callback invocation, coercion, and evaluation helpers for custom primitives."""

from __future__ import annotations

import inspect
from typing import Callable

from ..sx import SX, SXVector
from .model import RegisteredElementaryFunction


def build_custom_jacobian_expr(
    spec: RegisteredElementaryFunction,
    value: SX | SXVector,
    params: tuple[SX, ...],
) -> SX | SXVector:
    """Build the symbolic Jacobian or gradient expression for a custom function."""
    expr = invoke_custom_callback(spec.jacobian, value, SXVector(params), spec.parameter_dimension)
    if spec.is_scalar:
        return coerce_symbolic_scalar(
            expr,
            "scalar custom Jacobian builders must return a scalar-like value",
        )
    return coerce_symbolic_vector(
        expr,
        spec.vector_dim or 0,
        "vector custom Jacobian builders must return a vector-like value with matching length",
    )


def build_custom_hessian_expr(
    spec: RegisteredElementaryFunction,
    value: SX | SXVector,
    params: tuple[SX, ...],
) -> SX | tuple[SXVector, ...]:
    """Build the symbolic Hessian expression for a custom function."""
    expr = invoke_custom_callback(spec.hessian, value, SXVector(params), spec.parameter_dimension)
    if spec.is_scalar:
        return coerce_symbolic_scalar(
            expr,
            "scalar custom Hessian builders must return a scalar-like value",
        )
    return coerce_symbolic_matrix(
        expr,
        spec.vector_dim or 0,
        "vector custom Hessian builders must return a matrix-like value with matching shape",
    )


def build_custom_hvp_expr(
    spec: RegisteredElementaryFunction,
    value: SX | SXVector,
    tangent: SX | SXVector,
    params: tuple[SX, ...],
) -> SX | SXVector:
    """Build the symbolic Hessian-vector-product expression for a custom function."""
    if spec.hvp is None:
        return _build_custom_hvp_from_hessian(spec, value, tangent, params)

    expr = invoke_custom_hvp_callback(spec.hvp, value, tangent, SXVector(params), spec.parameter_dimension)
    if spec.is_scalar:
        return coerce_symbolic_scalar(
            expr,
            "scalar custom HVP builders must return a scalar-like value",
        )
    return coerce_symbolic_vector(
        expr,
        spec.vector_dim or 0,
        "vector custom HVP builders must return a vector-like value with matching length",
    )


def evaluate_custom_jacobian(
    spec: RegisteredElementaryFunction,
    value: float | tuple[float, ...],
    params: tuple[float, ...],
) -> float | tuple[float, ...]:
    """Evaluate the custom Jacobian callback numerically."""
    expr = invoke_custom_callback(spec.jacobian, value, params, spec.parameter_dimension)
    if spec.is_scalar:
        return coerce_numeric_scalar(
            expr,
            "scalar custom Jacobian builders must return a numeric scalar",
        )
    return coerce_numeric_vector(
        expr,
        spec.vector_dim or 0,
        "vector custom Jacobian builders must return a numeric vector-like value with matching length",
    )


def evaluate_custom_hessian(
    spec: RegisteredElementaryFunction,
    value: float | tuple[float, ...],
    params: tuple[float, ...],
) -> float | tuple[tuple[float, ...], ...]:
    """Evaluate the custom Hessian callback numerically."""
    expr = invoke_custom_callback(spec.hessian, value, params, spec.parameter_dimension)
    if spec.is_scalar:
        return coerce_numeric_scalar(
            expr,
            "scalar custom Hessian builders must return a numeric scalar",
        )
    return coerce_numeric_matrix(
        expr,
        spec.vector_dim or 0,
        "vector custom Hessian builders must return a numeric matrix-like value with matching shape",
    )


def evaluate_custom_hvp(
    spec: RegisteredElementaryFunction,
    value: float | tuple[float, ...],
    tangent: float | tuple[float, ...],
    params: tuple[float, ...],
) -> float | tuple[float, ...]:
    """Evaluate the custom Hessian-vector-product callback numerically."""
    if spec.hvp is not None:
        expr = invoke_custom_hvp_callback(spec.hvp, value, tangent, params, spec.parameter_dimension)
        if spec.is_scalar:
            return coerce_numeric_scalar(
                expr,
                "scalar custom HVP builders must return a numeric scalar",
            )
        return coerce_numeric_vector(
            expr,
            spec.vector_dim or 0,
            "vector custom HVP builders must return a numeric vector-like value with matching length",
        )

    hessian = evaluate_custom_hessian(spec, value, params)
    if spec.is_scalar:
        if not isinstance(tangent, (int, float)):
            raise TypeError("scalar custom HVP tangents must be numeric scalars")
        return coerce_numeric_scalar(
            hessian,
            "scalar custom Hessian builders must return a numeric scalar",
        ) * float(tangent)

    if not isinstance(tangent, tuple):
        raise TypeError("vector custom HVP tangents must be numeric vectors")
    matrix = coerce_numeric_matrix(
        hessian,
        spec.vector_dim or 0,
        "vector custom Hessian builders must return a numeric matrix-like value with matching shape",
    )
    return tuple(
        sum(row[column] * tangent[column] for column in range(len(tangent)))
        for row in matrix
    )


def validate_registered_function(spec: RegisteredElementaryFunction) -> None:
    """Validate a registration by numerically probing its callbacks."""
    if spec.is_scalar:
        x = 0.5
        tangent = 1.25
        params = tuple(spec.parameter_defaults)
        _ = evaluate_custom_jacobian(spec, x, params)
        _ = evaluate_custom_hessian(spec, x, params)
        if spec.hvp is not None:
            _ = evaluate_custom_hvp(spec, x, tangent, params)
        return

    dim = spec.vector_dim or 0
    x = tuple(float(index + 1) for index in range(dim))
    tangent = tuple(float(index + 2) for index in range(dim))
    params = tuple(spec.parameter_defaults)
    if spec.jacobian is not None:
        _ = evaluate_custom_jacobian(spec, x, params)
    if spec.hessian is not None:
        _ = evaluate_custom_hessian(spec, x, params)
    if spec.hvp is not None:
        _ = evaluate_custom_hvp(spec, x, tangent, params)


def invoke_custom_callback(
    callback: Callable[..., object],
    value: object,
    w: object,
    parameter_dimension: int,
) -> object:
    """Invoke a user callback, supporting optional omitted parameter vectors."""
    if parameter_dimension == 0:
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(value, w)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) <= 1:
            return callback(value)
    return callback(value, w)


def invoke_custom_hvp_callback(
    callback: Callable[..., object],
    value: object,
    tangent: object,
    w: object,
    parameter_dimension: int,
) -> object:
    """Invoke a custom HVP callback, handling supported argument orders."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(value, tangent, w)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) <= 2 and parameter_dimension == 0:
        return callback(value, tangent)
    if len(positional) >= 3:
        third_name = positional[2].name.lower()
        if "tangent" in third_name or third_name.startswith("v"):
            return callback(value, w, tangent)
    return callback(value, tangent, w)


def coerce_symbolic_scalar(value: object, error_message: str) -> SX:
    """Coerce ``value`` to a symbolic scalar."""
    if isinstance(value, SX):
        return value
    if isinstance(value, (int, float)):
        return SX.const(value)
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, (SX, int, float)):
            return coerce_symbolic_scalar(scalar, error_message)
    raise TypeError(error_message)


def coerce_symbolic_vector(value: object, size: int, error_message: str) -> SXVector:
    """Coerce ``value`` to a symbolic vector of length ``size``."""
    if isinstance(value, SXVector):
        if len(value) != size:
            raise TypeError(error_message)
        return value
    if isinstance(value, (str, bytes)):
        raise TypeError(error_message)
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(error_message) from exc
    if len(items) != size:
        raise TypeError(error_message)
    return SXVector(tuple(coerce_symbolic_scalar(item, error_message) for item in items))


def coerce_symbolic_matrix(
    value: object,
    size: int,
    error_message: str,
) -> tuple[SXVector, ...]:
    """Coerce ``value`` to a symbolic square matrix with side length ``size``."""
    if isinstance(value, (str, bytes)):
        raise TypeError(error_message)
    try:
        rows = list(value)
    except TypeError as exc:
        raise TypeError(error_message) from exc
    if len(rows) != size:
        raise TypeError(error_message)
    return tuple(coerce_symbolic_vector(row, size, error_message) for row in rows)


def coerce_numeric_scalar(value: object, error_message: str) -> float:
    """Coerce ``value`` to a numeric scalar."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, SX):
        if value.op == "const" and value.value is not None:
            return float(value.value)
        raise TypeError(error_message)
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, (int, float, SX)):
            return coerce_numeric_scalar(scalar, error_message)
    raise TypeError(error_message)


def coerce_numeric_vector(
    value: object,
    size: int,
    error_message: str,
) -> tuple[float, ...]:
    """Coerce ``value`` to a numeric vector of length ``size``."""
    if isinstance(value, SXVector):
        if len(value) != size:
            raise TypeError(error_message)
        return tuple(coerce_numeric_scalar(item, error_message) for item in value)
    if isinstance(value, (str, bytes)):
        raise TypeError(error_message)
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(error_message) from exc
    if len(items) != size:
        raise TypeError(error_message)
    return tuple(coerce_numeric_scalar(item, error_message) for item in items)


def coerce_numeric_matrix(
    value: object,
    size: int,
    error_message: str,
) -> tuple[tuple[float, ...], ...]:
    """Coerce ``value`` to a numeric square matrix with side length ``size``."""
    if isinstance(value, (str, bytes)):
        raise TypeError(error_message)
    try:
        rows = list(value)
    except TypeError as exc:
        raise TypeError(error_message) from exc
    if len(rows) != size:
        raise TypeError(error_message)
    return tuple(coerce_numeric_vector(row, size, error_message) for row in rows)


def _build_custom_hvp_from_hessian(
    spec: RegisteredElementaryFunction,
    value: SX | SXVector,
    tangent: SX | SXVector,
    params: tuple[SX, ...],
) -> SX | SXVector:
    """Fallback symbolic HVP by multiplying the Hessian with the tangent."""
    hessian_expr = build_custom_hessian_expr(spec, value, params)
    if spec.is_scalar:
        if not isinstance(tangent, SX):
            raise TypeError("scalar custom HVP tangents must be scalar")
        return hessian_expr * tangent
    if not isinstance(hessian_expr, tuple) or not isinstance(tangent, SXVector):
        raise TypeError("vector custom HVP fallback requires a vector Hessian and vector tangent")
    return SXVector(
        tuple(
            sum((row[column] * tangent[column] for column in range(len(tangent))), SX.const(0.0))
            for row in hessian_expr
        )
    )
