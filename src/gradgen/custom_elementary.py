"""Registry for user-defined elementary functions."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable, Sequence

from .sx import SX, SXNode, SXVector


ScalarJacobianBuilder = Callable[..., object]
ScalarHessianBuilder = Callable[..., object]
ScalarHvpBuilder = Callable[..., object]
VectorJacobianBuilder = Callable[..., object]
VectorHessianBuilder = Callable[..., object]
VectorHvpBuilder = Callable[..., object]
PythonEvalBuilder = Callable[..., float]


@dataclass(frozen=True, slots=True)
class RegisteredElementaryFunction:
    """Registered user-defined elementary function metadata."""

    name: str
    input_dimension: int
    parameter_dimension: int
    parameter_defaults: tuple[float, ...]
    eval_python: PythonEvalBuilder | None
    jacobian: ScalarJacobianBuilder | VectorJacobianBuilder
    hessian: ScalarHessianBuilder | VectorHessianBuilder
    hvp: ScalarHvpBuilder | VectorHvpBuilder | None
    rust_primal: str | None
    rust_jacobian: str | None
    rust_hvp: str | None
    rust_hessian: str | None

    @property
    def is_scalar(self) -> bool:
        return self.input_dimension == 1

    @property
    def vector_dim(self) -> int | None:
        if self.is_scalar:
            return None
        return self.input_dimension

    def resolve_parameters(self, w: Sequence[float | int] | None) -> tuple[SX, ...]:
        """Return parameter-vector values in registration order."""
        if w is None:
            resolved_values = self.parameter_defaults
        else:
            if len(w) != self.parameter_dimension:
                raise ValueError(
                    f"custom function {self.name!r} expects parameter vector length "
                    f"{self.parameter_dimension}, received {len(w)}"
                )
            resolved_values = tuple(float(value) for value in w)

        return tuple(_coerce_parameter_value(value) for value in resolved_values)

    def __call__(self, value: SX | SXVector, *, w: Sequence[float | int] | None = None) -> SX:
        """Build a symbolic call to the registered elementary function."""
        parameter_values = self.resolve_parameters(w)

        if self.is_scalar:
            if not isinstance(value, SX):
                raise TypeError(f"custom scalar function {self.name!r} expects an SX input")
            return SX(
                SXNode.make(
                    "custom_scalar",
                    (value.node, *(parameter.node for parameter in parameter_values)),
                    name=self.name,
                )
            )

        if not isinstance(value, SXVector):
            raise TypeError(f"custom vector function {self.name!r} expects an SXVector input")
        if len(value) != self.vector_dim:
            raise ValueError(
                f"custom vector function {self.name!r} expects length {self.vector_dim}, received {len(value)}"
            )
        return SX(
            SXNode.make(
                "custom_vector",
                (
                    *(element.node for element in value),
                    *(parameter.node for parameter in parameter_values),
                ),
                name=self.name,
            )
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
    normalized_parameter_defaults = _normalize_parameter_defaults(
        parameter_dimension=normalized_parameter_dimension,
        parameter_defaults=parameter_defaults,
    )
    spec = RegisteredElementaryFunction(
        name=name,
        input_dimension=_normalize_input_dimension(input_dimension),
        parameter_dimension=normalized_parameter_dimension,
        parameter_defaults=normalized_parameter_defaults,
        eval_python=eval_python,
        jacobian=jacobian,
        hessian=hessian,
        hvp=hvp,
        rust_primal=rust_primal,
        rust_jacobian=rust_jacobian,
        rust_hvp=rust_hvp,
        rust_hessian=rust_hessian,
    )
    _validate_registered_function(spec)
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


def render_custom_rust_snippet(snippet: str, *, scalar_type: str, math_library: str | None) -> tuple[str, ...]:
    """Render a user-provided Rust snippet with simple placeholders."""
    rendered = snippet.replace("{{ scalar_type }}", scalar_type)
    rendered = rendered.replace("{{ math_library }}", math_library or "")
    return tuple(line.rstrip() for line in rendered.strip().splitlines())


def custom_scalar_jacobian(name: str, x: SX, params: tuple[SX, ...]) -> SX:
    return SX(SXNode.make("custom_scalar_jacobian", (x.node, *(param.node for param in params)), name=name))


def custom_scalar_hvp(name: str, x: SX, tangent: SX, params: tuple[SX, ...]) -> SX:
    return SX(
        SXNode.make(
            "custom_scalar_hvp",
            (x.node, tangent.node, *(param.node for param in params)),
            name=name,
        )
    )


def custom_scalar_hessian(name: str, x: SX, params: tuple[SX, ...]) -> SX:
    return SX(SXNode.make("custom_scalar_hessian", (x.node, *(param.node for param in params)), name=name))


def custom_vector_jacobian_component(
    name: str,
    index: int,
    x: SXVector,
    params: tuple[SX, ...],
) -> SX:
    return SX(
        SXNode.make(
            "custom_vector_jacobian_component",
            (SX.const(index).node, *(element.node for element in x), *(param.node for param in params)),
            name=name,
        )
    )


def custom_vector_hvp_component(
    name: str,
    index: int,
    x: SXVector,
    tangent: SXVector,
    params: tuple[SX, ...],
) -> SX:
    return SX(
        SXNode.make(
            "custom_vector_hvp_component",
            (
                SX.const(index).node,
                *(element.node for element in x),
                *(element.node for element in tangent),
                *(param.node for param in params),
            ),
            name=name,
        )
    )


def custom_vector_hessian_entry(
    name: str,
    row: int,
    col: int,
    x: SXVector,
    params: tuple[SX, ...],
) -> SX:
    return SX(
        SXNode.make(
            "custom_vector_hessian_entry",
            (
                SX.const(row).node,
                SX.const(col).node,
                *(element.node for element in x),
                *(param.node for param in params),
            ),
            name=name,
        )
    )


def parse_custom_scalar_args(name: str | None, args: tuple[SX, ...]) -> tuple[RegisteredElementaryFunction, SX, tuple[SX, ...]]:
    spec = get_registered_elementary_function(_require_name(name))
    if not spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not scalar-input")
    return spec, args[0], args[1:]


def parse_custom_scalar_hvp_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, SX, SX, tuple[SX, ...]]:
    spec = get_registered_elementary_function(_require_name(name))
    if not spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not scalar-input")
    return spec, args[0], args[1], args[2:]


def parse_custom_vector_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, SXVector, tuple[SX, ...]]:
    spec = get_registered_elementary_function(_require_name(name))
    if spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not vector-input")
    dim = spec.vector_dim or 0
    return spec, SXVector(args[:dim]), args[dim:]


def parse_custom_vector_jacobian_component_args(
    name: str | None,
    args: tuple[SX, ...],
) -> tuple[RegisteredElementaryFunction, int, SXVector, tuple[SX, ...]]:
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
    spec = get_registered_elementary_function(_require_name(name))
    if spec.is_scalar:
        raise ValueError(f"custom function {spec.name!r} is not vector-input")
    dim = spec.vector_dim or 0
    row = _require_integral_const(args[0], "row")
    col = _require_integral_const(args[1], "col")
    return spec, row, col, SXVector(args[2 : 2 + dim]), args[2 + dim :]


def build_custom_jacobian_expr(spec: RegisteredElementaryFunction, value: SX | SXVector, params: tuple[SX, ...]) -> SX | SXVector:
    """Build the symbolic Jacobian/gradient expression for a custom function."""
    expr = _invoke_custom_callback(spec.jacobian, value, SXVector(params), spec.parameter_dimension)
    if spec.is_scalar:
        return _coerce_symbolic_scalar(expr, "scalar custom Jacobian builders must return a scalar-like value")
    return _coerce_symbolic_vector(
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
    expr = _invoke_custom_callback(spec.hessian, value, SXVector(params), spec.parameter_dimension)
    if spec.is_scalar:
        return _coerce_symbolic_scalar(expr, "scalar custom Hessian builders must return a scalar-like value")
    return _coerce_symbolic_matrix(
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
    """Build the symbolic Hessian-vector product expression for a custom function."""
    if spec.hvp is None:
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

    expr = _invoke_custom_hvp_callback(spec.hvp, value, tangent, SXVector(params), spec.parameter_dimension)
    if spec.is_scalar:
        return _coerce_symbolic_scalar(expr, "scalar custom HVP builders must return a scalar-like value")
    return _coerce_symbolic_vector(
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
    expr = _invoke_custom_callback(spec.jacobian, value, params, spec.parameter_dimension)
    if spec.is_scalar:
        return _coerce_numeric_scalar(expr, "scalar custom Jacobian builders must return a numeric scalar")
    return _coerce_numeric_vector(
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
    expr = _invoke_custom_callback(spec.hessian, value, params, spec.parameter_dimension)
    if spec.is_scalar:
        return _coerce_numeric_scalar(expr, "scalar custom Hessian builders must return a numeric scalar")
    return _coerce_numeric_matrix(
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
    """Evaluate the custom Hessian-vector product callback numerically."""
    if spec.hvp is not None:
        expr = _invoke_custom_hvp_callback(spec.hvp, value, tangent, params, spec.parameter_dimension)
        if spec.is_scalar:
            return _coerce_numeric_scalar(expr, "scalar custom HVP builders must return a numeric scalar")
        return _coerce_numeric_vector(
            expr,
            spec.vector_dim or 0,
            "vector custom HVP builders must return a numeric vector-like value with matching length",
        )

    hessian = evaluate_custom_hessian(spec, value, params)
    if spec.is_scalar:
        if not isinstance(tangent, (int, float)):
            raise TypeError("scalar custom HVP tangents must be numeric scalars")
        return _coerce_numeric_scalar(hessian, "scalar custom Hessian builders must return a numeric scalar") * float(tangent)

    if not isinstance(tangent, tuple):
        raise TypeError("vector custom HVP tangents must be numeric vectors")
    matrix = _coerce_numeric_matrix(
        hessian,
        spec.vector_dim or 0,
        "vector custom Hessian builders must return a numeric matrix-like value with matching shape",
    )
    return tuple(
        sum(row[column] * tangent[column] for column in range(len(tangent)))
        for row in matrix
    )


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
        raise ValueError(
            "parameter_defaults must match parameter_dimension"
        )
    return tuple(float(_coerce_parameter_value(value).value) for value in parameter_defaults)


def _validate_registered_function(spec: RegisteredElementaryFunction) -> None:
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
    _ = evaluate_custom_jacobian(spec, x, params)
    _ = evaluate_custom_hessian(spec, x, params)
    if spec.hvp is not None:
        _ = evaluate_custom_hvp(spec, x, tangent, params)


def _require_name(name: str | None) -> str:
    if name is None:
        raise ValueError("custom elementary nodes must carry a registered name")
    return name


def _require_integral_const(expr: SX, label: str) -> int:
    if expr.op != "const" or expr.value is None or expr.value != int(expr.value):
        raise ValueError(f"{label} must be an integer constant")
    return int(expr.value)


def _coerce_parameter_value(value: float | int) -> SX:
    if not isinstance(value, (int, float)):
        raise TypeError("custom function parameters must be numeric constants")
    return SX.const(value)


def _invoke_custom_callback(callback: Callable[..., object], value: object, w: object, parameter_dimension: int) -> object:
    if parameter_dimension == 0:
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(value, w)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) <= 1:
            return callback(value)
    return callback(value, w)


def _invoke_custom_hvp_callback(
    callback: Callable[..., object],
    value: object,
    tangent: object,
    w: object,
    parameter_dimension: int,
) -> object:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(value, tangent, w)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) <= 2 and parameter_dimension == 0:
        return callback(value, tangent)
    if len(positional) >= 3:
        third_name = positional[2].name.lower()
        if "tangent" in third_name or third_name.startswith("v"):
            return callback(value, w, tangent)
    return callback(value, tangent, w)


def _coerce_symbolic_scalar(value: object, error_message: str) -> SX:
    if isinstance(value, SX):
        return value
    if isinstance(value, (int, float)):
        return SX.const(value)
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, (SX, int, float)):
            return _coerce_symbolic_scalar(scalar, error_message)
    raise TypeError(error_message)


def _coerce_symbolic_vector(value: object, size: int, error_message: str) -> SXVector:
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
    return SXVector(tuple(_coerce_symbolic_scalar(item, error_message) for item in items))


def _coerce_symbolic_matrix(value: object, size: int, error_message: str) -> tuple[SXVector, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(error_message)
    try:
        rows = list(value)
    except TypeError as exc:
        raise TypeError(error_message) from exc
    if len(rows) != size:
        raise TypeError(error_message)
    return tuple(_coerce_symbolic_vector(row, size, error_message) for row in rows)


def _coerce_numeric_scalar(value: object, error_message: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, SX):
        if value.op == "const" and value.value is not None:
            return float(value.value)
        raise TypeError(error_message)
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, (int, float, SX)):
            return _coerce_numeric_scalar(scalar, error_message)
    raise TypeError(error_message)


def _coerce_numeric_vector(value: object, size: int, error_message: str) -> tuple[float, ...]:
    if isinstance(value, SXVector):
        if len(value) != size:
            raise TypeError(error_message)
        return tuple(_coerce_numeric_scalar(item, error_message) for item in value)
    if isinstance(value, (str, bytes)):
        raise TypeError(error_message)
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(error_message) from exc
    if len(items) != size:
        raise TypeError(error_message)
    return tuple(_coerce_numeric_scalar(item, error_message) for item in items)


def _coerce_numeric_matrix(value: object, size: int, error_message: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(error_message)
    try:
        rows = list(value)
    except TypeError as exc:
        raise TypeError(error_message) from exc
    if len(rows) != size:
        raise TypeError(error_message)
    return tuple(_coerce_numeric_vector(row, size, error_message) for row in rows)
