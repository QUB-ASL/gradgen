"""Registry for user-defined elementary functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .sx import SX, SXNode, SXVector


ScalarJacobianBuilder = Callable[..., SX]
ScalarHessianBuilder = Callable[..., SX]
VectorJacobianBuilder = Callable[..., SXVector]
VectorHessianBuilder = Callable[..., tuple[SXVector, ...]]
PythonEvalBuilder = Callable[..., float]


@dataclass(frozen=True, slots=True)
class RegisteredElementaryFunction:
    """Registered user-defined elementary function metadata."""

    name: str
    input_dimension: int
    parameter_defaults: tuple[tuple[str, float], ...]
    eval_python: PythonEvalBuilder | None
    jacobian: ScalarJacobianBuilder | VectorJacobianBuilder
    hessian: ScalarHessianBuilder | VectorHessianBuilder
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

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.parameter_defaults)

    def resolve_parameters(self, supplied: dict[str, float | int]) -> tuple[SX, ...]:
        """Return parameter values in registration order."""
        unexpected = set(supplied) - set(self.parameter_names)
        if unexpected:
            raise ValueError(
                f"unexpected parameters for custom function {self.name!r}: {sorted(unexpected)!r}"
            )

        resolved: list[SX] = []
        for name, default in self.parameter_defaults:
            value = supplied.get(name, default)
            if not isinstance(value, (int, float)):
                raise TypeError("custom function parameters must be numeric constants")
            resolved.append(SX.const(value))
        return tuple(resolved)

    def __call__(self, value: SX | SXVector, **parameters: float | int) -> SX:
        """Build a symbolic call to the registered elementary function."""
        parameter_values = self.resolve_parameters(parameters)

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
    parameters: dict[str, float | int] | None = None,
    eval_python: PythonEvalBuilder | None = None,
    jacobian: ScalarJacobianBuilder | VectorJacobianBuilder,
    hessian: ScalarHessianBuilder | VectorHessianBuilder,
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

    parameter_defaults = _normalize_parameter_defaults(parameters)
    spec = RegisteredElementaryFunction(
        name=name,
        input_dimension=_normalize_input_dimension(input_dimension),
        parameter_defaults=parameter_defaults,
        eval_python=eval_python,
        jacobian=jacobian,
        hessian=hessian,
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
    param_map = {name: param for name, param in zip(spec.parameter_names, params)}
    expr = spec.jacobian(value, **param_map)
    if spec.is_scalar:
        if not isinstance(expr, SX):
            raise TypeError("scalar custom Jacobian builders must return SX")
        return expr
    if not isinstance(expr, SXVector) or len(expr) != spec.vector_dim:
        raise TypeError("vector custom Jacobian builders must return an SXVector with matching length")
    return expr


def build_custom_hessian_expr(
    spec: RegisteredElementaryFunction,
    value: SX | SXVector,
    params: tuple[SX, ...],
) -> SX | tuple[SXVector, ...]:
    """Build the symbolic Hessian expression for a custom function."""
    param_map = {name: param for name, param in zip(spec.parameter_names, params)}
    expr = spec.hessian(value, **param_map)
    if spec.is_scalar:
        if not isinstance(expr, SX):
            raise TypeError("scalar custom Hessian builders must return SX")
        return expr
    if (
        not isinstance(expr, tuple)
        or len(expr) != spec.vector_dim
        or any(not isinstance(row, SXVector) or len(row) != spec.vector_dim for row in expr)
    ):
        raise TypeError("vector custom Hessian builders must return a tuple of SXVector rows")
    return expr


def _normalize_input_dimension(input_dimension: int) -> int:
    if not isinstance(input_dimension, int) or input_dimension <= 0:
        raise ValueError("input_dimension must be a positive integer")
    return input_dimension


def _normalize_parameter_defaults(
    parameters: dict[str, float | int] | None,
) -> tuple[tuple[str, float], ...]:
    if parameters is None:
        return ()
    normalized: list[tuple[str, float]] = []
    for name, value in parameters.items():
        if not name.isidentifier():
            raise ValueError("custom parameter names must be valid identifiers")
        if not isinstance(value, (int, float)):
            raise TypeError("custom parameter defaults must be numeric")
        normalized.append((name, float(value)))
    return tuple(normalized)


def _validate_registered_function(spec: RegisteredElementaryFunction) -> None:
    if spec.is_scalar:
        x = SX.sym(f"{spec.name}_x")
        jacobian = build_custom_jacobian_expr(
            spec,
            x,
            tuple(SX.const(value) for _, value in spec.parameter_defaults),
        )
        hessian = build_custom_hessian_expr(
            spec,
            x,
            tuple(SX.const(value) for _, value in spec.parameter_defaults),
        )
        if not isinstance(jacobian, SX) or not isinstance(hessian, SX):
            raise TypeError("scalar custom builders must return scalar expressions")
        return

    dim = spec.vector_dim or 0
    x = SXVector.sym(f"{spec.name}_x", dim)
    jacobian = build_custom_jacobian_expr(
        spec,
        x,
        tuple(SX.const(value) for _, value in spec.parameter_defaults),
    )
    hessian = build_custom_hessian_expr(
        spec,
        x,
        tuple(SX.const(value) for _, value in spec.parameter_defaults),
    )
    if not isinstance(jacobian, SXVector) or len(jacobian) != dim:
        raise TypeError("vector custom Jacobian builders must return an SXVector of matching length")
    if not isinstance(hessian, tuple) or len(hessian) != dim:
        raise TypeError("vector custom Hessian builders must return a tuple of SXVector rows")


def _require_name(name: str | None) -> str:
    if name is None:
        raise ValueError("custom elementary nodes must carry a registered name")
    return name


def _require_integral_const(expr: SX, label: str) -> int:
    if expr.op != "const" or expr.value is None or expr.value != int(expr.value):
        raise ValueError(f"{label} must be an integer constant")
    return int(expr.value)
