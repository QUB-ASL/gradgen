"""Core data structures for user-defined elementary functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..sx import SX, SXNode, SXVector


ScalarJacobianBuilder = Callable[..., object]
ScalarHessianBuilder = Callable[..., object]
ScalarHvpBuilder = Callable[..., object]
VectorJacobianBuilder = Callable[..., object]
VectorHessianBuilder = Callable[..., object]
VectorHvpBuilder = Callable[..., object]
PythonEvalBuilder = Callable[..., float]


def coerce_parameter_value(value: float | int) -> SX:
    """Return ``value`` as a constant symbolic scalar."""
    if not isinstance(value, (int, float)):
        raise TypeError("custom function parameters must be numeric constants")
    return SX.const(value)


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
        """Return whether the primitive accepts a scalar input."""
        return self.input_dimension == 1

    @property
    def vector_dim(self) -> int | None:
        """Return the vector input dimension, if any."""
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
        return tuple(coerce_parameter_value(value) for value in resolved_values)

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
                f"custom vector function {self.name!r} expects length {self.vector_dim}, "
                f"received {len(value)}"
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


def custom_scalar_jacobian(name: str, x: SX, params: tuple[SX, ...]) -> SX:
    """Create a symbolic node representing a custom scalar Jacobian."""
    return SX(SXNode.make("custom_scalar_jacobian", (x.node, *(param.node for param in params)), name=name))


def custom_scalar_hvp(name: str, x: SX, tangent: SX, params: tuple[SX, ...]) -> SX:
    """Create a symbolic node representing a custom scalar Hessian-vector product."""
    return SX(
        SXNode.make(
            "custom_scalar_hvp",
            (x.node, tangent.node, *(param.node for param in params)),
            name=name,
        )
    )


def custom_scalar_hessian(name: str, x: SX, params: tuple[SX, ...]) -> SX:
    """Create a symbolic node representing a custom scalar Hessian."""
    return SX(SXNode.make("custom_scalar_hessian", (x.node, *(param.node for param in params)), name=name))


def custom_vector_jacobian_component(
    name: str,
    index: int,
    x: SXVector,
    params: tuple[SX, ...],
) -> SX:
    """Create a symbolic node for one component of a custom vector Jacobian."""
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
    """Create a symbolic node for one component of a custom vector HVP."""
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
    """Create a symbolic node for one entry of a custom vector Hessian."""
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
