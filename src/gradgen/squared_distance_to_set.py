r"""Helpers for projection-backed half-squared distance primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ._custom_elementary import (
    RegisteredElementaryFunction,
    get_registered_elementary_function,
    register_elementary_function,
)
from .sx import SX, SXVector

SqDistanceCallback = Callable[[object], object]
ProjectionCallback = Callable[[object], object]


@dataclass(slots=True)
class SquaredDistanceToSet:
    r"""Represent a projection-backed half-squared distance primitive.

    This helper lets callers define a scalar-valued primitive whose gradient
    is derived from a projection callback. The represented scalar quantity is
    expected to be the half-squared Euclidean distance to a nonempty closed
    convex set:

    .. math::

        \tfrac12 \operatorname{dist}_C(x)^2.

    With that convention, the gradient satisfies

    .. math::

        \nabla \tfrac12 \operatorname{dist}_C(x)^2 = x - \Pi_C(x).

    Instances are configured fluently and can then be called with symbolic
    vectors to produce ``SX`` expressions that participate in regular gradgen
    composition and Rust code generation.

    Args:
        name: Public identifier used for the registered primitive.

    Example:
        >>> from gradgen import SXVector, SquaredDistanceToSet
        >>> distance = (
        ...     SquaredDistanceToSet(name="dist_to_axis")
        ...     .with_sq_distance(lambda x: 0.5 * x[1] * x[1])
        ...     .with_projection(lambda x: (x[0], 0.0))
        ... )
        >>> x = SXVector.sym("x", 2)
        >>> expr = distance(x)
        >>> expr.op
        'custom_vector'
    """

    name: str
    _sq_distance: SqDistanceCallback | None = None
    _projection: ProjectionCallback | None = None
    _rust_sq_distance: str | None = None
    _rust_projection: str | None = None
    _input_dimension: int | None = None

    def __post_init__(self) -> None:
        """Validate constructor arguments."""
        if not self.name or not self.name.isidentifier():
            raise ValueError("name must be a valid identifier")

    def with_sq_distance(
        self,
        callback: SqDistanceCallback,
    ) -> SquaredDistanceToSet:
        """Attach the half-squared distance callback.

        Args:
            callback: Callable that evaluates the represented scalar
                half-squared distance. The callback must accept either a
                symbolic ``SXVector`` or a numeric tuple and return a scalar.

        Returns:
            ``self`` to support fluent configuration.
        """
        self._sq_distance = callback
        return self

    def with_projection(
        self,
        callback: ProjectionCallback,
    ) -> SquaredDistanceToSet:
        """Attach the projection callback used to build gradients.

        Args:
            callback: Callable implementing the projection map ``Pi_C``. The
                callback must accept either a symbolic ``SXVector`` or a
                numeric tuple and return a vector-like value of matching
                length.

        Returns:
            ``self`` to support fluent configuration.
        """
        self._projection = callback
        return self

    def with_rust_sq_distance(self, snippet: str) -> SquaredDistanceToSet:
        """Attach the Rust primal helper snippet.

        Args:
            snippet: Rust source defining the registered primal helper with
                signature ``fn <name>(x: &[T], w: &[T]) -> T``.

        Returns:
            ``self`` to support fluent configuration.
        """
        self._rust_sq_distance = snippet
        return self

    def with_rust_projection(self, snippet: str) -> SquaredDistanceToSet:
        """Attach the Rust projection helper snippet.

        Args:
            snippet: Rust source defining a projection helper with signature
                ``fn <name>_projection(x: &[T], out: &mut [T])``.

        Returns:
            ``self`` to support fluent configuration.
        """
        self._rust_projection = snippet
        return self

    def __call__(self, value: SXVector) -> SX:
        """Build a symbolic call to the configured primitive.

        Args:
            value: Symbolic vector at which the half-squared distance should
                be evaluated.

        Returns:
            An ``SX`` scalar representing the configured primitive applied to
            ``value``.

        Raises:
            TypeError: If ``value`` is not an ``SXVector``.
            ValueError: If mandatory callbacks are missing or the primitive is
                reused with a different input dimension.
        """
        if not isinstance(value, SXVector):
            raise TypeError("SquaredDistanceToSet expects an SXVector input")
        spec = self._ensure_registered(len(value))
        return spec(value)

    def _ensure_registered(self, input_dimension: int) -> RegisteredElementaryFunction:
        """Register the backing custom primitive on first use."""
        self._validate_callbacks()
        if self._input_dimension is not None and input_dimension != self._input_dimension:
            raise ValueError(
                f"SquaredDistanceToSet {self.name!r} expects input length "
                f"{self._input_dimension}, received {input_dimension}"
            )

        try:
            registered = get_registered_elementary_function(self.name)
        except KeyError:
            registered = register_elementary_function(
                name=self.name,
                input_dimension=input_dimension,
                parameter_dimension=0,
                eval_python=self._sq_distance,
                jacobian=self._build_gradient_callback(),
                hessian=None,
                rust_primal=self._rust_sq_distance,
                rust_jacobian=self._build_rust_jacobian(input_dimension),
                rust_hvp=None,
                rust_hessian=None,
            )
        self._input_dimension = input_dimension
        return registered

    def _validate_callbacks(self) -> None:
        """Ensure the required Python callbacks have been provided."""
        if self._sq_distance is None:
            raise ValueError("SquaredDistanceToSet requires with_sq_distance")
        if self._projection is None:
            raise ValueError("SquaredDistanceToSet requires with_projection")

    def _build_gradient_callback(self) -> Callable[[object], tuple[object, ...]]:
        """Return a callback computing ``x - projection(x)``."""
        projection = self._projection
        if projection is None:
            raise ValueError("SquaredDistanceToSet requires with_projection")

        def gradient(value: object) -> tuple[object, ...]:
            projected = projection(value)
            projected_items = _coerce_vector_like(projected)
            value_items = _coerce_vector_like(value)
            if len(projected_items) != len(value_items):
                raise ValueError(
                    "projection callback must return a vector with matching "
                    "length"
                )
            return tuple(
                value_items[index] - projected_items[index]
                for index in range(len(value_items))
            )

        return gradient

    def _build_rust_jacobian(self, input_dimension: int) -> str | None:
        """Return a Rust Jacobian helper synthesized from the projection."""
        if self._rust_projection is None:
            return None
        return "\n".join(
            [
                self._rust_projection.strip(),
                "",
                (
                    f"fn {self.name}_jacobian("
                    "x: &[{{ scalar_type }}], "
                    "w: &[{{ scalar_type }}], "
                    "out: &mut [{{ scalar_type }}],"
                    ") {"
                ),
                "    let _ = w;",
                (
                    f"    let mut projection = "
                    f"[0.0_{{{{ scalar_type }}}}; {input_dimension}];"
                ),
                f"    {self.name}_projection(x, &mut projection);",
                *[
                    (
                        f"    out[{index}] = x[{index}] - "
                        f"projection[{index}];"
                    )
                    for index in range(input_dimension)
                ],
                "}",
            ]
        )


def _coerce_vector_like(value: object) -> tuple[object, ...]:
    """Return ``value`` as a tuple of vector components."""
    if isinstance(value, SXVector):
        return tuple(value)
    if isinstance(value, (str, bytes)):
        raise TypeError("projection callbacks must return vector-like values")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(
            "projection callbacks must return vector-like values"
        ) from exc
