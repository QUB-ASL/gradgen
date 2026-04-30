r"""Helpers for projection-backed half-squared distance primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ._custom_elementary import (
    RegisteredElementaryFunction,
    get_registered_elementary_function,
    register_elementary_function,
)
from .function import Function
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
    _sq_distance_function: Function | None = None
    _projection_function: Function | None = None
    _rust_sq_distance: str | None = None
    _rust_projection: str | None = None
    _input_dimension: int | None = None
    _function_cache: Function | None = None

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
        self._invalidate_function_cache()
        self._sq_distance = callback
        return self

    def with_sq_distance_function(
        self,
        function: Function,
    ) -> SquaredDistanceToSet:
        """Attach a symbolic half-squared distance function.

        Args:
            function: A gradgen ``Function`` with exactly one vector input
                block and one scalar output block representing the half-
                squared distance.

        Returns:
            ``self`` to support fluent configuration.
        """
        self._validate_single_vector_input_function(
            function,
            output_kind="scalar",
            label="sq_distance function",
        )
        self._invalidate_function_cache()
        self._sq_distance_function = function
        self._update_input_dimension_from_function(function)
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
        self._invalidate_function_cache()
        self._projection = callback
        return self

    def with_projection_function(
        self,
        function: Function,
    ) -> SquaredDistanceToSet:
        """Attach a symbolic projection function.

        Args:
            function: A gradgen ``Function`` with exactly one vector input
                block and one vector output block representing the
                projection map.

        Returns:
            ``self`` to support fluent configuration.
        """
        self._validate_single_vector_input_function(
            function,
            output_kind="vector",
            label="projection function",
        )
        self._invalidate_function_cache()
        self._projection_function = function
        self._update_input_dimension_from_function(function)
        return self

    def with_rust_sq_distance(self, snippet: str) -> SquaredDistanceToSet:
        """Attach the Rust primal helper snippet.

        Args:
            snippet: Rust source defining the registered primal helper with
                signature ``fn <name>(x: &[T], w: &[T]) -> T``.

        Returns:
            ``self`` to support fluent configuration.
        """
        self._invalidate_function_cache()
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
        self._invalidate_function_cache()
        self._rust_projection = snippet
        return self

    def __call__(
        self,
        value: SXVector | tuple[object, ...] | list[object],
    ) -> SX | float | tuple[float, ...] | SXVector | tuple[object, ...]:
        """Evaluate the configured distance as a function-like object.

        Args:
            value: Symbolic vector or numeric vector-like value at which the
                half-squared distance should be evaluated.

        Returns:
            The evaluated half-squared distance, using the same calling
            conventions as :class:`gradgen.function.Function`.

        Raises:
            TypeError: If ``value`` is not vector-like.
            ValueError: If mandatory callbacks are missing or the primitive
                is reused with a different input dimension.
        """
        vector_value = _coerce_vector_like(value)
        self._validate_input_dimension(len(vector_value), allow_unset=True)
        return self.to_function()(value)

    def to_function(self, name: str | None = None) -> Function:
        """Return a symbolic function view of the configured distance.

        Args:
            name: Optional symbolic function name. When omitted, the
                configured primitive name is used.

        Returns:
            A :class:`~gradgen.function.Function` representing the configured
            half-squared distance as a scalar-valued symbolic function.

        Raises:
            ValueError: If the input dimension is not known yet.
        """
        if name is None and self._function_cache is not None:
            return self._function_cache

        if self._input_dimension is None:
            raise ValueError(
                "SquaredDistanceToSet needs an input dimension before it can "
                "be materialized; call it once or configure a symbolic "
                "helper function first"
            )

        x = SXVector.sym("x", self._input_dimension)
        symbolic_expr = self._build_symbolic_expr(x)
        if symbolic_expr is None:
            spec = self._ensure_registered(self._input_dimension)
            symbolic_expr = spec(x)

        function = Function(
            name or self.name,
            [x],
            [symbolic_expr],
            input_names=["x"],
            output_names=["y"],
        )
        if name is None:
            self._function_cache = function
        return function

    def gradient(
        self,
        wrt_index: int = 0,
        name: str | None = None,
    ) -> Function:
        """Return the gradient of the materialized distance function.

        Args:
            wrt_index: Index of the input block to differentiate with
                respect to.
            name: Optional name for the returned symbolic function.

        Returns:
            A :class:`~gradgen.function.Function` representing the gradient
            with respect to the selected input block.
        """
        return self.to_function().gradient(wrt_index=wrt_index, name=name)

    def jacobian(
        self,
        wrt_index: int = 0,
        name: str | None = None,
    ) -> Function:
        """Return the Jacobian of the materialized distance function.

        Args:
            wrt_index: Index of the input block to differentiate with
                respect to.
            name: Optional name for the returned symbolic function.

        Returns:
            A :class:`~gradgen.function.Function` representing the Jacobian
            block with respect to the selected input block.
        """
        return self.to_function().jacobian(wrt_index=wrt_index, name=name)

    def __getattr__(self, name: str) -> object:
        """Delegate remaining function-like attributes to ``to_function()``.

        This keeps the wrapper aligned with :class:`gradgen.function.Function`
        without reimplementing the full surface area manually.
        """
        return getattr(self.to_function(), name)

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
        if self._sq_distance_function is not None:
            return
        if self._projection_function is not None:
            return
        if self._sq_distance is None:
            raise ValueError("SquaredDistanceToSet requires with_sq_distance")
        if self._projection is None:
            raise ValueError("SquaredDistanceToSet requires with_projection")

    def _build_symbolic_expr(self, value: SXVector) -> SX | None:
        """Build a symbolic expression from configured gradgen Functions."""
        if self._sq_distance_function is not None:
            self._validate_input_dimension(len(value))
            output = self._sq_distance_function(value)
            if not isinstance(output, SX):
                raise TypeError(
                    "sq_distance function must return a single scalar output"
                )
            return output

        if self._projection_function is not None:
            self._validate_input_dimension(len(value))
            projected = self._projection_function(value)
            if not isinstance(projected, SXVector):
                raise TypeError(
                    "projection function must return a single vector output"
                )
            delta = value - projected
            return 0.5 * delta.norm2sq()

        return None

    def _invalidate_function_cache(self) -> None:
        """Forget any cached symbolic :class:`Function` view."""
        self._function_cache = None

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

    def _update_input_dimension_from_function(self, function: Function) -> None:
        """Record and validate the shared symbolic input dimension."""
        vector_input = function.inputs[0]
        if not isinstance(vector_input, SXVector):
            raise TypeError("SquaredDistanceToSet symbolic inputs must be vectors")
        self._validate_input_dimension(len(vector_input), allow_unset=True)
        self._input_dimension = len(vector_input)

    def _validate_input_dimension(
        self,
        input_dimension: int,
        *,
        allow_unset: bool = False,
    ) -> None:
        """Validate the configured vector input dimension."""
        if self._input_dimension is None:
            if allow_unset:
                self._input_dimension = input_dimension
                return
            self._input_dimension = input_dimension
            return
        if input_dimension != self._input_dimension:
            raise ValueError(
                f"SquaredDistanceToSet {self.name!r} expects input length "
                f"{self._input_dimension}, received {input_dimension}"
            )

    def _validate_single_vector_input_function(
        self,
        function: Function,
        *,
        output_kind: str,
        label: str,
    ) -> None:
        """Validate a symbolic helper function shape."""
        if len(function.inputs) != 1:
            raise TypeError(f"{label} must accept exactly one input block")
        if len(function.outputs) != 1:
            raise TypeError(f"{label} must return exactly one output block")
        if not isinstance(function.inputs[0], SXVector):
            raise TypeError(f"{label} input must be an SXVector")
        if output_kind == "scalar" and not isinstance(function.outputs[0], SX):
            raise TypeError(f"{label} output must be an SX scalar")
        if (
            output_kind == "vector"
            and not isinstance(function.outputs[0], SXVector)
        ):
            raise TypeError(f"{label} output must be an SXVector")


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
