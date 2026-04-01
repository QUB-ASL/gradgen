"""Core `SX` scalar and vector expression types.

This module provides the first symbolic building block for the library:

- ``SXNode`` is the canonical internal graph node
- ``SX`` is the small user-facing wrapper around a scalar node
- ``SXVector`` is a lightweight vector container built from ``SX`` values
- helper functions such as :func:`sin` and :func:`sqrt` make expression
  construction feel natural from Python

The implementation uses structural interning, also called hash-consing.
That means two expressions with the same structure reuse the same
``SXNode`` instance. This keeps the symbolic graph compact and gives us a
directed acyclic graph (DAG) instead of repeatedly rebuilding identical
subtrees.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from threading import Lock
from typing import ClassVar, Iterable, Iterator, Sequence, overload


# Operations whose operands can be reordered without changing semantics.
# Normalizing these makes expressions like ``x + y`` and ``y + x`` share
# the same underlying node.
_COMMUTATIVE_OPS = frozenset({"add", "mul"})


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SXNode:
    """Canonical scalar expression node.

    Instances of this class are immutable and are intended to be created
    through :meth:`make`, not by calling the dataclass constructor
    directly. ``make`` looks up a node in a global cache and reuses an
    existing node when the operation and operands are structurally
    identical.

    Attributes:
        op: Operation code, such as ``"symbol"``, ``"const"``, ``"add"``,
            or ``"sin"``.
        args: Child nodes for non-leaf expressions.
        name: Symbol name for ``"symbol"`` nodes.
        metadata: Optional symbol metadata for ``"symbol"`` nodes.
        value: Numeric literal for ``"const"`` nodes.
    """

    op: str
    args: tuple[SXNode, ...] = ()
    name: str | None = None
    metadata: tuple[tuple[str, Hashable], ...] = ()
    value: float | None = None
    _cache: ClassVar[dict[tuple[object, ...], SXNode]] = {}
    _lock: ClassVar[Lock] = Lock()

    @classmethod
    def make(
        cls,
        op: str,
        args: tuple[SXNode, ...] = (),
        *,
        name: str | None = None,
        metadata: dict[str, Hashable] | None = None,
        value: float | None = None,
    ) -> SXNode:
        """Create or reuse a structurally identical node.

        This method is the heart of the interning strategy. It first
        normalizes the operand order for commutative operations, then uses
        the resulting structure as a cache key.

        Args:
            op: Operation code for the node being created.
            args: Child nodes of the expression.
            name: Optional symbol name for ``"symbol"`` nodes.
            metadata: Optional metadata attached to ``"symbol"`` nodes.
            value: Optional floating-point literal for ``"const"`` nodes.

        Returns:
            The unique canonical node matching the requested structure.
        """
        normalized_args = cls._normalize_args(op, args)
        normalized_metadata = _normalize_metadata(metadata)
        key = (op, normalized_args, name, normalized_metadata, value)

        with cls._lock:
            node = cls._cache.get(key)
            if node is not None:
                return node

            node = cls(
                op=op,
                args=normalized_args,
                name=name,
                metadata=normalized_metadata,
                value=value,
            )
            cls._cache[key] = node
            return node

    @staticmethod
    def _normalize_args(op: str, args: tuple[SXNode, ...]) -> tuple[SXNode, ...]:
        """Return a canonical ordering of arguments for supported ops.

        For commutative operations we sort operands so that equivalent
        expressions map to the same cache key. For example, ``add(x, y)``
        and ``add(y, x)`` become identical.
        """
        if op in _COMMUTATIVE_OPS:
            return tuple(sorted(args, key=lambda node: repr(node)))
        return args

    def __repr__(self) -> str:
        if self.op == "symbol":
            if self.metadata:
                return f"SXNode(symbol={self.name!r}, metadata={dict(self.metadata)!r})"
            return f"SXNode(symbol={self.name!r})"
        if self.op == "const":
            return f"SXNode(const={self.value!r})"
        return f"SXNode(op={self.op!r}, args={self.args!r})"


@dataclass(frozen=True, slots=True)
class SX:
    """User-facing scalar symbolic expression wrapper.

    ``SX`` is a lightweight immutable wrapper around an :class:`SXNode`.
    Most user code should work with ``SX`` values rather than raw nodes.
    Operator overloading is implemented here so expressions such as
    ``x + y`` or ``sin(x)`` produce symbolic graph nodes naturally.
    """

    node: SXNode = field(repr=False)

    @classmethod
    def sym(
        cls,
        name: str,
        metadata: dict[str, Hashable] | None = None,
    ) -> SX:
        """Create a symbolic scalar variable.

        Args:
            name: Symbol name used to identify the variable.
            metadata: Optional metadata associated with the symbol. This
                is recorded as part of the symbol identity but does not
                yet change algebraic, AD, or code-generation semantics.

        Returns:
            An ``SX`` wrapper around a canonical ``"symbol"`` node.
        """
        return cls(SXNode.make("symbol", name=name, metadata=metadata))

    @classmethod
    def const(cls, value: float | int) -> SX:
        """Create a scalar constant expression.

        Integer values are converted to ``float`` so the symbolic layer has
        one numeric literal representation for now.
        """
        return cls(SXNode.make("const", value=float(value)))

    @property
    def op(self) -> str:
        """Return the operation code of the underlying node."""
        return self.node.op

    @property
    def args(self) -> tuple[SX, ...]:
        """Return child expressions as ``SX`` wrappers."""
        return tuple(SX(arg) for arg in self.node.args)

    @property
    def name(self) -> str | None:
        """Return the symbol name, if this is a symbol node."""
        return self.node.name

    @property
    def value(self) -> float | None:
        """Return the numeric value, if this is a constant node."""
        return self.node.value

    @property
    def metadata(self) -> dict[str, Hashable]:
        """Return symbol metadata as a plain dictionary.

        Non-symbol expressions return an empty dictionary. The returned
        dictionary is detached from the interned node state.
        """
        return dict(self.node.metadata)

    def __add__(self, other: object) -> SX:
        return _binary("add", self, other)

    def __radd__(self, other: object) -> SX:
        return _binary("add", other, self)

    def __sub__(self, other: object) -> SX:
        return _binary("sub", self, other)

    def __rsub__(self, other: object) -> SX:
        return _binary("sub", other, self)

    def __mul__(self, other: object) -> SX | SXVector:
        if isinstance(other, SXVector):
            if len(other) == 1:
                return _binary("mul", self, other[0])
            return SXVector(tuple(self * element for element in other.elements))
        return _binary("mul", self, other)

    def __rmul__(self, other: object) -> SX | SXVector:
        if isinstance(other, SXVector):
            if len(other) == 1:
                return _binary("mul", other[0], self)
            return SXVector(tuple(element * self for element in other.elements))
        return _binary("mul", other, self)

    def __truediv__(self, other: object) -> SX:
        return _binary("div", self, other)

    def __rtruediv__(self, other: object) -> SX:
        return _binary("div", other, self)

    def __pow__(self, other: object) -> SX:
        return _binary("pow", self, other)

    def __rpow__(self, other: object) -> SX:
        return _binary("pow", other, self)

    def __neg__(self) -> SX:
        return _unary("neg", self)

    def sin(self) -> SX:
        return _unary("sin", self)

    def cos(self) -> SX:
        return _unary("cos", self)

    def tan(self) -> SX:
        return _unary("tan", self)

    def asin(self) -> SX:
        return _unary("asin", self)

    def acos(self) -> SX:
        return _unary("acos", self)

    def atan(self) -> SX:
        return _unary("atan", self)

    def atan2(self, other: object) -> SX:
        return _binary("atan2", self, other)

    def sinh(self) -> SX:
        return _unary("sinh", self)

    def cosh(self) -> SX:
        return _unary("cosh", self)

    def tanh(self) -> SX:
        return _unary("tanh", self)

    def asinh(self) -> SX:
        return _unary("asinh", self)

    def acosh(self) -> SX:
        return _unary("acosh", self)

    def atanh(self) -> SX:
        return _unary("atanh", self)

    def exp(self) -> SX:
        return _unary("exp", self)

    def expm1(self) -> SX:
        return _unary("expm1", self)

    def log(self) -> SX:
        return _unary("log", self)

    def log1p(self) -> SX:
        return _unary("log1p", self)

    def sqrt(self) -> SX:
        return _unary("sqrt", self)

    def cbrt(self) -> SX:
        return _unary("cbrt", self)

    def erf(self) -> SX:
        return _unary("erf", self)

    def erfc(self) -> SX:
        return _unary("erfc", self)

    def floor(self) -> SX:
        return _unary("floor", self)

    def ceil(self) -> SX:
        return _unary("ceil", self)

    def round(self) -> SX:
        return _unary("round", self)

    def trunc(self) -> SX:
        return _unary("trunc", self)

    def fract(self) -> SX:
        return _unary("fract", self)

    def signum(self) -> SX:
        return _unary("signum", self)

    def hypot(self, other: object) -> SX:
        return _binary("hypot", self, other)

    def abs(self) -> SX:
        return _unary("abs", self)

    def maximum(self, other: object) -> SX:
        return _binary("max", self, other)

    def minimum(self, other: object) -> SX:
        return _binary("min", self, other)

    def __repr__(self) -> str:
        if self.op == "symbol":
            if self.node.metadata:
                return f"SX.sym({self.name!r}, metadata={dict(self.node.metadata)!r})"
            return f"SX.sym({self.name!r})"
        if self.op == "const":
            return f"SX.const({self.value!r})"
        if len(self.node.args) == 1:
            return f"{self.op}({SX(self.node.args[0])!r})"
        if len(self.node.args) == 2:
            lhs, rhs = self.node.args
            return f"{self.op}({SX(lhs)!r}, {SX(rhs)!r})"
        return f"SX({self.node!r})"


@dataclass(frozen=True, slots=True)
class SXVector:
    """Vector of scalar symbolic expressions.

    ``SXVector`` intentionally stays small for the first iteration of the
    library. It is a thin immutable container around a tuple of ``SX``
    expressions and provides only vector operations that map cleanly onto
    the current scalar graph implementation.

    For now the class supports:

    - symbolic vector construction
    - indexing, iteration, and length
    - elementwise addition, subtraction, and division
    - scalar-vector multiplication
    - elementwise unary functions
    - dot products
    - common vector norms

    Elementwise vector-vector multiplication is intentionally unsupported
    at this stage so that ``*`` remains reserved for scalar-vector
    multiplication.
    """

    elements: tuple[SX, ...]

    @classmethod
    def sym(
        cls,
        name: str,
        length: int,
        metadata: dict[str, Hashable] | None = None,
    ) -> SXVector:
        """Create a symbolic vector with indexed scalar element names.

        Args:
            name: Base name for the vector.
            length: Number of scalar elements in the vector.
            metadata: Optional metadata copied onto each scalar symbol in
                the vector.

        Returns:
            An ``SXVector`` whose elements are named ``"{name}_{i}"``.
        """
        if length < 0:
            raise ValueError("vector length must be non-negative")
        return cls(tuple(SX.sym(f"{name}_{index}", metadata=metadata) for index in range(length)))

    def __len__(self) -> int:
        """Return the vector length."""
        return len(self.elements)

    def __iter__(self) -> Iterator[SX]:
        """Iterate over the scalar elements of the vector."""
        return iter(self.elements)

    @overload
    def __getitem__(self, index: int) -> SX:
        ...

    @overload
    def __getitem__(self, index: slice) -> SXVector:
        ...

    def __getitem__(self, index: int | slice) -> SX | SXVector:
        """Return either one scalar element or a sliced vector view."""
        if isinstance(index, slice):
            return SXVector(self.elements[index])
        return self.elements[index]

    def __add__(self, other: object) -> SXVector | SX:
        """Return the elementwise sum of two vectors.

        A singleton vector may also participate in scalar arithmetic for
        convenience, so ``SXVector.sym("u", 1) + x`` behaves like
        ``u[0] + x``.
        """
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return self.elements[0] + scalar
        return self._elementwise_binary("add", other)

    def __radd__(self, other: object) -> SXVector | SX:
        """Return the elementwise sum of two vectors."""
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return scalar + self.elements[0]
        return self._elementwise_binary("add", other, reverse=True)

    def __sub__(self, other: object) -> SXVector | SX:
        """Return the elementwise difference of two vectors."""
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return self.elements[0] - scalar
        return self._elementwise_binary("sub", other)

    def __rsub__(self, other: object) -> SXVector | SX:
        """Return the reversed elementwise difference of two vectors."""
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return scalar - self.elements[0]
        return self._elementwise_binary("sub", other, reverse=True)

    def __mul__(self, other: object) -> SXVector | SX:
        """Return a scalar-vector product.

        Vector-vector multiplication is intentionally unsupported for now.
        Use :meth:`dot` for an inner product.
        """
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return self.elements[0] * scalar
        scalar = _coerce_scalar(other)
        return SXVector(tuple(element * scalar for element in self.elements))

    def __rmul__(self, other: object) -> SXVector | SX:
        """Return a scalar-vector product."""
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return scalar * self.elements[0]
        scalar = _coerce_scalar(other)
        return SXVector(tuple(scalar * element for element in self.elements))

    def __truediv__(self, other: object) -> SXVector | SX:
        """Return elementwise division by a scalar or vector."""
        if isinstance(other, SXVector):
            return self._elementwise_binary("div", other)
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return self.elements[0] / scalar
        scalar = _coerce_scalar(other)
        return SXVector(tuple(element / scalar for element in self.elements))

    def __rtruediv__(self, other: object) -> SXVector | SX:
        """Return scalar divided by each vector element."""
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return scalar / self.elements[0]
        scalar = _coerce_scalar(other)
        return SXVector(tuple(scalar / element for element in self.elements))

    def __pow__(self, other: object) -> SXVector | SX:
        """Return elementwise power with a scalar exponent."""
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return self.elements[0] ** scalar
        scalar = _coerce_scalar(other)
        return SXVector(tuple(element ** scalar for element in self.elements))

    def __rpow__(self, other: object) -> SXVector | SX:
        """Return scalar raised to each vector element."""
        scalar = self._coerce_mixed_scalar(other)
        if scalar is not None:
            return scalar ** self.elements[0]
        scalar = _coerce_scalar(other)
        return SXVector(tuple(scalar ** element for element in self.elements))

    def __neg__(self) -> SXVector:
        """Return the elementwise negation of the vector."""
        return SXVector(tuple(-element for element in self.elements))

    def sin(self) -> SXVector:
        """Apply sine elementwise."""
        return SXVector(tuple(element.sin() for element in self.elements))

    def cos(self) -> SXVector:
        """Apply cosine elementwise."""
        return SXVector(tuple(element.cos() for element in self.elements))

    def tan(self) -> SXVector:
        """Apply tangent elementwise."""
        return SXVector(tuple(element.tan() for element in self.elements))

    def asin(self) -> SXVector:
        """Apply arcsine elementwise."""
        return SXVector(tuple(element.asin() for element in self.elements))

    def acos(self) -> SXVector:
        """Apply arccosine elementwise."""
        return SXVector(tuple(element.acos() for element in self.elements))

    def atan(self) -> SXVector:
        """Apply arctangent elementwise."""
        return SXVector(tuple(element.atan() for element in self.elements))

    def asinh(self) -> SXVector:
        """Apply inverse hyperbolic sine elementwise."""
        return SXVector(tuple(element.asinh() for element in self.elements))

    def acosh(self) -> SXVector:
        """Apply inverse hyperbolic cosine elementwise."""
        return SXVector(tuple(element.acosh() for element in self.elements))

    def atanh(self) -> SXVector:
        """Apply inverse hyperbolic tangent elementwise."""
        return SXVector(tuple(element.atanh() for element in self.elements))

    def sinh(self) -> SXVector:
        """Apply hyperbolic sine elementwise."""
        return SXVector(tuple(element.sinh() for element in self.elements))

    def cosh(self) -> SXVector:
        """Apply hyperbolic cosine elementwise."""
        return SXVector(tuple(element.cosh() for element in self.elements))

    def tanh(self) -> SXVector:
        """Apply hyperbolic tangent elementwise."""
        return SXVector(tuple(element.tanh() for element in self.elements))

    def exp(self) -> SXVector:
        """Apply exponential elementwise."""
        return SXVector(tuple(element.exp() for element in self.elements))

    def expm1(self) -> SXVector:
        """Apply ``exp(x) - 1`` elementwise."""
        return SXVector(tuple(element.expm1() for element in self.elements))

    def log(self) -> SXVector:
        """Apply natural logarithm elementwise."""
        return SXVector(tuple(element.log() for element in self.elements))

    def log1p(self) -> SXVector:
        """Apply ``log(1 + x)`` elementwise."""
        return SXVector(tuple(element.log1p() for element in self.elements))

    def sqrt(self) -> SXVector:
        """Apply square root elementwise."""
        return SXVector(tuple(element.sqrt() for element in self.elements))

    def cbrt(self) -> SXVector:
        """Apply cube root elementwise."""
        return SXVector(tuple(element.cbrt() for element in self.elements))

    def erf(self) -> SXVector:
        """Apply error function elementwise."""
        return SXVector(tuple(element.erf() for element in self.elements))

    def erfc(self) -> SXVector:
        """Apply complementary error function elementwise."""
        return SXVector(tuple(element.erfc() for element in self.elements))

    def floor(self) -> SXVector:
        """Apply floor elementwise."""
        return SXVector(tuple(element.floor() for element in self.elements))

    def ceil(self) -> SXVector:
        """Apply ceil elementwise."""
        return SXVector(tuple(element.ceil() for element in self.elements))

    def round(self) -> SXVector:
        """Apply round elementwise."""
        return SXVector(tuple(element.round() for element in self.elements))

    def trunc(self) -> SXVector:
        """Apply truncation elementwise."""
        return SXVector(tuple(element.trunc() for element in self.elements))

    def fract(self) -> SXVector:
        """Apply fractional-part extraction elementwise."""
        return SXVector(tuple(element.fract() for element in self.elements))

    def signum(self) -> SXVector:
        """Apply sign extraction elementwise."""
        return SXVector(tuple(element.signum() for element in self.elements))

    def abs(self) -> SXVector:
        """Apply absolute value elementwise."""
        return SXVector(tuple(element.abs() for element in self.elements))

    def dot(self, other: object) -> SX:
        """Return the symbolic dot product of two vectors."""
        vector = _coerce_vector(other)
        self._check_same_length(vector)

        if len(self) == 0:
            return SX.const(0.0)

        pairs = iter(zip(self.elements, vector.elements))
        first_left, first_right = next(pairs)
        total = first_left * first_right

        for left, right in pairs:
            total = total + (left * right)
        return total

    def sum(self) -> SX:
        """Return the sum of all vector elements."""
        if len(self) == 0:
            return SX.const(0.0)
        return SX(SXNode.make("sum", tuple(element.node for element in self.elements)))

    def prod(self) -> SX:
        """Return the product of all vector elements."""
        if len(self) == 0:
            return SX.const(1.0)
        return SX(SXNode.make("prod", tuple(element.node for element in self.elements)))

    def max(self) -> SX:
        """Return the maximum vector element."""
        if len(self) == 0:
            raise ValueError("vector max is undefined for empty vectors")
        return SX(SXNode.make("reduce_max", tuple(element.node for element in self.elements)))

    def min(self) -> SX:
        """Return the minimum vector element."""
        if len(self) == 0:
            raise ValueError("vector min is undefined for empty vectors")
        return SX(SXNode.make("reduce_min", tuple(element.node for element in self.elements)))

    def mean(self) -> SX:
        """Return the arithmetic mean of the vector elements."""
        if len(self) == 0:
            raise ValueError("vector mean is undefined for empty vectors")
        return SX(SXNode.make("mean", tuple(element.node for element in self.elements)))

    def norm2(self) -> SX:
        """Return the Euclidean norm of the vector."""
        if len(self) == 0:
            return SX.const(0.0)
        return SX(SXNode.make("norm2", tuple(element.node for element in self.elements)))

    def norm2sq(self) -> SX:
        """Return the squared Euclidean norm of the vector."""
        if len(self) == 0:
            return SX.const(0.0)
        return SX(SXNode.make("norm2sq", tuple(element.node for element in self.elements)))

    def norm1(self) -> SX:
        """Return the 1-norm of the vector."""
        if len(self) == 0:
            return SX.const(0.0)
        return SX(SXNode.make("norm1", tuple(element.node for element in self.elements)))

    def norm_inf(self) -> SX:
        """Return the infinity norm of the vector."""
        if len(self) == 0:
            return SX.const(0.0)
        return SX(SXNode.make("norm_inf", tuple(element.node for element in self.elements)))

    def norm_p(self, p: object) -> SX:
        """Return the p-norm of the vector."""
        if len(self) == 0:
            return SX.const(0.0)
        p_scalar = _coerce(p)
        return SX(SXNode.make("norm_p", tuple((*[element.node for element in self.elements], p_scalar.node))))

    def norm_p_to_p(self, p: object) -> SX:
        """Return the p-norm of the vector raised to the power p."""
        if len(self) == 0:
            return SX.const(0.0)
        p_scalar = _coerce(p)
        return SX(SXNode.make("norm_p_to_p", tuple((*[element.node for element in self.elements], p_scalar.node))))

    def __repr__(self) -> str:
        return f"SXVector(elements={self.elements!r})"

    def _elementwise_binary(
        self,
        op: str,
        other: object,
        *,
        reverse: bool = False,
    ) -> SXVector:
        """Apply a binary operation elementwise to two vectors."""
        vector = _coerce_vector(other)
        self._check_same_length(vector)

        if reverse:
            return SXVector(
                tuple(_binary(op, rhs, lhs) for lhs, rhs in zip(self.elements, vector.elements))
            )
        return SXVector(
            tuple(_binary(op, lhs, rhs) for lhs, rhs in zip(self.elements, vector.elements))
        )

    def _check_same_length(self, other: SXVector) -> None:
        """Raise when two vectors do not have the same length."""
        if len(self) != len(other):
            raise ValueError("vector lengths must match")

    def _coerce_mixed_scalar(self, other: object) -> SX | None:
        """Return a scalar view when a singleton vector meets a scalar."""
        if len(self) != 1 or isinstance(other, SXVector):
            return None
        return _coerce(other)


def const(value: float | int) -> SX:
    """Create a scalar constant expression.

    This is a convenience alias for :meth:`SX.const`.
    """
    return SX.const(value)


def sin(expr: object) -> SX:
    """Return the symbolic sine of an expression."""
    return _unary("sin", expr)


def cos(expr: object) -> SX:
    """Return the symbolic cosine of an expression."""
    return _unary("cos", expr)


def tan(expr: object) -> SX:
    """Return the symbolic tangent of an expression."""
    return _unary("tan", expr)


def asin(expr: object) -> SX:
    """Return the symbolic arcsine of an expression."""
    return _unary("asin", expr)


def acos(expr: object) -> SX:
    """Return the symbolic arccosine of an expression."""
    return _unary("acos", expr)


def atan(expr: object) -> SX:
    """Return the symbolic arctangent of an expression."""
    return _unary("atan", expr)


def atan2(lhs: object, rhs: object) -> SX:
    """Return the symbolic arctangent of ``lhs / rhs`` with quadrant handling."""
    return _binary("atan2", lhs, rhs)


def sinh(expr: object) -> SX:
    """Return the symbolic hyperbolic sine of an expression."""
    return _unary("sinh", expr)


def cosh(expr: object) -> SX:
    """Return the symbolic hyperbolic cosine of an expression."""
    return _unary("cosh", expr)


def tanh(expr: object) -> SX:
    """Return the symbolic hyperbolic tangent of an expression."""
    return _unary("tanh", expr)


def asinh(expr: object) -> SX:
    """Return the symbolic inverse hyperbolic sine of an expression."""
    return _unary("asinh", expr)


def acosh(expr: object) -> SX:
    """Return the symbolic inverse hyperbolic cosine of an expression."""
    return _unary("acosh", expr)


def atanh(expr: object) -> SX:
    """Return the symbolic inverse hyperbolic tangent of an expression."""
    return _unary("atanh", expr)


def exp(expr: object) -> SX:
    """Return the symbolic exponential of an expression."""
    return _unary("exp", expr)


def expm1(expr: object) -> SX:
    """Return the symbolic ``exp(x) - 1`` of an expression."""
    return _unary("expm1", expr)


def log(expr: object) -> SX:
    """Return the symbolic natural logarithm of an expression."""
    return _unary("log", expr)


def log1p(expr: object) -> SX:
    """Return the symbolic ``log(1 + x)`` of an expression."""
    return _unary("log1p", expr)


def sqrt(expr: object) -> SX:
    """Return the symbolic square root of an expression."""
    return _unary("sqrt", expr)


def cbrt(expr: object) -> SX:
    """Return the symbolic cube root of an expression."""
    return _unary("cbrt", expr)


def erf(expr: object) -> SX:
    """Return the symbolic error function of an expression."""
    return _unary("erf", expr)


def erfc(expr: object) -> SX:
    """Return the symbolic complementary error function of an expression."""
    return _unary("erfc", expr)


def floor(expr: object) -> SX:
    """Return the symbolic floor of an expression."""
    return _unary("floor", expr)


def ceil(expr: object) -> SX:
    """Return the symbolic ceil of an expression."""
    return _unary("ceil", expr)


def round(expr: object) -> SX:
    """Return the symbolic nearest integer of an expression."""
    return _unary("round", expr)


def trunc(expr: object) -> SX:
    """Return the symbolic truncation of an expression."""
    return _unary("trunc", expr)


def fract(expr: object) -> SX:
    """Return the symbolic fractional part of an expression."""
    return _unary("fract", expr)


def signum(expr: object) -> SX:
    """Return the symbolic sign of an expression."""
    return _unary("signum", expr)


def hypot(lhs: object, rhs: object) -> SX:
    """Return the symbolic Euclidean norm of two scalar-like values."""
    return _binary("hypot", lhs, rhs)


def maximum(lhs: object, rhs: object) -> SX:
    """Return the symbolic maximum of two scalar-like expressions."""
    return _binary("max", lhs, rhs)


def minimum(lhs: object, rhs: object) -> SX:
    """Return the symbolic minimum of two scalar-like expressions."""
    return _binary("min", lhs, rhs)


def vector(values: Iterable[object]) -> SXVector:
    """Create a symbolic vector from a sequence of scalar-like values."""
    return SXVector(tuple(_coerce(value) for value in values))


def matvec(matrix: Sequence[Sequence[float | int]], x: object) -> SXVector:
    """Return the symbolic matrix-vector product for a constant matrix."""
    x_vector = _coerce_vector(x)
    rows, cols, values = _coerce_constant_matrix(matrix)
    if cols != len(x_vector):
        raise ValueError("matrix column count must match vector length")
    return SXVector(
        tuple(
            SX(
                SXNode.make(
                    "matvec_component",
                    _build_matvec_component_args(rows, cols, row, values, x_vector.elements),
                )
            )
            for row in range(rows)
        )
    )


def quadform(
    matrix: Sequence[Sequence[float | int]],
    x: object,
    is_symmetric: bool = True,
) -> SX:
    """Return a symbolic quadratic form ``x^\top P x`` for a constant matrix ``P``.

    Args:
        matrix: Constant numeric matrix in row-major form.
        x: Symbolic vector operand with the same dimension as ``matrix``.
        is_symmetric: When ``True`` (default), ``matrix`` is treated as
            symmetric: the function verifies symmetry and builds a compact
            equivalent form that keeps diagonal terms and doubles
            upper-triangular cross terms (with lower-triangular entries set
            to zero). When ``False``, all entries of ``matrix`` are used.

    Returns:
        A scalar ``SX`` expression representing ``x^\top P x``.

    Raises:
        ValueError: If ``matrix`` is not square, dimensions do not match ``x``,
            or ``is_symmetric=True`` and ``matrix`` is not symmetric.
    """
    x_vector = _coerce_vector(x)
    rows, cols, values = _coerce_constant_matrix(matrix)
    if rows != cols:
        raise ValueError("quadratic form requires a square matrix")
    if rows != len(x_vector):
        raise ValueError("matrix size must match vector length")

    matrix_values = values
    if is_symmetric:
        for row in range(rows):
            for col in range(row + 1, rows):
                upper = values[row * rows + col]
                lower = values[col * rows + row]
                if upper != lower:
                    raise ValueError(
                        "quadratic form requires a symmetric matrix when is_symmetric=True"
                    )

        compact_values: list[float] = []
        for row in range(rows):
            for col in range(rows):
                value = values[row * rows + col]
                if col < row:
                    compact_values.append(0.0)
                elif col == row:
                    compact_values.append(value)
                else:
                    compact_values.append(2.0 * value)
        matrix_values = tuple(compact_values)

    return SX(SXNode.make("quadform", _build_quadform_args(rows, matrix_values, x_vector.elements)))


def bilinear_form(
    x: object,
    matrix: Sequence[Sequence[float | int]],
    y: object,
) -> SX:
    """Return the symbolic bilinear form ``x' P y`` for a constant matrix."""
    x_vector = _coerce_vector(x)
    y_vector = _coerce_vector(y)
    rows, cols, values = _coerce_constant_matrix(matrix)
    if rows != len(x_vector):
        raise ValueError("matrix row count must match left vector length")
    if cols != len(y_vector):
        raise ValueError("matrix column count must match right vector length")
    return SX(
        SXNode.make(
            "bilinear_form",
            _build_bilinear_form_args(rows, cols, values, x_vector.elements, y_vector.elements),
        )
    )


def _binary(op: str, lhs: object, rhs: object) -> SX:
    """Build a binary symbolic expression after coercing operands."""
    left = _coerce(lhs)
    right = _coerce(rhs)
    return SX(SXNode.make(op, (left.node, right.node)))


def _unary(op: str, expr: object) -> SX:
    """Build a unary symbolic expression after coercing the operand."""
    value = _coerce(expr)
    return SX(SXNode.make(op, (value.node,)))


def _coerce(value: object) -> SX:
    """Convert supported Python values into ``SX`` expressions.

    ``SX`` values pass through unchanged. Numeric literals become constant
    expressions. Singleton vectors are unwrapped to their sole element so
    they can participate naturally in scalar expressions. Everything else
    raises ``TypeError`` to keep graph construction explicit.
    """
    if isinstance(value, SX):
        return value
    if isinstance(value, SXVector):
        if len(value) == 1:
            return value[0]
        raise TypeError("cannot convert SXVector with length other than 1 to SX")
    if isinstance(value, (int, float)):
        return SX.const(value)
    raise TypeError(f"cannot convert {type(value).__name__} to SX")


def _coerce_scalar(value: object) -> SX:
    """Convert a supported scalar-like value into ``SX``.

    Singleton vectors are treated as scalar-like by unwrapping their sole
    element. Longer vectors remain invalid in scalar contexts.
    """
    return _coerce(value)


def _coerce_vector(value: object) -> SXVector:
    """Convert a supported vector-like value into ``SXVector``.

    ``SXVector`` values pass through unchanged. Python sequences are
    interpreted as element lists and coerced via :func:`vector`.
    """
    if isinstance(value, SXVector):
        return value
    if isinstance(value, (str, bytes)):
        raise TypeError(f"cannot convert {type(value).__name__} to SXVector")
    try:
        return vector(value)
    except TypeError as exc:
        raise TypeError(f"cannot convert {type(value).__name__} to SXVector") from exc


def _coerce_constant_matrix(
    matrix: Sequence[Sequence[float | int]],
) -> tuple[int, int, tuple[float, ...]]:
    """Validate and flatten a constant numeric matrix."""
    if isinstance(matrix, (str, bytes)):
        raise TypeError("matrix must be a sequence of numeric rows")
    rows = list(matrix)
    if not rows:
        return 0, 0, ()
    cols = len(rows[0])
    flattened: list[float] = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise TypeError("matrix rows must be sequences of numbers")
        if len(row) != cols:
            raise ValueError("matrix rows must all have the same length")
        for value in row:
            if not isinstance(value, (int, float)):
                raise TypeError("matrix entries must be numeric constants")
            flattened.append(float(value))
    return len(rows), cols, tuple(flattened)


def _build_matvec_component_args(
    rows: int,
    cols: int,
    row: int,
    values: tuple[float, ...],
    x_elements: tuple[SX, ...],
) -> tuple[SXNode, ...]:
    return (
        SX.const(rows).node,
        SX.const(cols).node,
        SX.const(row).node,
        *(SX.const(value).node for value in values),
        *(element.node for element in x_elements),
    )


def _build_quadform_args(
    size: int,
    values: tuple[float, ...],
    x_elements: tuple[SX, ...],
) -> tuple[SXNode, ...]:
    return (
        SX.const(size).node,
        *(SX.const(value).node for value in values),
        *(element.node for element in x_elements),
    )


def _build_bilinear_form_args(
    rows: int,
    cols: int,
    values: tuple[float, ...],
    x_elements: tuple[SX, ...],
    y_elements: tuple[SX, ...],
) -> tuple[SXNode, ...]:
    return (
        SX.const(rows).node,
        SX.const(cols).node,
        *(SX.const(value).node for value in values),
        *(element.node for element in x_elements),
        *(element.node for element in y_elements),
    )


def parse_matvec_component_args(
    args: tuple[SX, ...],
) -> tuple[int, int, int, tuple[float, ...], tuple[SX, ...]]:
    """Decode a ``matvec_component`` node payload."""
    rows = _require_integral_const(args[0], "rows")
    cols = _require_integral_const(args[1], "cols")
    row = _require_integral_const(args[2], "row")
    matrix_count = rows * cols
    matrix_values = tuple(_require_const(arg, "matrix entry") for arg in args[3 : 3 + matrix_count])
    x_values = args[3 + matrix_count :]
    if len(x_values) != cols:
        raise ValueError("matvec_component payload is malformed")
    return rows, cols, row, matrix_values, x_values


def parse_quadform_args(args: tuple[SX, ...]) -> tuple[int, tuple[float, ...], tuple[SX, ...]]:
    """Decode a ``quadform`` node payload."""
    size = _require_integral_const(args[0], "size")
    matrix_count = size * size
    matrix_values = tuple(_require_const(arg, "matrix entry") for arg in args[1 : 1 + matrix_count])
    x_values = args[1 + matrix_count :]
    if len(x_values) != size:
        raise ValueError("quadform payload is malformed")
    return size, matrix_values, x_values


def parse_bilinear_form_args(
    args: tuple[SX, ...],
) -> tuple[int, int, tuple[float, ...], tuple[SX, ...], tuple[SX, ...]]:
    """Decode a ``bilinear_form`` node payload."""
    rows = _require_integral_const(args[0], "rows")
    cols = _require_integral_const(args[1], "cols")
    matrix_count = rows * cols
    matrix_values = tuple(_require_const(arg, "matrix entry") for arg in args[2 : 2 + matrix_count])
    x_values = args[2 + matrix_count : 2 + matrix_count + rows]
    y_values = args[2 + matrix_count + rows :]
    if len(x_values) != rows or len(y_values) != cols:
        raise ValueError("bilinear_form payload is malformed")
    return rows, cols, matrix_values, x_values, y_values


def matrix_transpose(rows: int, cols: int, values: tuple[float, ...]) -> tuple[float, ...]:
    """Return the flattened transpose of a row-major matrix."""
    return tuple(values[row * cols + col] for col in range(cols) for row in range(rows))


def matrix_add(values_a: tuple[float, ...], values_b: tuple[float, ...]) -> tuple[float, ...]:
    """Return the elementwise sum of two flattened matrices."""
    if len(values_a) != len(values_b):
        raise ValueError("matrix sizes must match")
    return tuple(left + right for left, right in zip(values_a, values_b))


def _require_const(expr: SX, label: str) -> float:
    if expr.op != "const" or expr.value is None:
        raise ValueError(f"{label} must be stored as a constant")
    return expr.value


def _require_integral_const(expr: SX, label: str) -> int:
    value = _require_const(expr, label)
    if value != int(value):
        raise ValueError(f"{label} must be an integer constant")
    return int(value)


def _normalize_metadata(
    metadata: dict[str, Hashable] | None,
) -> tuple[tuple[str, Hashable], ...]:
    """Return a canonical immutable representation of symbol metadata."""
    if metadata is None:
        return ()
    if not isinstance(metadata, dict):
        raise TypeError("symbol metadata must be provided as a dictionary")

    normalized: list[tuple[str, Hashable]] = []
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("symbol metadata keys must be strings")
        if not isinstance(value, Hashable):
            raise TypeError("symbol metadata values must be hashable")
        normalized.append((key, value))
    return tuple(sorted(normalized))
