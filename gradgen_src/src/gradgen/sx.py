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
from typing import ClassVar, Iterable, Iterator


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

    def __mul__(self, other: object) -> SX:
        return _binary("mul", self, other)

    def __rmul__(self, other: object) -> SX:
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

    def exp(self) -> SX:
        return _unary("exp", self)

    def log(self) -> SX:
        return _unary("log", self)

    def sqrt(self) -> SX:
        return _unary("sqrt", self)

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

    def __getitem__(self, index: int) -> SX:
        """Return the scalar element at ``index``."""
        return self.elements[index]

    def __add__(self, other: object) -> SXVector:
        """Return the elementwise sum of two vectors."""
        return self._elementwise_binary("add", other)

    def __radd__(self, other: object) -> SXVector:
        """Return the elementwise sum of two vectors."""
        return self._elementwise_binary("add", other, reverse=True)

    def __sub__(self, other: object) -> SXVector:
        """Return the elementwise difference of two vectors."""
        return self._elementwise_binary("sub", other)

    def __rsub__(self, other: object) -> SXVector:
        """Return the reversed elementwise difference of two vectors."""
        return self._elementwise_binary("sub", other, reverse=True)

    def __mul__(self, other: object) -> SXVector:
        """Return a scalar-vector product.

        Vector-vector multiplication is intentionally unsupported for now.
        Use :meth:`dot` for an inner product.
        """
        scalar = _coerce_scalar(other)
        return SXVector(tuple(element * scalar for element in self.elements))

    def __rmul__(self, other: object) -> SXVector:
        """Return a scalar-vector product."""
        scalar = _coerce_scalar(other)
        return SXVector(tuple(scalar * element for element in self.elements))

    def __truediv__(self, other: object) -> SXVector:
        """Return elementwise division by a scalar or vector."""
        if isinstance(other, SXVector):
            return self._elementwise_binary("div", other)
        scalar = _coerce_scalar(other)
        return SXVector(tuple(element / scalar for element in self.elements))

    def __rtruediv__(self, other: object) -> SXVector:
        """Return scalar divided by each vector element."""
        scalar = _coerce_scalar(other)
        return SXVector(tuple(scalar / element for element in self.elements))

    def __neg__(self) -> SXVector:
        """Return the elementwise negation of the vector."""
        return SXVector(tuple(-element for element in self.elements))

    def sin(self) -> SXVector:
        """Apply sine elementwise."""
        return SXVector(tuple(element.sin() for element in self.elements))

    def cos(self) -> SXVector:
        """Apply cosine elementwise."""
        return SXVector(tuple(element.cos() for element in self.elements))

    def exp(self) -> SXVector:
        """Apply exponential elementwise."""
        return SXVector(tuple(element.exp() for element in self.elements))

    def log(self) -> SXVector:
        """Apply natural logarithm elementwise."""
        return SXVector(tuple(element.log() for element in self.elements))

    def sqrt(self) -> SXVector:
        """Apply square root elementwise."""
        return SXVector(tuple(element.sqrt() for element in self.elements))

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


def exp(expr: object) -> SX:
    """Return the symbolic exponential of an expression."""
    return _unary("exp", expr)


def log(expr: object) -> SX:
    """Return the symbolic natural logarithm of an expression."""
    return _unary("log", expr)


def sqrt(expr: object) -> SX:
    """Return the symbolic square root of an expression."""
    return _unary("sqrt", expr)


def vector(values: Iterable[object]) -> SXVector:
    """Create a symbolic vector from a sequence of scalar-like values."""
    return SXVector(tuple(_coerce(value) for value in values))


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
    expressions. Everything else raises ``TypeError`` to keep graph
    construction explicit.
    """
    if isinstance(value, SX):
        return value
    if isinstance(value, (int, float)):
        return SX.const(value)
    raise TypeError(f"cannot convert {type(value).__name__} to SX")


def _coerce_scalar(value: object) -> SX:
    """Convert a supported scalar-like value into ``SX``.

    ``SXVector`` values are rejected here because some operators, such as
    scalar-vector multiplication, intentionally only support scalar
    operands on one side.
    """
    if isinstance(value, SXVector):
        raise TypeError("expected a scalar-like value, got SXVector")
    return _coerce(value)


def _coerce_vector(value: object) -> SXVector:
    """Convert a supported vector-like value into ``SXVector``."""
    if isinstance(value, SXVector):
        return value
    raise TypeError(f"cannot convert {type(value).__name__} to SXVector")


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
