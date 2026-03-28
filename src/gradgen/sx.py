"""Core `SX` scalar expression types.

This module provides the first symbolic building block for the library:

- ``SXNode`` is the canonical internal graph node
- ``SX`` is the small user-facing wrapper around a node
- helper functions such as :func:`sin` and :func:`sqrt` make expression
  construction feel natural from Python

The implementation uses structural interning, also called hash-consing.
That means two expressions with the same structure reuse the same
``SXNode`` instance. This keeps the symbolic graph compact and gives us a
directed acyclic graph (DAG) instead of repeatedly rebuilding identical
subtrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import ClassVar


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
        value: Numeric literal for ``"const"`` nodes.
    """

    op: str
    args: tuple[SXNode, ...] = ()
    name: str | None = None
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
            value: Optional floating-point literal for ``"const"`` nodes.

        Returns:
            The unique canonical node matching the requested structure.
        """
        normalized_args = cls._normalize_args(op, args)
        key = (op, normalized_args, name, value)

        with cls._lock:
            node = cls._cache.get(key)
            if node is not None:
                return node

            node = cls(op=op, args=normalized_args, name=name, value=value)
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
    def sym(cls, name: str) -> SX:
        """Create a symbolic scalar variable.

        Args:
            name: Symbol name used to identify the variable.

        Returns:
            An ``SX`` wrapper around a canonical ``"symbol"`` node.
        """
        return cls(SXNode.make("symbol", name=name))

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
            return f"SX.sym({self.name!r})"
        if self.op == "const":
            return f"SX.const({self.value!r})"
        if len(self.node.args) == 1:
            return f"{self.op}({SX(self.node.args[0])!r})"
        if len(self.node.args) == 2:
            lhs, rhs = self.node.args
            return f"{self.op}({SX(lhs)!r}, {SX(rhs)!r})"
        return f"SX({self.node!r})"


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
