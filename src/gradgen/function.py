"""Symbolic function abstraction built on top of ``SX`` expressions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .sx import SX, SXNode, SXVector, vector


FunctionArg = SX | SXVector
BoundValue = SX | SXVector | float | int | list[object] | tuple[object, ...]


@dataclass(frozen=True, slots=True)
class Function:
    """Symbolic multi-input, multi-output function.

    A ``Function`` defines a boundary around symbolic expressions. Inputs
    must be symbolic leaves, while outputs can be arbitrary ``SX`` or
    ``SXVector`` expressions that depend on those inputs.

    The class supports:

    - explicit input and output grouping
    - validation of names and symbolic inputs
    - topological ordering of the output dependency graph
    - calling with symbolic values for substitution
    - calling with numeric values for direct evaluation
    """

    name: str
    inputs: tuple[FunctionArg, ...]
    outputs: tuple[FunctionArg, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def __init__(
        self,
        name: str,
        inputs: Iterable[FunctionArg],
        outputs: Iterable[FunctionArg],
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
    ) -> None:
        normalized_inputs = tuple(_coerce_function_arg(item, label="input") for item in inputs)
        normalized_outputs = tuple(_coerce_function_arg(item, label="output") for item in outputs)

        if not normalized_inputs:
            raise ValueError("Function must have at least one input")
        if not normalized_outputs:
            raise ValueError("Function must have at least one output")

        resolved_input_names = _resolve_names(
            names=input_names,
            count=len(normalized_inputs),
            prefix="i",
            label="input",
        )
        resolved_output_names = _resolve_names(
            names=output_names,
            count=len(normalized_outputs),
            prefix="o",
            label="output",
        )

        _validate_inputs(normalized_inputs)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "inputs", normalized_inputs)
        object.__setattr__(self, "outputs", normalized_outputs)
        object.__setattr__(self, "input_names", resolved_input_names)
        object.__setattr__(self, "output_names", resolved_output_names)

    @property
    def flat_inputs(self) -> tuple[SX, ...]:
        """Return all scalar inputs flattened into a single tuple."""
        return tuple(_flatten_args(self.inputs))

    @property
    def flat_outputs(self) -> tuple[SX, ...]:
        """Return all scalar outputs flattened into a single tuple."""
        return tuple(_flatten_args(self.outputs))

    @property
    def nodes(self) -> tuple[SXNode, ...]:
        """Return output dependency nodes in topological order."""
        ordered: list[SXNode] = []
        seen: set[SXNode] = set()

        for expr in self.flat_outputs:
            _visit_node(expr.node, seen, ordered)

        return tuple(ordered)

    def __call__(self, *args: BoundValue) -> FunctionArg | tuple[FunctionArg, ...] | float | tuple[float, ...]:
        """Call the function with symbolic or numeric arguments.

        Args:
            *args: One argument for each declared input. Scalar inputs accept
                ``SX``, ``int``, or ``float``. Vector inputs accept
                ``SXVector`` or a sequence of scalar-like values.

        Returns:
            The evaluated outputs. If the call stays symbolic, the return
            value contains ``SX`` and ``SXVector`` objects. If all outputs
            reduce to numeric constants, the result is returned as ``float``
            values.
        """
        if len(args) != len(self.inputs):
            raise ValueError(
                f"expected {len(self.inputs)} arguments, received {len(args)}"
            )

        mapping: dict[SXNode, SX] = {}
        for formal, actual in zip(self.inputs, args):
            bound = _coerce_bound_arg(actual, formal)
            for formal_scalar, actual_scalar in zip(_flatten_single(formal), _flatten_single(bound)):
                mapping[formal_scalar.node] = actual_scalar

        evaluated = tuple(
            _finalize_output(_substitute_output(output, mapping)) for output in self.outputs
        )
        return _collapse_outputs(evaluated)

    def __repr__(self) -> str:
        return (
            f"Function(name={self.name!r}, input_names={self.input_names!r}, "
            f"output_names={self.output_names!r})"
        )


def _coerce_function_arg(item: object, *, label: str) -> FunctionArg:
    """Coerce a supported function input or output declaration."""
    if isinstance(item, (SX, SXVector)):
        return item
    raise TypeError(f"{label} must be an SX or SXVector")


def _resolve_names(
    *,
    names: Iterable[str] | None,
    count: int,
    prefix: str,
    label: str,
) -> tuple[str, ...]:
    """Resolve user-provided names or generate defaults."""
    if names is None:
        return tuple(f"{prefix}{index}" for index in range(count))

    resolved = tuple(names)
    if len(resolved) != count:
        raise ValueError(f"expected {count} {label} names, received {len(resolved)}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} names must be unique")
    return resolved


def _validate_inputs(inputs: tuple[FunctionArg, ...]) -> None:
    """Ensure every input is built from unique symbolic leaves."""
    seen_names: set[str] = set()

    for item in inputs:
        for scalar in _flatten_single(item):
            if scalar.op != "symbol":
                raise ValueError("Function inputs must be symbolic variables")
            if scalar.name in seen_names:
                raise ValueError("Function input symbols must be unique")
            seen_names.add(scalar.name)


def _flatten_args(items: Iterable[FunctionArg]) -> Iterable[SX]:
    """Yield scalar expressions from a sequence of scalar or vector values."""
    for item in items:
        yield from _flatten_single(item)


def _flatten_single(item: FunctionArg) -> tuple[SX, ...]:
    """Flatten a single scalar or vector value into scalar expressions."""
    if isinstance(item, SX):
        return (item,)
    return item.elements


def _visit_node(node: SXNode, seen: set[SXNode], ordered: list[SXNode]) -> None:
    """Depth-first topological traversal of expression nodes."""
    if node in seen:
        return

    for arg in node.args:
        _visit_node(arg, seen, ordered)

    seen.add(node)
    ordered.append(node)


def _coerce_bound_arg(value: BoundValue, formal: FunctionArg) -> FunctionArg:
    """Coerce a call-time argument to match a declared input shape."""
    if isinstance(formal, SX):
        return _coerce_scalar_value(value)
    if isinstance(value, SXVector):
        if len(value) != len(formal):
            raise ValueError("vector argument length must match the formal input")
        return value
    if isinstance(value, (list, tuple)):
        coerced = vector(value)
        if len(coerced) != len(formal):
            raise ValueError("vector argument length must match the formal input")
        return coerced
    raise TypeError("vector inputs require an SXVector or a sequence of scalar-like values")


def _coerce_scalar_value(value: BoundValue) -> SX:
    """Coerce a scalar call-time value into an ``SX`` expression."""
    if isinstance(value, SX):
        return value
    if isinstance(value, (int, float)):
        return SX.const(value)
    raise TypeError("scalar inputs require an SX, int, or float")


def _substitute_output(output: FunctionArg, mapping: dict[SXNode, SX]) -> FunctionArg:
    """Apply input substitution to a function output."""
    if isinstance(output, SX):
        return _substitute_scalar(output, mapping)
    return SXVector(tuple(_substitute_scalar(element, mapping) for element in output))


def _substitute_scalar(expr: SX, mapping: dict[SXNode, SX]) -> SX:
    """Recursively substitute formal input symbols with bound values."""
    if expr.node in mapping:
        return mapping[expr.node]
    if expr.op in {"symbol", "const"}:
        return expr

    substituted_args = tuple(_substitute_scalar(SX(arg), mapping).node for arg in expr.node.args)
    return SX(SXNode.make(expr.op, substituted_args, name=expr.name, value=expr.value))


def _finalize_output(output: FunctionArg) -> FunctionArg | float:
    """Evaluate a substituted output when it has become fully numeric."""
    if isinstance(output, SX):
        if _is_numeric_expr(output):
            return _evaluate_scalar(output)
        return output

    if all(_is_numeric_expr(element) for element in output):
        return tuple(_evaluate_scalar(element) for element in output)
    return output


def _collapse_outputs(
    outputs: tuple[FunctionArg | float | tuple[float, ...], ...]
) -> FunctionArg | tuple[FunctionArg | float | tuple[float, ...], ...] | float | tuple[float, ...]:
    """Return a single output directly and preserve tuples for multi-output calls."""
    if len(outputs) == 1:
        return outputs[0]
    return outputs


def _is_numeric_expr(expr: SX) -> bool:
    """Return ``True`` when an expression depends only on constants."""
    if expr.op == "const":
        return True
    if expr.op == "symbol":
        return False
    return all(_is_numeric_expr(SX(arg)) for arg in expr.node.args)


def _evaluate_scalar(expr: SX) -> float:
    """Evaluate a scalar expression made only of constants."""
    if expr.op == "const":
        if expr.value is None:
            raise ValueError("constant expression is missing a value")
        return expr.value

    args = tuple(_evaluate_scalar(SX(arg)) for arg in expr.node.args)

    if expr.op == "add":
        return args[0] + args[1]
    if expr.op == "sub":
        return args[0] - args[1]
    if expr.op == "mul":
        return args[0] * args[1]
    if expr.op == "div":
        return args[0] / args[1]
    if expr.op == "pow":
        return args[0] ** args[1]
    if expr.op == "neg":
        return -args[0]
    if expr.op == "sin":
        return math.sin(args[0])
    if expr.op == "cos":
        return math.cos(args[0])
    if expr.op == "exp":
        return math.exp(args[0])
    if expr.op == "log":
        return math.log(args[0])
    if expr.op == "sqrt":
        return math.sqrt(args[0])

    raise ValueError(f"cannot evaluate operation {expr.op!r}")
