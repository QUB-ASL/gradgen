"""Common subexpression elimination utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .sx import SX, SXNode, SXVector


ExpressionLike = SX | SXVector


@dataclass(frozen=True, slots=True)
class CSEAssignment:
    """A named reusable intermediate expression."""

    name: str
    expr: SX
    use_count: int


@dataclass(frozen=True, slots=True)
class CSEPlan:
    """A computation plan extracted from a symbolic DAG."""

    assignments: tuple[CSEAssignment, ...]
    outputs: tuple[SX, ...]
    use_counts: dict[SXNode, int]


def cse(
    outputs: Iterable[ExpressionLike],
    *,
    prefix: str = "w",
    min_uses: int = 2,
) -> CSEPlan:
    """Build a common-subexpression elimination plan for symbolic outputs.

    Args:
        outputs: Scalar or vector symbolic outputs to analyze.
        prefix: Prefix used for temporary names.
        min_uses: Minimum number of uses required before an expression is
            promoted to a named temporary.

    Returns:
        A ``CSEPlan`` containing topologically ordered assignments for
        reusable intermediates and the flattened scalar outputs.
    """
    if min_uses < 2:
        raise ValueError("min_uses must be at least 2")

    flat_outputs = tuple(_flatten_outputs(outputs))
    ordered = _topological_nodes(flat_outputs)
    use_counts = _count_uses(flat_outputs)

    assignments: list[CSEAssignment] = []
    temp_index = 0
    for node in ordered:
        if node.op in {"symbol", "const"}:
            continue
        if use_counts.get(node, 0) < min_uses:
            continue
        assignments.append(
            CSEAssignment(
                name=f"{prefix}{temp_index}",
                expr=SX(node),
                use_count=use_counts[node],
            )
        )
        temp_index += 1

    return CSEPlan(
        assignments=tuple(assignments),
        outputs=flat_outputs,
        use_counts=use_counts,
    )


def _flatten_outputs(outputs: Iterable[ExpressionLike]) -> Iterable[SX]:
    """Flatten scalar and vector outputs into scalar expressions."""
    for output in outputs:
        if isinstance(output, SX):
            yield output
        else:
            yield from output


def _topological_nodes(outputs: tuple[SX, ...]) -> tuple[SXNode, ...]:
    """Return output dependency nodes in topological order."""
    ordered: list[SXNode] = []
    seen: set[SXNode] = set()

    for output in outputs:
        _visit_node(output.node, seen, ordered)

    return tuple(ordered)


def _visit_node(node: SXNode, seen: set[SXNode], ordered: list[SXNode]) -> None:
    """Depth-first topological traversal of expression nodes."""
    if node in seen:
        return

    for arg in node.args:
        _visit_node(arg, seen, ordered)

    seen.add(node)
    ordered.append(node)


def _count_uses(outputs: tuple[SX, ...]) -> dict[SXNode, int]:
    """Count how many parent/output references each node receives."""
    counts: dict[SXNode, int] = {}

    def record(node: SXNode) -> None:
        counts[node] = counts.get(node, 0) + 1
        for arg in node.args:
            record(arg)

    for output in outputs:
        record(output.node)

    return counts
