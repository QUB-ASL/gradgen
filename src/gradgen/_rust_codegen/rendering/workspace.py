"""Workspace allocation and assignment helpers for Rust code generation."""

from __future__ import annotations

from collections import Counter

from ...sx import SX, SXNode
from ..config import RustBackendMode, RustScalarType
from .expression import _emit_expr_ref
from .util import _format_float
from .util import _flatten_arg


def _workspace_ref_for_node(node: SXNode, workspace_map: dict[SXNode, int]) -> str | None:
    """Return the workspace reference for ``node`` when it already has one."""
    work_index = workspace_map.get(node)
    if work_index is None:
        return None
    return f"work[{work_index}]"


def _emit_workspace_assignment(
    node: SXNode,
    work_index: int,
    rhs: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str,
) -> str:
    """Emit one workspace assignment."""
    del node, scalar_bindings, workspace_map, backend_mode, scalar_type
    del math_library
    return f"work[{work_index}] = {rhs};"


def _emit_exact_length_assert(rust_name: str, display_name: str, expected_size: int) -> tuple[str, str, str]:
    """Emit exact-length checks for generated Rust entrypoints.

    Returns a triple of strings: (assert_line, input_return_line, output_return_line).
    Private helpers intentionally emit no assert line so all runtime shape
    validation stays in public ``Result``-returning functions.
    """
    assert_line = ""
    input_return = (
        f'if {rust_name}.len() != {expected_size} {{ '
        f'return Err(GradgenError::InputTooSmall("{display_name} expected length {expected_size}")); '
        f'}};'
    )
    output_return = (
        f'if {rust_name}.len() != {expected_size} {{ '
        f'return Err(GradgenError::OutputTooSmall("{display_name} expected length {expected_size}")); '
        f'}};'
    )
    return (assert_line, input_return, output_return)


def _emit_min_length_assert(rust_name: str, display_name: str, minimum_size: int) -> tuple[str, str]:
    """Emit minimum-length checks for generated Rust entrypoints.

    Returns a pair: (assert_line, return_line). Private helpers intentionally
    emit no assert line so all runtime workspace validation stays in public
    ``Result``-returning functions.
    """
    if minimum_size == 1:
        assert_line = ""
        return_line = (
            f'if {rust_name}.is_empty() {{ '
            f'return Err(GradgenError::WorkspaceTooSmall("{display_name} expected at least 1")); '
            f'}};'
        )
        return (assert_line, return_line)
    assert_line = ""
    return_line = (
        f'if {rust_name}.len() < {minimum_size} {{ '
        f'return Err(GradgenError::WorkspaceTooSmall("{display_name} expected at least {minimum_size}")); '
        f'}};'
    )
    return (assert_line, return_line)


def _collect_required_workspace_nodes(output_refs: tuple[SX, ...]) -> set[SXNode]:
    """Return non-trivial nodes reachable from the materialized outputs."""
    required: set[SXNode] = set()
    stack = [expr.node for expr in output_refs]
    while stack:
        node = stack.pop()
        if node in required or node.op in {"symbol", "const"}:
            continue
        required.add(node)
        stack.extend(node.args)
    return required


def _collect_reachable_nodes(output_refs: tuple[SX, ...]) -> set[SXNode]:
    """Return all reachable nodes (including symbols/constants) from outputs."""
    reachable: set[SXNode] = set()
    stack = [expr.node for expr in output_refs]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(node.args)
    return reachable


def _count_node_uses(output_refs: tuple[SX, ...]) -> dict[SXNode, int]:
    """Count how many times each node is referenced in the output DAG."""
    counts: dict[SXNode, int] = Counter()

    def record(node: SXNode) -> None:
        counts[node] += 1
        for arg in node.args:
            record(arg)

    for expr in output_refs:
        record(expr.node)
    return counts


def _allocate_workspace_slots(
    function,
    *,
    output_refs: tuple[SX, ...] | None = None,
    prioritize_expensive_nodes: bool = False,
) -> tuple[dict[SXNode, int], int]:
    """Assign reusable workspace slots based on each node's last use.

    The allocator uses a small cost model so repeated expensive
    expressions are more likely to be cached than repeated cheap ones.
    """
    if output_refs is None:
        output_refs = tuple(
            scalar
            for output in function.outputs
            for scalar in _flatten_arg(output)
        )
    required_nodes = _collect_required_workspace_nodes(output_refs)
    use_counts = _count_node_uses(output_refs)
    workspace_nodes = [
        node
        for node in function.nodes
        if (
            node.op not in {"symbol", "const"}
            and node in required_nodes
            and (
                _should_materialize_node(node, use_counts.get(node, 0))
                if prioritize_expensive_nodes
                else use_counts.get(node, 0) > 1
            )
        )
    ]
    if not workspace_nodes:
        return {}, 0

    node_index = {node: index for index, node in enumerate(workspace_nodes)}
    last_use = {node: index for index, node in enumerate(workspace_nodes)}

    for index, node in enumerate(function.nodes):
        for child in node.args:
            if child in node_index:
                last_use[child] = max(last_use[child], index)

    output_base = len(workspace_nodes)
    for offset, scalar in enumerate(output_refs):
        if scalar.node in node_index:
            last_use[scalar.node] = max(
                last_use[scalar.node], output_base + offset + 1
            )

    workspace_map: dict[SXNode, int] = {}
    next_slot = 0

    for index, node in enumerate(workspace_nodes):
        slot = next_slot
        next_slot += 1
        workspace_map[node] = slot

    return workspace_map, next_slot


def _should_materialize_node(node: SXNode, use_count: int) -> bool:
    """Return whether a repeated node is worth caching in workspace."""
    if use_count < 2:
        return False
    expr_cost = _estimate_node_cost(node)
    if expr_cost <= 1:
        materialization_cost = 3
    elif expr_cost <= 2:
        materialization_cost = 2
    else:
        materialization_cost = 1
    return (use_count - 1) * expr_cost > materialization_cost


def _estimate_node_cost(node: SXNode) -> int:
    """Estimate the relative cost of recomputing ``node``."""
    if node.op in {"symbol", "const"}:
        return 0
    if node.op in {"neg", "abs"}:
        return 1
    if node.op in {
        "add",
        "sub",
        "mul",
        "div",
        "min",
        "max",
        "sign",
        "signum",
        "floor",
        "ceil",
        "trunc",
        "round",
        "remainder",
    }:
        return 1
    if node.op in {
        "pow",
        "sqrt",
        "exp",
        "log",
        "log1p",
        "expm1",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "atan2",
        "hypot",
        "norm2",
        "norm2sq",
        "sum",
        "mean",
        "prod",
        "reduce_max",
        "reduce_min",
        "matvec",
        "transpose_matvec",
        "matvec_component",
        "transpose_matvec_component",
        "bilinear_form",
        "quadform",
        "cross",
    }:
        return 4
    return 2
