"""Workspace allocation and assignment helpers for Rust code generation."""

from __future__ import annotations

from ...sx import SX, SXNode
from ..config import RustBackendMode, RustScalarType
from .expression import _emit_expr_ref
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
    """Emit one workspace assignment, using compound operators when safe."""
    target = f"work[{work_index}]"
    expr = SX(node)

    if expr.op in {"add", "sub", "mul", "div"}:
        left, right = expr.args
        left_is_target = _workspace_ref_for_node(left.node, workspace_map) == target
        right_is_target = _workspace_ref_for_node(right.node, workspace_map) == target
        if left_is_target:
            other_ref = _emit_expr_ref(
                right,
                scalar_bindings,
                workspace_map,
                backend_mode,
                scalar_type,
                math_library,
            )
            operator = {
                "add": "+=",
                "sub": "-=",
                "mul": "*=",
                "div": "/=",
            }[expr.op]
            return f"{target} {operator} {other_ref};"
        if expr.op in {"add", "mul"} and right_is_target:
            other_ref = _emit_expr_ref(
                left,
                scalar_bindings,
                workspace_map,
                backend_mode,
                scalar_type,
                math_library,
            )
            operator = {
                "add": "+=",
                "mul": "*=",
            }[expr.op]
            return f"{target} {operator} {other_ref};"

    return f"{target} = {rhs};"


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


def _allocate_workspace_slots(
    function,
    *,
    output_refs: tuple[SX, ...] | None = None,
) -> tuple[dict[SXNode, int], int]:
    """Assign reusable workspace slots based on each node's last use."""
    if output_refs is None:
        output_refs = tuple(scalar for output in function.outputs for scalar in _flatten_arg(output))
    required_nodes = _collect_required_workspace_nodes(output_refs)
    workspace_nodes = [
        node
        for node in function.nodes
        if node.op not in {"symbol", "const"} and node in required_nodes
    ]
    if not workspace_nodes:
        return {}, 0

    node_index = {node: index for index, node in enumerate(workspace_nodes)}
    last_use = {node: index for index, node in enumerate(workspace_nodes)}

    for index, node in enumerate(workspace_nodes):
        for child in node.args:
            if child in node_index:
                last_use[child] = max(last_use[child], index)

    output_base = len(workspace_nodes)
    for offset, scalar in enumerate(output_refs):
        if scalar.node in node_index:
            last_use[scalar.node] = max(last_use[scalar.node], output_base + offset)

    available_slots: list[int] = []
    expiring_by_index: dict[int, list[int]] = {}
    workspace_map: dict[SXNode, int] = {}
    next_slot = 0

    for index, node in enumerate(workspace_nodes):
        for slot in expiring_by_index.pop(index, []):
            available_slots.append(slot)

        if available_slots:
            slot = min(available_slots)
            available_slots.remove(slot)
        else:
            slot = next_slot
            next_slot += 1

        workspace_map[node] = slot
        expiring_by_index.setdefault(last_use[node], []).append(slot)

    return workspace_map, next_slot
