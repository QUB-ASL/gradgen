import unittest

from gradgen import Function, SXVector
from gradgen._rust_codegen.rendering.workspace import (
    _allocate_workspace_slots,
    _collect_reachable_nodes,
    _collect_required_workspace_nodes,
    _emit_exact_length_assert,
    _emit_min_length_assert,
    _workspace_ref_for_node,
)
from gradgen.sx import SX, SXNode


class RenderingWorkspaceTests(unittest.TestCase):
    def test_workspace_ref_for_node_returns_reference_when_present(self) -> None:
        node = SXNode.make("add", (SX.const(1.0).node, SX.const(2.0).node))
        self.assertEqual(_workspace_ref_for_node(node, {node: 3}), "work[3]")

    def test_length_assert_helpers_emit_result_returning_checks(self) -> None:
        exact = _emit_exact_length_assert("x", "x", 2)
        self.assertIn("InputTooSmall", exact[1])
        self.assertIn("OutputTooSmall", exact[2])

        minimum = _emit_min_length_assert("work", "work", 1)
        self.assertIn("WorkspaceTooSmall", minimum[1])

    def test_workspace_collection_identifies_reachable_nodes(self) -> None:
        x = SX.sym("x")
        expr = (x + 1.0) * (x + 2.0)
        reachable = _collect_reachable_nodes((expr,))
        required = _collect_required_workspace_nodes((expr,))

        self.assertTrue(any(node.op == "symbol" for node in reachable))
        self.assertTrue(any(node.op == "const" for node in reachable))
        self.assertTrue(any(node.op == "mul" for node in required))
        self.assertTrue(all(node.op not in {"symbol", "const"} for node in required))

    def test_allocate_workspace_slots_returns_non_empty_allocation_for_nontrivial_function(self) -> None:
        x = SXVector.sym("x", 1)
        function = Function(
            "demo",
            [x],
            [(x[0] + 1.0) * (x[0] + 2.0)],
            input_names=["x"],
            output_names=["y"],
        )

        workspace_map, workspace_size = _allocate_workspace_slots(function)
        self.assertGreaterEqual(workspace_size, 1)
        self.assertTrue(workspace_map)
