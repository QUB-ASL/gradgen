import unittest

from gradgen._rust_codegen.rendering.helpers import _build_shared_helper_lines
from gradgen.sx import SX, SXNode


class RenderingHelpersTests(unittest.TestCase):
    def test_build_shared_helper_lines_emits_expected_helpers(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        nodes = (
            SXNode.make("sum", (x.node, y.node)),
            SXNode.make("matvec_component", ()),
            SXNode.make("norm2", (x.node, y.node)),
        )

        lines = "\n".join(_build_shared_helper_lines(nodes, "std", "f64", None))
        self.assertIn("fn vec_sum", lines)
        self.assertIn("fn matvec_component", lines)
        self.assertIn("fn norm2", lines)
