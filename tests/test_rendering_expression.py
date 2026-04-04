import unittest

from gradgen._rust_codegen.rendering.expression import (
    _emit_expr_ref,
    _emit_math_call,
    _emit_matrix_literal,
    _emit_node_expr,
    _emit_norm_abs_expr,
    _match_contiguous_slice,
)
from gradgen.sx import SX


class RenderingExpressionTests(unittest.TestCase):
    def test_match_contiguous_slice_detects_known_slices(self) -> None:
        self.assertEqual(_match_contiguous_slice(("a[0]", "a[1]")), "a")

    def test_emit_math_call_uses_std_backend_methods(self) -> None:
        self.assertEqual(_emit_math_call("sin", ("x",), "std", "f64", None), "x.sin()")

    def test_emit_math_call_uses_no_std_math_library(self) -> None:
        self.assertEqual(_emit_math_call("pow", ("x", "y"), "no_std", "f32", "libm"), "libm::powf(x, y)")

    def test_emit_matrix_literal_formats_scalar_values(self) -> None:
        self.assertEqual(_emit_matrix_literal((1.0, 2.0), "f64"), "&[1.0_f64, 2.0_f64]")

    def test_emit_expr_ref_returns_symbol_binding(self) -> None:
        x = SX.sym("x")
        self.assertEqual(
            _emit_expr_ref(
                x,
                {x.node: "x"},
                {},
                "std",
                "f64",
                None,
            ),
            "x",
        )

    def test_emit_node_expr_handles_simple_additions(self) -> None:
        x = SX.sym("x")
        expr = x + 1.0
        self.assertEqual(
            _emit_node_expr(
                expr,
                {x.node: "x"},
                {},
                "std",
                "f64",
                None,
            ),
            "1.0_f64 + x",
        )

    def test_emit_norm_abs_expr_uses_std_abs(self) -> None:
        self.assertEqual(_emit_norm_abs_expr("value", "std", "f64", None), "value.abs()")
