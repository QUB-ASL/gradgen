import unittest

from gradgen._rust_codegen.rendering.util import (
    _arg_size,
    _describe_input_arg,
    _describe_output_arg,
    _flatten_arg,
    _format_float,
    _format_rust_string_literal,
    _scaled_index_expr,
)
from gradgen.sx import SX, SXVector


class RenderingUtilTests(unittest.TestCase):
    def test_flatten_arg_returns_scalar_tuple_for_symbolic_scalar(self) -> None:
        scalar = SX.sym("x")
        self.assertEqual(_flatten_arg(scalar), (scalar,))

    def test_flatten_arg_returns_vector_elements(self) -> None:
        vector = SXVector.sym("x", 2)
        self.assertEqual(_flatten_arg(vector), tuple(vector.elements))

    def test_arg_size_matches_flattened_length(self) -> None:
        vector = SXVector.sym("x", 3)
        self.assertEqual(_arg_size(vector), 3)

    def test_scaled_index_expr_removes_identity_multiplier(self) -> None:
        self.assertEqual(_scaled_index_expr("i", 1), "i")

    def test_scaled_index_expr_parenthesizes_complex_bases(self) -> None:
        self.assertEqual(_scaled_index_expr("i + 1", 3), "((i + 1) * 3)")

    def test_format_float_matches_scalar_type(self) -> None:
        self.assertEqual(_format_float(1.25, "f64"), "1.25_f64")

    def test_format_rust_string_literal_escapes_special_characters(self) -> None:
        literal = _format_rust_string_literal('a"\\b\n')
        self.assertTrue(literal.startswith('"'))
        self.assertTrue(literal.endswith('"'))
        self.assertIn('\\"', literal)
        self.assertIn('\\\\', literal)
        self.assertIn('\\n', literal)

    def test_describe_input_arg_mentions_cotangent_seeds(self) -> None:
        self.assertIn("cotangent seed", _describe_input_arg("cotangent_cost"))

    def test_describe_output_arg_mentions_gradients(self) -> None:
        self.assertIn("gradient", _describe_output_arg("gradient_cost"))
