import unittest

from gradgen._rust_codegen.rendering.expression import (
    _emit_math_call, _match_contiguous_slice
)
from gradgen._rust_codegen.rendering.util import _format_float


class RenderTests(unittest.TestCase):
    def test_format_float_matches_scalar_type(self) -> None:
        self.assertEqual(_format_float(1.25, "f64"), "1.25_f64")

    def test_match_contiguous_slice_detects_known_slices(self) -> None:
        self.assertEqual(_match_contiguous_slice(("a[0]", "a[1]")), "a")

    def test_emit_math_call_uses_std_backend_methods(self) -> None:
        self.assertEqual(
            _emit_math_call("sin", ("x",), "std", "f64", None), "x.sin()")
